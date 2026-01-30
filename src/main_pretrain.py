"""FCMAE 预训练入口脚本（自监督）。

你可以把它当作“组装器”：
- 解析命令行参数（模型大小、mask_ratio、LR、epochs、分布式等）
- 构建 transform + `ImageFolder(data_path/train)`
- 构建 FCMAE 模型（`src.models.fcmae`）
- 构建 optimizer + scaler
- 进入训练循环：`src.engine_pretrain.train_one_epoch`

数据约定：`--data_path` 指向一个目录，且其下必须有 `train/`（ImageFolder 格式）。
注意：RHEED 图片常见为灰度图，本文件会在 transform 最前面把 PIL 图像转为 RGB，
以匹配 ImageNet 的 mean/std 归一化。
"""


import argparse
import datetime
import numpy as np

import time
import json
import os
import faulthandler
from pathlib import Path


import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import timm
# The original upstream code pinned timm==0.3.2. For local experimentation, we allow newer versions.
if getattr(timm, "__version__", "") != "0.3.2":
    print(f"Warning: this repo was originally tested with timm==0.3.2, current={timm.__version__}")

# timm API compatibility:
# - old: timm.optim.optim_factory.add_weight_decay
# - new: timm.optim.optim_factory.param_groups_weight_decay
import timm.optim.optim_factory as optim_factory


def _timm_param_groups(model, weight_decay: float):
    if hasattr(optim_factory, "param_groups_weight_decay"):
        return optim_factory.param_groups_weight_decay(model, weight_decay=weight_decay)
    if hasattr(optim_factory, "add_weight_decay"):
        return optim_factory.add_weight_decay(model, weight_decay)

    # final fallback: minimal local implementation
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if len(param.shape) == 1 or name.endswith(".bias"):
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {"params": no_decay, "weight_decay": 0.0},
        {"params": decay, "weight_decay": weight_decay},
    ]


from src.engine_pretrain import train_one_epoch
from src.models import fcmae

from src import utils
from src.utils import NativeScalerWithGradNormCount as NativeScaler
from src.utils import str2bool


def _to_rgb(img):
    """确保输入为 3 通道 RGB（RHEED 常见为灰度图）。"""
    try:
        return img.convert("RGB")
    except Exception:
        return img


def get_args_parser():
    """构建预训练阶段的命令行参数.

    你最关心的几个参数: 
    - `--model`: 选择 `fcmae.py` 里定义的模型规模 (tiny/base/large/...) .
    - `--data_path`: 数据根目录 (需要包含 `train/` 子目录) .
    - `--mask_ratio`: MAE mask 比例, 越大越难.
    - `--batch_size`/`--update_freq`: 单卡 batch 与梯度累积, 决定有效 batch size.
    - `--blr`: base lr (会按有效 batch size 自动换算成 `--lr`) .
    - `--output_dir`/`--log_dir`: checkpoint 与 tensorboard.
    """
    parser = argparse.ArgumentParser('FCMAE pre-training', add_help=False)

    parser.add_argument('--batch_size', default=64, type=int,
                        help='Per GPU batch size')
    parser.add_argument('--epochs', default=800, type=int)
    parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N',
                        help='epochs to warmup LR')
    parser.add_argument('--update_freq', default=1, type=int,
                        help='gradient accumulation step')
    
    # Model parameters
    parser.add_argument('--model', default='convnextv2_base', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--input_size', default=224, type=int,
                        help='image input size')
    parser.add_argument('--mask_ratio', default=0.6, type=float,
                        help='Masking ratio (percentage of removed patches).')
    parser.add_argument('--norm_pix_loss', action='store_true',
                        help='Use (per-patch) normalized pixels as targets for computing loss')
    parser.set_defaults(norm_pix_loss=True)
    parser.add_argument('--decoder_depth', type=int, default=1)
    parser.add_argument('--decoder_embed_dim', type=int, default=512)
    
    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1.5e-4, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')
    
    # Dataset parameters
    parser.add_argument('--data_path', default='/datasets01/imagenet_full_size/061417/', type=str,
                        help='dataset path')
    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default=None,
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')

    parser.add_argument('--auto_resume', type=str2bool, default=True)
    parser.add_argument('--save_ckpt', type=str2bool, default=True)
    parser.add_argument('--save_ckpt_freq', default=1, type=int)
    parser.add_argument('--save_ckpt_num', default=3, type=int)

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', type=str2bool, default=True,
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    
    # Evaluation parameters
    parser.add_argument('--crop_pct', type=float, default=None)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', type=str2bool, default=False)
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    parser.add_argument('--dist_backend', default='nccl', choices=['nccl', 'gloo'],
                        help='Distributed backend (default: nccl). Use gloo to debug NCCL issues.')
    return parser


def main(args):
    """预训练主函数.

    结构上分为 6 步: 
    1) 初始化分布式 (可选)
    2) 固定随机种子 + cudnn 配置
    3) 构建训练集 (`ImageFolder(data_path/train)`) 与 sampler/dataloader
    4) 构建模型 (FCMAE) 并包 DDP (可选) 
    5) 构建 optimizer + AMP scaler, 支持自动恢复 (resume/auto_resume) 
    6) 循环epoch: 调用 `engine_pretrain.train_one_epoch` + 保存checkpoint

    额外说明：如果你在多卡上遇到 SIGSEGV（native 崩溃），本函数会为每个 rank
    生成一个 `debug_rank*.log`，并启用 `faulthandler` 尝试打印 Python 栈，方便定位卡在哪一步。
    """

    # per-rank debug log (helps diagnose SIGSEGV / native crashes)
    rank = os.environ.get("RANK", "0")
    local_rank = os.environ.get("LOCAL_RANK", "")
    debug_fp = None
    debug_path = None

    try:
        if args.output_dir:
            os.makedirs(args.output_dir, exist_ok=True)
            debug_path = os.path.join(args.output_dir, f"debug_rank{rank}.log")
        else:
            debug_path = f"debug_rank{rank}.log"
        debug_fp = open(debug_path, "a", buffering=1, encoding="utf-8")
        faulthandler.enable(file=debug_fp, all_threads=True)
    except Exception:
        debug_fp = None

    def _dbg(msg: str):
        if debug_fp is None:
            return
        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        debug_fp.write(f"[{ts}] rank={rank} local_rank={local_rank} {msg}\n")

    _dbg("enter main")
    _dbg(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES','')} OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS','')} MKL_NUM_THREADS={os.environ.get('MKL_NUM_THREADS','')}")

    utils.init_distributed_mode(args)
    _dbg("after init_distributed_mode")

    print(args)
    device = torch.device(args.device)
    _dbg(f"device={device}")


    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    cudnn.benchmark = True
    
    # simple augmentation
    transform_train = transforms.Compose([
            transforms.Lambda(_to_rgb),
            transforms.RandomResizedCrop(args.input_size, scale=(0.2, 1.0), interpolation=3),  # 3 is bicubic
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    dataset_train = datasets.ImageFolder(os.path.join(args.data_path, 'train'), transform=transform_train)
    print(dataset_train)

    num_tasks = utils.get_world_size()
    global_rank = utils.get_rank()

    sampler_train = torch.utils.data.DistributedSampler(
        dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True, seed=args.seed,
    )
    print("Sampler_train = %s" % str(sampler_train))

    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        # log_writer = SummaryWriter(log_dir=args.log_dir)
        log_writer = utils.TensorboardLogger(log_dir=args.log_dir)
    else:
        log_writer = None

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    # define the model
    model = fcmae.__dict__[args.model](
        mask_ratio=args.mask_ratio,
        decoder_depth=args.decoder_depth,
        decoder_embed_dim=args.decoder_embed_dim,
        norm_pix_loss=args.norm_pix_loss
    )
    model.to(device)

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Model = %s" % str(model_without_ddp))
    print('number of params:', n_parameters)

    eff_batch_size = args.batch_size * args.update_freq * utils.get_world_size()
    num_training_steps_per_epoch = len(dataset_train) // eff_batch_size

    if args.lr is None:
        args.lr = args.blr * eff_batch_size / 256
        
    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.update_freq)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module

    param_groups = _timm_param_groups(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    print(optimizer)
    loss_scaler = NativeScaler()

    utils.auto_load_model(
        args=args, model=model, model_without_ddp=model_without_ddp,
        optimizer=optimizer, loss_scaler=loss_scaler)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        if log_writer is not None:
            log_writer.set_step(epoch * num_training_steps_per_epoch * args.update_freq)
        train_stats = train_one_epoch(
            model, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            log_writer=log_writer,
            args=args
        )
        if args.output_dir and args.save_ckpt:
            if (epoch + 1) % args.save_ckpt_freq == 0 or epoch + 1 == args.epochs:
                utils.save_model(
                    args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                    loss_scaler=loss_scaler, epoch=epoch)
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        'epoch': epoch,
                        'n_parameters': n_parameters}
        if args.output_dir and utils.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")
    
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)