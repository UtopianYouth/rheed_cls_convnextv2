"""预训练训练循环（FCMAE / Masked AutoEncoder）。

这个文件只做一件事：定义 `train_one_epoch`。
- `main_pretrain.py` 会构建 dataloader / optimizer / scaler 后调用它。
- 预训练的 loss 来自 FCMAE 的重建误差（只在 mask 的 patch 上计算）。

注意：这里的 `labels` 在预训练阶段并不参与分类，只是为了兼容 `ImageFolder` 的返回值。
"""

import math
import sys
from typing import Iterable


import torch

from src import utils

def train_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    log_writer=None,
                    args=None):
    """预训练阶段训练一个 epoch（FCMAE）。

    这里的训练和监督分类不同：
    - `data_loader` 来自 `ImageFolder`，会产出 `(samples, labels)`，但 `labels` 在自监督中不参与 loss。
    - loss 由 `model(samples, labels, mask_ratio=...)` 返回的重建误差决定（仅 mask 区域）。
    - 学习率采用 **iteration 级别** 的 cosine schedule（见 `utils.adjust_learning_rate`）。
    - 支持 `update_freq` 梯度累积；每累积完成一次会清理一次 CUDA cache（降低稀疏算子碎片）。

    Returns:
        dict: 统计信息（loss、lr）。
    """

    model.train(True)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    update_freq = args.update_freq

    optimizer.zero_grad()
    for data_iter_step, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # batch can be (samples, labels) or (samples, labels, paths)
        paths = None
        if isinstance(batch, (list, tuple)) and len(batch) == 3:
            samples, labels, paths = batch
        else:
            samples, labels = batch

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % update_freq == 0:
            utils.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        # Detect NaN/Inf early (on CPU) to localize the root cause to data/augmentation.
        if not isinstance(samples, list):
            try:
                if torch.is_tensor(samples):
                    finite = torch.isfinite(samples)
                    if not finite.all().item():
                        # find which samples in the batch are bad
                        per_sample_finite = finite.reshape(finite.shape[0], -1).all(dim=1)
                        bad_idx = (~per_sample_finite).nonzero(as_tuple=False).view(-1).tolist()
                        nan_cnt = int(torch.isnan(samples).sum().item())
                        inf_cnt = int(torch.isinf(samples).sum().item())

                        print(
                            f"[DATA ERROR] non-finite samples BEFORE .to(device) at epoch={epoch} iter={data_iter_step} "
                            f"(bad_idx={bad_idx}, nan_cnt={nan_cnt}, inf_cnt={inf_cnt})"
                        )
                        if paths is not None:
                            try:
                                bad_paths = [paths[i] for i in bad_idx]
                            except Exception:
                                bad_paths = paths
                            print(f"[DATA ERROR] bad_paths: {bad_paths}")

                        action = getattr(args, 'bad_sample_action', 'error')
                        if action == 'zero':
                            samples = torch.nan_to_num(samples, nan=0.0, posinf=0.0, neginf=0.0)
                            print(f"[DATA WARN] bad_sample_action=zero applied; training continues.")
                        else:
                            sys.exit(1)
            except Exception:
                # if anything goes wrong in the diagnostic path, don't block training
                pass

            samples = samples.to(device, non_blocking=True)


        labels = labels.to(device, non_blocking=True)

        loss, _, _ = model(samples, labels, mask_ratio=args.mask_ratio)


        loss_value = loss.item()

        if not math.isfinite(loss_value):
            lr = optimizer.param_groups[0].get("lr", float('nan'))
            rank = utils.get_rank() if hasattr(utils, 'get_rank') else 0
            # try to provide actionable debugging info
            try:
                s_min = float(samples.min().item())
                s_max = float(samples.max().item())
                s_finite = bool(torch.isfinite(samples).all().item())
            except Exception:
                s_min, s_max, s_finite = float('nan'), float('nan'), False

            print(
                "Loss is {}, stopping training. "
                "(rank={}, epoch={}, iter={}, lr={}, samples_finite={}, samples_min={}, samples_max={})".format(
                    loss_value, rank, epoch, data_iter_step, lr, s_finite, s_min, s_max
                )
            )
            if paths is not None:
                print(f"[DATA TRACE] paths: {paths}")
            print("Tip: try enabling grad clipping (--clip_grad 3.0) or lowering --blr if this repeats.")
            sys.exit(1)


    
        loss /= update_freq
        loss_scaler(
            loss,
            optimizer,
            clip_grad=getattr(args, 'clip_grad', None),
            parameters=model.parameters(),
            update_grad=(data_iter_step + 1) % update_freq == 0,
        )
        if (data_iter_step + 1) % update_freq == 0:
            optimizer.zero_grad()

            # MinkowskiEngine + CUDA 下频繁 empty_cache 可能引发 native 不稳定；改为可控频率，并在前面 synchronize。
            empty_cache_freq = int(getattr(args, 'empty_cache_freq', 0) or 0)
            if empty_cache_freq > 0 and (data_iter_step + 1) % empty_cache_freq == 0 and torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()

        
        metric_logger.update(loss=loss_value)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)
        
        loss_value_reduce = utils.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % update_freq == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.update(train_loss=loss_value_reduce, head="loss", step=epoch_1000x)
            log_writer.update(lr=lr, head="opt", step=epoch_1000x)
    
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}