"""RHEED sequence self-supervised pretraining (SimCLR-style)."""

import argparse
import datetime
import json
import os
import time
from pathlib import Path

from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms

import timm
from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

from src.train_timm import build_sequence_samples, _to_rgb, _load_image


class SSLSequenceWindowDataset(torch.utils.data.Dataset):
    def __init__(self, samples, transform1, transform2):
        self.samples = samples
        self.transform1 = transform1
        self.transform2 = transform2

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        frame_paths, _ = self.samples[idx]
        frames1 = []
        frames2 = []
        for path in frame_paths:
            img = _load_image(path)
            img1 = self.transform1(img) if self.transform1 is not None else img
            img2 = self.transform2(img) if self.transform2 is not None else img
            frames1.append(img1)
            frames2.append(img2)
        images1 = torch.cat(frames1, dim=0)
        images2 = torch.cat(frames2, dim=0)
        return images1, images2


class ProjectionHead(nn.Module):
    def __init__(self, in_dim, proj_dim=128, hidden_dim=2048):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, proj_dim),
        )

    def forward(self, x):
        return self.net(x)


class SSLModel(nn.Module):
    def __init__(self, backbone, proj_dim=128, hidden_dim=2048):
        super().__init__()
        self.backbone = backbone
        self.projector = ProjectionHead(backbone.num_features, proj_dim, hidden_dim)

    def forward(self, x):
        feats = self.backbone(x)
        if feats.dim() > 2:
            feats = feats.flatten(1)
        z = self.projector(feats)
        return z


def nt_xent_loss(z1, z2, temperature):
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    z = torch.cat([z1, z2], dim=0)
    sim = torch.matmul(z, z.T) / temperature
    batch_size = z1.size(0)
    mask = torch.eye(2 * batch_size, device=z.device, dtype=torch.bool)
    sim = sim.masked_fill(mask, torch.finfo(sim.dtype).min)
    labels = torch.arange(batch_size, device=z.device)
    labels = torch.cat([labels + batch_size, labels], dim=0)
    loss = F.cross_entropy(sim, labels)
    return loss


def build_ssl_transform(args):
    if args.aa is not None and args.aa.lower() == "none":
        args.aa = None
    if args.color_jitter is None:
        args.color_jitter = 0.4
    if args.reprob is None:
        args.reprob = 0.0

    transform = create_transform(
        input_size=args.input_size,
        is_training=True,
        color_jitter=args.color_jitter,
        auto_augment=args.aa,
        interpolation=args.train_interpolation,
        re_prob=args.reprob,
        re_mode=args.remode,
        re_count=args.recount,
        mean=IMAGENET_DEFAULT_MEAN,
        std=IMAGENET_DEFAULT_STD,
    )
    transform.transforms.insert(0, transforms.Lambda(_to_rgb))
    return transform


def parse_args():
    parser = argparse.ArgumentParser("RHEED sequence SSL pretraining (SimCLR)")

    parser.add_argument("--sequence_root", type=str, required=True)
    parser.add_argument("--window_size", type=int, default=10)
    parser.add_argument("--window_stride", type=int, default=2)
    parser.add_argument("--split_ratio", type=str, default="0.7,0.2,0.1")
    parser.add_argument("--strict_time_split", type=lambda x: x.lower() == "true", default=True)
    parser.add_argument("--seq_train", type=str, default=None)
    parser.add_argument("--seq_val", type=str, default=None)
    parser.add_argument("--seq_test", type=str, default=None)

    parser.add_argument("--model", type=str, default="convnextv2_tiny")
    parser.add_argument("--pretrained", type=lambda x: x.lower() == "true", default=False)
    parser.add_argument("--pretrained_weights", type=str, default="")

    parser.add_argument("--input_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--pin_memory", type=lambda x: x.lower() == "true", default=True)
    parser.add_argument("--persistent_workers", type=lambda x: x.lower() == "true", default=True)
    parser.add_argument("--prefetch_factor", type=int, default=2)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_amp", type=lambda x: x.lower() == "true", default=True)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--proj_dim", type=int, default=128)
    parser.add_argument("--proj_hidden_dim", type=int, default=2048)

    parser.add_argument("--aa", type=str, default="rand-m9-mstd0.5-inc1")
    parser.add_argument("--color_jitter", type=float, default=None)
    parser.add_argument("--train_interpolation", type=str, default="bicubic")
    parser.add_argument("--reprob", type=float, default=0.0)
    parser.add_argument("--remode", type=str, default="pixel")
    parser.add_argument("--recount", type=int, default=1)

    parser.add_argument("--output_dir", type=str, default="outputs/pretrain_ssl")

    return parser.parse_args()


def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = True

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if not args.sequence_root or not os.path.isdir(args.sequence_root):
        raise RuntimeError("--sequence_root 必须是有效目录")

    train_samples, val_samples, test_samples, class_to_idx = build_sequence_samples(
        args.sequence_root,
        args.window_size,
        args.window_stride,
        args.split_ratio,
        args.strict_time_split,
        args.seed,
        args.seq_train,
        args.seq_val,
        args.seq_test,
    )

    all_samples = train_samples
    if len(all_samples) == 0:
        raise RuntimeError("预训练样本为空，检查window_size或序列划分")

    print(f"[SEQ-SSL] class_to_idx: {class_to_idx}")
    print(f"[SEQ-SSL] samples: train={len(train_samples)}, val={len(val_samples)}, test={len(test_samples)}, all={len(all_samples)}")

    transform1 = build_ssl_transform(args)
    transform2 = build_ssl_transform(args)

    dataset = SSLSequenceWindowDataset(all_samples, transform1, transform2)

    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": args.pin_memory,
    }
    if args.num_workers > 0:
        loader_kwargs["persistent_workers"] = args.persistent_workers
        loader_kwargs["prefetch_factor"] = args.prefetch_factor

    loader = DataLoader(dataset, shuffle=True, **loader_kwargs)

    in_chans = 3 * args.window_size
    pretrained_flag = args.pretrained and not args.pretrained_weights
    if in_chans != 3 and pretrained_flag:
        print("[WARN] in_chans != 3，已禁用timm预训练权重")
        pretrained_flag = False

    backbone = timm.create_model(
        args.model,
        pretrained=pretrained_flag,
        num_classes=0,
        in_chans=in_chans,
        global_pool="avg",
    )

    if args.pretrained_weights:
        ckpt = torch.load(args.pretrained_weights, map_location="cpu")
        if isinstance(ckpt, dict) and "model" in ckpt:
            state_dict = ckpt["model"]
        else:
            state_dict = ckpt
        missing, unexpected = backbone.load_state_dict(state_dict, strict=False)
        print(f"Loaded pretrained weights from {args.pretrained_weights}")
        print(f"Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")

    model = SSLModel(backbone, proj_dim=args.proj_dim, hidden_dim=args.proj_hidden_dim)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    print(f"[DEBUG] Optimizer: AdamW(lr={args.lr}, weight_decay={args.weight_decay})")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "log.txt"

    print(f"Start pretraining for {args.epochs} epochs")
    start_time = time.time()

    scaler = torch.amp.GradScaler('cuda', enabled=args.use_amp)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        num_batches = 0

        for images1, images2 in loader:
            images1 = images1.to(device, non_blocking=True)
            images2 = images2.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            if args.use_amp:
                with torch.amp.autocast('cuda'):
                    z1 = model(images1)
                    z2 = model(images2)
                    loss = nt_xent_loss(z1, z2, args.temperature)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                z1 = model(images1)
                z2 = model(images2)
                loss = nt_xent_loss(z1, z2, args.temperature)
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        epoch_loss = total_loss / max(num_batches, 1)
        print(f"Epoch {epoch:03d}: ssl_loss={epoch_loss:.4f}")

        log_stats = {
            "epoch": epoch,
            "ssl_loss": epoch_loss,
        }
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(log_stats, ensure_ascii=False) + "\n")

        ckpt = {
            "epoch": epoch,
            "model": backbone.state_dict(),
            "ssl_model": model.state_dict(),
            "args": vars(args),
        }
        torch.save(ckpt, output_dir / "checkpoint_last.pth")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Pretraining time: {total_time_str}")


if __name__ == "__main__":
    main()
