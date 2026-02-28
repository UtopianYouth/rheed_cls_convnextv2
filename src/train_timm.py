"""基于timm的ConvNeXtV2图像分类训练脚本（RHEED任务）。

功能：
- 直接使用timm内置模型（可选ImageNet预训练权重）
- ImageFolder数据集读取（train/val/test）
- 输出Accuracy / Macro-F1 / 混淆矩阵
- 保存最佳模型

示例：
python -m src.train_timm \
  --model convnextv2_tiny \
  --train_dir data_rheed/train \
  --val_dir data_rheed/val \
  --test_dir data_rheed/test \
  --num_classes 2 \
  --input_size 224 \
  --batch_size 16 \
  --epochs 80 \
  --lr 5e-4 \
  --output_dir outputs/finetune_demo
"""

import argparse
import datetime
import json
import os
import random
import time
from pathlib import Path

from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms

import timm
from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


try:
    from sklearn.metrics import classification_report, confusion_matrix, f1_score
    _HAS_SKLEARN = True
except Exception:
    _HAS_SKLEARN = False

import numpy as np


def _to_rgb(img):
    try:
        return img.convert("RGB")
    except Exception:
        return img


def _load_image(path):
    with Image.open(path) as img:
        img = _to_rgb(img)
        return img


def summarize_dataset_stats(dataset, class_to_idx):
    stats = {}
    for i, (img, label) in enumerate(dataset):
        cls_name = list(class_to_idx.keys())[label]
        img_np = img.cpu().numpy()
        if cls_name not in stats:
            stats[cls_name] = []
        stats[cls_name].append(img_np)
        if i > 999:  # 采样1000张足够判断分布
            break
    for cls_name, arrs in stats.items():
        all_vals = np.concatenate([a.flatten() for a in arrs])
        print(f"[STATS] {cls_name}:")
        print(f"  count = {len(arrs)}")
        print(f"  mean  = {all_vals.mean():.6f}")
        print(f"  std   = {all_vals.std():.6f}")
        print(f"  min   = {all_vals.min():.6f}")
        print(f"  max   = {all_vals.max():.6f}")
    print("=" * 40)


def _list_image_files(dir_path):
    files = []
    for name in os.listdir(dir_path):
        lower = name.lower()
        if lower.endswith(".png"):
            files.append(os.path.join(dir_path, name))
    files.sort()
    return files


def _parse_split_ratio(ratio_str):
    parts = [float(p.strip()) for p in ratio_str.split(",") if p.strip() != ""]
    if len(parts) != 3:
        raise ValueError("--split_ratio 必须是3段比例，例如 0.7,0.2,0.1")
    total = sum(parts)
    if total <= 0:
        raise ValueError("--split_ratio 总和必须大于0")
    return [p / total for p in parts]


def _parse_seq_list(seq_str):
    if not seq_str:
        return []
    return [s.strip() for s in seq_str.split(",") if s.strip()]


def _normalize_seq_name(name):
    raw = name.strip()
    if raw.startswith("seq_"):
        return raw
    if raw.startswith("seq"):
        return raw
    if raw.isdigit():
        return f"seq_{int(raw):03d}"
    return raw


def _build_fixed_seq_map(seq_train, seq_val, seq_test):
    train_list = [_normalize_seq_name(s) for s in _parse_seq_list(seq_train)]
    val_list = [_normalize_seq_name(s) for s in _parse_seq_list(seq_val)]
    test_list = [_normalize_seq_name(s) for s in _parse_seq_list(seq_test)]

    all_list = train_list + val_list + test_list
    if len(all_list) != len(set(all_list)):
        raise ValueError("seq_train/seq_val/seq_test 之间存在重复序列名")

    split_map = {}
    for s in train_list:
        split_map[s] = "train"
    for s in val_list:
        split_map[s] = "val"
    for s in test_list:
        split_map[s] = "test"
    return split_map, train_list, val_list, test_list


def _build_windows(frames, start_idx, end_idx, window_size, stride):
    windows = []
    if end_idx - start_idx < window_size:
        return windows
    last_start = end_idx - window_size
    for s in range(start_idx, last_start + 1, stride):
        windows.append(frames[s:s + window_size])
    return windows


def _compute_class_weights_from_labels(labels, num_classes):
    if not labels:
        raise RuntimeError("无法从空标签列表计算类别权重")
    counts = np.bincount(np.array(labels), minlength=num_classes).astype(np.float64)
    if np.any(counts == 0):
        raise RuntimeError(f"某些类别没有样本，无法计算权重: {counts.tolist()}")
    total = counts.sum()
    weights = total / (num_classes * counts)
    return torch.tensor(weights, dtype=torch.float32)


def _print_label_distribution(name, labels, class_to_idx):
    from collections import Counter
    if labels is None:
        return
    if len(labels) == 0:
        print(f"[DATA] {name} label distribution: empty")
        return
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    counts = Counter(labels)
    pretty = {idx_to_class.get(k, k): v for k, v in counts.items()}
    print(f"[DATA] {name} label distribution: {pretty}")


def _print_confidence_stats(name, max_probs):
    if max_probs is None or len(max_probs) == 0:
        print(f"[DEBUG] {name} confidence stats: empty")
        return
    arr = np.array(max_probs, dtype=np.float32)
    qs = np.quantile(arr, [0.1, 0.25, 0.5, 0.75, 0.9])
    hist, edges = np.histogram(arr, bins=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    buckets = [f"{edges[i]:.1f}-{edges[i+1]:.1f}" for i in range(len(edges) - 1)]
    hist_dict = {buckets[i]: int(hist[i]) for i in range(len(hist))}
    print(
        f"[DEBUG] {name} confidence stats: mean={arr.mean():.4f} std={arr.std():.4f} "
        f"min={arr.min():.4f} max={arr.max():.4f} "
        f"q10={qs[0]:.4f} q25={qs[1]:.4f} q50={qs[2]:.4f} q75={qs[3]:.4f} q90={qs[4]:.4f}"
    )
    print(f"[DEBUG] {name} confidence hist: {hist_dict}")


def _print_logit_diff_stats(name, diffs):
    if diffs is None or len(diffs) == 0:
        print(f"[DEBUG] {name} logit diff stats: empty")
        return
    arr = np.array(diffs, dtype=np.float32)
    qs = np.quantile(arr, [0.1, 0.25, 0.5, 0.75, 0.9])
    bins = [-5.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 5.0]
    hist, edges = np.histogram(arr, bins=bins)
    buckets = [f"{edges[i]:.1f}-{edges[i+1]:.1f}" for i in range(len(edges) - 1)]
    hist_dict = {buckets[i]: int(hist[i]) for i in range(len(hist))}
    print(
        f"[DEBUG] {name} logit diff stats: mean={arr.mean():.4f} std={arr.std():.4f} "
        f"min={arr.min():.4f} max={arr.max():.4f} "
        f"q10={qs[0]:.4f} q25={qs[1]:.4f} q50={qs[2]:.4f} q75={qs[3]:.4f} q90={qs[4]:.4f}"
    )
    print(f"[DEBUG] {name} logit diff hist: {hist_dict}")


def build_sequence_samples(
    sequence_root,
    window_size,
    stride,
    split_ratio,
    strict_time_split,
    seed,
    seq_train=None,
    seq_val=None,
    seq_test=None,
):
    if window_size < 1:
        raise ValueError("window_size 必须 >= 1")
    if stride < 1:
        raise ValueError("window_stride 必须 >= 1")

    class_names = sorted([
        d for d in os.listdir(sequence_root)
        if os.path.isdir(os.path.join(sequence_root, d))
    ])
    if not class_names:
        raise RuntimeError(f"sequence_root下未找到类别目录: {sequence_root}")

    class_to_idx = {name: i for i, name in enumerate(class_names)}
    ratios = _parse_split_ratio(split_ratio)

    train_samples = []
    val_samples = []
    test_samples = []

    rng = random.Random(seed)

    fixed_mode = any([seq_train, seq_val, seq_test])
    split_map = {}
    if fixed_mode:
        split_map, train_list, val_list, test_list = _build_fixed_seq_map(seq_train, seq_val, seq_test)
        parts = [f"train={train_list}"]
        if val_list:
            parts.append(f"val={val_list}")
        if test_list:
            parts.append(f"test={test_list}")
        print(f"[SEQ] 固定划分: {', '.join(parts)}")

    for class_name in class_names:
        class_dir = os.path.join(sequence_root, class_name)
        seq_dirs = sorted([
            d for d in os.listdir(class_dir)
            if os.path.isdir(os.path.join(class_dir, d))
        ])
        if not seq_dirs:
            print(f"[WARN] 类别 {class_name} 下未找到序列目录")
            continue

        missing = []
        for seq in seq_dirs:
            seq_dir = os.path.join(class_dir, seq)
            frames = _list_image_files(seq_dir)
            if not frames:
                print(f"[WARN] 序列 {seq_dir} 下未找到png文件")
                continue

            label = class_to_idx[class_name]
            if fixed_mode:
                split = split_map.get(seq)
                if split is None:
                    missing.append(seq)
                    continue

                windows = _build_windows(frames, 0, len(frames), window_size, stride)
                if split == "train":
                    train_samples.extend([(w, label) for w in windows])
                elif split == "val":
                    val_samples.extend([(w, label) for w in windows])
                else:
                    test_samples.extend([(w, label) for w in windows])
            else:
                if strict_time_split:
                    n = len(frames)
                    train_end = int(n * ratios[0])
                    val_end = train_end + int(n * ratios[1])

                    train_windows = _build_windows(frames, 0, train_end, window_size, stride)
                    val_windows = _build_windows(frames, train_end, val_end, window_size, stride)
                    test_windows = _build_windows(frames, val_end, n, window_size, stride)

                    train_samples.extend([(w, label) for w in train_windows])
                    val_samples.extend([(w, label) for w in val_windows])
                    test_samples.extend([(w, label) for w in test_windows])
                else:
                    windows = _build_windows(frames, 0, len(frames), window_size, stride)
                    rng.shuffle(windows)
                    n_win = len(windows)
                    train_end = int(n_win * ratios[0])
                    val_end = train_end + int(n_win * ratios[1])

                    train_samples.extend([(w, label) for w in windows[:train_end]])
                    val_samples.extend([(w, label) for w in windows[train_end:val_end]])
                    test_samples.extend([(w, label) for w in windows[val_end:]])

        if fixed_mode and missing:
            raise RuntimeError(f"固定划分模式下，以下序列未被分配到train/val/test: {sorted(missing)}")

    return train_samples, val_samples, test_samples, class_to_idx


class SequenceWindowDataset(torch.utils.data.Dataset):
    def __init__(self, samples, transform):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        frame_paths, label = self.samples[idx]
        frames = []
        for path in frame_paths:
            img = _load_image(path)
            if self.transform is not None:
                img = self.transform(img)
            frames.append(img)

        if not frames:
            raise RuntimeError("空窗口样本，检查window_size/stride设置")

        images = torch.cat(frames, dim=0)
        return images, label

def build_train_transform(args):
    # 极简增强：排除RandAug和color_jitter，避免破坏rough类纹理
    if args.aa is not None and args.aa.lower() != "none":
        args.aa = None
    if args.color_jitter is not None:
        args.color_jitter = None
    if args.reprob is not None:
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


def build_eval_transform(args):
    t = [transforms.Lambda(_to_rgb)]
    if args.input_size >= 384:
        t.append(transforms.Resize((args.input_size, args.input_size), interpolation=transforms.InterpolationMode.BICUBIC))
    else:
        size = int(args.input_size / args.crop_pct)
        t.append(transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC))
        t.append(transforms.CenterCrop(args.input_size))
    t.extend([
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
    ])
    return transforms.Compose(t)


def parse_args():
    parser = argparse.ArgumentParser("RHEED classification with timm ConvNeXtV2")

    # data
    parser.add_argument("--train_dir", type=str, default=None)
    parser.add_argument("--val_dir", type=str, default=None)
    parser.add_argument("--test_dir", type=str, default=None)
    parser.add_argument("--num_classes", type=int, default=2)

    # sequence mode
    parser.add_argument("--sequence_mode", type=lambda x: x.lower() == "true", default=False)
    parser.add_argument("--sequence_root", type=str, default=None)
    parser.add_argument("--window_size", type=int, default=10)
    parser.add_argument("--window_stride", type=int, default=1)
    parser.add_argument("--split_ratio", type=str, default="0.7,0.2,0.1")
    parser.add_argument("--strict_time_split", type=lambda x: x.lower() == "true", default=True)
    parser.add_argument("--seq_train", type=str, default=None, help="逗号分隔序列名，如 seq_001,seq_002")
    parser.add_argument("--seq_val", type=str, default=None, help="逗号分隔序列名，如 seq_008")
    parser.add_argument("--seq_test", type=str, default=None, help="逗号分隔序列名，如 seq_009,seq_010")

    # model
    parser.add_argument("--model", type=str, default="convnextv2_tiny")
    parser.add_argument("--pretrained", type=lambda x: x.lower() == "true", default=False)
    parser.add_argument("--pretrained_weights", type=str, default="")

    # train
    parser.add_argument("--input_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--pin_memory", type=lambda x: x.lower() == "true", default=True)
    parser.add_argument("--persistent_workers", type=lambda x: x.lower() == "true", default=True)
    parser.add_argument("--prefetch_factor", type=int, default=2)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_amp", type=lambda x: x.lower() == "true", default=True)
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--class_weights", type=float, nargs="+", default=None)
    parser.add_argument("--auto_class_weights", type=lambda x: x.lower() == "true", default=True)
    parser.add_argument("--balanced_sampler", type=lambda x: x.lower() == "true", default=False)
    parser.add_argument("--debug_pred_stats", type=lambda x: x.lower() == "true", default=False)
    parser.add_argument("--use_data_parallel", type=lambda x: x.lower() == "true", default=False)

    # aug
    parser.add_argument("--aa", type=str, default="rand-m9-mstd0.5-inc1")
    parser.add_argument("--color_jitter", type=float, default=None)
    parser.add_argument("--train_interpolation", type=str, default="bicubic")
    parser.add_argument("--reprob", type=float, default=0.25)
    parser.add_argument("--remode", type=str, default="pixel")
    parser.add_argument("--recount", type=int, default=1)
    parser.add_argument("--crop_pct", type=float, default=224 / 256)

    # output
    parser.add_argument("--output_dir", type=str, default="outputs/finetune_timm")
    parser.add_argument("--log_dir", type=str, default=None)
    parser.add_argument("--save_best_metric", type=str, default="acc1", choices=["acc1", "f1"])

    return parser.parse_args()


def _save_checkpoint(path, model, epoch, args, best_metric):
    model_to_save = model.module if hasattr(model, "module") else model
    state = {
        "epoch": epoch,
        "model": model_to_save.state_dict(),
        "args": vars(args),
        "best_metric": best_metric,
    }
    torch.save(state, path)


def _adapt_first_conv(state_dict, model_state, in_chans):
    for k, v in state_dict.items():
        if k not in model_state:
            continue
        if v.ndim == 4 and model_state[k].ndim == 4:
            if v.shape[1] == 3 and model_state[k].shape[1] == in_chans:
                if in_chans % 3 != 0:
                    raise ValueError(f"in_chans={in_chans} 不能被3整除，无法通道扩展")
                repeat = in_chans // 3
                state_dict[k] = v.repeat(1, repeat, 1, 1) / repeat
                print(f"[PRETRAIN] 扩展首层卷积通道: {k} 3 -> {in_chans}")
                break


def _load_imagenet_pretrained(model, model_name, in_chans):
    ref_model = timm.create_model(
        model_name,
        pretrained=True,
        num_classes=1000,
        in_chans=3,
    )
    state_dict = ref_model.state_dict()
    model_state = model.state_dict()

    _adapt_first_conv(state_dict, model_state, in_chans)

    for k in list(state_dict.keys()):
        if k not in model_state:
            del state_dict[k]
            continue
        if state_dict[k].shape != model_state[k].shape:
            del state_dict[k]

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"Loaded ImageNet pretrained weights for {model_name}")
    print(f"Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")


def _load_pretrained(model, weight_path):
    ckpt = torch.load(weight_path, map_location="cpu")
    if isinstance(ckpt, dict) and "model" in ckpt:
        state_dict = ckpt["model"]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    else:
        state_dict = ckpt

    # remove mismatched shapes (e.g., head or first conv in sequence mode)
    model_state = model.state_dict()
    for k in list(state_dict.keys()):
        if k in model_state and state_dict[k].shape != model_state[k].shape:
            del state_dict[k]

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"Loaded pretrained weights from {weight_path}")
    print(f"Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, label_smoothing=0.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.reduction = reduction

    def forward(self, logits, targets):
        n_classes = logits.size(-1)
        log_probs = F.log_softmax(logits, dim=-1)

        # label smoothing
        if self.label_smoothing > 0:
            smooth_targets = torch.full_like(logits, self.label_smoothing / (n_classes - 1))
            smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.label_smoothing)
        else:
            smooth_targets = targets

        pt = torch.exp(log_probs)
        pt = pt.gather(1, targets.unsqueeze(1)).squeeze(1)

        loss = -((1 - pt) ** self.gamma) * log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)

        if self.alpha is not None:
            alpha_t = self.alpha.gather(0, targets)
            loss = loss * alpha_t

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


def _accuracy(output, target):
    """兼容旧代码：返回该 batch 的 acc1（百分比）。"""
    with torch.no_grad():
        pred = output.argmax(dim=1)
        acc = (pred == target).float().mean().item() * 100.0
    return acc


def train_one_epoch(model, loader, optimizer, criterion, device, use_amp):
    model.train()
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    # 用“按样本计数”的方式统计，避免“按 batch 平均”带来的歧义
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        if use_amp:
            with torch.amp.autocast('cuda'):
                outputs = model(images)
                loss = criterion(outputs, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        batch_size = targets.size(0)
        total_loss += loss.item() * batch_size

        with torch.no_grad():
            preds = outputs.argmax(dim=1)
            total_correct += int((preds == targets).sum().item())
            total_samples += int(batch_size)

    avg_loss = total_loss / max(total_samples, 1)
    acc1 = (total_correct / max(total_samples, 1)) * 100.0

    return {
        "loss": float(avg_loss),
        "acc1": float(acc1),
    }


def evaluate(model, loader, criterion, device, use_amp, split_name="val", debug_pred_stats=False, epoch=None):
    model.eval()

    # 用“按样本计数”的方式统计，避免“按 batch 平均”带来的歧义
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    all_preds = []
    all_targets = []
    all_max_probs = []
    all_logit_diffs = []

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            if use_amp:
                with torch.amp.autocast('cuda'):
                    outputs = model(images)
                    loss = criterion(outputs, targets)
            else:
                outputs = model(images)
                loss = criterion(outputs, targets)

            batch_size = targets.size(0)
            total_loss += loss.item() * batch_size

            preds_t = outputs.argmax(dim=1)
            total_correct += int((preds_t == targets).sum().item())
            total_samples += int(batch_size)

            preds = preds_t.detach().cpu().tolist()
            tars = targets.detach().cpu().tolist()
            all_preds.extend(preds)
            all_targets.extend(tars)

            if debug_pred_stats:
                probs = torch.softmax(outputs, dim=1)
                max_probs = probs.max(dim=1).values.detach().cpu().tolist()
                all_max_probs.extend(max_probs)

                logits_cpu = outputs.detach().float().cpu()
                if logits_cpu.size(1) == 2:
                    diffs = (logits_cpu[:, 1] - logits_cpu[:, 0]).tolist()
                else:
                    top2 = logits_cpu.topk(2, dim=1).values
                    diffs = (top2[:, 0] - top2[:, 1]).tolist()
                all_logit_diffs.extend(diffs)

    avg_loss = total_loss / max(total_samples, 1)
    acc1 = (total_correct / max(total_samples, 1)) * 100.0

    stats = {
        "loss": float(avg_loss),
        "acc1": float(acc1),
    }

    # 额外一致性校验（不依赖sklearn）：acc1 必须与 all_preds/all_targets 完全一致
    if len(all_targets) > 0:
        acc_check = (sum(int(p == t) for p, t in zip(all_preds, all_targets)) / len(all_targets)) * 100.0
        if abs(acc_check - stats["acc1"]) > 1e-6:
            if epoch is None:
                tag = split_name
            elif isinstance(epoch, int):
                tag = f"{split_name}@{epoch:03d}"
            else:
                tag = f"{split_name}@{epoch}"
            print(
                f"[WARN] {tag} acc1不一致: stats.acc1={stats['acc1']:.6f} vs list_check={acc_check:.6f}. "
                "可能是日志输出错位/混跑/代码被修改导致。",
                flush=True,
            )

    if _HAS_SKLEARN:
        stats["f1_macro"] = float(f1_score(all_targets, all_preds, average="macro")) * 100.0

        # debug: print prediction distribution（带epoch标识，避免日志对不上）
        from collections import Counter
        if epoch is None:
            tag = split_name
        elif isinstance(epoch, int):
            tag = f"{split_name}@{epoch:03d}"
        else:
            tag = f"{split_name}@{epoch}"
        pred_counts = Counter(all_preds)
        print(f"[DEBUG] {tag} 预测类别分布: {dict(pred_counts)}", flush=True)
        print(f"[DEBUG] {tag} 真实类别分布: {dict(Counter(all_targets))}", flush=True)
        stats["confusion_matrix"] = confusion_matrix(all_targets, all_preds).tolist()

    if debug_pred_stats:
        _print_confidence_stats(split_name, all_max_probs)
        _print_logit_diff_stats(split_name, all_logit_diffs)

    return stats


def main():
    args = parse_args()

    if args.pretrained_weights:
        if not os.path.isfile(args.pretrained_weights):
            raise RuntimeError(f"预训练权重文件不存在: {args.pretrained_weights}")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = True

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_transform = build_train_transform(args)
    eval_transform = build_eval_transform(args)

    if args.sequence_mode:
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

        if args.num_classes != len(class_to_idx):
            raise RuntimeError(
                f"num_classes={args.num_classes} 与数据类别数 {len(class_to_idx)} 不一致: {class_to_idx}"
            )

        if len(train_samples) == 0 or len(val_samples) == 0:
            raise RuntimeError("训练/验证样本为空，检查window_size或split_ratio设置")

        print(f"[SEQ] class_to_idx: {class_to_idx}")
        print(
            f"[SEQ] samples: train={len(train_samples)}, val={len(val_samples)}, test={len(test_samples)}"
        )

        train_set = SequenceWindowDataset(train_samples, train_transform)
        val_set = SequenceWindowDataset(val_samples, eval_transform)
        test_set = SequenceWindowDataset(test_samples, eval_transform) if len(test_samples) > 0 else None
        train_labels = [label for _, label in train_samples]
        val_labels = [label for _, label in val_samples]
        test_labels = [label for _, label in test_samples]
    else:
        if not args.train_dir or not args.val_dir:
            raise RuntimeError("非序列模式必须提供 --train_dir 和 --val_dir")

        train_set = datasets.ImageFolder(args.train_dir, transform=train_transform)
        val_set = datasets.ImageFolder(args.val_dir, transform=eval_transform)

        if train_set.class_to_idx != val_set.class_to_idx:
            raise RuntimeError(f"train/val class_to_idx mismatch: {train_set.class_to_idx} vs {val_set.class_to_idx}")

        test_set = None
        if args.test_dir and os.path.isdir(args.test_dir):
            test_set = datasets.ImageFolder(args.test_dir, transform=eval_transform)
            if test_set.class_to_idx != train_set.class_to_idx:
                raise RuntimeError("test class_to_idx mismatch with train")

        class_to_idx = train_set.class_to_idx
        train_labels = train_set.targets
        val_labels = val_set.targets
        test_labels = test_set.targets if test_set is not None else []
        print(f"class_to_idx: {class_to_idx}")

    _print_label_distribution("train", train_labels, class_to_idx)
    _print_label_distribution("val", val_labels, class_to_idx)
    if test_set is not None:
        _print_label_distribution("test", test_labels, class_to_idx)

    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": args.pin_memory,
    }
    if args.num_workers > 0:
        loader_kwargs["persistent_workers"] = args.persistent_workers
        loader_kwargs["prefetch_factor"] = args.prefetch_factor

    train_sampler = None
    if args.balanced_sampler:
        counts = np.bincount(np.array(train_labels), minlength=args.num_classes).astype(np.float64)
        if np.any(counts == 0):
            raise RuntimeError(f"训练集中存在空类别，无法使用balanced_sampler: {counts.tolist()}")
        class_weights = 1.0 / counts
        sample_weights = torch.tensor(class_weights[np.array(train_labels)], dtype=torch.float32)
        train_sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
        print(f"[DEBUG] Using WeightedRandomSampler with class_counts: {counts.tolist()}")

    train_loader = DataLoader(
        train_set, shuffle=(train_sampler is None), sampler=train_sampler, **loader_kwargs
    )
    val_loader = DataLoader(
        val_set, shuffle=False, **loader_kwargs
    )
    test_loader = None
    if test_set is not None:
        test_loader = DataLoader(
            test_set, shuffle=False, **loader_kwargs
        )

    # build model
    in_chans = 3 * args.window_size if args.sequence_mode else 3

    model = timm.create_model(
        args.model,
        pretrained=False,
        num_classes=args.num_classes,
        in_chans=in_chans,
    )

    if args.pretrained and not args.pretrained_weights:
        _load_imagenet_pretrained(model, args.model, in_chans)

    if args.pretrained_weights:
        _load_pretrained(model, args.pretrained_weights)

    if args.use_data_parallel and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    model.to(device)

    if args.class_weights is not None:
        if len(args.class_weights) != args.num_classes:
            raise ValueError("--class_weights长度必须与num_classes一致")
        class_weights = torch.tensor(args.class_weights, dtype=torch.float32, device=device)
        criterion = FocalLoss(alpha=class_weights, gamma=2.0, label_smoothing=args.label_smoothing)
        print(f"[DEBUG] Using FocalLoss with manual class weights: {class_weights.tolist()}")
    elif args.auto_class_weights:
        class_weights = _compute_class_weights_from_labels(train_labels, args.num_classes).to(device)
        criterion = FocalLoss(alpha=class_weights, gamma=2.0, label_smoothing=args.label_smoothing)
        print(f"[DEBUG] Using FocalLoss with auto class weights: {class_weights.tolist()}")
    else:
        criterion = FocalLoss(gamma=2.0, label_smoothing=args.label_smoothing)
        print("[DEBUG] Using FocalLoss (no class weights)")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    print(f"[DEBUG] Optimizer: AdamW(lr={args.lr}, weight_decay={args.weight_decay})")

    # DEBUG: 统计训练集数值分布（序列模式下跳过）
    if not args.sequence_mode:
        train_dataset_for_stat = datasets.ImageFolder(args.train_dir, transform=build_eval_transform(args))
        summarize_dataset_stats(train_dataset_for_stat, train_dataset_for_stat.class_to_idx)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "log.txt"

    best_metric = -1.0
    best_epoch = -1

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        train_stats = train_one_epoch(
            model, train_loader, optimizer, criterion, device, args.use_amp
        )
        val_stats = evaluate(
            model, val_loader, criterion, device, args.use_amp,
            split_name="val", debug_pred_stats=args.debug_pred_stats, epoch=epoch
        )

        metric_value = val_stats["acc1"] if args.save_best_metric == "acc1" else val_stats.get("f1_macro", 0.0)
        if metric_value > best_metric:
            best_metric = metric_value
            best_epoch = epoch
            _save_checkpoint(output_dir / "checkpoint_best.pth", model, epoch, args, best_metric)

        # save last checkpoint
        _save_checkpoint(output_dir / "checkpoint_last.pth", model, epoch, args, best_metric)

        log_stats = {
            "epoch": epoch,
            "train": train_stats,
            "val": val_stats,
            "best_metric": best_metric,
            "best_epoch": best_epoch,
        }

        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(log_stats, ensure_ascii=False) + "\n")

        print(
            f"Epoch {epoch:03d}: "
            f"train_loss={train_stats['loss']:.4f} train_acc1={train_stats['acc1']:.2f} | "
            f"val_loss={val_stats['loss']:.4f} val_acc1={val_stats['acc1']:.2f} "
            f"val_f1={val_stats.get('f1_macro', 0.0):.2f}"
        )

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Training time: {total_time_str}")

    if test_loader is not None:
        test_stats = evaluate(
            model, test_loader, criterion, device, args.use_amp,
            split_name="test", debug_pred_stats=args.debug_pred_stats, epoch="test"
        )
        print(f"Test acc1={test_stats['acc1']:.2f} f1={test_stats.get('f1_macro', 0.0):.2f}")
        with (output_dir / "test_metrics.json").open("w", encoding="utf-8") as f:
            json.dump(test_stats, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
