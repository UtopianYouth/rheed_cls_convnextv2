#!/usr/bin/env python3
"""
对比2D/3D类别的seq_001图像统计特征：
- 平均图像（mean image）
- 灰度直方图（0-255）
- 边缘能量（Sobel）

用法：
python -m src.compare_seq001_stats \
  --sequence_root /home/utopianyouth/rheed_cls_convnextv2/rheed_images \
  --output_dir /home/utopianyouth/rheed_cls_convnextv2/outputs/seq001_stats
"""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


SUPPORTED_EXTS = (".tif", ".tiff", ".png", ".jpg", ".jpeg")


def parse_args():
    parser = argparse.ArgumentParser(description="统计对比seq_001图像特征")
    parser.add_argument("--sequence_root", type=str, required=True, help="rheed_images根目录")
    parser.add_argument("--seq_name", type=str, default="seq_001", help="序列名")
    parser.add_argument("--class_a", type=str, default="2d", help="类别A目录名")
    parser.add_argument("--class_b", type=str, default="3d", help="类别B目录名")
    parser.add_argument("--bins", type=int, default=256, help="直方图bin数量")
    parser.add_argument("--output_dir", type=str, required=True, help="输出目录")
    return parser.parse_args()


def list_images(folder: Path):
    files = [p for p in folder.iterdir() if p.suffix.lower() in SUPPORTED_EXTS]
    files.sort()
    if not files:
        raise FileNotFoundError(f"未找到图像文件：{folder}")
    return files


def _linear_normalize(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float32)
    vmin = float(arr.min())
    vmax = float(arr.max())
    if vmax <= vmin:
        return np.zeros_like(arr, dtype=np.uint8)
    norm = (arr - vmin) / (vmax - vmin)
    norm = np.clip(norm * 255.0, 0, 255).astype(np.uint8)
    return norm


def _load_gray_image(fp: Path) -> np.ndarray:
    suffix = fp.suffix.lower()
    with Image.open(fp) as im:
        if suffix in (".tif", ".tiff"):
            im = im.convert("I")
            arr = np.array(im)
            arr = _linear_normalize(arr)
            return arr.astype(np.float32)
        im = im.convert("L")
        return np.array(im, dtype=np.float32)


def conv2d(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    kh, kw = kernel.shape
    pad = kh // 2
    padded = np.pad(img, ((pad, pad), (pad, pad)), mode="edge")
    h, w = img.shape
    out = np.zeros((h, w), dtype=np.float32)
    for i in range(kh):
        for j in range(kw):
            out += kernel[i, j] * padded[i:i + h, j:j + w]
    return out


def sobel_edge_energy(img: np.ndarray) -> float:
    kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    ky = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)
    gx = conv2d(img, kx)
    gy = conv2d(img, ky)
    mag = np.sqrt(gx ** 2 + gy ** 2)
    return float(mag.mean())


def process_class(seq_dir: Path, bins: int):
    files = list_images(seq_dir)
    sum_img = None
    hist = np.zeros((bins,), dtype=np.float64)
    edge_energies = []

    for fp in files:
        arr = _load_gray_image(fp)

        if sum_img is None:
            sum_img = np.zeros_like(arr, dtype=np.float64)

        sum_img += arr
        hist += np.histogram(arr, bins=bins, range=(0, 255))[0]
        edge_energies.append(sobel_edge_energy(arr))

    mean_img = sum_img / len(files)
    hist = hist / hist.sum()
    edge_mean = float(np.mean(edge_energies))

    return {
        "num": len(files),
        "mean_img": mean_img,
        "hist": hist,
        "edge_mean": edge_mean,
    }


def plot_results(stats_a, stats_b, class_a, class_b, save_path: Path):
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), dpi=300)

    ax = axes[0, 0]
    ax.imshow(stats_a["mean_img"], cmap="gray")
    ax.set_title(f"Mean Image - {class_a}")
    ax.axis("off")

    ax = axes[0, 1]
    ax.imshow(stats_b["mean_img"], cmap="gray")
    ax.set_title(f"Mean Image - {class_b}")
    ax.axis("off")

    ax = axes[1, 0]
    xs = np.arange(len(stats_a["hist"]))
    ax.plot(xs, stats_a["hist"], label=class_a, color="#1f77b4", linewidth=1.5)
    ax.plot(xs, stats_b["hist"], label=class_b, color="#d62728", linewidth=1.5)
    ax.set_title("Gray Histogram (normalized)")
    ax.set_xlabel("Gray value")
    ax.set_ylabel("Probability")
    ax.legend()

    ax = axes[1, 1]
    ax.bar([class_a, class_b], [stats_a["edge_mean"], stats_b["edge_mean"]], color=["#1f77b4", "#d62728"])
    ax.set_title("Edge Energy (Sobel mean)")
    ax.set_ylabel("Mean edge magnitude")

    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    root = Path(args.sequence_root)

    dir_a = root / args.class_a / args.seq_name
    dir_b = root / args.class_b / args.seq_name

    stats_a = process_class(dir_a, args.bins)
    stats_b = process_class(dir_b, args.bins)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_path = out_dir / f"seq001_stats_{args.class_a}_vs_{args.class_b}.png"
    plot_results(stats_a, stats_b, args.class_a, args.class_b, fig_path)

    print("=== Seq001 Stats Summary ===")
    print(f"{args.class_a}: num={stats_a['num']} edge_mean={stats_a['edge_mean']:.4f}")
    print(f"{args.class_b}: num={stats_b['num']} edge_mean={stats_b['edge_mean']:.4f}")
    print(f"Figure saved: {fig_path}")


if __name__ == "__main__":
    main()
