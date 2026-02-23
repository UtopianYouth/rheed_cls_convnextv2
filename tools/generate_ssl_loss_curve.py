#!/usr/bin/env python3
"""
生成自监督预训练 SSL loss 曲线（单图）。

用法：
    python -m src.generate_ssl_loss_curve \
        --log_path outputs/pretrain_ssl_20260221_151605/log.txt \
        --output_path docs/my_graduate_paper/figure/cp2/ssl_loss_curve_20260221_151605.png
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(description="生成SSL loss曲线")
    parser.add_argument(
        "--log_path",
        type=str,
        required=True,
        help="自监督预训练log.txt路径（jsonl格式）",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="输出图片路径（.png）",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="图片分辨率（默认300）",
    )
    return parser.parse_args()


def load_ssl_log(log_path: Path):
    records = []
    if not log_path.exists():
        raise FileNotFoundError(f"日志不存在：{log_path}")

    with log_path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                print(f"警告：第{line_num}行解析失败，跳过: {exc}")
                continue

    if not records:
        raise ValueError("日志为空或无有效记录")

    records.sort(key=lambda x: int(x.get("epoch", 0)))
    return records


def plot_ssl_loss(records, save_path: Path, dpi: int):
    epochs = [r["epoch"] for r in records]
    losses = [r["ssl_loss"] for r in records]

    try:
        plt.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False
    except Exception:
        pass

    fig, ax = plt.subplots(figsize=(6.5, 4.5), dpi=dpi)
    ax.plot(epochs, losses, linewidth=2, color="#1f77b4", label="SSL Loss")
    ax.set_xlabel("Epoch", fontsize=12, fontweight="bold")
    ax.set_ylabel("Loss", fontsize=12, fontweight="bold")
    ax.set_title("SSL Pretraining Loss Curve", fontsize=13, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(loc="best", fontsize=11)
    ax.set_xlim([1, max(epochs)])

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight", dpi=dpi)
    plt.close(fig)
    print(f"✓ SSL loss曲线已保存：{save_path}")


def main():
    args = parse_args()
    log_path = Path(args.log_path)
    output_path = Path(args.output_path)

    records = load_ssl_log(log_path)
    plot_ssl_loss(records, output_path, args.dpi)


if __name__ == "__main__":
    main()
