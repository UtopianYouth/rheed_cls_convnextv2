#!/usr/bin/env python3
"""
生成第二章实验结果图表：训练曲线与测试集混淆矩阵

用法：
    python -m src.plot_cp2_figures --exp_dir outputs/finetune_timm_20260212_145658 \
                                    --output_dir docs/my_graduate_paper/figure/cp2

生成文件：
    - training_curves.png：训练/验证 loss、acc、f1 曲线
    - confusion_matrix_test.png：测试集混淆矩阵（归一化显示）
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# 默认使用seaborn绘制美化热力图
try:
    import seaborn as sns
    _HAS_SEABORN = True
except ImportError:
    _HAS_SEABORN = False
    raise ImportError("未检测到seaborn，请先安装：conda install seaborn 或 pip install seaborn")


def parse_args():
    parser = argparse.ArgumentParser(description="生成第二章实验图表")
    parser.add_argument(
        "--exp_dir",
        type=str,
        required=True,
        help="实验输出目录（包含log.txt和test_metrics.json）"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="图表保存目录"
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="图片分辨率（默认300）"
    )
    parser.add_argument(
        "--normalize_cm",
        action="store_true",
        default=True,
        help="混淆矩阵是否归一化显示（默认True）"
    )
    return parser.parse_args()


def load_jsonl_log(log_path: Path):
    """加载训练日志（每行一个JSON）"""
    records = []
    if not log_path.exists():
        raise FileNotFoundError(f"训练日志不存在：{log_path}")
    
    with open(log_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"警告：第{line_num}行解析失败，跳过: {e}")
                continue
    
    if not records:
        raise ValueError("训练日志为空或无有效记录")
    
    records.sort(key=lambda x: int(x.get("epoch", 0)))
    return records


def plot_training_curves(records, save_path: Path, dpi=300):
    """
    绘制训练与验证曲线（双子图）
    
    左图：训练/验证 loss
    右图：训练/验证 acc 和验证 f1
    """
    epochs = [r["epoch"] for r in records]
    
    train_loss = [r["train"]["loss"] for r in records]
    val_loss = [r["val"]["loss"] for r in records]
    
    train_acc = [r["train"]["acc1"] for r in records]
    val_acc = [r["val"]["acc1"] for r in records]
    val_f1 = [r["val"].get("f1_macro", np.nan) for r in records]
    
    # 设置中文字体（如果环境支持）
    try:
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
    except Exception:
        pass
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), dpi=dpi)
    
    # ========== 左图：Loss ==========
    ax = axes[0]
    ax.plot(epochs, train_loss, label="Train Loss", linewidth=2, alpha=0.8, color="#1f77b4")
    ax.plot(epochs, val_loss, label="Val Loss", linewidth=2, alpha=0.8, color="#ff7f0e")
    ax.set_xlabel("Epoch", fontsize=12, fontweight="bold")
    ax.set_ylabel("Loss", fontsize=12, fontweight="bold")
    ax.set_title("Training / Validation Loss", fontsize=13, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(loc="best", fontsize=11)
    ax.set_xlim([1, max(epochs)])
    
    # ========== 右图：Accuracy & F1 ==========
    ax = axes[1]
    ax.plot(epochs, train_acc, label="Train Acc (%)", linewidth=2, alpha=0.8, color="#1f77b4")
    ax.plot(epochs, val_acc, label="Val Acc (%)", linewidth=2, alpha=0.8, color="#ff7f0e")
    ax.plot(epochs, val_f1, label="Val Macro-F1 (%)", linewidth=2, alpha=0.8, 
            color="#2ca02c", linestyle="--")
    ax.set_xlabel("Epoch", fontsize=12, fontweight="bold")
    ax.set_ylabel("Metric (%)", fontsize=12, fontweight="bold")
    ax.set_title("Accuracy / Macro-F1", fontsize=13, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(loc="best", fontsize=11)
    ax.set_xlim([1, max(epochs)])
    ax.set_ylim([0, 100])
    
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight", dpi=dpi)
    plt.close(fig)
    print(f"✓ 训练曲线已保存：{save_path}")


def plot_confusion_matrix(cm, class_names, save_path: Path, normalize=True, dpi=300):
    """
    绘制混淆矩阵热力图
    
    参数：
        cm: 2D列表或numpy数组，shape=(n_classes, n_classes)
        class_names: 类别名称列表
        save_path: 保存路径
        normalize: 是否按行归一化（显示百分比）
        dpi: 图片分辨率
    """
    cm = np.array(cm, dtype=np.float64)
    
    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0  # 防止除零
        cm_show = cm / row_sums
        fmt = ".2%"
        title = "Test Set Confusion Matrix (Normalized)"
        vmin, vmax = 0.0, 1.0
    else:
        cm_show = cm
        fmt = "d"
        title = "Test Set Confusion Matrix (Counts)"
        vmin, vmax = None, None
    
    # 设置中文字体
    try:
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
    except Exception:
        pass
    
    fig, ax = plt.subplots(figsize=(6, 5.2), dpi=dpi)
    
    # 使用seaborn绘制（默认美化热力图）
    sns.heatmap(
        cm_show,
        annot=True,
        fmt=fmt if not normalize else ".2f",
        cmap="Blues",
        xticklabels=[f"Pred {c}" for c in class_names],
        yticklabels=[f"True {c}" for c in class_names],
        cbar=True,
        vmin=vmin,
        vmax=vmax,
        linewidths=0.5,
        linecolor='gray',
        ax=ax,
        cbar_kws={'label': 'Proportion' if normalize else 'Count'}
    )
    
    ax.set_title(title, fontsize=13, fontweight="bold", pad=15)
    ax.set_xlabel("Predicted Label", fontsize=12, fontweight="bold")
    ax.set_ylabel("True Label", fontsize=12, fontweight="bold")
    
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight", dpi=dpi)
    plt.close(fig)
    print(f"✓ 混淆矩阵已保存：{save_path}")


def main():
    args = parse_args()
    
    exp_dir = Path(args.exp_dir)
    output_dir = Path(args.output_dir)
    
    # 检查输入目录
    if not exp_dir.exists():
        raise FileNotFoundError(f"实验目录不存在：{exp_dir}")
    
    log_path = exp_dir / "log.txt"
    test_metrics_path = exp_dir / "test_metrics.json"
    
    if not test_metrics_path.exists():
        raise FileNotFoundError(f"测试指标文件不存在：{test_metrics_path}")
    
    # 创建输出目录
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("第二章实验图表生成工具")
    print("=" * 60)
    print(f"实验目录: {exp_dir}")
    print(f"输出目录: {output_dir}")
    print(f"分辨率: {args.dpi} DPI")
    print("-" * 60)
    
    # ========== 1. 生成训练曲线 ==========
    print("\n[1/2] 生成训练曲线...")
    records = load_jsonl_log(log_path)
    print(f"      加载了 {len(records)} 个epoch的训练记录")
    
    training_curves_path = output_dir / "training_curves.png"
    plot_training_curves(records, training_curves_path, dpi=args.dpi)
    
    # ========== 2. 生成混淆矩阵 ==========
    print("\n[2/2] 生成测试集混淆矩阵...")
    with open(test_metrics_path, "r", encoding="utf-8") as f:
        test_metrics = json.load(f)
    
    cm = test_metrics["confusion_matrix"]
    test_acc = test_metrics["acc1"]
    test_f1 = test_metrics["f1_macro"]
    
    print(f"      测试集准确率: {test_acc:.2f}%")
    print(f"      测试集宏平均F1: {test_f1:.2f}%")
    
    # 类别名称（根据你的数据集）
    class_names = ["2D Layer", "3D Island"]
    
    cm_path = output_dir / "confusion_matrix_test.png"
    plot_confusion_matrix(cm, class_names, cm_path, 
                         normalize=args.normalize_cm, dpi=args.dpi)
    
    # ========== 完成 ==========
    print("\n" + "=" * 60)
    print("✓ 所有图表生成完成！")
    print("=" * 60)
    print(f"输出文件：")
    print(f"  1. {training_curves_path}")
    print(f"  2. {cm_path}")
    print("\n可直接在LaTeX中引用这些图片。")


if __name__ == "__main__":
    main()
