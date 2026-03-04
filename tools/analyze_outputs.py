"""Analyze training outputs (pretrain + finetune) and generate plots/tables.

This script is intentionally lightweight and engineering-oriented:
- Parse JSONL logs produced by src/train_timm.py and src/pretrain_fcmae.py
- Generate clear plots (loss/acc curves, confusion matrices)
- Export summary tables (CSV + LaTeX)

Typical usage:
  python -m src.analyze_outputs --outputs_dir outputs

It will create a new folder under outputs/: analysis_report_YYYYmmdd_HHMMSS
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from matplotlib.ticker import AutoMinorLocator, MaxNLocator, MultipleLocator


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Analyze RHEED classification outputs")
    p.add_argument(
        "--outputs_dir",
        type=str,
        default="outputs",
        help="Root outputs directory (default: outputs)",
    )
    p.add_argument(
        "--out_dir",
        type=str,
        default="",
        help="Output directory for report. If empty, create outputs/analysis_report_<timestamp>/",
    )
    p.add_argument(
        "--finetune_glob",
        type=str,
        default="finetune_*",
        help="Glob pattern under outputs_dir for finetune runs (default: finetune_*)",
    )
    p.add_argument(
        "--pretrain_glob",
        type=str,
        default="pretrain_*",
        help="Glob pattern under outputs_dir for pretrain runs (default: pretrain_*)",
    )
    p.add_argument(
        "--max_epochs_for_cm",
        type=int,
        default=6,
        help="How many epochs to visualize confusion matrices per run (default: 6)",
    )
    return p.parse_args()


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def safe_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        return float(x)
    except Exception:
        return None


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def parse_finetune_name(run_name: str) -> Tuple[str, str]:
    name = run_name
    if name.startswith("finetune_timm_"):
        name = name[len("finetune_timm_"):]
    elif name.startswith("finetune_"):
        name = name[len("finetune_"):]

    parts = [p for p in name.split("_") if p]
    # 兼容目录名: finetune_YYYYmmdd_HHMMSS_model_weight
    if len(parts) >= 3 and parts[0].isdigit() and parts[1].isdigit():
        parts = parts[2:]

    if len(parts) >= 2:
        weight = parts[-1]
        model = "_".join(parts[:-1])
    elif len(parts) == 1:
        model = parts[0]
        weight = "unknown"
    else:
        model = run_name
        weight = "unknown"
    return model, weight


COLOR_CYCLE = [
    "#1f77b4",  # blue
    "#d62728",  # red
    "#2ca02c",  # green
    "#9467bd",  # purple
    "#ff7f0e",  # orange
    "#17becf",  # cyan
    "#8c564b",  # brown
]
LINE_STYLES = ["-", "--", "-.", ":"]


def _indices_every_n_epochs(epochs: Sequence[int], n: int = 10) -> List[int]:
    if n <= 0:
        return []
    return [i for i, e in enumerate(epochs) if int(e) % n == 0]


def _line_style_by_idx(idx: int) -> Tuple[str, str]:
    color = COLOR_CYCLE[idx % len(COLOR_CYCLE)]
    linestyle = LINE_STYLES[idx % len(LINE_STYLES)]
    return color, linestyle


def _set_x_epoch_ticks(ax: Any, epochs: Sequence[int]) -> None:
    if not epochs:
        return
    e_max = int(max(epochs))
    if e_max >= 10:
        ax.xaxis.set_major_locator(MultipleLocator(10))
    else:
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))


def _set_finer_y_ticks(ax: Any, y_major_step: Optional[float] = None) -> None:
    if y_major_step is not None:
        ax.yaxis.set_major_locator(MultipleLocator(y_major_step))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))

    # 不显示横向网格线（纵轴刻度线对应的横线）
    ax.grid(False, which="both", axis="both")

    # 轴线样式：只保留左侧 y 轴与底部 x 轴
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["left"].set_linewidth(1.05)
    ax.spines["bottom"].set_linewidth(1.0)
    ax.spines["left"].set_color("#333333")
    ax.spines["bottom"].set_color("#333333")

    # 刻度线内向，论文图更紧凑
    ax.tick_params(axis="x", which="major", direction="in", length=4, width=0.8, colors="#333333")
    ax.tick_params(axis="y", which="major", direction="in", length=4, width=0.8, colors="#333333")
    ax.tick_params(axis="y", which="minor", direction="in", length=2.5, width=0.6, colors="#666666")


def _setup_plot_style() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.labelsize": 11,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "figure.dpi": 140,
            "savefig.bbox": "tight",
        }
    )


def _line_render_kwargs() -> Dict[str, Any]:
    return {
        "linewidth": 1.75,
        "alpha": 0.95,
        "solid_capstyle": "round",
        "solid_joinstyle": "round",
    }


def _smooth_for_display(epochs: Sequence[int], values: Sequence[float], window: int = 5) -> List[float]:
    arr = np.array(values, dtype=np.float64)
    if arr.size == 0:
        return []
    if window <= 1:
        return arr.tolist()
    if window % 2 == 0:
        window += 1

    valid = ~np.isnan(arr)
    arr_filled = np.where(valid, arr, 0.0)
    kernel = np.ones(window, dtype=np.float64)
    num = np.convolve(arr_filled, kernel, mode="same")
    den = np.convolve(valid.astype(np.float64), kernel, mode="same")
    smooth = np.divide(num, den, out=np.copy(arr), where=den > 0)

    # 保留每10 epoch锚点与首尾点，保证关键标记点仍在曲线上
    out = smooth.tolist()
    for i, e in enumerate(epochs):
        if i == 0 or i == len(epochs) - 1 or int(e) % 10 == 0:
            out[i] = float(arr[i])
    return out


def _pick_annotation_offset(
    idx: int,
    y: float,
    used_label_pos: List[Tuple[float, float]],
) -> Tuple[Tuple[int, int], Tuple[float, float]]:
    candidates = [(14, 14), (14, -20), (-64, 14), (-64, -20), (20, 26), (20, -30), (-72, 26), (-72, -30)]
    for offx, offy in candidates:
        px = float(idx) + offx * 0.30
        py = float(y) + offy * 0.16
        if all(abs(px - ux) > 10.0 or abs(py - uy) > 2.2 for ux, uy in used_label_pos):
            return (offx, offy), (px, py)

    offx, offy = candidates[idx % len(candidates)]
    px = float(idx) + offx * 0.30
    py = float(y) + offy * 0.16
    return (offx, offy), (px, py)


def plot_pretrain_fcmae_curve(pre_dir: Path, out_dir: Path) -> Optional[Path]:
    log_path = pre_dir / "log.txt"
    rows = read_jsonl(log_path)
    if not rows:
        return None

    epochs = [int(r.get("epoch")) for r in rows if "epoch" in r]
    losses = [float(r.get("fcmae_loss")) for r in rows if "fcmae_loss" in r]
    if not epochs or not losses:
        return None

    losses_plot = _smooth_for_display(epochs, losses, window=5)

    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    color, linestyle = _line_style_by_idx(0)
    mark_idx = _indices_every_n_epochs(epochs, n=10)
    ax.plot(
        epochs,
        losses_plot,
        color=color,
        linestyle=linestyle,
        marker="o",
        markevery=mark_idx if mark_idx else None,
        markersize=3.4,
        markerfacecolor="white",
        markeredgewidth=0.95,
        markeredgecolor=color,
        label="FCMAE Loss",
        **_line_render_kwargs(),
    )

    best_idx = int(np.argmin(np.array(losses)))
    best_epoch = epochs[best_idx]
    best_loss_raw = losses[best_idx]
    best_loss_plot = losses_plot[best_idx]
    y_off = -20 if best_loss_plot > float(np.nanmedian(losses_plot)) else 14
    ax.scatter([best_epoch], [best_loss_plot], s=40, color="#111111", zorder=5)
    ax.annotate(
        f"E{best_epoch}, {best_loss_raw:.4f}",
        (best_epoch, best_loss_plot),
        textcoords="offset points",
        xytext=(14, y_off),
        fontsize=8.8,
        bbox=dict(boxstyle="round,pad=0.16", facecolor="white", edgecolor="none", alpha=0.72),
        arrowprops=dict(arrowstyle="-", color="#666666", lw=0.8, shrinkA=0, shrinkB=4),
    )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    _set_x_epoch_ticks(ax, epochs)
    _set_finer_y_ticks(ax)
    ax.margins(x=0.04, y=0.08)
    ax.legend(frameon=False)

    out_path = out_dir / f"fcmae_loss_curve_{pre_dir.name}.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=240)
    plt.close(fig)
    return out_path


def plot_finetune_curves(ft_dir: Path, out_dir: Path) -> Optional[Path]:
    rows = read_jsonl(ft_dir / "log.txt")
    if not rows:
        return None

    epochs = [int(r["epoch"]) for r in rows]
    train_loss = [float(r["train"]["loss"]) for r in rows]
    train_acc = [float(r["train"]["acc1"]) for r in rows]
    val_loss = [float(r["val"]["loss"]) for r in rows]
    val_acc = [float(r["val"]["acc1"]) for r in rows]
    val_f1 = [safe_float(r["val"].get("f1_macro")) for r in rows]

    train_loss_plot = _smooth_for_display(epochs, train_loss, window=5)
    val_loss_plot = _smooth_for_display(epochs, val_loss, window=5)
    train_acc_plot = _smooth_for_display(epochs, train_acc, window=5)
    val_acc_plot = _smooth_for_display(epochs, val_acc, window=5)

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.8))
    mark_idx = _indices_every_n_epochs(epochs, n=10)

    c0, s0 = _line_style_by_idx(0)
    c1, s1 = _line_style_by_idx(1)
    axes[0].plot(
        epochs,
        train_loss_plot,
        label="Train Loss",
        color=c0,
        linestyle=s0,
        marker="o",
        markevery=mark_idx if mark_idx else None,
        markersize=3.2,
        markerfacecolor="white",
        markeredgecolor=c0,
        markeredgewidth=0.9,
        **_line_render_kwargs(),
    )
    axes[0].plot(
        epochs,
        val_loss_plot,
        label="Val Loss",
        color=c1,
        linestyle=s1,
        marker="o",
        markevery=mark_idx if mark_idx else None,
        markersize=3.2,
        markerfacecolor="white",
        markeredgecolor=c1,
        markeredgewidth=0.9,
        **_line_render_kwargs(),
    )

    min_val_loss_idx = int(np.argmin(np.array(val_loss)))
    min_val_loss_raw = val_loss[min_val_loss_idx]
    min_val_loss_plot = val_loss_plot[min_val_loss_idx]
    axes[0].scatter([epochs[min_val_loss_idx]], [min_val_loss_plot], s=36, color="#111111", zorder=5)
    axes[0].annotate(
        f"E{epochs[min_val_loss_idx]}, {min_val_loss_raw:.4f}",
        (epochs[min_val_loss_idx], min_val_loss_plot),
        textcoords="offset points",
        xytext=(14, -20),
        fontsize=8.8,
        bbox=dict(boxstyle="round,pad=0.14", facecolor="white", edgecolor="none", alpha=0.72),
        arrowprops=dict(arrowstyle="-", color="#666666", lw=0.8, shrinkA=0, shrinkB=4),
    )
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    _set_x_epoch_ticks(axes[0], epochs)
    _set_finer_y_ticks(axes[0])
    axes[0].margins(x=0.04, y=0.08)
    axes[0].legend(frameon=False)

    c2, s2 = _line_style_by_idx(2)
    c3, s3 = _line_style_by_idx(3)
    axes[1].plot(
        epochs,
        train_acc_plot,
        label="Train Acc@1",
        color=c2,
        linestyle=s2,
        marker="o",
        markevery=mark_idx if mark_idx else None,
        markersize=3.2,
        markerfacecolor="white",
        markeredgecolor=c2,
        markeredgewidth=0.9,
        **_line_render_kwargs(),
    )
    axes[1].plot(
        epochs,
        val_acc_plot,
        label="Val Acc@1",
        color=c3,
        linestyle=s3,
        marker="o",
        markevery=mark_idx if mark_idx else None,
        markersize=3.2,
        markerfacecolor="white",
        markeredgecolor=c3,
        markeredgewidth=0.9,
        **_line_render_kwargs(),
    )
    if any(v is not None for v in val_f1):
        vf = [v if v is not None else math.nan for v in val_f1]
        vf_plot = _smooth_for_display(epochs, vf, window=5)
        c4, s4 = _line_style_by_idx(4)
        axes[1].plot(
            epochs,
            vf_plot,
            label="Val Macro-F1",
            color=c4,
            linestyle=s4,
            marker="o",
            markevery=mark_idx if mark_idx else None,
            markersize=3.2,
            markerfacecolor="white",
            markeredgecolor=c4,
            markeredgewidth=0.9,
            **_line_render_kwargs(),
        )
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Metric (%)")
    axes[1].set_ylim(0, 100)
    _set_x_epoch_ticks(axes[1], epochs)
    _set_finer_y_ticks(axes[1], y_major_step=5)
    axes[1].margins(x=0.04, y=0.06)

    # 标注按日志best_metric/best_epoch选取的点
    best_row = max(rows, key=lambda r: r.get("best_metric", -1))
    best_epoch = best_row.get("best_epoch", best_row.get("epoch"))
    if not isinstance(best_epoch, int) or best_epoch not in epochs:
        best_epoch = epochs[int(np.argmax(np.array(val_acc)))]
    best_idx = epochs.index(int(best_epoch))
    best_acc_raw = val_acc[best_idx]
    best_acc_plot = val_acc_plot[best_idx]
    y_off = -20 if best_acc_plot > 90 else 14
    axes[1].scatter([best_epoch], [best_acc_plot], s=44, color="#111111", zorder=6)
    axes[1].annotate(
        f"E{best_epoch}, {best_acc_raw:.2f}",
        (best_epoch, best_acc_plot),
        textcoords="offset points",
        xytext=(14, y_off),
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.16", facecolor="white", edgecolor="none", alpha=0.72),
        arrowprops=dict(arrowstyle="-", color="#666666", lw=0.8, shrinkA=0, shrinkB=4),
    )
    axes[1].legend(frameon=False)

    model_name, weight_name = parse_finetune_name(ft_dir.name)
    out_path = out_dir / f"curves_{model_name}_{weight_name}.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=240)
    plt.close(fig)
    return out_path


def plot_confusion_matrix(cm: Sequence[Sequence[int]], out_path: Path, title: Optional[str] = None) -> None:
    cm_arr = np.array(cm, dtype=np.int64)
    if cm_arr.ndim != 2 or cm_arr.shape[0] != cm_arr.shape[1]:
        return

    row_sum = cm_arr.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm_arr, row_sum, out=np.zeros_like(cm_arr, dtype=np.float64), where=row_sum > 0)
    cm_pct = cm_norm * 100.0

    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    im = ax.imshow(cm_pct, cmap="YlGnBu", vmin=0.0, vmax=100.0)

    # 按用户要求：图内不再显示标题（即便传入 title）
    _ = title

    n_cls = cm_arr.shape[0]
    labels = ["2D", "3D"] if n_cls == 2 else [str(i) for i in range(n_cls)]
    ax.set_xlabel("Predicted Class")
    ax.set_ylabel("Ground Truth Class")
    ax.set_xticks(np.arange(n_cls))
    ax.set_yticks(np.arange(n_cls))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)

    # 不绘制内部网格线，仅保留外边框
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.1)
        spine.set_color("#333333")

    # 标注：计数 + 行归一化百分比
    for (i, j), v in np.ndenumerate(cm_arr):
        pct = cm_pct[i, j]
        text_color = "white" if pct >= 55 else "#1a1a1a"
        ax.text(
            j,
            i,
            f"{int(v)}\n({pct:.1f}%)",
            ha="center",
            va="center",
            fontsize=9,
            color=text_color,
        )

    # 对角线高亮（正确分类项）
    for k in range(n_cls):
        rect = Rectangle((k - 0.5, k - 0.5), 1, 1, fill=False, edgecolor="#ff7f0e", linewidth=2.0)
        ax.add_patch(rect)

    # 右侧颜色条
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Row-wise Ratio (%)", rotation=90)

    total = int(cm_arr.sum())
    diag = int(np.trace(cm_arr))
    acc = (diag / total * 100.0) if total > 0 else 0.0
    bal_acc = float(np.mean(np.diag(cm_norm)) * 100.0)
    ax.text(
        0.5,
        -0.14,
        f"Samples={total}, Acc={acc:.2f}%, Balanced-Acc={bal_acc:.2f}%",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=9,
        color="#333333",
    )

    fig.tight_layout()
    fig.savefig(out_path, dpi=240)
    plt.close(fig)


def pick_epochs(rows: List[Dict[str, Any]], max_k: int) -> List[int]:
    epochs = [int(r["epoch"]) for r in rows if "epoch" in r]
    if not epochs:
        return []

    best_row = max(rows, key=lambda r: r.get("best_metric", -1))
    best_epoch = best_row.get("best_epoch", None)
    last_epoch = epochs[-1]

    cand: List[int] = []
    if isinstance(best_epoch, int):
        cand.append(best_epoch)
    cand.append(last_epoch)

    # unique keep order
    seen = set()
    out: List[int] = []
    for e in cand:
        if e not in seen:
            out.append(e)
            seen.add(e)
    return out[:max_k]


def finetune_summary(ft_dir: Path) -> Optional[Dict[str, Any]]:
    rows = read_jsonl(ft_dir / "log.txt")
    if not rows:
        return None

    best_row = max(rows, key=lambda r: r.get("best_metric", -1))
    best_epoch = int(best_row.get("best_epoch", best_row.get("epoch")))

    row_at_best = next((r for r in rows if int(r.get("epoch", -1)) == best_epoch), best_row)

    test_path = ft_dir / "test_metrics.json"
    test = json.loads(test_path.read_text(encoding="utf-8")) if test_path.exists() else {}

    return {
        "run": ft_dir.name,
        "best_epoch": best_epoch,
        "best_val_acc": safe_float(row_at_best.get("val", {}).get("acc1")),
        "best_val_f1": safe_float(row_at_best.get("val", {}).get("f1_macro")),
        "best_val_loss": safe_float(row_at_best.get("val", {}).get("loss")),
        "last_val_acc": safe_float(rows[-1].get("val", {}).get("acc1")),
        "last_val_f1": safe_float(rows[-1].get("val", {}).get("f1_macro")),
        "test_acc": safe_float(test.get("acc1")),
        "test_f1": safe_float(test.get("f1_macro")),
        "test_loss": safe_float(test.get("loss")),
        "test_cm": test.get("confusion_matrix"),
    }


def write_csv(rows: List[Dict[str, Any]], out_path: Path) -> None:
    fields = [
        "run",
        "best_epoch",
        "best_val_acc",
        "best_val_f1",
        "best_val_loss",
        "last_val_acc",
        "last_val_f1",
        "test_acc",
        "test_f1",
        "test_loss",
    ]

    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fields})


def write_latex_table(rows: List[Dict[str, Any]], out_path: Path) -> None:
    with out_path.open("w", encoding="utf-8") as f:
        f.write("% Auto-generated table. Copy into LaTeX as needed.\n")
        f.write("\\begin{table}[h]\n")
        f.write("\\centering\n")
        f.write("\\caption{RHEED分类微调实验结果汇总（验证集最优与测试集表现）}\\n")
        f.write("\\label{tab:finetune_summary}\\n")
        f.write("\\begin{tabular}{lcccccc}\\n")
        f.write("\\hline\\n")
        f.write("Run & BestEp & ValAcc(\\%) & ValF1(\\%) & ValLoss & TestAcc(\\%) & TestF1(\\%) \\\\\\ \\hline\\n")

        for r in rows:
            val_acc = r.get("best_val_acc")
            val_f1 = r.get("best_val_f1")
            val_loss = r.get("best_val_loss")
            test_acc = r.get("test_acc")
            test_f1 = r.get("test_f1")

            def fmt(x: Any, nd: int = 2) -> str:
                if x is None:
                    return "-"
                return f"{float(x):.{nd}f}"

            f.write(
                f"{r['run']} & {r['best_epoch']} & {fmt(val_acc)} & {fmt(val_f1)} & {fmt(val_loss, 4)} & {fmt(test_acc)} & {fmt(test_f1)} \\\\\\ \n"
            )

        f.write("\\hline\\n")
        f.write("\\end{tabular}\\n")
        f.write("\\end{table}\\n")


def plot_val_acc_comparison(
    finetune_dirs: List[Path],
    out_dir: Path,
    out_name: str,
) -> Optional[Path]:
    fig, ax = plt.subplots(figsize=(9.2, 5.4))

    any_ok = False
    all_epochs: List[int] = []
    used_label_points: List[Tuple[float, float]] = []
    for i, d in enumerate(finetune_dirs):
        rows = read_jsonl(d / "log.txt")
        if not rows:
            continue

        epochs = [int(r["epoch"]) for r in rows]
        val_acc = [float(r["val"]["acc1"]) for r in rows]
        val_acc_plot = _smooth_for_display(epochs, val_acc, window=5)
        all_epochs.extend(epochs)

        model_name, weight_name = parse_finetune_name(d.name)
        label = f"{model_name}_{weight_name}"
        color, linestyle = _line_style_by_idx(i)
        mark_idx = _indices_every_n_epochs(epochs, n=10)

        ax.plot(
            epochs,
            val_acc_plot,
            color=color,
            linestyle=linestyle,
            marker="o",
            markevery=mark_idx if mark_idx else None,
            markersize=3.2,
            markerfacecolor="white",
            markeredgecolor=color,
            markeredgewidth=0.9,
            label=label,
            **_line_render_kwargs(),
        )

        best_idx = int(np.argmax(np.array(val_acc)))
        best_epoch = epochs[best_idx]
        best_acc_raw = val_acc[best_idx]
        best_acc_plot = val_acc_plot[best_idx]
        ax.scatter([best_epoch], [best_acc_plot], s=36, color=color, edgecolors="#111111", zorder=6)
        offset, label_pos = _pick_annotation_offset(best_epoch, best_acc_plot, used_label_points)
        ax.annotate(
            f"{best_acc_raw:.2f}@{best_epoch}",
            (best_epoch, best_acc_plot),
            textcoords="offset points",
            xytext=offset,
            fontsize=8.4,
            color=color,
            bbox=dict(boxstyle="round,pad=0.16", facecolor="white", edgecolor="none", alpha=0.74),
            arrowprops=dict(arrowstyle="-", color=color, lw=0.75, shrinkA=0, shrinkB=4, alpha=0.8),
        )
        used_label_points.append(label_pos)
        any_ok = True

    if not any_ok:
        plt.close(fig)
        return None

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Val Acc@1 (%)")
    ax.set_ylim(50, 100)
    _set_x_epoch_ticks(ax, all_epochs)
    _set_finer_y_ticks(ax, y_major_step=5)
    ax.margins(x=0.04, y=0.06)
    ax.legend(loc="lower left", frameon=False)
    fig.tight_layout()

    out_path = out_dir / out_name
    fig.savefig(out_path, dpi=240)
    plt.close(fig)
    return out_path


def main() -> None:
    args = parse_args()
    _setup_plot_style()

    outputs_dir = Path(args.outputs_dir)
    if not outputs_dir.exists():
        raise FileNotFoundError(f"outputs_dir not found: {outputs_dir}")

    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = outputs_dir / f"analysis_report_{ts}"
    ensure_dir(out_dir)

    finetune_dirs = sorted([p for p in outputs_dir.glob(args.finetune_glob) if p.is_dir()])
    pretrain_dirs = sorted([p for p in outputs_dir.glob(args.pretrain_glob) if p.is_dir()])

    if not finetune_dirs:
        print(
            f"[WARN] No finetune runs found with glob '{args.finetune_glob}' under: {outputs_dir.resolve()}"
        )
    else:
        print(f"[INFO] Found {len(finetune_dirs)} finetune runs")

    if not pretrain_dirs:
        print(
            f"[WARN] No pretrain runs found with glob '{args.pretrain_glob}' under: {outputs_dir.resolve()}"
        )
    else:
        print(f"[INFO] Found {len(pretrain_dirs)} pretrain runs")

    # 1) pretrain curves
    for d in pretrain_dirs:
        plot_pretrain_fcmae_curve(d, out_dir)

    # 2) finetune curves + confusion matrices
    summaries: List[Dict[str, Any]] = []
    for d in finetune_dirs:
        rows = read_jsonl(d / "log.txt")
        if not rows:
            continue

        plot_finetune_curves(d, out_dir)

        model_name, weight_name = parse_finetune_name(d.name)

        # 仅导出验证集 best epoch 的混淆矩阵
        best_row = max(rows, key=lambda r: r.get("best_metric", -1))
        best_epoch = best_row.get("best_epoch", best_row.get("epoch", None))
        best_val_row = None
        if isinstance(best_epoch, int):
            best_val_row = next((x for x in rows if int(x.get("epoch", -1)) == best_epoch), None)
        if best_val_row is None:
            best_val_row = best_row

        best_val_cm = best_val_row.get("val", {}).get("confusion_matrix")
        if best_val_cm is not None:
            plot_confusion_matrix(
                best_val_cm,
                out_path=out_dir / f"cm_{model_name}_{weight_name}_best.png",
            )

        s = finetune_summary(d)
        if s is not None:
            summaries.append(s)
            test_cm = s.get("test_cm")
            if test_cm is not None:
                plot_confusion_matrix(
                    test_cm,
                    out_path=out_dir / f"cm_{model_name}_{weight_name}_test.png",
                )

    # 3) summary tables
    write_csv(summaries, out_dir / "metrics_summary.csv")
    write_latex_table(summaries, out_dir / "metrics_summary.tex")

    # 4) overview comparison plots
    model_weight_dirs: List[Path] = []
    convnext_weight_dirs: List[Path] = []
    for d in finetune_dirs:
        model_name, weight_name = parse_finetune_name(d.name)
        if weight_name == "imagenet":
            model_weight_dirs.append(d)
        if model_name == "convnextv2_atto":
            convnext_weight_dirs.append(d)

    plot_val_acc_comparison(
        model_weight_dirs,
        out_dir,
        "val_acc_comparison_models_imagenet.png",
    )
    plot_val_acc_comparison(
        convnext_weight_dirs,
        out_dir,
        "val_acc_comparison_convnextv2_atto_weights.png",
    )

    print(f"[OK] Report generated at: {out_dir.resolve()}")
    print("[OK] Key files:")
    for name in [
        "metrics_summary.csv",
        "metrics_summary.tex",
        "val_acc_comparison_models_imagenet.png",
        "val_acc_comparison_convnextv2_atto_weights.png",
    ]:
        p = out_dir / name
        if p.exists():
            print(" -", p.name)


if __name__ == "__main__":
    main()
