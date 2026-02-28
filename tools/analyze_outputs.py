"""Analyze training outputs (pretrain + finetune) and generate plots/tables.

This script is intentionally lightweight and engineering-oriented:
- Parse JSONL logs produced by src/train_timm.py and src/pretrain_ssl.py
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
        default="finetune_timm_*",
        help="Glob pattern under outputs_dir for finetune runs",
    )
    p.add_argument(
        "--pretrain_glob",
        type=str,
        default="pretrain_ssl_*",
        help="Glob pattern under outputs_dir for pretrain runs",
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
    parts = [p for p in name.split("_") if p]
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


def plot_pretrain_ssl_curve(pre_dir: Path, out_dir: Path) -> Optional[Path]:
    log_path = pre_dir / "log.txt"
    rows = read_jsonl(log_path)
    if not rows:
        return None

    epochs = [int(r.get("epoch")) for r in rows if "epoch" in r]
    losses = [float(r.get("ssl_loss")) for r in rows if "ssl_loss" in r]
    if not epochs or not losses:
        return None

    plt.figure(figsize=(7.6, 4.6))
    plt.plot(epochs, losses, linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("SSL Loss (NT-Xent)")
    plt.grid(True, alpha=0.3)

    idx = int(np.argmin(np.array(losses)))
    plt.scatter([epochs[idx]], [losses[idx]], s=40)
    plt.annotate(
        f"min={losses[idx]:.4f}@{epochs[idx]}",
        (epochs[idx], losses[idx]),
        textcoords="offset points",
        xytext=(10, -12),
    )

    out_path = out_dir / f"ssl_loss_curve_{pre_dir.name}.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()
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

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.4))

    axes[0].plot(epochs, train_loss, label="Train Loss", linewidth=2)
    axes[0].plot(epochs, val_loss, label="Val Loss", linewidth=2)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(epochs, train_acc, label="Train Acc@1", linewidth=2)
    axes[1].plot(epochs, val_acc, label="Val Acc@1", linewidth=2)
    if any(v is not None for v in val_f1):
        vf = [v if v is not None else math.nan for v in val_f1]
        axes[1].plot(epochs, vf, label="Val Macro-F1", linewidth=2)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Metric (%)")
    axes[1].set_ylim(0, 105)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    # 标注 best epoch（按 log 中 best_metric/best_epoch）
    best_row = max(rows, key=lambda r: r.get("best_metric", -1))
    best_epoch = best_row.get("best_epoch", best_row.get("epoch"))
    if isinstance(best_epoch, int) and best_epoch in epochs:
        idx = epochs.index(best_epoch)
        axes[1].scatter([epochs[idx]], [val_acc[idx]], s=40)
        axes[1].annotate(
            f"best@{best_epoch}\nval_acc={val_acc[idx]:.2f}",
            (epochs[idx], val_acc[idx]),
            textcoords="offset points",
            xytext=(10, -15),
        )

    model_name, weight_name = parse_finetune_name(ft_dir.name)
    out_path = out_dir / f"curves_{model_name}_{weight_name}.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close(fig)
    return out_path


def plot_confusion_matrix(cm: Sequence[Sequence[int]], out_path: Path, title: Optional[str] = None) -> None:
    cm_arr = np.array(cm, dtype=np.int64)

    fig, ax = plt.subplots(figsize=(4.9, 4.3))
    im = ax.imshow(cm_arr, cmap="Blues")

    if title:
        ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Ground Truth")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["2D", "3D"])
    ax.set_yticklabels(["2D", "3D"])

    for (i, j), v in np.ndenumerate(cm_arr):
        ax.text(j, i, str(int(v)), ha="center", va="center", fontsize=10)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
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
    plt.figure(figsize=(8.8, 5.2))

    any_ok = False
    for d in finetune_dirs:
        rows = read_jsonl(d / "log.txt")
        if not rows:
            continue
        epochs = [int(r["epoch"]) for r in rows]
        val_acc = [float(r["val"]["acc1"]) for r in rows]
        model_name, weight_name = parse_finetune_name(d.name)
        label = f"{model_name}_{weight_name}"
        plt.plot(epochs, val_acc, linewidth=2, label=label)
        any_ok = True

    if not any_ok:
        plt.close()
        return None

    plt.xlabel("Epoch")
    plt.ylabel("Val Acc@1 (%)")
    plt.ylim(0, 105)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    out_path = out_dir / out_name
    plt.savefig(out_path, dpi=220)
    plt.close()
    return out_path


def main() -> None:
    args = parse_args()

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

    # 1) pretrain curves
    for d in pretrain_dirs:
        plot_pretrain_ssl_curve(d, out_dir)

    # 2) finetune curves + confusion matrices
    summaries: List[Dict[str, Any]] = []
    for d in finetune_dirs:
        rows = read_jsonl(d / "log.txt")
        if not rows:
            continue

        plot_finetune_curves(d, out_dir)

        model_name, weight_name = parse_finetune_name(d.name)
        epochs_for_cm = pick_epochs(rows, max_k=args.max_epochs_for_cm)
        best_row = max(rows, key=lambda r: r.get("best_metric", -1))
        best_epoch = best_row.get("best_epoch", None)
        last_epoch = int(rows[-1].get("epoch", rows[-1].get("epoch", -1)))

        for e in epochs_for_cm:
            r = next((x for x in rows if int(x.get("epoch", -1)) == e), None)
            if r is None:
                continue
            cm = r.get("val", {}).get("confusion_matrix")
            if cm is None:
                continue
            tag = "best" if isinstance(best_epoch, int) and e == best_epoch else "last"
            if e != last_epoch and tag == "last":
                continue
            plot_confusion_matrix(
                cm,
                out_path=out_dir / f"cm_{model_name}_{weight_name}_{tag}.png",
            )

        s = finetune_summary(d)
        if s is not None:
            summaries.append(s)

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
