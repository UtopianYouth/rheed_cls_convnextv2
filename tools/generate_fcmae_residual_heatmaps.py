"""Generate residual visualizations for FCMAE reconstruction pairs (src vs dst).

Improvements:
1) Export four independent figures per sample (SRC / DST / residual-gray / residual-heatmap).
2) Use publication-style heatmap aesthetics (Matplotlib + Seaborn).
3) Keep output canvas size exactly identical to input image size.
4) Export text logs for residual/heatmap results and size checks.

Usage:
  python -m tools.generate_fcmae_residual_heatmaps \
      --input_dir /path/to/fcmae_images \
      --output_dir /path/to/fcmae_images/residual_maps
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from PIL import Image
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Generate FCMAE residual heatmaps")
    p.add_argument("--input_dir", type=str, default="fcmae_images", help="Directory containing src/dst images")
    p.add_argument("--output_dir", type=str, default="", help="Output directory. Default: <input_dir>/residual_maps")
    p.add_argument("--heatmap_cmap", type=str, default="lightgreen", help="Seaborn/Matplotlib colormap for residual heatmap")
    p.add_argument("--vmax_percentile", type=float, default=99.0, help="Percentile for residual heatmap vmax")
    p.add_argument("--dpi", type=int, default=100, help="Rendering DPI for exact-size export")
    p.add_argument("--log_file", type=str, default="residual_report.txt", help="Text log file name under output_dir")
    return p.parse_args()


def _ensure_gray_2d(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 2:
        return arr.astype(np.float32)
    if arr.ndim == 3:
        return arr[..., :3].astype(np.float32).mean(axis=2)
    raise ValueError(f"Unsupported image shape: {arr.shape}")


def load_gray_image(path: Path) -> np.ndarray:
    try:
        arr = np.array(Image.open(path))
        return _ensure_gray_2d(arr)
    except Exception:
        pass

    try:
        import tifffile as tiff  # type: ignore

        arr = tiff.imread(str(path))
        return _ensure_gray_2d(np.array(arr))
    except Exception as e:
        raise RuntimeError(f"Failed to read image: {path} ({e})")


def resize_to_shape(arr: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    img = Image.fromarray(arr.astype(np.float32), mode="F")
    img = img.resize((target_w, target_h), Image.BILINEAR)
    return np.array(img, dtype=np.float32)


def collect_pairs(input_dir: Path) -> Dict[str, Dict[str, Path]]:
    pattern = re.compile(
        r"^(?P<prefix>.+?)_(?P<tag>src|dst)_(?P<idx>.+?)\.(?P<ext>tiff|tif|png|jpg|jpeg)$",
        re.IGNORECASE,
    )
    pairs: Dict[str, Dict[str, Path]] = {}

    for p in sorted(input_dir.iterdir()):
        if not p.is_file():
            continue
        m = pattern.match(p.name)
        if not m:
            continue
        key = f"{m.group('prefix')}_{m.group('idx')}"
        pairs.setdefault(key, {})[m.group("tag").lower()] = p

    return {k: v for k, v in pairs.items() if "src" in v and "dst" in v}


def calc_metrics(src: np.ndarray, dst: np.ndarray) -> Dict[str, float]:
    diff = np.abs(src - dst)
    mse = float(np.mean((src - dst) ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(diff))
    max_abs = float(np.max(diff))
    p95 = float(np.percentile(diff, 95))
    p99 = float(np.percentile(diff, 99))

    signal_range = float(np.max(src) - np.min(src))
    if signal_range <= 1e-12:
        nmae = 0.0
    else:
        nmae = mae / signal_range

    return {
        "mae": mae,
        "rmse": rmse,
        "max_abs": max_abs,
        "p95_abs": p95,
        "p99_abs": p99,
        "nmae": nmae,
    }


def _robust_display_range(src: np.ndarray, dst: np.ndarray) -> tuple[float, float]:
    stack = np.concatenate([src.ravel(), dst.ravel()])
    lo = float(np.percentile(stack, 1.0))
    hi = float(np.percentile(stack, 99.5))
    if hi <= lo:
        lo, hi = float(np.min(stack)), float(np.max(stack))
    if hi <= lo:
        hi = lo + 1e-6
    return lo, hi


def _save_exact_heatmap(
    diff: np.ndarray,
    out_path: Path,
    cmap_name: str,
    vmax_percentile: float,
    dpi: int,
) -> None:
    h, w = diff.shape
    vmax = float(np.percentile(diff, vmax_percentile))
    vmax = max(vmax, 1e-6)

    sns.set_theme(style="white", context="paper")
    if cmap_name.lower() == "lightgreen":
        cmap = sns.light_palette("#2ca25f", as_cmap=True)
    else:
        cmap = sns.color_palette(cmap_name, as_cmap=True)

    fig = plt.figure(figsize=(w / dpi, h / dpi), dpi=dpi, frameon=False)
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])

    cax = inset_axes(ax, width="3.5%", height="42%", loc="lower right", borderpad=1.0)
    sns.heatmap(
        diff,
        ax=ax,
        cmap=cmap,
        vmin=0.0,
        vmax=vmax,
        xticklabels=False,
        yticklabels=False,
        cbar=True,
        cbar_ax=cax,
        square=False,
        linewidths=0,
    )

    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xlabel("")
    ax.set_ylabel("")

    ticks = [0.0, vmax * 0.25, vmax * 0.50, vmax * 0.75, vmax]
    cax.set_yticks(ticks)
    cax.set_yticklabels([f"{t:.1f}" for t in ticks], fontsize=7)
    cax.set_title("|Δ|", fontsize=7, pad=2)

    txt = (
        f"mean={np.mean(diff):.2f}\n"
        f"p95={np.percentile(diff, 95):.2f}\n"
        f"p99={np.percentile(diff, 99):.2f}"
    )
    ax.text(
        0.012,
        0.988,
        txt,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7,
        color="white",
        bbox=dict(boxstyle="round,pad=0.2", facecolor=(0, 0, 0, 0.35), edgecolor="none"),
    )

    fig.savefig(out_path, dpi=dpi, bbox_inches=None, pad_inches=0)
    plt.close(fig)


def _image_wh(path: Path) -> Tuple[int, int]:
    with Image.open(path) as img:
        w, h = img.size
    return w, h


def save_visuals(
    key: str,
    diff: np.ndarray,
    out_dir: Path,
    cmap: str,
    vmax_percentile: float,
    dpi: int,
) -> Path:
    out_res_heat = out_dir / f"residual_heatmap_{key}.png"
    _save_exact_heatmap(diff, out_res_heat, cmap, vmax_percentile, dpi)
    return out_res_heat


def write_csv(rows: List[Dict[str, float]], out_path: Path) -> None:
    if not rows:
        return
    fields = ["sample", "src", "dst", "mae", "rmse", "max_abs", "p95_abs", "p99_abs", "nmae"]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _format_line(sample: str, metrics: Dict[str, float], src_wh: Tuple[int, int], heat_wh: Tuple[int, int]) -> str:
    status = "PASS" if heat_wh == src_wh else "FAIL"
    return (
        f"[{sample}] "
        f"MAE={metrics['mae']:.4f}, RMSE={metrics['rmse']:.4f}, "
        f"NMAE={metrics['nmae']:.6f}, P95={metrics['p95_abs']:.4f}, P99={metrics['p99_abs']:.4f}, "
        f"MaxAbs={metrics['max_abs']:.4f} | "
        f"Input={src_wh[0]}x{src_wh[1]} | residual_heatmap={heat_wh[0]}x{heat_wh[1]}({status})"
    )


def write_text_report(log_path: Path, lines: List[str], summary: Dict[str, float]) -> None:
    with log_path.open("w", encoding="utf-8") as f:
        f.write("FCMAE Residual Analysis Report\n")
        f.write("=" * 72 + "\n")
        for line in lines:
            f.write(line + "\n")
        f.write("-" * 72 + "\n")
        f.write(
            "Summary: "
            f"samples={int(summary['n'])}, "
            f"mae_mean={summary['mae_mean']:.4f}, rmse_mean={summary['rmse_mean']:.4f}, "
            f"nmae_mean={summary['nmae_mean']:.6f}, p95_mean={summary['p95_mean']:.4f}, "
            f"p99_mean={summary['p99_mean']:.4f}, size_pass={int(summary['size_pass'])}/{int(summary['n'])}\n"
        )


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"input_dir not found: {input_dir}")

    out_dir = Path(args.output_dir) if args.output_dir else (input_dir / "residual_maps")
    out_dir.mkdir(parents=True, exist_ok=True)

    pairs = collect_pairs(input_dir)
    if not pairs:
        raise RuntimeError(f"No src/dst pairs found in: {input_dir}")

    summary_rows: List[Dict[str, float]] = []
    report_lines: List[str] = []
    size_pass_count = 0

    for key, pp in sorted(pairs.items()):
        src = load_gray_image(pp["src"])
        dst = load_gray_image(pp["dst"])
        if src.shape != dst.shape:
            dst = resize_to_shape(dst, src.shape[0], src.shape[1])

        diff = np.abs(src - dst)
        metrics = calc_metrics(src, dst)
        out_heatmap = save_visuals(key, diff, out_dir, args.heatmap_cmap, args.vmax_percentile, args.dpi)

        src_wh = (src.shape[1], src.shape[0])
        heat_wh = _image_wh(out_heatmap)
        sample_size_pass = heat_wh == src_wh
        if sample_size_pass:
            size_pass_count += 1

        line = _format_line(key, metrics, src_wh, heat_wh)
        report_lines.append(line)
        print(line)

        summary_rows.append({"sample": key, "src": pp["src"].name, "dst": pp["dst"].name, **metrics})

    write_csv(summary_rows, out_dir / "residual_metrics_summary.csv")

    summary = {
        "n": float(len(summary_rows)),
        "mae_mean": float(np.mean([r["mae"] for r in summary_rows])) if summary_rows else 0.0,
        "rmse_mean": float(np.mean([r["rmse"] for r in summary_rows])) if summary_rows else 0.0,
        "nmae_mean": float(np.mean([r["nmae"] for r in summary_rows])) if summary_rows else 0.0,
        "p95_mean": float(np.mean([r["p95_abs"] for r in summary_rows])) if summary_rows else 0.0,
        "p99_mean": float(np.mean([r["p99_abs"] for r in summary_rows])) if summary_rows else 0.0,
        "size_pass": float(size_pass_count),
    }

    log_path = out_dir / args.log_file
    write_text_report(log_path, report_lines, summary)

    print(f"[OK] residual maps generated: {out_dir.resolve()}")
    print(f"[OK] paired samples: {len(summary_rows)}")
    print(f"[OK] text report: {log_path.resolve()}")


if __name__ == "__main__":
    main()
