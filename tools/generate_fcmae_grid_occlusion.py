#!/usr/bin/env python3
"""基于输入图像生成 8x8 网格遮挡图（灰色版 + 橙色半透明版）。

功能：
1) 在图像内部绘制 8x8 网格线（不绘制图像边缘线）。
2) 将左上 4x4 与右下 4x4 共 32 个网格进行遮挡：
   - 输出1：全灰色填充
   - 输出2：90% 不透明橙色覆盖

示例：
python tools/generate_fcmae_grid_occlusion.py \
  --input /path/to/image.png \
  --out_gray /path/to/image_mask_gray.png \
  --out_orange /path/to/image_mask_orange.png
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

from PIL import Image, ImageDraw


GridBox = Tuple[int, int, int, int]


def grid_boundaries(length: int, n: int = 8) -> List[int]:
    return [round(i * length / n) for i in range(n + 1)]


def build_mask_regions(width: int, height: int) -> List[GridBox]:
    xs = grid_boundaries(width, 8)
    ys = grid_boundaries(height, 8)

    regions: List[GridBox] = []

    # 左上角 4x4
    for r in range(4):
        for c in range(4):
            regions.append((xs[c], ys[r], xs[c + 1], ys[r + 1]))

    # 右下角 4x4
    for r in range(4, 8):
        for c in range(4, 8):
            regions.append((xs[c], ys[r], xs[c + 1], ys[r + 1]))

    return regions


def draw_grid_lines(img: Image.Image, width_px: int, color=(245, 245, 245, 255)) -> None:
    w, h = img.size
    xs = grid_boundaries(w, 8)
    ys = grid_boundaries(h, 8)

    draw = ImageDraw.Draw(img, "RGBA")

    # 仅绘制内部网格线，不绘制边缘线
    for x in xs[1:-1]:
        draw.line([(x, 0), (x, h)], fill=color, width=width_px)
    for y in ys[1:-1]:
        draw.line([(0, y), (w, y)], fill=color, width=width_px)


def make_gray_masked(base_rgb: Image.Image, regions: List[GridBox], grid_width: int) -> Image.Image:
    out = base_rgb.copy().convert("RGBA")
    draw = ImageDraw.Draw(out, "RGBA")

    for box in regions:
        draw.rectangle(box, fill=(128, 128, 128, 255))

    draw_grid_lines(out, grid_width)
    return out.convert("RGB")


def make_orange_overlay(base_rgb: Image.Image, regions: List[GridBox], grid_width: int, alpha: float = 0.9) -> Image.Image:
    out = base_rgb.copy().convert("RGBA")
    overlay = Image.new("RGBA", out.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay, "RGBA")

    a = int(max(0.0, min(1.0, alpha)) * 255)
    orange = (255, 140, 0, a)

    for box in regions:
        draw.rectangle(box, fill=orange)

    out = Image.alpha_composite(out, overlay)
    draw_grid_lines(out, grid_width)
    return out.convert("RGB")


def main() -> None:
    parser = argparse.ArgumentParser(description="生成 8x8 网格遮挡图（灰色 + 橙色可调透明度）")
    parser.add_argument("--input", required=True, help="输入图片路径")
    parser.add_argument("--out_gray", required=True, help="灰色遮挡输出路径")
    parser.add_argument("--out_orange", required=True, help="橙色遮挡输出路径")
    parser.add_argument("--grid_width", type=int, default=2, help="网格线宽（像素）")
    parser.add_argument("--orange_alpha", type=float, default=0.05, help="橙色覆盖不透明度，范围[0,1]，例如0.05表示5%")
    args = parser.parse_args()

    in_path = Path(args.input)
    out_gray = Path(args.out_gray)
    out_orange = Path(args.out_orange)

    if not in_path.exists():
        raise FileNotFoundError(f"输入图片不存在: {in_path}")

    img = Image.open(in_path).convert("RGB")
    w, h = img.size

    # 线宽做一个轻度自适应，确保“可见但不过分抢眼”
    grid_width = max(1, min(args.grid_width, max(1, min(w, h) // 80)))

    regions = build_mask_regions(w, h)

    gray_img = make_gray_masked(img, regions, grid_width)
    orange_img = make_orange_overlay(img, regions, grid_width, alpha=args.orange_alpha)

    out_gray.parent.mkdir(parents=True, exist_ok=True)
    out_orange.parent.mkdir(parents=True, exist_ok=True)

    gray_img.save(out_gray)
    orange_img.save(out_orange)

    print(f"[OK] input: {in_path} ({w}x{h})")
    print("[OK] grid: 8x8 (internal lines only)")
    print("[OK] masked regions: top-left 4x4 + bottom-right 4x4")
    print(f"[OK] orange alpha: {max(0.0, min(1.0, args.orange_alpha)):.3f}")
    print(f"[OK] gray output: {out_gray}")
    print(f"[OK] orange output: {out_orange}")


if __name__ == "__main__":
    main()
