#!/usr/bin/env python3
"""
将指定目录中的TIFF图像批量转换为PNG（线性归一化），用于可视化与后续处理。

用法：
python -m tools.convert_single_tiff_to_png \
  --input_dir /path/to/seq_001 \
  --output_dir /path/to/rheed_images_png/2d
"""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser(description="TIFF目录转PNG（线性归一化）")
    parser.add_argument("--input_dir", type=str, required=True, help="输入目录（seq_001）")
    parser.add_argument("--output_dir", type=str, required=True, help="输出目录（不含seq子目录）")
    return parser.parse_args()


def _linear_normalize(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float32)
    vmin = float(arr.min())
    vmax = float(arr.max())
    if vmax <= vmin:
        return np.zeros_like(arr, dtype=np.uint8)
    norm = (arr - vmin) / (vmax - vmin)
    norm = np.clip(norm * 255.0, 0, 255).astype(np.uint8)
    return norm


def _list_tiff_files(input_dir: Path):
    files = list(input_dir.glob("*.tif")) + list(input_dir.glob("*.tiff"))
    files.sort()
    if not files:
        raise FileNotFoundError(f"未找到TIFF文件：{input_dir}")
    return files


def _convert_one(input_path: Path, output_dir: Path):
    with Image.open(input_path) as img:
        img = img.convert("I")
        arr = np.array(img)

    norm = _linear_normalize(arr)
    out_img = Image.fromarray(norm, mode="L")
    output_path = output_dir / f"{input_path.stem}.png"
    out_img.save(output_path)
    return output_path


def main():
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.exists():
        raise FileNotFoundError(f"输入目录不存在：{input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    files = _list_tiff_files(input_dir)

    for fp in files:
        _convert_one(fp, output_dir)

    print(f"转换完成：{input_dir} -> {output_dir} (共{len(files)}张)")


if __name__ == "__main__":
    main()
