#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# 仅转换seq_001，输出到rheed_images_png/{2d,3d}，不保留seq子目录
INPUT_DIR_2D="$PROJECT_ROOT/rheed_images/2d/seq_001"
INPUT_DIR_3D="$PROJECT_ROOT/rheed_images/3d/seq_001"
OUTPUT_DIR_2D="$PROJECT_ROOT/rheed_images_png/2d"
OUTPUT_DIR_3D="$PROJECT_ROOT/rheed_images_png/3d"

cd "$PROJECT_ROOT"

echo "转换2d/seq_001 -> rheed_images_png/2d"
python -m tools.convert_single_tiff_to_png \
  --input_dir "$INPUT_DIR_2D" \
  --output_dir "$OUTPUT_DIR_2D"

echo "转换3d/seq_001 -> rheed_images_png/3d"
python -m tools.convert_single_tiff_to_png \
  --input_dir "$INPUT_DIR_3D" \
  --output_dir "$OUTPUT_DIR_3D"
