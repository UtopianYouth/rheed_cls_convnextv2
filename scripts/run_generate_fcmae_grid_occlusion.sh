#!/usr/bin/env bash
# 批量生成8x8网格遮挡图（灰色版 + 橙色可调透明度覆盖）

set -euo pipefail
cd "$(dirname "$0")/.."

INPUT_DIR=${INPUT_DIR:-"fcmae_images"}
OUT_DIR=${OUT_DIR:-"$INPUT_DIR/grid_occlusion"}
GRID_WIDTH=${GRID_WIDTH:-2}
ORANGE_ALPHA=${ORANGE_ALPHA:-0.20}

mkdir -p "$OUT_DIR"

echo "输入目录: $INPUT_DIR"
echo "输出目录: $OUT_DIR"

count=0
while IFS= read -r -d '' img; do
  base="$(basename "$img")"
  stem="${base%.*}"

  out_gray="$OUT_DIR/${stem}_grid_mask_gray.png"
  out_orange="$OUT_DIR/${stem}_grid_mask_orange_alpha${ORANGE_ALPHA}.png"

  python -m tools.generate_fcmae_grid_occlusion \
    --input "$img" \
    --out_gray "$out_gray" \
    --out_orange "$out_orange" \
    --grid_width "$GRID_WIDTH" \
    --orange_alpha "$ORANGE_ALPHA"

  count=$((count + 1))
done < <(find "$INPUT_DIR" -maxdepth 1 -type f \( -iname "*.png" -o -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.tif" -o -iname "*.tiff" -o -iname "*.bmp" \) -print0 | sort -z)

if [ "$count" -eq 0 ]; then
  echo "[WARN] 未在 $INPUT_DIR 找到可处理图像"
  exit 0
fi

echo "[DONE] 共处理 $count 张图像，结果位于: $OUT_DIR"
