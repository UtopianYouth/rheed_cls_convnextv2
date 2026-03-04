#!/usr/bin/env bash
# 基于fcmae_images中的src/dst图像生成残差热力图与对比图

set -e
cd "$(dirname "$0")/.."

INPUT_DIR=${INPUT_DIR:-"fcmae_images"}
OUT_DIR=${OUT_DIR:-"$INPUT_DIR/residual_maps"}
CMAP=${CMAP:-"lightgreen"}
VMAX_PERCENTILE=${VMAX_PERCENTILE:-"99"}

python -m tools.generate_fcmae_residual_heatmaps \
  --input_dir "$INPUT_DIR" \
  --output_dir "$OUT_DIR" \
  --heatmap_cmap "$CMAP" \
  --vmax_percentile "$VMAX_PERCENTILE"

echo "[DONE] Residual heatmaps generated at: $OUT_DIR"
