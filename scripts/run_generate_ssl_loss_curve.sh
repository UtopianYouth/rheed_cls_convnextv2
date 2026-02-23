#!/usr/bin/env bash
# 生成自监督SSL loss曲线（极简脚本）

set -e
cd "$(dirname "$0")/.."

EXP_DIR=${EXP_DIR:-"outputs/pretrain_ssl_20260223_150207"}
OUTPUT_DIR=${OUTPUT_DIR:-"docs/my_graduate_paper/figure/cp2"}
DPI=${DPI:-300}

LOG_PATH="$EXP_DIR/log.txt"
OUTPUT_PATH="$OUTPUT_DIR/ssl_loss_curve_$(basename "$EXP_DIR").png"

if [ ! -f "$LOG_PATH" ]; then
  echo "错误：未找到日志文件：$LOG_PATH"
  exit 1
fi

mkdir -p "$OUTPUT_DIR"

python -m tools.generate_ssl_loss_curve \
  --log_path "$LOG_PATH" \
  --output_path "$OUTPUT_PATH" \
  --dpi "$DPI"

echo "生成完成：$OUTPUT_PATH"
