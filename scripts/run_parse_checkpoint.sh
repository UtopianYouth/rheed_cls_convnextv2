#!/usr/bin/env bash
# 极简解析checkpoint脚本（预训练/微调通用）

set -e
cd "$(dirname "$0")/.."

CKPT_PATH=${CKPT_PATH:-"outputs/pretrain_ssl_20260221_151605/checkpoint_last.pth"}

if [ ! -f "$CKPT_PATH" ]; then
  echo "错误：未找到checkpoint：$CKPT_PATH"
  exit 1
fi

python -m src.parse_checkpoint \
  --ckpt_path "$CKPT_PATH"
