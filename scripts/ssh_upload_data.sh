#!/usr/bin/env bash
set -e

SSH_HOST="172.31.233.200"
SSH_USER="omnisky"
REMOTE_DIR="/home/omnisky/public/hqing2025"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PROJECT_NAME="$(basename "$PROJECT_ROOT")"

echo "上传数据集..."
rsync -az "$PROJECT_ROOT/data_rheed_split/" \
  "$SSH_USER@$SSH_HOST:$REMOTE_DIR/$PROJECT_NAME/data_rheed_split/"
echo "✅ 数据集上传完成"
