#!/usr/bin/env bash
set -e

SSH_HOST="172.31.233.227"
SSH_USER="omnisky"
REMOTE_DIR="/rheed/"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PROJECT_NAME="$(basename "$PROJECT_ROOT")"

echo "上传数据集..."
rsync -az "$PROJECT_ROOT/rheed_images/" \
  "$SSH_USER@$SSH_HOST:$REMOTE_DIR/$PROJECT_NAME/rheed_images/"
echo "✅ 数据集上传完成"
