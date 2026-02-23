#!/usr/bin/env bash
set -e

SSH_HOST="172.31.233.227"
SSH_USER="omnisky"
REMOTE_DIR="/rheed/"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PROJECT_NAME="$(basename "$PROJECT_ROOT")"

echo "上传数据集（仅seq_001）..."
rsync -az "$PROJECT_ROOT/rheed_images/2d/seq_001/" \
  "$SSH_USER@$SSH_HOST:$REMOTE_DIR/$PROJECT_NAME/rheed_images/2d/seq_001/"
rsync -az "$PROJECT_ROOT/rheed_images/3d/seq_001/" \
  "$SSH_USER@$SSH_HOST:$REMOTE_DIR/$PROJECT_NAME/rheed_images/3d/seq_001/"
echo "✅ 数据集上传完成（仅seq_001）"
