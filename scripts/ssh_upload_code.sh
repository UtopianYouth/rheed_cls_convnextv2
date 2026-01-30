#!/usr/bin/env bash
set -e

SSH_HOST="172.31.233.200"
SSH_USER="omnisky"
REMOTE_DIR="/home/omnisky/public/hqing2025"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PROJECT_NAME="$(basename "$PROJECT_ROOT")"
 
echo "上传代码文件..."
rsync -az --exclude='outputs/' --exclude='__pycache__/' --exclude='*.pyc' \
  --exclude='.git/' --exclude='env/' --exclude='.DS_Store' --exclude='.vscode/' \
  --exclude='data_rheed/' --exclude='data_rheed_split/' --exclude='docs/' \
  --exclude='*.tar' \
  --exclude='*.tar.gz' \
  "$PROJECT_ROOT/" "$SSH_USER@$SSH_HOST:$REMOTE_DIR/$PROJECT_NAME/"
echo "✅ 代码文件上传完成"