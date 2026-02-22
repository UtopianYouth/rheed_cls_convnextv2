#!/usr/bin/env bash
set -e

SSH_HOST="172.31.233.227"
SSH_USER="omnisky"
REMOTE_DIR="/rheed/"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PROJECT_NAME="$(basename "$PROJECT_ROOT")"
 
echo "上传代码文件..."

rsync -az "$PROJECT_ROOT/src/" "$SSH_USER@$SSH_HOST:$REMOTE_DIR/$PROJECT_NAME/src/"
rsync -az "$PROJECT_ROOT/scripts/" "$SSH_USER@$SSH_HOST:$REMOTE_DIR/$PROJECT_NAME/scripts/"
rsync -az "$PROJECT_ROOT/docs/" "$SSH_USER@$SSH_HOST:$REMOTE_DIR/$PROJECT_NAME/docs/"

echo "✅ 代码文件上传完成"
