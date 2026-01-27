#!/usr/bin/env bash
set -e

SSH_HOST="172.31.233.200"
SSH_USER="omnisky"
REMOTE_DIR="/home/omnisky/public/hqing2025"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PROJECT_NAME="$(basename "$PROJECT_ROOT")"
ENV_FILE="$PROJECT_ROOT/rheed_cls_gpu.tar"

echo "Calculating local MD5..."
LOCAL_MD5=$(md5sum "$ENV_FILE" | awk '{print $1}')
echo "Local MD5: $LOCAL_MD5"

echo "Uploading conda environment..."
rsync -avz --partial --progress --checksum \
  "$ENV_FILE" "$SSH_USER@$SSH_HOST:$REMOTE_DIR/$PROJECT_NAME/"

echo "Verifying remote MD5..."
REMOTE_MD5=$(ssh "$SSH_USER@$SSH_HOST" "md5sum $REMOTE_DIR/$PROJECT_NAME/rheed_cls_gpu.tar | awk '{print \$1}'")
echo "Remote MD5: $REMOTE_MD5"

if [ "$LOCAL_MD5" = "$REMOTE_MD5" ]; then
    echo "Upload complete, MD5 verified"
else
    echo "MD5 mismatch, retrying..."
    rsync -avz --partial --progress --checksum \
      "$ENV_FILE" "$SSH_USER@$SSH_HOST:$REMOTE_DIR/$PROJECT_NAME/"
    
    REMOTE_MD5=$(ssh "$SSH_USER@$SSH_HOST" "md5sum $REMOTE_DIR/$PROJECT_NAME/rheed_cls_gpu.tar | awk '{print \$1}'")
    
    if [ "$LOCAL_MD5" = "$REMOTE_MD5" ]; then
        echo "Retry successful"
    else
        echo "Retry failed"
        exit 1
    fi
fi
