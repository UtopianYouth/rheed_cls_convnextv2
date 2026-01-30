#!/usr/bin/env bash
set -e
CONDA_BASE=$(conda info --base)
TARGET_PATH="$CONDA_BASE/envs/rheed_cls_gpu"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
ENV_FILE="$PROJECT_ROOT/rheed_cls_gpu.tar"

[ ! -f "$ENV_FILE" ] && echo "Error: $ENV_FILE not found" && exit 1

echo "Unpacking conda environment..."
if [ -d "$TARGET_PATH" ]; then
    echo "Existing env found, removing: $TARGET_PATH"
    rm -rf "$TARGET_PATH"
fi
mkdir -p "$TARGET_PATH"
tar -xf "$ENV_FILE" -C "$TARGET_PATH"

echo "Fixing environment paths..."
source "$TARGET_PATH/bin/activate"
conda-unpack

echo "Done: $TARGET_PATH"
echo "Activate: conda activate rheed_cls_gpu"
