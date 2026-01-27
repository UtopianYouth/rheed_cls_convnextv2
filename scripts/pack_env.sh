#!/usr/bin/env bash
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
OUTPUT_FILE="$PROJECT_ROOT/rheed_cls_gpu.tar"

echo "Packing conda environment (tar format)..."
conda pack -n rheed_cls_gpu -o "$OUTPUT_FILE" --format tar

echo "Verifying package..."
if [ -f "$OUTPUT_FILE" ]; then
    echo "Package complete: $OUTPUT_FILE"
    ls -lh "$OUTPUT_FILE"
else
    echo "Package failed"
    exit 1
fi
