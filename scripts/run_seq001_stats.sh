#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

SEQUENCE_ROOT=${SEQUENCE_ROOT:-"$PROJECT_ROOT/rheed_images"}
OUTPUT_DIR=${OUTPUT_DIR:-"$PROJECT_ROOT/outputs/seq001_stats"}

cd "$PROJECT_ROOT"

python -m tools.compare_seq001_stats \
  --sequence_root "$SEQUENCE_ROOT" \
  --output_dir "$OUTPUT_DIR"
