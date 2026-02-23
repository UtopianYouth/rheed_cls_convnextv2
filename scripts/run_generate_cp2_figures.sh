#!/bin/bash
# 生成第二章实验图表

set -e

EXP_DIR="outputs/finetune_timm_20260212_175658"
OUTPUT_DIR="docs/my_graduate_paper/figure/cp2"
DPI=300
NORMALIZE_CM="--normalize_cm"

if [ ! -d "$EXP_DIR" ]; then
    echo "错误：实验目录不存在：$EXP_DIR"
    exit 1
fi

if [ ! -f "$EXP_DIR/log.txt" ] || [ ! -f "$EXP_DIR/test_metrics.json" ]; then
    echo "错误：缺少log.txt或test_metrics.json"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

EXP_BASENAME=$(basename "$EXP_DIR")
TIME_SUFFIX=$(echo "$EXP_BASENAME" | grep -oE '[0-9]{8}_[0-9]{6}$')
if [ -z "$TIME_SUFFIX" ]; then
    TIME_SUFFIX="unknown"
fi

python -m tools.generate_cp2_figures \
    --exp_dir "$EXP_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --dpi "$DPI" \
    $NORMALIZE_CM

TRAINING_SRC="$OUTPUT_DIR/training_curves.png"
CONFUSION_SRC="$OUTPUT_DIR/confusion_matrix_test.png"
TRAINING_DST="$OUTPUT_DIR/training_curves_${TIME_SUFFIX}.png"
CONFUSION_DST="$OUTPUT_DIR/confusion_matrix_test_${TIME_SUFFIX}.png"

if [ -f "$TRAINING_SRC" ] && [ -f "$CONFUSION_SRC" ]; then
    mv "$TRAINING_SRC" "$TRAINING_DST"
    mv "$CONFUSION_SRC" "$CONFUSION_DST"
    echo "生成完成："
    echo "  $TRAINING_DST"
    echo "  $CONFUSION_DST"
else
    echo "错误：图表生成失败"
    exit 1
fi
