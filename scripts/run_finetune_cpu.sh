#!/usr/bin/env bash
# RHEED图像分类训练脚本 (CPU版本 - 本地调试用)

set -e
cd "$(dirname "$0")/.."

# 默认参数（CPU优化：小batch、少epoch）
OUTPUT_DIR="finetune_rheed_cpu_$(date +%Y%m%d_%H%M%S)"
BATCH_SIZE=1   
EPOCHS=1
BASE_LR=5e-4
NUM_WORKERS=2

# 使用划分后的数据集
DATA_PATH="data_rheed_split/train"
VAL_PATH="data_rheed_split/val"

# 检查数据集是否存在
if [ ! -d "$DATA_PATH" ]; then
    echo "❌ 错误: 数据集目录不存在: $DATA_PATH"
    echo "请先运行: python scripts/split_dataset_simple.py"
    exit 1
fi

# 类别权重配置（可选）
read -p "是否使用类别权重? (y/N): " use_weights
EXTRA_ARGS=()
if [[ "$use_weights" =~ ^[Yy]$ ]]; then
    read -p "Class weights (例如: 1.0 4.8): " CLASS_WEIGHTS
    if [ -n "$CLASS_WEIGHTS" ]; then
        EXTRA_ARGS=(--class_weights $CLASS_WEIGHTS)
    fi
fi

echo "开始训练, 输出目录: outputs/$OUTPUT_DIR"

# 开始训练
python -m src.main_finetune \
  --model convnextv2_tiny \
  --batch_size $BATCH_SIZE \
  --epochs $EPOCHS \
  --blr $BASE_LR \
  --layer_decay 0.9 \
  --warmup_epochs 2 \
  --min_lr 1e-6 \
  --weight_decay 0.05 \
  --data_set image_folder \
  --data_path $DATA_PATH \
  --eval_data_path $VAL_PATH \
  --nb_classes 2 \
  --input_size 224 \
  --drop_path 0.1 \
  --mixup 0.8 \
  --cutmix 1.0 \
  --mixup_prob 1.0 \
  --mixup_switch_prob 0.5 \
  --reprob 0.25 \
  --remode pixel \
  --smoothing 0.1 \
  "${EXTRA_ARGS[@]}" \
  --aa rand-m9-mstd0.5-inc1 \
  --train_interpolation bicubic \
  --output_dir outputs/$OUTPUT_DIR \
  --log_dir outputs/$OUTPUT_DIR/logs \
  --auto_resume true \
  --save_ckpt true \
  --save_ckpt_freq 2 \
  --save_ckpt_num 2 \
  --num_workers $NUM_WORKERS \
  --device cpu \
  --use_amp false \
  --pin_mem false \
  --imagenet_default_mean_and_std true

echo "✅ 训练完成! 输出: outputs/$OUTPUT_DIR"
echo "查看训练曲线: tensorboard --logdir outputs/$OUTPUT_DIR/logs"
