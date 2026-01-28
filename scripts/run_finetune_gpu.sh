#!/usr/bin/env bash
# RHEED图像分类训练脚本 (GPU自动选择: 单卡/多卡)

set -e
cd "$(dirname "$0")/.."

# 检查可用GPU
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv
    echo "------------------------------------------------"
    
    # GPU选择
    read -p "GPU IDs (多个用逗号分隔, 默认: 0): " gpu_ids
    gpu_ids=${gpu_ids:-"0"}
    
    export CUDA_VISIBLE_DEVICES=$gpu_ids
    IFS=',' read -ra GPU_ARRAY <<< "$gpu_ids"
    NUM_GPUS=${#GPU_ARRAY[@]}
    echo "使用GPU: $CUDA_VISIBLE_DEVICES (共 $NUM_GPUS 张)"
else
    echo "⚠️  未检测到GPU, 将使用CPU训练"
    NUM_GPUS=0
fi

# 默认参数
OUTPUT_DIR="finetune_rheed_$(date +%Y%m%d_%H%M%S)"
BATCH_SIZE=32
EPOCHS=100
BASE_LR=5e-4

# 类别权重
read -p "Class weights (例如: 1.0 4.8, 留空禁用): " CLASS_WEIGHTS
EXTRA_ARGS=()
if [ -n "$CLASS_WEIGHTS" ]; then
    EXTRA_ARGS=(--class_weights $CLASS_WEIGHTS)
fi

# 使用划分后的数据集
DATA_PATH="data_rheed/train"
VAL_PATH="data_rheed/val"

echo "开始训练 - 输出目录: outputs/$OUTPUT_DIR"

# 单GPU/CPU 用 python, 多GPU用 torchrun
if [ "$NUM_GPUS" -gt 1 ]; then
    export MASTER_ADDR=127.0.0.1
    export MASTER_PORT=29500

    torchrun --standalone \
      --nproc_per_node=$NUM_GPUS \
      -m src.main_finetune \
      --model convnextv2_tiny \
      --batch_size $BATCH_SIZE \
      --epochs $EPOCHS \
      --blr $BASE_LR \
      --layer_decay 0.9 \
      --warmup_epochs 10 \
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
      --save_ckpt_freq 10 \
      --save_ckpt_num 3 \
      --num_workers 8 \
      --device cuda \
      --use_amp false \
      --pin_mem true \
      --imagenet_default_mean_and_std true
else
    DEVICE="cuda"
    if [ "$NUM_GPUS" -eq 0 ]; then
        DEVICE="cpu"
    fi

    python -m src.main_finetune \
      --model convnextv2_tiny \
      --batch_size $BATCH_SIZE \
      --epochs $EPOCHS \
      --blr $BASE_LR \
      --layer_decay 0.9 \
      --warmup_epochs 10 \
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
      --save_ckpt_freq 10 \
      --save_ckpt_num 3 \
      --num_workers 8 \
      --device $DEVICE \
      --use_amp false \
      --pin_mem true \
      --imagenet_default_mean_and_std true
fi

echo "训练完成, 输出: outputs/$OUTPUT_DIR"
echo "查看训练曲线: tensorboard --logdir outputs/$OUTPUT_DIR/logs"
