# FCMAE/ConvNeXtV2 预训练脚本 (GPU自动选择: 单卡/多卡)
#
# 数据目录约定：
# - --data_path 指向一个目录，且其下有 train/
# - train/ 采用 ImageFolder 格式：train/<class_name>/*.tiff
#   预训练是自监督，不用真实标签；这里的 class_name 仅用于满足 ImageFolder 读取格式
#   数据来源：data_rheed/train（rough, smooth 两类）
#   注意：预训练只使用 train 数据，不使用 val/test 数据

set -e
cd "$(dirname "$0")/.."

# 检查可用GPU
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv
    echo "------------------------------------------------"

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

# 默认参数 (你可以按显存/数据量自行调整)
OUTPUT_DIR="pretrain_rheed_$(date +%Y%m%d_%H%M%S)"
MODEL="convnextv2_base"
BATCH_SIZE=64
EPOCHS=200
BASE_LR=1.5e-4
MASK_RATIO=0.6
INPUT_SIZE=224
NUM_WORKERS=8

# 数据根目录 (其下需要有 train/)
DATA_PATH="data_rheed"

echo "开始预训练 - 输出目录: outputs/$OUTPUT_DIR"

if [ "$NUM_GPUS" -gt 1 ]; then
    export MASTER_ADDR=127.0.0.1
    export MASTER_PORT=29501

    torchrun --standalone \
      --nproc_per_node=$NUM_GPUS \
      -m src.main_pretrain \
      --model $MODEL \
      --batch_size $BATCH_SIZE \
      --epochs $EPOCHS \
      --blr $BASE_LR \
      --mask_ratio $MASK_RATIO \
      --input_size $INPUT_SIZE \
      --weight_decay 0.05 \
      --warmup_epochs 20 \
      --min_lr 0.0 \
      --data_path $DATA_PATH \
      --output_dir outputs/$OUTPUT_DIR \
      --log_dir outputs/$OUTPUT_DIR/logs \
      --auto_resume true \
      --save_ckpt true \
      --save_ckpt_freq 10 \
      --save_ckpt_num 3 \
      --num_workers $NUM_WORKERS \
      --device cuda \
      --pin_mem true
else
    DEVICE="cuda"
    if [ "$NUM_GPUS" -eq 0 ]; then
        DEVICE="cpu"
    fi

    python -m src.main_pretrain \
      --model $MODEL \
      --batch_size $BATCH_SIZE \
      --epochs $EPOCHS \
      --blr $BASE_LR \
      --mask_ratio $MASK_RATIO \
      --input_size $INPUT_SIZE \
      --weight_decay 0.05 \
      --warmup_epochs 20 \
      --min_lr 0.0 \
      --data_path $DATA_PATH \
      --output_dir outputs/$OUTPUT_DIR \
      --log_dir outputs/$OUTPUT_DIR/logs \
      --auto_resume true \
      --save_ckpt true \
      --save_ckpt_freq 10 \
      --save_ckpt_num 3 \
      --num_workers $NUM_WORKERS \
      --device $DEVICE \
      --pin_mem true
fi

echo "预训练完成, 输出: outputs/$OUTPUT_DIR"
echo "查看训练曲线: tensorboard --logdir outputs/$OUTPUT_DIR/logs"
