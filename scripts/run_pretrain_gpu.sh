# FCMAE/ConvNeXtV2 预训练脚本 (GPU自动选择: 单卡/多卡)
#
# 数据目录约定：
# - --data_path 指向一个目录，且其下有 train/
# - train/ 采用 ImageFolder 格式：train/<class_name>/*.tiff
#   预训练是自监督，不用真实标签；这里的 class_name 仅用于满足 ImageFolder 读取格式
#   数据来源：data_rheed/train（rough, smooth 两类）
#   预训练只使用 train 数据
#

set -e
cd "$(dirname "$0")/.."

# 避免 DDP 下每个进程默认开太多 CPU 线程导致系统过载
: "${OMP_NUM_THREADS:=8}"
: "${MKL_NUM_THREADS:=8}"
export OMP_NUM_THREADS MKL_NUM_THREADS

# 分布式后端 (默认 nccl；若遇到 NCCL/驱动相关崩溃可临时用 gloo 排查)
DIST_BACKEND=${DIST_BACKEND:-"nccl"}

# NCCL 排障常用开关 (如需可在命令行预先 export 覆盖)
: "${NCCL_ASYNC_ERROR_HANDLING:=1}"
export NCCL_ASYNC_ERROR_HANDLING

# Debug: 开启更多 native traceback / 分布式日志（默认关闭，避免刷屏）
DEBUG_PRETRAIN=${DEBUG_PRETRAIN:-0}
if [ "$DEBUG_PRETRAIN" = "1" ]; then
  : "${PYTHONFAULTHANDLER:=1}"
  : "${TORCH_SHOW_CPP_STACKTRACES:=1}"
  : "${TORCH_DISTRIBUTED_DEBUG:=DETAIL}"
  export PYTHONFAULTHANDLER TORCH_SHOW_CPP_STACKTRACES TORCH_DISTRIBUTED_DEBUG
  # gloo 调试日志（仅在你设置 DIST_BACKEND=gloo 时更有意义）
  : "${GLOO_LOG_LEVEL:=DEBUG}"
  export GLOO_LOG_LEVEL
  echo "[debug] DEBUG_PRETRAIN=1 enabled"
fi

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
BATCH_SIZE=16
EPOCHS=150
BASE_LR=1.5e-4
MASK_RATIO=0.6
INPUT_SIZE=384
NUM_WORKERS=8

# 数据根目录 (其下需要有 train/)
DATA_PATH="data_rheed"

echo "开始预训练 - 输出目录: outputs/$OUTPUT_DIR"

echo "dist_backend=$DIST_BACKEND"

auto_args=(
  --model $MODEL
  --batch_size $BATCH_SIZE
  --epochs $EPOCHS
  --blr $BASE_LR
  --mask_ratio $MASK_RATIO
  --input_size $INPUT_SIZE
  --weight_decay 0.05
  --warmup_epochs 10
  --min_lr 0.0
  --data_path $DATA_PATH
  --output_dir outputs/$OUTPUT_DIR
  --log_dir outputs/$OUTPUT_DIR/logs
  --auto_resume true
  --save_ckpt true
  --save_ckpt_freq 10
  --save_ckpt_num 3
  --num_workers $NUM_WORKERS
  --pin_mem true
  --dist_backend $DIST_BACKEND
)

if [ "$NUM_GPUS" -gt 1 ]; then
    export MASTER_ADDR=127.0.0.1
    export MASTER_PORT=29501

    torchrun --standalone \
      --nproc_per_node=$NUM_GPUS \
      -m src.main_pretrain \
      --device cuda \
      "${auto_args[@]}"
else
    DEVICE="cuda"
    if [ "$NUM_GPUS" -eq 0 ]; then
        DEVICE="cpu"
    fi

    python -m src.main_pretrain \
      --device $DEVICE \
      "${auto_args[@]}"
fi

echo "预训练完成, 输出: outputs/$OUTPUT_DIR"
echo "查看训练曲线: tensorboard --logdir outputs/$OUTPUT_DIR/logs"
