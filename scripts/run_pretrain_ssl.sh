#!/usr/bin/env bash
# RHEED 序列自监督预训练脚本（SimCLR风格）
# 支持单卡/多卡（DataParallel），默认使用GPU 0

set -e
cd "$(dirname "$0")/.."

# GPU状态展示（显存百分比）
if command -v nvidia-smi &> /dev/null; then
  echo "=== 当前GPU使用状态 ==="
  nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits | \
  while IFS=, read -r idx name util used total; do
    mem_percent=$(( used * 100 / total ))
    printf "GPU %s: %s | GPU利用率: %s%% | 显存: %s/%s MB (%s%%)\n" "$idx" "$name" "$util" "$used" "$total" "$mem_percent"
  done
  echo "========================"
else
  echo "未检测到nvidia-smi，使用CPU训练"
  NUM_GPUS=0
fi

# GPU选择（强制单卡；若未指定GPU_IDS则交互输入）
if command -v nvidia-smi &> /dev/null; then
  if [ -n "${GPU_IDS:-}" ]; then
    gpu_id="$GPU_IDS"
  else
    read -p "请输入要使用的GPU编号（单卡，例如 0）：" gpu_id
    gpu_id=${gpu_id:-0}
  fi
  export CUDA_VISIBLE_DEVICES=$gpu_id
  NUM_GPUS=1
  echo "使用GPU: $CUDA_VISIBLE_DEVICES (单卡)"
fi

# 快速测试交互（默认不进入测试模式）
if [ -n "${QUICK_TEST:-}" ]; then
  epochs_val=1
else
  read -p "是否进行快速测试（仅1个epoch）？[y/N]：" ans
  if [[ "$ans" =~ ^[yY]([eE][sS])?$ ]]; then
    epochs_val=1
  else
    epochs_val=${EPOCHS:-80}
  fi
fi

# 可通过环境变量覆盖
MODEL=${MODEL:-"convnextv2_tiny"}
BATCH_SIZE=${BATCH_SIZE:-64}
EPOCHS=$epochs_val
LR=${LR:-3e-4}
INPUT_SIZE=${INPUT_SIZE:-224}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.01}
NUM_WORKERS=${NUM_WORKERS:-16}
PIN_MEMORY=${PIN_MEMORY:-true}
PERSISTENT_WORKERS=${PERSISTENT_WORKERS:-true}
PREFETCH_FACTOR=${PREFETCH_FACTOR:-4}
USE_AMP=${USE_AMP:-true}
TEMPERATURE=${TEMPERATURE:-0.1}
AA=${AA:-none}
COLOR_JITTER=${COLOR_JITTER:-0.0}
REPROB=${REPROB:-0.0}

# 序列数据路径
SEQUENCE_ROOT=${SEQUENCE_ROOT:-"rheed_images_png"}
WINDOW_SIZE=${WINDOW_SIZE:-10}
WINDOW_STRIDE=${WINDOW_STRIDE:-2}
SPLIT_RATIO=${SPLIT_RATIO:-"0.7,0.2,0.1"}
STRICT_TIME_SPLIT=${STRICT_TIME_SPLIT:-true}
SEQ_TRAIN=${SEQ_TRAIN:-"seq_001,seq_002,seq_003,seq_004,seq_005,seq_006,seq_007"}
SEQ_VAL=${SEQ_VAL:-"seq_008"}
SEQ_TEST=${SEQ_TEST:-"seq_009,seq_010"}

# 预训练权重（可选）
PRETRAINED=${PRETRAINED:-false}
PRETRAINED_WEIGHTS=${PRETRAINED_WEIGHTS:-""}

EXTRA_ARGS=()
if [ -n "$PRETRAINED_WEIGHTS" ]; then
  PRETRAINED=true
  EXTRA_ARGS+=(--pretrained_weights "$PRETRAINED_WEIGHTS")
fi

echo "运行模式: sequence_pretrain_ssl | WINDOW_SIZE=$WINDOW_SIZE STRIDE=$WINDOW_STRIDE LR=$LR WD=$WEIGHT_DECAY TEMP=$TEMPERATURE AA=$AA CJ=$COLOR_JITTER REPROB=$REPROB"

OUTPUT_DIR="outputs/pretrain_ssl_$(date +%Y%m%d_%H%M%S)"

PRETRAIN_ARGS=(
  --model "$MODEL"
  --sequence_root "$SEQUENCE_ROOT"
  --window_size "$WINDOW_SIZE"
  --window_stride "$WINDOW_STRIDE"
  --split_ratio "$SPLIT_RATIO"
  --strict_time_split "$STRICT_TIME_SPLIT"
  --seq_train "$SEQ_TRAIN"
  --seq_val "$SEQ_VAL"
  --seq_test "$SEQ_TEST"
  --input_size "$INPUT_SIZE"
  --batch_size "$BATCH_SIZE"
  --epochs "$EPOCHS"
  --lr "$LR"
  --weight_decay "$WEIGHT_DECAY"
  --temperature "$TEMPERATURE"
  --aa "$AA"
  --color_jitter "$COLOR_JITTER"
  --reprob "$REPROB"
  --num_workers "$NUM_WORKERS"
  --pin_memory "$PIN_MEMORY"
  --persistent_workers "$PERSISTENT_WORKERS"
  --prefetch_factor "$PREFETCH_FACTOR"
  --device cuda
  --use_amp "$USE_AMP"
  --pretrained "$PRETRAINED"
  --output_dir "$OUTPUT_DIR"
)

python -m src.pretrain_ssl "${PRETRAIN_ARGS[@]}" "${EXTRA_ARGS[@]}"

echo "预训练完成, 输出: $OUTPUT_DIR"
