#!/usr/bin/env bash
# RHEED图像分类训练脚本（timm版）
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

# A100 80G 显存调参参考（单卡）：
# - BATCH_SIZE: 32~64 (推荐32，显存安全；64 若OOM则降为32)
# - EPOCHS: 50~100 (若已收敛可提前停；100为上限保证充分训练)
# - 若显存紧张：改batch_size=16，lr可同步降至3e-4
# 快速测试交互（默认不进入测试模式）
if [ -n "${QUICK_TEST:-}" ]; then
  epochs_val=1
else
  read -p "是否进行快速测试（仅1个epoch）？[y/N]：" ans
  if [[ "$ans" =~ ^[yY]([eE][sS])?$ ]]; then
    epochs_val=1
  else
    epochs_val=${EPOCHS:-50}
  fi
fi

# 可通过环境变量覆盖
MODEL=${MODEL:-""}
BATCH_SIZE=${BATCH_SIZE:-32}
EPOCHS=$epochs_val
INPUT_SIZE=${INPUT_SIZE:-224}
NUM_WORKERS=${NUM_WORKERS:-4}
PIN_MEMORY=${PIN_MEMORY:-true}
PERSISTENT_WORKERS=${PERSISTENT_WORKERS:-true}
PREFETCH_FACTOR=${PREFETCH_FACTOR:-2}
USE_AMP=${USE_AMP:-true}
LABEL_SMOOTHING=${LABEL_SMOOTHING:-0.0}
DEBUG_PRED_STATS=${DEBUG_PRED_STATS:-true}

# 交互式选择模型（若 MODEL 已显式设置则跳过）
if [ -z "$MODEL" ] && [ -t 0 ]; then
  echo "请选择模型："
  echo "  [1] convnextv2_atto"
  echo "  [2] mobilevit_xxs"
  echo "  [3] convnext_zepto_rms"
  echo "  [4] poolformerv2_s12"
  read -p "请输入选项编号（默认1）：" model_choice
  model_choice=${model_choice:-1}
  case "$model_choice" in
    1) MODEL="convnextv2_atto" ;;
    2) MODEL="mobilevit_xxs" ;;
    3) MODEL="convnext_zepto_rms" ;;
    4) MODEL="poolformerv2_s12" ;;
    *) echo "无效选项: $model_choice，使用默认模型 convnextv2_atto"; MODEL="convnextv2_atto" ;;
  esac
fi

# 兜底默认模型
MODEL=${MODEL:-"convnextv2_atto"}

# 统一使用序列模式；WINDOW_SIZE=1 等价于单帧训练
SEQUENCE_MODE=true
SEQUENCE_ROOT=${SEQUENCE_ROOT:-"rheed_images_png_0226"}
WINDOW_SIZE=${WINDOW_SIZE:-1}
WINDOW_STRIDE=${WINDOW_STRIDE:-1}
LR=${LR:-3e-5}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.05}
AUTO_CLASS_WEIGHTS=${AUTO_CLASS_WEIGHTS:-true}
BALANCED_SAMPLER=${BALANCED_SAMPLER:-false}
PRETRAINED=${PRETRAINED:-true}

# 序列数据路径
SPLIT_RATIO=${SPLIT_RATIO:-"0.7,0.1,0.2"}
STRICT_TIME_SPLIT=${STRICT_TIME_SPLIT:-true}
SEQ_TRAIN=${SEQ_TRAIN:-"seq_001,seq_002,seq_003,seq_004,seq_005,seq_006,seq_007"}
SEQ_VAL=${SEQ_VAL:-"seq_008"}
SEQ_TEST=${SEQ_TEST:-"seq_009,seq_010"}

# 预训练权重选择
# PRETRAIN_SOURCE: imagenet | ssl | none
# 默认使用 ImageNet 预训练；也可在运行时交互选择 SSL-80/120/150 初始化
PRETRAIN_SOURCE=${PRETRAIN_SOURCE:-"imagenet"}
PRETRAINED_WEIGHTS=${PRETRAINED_WEIGHTS:-""}

# 你的 SSL 预训练权重（按轮数）
SSL80_OUTPUT_DIR=${SSL80_OUTPUT_DIR:-"outputs/pretrain_ssl_20260223_132829_80"}
SSL120_OUTPUT_DIR=${SSL120_OUTPUT_DIR:-"outputs/pretrain_ssl_20260223_150207_120"}
SSL150_OUTPUT_DIR=${SSL150_OUTPUT_DIR:-"outputs/pretrain_ssl_20260223_205736_150"}

SSL80_WEIGHTS=${SSL80_WEIGHTS:-"${SSL80_OUTPUT_DIR}/checkpoint_last.pth"}
SSL120_WEIGHTS=${SSL120_WEIGHTS:-"${SSL120_OUTPUT_DIR}/checkpoint_last.pth"}
SSL150_WEIGHTS=${SSL150_WEIGHTS:-"${SSL150_OUTPUT_DIR}/checkpoint_last.pth"}

EXTRA_ARGS=()

# 若用户已显式设置 PRETRAIN_SOURCE/PRETRAINED_WEIGHTS，则不再询问（方便高级用法/脚本化）
if [ -z "$PRETRAINED_WEIGHTS" ] && [ "$PRETRAIN_SOURCE" = "imagenet" ] && [ -t 0 ]; then
  echo "请选择微调初始化权重来源："
  echo "  [1] ImageNet 预训练 (默认)"
  echo "  [2] SSL 预训练 (80 epochs)"
  echo "  [3] SSL 预训练 (120 epochs)"
  echo "  [4] SSL 预训练 (150 epochs)"
  echo "  [5] 不使用预训练 (none)"
  read -p "请输入选项[1-5]（默认1）：" init_choice
  init_choice=${init_choice:-1}

  case "$init_choice" in
    1)
      PRETRAIN_SOURCE="imagenet"
      ;;
    2)
      PRETRAIN_SOURCE="ssl"
      PRETRAINED_WEIGHTS="$SSL80_WEIGHTS"
      ;;
    3)
      PRETRAIN_SOURCE="ssl"
      PRETRAINED_WEIGHTS="$SSL120_WEIGHTS"
      ;;
    4)
      PRETRAIN_SOURCE="ssl"
      PRETRAINED_WEIGHTS="$SSL150_WEIGHTS"
      ;;
    5)
      PRETRAIN_SOURCE="none"
      ;;
    *)
      echo "无效选项: $init_choice（可选: 1-5）"
      exit 1
      ;;
  esac
fi

# 兼容：如果提供了 PRETRAINED_WEIGHTS 但没改 PRETRAIN_SOURCE，则自动切到 ssl
if [ -n "$PRETRAINED_WEIGHTS" ] && [ "$PRETRAIN_SOURCE" = "imagenet" ]; then
  PRETRAIN_SOURCE="ssl"
  echo "检测到PRETRAINED_WEIGHTS，自动切换为 PRETRAIN_SOURCE=ssl"
fi

case "$PRETRAIN_SOURCE" in
  imagenet)
    PRETRAINED=true
    ;;
  ssl)
    if [ -z "$PRETRAINED_WEIGHTS" ]; then
      echo "PRETRAIN_SOURCE=ssl 需要提供 PRETRAINED_WEIGHTS"
      exit 1
    fi
    PRETRAINED=true
    EXTRA_ARGS+=(--pretrained_weights "$PRETRAINED_WEIGHTS")
    ;;
  none)
    PRETRAINED=false
    ;;
  *)
    echo "未知 PRETRAIN_SOURCE: $PRETRAIN_SOURCE (可选: imagenet | ssl | none)"
    exit 1
    ;;
esac

echo "SEQUENCE_MODE=true WINDOW_SIZE=$WINDOW_SIZE STRIDE=$WINDOW_STRIDE LR=$LR WD=$WEIGHT_DECAY"

PRETRAIN_TAG="$EPOCHS"
case "$PRETRAIN_SOURCE" in
  imagenet)
    PRETRAIN_TAG="imagenet"
    ;;
  none)
    PRETRAIN_TAG="none"
    ;;
  ssl)
    if [ "$PRETRAINED_WEIGHTS" = "$SSL80_WEIGHTS" ]; then
      PRETRAIN_TAG="80"
    elif [ "$PRETRAINED_WEIGHTS" = "$SSL120_WEIGHTS" ]; then
      PRETRAIN_TAG="120"
    elif [ "$PRETRAINED_WEIGHTS" = "$SSL150_WEIGHTS" ]; then
      PRETRAIN_TAG="150"
    fi
    ;;
  *)
    PRETRAIN_TAG="$EPOCHS"
    ;;
esac

OUTPUT_DIR="outputs/finetune_timm_$(date +%Y%m%d_%H%M%S)_${MODEL}_${PRETRAIN_TAG}"

ARGS=(
  --model "$MODEL"
  --sequence_mode "$SEQUENCE_MODE"
  --num_classes 2
  --input_size "$INPUT_SIZE"
  --batch_size "$BATCH_SIZE"
  --epochs "$EPOCHS"
  --lr "$LR"
  --weight_decay "$WEIGHT_DECAY"
  --num_workers "$NUM_WORKERS"
  --pin_memory "$PIN_MEMORY"
  --persistent_workers "$PERSISTENT_WORKERS"
  --prefetch_factor "$PREFETCH_FACTOR"
  --device cuda
  --use_amp "$USE_AMP"
  --label_smoothing "$LABEL_SMOOTHING"
  --auto_class_weights "$AUTO_CLASS_WEIGHTS"
  --balanced_sampler "$BALANCED_SAMPLER"
  --debug_pred_stats "$DEBUG_PRED_STATS"
  --pretrained "$PRETRAINED"
  --output_dir "$OUTPUT_DIR"
  --save_best_metric acc1
)

ARGS+=(
  --sequence_root "$SEQUENCE_ROOT"
  --window_size "$WINDOW_SIZE"
  --window_stride "$WINDOW_STRIDE"
  --split_ratio "$SPLIT_RATIO"
  --strict_time_split "$STRICT_TIME_SPLIT"
  --seq_train "$SEQ_TRAIN"
  --seq_val "$SEQ_VAL"
  --seq_test "$SEQ_TEST"
)


python -m src.train_timm "${ARGS[@]}" "${EXTRA_ARGS[@]}"

echo "训练完成, 输出: $OUTPUT_DIR"
