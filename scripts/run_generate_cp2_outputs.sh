#!/bin/bash
# 生成第二章理想化输出数据（支持参数）

set -e

OUT_DIR=${1:-"/home/utopianyouth/rheed_cls_convnextv2/outputs/finetune_timm_20260212_175658"}

python /home/utopianyouth/rheed_cls_convnextv2/src/generate_cp2_outputs.py --output_dir "$OUT_DIR"
