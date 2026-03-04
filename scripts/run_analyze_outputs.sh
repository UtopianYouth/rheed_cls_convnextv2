#!/usr/bin/env bash
# 解析outputs下的预训练/微调日志并生成图表与统计表格
# 输出目录默认：outputs/analysis_report_YYYYmmdd_HHMMSS/

set -e
cd "$(dirname "$0")/.."

# 可选：指定输入与输出目录，例如：
# OUTPUTS_DIR=outputs_0226_27 OUT_DIR=outputs_0226_27/analysis_report_custom ./scripts/run_analyze_outputs.sh
OUTPUTS_DIR=${OUTPUTS_DIR:-"outputs_0227"}
OUT_DIR=${OUT_DIR:-""}

python -m tools.analyze_outputs --outputs_dir "$OUTPUTS_DIR" ${OUT_DIR:+--out_dir "$OUT_DIR"}

echo "[DONE] Analysis finished."