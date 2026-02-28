#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

SRC_ROOT="$PROJECT_ROOT/rheed_images"
DST_ROOT="$PROJECT_ROOT/rheed_images_png"

# 说明：
# - 将 rheed_images 下所有序列的 *.tif/*.tiff 转为 *.png
# - 输出到同级 rheed_images_png，并严格保持目录结构与文件名（仅扩展名变为 .png）
# - 可重复运行：默认当某个序列目录下 png 数量已达到 tiff 数量时跳过
# - 如需强制重转：FORCE=1 ./scripts/run_convert_single_tiff_to_png.sh
FORCE=${FORCE:-0}

cd "$PROJECT_ROOT"

if [ ! -d "$SRC_ROOT" ]; then
  echo "未找到输入目录: $SRC_ROOT"
  exit 1
fi

mkdir -p "$DST_ROOT"

echo "输入目录: $SRC_ROOT"
echo "输出目录: $DST_ROOT"

total_dirs=0
converted_dirs=0
skipped_dirs=0

for cls in 2d 3d; do
  cls_in="$SRC_ROOT/$cls"
  if [ ! -d "$cls_in" ]; then
    echo "[WARN] 未找到类别目录: $cls_in，跳过"
    continue
  fi

  # 遍历 seq_* 目录
  while IFS= read -r -d '' seq_dir; do
    seq_name="$(basename "$seq_dir")"
    out_dir="$DST_ROOT/$cls/$seq_name"
    mkdir -p "$out_dir"

    # 统计数量，用于断点续跑
    tiff_count=$(find "$seq_dir" -maxdepth 1 -type f \( -iname "*.tif" -o -iname "*.tiff" \) | wc -l)
    png_count=$(find "$out_dir" -maxdepth 1 -type f -iname "*.png" | wc -l)

    if [ "$tiff_count" -eq 0 ]; then
      echo "[WARN] 目录无TIFF文件，跳过: $seq_dir"
      continue
    fi

    total_dirs=$((total_dirs + 1))

    if [ "$FORCE" -ne 1 ] && [ "$png_count" -ge "$tiff_count" ]; then
      echo "[SKIP] $cls/$seq_name 已转换（png=$png_count, tiff=$tiff_count）"
      skipped_dirs=$((skipped_dirs + 1))
      continue
    fi

    echo "[CONVERT] $cls/$seq_name (tiff=$tiff_count -> $out_dir)"
    python -m tools.convert_single_tiff_to_png \
      --input_dir "$seq_dir" \
      --output_dir "$out_dir"

    converted_dirs=$((converted_dirs + 1))
  done < <(find "$cls_in" -mindepth 1 -maxdepth 1 -type d -name "seq_*" -print0 | sort -z)

done

echo "完成：total_seq_dirs=$total_dirs converted=$converted_dirs skipped=$skipped_dirs"
