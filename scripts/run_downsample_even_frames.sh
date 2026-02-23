#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

INPUT_ROOT="$PROJECT_ROOT/rheed_images_png"
OUTPUT_2D="$PROJECT_ROOT/rheed_images_png/4_2d"
OUTPUT_3D="$PROJECT_ROOT/rheed_images_png/4_3d"

process_class() {
  local class_name="$1"
  local input_dir="$INPUT_ROOT/$class_name"
  local output_dir="$2"

  if [ ! -d "$input_dir" ]; then
    echo "目录不存在: $input_dir"
    exit 1
  fi

  mkdir -p "$output_dir"

  mapfile -t seq_dirs < <(find "$input_dir" -maxdepth 1 -type d -name "seq_*" | sort)

  if [ ${#seq_dirs[@]} -eq 0 ]; then
    echo "${class_name}: 未发现seq子目录，按平铺目录处理"
    _process_dir "$input_dir" "$output_dir" "$class_name"
  else
    for seq_dir in "${seq_dirs[@]}"; do
      seq_name=$(basename "$seq_dir")
      out_seq_dir="$output_dir/$seq_name"
      _process_dir "$seq_dir" "$out_seq_dir" "${class_name}/${seq_name}"
    done
  fi
}

_process_dir() {
  local src_dir="$1"
  local dst_dir="$2"
  local tag="$3"

  mkdir -p "$dst_dir"

  local count=0
  local kept=0
  while IFS= read -r -d '' img; do
    fname=$(basename "$img")
    if [[ "$fname" =~ ([0-9]+)\.png$ ]]; then
      frame_num=${BASH_REMATCH[1]}
      frame_num=$((10#$frame_num))
      count=$((count + 1))
      if [ $((frame_num % 4)) -eq 0 ]; then
        cp "$img" "$dst_dir/"
        kept=$((kept + 1))
      fi
    fi
  done < <(find "$src_dir" -maxdepth 1 -type f -name "*.png" -print0 | sort -z)

  echo "$tag: copied $kept frames from $count to $dst_dir"
}

echo "=== 降采样到7.5fps（每4帧保留1帧） ==="
process_class "2d" "$OUTPUT_2D"
process_class "3d" "$OUTPUT_3D"

echo "完成：输出目录"
echo "  $OUTPUT_2D"
echo "  $OUTPUT_3D"
