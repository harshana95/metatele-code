#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export MPLBACKEND=Agg
export QT_QPA_PLATFORM=offscreen

# Symlink only the highest _{x}k variant per image (e.g. out_0001_0870k over out_0001_0835k).
link_highest_k() {
  local exp_dir="$1"
  local tag="$2"
  local dest_subdir="$3"
  declare -A best_k=()
  declare -A best_path=()

  shopt -s nullglob
  for f in "$exp_dir"/${tag}_*.jpg; do
    local base group k
    base=$(basename "$f")
    if [[ "$base" =~ ^(.+)_([0-9]+)k\.jpg$ ]]; then
      group="${BASH_REMATCH[1]}"
      k="${BASH_REMATCH[2]}"
      if [[ -z "${best_k[$group]+x}" || 10#$k -gt 10#${best_k[$group]} ]]; then
        best_k[$group]="$k"
        best_path[$group]="$f"
      fi
    else
      ln -sf "../$base" "$exp_dir/$dest_subdir/$base"
    fi
  done
  shopt -u nullglob

  for group in "${!best_path[@]}"; do
    local f base
    f="${best_path[$group]}"
    base=$(basename "$f")
    ln -sf "../$base" "$exp_dir/$dest_subdir/$base"
  done
}

for EXP_DIR in experiments/infer/*/images; do
  rm -rf "$EXP_DIR/pred" "$EXP_DIR/gt"
  mkdir -p "$EXP_DIR/pred" "$EXP_DIR/gt"
  link_highest_k "$EXP_DIR" "out" "pred"
  link_highest_k "$EXP_DIR" "gt" "gt"
done

METHODS=(
  # null_4
  # dape_4
  # gemma_4
  # qwen_4
  # florence_4
  nodc_4
  sharp_4
  # nodc
  # sharp
  # vsd
  # novsd
)

for NAME in "${METHODS[@]}"; do
  DIR="experiments/infer/MetaZoom_infer_${NAME}/images"
  python analysis/calculate_metrics.py \
    --pred_folder "$DIR/pred" \
    --gt_folder "$DIR/gt" \
    --image_size 512 \
    --device cuda \
    --name "$NAME" \
    --save_pairs --save_outputs \
    --results_file "./results.txt"
done