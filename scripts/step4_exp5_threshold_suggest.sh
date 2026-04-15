#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-/home/projectx/miniconda/envs/sam3/bin/python}"
PYTHONPATH_DIR="${PYTHONPATH_DIR:-$PROJECT_ROOT/sam3}"

CONFIG="${CONFIG:-$PROJECT_ROOT/experiments/exp5/config_resolved.yaml}"
FINETUNE_CKPT="${FINETUNE_CKPT:-$PROJECT_ROOT/experiments/exp5/checkpoints/best_train_loss.pt}"
ISOLATED_VAL_ROOT="${ISOLATED_VAL_ROOT:-$PROJECT_ROOT/dataset/Golden_set/val_full50}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/experiments/exp5/visualization_label_threshold_custom}"

NUM_IMAGES="${NUM_IMAGES:-50}"
PRED_THRESHOLD="${PRED_THRESHOLD:-0.7}"
THRESHOLD_CANDIDATES="${THRESHOLD_CANDIDATES:-0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95}"
COMBO_TOP_N_PER_LABEL="${COMBO_TOP_N_PER_LABEL:-5}"
COMBO_TOP_K="${COMBO_TOP_K:-10}"

# CSV format, e.g. "1:0.55,2:0.8,3:0.75,4:0.8,5:0.75"
LABEL_THRESHOLD_OVERRIDES_CSV="${LABEL_THRESHOLD_OVERRIDES_CSV:-1:0.55,2:0.8,3:0.7,4:0.75,5:0.6}"

IFS=',' read -r -a label_overrides <<< "$LABEL_THRESHOLD_OVERRIDES_CSV"
label_args=()
for item in "${label_overrides[@]}"; do
  [[ -z "$item" ]] && continue
  label_args+=("--label-threshold" "$item")
done

cmd=(
  "$PYTHON_BIN"
  "$PROJECT_ROOT/scripts/step4_evaluation.py"
  --config "$CONFIG"
  --finetune-ckpt "$FINETUNE_CKPT"
  --isolated-val-root "$ISOLATED_VAL_ROOT"
  --num-images "$NUM_IMAGES"
  --pred-threshold "$PRED_THRESHOLD"
  --suggest-label-thresholds
  --threshold-candidates "$THRESHOLD_CANDIDATES"
  --combo-top-n-per-label "$COMBO_TOP_N_PER_LABEL"
  --combo-top-k "$COMBO_TOP_K"
  --output-dir "$OUTPUT_DIR"
)
cmd+=("${label_args[@]}")
cmd+=("$@")

echo "[INFO] Running threshold suggestion for exp5"
echo "[INFO] Output dir: $OUTPUT_DIR"

env PYTHONPATH="$PYTHONPATH_DIR" "${cmd[@]}"
