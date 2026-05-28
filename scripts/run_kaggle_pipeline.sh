#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

MODE="${MODE:-all}"
SEEDS="${SEEDS:-0 1 2 3 4}"
SEEDS_GPU0="${SEEDS_GPU0:-0 2 4}"
SEEDS_GPU1="${SEEDS_GPU1:-1 3}"
OUTPUT_ROOT="${OUTPUT_ROOT:-data/imagenet_resnet50_selected}"
CLASS_INDEX_JSON="${CLASS_INDEX_JSON:-data/imagenet_class_index${GPU_ID:+_gpu${GPU_ID}}.json}"
SWEEP_DIR="${SWEEP_DIR:-results/sweeps}"
RESULTS_DIR="${RESULTS_DIR:-results}"
MODELS="${MODELS:-resnet50 vgg16 densenet121}"
TAU_GRID="${TAU_GRID:-0.001 0.005 0.01 0.05 0.1 1.0}"
STEPS_GRID="${STEPS_GRID:-16 32 64 128}"
CANDIDATE_COUNT="${CANDIDATE_COUNT:-5000}"
SELECT_COUNT="${SELECT_COUNT:-200}"
SWEEP_NUM_IMAGES="${SWEEP_NUM_IMAGES:-20}"
FULL_NUM_IMAGES="${FULL_NUM_IMAGES:-200}"
ITERS="${ITERS:-300}"
LR="${LR:-0.05}"
INSDEL_STEPS="${INSDEL_STEPS:-50}"
BATCH_SIZE="${BATCH_SIZE:-64}"
SKIP_ERRORS="${SKIP_ERRORS:-1}"
IMAGENET_ROOT="${IMAGENET_ROOT:-}"
DEVICE="${DEVICE:-cuda}"

if [[ -n "${GPU_ID:-}" ]]; then
  export CUDA_VISIBLE_DEVICES="$GPU_ID"
fi

if [[ "${RUN_TWO_GPU:-0}" == "1" ]]; then
  echo "Launching two process-level workers"
  if [[ "$MODE" == "all" ]]; then
    for stage in prepare sweep eval; do
      MODE="$stage" SEEDS="$SEEDS_GPU0" GPU_ID=0 RUN_TWO_GPU=0 bash "$0" &
      pid0=$!
      MODE="$stage" SEEDS="$SEEDS_GPU1" GPU_ID=1 RUN_TWO_GPU=0 bash "$0" &
      pid1=$!
      wait "$pid0"
      wait "$pid1"
    done
    MODE=summarize RUN_TWO_GPU=0 bash "$0"
  elif [[ "$MODE" == "summarize" ]]; then
    MODE=summarize RUN_TWO_GPU=0 bash "$0"
  else
    MODE="$MODE" SEEDS="$SEEDS_GPU0" GPU_ID=0 RUN_TWO_GPU=0 bash "$0" &
    pid0=$!
    MODE="$MODE" SEEDS="$SEEDS_GPU1" GPU_ID=1 RUN_TWO_GPU=0 bash "$0" &
    pid1=$!
    wait "$pid0"
    wait "$pid1"
  fi
  exit 0
fi

if [[ -f /opt/conda/etc/profile.d/conda.sh ]]; then
  source /opt/conda/etc/profile.d/conda.sh
elif [[ -f /mnt/data/miniconda3/etc/profile.d/conda.sh ]]; then
  source /mnt/data/miniconda3/etc/profile.d/conda.sh
fi
if command -v conda >/dev/null 2>&1 && conda env list | awk '{print $1}' | grep -qx ai_stable; then
  conda activate ai_stable
fi

mkdir -p "$RESULTS_DIR" "$SWEEP_DIR"

maybe_skip_errors=()
if [[ "$SKIP_ERRORS" == "1" ]]; then
  maybe_skip_errors=(--skip-errors)
fi

prepare_dataset() {
  local imagenet_args=()
  if [[ -n "$IMAGENET_ROOT" ]]; then
    imagenet_args=(--imagenet-root "$IMAGENET_ROOT")
  fi
  echo "Preparing dataset for seeds: $SEEDS"
  python prepare_imagenet_resnet_dataset.py \
    --seeds $SEEDS \
    --candidate-count "$CANDIDATE_COUNT" \
    --select-count "$SELECT_COUNT" \
    --output-root "$OUTPUT_ROOT" \
    --class-index-json "$CLASS_INDEX_JSON" \
    --batch-size "$BATCH_SIZE" \
    --device "$DEVICE" \
    --resume \
    "${imagenet_args[@]}"
}

run_sweeps() {
  for seed in $SEEDS; do
    local selection_csv="$OUTPUT_ROOT/seed_${seed}/selected.csv"
    echo "Sweeping seed $seed from $selection_csv"
    python sweep_mu_config.py \
      --selection-csv "$selection_csv" \
      --seed "$seed" \
      --num-images "$SWEEP_NUM_IMAGES" \
      --tau-grid $TAU_GRID \
      --steps-grid $STEPS_GRID \
      --iters "$ITERS" \
      --lr "$LR" \
      --insdel-steps "$INSDEL_STEPS" \
      --output-dir "$SWEEP_DIR" \
      --device "$DEVICE" \
      "${maybe_skip_errors[@]}"
  done
}

best_value() {
  local seed="$1"
  local key="$2"
  python - "$SWEEP_DIR/best_config_seed${seed}.json" "$key" <<'PY'
import json, sys
path, key = sys.argv[1], sys.argv[2]
print(json.load(open(path))[key])
PY
}

run_full_eval() {
  for seed in $SEEDS; do
    local selection_csv="$OUTPUT_ROOT/seed_${seed}/selected.csv"
    local best_tau best_steps tau_tag
    best_tau="$(best_value "$seed" tau)"
    best_steps="$(best_value "$seed" steps)"
    tau_tag="$(python - "$best_tau" <<'PY'
import sys
print(sys.argv[1].replace(".", "p").replace("-", "m"))
PY
)"
    for model in $MODELS; do
      local out="$RESULTS_DIR/full_eval_seed${seed}_${model}_N${best_steps}_tau${tau_tag}.json"
      echo "Full eval seed=$seed model=$model N=$best_steps tau=$best_tau"
      python batch_eval.py \
        --selected-csv "$selection_csv" \
        --num-images "$FULL_NUM_IMAGES" \
        --model-name "$model" \
        --steps "$best_steps" \
        --tau "$best_tau" \
        --iters "$ITERS" \
        --lr "$LR" \
        --insdel \
        --insdel-steps "$INSDEL_STEPS" \
        --seed "$seed" \
        --device "$DEVICE" \
        --output-json "$out" \
        "${maybe_skip_errors[@]}"
    done
  done
}

summarize_results() {
  python summarize_full_eval.py \
    --input-glob "$RESULTS_DIR/full_eval_seed*_*.json" \
    --output-csv "$RESULTS_DIR/full_eval_summary.csv" \
    --output-md "$RESULTS_DIR/full_eval_summary.md"
}

case "$MODE" in
  prepare) prepare_dataset ;;
  sweep) run_sweeps ;;
  eval) run_full_eval ;;
  summarize) summarize_results ;;
  all)
    prepare_dataset
    run_sweeps
    run_full_eval
    summarize_results
    ;;
  *)
    echo "Unknown MODE=$MODE. Use prepare, sweep, eval, summarize, or all." >&2
    exit 2
    ;;
esac
