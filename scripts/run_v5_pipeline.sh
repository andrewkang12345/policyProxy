#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

log() {
  local msg="$1"
  printf '\n[%(%Y-%m-%d %H:%M:%S)T] %s\n' -1 "$msg"
}

detect_device() {
  if command -v nvidia-smi >/dev/null 2>&1; then
    local gpu_count
    gpu_count=$(nvidia-smi -L 2>/dev/null | grep -c 'GPU')
    if [[ "$gpu_count" -gt 0 ]]; then
      log "Detected ${gpu_count} GPU(s); using CUDA."
      DEVICE="cuda"
      return
    fi
  fi
  log "No GPUs detected; falling back to CPU."
  DEVICE="cpu"
}

ensure_dirs() {
  mkdir -p data runs reports/task_b_results/b1 reports/task_b_results/b2
}

run_cmd() {
  local desc="$1"
  shift
  log "$desc"
  "$@"
}

main() {
  detect_device
  ensure_dirs

  local DATA_ROOT="data/v5_three_shifts"

  run_cmd "Generating v5 three-shift dataset" \
    python make_data_v5.py --config configs/base_v5.yaml --out "$DATA_ROOT" --device "$DEVICE"

  run_cmd "Training CVAE-PID baseline" \
    python baselines/state_cond/train_cvae_pid.py --data_root "$DATA_ROOT" --save_dir runs/cvae_pid_v5 --device "$DEVICE" --epochs 100

  run_cmd "Training CVAE-REP baseline" \
    python baselines/state_cond/train_cvae.py --data_root "$DATA_ROOT" --save_dir runs/cvae_repr_v5 --device "$DEVICE" --epochs 100 --deterministic_latent

  run_cmd "Training CVAE baseline (free latent)" \
    python baselines/state_cond/train_cvae.py --data_root "$DATA_ROOT" --save_dir runs/cvae_free_v5 --device "$DEVICE" --epochs 100 --no_global_latent

  run_cmd "Training GRU baseline" \
    python baselines/state_cond/train_gru.py --data_root "$DATA_ROOT" --save_dir runs/gru_v5 --device "$DEVICE" --epochs 100

  run_cmd "Training Transformer CVAE baseline" \
    python baselines/state_cond/train_cvae_transformer.py --data_root "$DATA_ROOT" --save_dir runs/trans_cvae_v5 --device "$DEVICE" --epochs 100

  run_cmd "Training Regularized CVAE baseline" \
    python baselines/state_cond/train_cvae_reg.py --data_root "$DATA_ROOT" --save_dir runs/cvae_reg_v5 --device "$DEVICE" --epochs 100

  run_cmd "Training Regularized CVAE baseline (free latent)" \
    python baselines/state_cond/train_cvae_reg.py --data_root "$DATA_ROOT" --save_dir runs/cvae_reg_free_v5 --device "$DEVICE" --epochs 100 --no_global_latent

  run_cmd "Training Transformer state-only baseline" \
    python baselines/state_cond/train_transformer_state.py --data_root "$DATA_ROOT" --save_dir runs/transformer_state_v5 --device "$DEVICE" --epochs 100

  declare -A MODEL_PATHS=(
    [cvae_pid]=runs/cvae_pid_v5/model_best.pt
    [cvae_repr]=runs/cvae_repr_v5/model_best.pt
    [cvae_free]=runs/cvae_free_v5/model_best.pt
    [gru]=runs/gru_v5/model_best.pt
    [trans_cvae]=runs/trans_cvae_v5/model_best.pt
    [cvae_reg]=runs/cvae_reg_v5/model_best.pt
    [cvae_reg_free]=runs/cvae_reg_free_v5/model_best.pt
    [transformer_state]=runs/transformer_state_v5/model_best.pt
  )

  for model in "${!MODEL_PATHS[@]}"; do
    local ckpt="${MODEL_PATHS[$model]}"
    if [[ ! -f "$ckpt" ]]; then
      log "Warning: checkpoint not found for $model at $ckpt; skipping rollout."
      continue
    fi
    local save_dir="runs/${model}_v5"
    mkdir -p "$save_dir"
    local rollout_json="$save_dir/rollout_all.json"
    run_cmd "Running rollout for $model" \
      python eval/rollout.py --data_root "$DATA_ROOT" --model "$ckpt" --device "$DEVICE" --save_json "$rollout_json"
  done

  if [[ -f runs/cvae_pid_v5/model_best.pt ]]; then
    run_cmd "Running diagnostics for CVAE-PID" \
      python eval/diagnostics.py --data_dir "$DATA_ROOT" --model_dir runs/cvae_pid_v5 --output_file runs/cvae_pid_v5/diagnostics.json
  fi

  if [[ -f runs/cvae_repr_v5/model_best.pt ]]; then
    run_cmd "Running diagnostics for CVAE-REP" \
      python eval/diagnostics.py --data_dir "$DATA_ROOT" --model_dir runs/cvae_repr_v5 --output_file runs/cvae_repr_v5/diagnostics.json
  fi

  run_cmd "Policy classification task" \
    python eval/policy_classification.py --data_root "$DATA_ROOT" --save_dir reports/task_b_results/b1 --epochs 10 --task demo_proper --device "$DEVICE"

  run_cmd "Policy changepoint detection task" \
    python eval/policy_changepoint_detection.py --data_root "$DATA_ROOT" --save_dir reports/task_b_results/b2 --epochs 10 --task demo_proper --device "$DEVICE"

  log "Pipeline complete."
}

main "$@"
