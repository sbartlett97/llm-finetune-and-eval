#!/usr/bin/env bash
set -euo pipefail

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

EVAL_CONFIG="${EVAL_CONFIG:-configs/eval_config.yaml}"
DATA_CONFIG="${DATA_CONFIG:-configs/data_config.yaml}"

run_experiment() {
    local config="$1"
    local run_id="$2"
    log "Starting training: $run_id"
    python scripts/train.py --config "$config" --data-config "$DATA_CONFIG"
    log "Training complete: $run_id"

    log "Starting eval: $run_id"
    python scripts/eval.py --run-id "$run_id" --eval-config "$EVAL_CONFIG" --data-config "$DATA_CONFIG"
    log "Eval complete: $run_id"
}

log "=== Baseline (no fine-tuning) ==="
python scripts/eval.py --model-path unsloth/SmolLM3-3B-128K --run-name run_baseline \
    --eval-config "$EVAL_CONFIG" --data-config "$DATA_CONFIG"

log "=== LoRA r=8 ==="
run_experiment configs/training/lora_r8.yaml run_lora_r8

log "=== LoRA r=16 (default) ==="
run_experiment configs/training/lora_r16.yaml run_lora_r16

log "=== LoRA r=32 ==="
run_experiment configs/training/lora_r32.yaml run_lora_r32

log "=== LoRA r=16, 2 epochs ==="
run_experiment configs/training/lora_r16_2ep.yaml run_lora_r16_2ep

log "=== All experiments complete ==="
