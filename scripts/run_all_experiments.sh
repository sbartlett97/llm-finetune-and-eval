#!/usr/bin/env bash
set -euo pipefail

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

EVAL_CONFIG="${EVAL_CONFIG:-configs/eval_config.yaml}"
DATA_CONFIG="${DATA_CONFIG:-configs/data_config.yaml}"
DUAL_GPU="${DUAL_GPU:-0}"
STATE_FILE="${STATE_FILE:-.experiment_state}"

if [[ "${1:-}" == "--reset" ]]; then
    log "Resetting experiment state: $STATE_FILE"
    rm -f "$STATE_FILE"
fi

is_done()  { grep -qxF "$1" "$STATE_FILE" 2>/dev/null; }
mark_done() { echo "$1" >> "$STATE_FILE"; }

run_experiment() {
    local config="$1"
    local run_id="$2"

    if is_done "train_${run_id}"; then
        log "Skipping training (already done): $run_id"
    else
        log "Starting training: $run_id"
        python scripts/train.py --config "$config" --data-config "$DATA_CONFIG"
        mark_done "train_${run_id}"
        log "Training complete: $run_id"
    fi

    if is_done "eval_${run_id}"; then
        log "Skipping eval (already done): $run_id"
    else
        log "Starting eval: $run_id"
        python scripts/eval.py --run-id "$run_id" --eval-config "$EVAL_CONFIG" --data-config "$DATA_CONFIG"
        mark_done "eval_${run_id}"
        log "Eval complete: $run_id"
    fi
}

run_experiment_on_gpu() {
    local gpu_id="$1" config="$2" run_id="$3"

    if is_done "train_${run_id}"; then
        log "Skipping training (already done): $run_id"
    else
        CUDA_VISIBLE_DEVICES="$gpu_id" python scripts/train.py --config "$config" --data-config "$DATA_CONFIG"
        mark_done "train_${run_id}"
    fi

    if is_done "eval_${run_id}"; then
        log "Skipping eval (already done): $run_id"
    else
        CUDA_VISIBLE_DEVICES="$gpu_id" python scripts/eval.py --run-id "$run_id" --eval-config "$EVAL_CONFIG" --data-config "$DATA_CONFIG"
        mark_done "eval_${run_id}"
    fi
}

log "=== Baseline (no fine-tuning) ==="
if is_done "eval_baseline"; then
    log "Skipping baseline eval (already done)"
else
    python scripts/eval.py --model-path unsloth/SmolLM3-3B-128K --run-name run_baseline \
        --eval-config "$EVAL_CONFIG" --data-config "$DATA_CONFIG"
    mark_done "eval_baseline"
fi

if [[ "$DUAL_GPU" == "1" ]]; then
    log "=== Dual-GPU mode: GPU 0 -> r8, r16 | GPU 1 -> r32, r16_2ep ==="
    (
        run_experiment_on_gpu 0 configs/training/lora_r8.yaml run_lora_r8
        run_experiment_on_gpu 0 configs/training/lora_r16.yaml run_lora_r16
    ) &
    pid0=$!
    (
        run_experiment_on_gpu 1 configs/training/lora_r32.yaml run_lora_r32
        run_experiment_on_gpu 1 configs/training/lora_r16_2ep.yaml run_lora_r16_2ep
    ) &
    pid1=$!

    wait "$pid0" || { log "ERROR: GPU 0 subprocess failed"; exit 1; }
    wait "$pid1" || { log "ERROR: GPU 1 subprocess failed"; exit 1; }
else
    log "=== LoRA r=8 ==="
    run_experiment configs/training/lora_r8.yaml run_lora_r8

    log "=== LoRA r=16 (default) ==="
    run_experiment configs/training/lora_r16.yaml run_lora_r16

    log "=== LoRA r=32 ==="
    run_experiment configs/training/lora_r32.yaml run_lora_r32

    log "=== LoRA r=16, 2 epochs ==="
    run_experiment configs/training/lora_r16_2ep.yaml run_lora_r16_2ep
fi

log "=== All experiments complete ==="
