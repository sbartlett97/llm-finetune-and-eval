#!/usr/bin/env bash
set -euo pipefail

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

EVAL_CONFIG="${EVAL_CONFIG:-configs/eval_config.yaml}"
DATA_CONFIG="${DATA_CONFIG:-configs/data_config.yaml}"
DUAL_GPU="${DUAL_GPU:-0}"

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

run_experiment_on_gpu() {
    local gpu_id="$1" config="$2" run_id="$3"
    CUDA_VISIBLE_DEVICES="$gpu_id" python scripts/train.py --config "$config" --data-config "$DATA_CONFIG"
    CUDA_VISIBLE_DEVICES="$gpu_id" python scripts/eval.py --run-id "$run_id" --eval-config "$EVAL_CONFIG" --data-config "$DATA_CONFIG"
}

log "=== Baseline (no fine-tuning) ==="
python scripts/eval.py --model-path unsloth/SmolLM3-3B-128K --run-name run_baseline \
    --eval-config "$EVAL_CONFIG" --data-config "$DATA_CONFIG"

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
