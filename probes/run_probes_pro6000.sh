#!/bin/bash
# Run Qwen3.5 knowledge probes on RTX Pro 6000 Blackwell.
#
# Mirrors run_pro6000.sh's serve-then-query pattern: launches SGLang on one
# GPU, waits for /health, runs probes/run_probes.py against localhost:8000,
# then shuts down the server on exit.
#
# Only needs 1 GPU (probes just call the HTTP API — no local GPU tensors).
#
# Usage:
#   bash probes/run_probes_pro6000.sh              # single-shot probes
#   RETRIES=2 bash probes/run_probes_pro6000.sh    # allow 2 retries on BAD_JSON

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCAFFOLD_DIR="${SCAFFOLD_DIR:-$(dirname "$SCRIPT_DIR")}"
VENV_DIR="${VENV_DIR:-$SCAFFOLD_DIR/.venv_pro6000}"

MODEL="${MODEL:-Qwen/Qwen3.5-9B}"
VLLM_PORT="${VLLM_PORT:-8000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-30000}"
RETRIES="${RETRIES:-0}"
TRAIN_PY="${TRAIN_PY:-$SCAFFOLD_DIR/autoresearch/train.py}"

cd "$SCAFFOLD_DIR"

# ── Activate the same venv the frozen pipeline uses ──────────
if [ ! -d "$VENV_DIR" ]; then
    echo "ERROR: venv not found at $VENV_DIR. Run run_pro6000.sh first to create it."
    exit 1
fi
source "$VENV_DIR/bin/activate"

# ── Pick a free GPU ──────────────────────────────────────────
mapfile -t FREE_GPUS < <(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
    | sort -t',' -k2 -n | awk -F',' '{gsub(/ /,""); print $1}')
if [ "${#FREE_GPUS[@]}" -lt 1 ]; then
    echo "ERROR: no free GPU found"
    exit 1
fi
SGLANG_GPU="${FREE_GPUS[0]}"

echo "Node:    $(hostname)"
echo "Model:   $MODEL"
echo "GPU:     $SGLANG_GPU (SGLang)"
echo "Port:    $VLLM_PORT"
echo "Retries: $RETRIES"
echo "---"

# ── Start SGLang on the chosen GPU (same flags as run_pro6000.sh) ──
echo "Starting SGLang server..."
CUDA_VISIBLE_DEVICES="$SGLANG_GPU" python -m sglang.launch_server \
    --model-path "$MODEL" \
    --port "$VLLM_PORT" \
    --tp-size 1 \
    --mem-fraction-static 0.5 \
    --context-length "$MAX_MODEL_LEN" \
    --attention-backend triton \
    --reasoning-parser qwen3 \
    &> "$SCRIPT_DIR/sglang_probes.log" &
SERVER_PID=$!

cleanup() {
    echo "Stopping SGLang server (PID $SERVER_PID)..."
    kill $SERVER_PID 2>/dev/null || true
    wait $SERVER_PID 2>/dev/null || true
}
trap cleanup EXIT

echo "Waiting for server..."
for i in $(seq 1 360); do
    if curl -s "http://localhost:$VLLM_PORT/health" > /dev/null 2>&1; then
        echo "Server ready after ${i}s"
        break
    fi
    if ! kill -0 $SERVER_PID 2>/dev/null; then
        echo "ERROR: Server died. Check $SCRIPT_DIR/sglang_probes.log"
        tail -30 "$SCRIPT_DIR/sglang_probes.log"
        exit 1
    fi
    sleep 1
done

if ! curl -s "http://localhost:$VLLM_PORT/health" > /dev/null 2>&1; then
    echo "ERROR: Server failed to start within 360s"
    tail -30 "$SCRIPT_DIR/sglang_probes.log"
    exit 1
fi

# ── Run probes ───────────────────────────────────────────────
echo ""
echo "Running knowledge probes..."
python "$SCRIPT_DIR/run_probes.py" \
    --base-url "http://localhost:$VLLM_PORT/v1" \
    --model "$MODEL" \
    --train-py "$TRAIN_PY" \
    --retries "$RETRIES"

echo ""
echo "Outputs: $SCRIPT_DIR/outputs/"
echo "Summary: $SCRIPT_DIR/outputs/summary.json"
