#!/usr/bin/env bash
set -euo pipefail

# ---- Configuration ----
MODEL="${MODEL:-Qwen/Qwen3.5-9B}"
PORT="${PORT:-8000}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"

echo "Serving: $MODEL"
echo "Port:    $PORT"
echo ""

vllm serve "$MODEL" \
    --port "$PORT" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --max-model-len "$MAX_MODEL_LEN" \
    --dtype bfloat16 \
    --reasoning-parser qwen3
