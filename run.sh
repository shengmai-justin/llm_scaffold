#!/bin/bash
#SBATCH --job-name=autoresearch
#SBATCH --output=autoresearch_%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64gb
#SBATCH --time=04:00:00
#SBATCH --partition=hpg-b200
#SBATCH --gpus=2

# ── Configuration ─────────────────────────────────────────────
MODEL="Qwen/Qwen3.5-9B"
VLLM_PORT=8000
MAX_EXPERIMENTS=0
MAX_MODEL_LEN=30000
PROJ_DIR="/blue/buyuheng/li_an.ucsb/proj_yepeng"
CONDA_ENV="${PROJ_DIR}/envs/myenv"
SCAFFOLD_DIR="${PROJ_DIR}/llm_scaffold"
SOURCE_REPO="${SCAFFOLD_DIR}/autoresearch"
REPO_PATH="${SCAFFOLD_DIR}/autoresearch_frozen"
LOG_DIR="${SCAFFOLD_DIR}/frozen_log"

# ── Navigate to project dir ───────────────────────────────────
cd "$SCAFFOLD_DIR"

# ── Clone source repo if not present ─────────────────────────
if [ ! -d "$SOURCE_REPO" ]; then
    echo "Cloning autoresearch repo..."
    git clone https://github.com/karpathy/autoresearch.git "$SOURCE_REPO"
fi

# ── Copy to frozen working directory ─────────────────────────
if [ ! -d "$REPO_PATH" ]; then
    echo "Copying autoresearch to frozen working dir..."
    cp -r "$SOURCE_REPO" "$REPO_PATH"
fi

# ── Load modules ──────────────────────────────────────────────
module load gcc/14.2.0
module load cuda/12.8.1
module load conda

# ── Activate conda env ────────────────────────────────────────
conda activate "$CONDA_ENV"

# ── Cache HuggingFace models on blue storage ──────────────────
export HF_HOME="${PROJ_DIR}/.cache/huggingface"
export SGLANG_DISABLE_CUDNN_CHECK=1

# ── Info ──────────────────────────────────────────────────────
echo "Job ID:    $SLURM_JOB_ID"
echo "Node:      $(hostname)"
echo "GPUs:      $(nvidia-smi -L 2>/dev/null | wc -l)"
echo "Model:     $MODEL"
echo "Repo:      $REPO_PATH"
echo "Started:   $(date)"
echo "---"

# ── Install deps ──────────────────────────────────────────────
pip install openai "sglang[all]" --upgrade --quiet
cd "$REPO_PATH" && uv sync && cd "$SCAFFOLD_DIR"

# ── Start SGLang server ──────────────────────────────────────
echo "Starting SGLang server on GPU 0..."
CUDA_VISIBLE_DEVICES=0 python -m sglang.launch_server \
    --model-path "$MODEL" \
    --port "$VLLM_PORT" \
    --tp-size 1 \
    --mem-fraction-static 0.3 \
    --context-length "$MAX_MODEL_LEN" \
    --reasoning-parser qwen3 \
    &> sglang_server.log &
SERVER_PID=$!

# # ── (Alternative) Start vllm server ─────────────────────────
# echo "Starting vllm server..."
# vllm serve "$MODEL" \
#     --port "$VLLM_PORT" \
#     --max-model-len "$MAX_MODEL_LEN" \
#     --gpu-memory-utilization 0.90 \
#     --dtype bfloat16 \
#     --reasoning-parser qwen3 \
#     --language-model-only \
#     &> vllm_server.log &
# SERVER_PID=$!

# Cleanup server on exit (normal, error, or scancel)
cleanup() {
    echo "Cleaning up server (PID $SERVER_PID)..."
    kill $SERVER_PID 2>/dev/null || true
    wait $SERVER_PID 2>/dev/null || true
    echo "Done. $(date)"
}
trap cleanup EXIT

# Wait for server to be ready
echo "Waiting for server to be ready..."
for i in $(seq 1 360); do
    if curl -s "http://localhost:$VLLM_PORT/health" > /dev/null 2>&1; then
        echo "Server ready after ${i}s"
        break
    fi
    if ! kill -0 $SERVER_PID 2>/dev/null; then
        echo "ERROR: Server died. Check sglang_server.log"
        tail -20 sglang_server.log
        exit 1
    fi
    sleep 1
done

if ! curl -s "http://localhost:$VLLM_PORT/health" > /dev/null 2>&1; then
    echo "ERROR: Server failed to start within 360s"
    tail -20 sglang_server.log
    exit 1
fi

# ── Run experiments on GPU 1 ──────────────────────────────────
echo "Starting experiment loop on GPU 1..."
CUDA_VISIBLE_DEVICES=1 python main.py \
    --repo-path "$REPO_PATH" \
    --log-dir "$LOG_DIR" \
    --max-experiments "$MAX_EXPERIMENTS" \
    --llm-base-url "http://localhost:$VLLM_PORT/v1" \
    --llm-model "$MODEL" \
    --resume

echo "Finished. $(date)"
