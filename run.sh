#!/bin/bash
#SBATCH --job-name=autoresearch
#SBATCH --output=autoresearch_%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32gb
#SBATCH --time=10-12:00:00
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1

# ── Configuration ─────────────────────────────────────────────
MODEL="Qwen/Qwen3.5-9B"
VLLM_PORT=8000
MAX_EXPERIMENTS=100
MAX_MODEL_LEN=8192
PROJ_DIR="/blue/buyuheng/li_an.ucsb/proj_yepeng"
CONDA_ENV="${PROJ_DIR}/envs/myenv"
SCAFFOLD_DIR="${PROJ_DIR}/llm_scaffold"
REPO_PATH="${SCAFFOLD_DIR}/autoresearch"

# ── Navigate to project dir ───────────────────────────────────
cd "$SCAFFOLD_DIR"

# ── Clone autoresearch if not present ─────────────────────────
if [ ! -d "$REPO_PATH" ]; then
    echo "Cloning autoresearch repo..."
    git clone https://github.com/karpathy/autoresearch.git "$REPO_PATH"
fi

# ── Load modules ──────────────────────────────────────────────
module load gcc/12.2.0
module load cuda/12.8.1
module load conda

# ── Activate conda env ────────────────────────────────────────
conda activate "$CONDA_ENV"

# ── Info ──────────────────────────────────────────────────────
echo "Job ID:    $SLURM_JOB_ID"
echo "Node:      $(hostname)"
echo "GPUs:      $(nvidia-smi -L 2>/dev/null | wc -l)"
echo "Model:     $MODEL"
echo "Repo:      $REPO_PATH"
echo "Started:   $(date)"
echo "---"

# ── Install deps ──────────────────────────────────────────────
pip install openai --quiet
cd "$REPO_PATH" && uv sync && cd "$SCAFFOLD_DIR"

# ── Start vllm server on GPU 0 ───────────────────────────────
echo "Starting vllm server..."
vllm serve "$MODEL" \
    --port "$VLLM_PORT" \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization 0.90 \
    --dtype bfloat16 \
    --reasoning-parser qwen3 \
    &> vllm_server.log &
VLLM_PID=$!

# Cleanup vllm on exit (normal, error, or scancel)
cleanup() {
    echo "Cleaning up vllm server (PID $VLLM_PID)..."
    kill $VLLM_PID 2>/dev/null || true
    wait $VLLM_PID 2>/dev/null || true
    echo "Done. $(date)"
}
trap cleanup EXIT

# Wait for vllm to be ready
echo "Waiting for vllm to be ready..."
for i in $(seq 1 180); do
    if curl -s "http://localhost:$VLLM_PORT/health" > /dev/null 2>&1; then
        echo "vllm ready after ${i}s"
        break
    fi
    if ! kill -0 $VLLM_PID 2>/dev/null; then
        echo "ERROR: vllm server died. Check vllm_server.log"
        tail -20 vllm_server.log
        exit 1
    fi
    sleep 1
done

if ! curl -s "http://localhost:$VLLM_PORT/health" > /dev/null 2>&1; then
    echo "ERROR: vllm failed to start within 180s"
    tail -20 vllm_server.log
    exit 1
fi

# ── Run experiments on GPU 1 ──────────────────────────────────
echo "Starting experiment loop..."
python main.py \
    --repo-path "$REPO_PATH" \
    --max-experiments "$MAX_EXPERIMENTS" \
    --llm-base-url "http://localhost:$VLLM_PORT/v1" \
    --llm-model "$MODEL"

echo "Finished. $(date)"
