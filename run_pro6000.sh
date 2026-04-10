#!/bin/bash
# Run the FROZEN pipeline on RTX Pro 6000 Blackwell.
#
# Layout: 2 GPUs — one hosts the SGLang LLM server, the other runs the
# experiment loop (main.py → planner → train.py).  Only 2 of the 8 Pro
# 6000s are used because the frozen pipeline is strictly serial — one
# experiment at a time, waiting on the LLM's response — so extra GPUs
# would sit idle.  Leave the remaining 6 GPUs free for the ERL job.
#
# This cluster has NO SLURM, so the script auto-picks the 2 GPUs with
# the least memory used at launch time instead of hardcoding indices.
#
# Differences from run.sh (the B200 variant):
#   1. No `module load` — this cluster has no HPC modules; deps are
#      installed into a local uv venv instead.
#   2. No flash-attn-4 / kernels-community / cutlass-dsl install — the
#      SDPA train.py doesn't need them, and SGLang's FA4 JIT kernel is
#      non-fatal when flash-attn-4 is absent (Scheduler logs an import
#      error, but the server still serves via a fallback backend).
#   3. SGLang is launched with `--attention-backend triton` to keep it
#      off the FA4 codepath entirely on SM 12.0.
#   4. `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is exported
#      for both the SGLang process and the experiment loop.
#
# ── One-time setup on the Pro 6000 cluster ────────────────────
#   1. Clone this scaffold; set $PROJ_DIR below (or export it).
#   2. NVIDIA driver + CUDA runtime (12.8+) must be installed.
#   3. Copy train_sdpa.py from this repo into autoresearch/train.py.
#      (The script auto-syncs it on every run, so you only need to make
#      sure train_sdpa.py exists at $SCAFFOLD_DIR/train_sdpa.py.)
#   4. Submit: sbatch run_pro6000.sh     (or: bash run_pro6000.sh if
#      the cluster has no SLURM — SBATCH lines are comments to bash.)

#SBATCH --job-name=autoresearch-pro6000
#SBATCH --output=autoresearch_pro6000_%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64gb
#SBATCH --time=12:00:00
#SBATCH --gpus=2
# #SBATCH --partition=<your-partition>    # uncomment + set if your cluster requires it

set -euo pipefail

# ── Paths (auto-detected from script location; override via env) ─
# SCAFFOLD_DIR defaults to the dir containing this script, so the
# script works from any layout (~/auto_proj/llm_scaffold,
# ~/autoresearch_pro6000/llm_scaffold, etc.) without env vars.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCAFFOLD_DIR="${SCAFFOLD_DIR:-$SCRIPT_DIR}"
PROJ_DIR="${PROJ_DIR:-$(dirname "$SCAFFOLD_DIR")}"
SOURCE_REPO="${SOURCE_REPO:-$SCAFFOLD_DIR/autoresearch}"
REPO_PATH="${REPO_PATH:-$SCAFFOLD_DIR/autoresearch_frozen}"
LOG_DIR="${LOG_DIR:-$SCAFFOLD_DIR/frozen_log}"
VENV_DIR="${VENV_DIR:-$SCAFFOLD_DIR/.venv_pro6000}"

# ── Frozen pipeline configuration ────────────────────────────
MODEL="Qwen/Qwen3.5-9B"
VLLM_PORT=8000
MAX_EXPERIMENTS=0
MAX_MODEL_LEN=30000

cd "$SCAFFOLD_DIR"

# ── Clone autoresearch source if missing ─────────────────────
if [ ! -d "$SOURCE_REPO" ]; then
    echo "Cloning autoresearch repo..."
    git clone https://github.com/karpathy/autoresearch.git "$SOURCE_REPO"
fi

# ── Ensure Pro 6000 SDPA train.py is in place ────────────────
if [ -f "${SCAFFOLD_DIR}/train_sdpa.py" ]; then
    cp "${SCAFFOLD_DIR}/train_sdpa.py" "${SOURCE_REPO}/train.py"
    echo "Installed train_sdpa.py -> autoresearch/train.py"
else
    echo "WARNING: ${SCAFFOLD_DIR}/train_sdpa.py not found; using whatever is in autoresearch/train.py."
fi

# ── Install uv if missing ────────────────────────────────────
if ! command -v uv >/dev/null 2>&1; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
fi

# ── Create + activate venv ───────────────────────────────────
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating venv at $VENV_DIR..."
    uv venv "$VENV_DIR" --python 3.10
fi
source "$VENV_DIR/bin/activate"

# ── Install pinned dependencies (Pro 6000 frozen stack) ──────
# NOTE: sglang 0.5.10.post1 hard-depends on flash-attn-4>=4.0.0b4 in
# its metadata, so we must pass --prerelease=allow for uv to resolve
# it.  flash-attn-4 installs as a wheel but is never loaded at
# runtime because --attention-backend triton keeps SGLang off the FA4
# codepath.  SGLang will log a non-fatal FA4 import error at startup.
echo "Installing/updating Python dependencies..."
uv pip install --quiet --prerelease=allow \
    "torch==2.9.1" \
    "torchvision==0.24.1" \
    "torchaudio==2.9.1" \
    "sglang[all]==0.5.10.post1" \
    "openai" \
    "numpy>=2.2" \
    "pandas>=2.3" \
    "pyarrow>=21.0" \
    "tiktoken>=0.11" \
    "rustbpe>=0.1" \
    "matplotlib>=3.10" \
    "requests>=2.32"

# ── Runtime environment ──────────────────────────────────────
export HF_HOME="${HF_HOME:-$PROJ_DIR/.cache/huggingface}"
export SGLANG_DISABLE_CUDNN_CHECK=1
# Reduce PyTorch allocator fragmentation for both the SGLang process
# and the experiment-loop process.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ── CUDA toolkit sanity check ────────────────────────────────
# sglang transitively imports deep_gemm, which asserts on a missing
# CUDA_HOME at import time.  The Pro 6000 cluster ships only the
# driver, not the toolkit — install it to $HOME/cuda-XX.Y and export
# CUDA_HOME in your shell (or inherit from nvcc on PATH).
if [ -z "${CUDA_HOME:-}" ]; then
    if command -v nvcc >/dev/null 2>&1; then
        CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"
        export CUDA_HOME
    fi
fi
if [ -z "${CUDA_HOME:-}" ] || [ ! -x "$CUDA_HOME/bin/nvcc" ]; then
    echo "ERROR: CUDA toolkit not found."
    echo "  nvcc not on PATH and CUDA_HOME is unset or invalid."
    echo "  Install the toolkit (e.g. local runfile install to \$HOME/cuda-12.8)"
    echo "  and export CUDA_HOME + PATH + LD_LIBRARY_PATH in your shell."
    exit 1
fi
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
echo "CUDA_HOME: $CUDA_HOME ($(nvcc --version | grep -oE 'release [0-9.]+' | head -1))"

# ── Info ─────────────────────────────────────────────────────
# ── Auto-pick 2 least-used GPUs (no SLURM on this cluster) ───
mapfile -t FREE_GPUS < <(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
    | sort -t',' -k2 -n | awk -F',' '{gsub(/ /,""); print $1}')
if [ "${#FREE_GPUS[@]}" -lt 2 ]; then
    echo "ERROR: need 2 GPUs, found ${#FREE_GPUS[@]}"
    nvidia-smi --query-gpu=index,memory.used --format=csv
    exit 1
fi
SGLANG_GPU="${FREE_GPUS[0]}"
MAIN_GPU="${FREE_GPUS[1]}"

echo "Job ID:    ${SLURM_JOB_ID:-local}"
echo "Node:      $(hostname)"
echo "GPUs:      $(nvidia-smi -L 2>/dev/null | wc -l)"
echo "Model:     $MODEL"
echo "Mode:      Frozen Pro 6000 2-GPU (SGLang=GPU${SGLANG_GPU}, main.py=GPU${MAIN_GPU})"
echo "Repo:      $REPO_PATH"
echo "Started:   $(date)"
echo "---"

# ── Start SGLang server on the chosen GPU ────────────────────
# --attention-backend triton forces SGLang off its FA4 codepath so we
# don't depend on flash-attn-4 being installed.  Triton JIT-compiles
# per device, which is fine on SM 12.0 (Pro 6000 Blackwell).
echo "Starting SGLang server on GPU ${SGLANG_GPU}..."
CUDA_VISIBLE_DEVICES="$SGLANG_GPU" python -m sglang.launch_server \
    --model-path "$MODEL" \
    --port "$VLLM_PORT" \
    --tp-size 1 \
    --mem-fraction-static 0.5 \
    --context-length "$MAX_MODEL_LEN" \
    --attention-backend triton \
    --reasoning-parser qwen3 \
    &> sglang_server.log &
SERVER_PID=$!

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
        tail -30 sglang_server.log
        exit 1
    fi
    sleep 1
done

if ! curl -s "http://localhost:$VLLM_PORT/health" > /dev/null 2>&1; then
    echo "ERROR: Server failed to start within 360s"
    tail -30 sglang_server.log
    exit 1
fi

# ── Run experiments on the chosen GPU ────────────────────────
echo "Starting experiment loop on GPU ${MAIN_GPU}..."
CUDA_VISIBLE_DEVICES="$MAIN_GPU" python main.py \
    --repo-path "$REPO_PATH" \
    --source-repo "$SOURCE_REPO" \
    --log-dir "$LOG_DIR" \
    --max-experiments "$MAX_EXPERIMENTS" \
    --llm-base-url "http://localhost:$VLLM_PORT/v1" \
    --llm-model "$MODEL" \
    --resume

echo "Finished. $(date)"
