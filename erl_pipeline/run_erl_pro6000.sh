#!/bin/bash
# Run the ERL pipeline on 8× RTX Pro 6000 Blackwell.
#
# Differences from run_erl.sh (the 8× B200 variant):
#   1. No `module load` — this cluster has no HPC modules; Python deps are
#      installed into a local uv-managed venv instead.
#   2. No memlimit — each 96 GB Pro 6000 easily fits one train.py eval
#      worker, and we're running 1 worker/GPU × 7 eval GPUs = 7 workers.
#   3. No flash-attn-4 / kernels-community install — Pro 6000 (SM 12.0) uses
#      the SDPA-based train.py (train_sdpa.py in this repo) which has no
#      CUDA-kernel dependencies beyond PyTorch itself.
#
# ── One-time setup on the Pro 6000 cluster ────────────────────
#   1. Clone this scaffold somewhere and set $PROJ_DIR below (or export it).
#   2. Make sure NVIDIA driver + CUDA runtime (12.8+) are installed.
#   3. Copy train_sdpa.py from the scaffold into autoresearch/train.py:
#        cp ../train_sdpa.py "$SOURCE_REPO/train.py"
#      (Only needs to be done once. autoresearch_erl/train.py will be
#      refreshed automatically on the next clean + run.)
#   4. Submit: sbatch run_erl_pro6000.sh   (or: bash run_erl_pro6000.sh if
#      the cluster has no SLURM — the SBATCH headers are comments to bash.)

#SBATCH --job-name=erl-pro6000
#SBATCH --output=autoresearch_erl_pro6000_%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128gb
#SBATCH --time=12:00:00
#SBATCH --gpus=8
# #SBATCH --partition=<your-partition>    # uncomment + set if your cluster requires it

set -euo pipefail

# ── Paths (override via env vars if layout differs) ──────────
PROJ_DIR="${PROJ_DIR:-$HOME/autoresearch_pro6000}"
SCAFFOLD_DIR="${SCAFFOLD_DIR:-$PROJ_DIR/llm_scaffold}"
SOURCE_REPO="${SOURCE_REPO:-$SCAFFOLD_DIR/autoresearch}"
ERL_REPO="${ERL_REPO:-$SCAFFOLD_DIR/autoresearch_erl}"
VENV_DIR="${VENV_DIR:-$SCAFFOLD_DIR/.venv_pro6000}"

# ── ERL configuration ────────────────────────────────────────
MODEL="Qwen/Qwen3.5-9B"
NUM_STEPS=50
MODEL_GPU=0
EVAL_GPUS="1,2,3,4,5,6,7"
# batch_size=7: 7 attempt1 + 7 attempt2 = 14 evals per step, 1 wave per phase on 7 GPUs
BATCH_SIZE=7
WORKERS_PER_GPU=1
KL_COEF=0.1
LR=4e-5
LORA_RANK=32
LORA_ALPHA=64

cd "${SCAFFOLD_DIR}/erl_pipeline"

# ── Clone autoresearch source if missing ─────────────────────
if [ ! -d "$SOURCE_REPO" ]; then
    echo "Cloning autoresearch repo..."
    git clone https://github.com/karpathy/autoresearch.git "$SOURCE_REPO"
fi

# ── Ensure Pro 6000 SDPA train.py is in place ────────────────
# train_sdpa.py lives in $SCAFFOLD_DIR (one level above erl_pipeline). We
# overwrite autoresearch/train.py with it every run so source stays in sync.
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

# ── Install pinned dependencies (Pro 6000 stack) ─────────────
# NOTE: no flash-attn-4 and no `kernels` package — the SDPA train.py
# doesn't need either, and flash-attn-4 would otherwise drag in
# cutlass-dsl + pin torch to a specific version.
echo "Installing/updating Python dependencies..."
uv pip install --quiet \
    "torch==2.9.1" \
    "torchvision==0.24.1" \
    "torchaudio==2.9.1" \
    "transformers>=4.46" \
    "peft>=0.13" \
    "accelerate>=1.0" \
    "huggingface_hub" \
    "ray>=2.35" \
    "numpy>=2.2" \
    "pandas>=2.3" \
    "pyarrow>=21.0" \
    "tiktoken>=0.11" \
    "rustbpe>=0.1" \
    "matplotlib>=3.10" \
    "requests>=2.32"

# ── Runtime environment ──────────────────────────────────────
export HF_HOME="${HF_HOME:-$PROJ_DIR/.cache/huggingface}"
# Reduce PyTorch allocator fragmentation on the main-process GPU (GPU 0).
# Same reason as the B200 variant: ERL's phased loop accumulates
# reserved-but-unused blocks across generation / reflection / train.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ── Info ─────────────────────────────────────────────────────
echo "Job ID:    ${SLURM_JOB_ID:-local}"
echo "Node:      $(hostname)"
echo "GPUs:      $(nvidia-smi -L 2>/dev/null | wc -l)"
echo "Model:     $MODEL"
echo "Mode:      ERL Pro 6000 8-GPU (model=GPU0, eval=GPU1-7, 1 worker/GPU, no memlimit)"
echo "Workers:   ${WORKERS_PER_GPU}/GPU x 7 eval GPUs = 7 workers, batch_size=${BATCH_SIZE}"
echo "Started:   $(date)"
echo "---"

# ── Run ERL ──────────────────────────────────────────────────
# To use the TTT-Discover entropic advantages instead of GRPO,
# add: --adv-type ttt
python erl_main.py \
    --repo-path "$ERL_REPO" \
    --source-repo "$SOURCE_REPO" \
    --model-dir "$MODEL" \
    --model-gpu "$MODEL_GPU" \
    --eval-gpus "$EVAL_GPUS" \
    --workers-per-gpu "$WORKERS_PER_GPU" \
    --num-steps "$NUM_STEPS" \
    --batch-size "$BATCH_SIZE" \
    --kl-coef "$KL_COEF" \
    --lr "$LR" \
    --lora-rank "$LORA_RANK" \
    --lora-alpha "$LORA_ALPHA" \
    --temperature 0.7 \
    --max-new-tokens 16384 \
    --attn-impl sdpa \
    --log-dir ./erl_log_pro6000

echo "Finished. $(date)"
