#!/bin/bash
# Run the ERL pipeline on 8× RTX Pro 6000 Blackwell with the split pipeline
# (Ideator + Implementer). See docs/SPLIT_PIPELINE.md.
#
# Differences from run_erl_pro6000.sh:
#   1. Adds --split-pipeline (RL signal flows only through stage A).
#   2. Writes to ./erl_log_pro6000_split to keep artifacts separate.
#   3. Starts from step 0 (no --resume-step).

#SBATCH --job-name=erl-pro6000-split
#SBATCH --output=autoresearch_erl_pro6000_split_%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128gb
#SBATCH --time=12:00:00
#SBATCH --gpus=8

set -euo pipefail

# ── Paths (override via env vars if layout differs) ──────────
PROJ_DIR="${PROJ_DIR:-$HOME/auto_proj}"
SCAFFOLD_DIR="${SCAFFOLD_DIR:-$PROJ_DIR/llm_scaffold}"
SOURCE_REPO="${SOURCE_REPO:-$SCAFFOLD_DIR/autoresearch}"
ERL_REPO="${ERL_REPO:-$SCAFFOLD_DIR/autoresearch_erl_split}"
VENV_DIR="${VENV_DIR:-$SCAFFOLD_DIR/.venv_pro6000}"

# ── ERL configuration (override via env for smoke runs) ──────
MODEL="${MODEL:-Qwen/Qwen3.5-9B}"
NUM_STEPS="${NUM_STEPS:-100}"
BATCH_SIZE="${BATCH_SIZE:-4}"
WORKERS_PER_GPU="${WORKERS_PER_GPU:-1}"
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

# ── Clone the ERL working repo if missing (erl_main also does this, but
#    we need the dir to exist before the .venv symlink step below).
if [ ! -d "$ERL_REPO" ]; then
    echo "Cloning $SOURCE_REPO -> $ERL_REPO"
    cp -r "$SOURCE_REPO" "$ERL_REPO"
fi

# ── Ensure the ERL repo's .venv points at our activated env ──
# uv run resolves ./.venv in the cwd, ignoring VIRTUAL_ENV. Without this
# symlink, `uv run train.py` inside autoresearch_erl_split/ auto-creates
# a broken local .venv and the baseline fails with "no Python executable".
if [ -e "$ERL_REPO/.venv" ] && [ ! -L "$ERL_REPO/.venv" ]; then
    rm -rf "$ERL_REPO/.venv"
fi
ln -sfn "$VENV_DIR" "$ERL_REPO/.venv"

# ── Install pinned dependencies (Pro 6000 stack) ─────────────
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
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export RAY_TMPDIR="/tmp/raye_split_${USER}"
mkdir -p "$RAY_TMPDIR"
# NOTE: do NOT wipe RAY_TMPDIR here — that would break --resume-step flows.
# Use clean_pro6000_split.sh for a fresh start.

# ── Info ─────────────────────────────────────────────────────
echo "Job ID:    ${SLURM_JOB_ID:-local}"
echo "Node:      $(hostname)"
echo "GPUs:      $(nvidia-smi -L 2>/dev/null | wc -l)"
echo "Model:     $MODEL"

mapfile -t FREE_GPUS < <(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
    | sort -t',' -k2 -n | awk -F',' '{gsub(/ /,""); print $1}')
NEED=8
if [ "${#FREE_GPUS[@]}" -lt "$NEED" ]; then
    echo "ERROR: need $NEED free GPUs, found ${#FREE_GPUS[@]}"
    nvidia-smi --query-gpu=index,memory.used --format=csv
    exit 1
fi
MODEL_GPUS="${FREE_GPUS[0]},${FREE_GPUS[1]},${FREE_GPUS[2]},${FREE_GPUS[3]}"
EVAL_GPUS="${FREE_GPUS[4]},${FREE_GPUS[5]},${FREE_GPUS[6]},${FREE_GPUS[7]}"

echo "Mode:      ERL Pro 6000 8-GPU SPLIT (model=GPU${MODEL_GPUS}, eval=GPU${EVAL_GPUS})"
echo "Workers:   ${WORKERS_PER_GPU}/GPU x 4 eval GPUs = 4 workers, batch_size=${BATCH_SIZE}"
echo "Started:   $(date)"
echo "---"

python erl_main.py \
    --repo-path "$ERL_REPO" \
    --source-repo "$SOURCE_REPO" \
    --model-dir "$MODEL" \
    --model-gpus "$MODEL_GPUS" \
    --eval-gpus "$EVAL_GPUS" \
    --workers-per-gpu "$WORKERS_PER_GPU" \
    --num-steps "$NUM_STEPS" \
    --batch-size "$BATCH_SIZE" \
    --kl-coef "$KL_COEF" \
    --lr "$LR" \
    --lora-rank "$LORA_RANK" \
    --lora-alpha "$LORA_ALPHA" \
    --temperature 0.7 \
    --max-new-tokens 16000 \
    --think-budget 6000 \
    --split-pipeline \
    --attn-impl sdpa \
    --log-dir ./erl_log_pro6000_split

echo "Finished. $(date)"
