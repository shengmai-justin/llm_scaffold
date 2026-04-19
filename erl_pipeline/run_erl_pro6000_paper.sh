#!/bin/bash
# ERL on Pro 6000 with ERL-paper-parity hyperparameters.
#
# Same infra as run_erl_pro6000.sh; only training-loop hyperparameters
# differ to match `microsoft/experiential_rl/train_scripts/train_erl_sokoban.sh`.
#
# Changes from run_erl_pro6000.sh:
#   - lr: 4e-5 → 1e-6        (paper value)
#   - kl_coef: 0.1 → 0.001   (paper value)
#   - loss_agg_mode: default (seq-sum-token-mean) → seq-mean-token-sum
#   - clip_ratio_high: off → 0.28 (DAPO asymmetric)
#   - max_new_tokens: 16000 → 10240 (paper max_response_length)
#
# Everything else (model, GPU layout, batch_size, LoRA rank, temperature,
# think_budget) identical to run_erl_pro6000.sh.

#SBATCH --job-name=erl-pro6000-paper
#SBATCH --output=autoresearch_erl_pro6000_paper_%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128gb
#SBATCH --time=12:00:00
#SBATCH --gpus=8

set -euo pipefail

# ── Paths ────────────────────────────────────────────────────
PROJ_DIR="${PROJ_DIR:-$HOME/auto_proj}"
SCAFFOLD_DIR="${SCAFFOLD_DIR:-$PROJ_DIR/llm_scaffold}"
SOURCE_REPO="${SOURCE_REPO:-$SCAFFOLD_DIR/autoresearch}"
ERL_REPO="${ERL_REPO:-$SCAFFOLD_DIR/autoresearch_erl}"
VENV_DIR="${VENV_DIR:-$SCAFFOLD_DIR/.venv_pro6000}"

# ── ERL configuration (paper-parity) ─────────────────────────
MODEL="Qwen/Qwen3.5-9B"
NUM_STEPS=100
BATCH_SIZE=4
WORKERS_PER_GPU=1
KL_COEF=0.001                       # paper: 0.001 (ours default: 0.1)
LR=1e-6                             # paper: 1e-6 (ours default: 4e-5)
LORA_RANK=32
LORA_ALPHA=64
LOSS_AGG_MODE="seq-mean-token-sum"  # paper / verl default
CLIP_RATIO_HIGH=0.28                # DAPO asymmetric upper clip

cd "${SCAFFOLD_DIR}/erl_pipeline"

if [ ! -d "$SOURCE_REPO" ]; then
    echo "Cloning autoresearch repo..."
    git clone https://github.com/karpathy/autoresearch.git "$SOURCE_REPO"
fi

if [ -f "${SCAFFOLD_DIR}/train_sdpa.py" ]; then
    cp "${SCAFFOLD_DIR}/train_sdpa.py" "${SOURCE_REPO}/train.py"
    echo "Installed train_sdpa.py -> autoresearch/train.py"
else
    echo "WARNING: ${SCAFFOLD_DIR}/train_sdpa.py not found."
fi

if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
fi

if [ ! -d "$VENV_DIR" ]; then
    uv venv "$VENV_DIR" --python 3.10
fi
source "$VENV_DIR/bin/activate"

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

export HF_HOME="${HF_HOME:-$PROJ_DIR/.cache/huggingface}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export RAY_TMPDIR="/tmp/raye_${USER}"
mkdir -p "$RAY_TMPDIR"

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

echo "Mode:      ERL Pro 6000 paper-parity (model=GPU${MODEL_GPUS}, eval=GPU${EVAL_GPUS})"
echo "           loss_agg_mode=${LOSS_AGG_MODE}  clip_ratio_high=${CLIP_RATIO_HIGH}"
echo "           lr=${LR}  kl_coef=${KL_COEF}"
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
    --max-new-tokens 10240 \
    --think-budget 6000 \
    --attn-impl sdpa \
    --loss-agg-mode "$LOSS_AGG_MODE" \
    --clip-ratio-high "$CLIP_RATIO_HIGH" \
    --log-dir ./erl_log_pro6000_paper

echo "Finished. $(date)"
