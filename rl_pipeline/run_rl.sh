#!/bin/bash
#SBATCH --job-name=autoresearch-rl
#SBATCH --output=autoresearch_rl_%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=100gb
#SBATCH --time=12:00:00
#SBATCH --partition=hpg-b200
#SBATCH --gpus=8

# ── Configuration ─────────────────────────────────────────────
MODEL="Qwen/Qwen3.5-9B"
NUM_STEPS=50
MODEL_GPU=0
EVAL_GPUS="1,2,3,4,5,6,7"
BATCH_SIZE=7
WORKERS_PER_GPU=1
KL_COEF=0.1
PUCT_C=1.0
LR=4e-5
LORA_RANK=32
LORA_ALPHA=64
PROJ_DIR="/blue/buyuheng/li_an.ucsb/proj_yepeng"
CONDA_ENV="${PROJ_DIR}/envs/myenv"
SCAFFOLD_DIR="${PROJ_DIR}/llm_scaffold_rl/llm_scaffold"
SOURCE_REPO="${SCAFFOLD_DIR}/autoresearch"
RL_REPO="${SCAFFOLD_DIR}/autoresearch_rl"

# ── Navigate to RL pipeline dir ───────────────────────────────
cd "${SCAFFOLD_DIR}/rl_pipeline"

# ── Clone autoresearch if not present ─────────────────────────
if [ ! -d "$SOURCE_REPO" ]; then
    echo "Cloning autoresearch repo..."
    git clone https://github.com/karpathy/autoresearch.git "$SOURCE_REPO"
fi

# ── Load modules ──────────────────────────────────────────────
module load gcc/14.2.0
module load cuda/12.8.1
module load conda

# ── Activate conda env ────────────────────────────────────────
conda activate "$CONDA_ENV"

# ── Cache HuggingFace models on blue storage ──────────────────
export HF_HOME="${PROJ_DIR}/.cache/huggingface"

# ── Info ──────────────────────────────────────────────────────
echo "Job ID:    $SLURM_JOB_ID"
echo "Node:      $(hostname)"
echo "GPUs:      $(nvidia-smi -L 2>/dev/null | wc -l)"
echo "Model:     $MODEL"
echo "Mode:      RL (local model, no SGLang)"
echo "Started:   $(date)"
echo "---"

# ── Sync autoresearch deps ────────────────────────────────────
cd "$SOURCE_REPO" && uv sync && cd "${SCAFFOLD_DIR}/rl_pipeline"

# ── Run RL experiment loop (no server needed) ─────────────────
echo "Starting RL experiment loop..."
python rl_main.py \
    --repo-path "$RL_REPO" \
    --source-repo "$SOURCE_REPO" \
    --model-dir "$MODEL" \
    --model-gpu "$MODEL_GPU" \
    --eval-gpus "$EVAL_GPUS" \
    --workers-per-gpu "$WORKERS_PER_GPU" \
    --num-steps "$NUM_STEPS" \
    --batch-size "$BATCH_SIZE" \
    --kl-coef "$KL_COEF" \
    --puct-c "$PUCT_C" \
    --lr "$LR" \
    --lora-rank "$LORA_RANK" \
    --lora-alpha "$LORA_ALPHA" \
    --temperature 0.7 \
    --max-new-tokens 8192 \
    --attn-impl sdpa \
    --log-dir ./rl_log

echo "Finished. $(date)"
