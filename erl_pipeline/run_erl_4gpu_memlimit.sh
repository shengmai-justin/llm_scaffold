#!/bin/bash
#SBATCH --job-name=erl-4gpu-memlimit
#SBATCH --output=autoresearch_erl_%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=100gb
#SBATCH --time=12:00:00
#SBATCH --partition=hpg-b200
#SBATCH --gpus=3

# ── Configuration ─────────────────────────────────────────────
MODEL="Qwen/Qwen3.5-9B"
NUM_STEPS=100
MODEL_GPU=0
EVAL_GPUS="1,2"
# batch_size=4: 4 attempt1 + 4 attempt2 = 8 evals per step (all parallel on 4 workers)
BATCH_SIZE=4
WORKERS_PER_GPU=2
GPU_MEM_LIMIT_MB=88000
KL_COEF=0.1
LR=4e-5
LORA_RANK=32
LORA_ALPHA=64
PROJ_DIR="/blue/buyuheng/li_an.ucsb/proj_yepeng"
CONDA_ENV="${PROJ_DIR}/envs/myenv"
SCAFFOLD_DIR="${PROJ_DIR}/llm_scaffold_rl/llm_scaffold"
SOURCE_REPO="${SCAFFOLD_DIR}/autoresearch"
ERL_REPO="${SCAFFOLD_DIR}/autoresearch_erl"

cd "${SCAFFOLD_DIR}/erl_pipeline"

if [ ! -d "$SOURCE_REPO" ]; then
    git clone https://github.com/karpathy/autoresearch.git "$SOURCE_REPO"
fi

module load gcc/14.2.0
module load cuda/12.8.1
module load conda
conda activate "$CONDA_ENV"
export HF_HOME="${PROJ_DIR}/.cache/huggingface"

# Compile memory limiter
make -C "${SCAFFOLD_DIR}/gpu_mem_limit" clean && make -C "${SCAFFOLD_DIR}/gpu_mem_limit"

echo "Job ID:    $SLURM_JOB_ID"
echo "Node:      $(hostname)"
echo "GPUs:      $(nvidia-smi -L 2>/dev/null | wc -l)"
echo "Model:     $MODEL"
echo "Mode:      ERL 3-GPU memlimit (model=GPU0, eval=GPU1-2, 2 workers/GPU, 88GB cap)"
echo "Workers:   ${WORKERS_PER_GPU}/GPU x 2 eval GPUs = 4 workers, batch_size=${BATCH_SIZE}"
echo "Started:   $(date)"
echo "---"

cd "$SOURCE_REPO" && uv sync && cd "${SCAFFOLD_DIR}/erl_pipeline"

python erl_main.py \
    --repo-path "$ERL_REPO" \
    --source-repo "$SOURCE_REPO" \
    --model-dir "$MODEL" \
    --model-gpu "$MODEL_GPU" \
    --eval-gpus "$EVAL_GPUS" \
    --workers-per-gpu "$WORKERS_PER_GPU" \
    --gpu-mem-limit-mb "$GPU_MEM_LIMIT_MB" \
    --num-steps "$NUM_STEPS" \
    --batch-size "$BATCH_SIZE" \
    --kl-coef "$KL_COEF" \
    --lr "$LR" \
    --lora-rank "$LORA_RANK" \
    --lora-alpha "$LORA_ALPHA" \
    --temperature 0.7 \
    --max-new-tokens 16384 \
    --attn-impl sdpa \
    --log-dir ./erl_log

echo "Finished. $(date)"
