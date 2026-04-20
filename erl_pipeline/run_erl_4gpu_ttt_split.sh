#!/bin/bash
#SBATCH --job-name=erl-4gpu-ttt-split
#SBATCH --output=autoresearch_erl_ttt_split_%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=100gb
#SBATCH --time=10:00:00
#SBATCH --partition=hpg-b200
#SBATCH --gpus=4

# ── Configuration ─────────────────────────────────────────────
# ERL with TTT-Discover entropic LOO advantages for attempt1 / attempt2,
# using the split pipeline (Ideator + Implementer). See docs/SPLIT_PIPELINE.md.
# Only stage A (ideator) receives RL gradient; stage B runs under no_grad.
MODEL="Qwen/Qwen3.5-9B"
NUM_STEPS=100
MODEL_GPUS="0,1"
EVAL_GPUS="2,3"
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
# Namespaced so TTT+split runs don't share dirs with plain TTT or plain split.
ERL_REPO="${SCAFFOLD_DIR}/autoresearch_erl_ttt_split"

cd "${SCAFFOLD_DIR}/erl_pipeline"

if [ ! -d "$SOURCE_REPO" ]; then
    git clone https://github.com/karpathy/autoresearch.git "$SOURCE_REPO"
fi

module load gcc/14.2.0
module load cuda/12.8.1
module load conda
conda activate "$CONDA_ENV"
export HF_HOME="${PROJ_DIR}/.cache/huggingface"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Compile memory limiter
make -C "${SCAFFOLD_DIR}/gpu_mem_limit" clean && make -C "${SCAFFOLD_DIR}/gpu_mem_limit"

echo "Job ID:    $SLURM_JOB_ID"
echo "Node:      $(hostname)"
echo "GPUs:      $(nvidia-smi -L 2>/dev/null | wc -l)"
echo "Model:     $MODEL"
echo "Mode:      ERL 4-GPU TTT-adv SPLIT (model=GPU0-1, eval=GPU2-3, 2 workers/GPU, 88GB cap)"
echo "Advantage: ttt (entropic LOO with adaptive beta)"
echo "Pipeline:  split (Ideator RL-trained + Implementer frozen)"
echo "Workers:   ${WORKERS_PER_GPU}/GPU x 2 eval GPUs = 4 workers, batch_size=${BATCH_SIZE}"
echo "Started:   $(date)"
echo "---"

cd "$SOURCE_REPO" && uv sync && cd "${SCAFFOLD_DIR}/erl_pipeline"

python erl_main.py \
    --repo-path "$ERL_REPO" \
    --source-repo "$SOURCE_REPO" \
    --model-dir "$MODEL" \
    --model-gpus "$MODEL_GPUS" \
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
    --max-new-tokens 16000 \
    --think-budget 6000 \
    --attn-impl sdpa \
    --adv-type ttt \
    --split-pipeline \
    --log-dir ./erl_log_ttt_split

echo "Python exited at $(date)."
