#!/bin/bash
#SBATCH --job-name=erl-4gpu-ttt
#SBATCH --output=autoresearch_erl_ttt_%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=100gb
#SBATCH --time=04:00:00
#SBATCH --partition=hpg-b200
#SBATCH --gpus=4

# ── Configuration ─────────────────────────────────────────────
# ERL with TTT-Discover entropic LOO advantages for attempt1 / attempt2
# (same phased loop as run_erl_4gpu_memlimit.sh, only the advantage
# computation for attempts is swapped from GRPO to TTT entropic).
MODEL="Qwen/Qwen3.5-9B"
NUM_STEPS=100
MODEL_GPUS="0,1"
EVAL_GPUS="2,3"
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
# Namespaced by --adv-type so TTT runs don't share the working repo
# (and therefore the worker dirs) with the default GRPO runs.
ERL_REPO="${SCAFFOLD_DIR}/autoresearch_erl_ttt"

cd "${SCAFFOLD_DIR}/erl_pipeline"

if [ ! -d "$SOURCE_REPO" ]; then
    git clone https://github.com/karpathy/autoresearch.git "$SOURCE_REPO"
fi

module load gcc/14.2.0
module load cuda/12.8.1
module load conda
conda activate "$CONDA_ENV"
export HF_HOME="${PROJ_DIR}/.cache/huggingface"
# Reduce PyTorch allocator fragmentation on long training loops.
# Without this, the ERL main process on GPU 0 accumulates reserved-but-unused
# blocks across phases (generation, reflection, distillation, train) and OOMs
# even though live memory is well under the B200's 180 GB.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Compile memory limiter
make -C "${SCAFFOLD_DIR}/gpu_mem_limit" clean && make -C "${SCAFFOLD_DIR}/gpu_mem_limit"

echo "Job ID:    $SLURM_JOB_ID"
echo "Node:      $(hostname)"
echo "GPUs:      $(nvidia-smi -L 2>/dev/null | wc -l)"
echo "Model:     $MODEL"
echo "Mode:      ERL 4-GPU TTT-adv (model=GPU0-1, eval=GPU2-3, 2 workers/GPU, 88GB cap)"
echo "Advantage: ttt (entropic LOO with adaptive beta)"
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
    --log-dir ./erl_log_ttt

echo "Python exited at $(date)."
