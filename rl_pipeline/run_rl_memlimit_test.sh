#!/bin/bash
#SBATCH --job-name=rl-memlimit-test
#SBATCH --output=rl_memlimit_test_%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=100gb
#SBATCH --time=01:00:00
#SBATCH --partition=hpg-b200
#SBATCH --gpus=3

PROJ_DIR="/blue/buyuheng/li_an.ucsb/proj_yepeng"
SCAFFOLD_DIR="${PROJ_DIR}/llm_scaffold_rl/llm_scaffold"

module load gcc/14.2.0
module load cuda/12.8.1
module load conda
conda activate "${PROJ_DIR}/envs/myenv"
export HF_HOME="${PROJ_DIR}/.cache/huggingface"

cd "${SCAFFOLD_DIR}/rl_pipeline"

echo "Job ID:    $SLURM_JOB_ID"
echo "Node:      $(hostname)"
echo "GPUs:      $(nvidia-smi -L 2>/dev/null | wc -l)"
echo "Mode:      RL memlimit test (2 workers/GPU, 88GB cap)"
echo "Started:   $(date)"
echo "---"

cd "$SCAFFOLD_DIR/autoresearch" && uv sync && cd "${SCAFFOLD_DIR}/rl_pipeline"

python rl_main.py \
    --repo-path ../autoresearch_rl \
    --source-repo ../autoresearch \
    --model-dir Qwen/Qwen3.5-9B \
    --model-gpu 0 \
    --eval-gpus "1,2" \
    --workers-per-gpu 2 \
    --gpu-mem-limit-mb 88000 \
    --num-steps 3 \
    --batch-size 4 \
    --lr 4e-5 \
    --lora-rank 32 \
    --lora-alpha 64 \
    --temperature 0.7 \
    --max-new-tokens 8192 \
    --attn-impl sdpa \
    --log-dir ./rl_log

echo "Finished. $(date)"
