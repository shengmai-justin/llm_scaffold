#!/bin/bash
#SBATCH --job-name=gpu-hold
#SBATCH --output=gpu_hold_%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=100gb
#SBATCH --time=24:00:00
#SBATCH --partition=hpg-b200
#SBATCH --gpus=4

# Hold GPU allocation for interactive use.
# SSH into the node and run your scripts manually.
#
# Usage:
#   sbatch erl_pipeline/hold_gpus.sh
#   squeue -u $(whoami)          # find the node
#   ssh <node>                   # SSH in, run whatever you want
#   scancel <JOBID>              # release when done

ulimit -c 0

echo "Job ID:    $SLURM_JOB_ID"
echo "Node:      $(hostname)"
echo "GPUs:      $(nvidia-smi -L 2>/dev/null | wc -l)"
echo "Started:   $(date)"
echo "SSH in and run your scripts. scancel $SLURM_JOB_ID to release."

sleep infinity
