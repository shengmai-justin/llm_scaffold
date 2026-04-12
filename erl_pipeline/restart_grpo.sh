#!/bin/bash
# Restart ERL GRPO pipeline on an existing SLURM allocation.
# Run from the srun --overlap shell on the compute node.
#
# Usage:
#   bash erl_pipeline/restart_grpo.sh

set -euo pipefail

PROJ_DIR="${PROJ_DIR:-/blue/buyuheng/li_an.ucsb/proj_yepeng}"
SCAFFOLD_DIR="${PROJ_DIR}/llm_scaffold_rl/llm_scaffold"

cd "$SCAFFOLD_DIR"

# Hold allocation while restarting
sleep infinity &
HOLDER_PID=$!

pkill -f "erl_main.py.*--log-dir ./erl_log" || true
git pull
bash erl_pipeline/clean.sh
kill $HOLDER_PID

exec bash erl_pipeline/run_erl_4gpu_memlimit.sh
