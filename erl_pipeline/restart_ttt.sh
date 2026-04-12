#!/bin/bash
# Restart ERL TTT pipeline on an existing SLURM allocation.
# Run from the srun --overlap shell on the compute node.
#
# Usage:
#   bash erl_pipeline/restart_ttt.sh

set -euo pipefail

PROJ_DIR="${PROJ_DIR:-/blue/buyuheng/li_an.ucsb/proj_yepeng}"
SCAFFOLD_DIR="${PROJ_DIR}/llm_scaffold_rl/llm_scaffold"

cd "$SCAFFOLD_DIR"

# Hold allocation while restarting
sleep infinity &
HOLDER_PID=$!

pkill -f "erl_main.py.*--adv-type ttt" || true
git pull
bash erl_pipeline/clean_ttt.sh
kill $HOLDER_PID

exec bash erl_pipeline/run_erl_4gpu_ttt.sh
