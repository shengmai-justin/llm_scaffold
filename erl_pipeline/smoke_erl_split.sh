#!/bin/bash
# 2-step smoke test for the split pipeline on Pro 6000.
# Calls clean_pro6000_split.sh first, then runs the real launch script with
# NUM_STEPS=2 / BATCH_SIZE=2 via env overrides.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== Cleaning split-pipeline artifacts ==="
bash "${SCRIPT_DIR}/clean_pro6000_split.sh"

echo
echo "=== Running mock_split_test.py ==="
PROJ_DIR="${PROJ_DIR:-$HOME/auto_proj}"
SCAFFOLD_DIR="${SCAFFOLD_DIR:-$PROJ_DIR/llm_scaffold}"
VENV_DIR="${VENV_DIR:-$SCAFFOLD_DIR/.venv_pro6000}"
if [ -f "$VENV_DIR/bin/activate" ]; then
    # shellcheck disable=SC1091
    source "$VENV_DIR/bin/activate"
    python "${SCRIPT_DIR}/mock_split_test.py"
    deactivate
else
    echo "  (venv not found at $VENV_DIR; skipping python mock test — run_erl_pro6000_split.sh will create it)"
fi

echo
echo "=== Cluster smoke: 2 steps, batch_size 2, split pipeline ==="
NUM_STEPS=2 BATCH_SIZE=2 bash "${SCRIPT_DIR}/run_erl_pro6000_split.sh"
