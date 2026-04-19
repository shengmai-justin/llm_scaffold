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
python "${SCRIPT_DIR}/mock_split_test.py"

echo
echo "=== Cluster smoke: 2 steps, batch_size 2, split pipeline ==="
NUM_STEPS=2 BATCH_SIZE=2 bash "${SCRIPT_DIR}/run_erl_pro6000_split.sh"
