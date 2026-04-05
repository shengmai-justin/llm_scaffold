#!/bin/bash
# Real-world test: two train.py processes on the same GPU, each capped at 88GB.
#
# Run on HiPerGator (B200, 1 GPU):
#   srun --partition=hpg-b200 --gpus=1 --mem=100gb --time=00:30:00 --pty bash gpu_mem_limit/test_dual_train.sh

set -e

# ── Paths (edit these to match your setup) ────────────────────
PROJ_DIR="/blue/buyuheng/li_an.ucsb/proj_yepeng"
SCAFFOLD_DIR="${PROJ_DIR}/llm_scaffold_rl/llm_scaffold"
SOURCE_REPO="${SCAFFOLD_DIR}/autoresearch"
MEMLIMIT_DIR="${SCAFFOLD_DIR}/gpu_mem_limit"
LIB_PATH="${MEMLIMIT_DIR}/libgpumemlimit.so"
LIMIT_MB=88000

# ── Load modules + conda ──────────────────────────────────────
module load gcc/14.2.0
module load cuda/12.8.1
module load conda
conda activate "${PROJ_DIR}/envs/myenv"
export HF_HOME="${PROJ_DIR}/.cache/huggingface"

# ── Compile memory limiter ────────────────────────────────────
echo "=== Compiling libgpumemlimit.so ==="
cd "$MEMLIMIT_DIR" && make clean && make && cd -
echo "OK"
echo ""

# ── Create two worker repos ───────────────────────────────────
WORKER_A="${SCAFFOLD_DIR}/autoresearch_memlimit_A"
WORKER_B="${SCAFFOLD_DIR}/autoresearch_memlimit_B"

for W in "$WORKER_A" "$WORKER_B"; do
    if [ ! -d "$W" ]; then
        echo "Creating worker repo: $W"
        cp -r "$SOURCE_REPO" "$W"
    fi
done

# ── Memory monitor (background) ──────────────────────────────
monitor_gpu() {
    while true; do
        echo ""
        echo "──── GPU Memory @ $(date +%H:%M:%S) ────"
        nvidia-smi --query-gpu=index,memory.used,memory.total,memory.free \
                   --format=csv,noheader,nounits | \
            awk -F', ' '{printf "GPU %s: used=%sMiB / total=%sMiB (free=%sMiB)\n", $1, $2, $3, $4}'
        sleep 15
    done
}
monitor_gpu &
MONITOR_PID=$!
trap "kill $MONITOR_PID 2>/dev/null" EXIT

# ── Helper to print torch memory inside a process ─────────────
MEMCHECK_SCRIPT='
import torch, os, time

# Print what the process sees
free, total = torch.cuda.mem_get_info()
tag = os.environ.get("WORKER_TAG", "?")
print(f"[Worker {tag}] pid={os.getpid()}")
print(f"[Worker {tag}] GPU mem reported: total={total/1e9:.2f}GB  free={free/1e9:.2f}GB")
print(f"[Worker {tag}] limit should be ~{88000*1024*1024/1e9:.2f}GB")
'

# ── Launch Worker A ───────────────────────────────────────────
echo ""
echo "=== Launching Worker A (GPU 0, limit=${LIMIT_MB}MB) ==="
(
    cd "$WORKER_A"
    uv sync 2>/dev/null

    # Print memory info before training
    CUDA_VISIBLE_DEVICES=0 GPU_MEM_LIMIT_MB=$LIMIT_MB LD_PRELOAD="$LIB_PATH" \
        WORKER_TAG=A python3 -c "$MEMCHECK_SCRIPT"

    # Run actual training
    echo "[Worker A] Starting train.py..."
    CUDA_VISIBLE_DEVICES=0 GPU_MEM_LIMIT_MB=$LIMIT_MB LD_PRELOAD="$LIB_PATH" \
        WORKER_TAG=A uv run train.py 2>&1 | sed 's/^/[A] /'
    echo "[Worker A] exit=$?"
) &
PID_A=$!
echo "Worker A pid=$PID_A"

# ── Small delay so logs don't interleave at start ─────────────
sleep 5

# ── Launch Worker B ───────────────────────────────────────────
echo ""
echo "=== Launching Worker B (GPU 0, limit=${LIMIT_MB}MB) ==="
(
    cd "$WORKER_B"
    uv sync 2>/dev/null

    # Print memory info before training
    CUDA_VISIBLE_DEVICES=0 GPU_MEM_LIMIT_MB=$LIMIT_MB LD_PRELOAD="$LIB_PATH" \
        WORKER_TAG=B python3 -c "$MEMCHECK_SCRIPT"

    # Run actual training
    echo "[Worker B] Starting train.py..."
    CUDA_VISIBLE_DEVICES=0 GPU_MEM_LIMIT_MB=$LIMIT_MB LD_PRELOAD="$LIB_PATH" \
        WORKER_TAG=B uv run train.py 2>&1 | sed 's/^/[B] /'
    echo "[Worker B] exit=$?"
) &
PID_B=$!
echo "Worker B pid=$PID_B"

# ── Wait for both ────────────────────────────────────────────
echo ""
echo "=== Waiting for both workers... ==="
set +e
wait $PID_A; STATUS_A=$?
wait $PID_B; STATUS_B=$?
set -e

echo ""
echo "=== Results ==="
echo "Worker A: exit=$STATUS_A"
echo "Worker B: exit=$STATUS_B"

if [ $STATUS_A -eq 0 ] && [ $STATUS_B -eq 0 ]; then
    echo "PASS: both train.py ran on same GPU with ${LIMIT_MB}MB cap each"
else
    echo "FAIL: check logs above"
fi

# Cleanup monitor
kill $MONITOR_PID 2>/dev/null
echo "Done. $(date)"
