#!/bin/bash
# GPU memory limiter test suite.
# Run on HiPerGator:
#   srun --partition=hpg-b200 --gpus=1 --mem=32gb --time=00:15:00 --pty bash test_memlimit.sh

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
LIB_PATH="$SCRIPT_DIR/libgpumemlimit.so"

# ── Load modules + conda (same as run_rl.sh) ────────────────
PROJ_DIR="/blue/buyuheng/li_an.ucsb/proj_yepeng"
module load gcc/14.2.0
module load cuda/12.8.1
module load conda
conda activate "${PROJ_DIR}/envs/myenv"

echo "gcc:    $(gcc --version | head -1)"
echo "python: $(python3 --version)"
echo "torch:  $(python3 -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'not found')"
echo ""

echo "=== Compiling libgpumemlimit.so ==="
make clean && make
echo "OK"

# Verify .so exports the right symbols
echo ""
echo "=== Symbol check ==="
nm -D libgpumemlimit.so | grep -E 'cuMem(Alloc|Free|GetInfo)' || { echo "FAIL: missing symbols"; exit 1; }
echo "OK"

# ── Test 1: Memory reporting is faked ────────────────────────
echo ""
echo "=== Test 1: Memory limit reporting ==="
GPU_MEM_LIMIT_MB=1000 LD_PRELOAD="$LIB_PATH" python3 -c "
import torch
free, total = torch.cuda.mem_get_info()
print(f'Reported total: {total/1e9:.2f} GB (should be ~1.05 GB)')
print(f'Reported free:  {free/1e9:.2f} GB')
expected = 1000 * 1024 * 1024
assert abs(total - expected) < 1e6, f'Expected ~1GB total, got {total}'
print('PASS')
"

# ── Test 2: Allocation rejected when over limit ─────────────
echo ""
echo "=== Test 2: OOM when over limit ==="
GPU_MEM_LIMIT_MB=500 LD_PRELOAD="$LIB_PATH" python3 -c "
import torch
try:
    # 200M floats = 800MB, should exceed 500MB limit
    t = torch.zeros(200_000_000, device='cuda')
    print('FAIL: should have OOMed')
    exit(1)
except torch.cuda.OutOfMemoryError:
    print('PASS: correctly rejected allocation over 500MB limit')
"

# ── Test 3: Two processes sharing one GPU ────────────────────
echo ""
echo "=== Test 3: Two processes on same GPU, each capped at 88GB ==="

CUDA_VISIBLE_DEVICES=0 GPU_MEM_LIMIT_MB=88000 LD_PRELOAD="$LIB_PATH" python3 -c "
import torch, time, os
t = torch.zeros(int(20e9 / 4), device='cuda')  # ~20GB
print(f'Process A [pid={os.getpid()}]: allocated {t.numel()*4/1e9:.1f} GB')
time.sleep(10)
print('Process A: done')
" &
PID_A=$!

sleep 2

CUDA_VISIBLE_DEVICES=0 GPU_MEM_LIMIT_MB=88000 LD_PRELOAD="$LIB_PATH" python3 -c "
import torch, time, os
t = torch.zeros(int(20e9 / 4), device='cuda')  # ~20GB
print(f'Process B [pid={os.getpid()}]: allocated {t.numel()*4/1e9:.1f} GB')
time.sleep(5)
print('Process B: done')
" &
PID_B=$!

set +e
wait $PID_A; STATUS_A=$?
wait $PID_B; STATUS_B=$?
set -e

if [ $STATUS_A -eq 0 ] && [ $STATUS_B -eq 0 ]; then
    echo "PASS: both processes ran successfully on the same GPU"
else
    echo "FAIL: process A=$STATUS_A, process B=$STATUS_B"
    exit 1
fi

echo ""
echo "=== All tests passed ==="
