# Pipeline Configurations

## Overview

Two pipelines share a common eval infrastructure (`rl_eval.py`) but differ in how they generate proposals and train.

| | RL | ERL |
|---|---|---|
| Approach | PUCT tree search + entropic policy gradient | Phased: attempt1 -> reflection -> attempt2 -> GRPO train |
| Proposals | LLM generates `search/replace` edits via PUCT-guided sampling | LLM generates edits, reflects on batch, retries with feedback |
| Training signals | Entropic advantages from rollout rewards | 4 signals: attempt1, reflection, attempt2, distill |
| Tree search | Yes (PUCT with exploration constant `C`) | No |
| Reflection | No | Yes (one per step, shared by all attempt2s) |
| Distillation | No | Yes (best edit distilled into training) |

## ERL Step Structure

Each ERL step runs 5 phases. With `batch_size=N`, one step produces `2N` evals (N attempt1 + N attempt2).

```
Step k
│
├── Phase 1: First attempts (generate N proposals, eval all in parallel)
│   ├── Generate proposal_1 ... proposal_N  (LLM, search/replace edits)
│   ├── Dispatch eval_1 ... eval_N          (Ray workers run train.py)
│   └── Collect results                     (val_bpb, reward for each)
│
├── Phase 2: Batch reflection (one reflection shared by all attempt2s)
│   ├── Summarize all attempt1 results into batch feedback
│   ├── LLM generates ONE reflection analyzing what worked/failed
│   └── Build reflection context for attempt2 prompts
│
├── Phase 3: Second attempts (generate N new proposals with reflection, eval all)
│   ├── Generate proposal_1 ... proposal_N  (LLM, with reflection context)
│   ├── Dispatch eval_1 ... eval_N          (Ray workers run train.py)
│   └── Collect results
│
├── Phase 4: Update best + build distillation targets
│   ├── If any attempt2 beats best_bpb → build distillation target
│   │   (re-score attempt2's response under the original prompt, no reflection)
│   └── Update best_code, best_bpb from all attempts
│
└── Phase 5: GRPO training (4 loss signals in one backward pass)
    ├── attempt1 rollouts   → policy gradient with GRPO advantages
    ├── reflection          → reward = mean of attempt2 rewards
    ├── attempt2 rollouts   → policy gradient with GRPO advantages
    └── distillation        → KL toward attempt2 response (if it improved)
```

**Key design choices:**

- One reflection per step, not per episode — keeps reflection cost constant regardless of batch size
- Distillation only for attempt2s that beat `best_bpb` — avoids training toward bad edits
- Distillation uses the original prompt (without reflection) so the model learns to produce good edits without needing reflection at inference time
- Reflection reward is the mean of all attempt2 rewards, connecting reflection quality to downstream outcomes

## Shared Configuration

Two supported hardware targets:

- **HiPerGator B200** — original target. Uses `module load gcc/14.2.0 cuda/12.8.1 conda` + shared conda env. Scripts: `run.sh`, `run_rl*.sh`, `run_erl*.sh` (non-`pro6000` variants).
- **Pro 6000 Blackwell (SM 12.0)** — newer target. No modules, no conda. Scripts (`run_pro6000.sh`, `run_erl_pro6000.sh`) install a pinned stack into a local uv venv and use SDPA everywhere (no flash-attn-4 / kernels-community / cutlass-dsl). `train_sdpa.py` at the scaffold root is the SDPA variant of `autoresearch/train.py` that the Pro 6000 scripts auto-install into the source repo on every run.

| Parameter | Value |
|---|---|
| Model | `Qwen/Qwen3.5-9B` |
| LoRA rank / alpha | 32 / 64 |
| Learning rate | 4e-5 |
| KL coefficient | 0.1 |
| Temperature | 0.7 |
| Max new tokens | 8192 (small B200) / 16384 (memlimit + Pro 6000) |
| Attention | SDPA (all variants) |
| Eval timeout | 600s per `train.py` run |
| Reward | `1/val_bpb` (success) or `0.0` (crash) |
| ERL attempt advantages | `--adv-type grpo` (default) or `--adv-type ttt` (entropic LOO) |
| B200 modules | gcc/14.2.0, cuda/12.8.1, conda |
| Pro 6000 env setup | uv venv + pinned pip install (no modules) |

## Run Scripts

### Small (3 GPUs, dev/test)

| | RL | ERL |
|---|---|---|
| Script | `rl_pipeline/run_rl_4gpu.sh` | `erl_pipeline/run_erl_4gpu.sh` |
| SBATCH GPUs | 3 | 3 |
| Model GPU | 0 | 0 |
| Eval GPUs | 1,2 | 1,2 |
| Workers/GPU | 1 | 1 |
| Batch size | 4 | 4 |
| Steps | 50 | 50 |
| Time limit | 4 hours | 4 hours |
| Submit | `sbatch rl_pipeline/run_rl_4gpu.sh` | `sbatch erl_pipeline/run_erl_4gpu.sh` |

### Full (8 GPUs, production)

| | RL | ERL |
|---|---|---|
| Script | `rl_pipeline/run_rl.sh` | `erl_pipeline/run_erl.sh` |
| SBATCH GPUs | 8 | 8 |
| Model GPU | 0 | 0 |
| Eval GPUs | 1,2,3,4,5,6,7 | 1,2,3,4,5,6,7 |
| Workers/GPU | 1 | 1 |
| Batch size | 7 | 7 |
| Steps | 50 | 50 |
| Time limit | 10 days | 4 days |
| Submit | `sbatch rl_pipeline/run_rl.sh` | `sbatch erl_pipeline/run_erl.sh` |

### Memory-limited (3 GPUs, 2 workers/GPU, B200)

| | RL | ERL |
|---|---|---|
| Script | `rl_pipeline/run_rl_4gpu_memlimit.sh` | `erl_pipeline/run_erl_4gpu_memlimit.sh` |
| SBATCH GPUs | 3 | 3 |
| Model GPU | 0 | 0 |
| Eval GPUs | 1,2 | 1,2 |
| Workers/GPU | 2 | 2 |
| GPU mem limit | 88000 MB per worker | 88000 MB per worker |
| Batch size | 8 | 4 |
| Max new tokens | 8192 | 16384 |
| Steps | 100 | 100 |
| Time limit | 12 hours | 12 hours |
| Submit | `sbatch rl_pipeline/run_rl_4gpu_memlimit.sh` | `sbatch erl_pipeline/run_erl_4gpu_memlimit.sh` |

### ERL TTT-advantages variant (3 GPUs, B200)

Same layout as `run_erl_4gpu_memlimit.sh` but passes `--adv-type ttt` to swap attempt1 / attempt2 advantages from GRPO to TTT-Discover entropic LOO. Reflection + distillation signals unchanged.

| Script | `erl_pipeline/run_erl_4gpu_ttt.sh` |
|---|---|
| SBATCH GPUs | 3 |
| Workers/GPU | 2 |
| Batch size | 4 |
| Adv type | `ttt` |
| Log dir | `./erl_log_ttt` (isolated from GRPO runs) |
| Submit | `sbatch erl_pipeline/run_erl_4gpu_ttt.sh` |

### Pro 6000 Blackwell (8 GPUs)

| | Frozen | ERL |
|---|---|---|
| Script | `run_pro6000.sh` | `erl_pipeline/run_erl_pro6000.sh` |
| SBATCH GPUs | 2 | 8 |
| Model GPU | 0 (SGLang) | 0 |
| Experiment / eval GPUs | 1 (main.py) | 1,2,3,4,5,6,7 |
| Workers/GPU | — | 1 |
| GPU mem limit | none (single SGLang + single main.py) | none (96 GB is plenty) |
| Batch size | — | 7 |
| Steps | unlimited | 50 |
| Env setup | uv venv + pinned pip install, no modules | same |
| Attention backend | SGLang `--attention-backend triton` | SDPA (`--attn-impl sdpa`) |
| train.py variant | `train_sdpa.py` auto-installed into source | same |
| Time limit | 12 hours | 12 hours |
| Submit | `sbatch run_pro6000.sh` | `sbatch erl_pipeline/run_erl_pro6000.sh` |

### Memory-limited test (3 GPUs, quick validation)

| | RL (test) |
|---|---|
| Script | `rl_pipeline/run_rl_memlimit_test.sh` |
| SBATCH GPUs | 3 |
| Model GPU | 0 |
| Eval GPUs | 1,2 |
| Workers/GPU | 2 |
| GPU mem limit | 88000 MB per worker |
| Batch size | 4 |
| Steps | 3 |
| Submit | `sbatch rl_pipeline/run_rl_memlimit_test.sh` |

## GPU Memory Limiter

LD_PRELOAD library (`gpu_mem_limit/libgpumemlimit.so`) that caps per-process GPU memory. Required when running multiple workers on the same GPU.

**How it works:** Intercepts CUDA runtime API (`cudaMalloc`, `cudaFree`, `cudaMemGetInfo`) and driver API (`cuMemCreate`, `cuMemRelease`, `cuMemAlloc_v2`, `cuMemFree_v2`, `cuMemGetInfo_v2`) via LD_PRELOAD. Tracks allocations in a hash table, rejects requests that would exceed the limit.

**Why both APIs:** PyTorch 2.x with CUDA 12.x can use "expandable segments" which allocates via driver API (`cuMemCreate`) instead of `cudaMalloc`. Intercepting only runtime API misses those allocations.

### Enabling

Add two flags to any pipeline run:

```bash
--workers-per-gpu 2
--gpu-mem-limit-mb 88000
```

When `--gpu-mem-limit-mb` is 0 (default), no memory limiting is applied. Existing scripts work unchanged.

### Compile

Must compile the `.so` before first use on the cluster:

```bash
make -C gpu_mem_limit clean && make -C gpu_mem_limit
```

### B200 sizing

| | Per GPU | Per worker (2/GPU) |
|---|---|---|
| Total VRAM | ~180 GB | 88 GB cap |
| train.py peak | ~74 GB | ~74 GB |
| Headroom | 106 GB | 14 GB |

## Isolation

RL and ERL pipelines are fully isolated and can run simultaneously.

| | RL | ERL |
|---|---|---|
| Repo copy | `autoresearch_rl/` | `autoresearch_erl/` |
| Workers | `autoresearch_rl_worker_*/` | `autoresearch_erl_worker_*/` |
| Logs | `rl_pipeline/rl_log/` | `erl_pipeline/erl_log/` |
| Results | `rl_pipeline/rl_log/results.tsv` | `erl_pipeline/erl_log/results.tsv` |
| Run log | `rl_pipeline/rl_log/run.log` | `erl_pipeline/erl_log/run.log` |
| Clean | `bash rl_pipeline/clean.sh` | `bash erl_pipeline/clean.sh` |

## Quick Reference

```bash
# ── B200 ────────────────────────────────────────────────────
# Dev test (3 GPUs, 1 worker/GPU)
sbatch rl_pipeline/run_rl_4gpu.sh
sbatch erl_pipeline/run_erl_4gpu.sh

# Production (8 GPUs, 1 worker/GPU)
sbatch rl_pipeline/run_rl.sh
sbatch erl_pipeline/run_erl.sh

# Memory-limited (3 GPUs, 2 workers/GPU)
sbatch rl_pipeline/run_rl_4gpu_memlimit.sh
sbatch erl_pipeline/run_erl_4gpu_memlimit.sh

# ERL with TTT-Discover entropic advantages (3 GPUs)
sbatch erl_pipeline/run_erl_4gpu_ttt.sh

# Frozen (2 GPUs, SGLang + experiment loop)
sbatch run.sh

# ── Pro 6000 Blackwell ──────────────────────────────────────
# Frozen (2 GPUs)
sbatch run_pro6000.sh

# ERL (8 GPUs, no memlimit)
sbatch erl_pipeline/run_erl_pro6000.sh

# ── Ops ─────────────────────────────────────────────────────
# Check job status
squeue -u $USER

# Tail logs
tail -f rl_pipeline/rl_log/run.log
tail -f erl_pipeline/erl_log/run.log

# Clean up worker repos
bash rl_pipeline/clean.sh
bash erl_pipeline/clean.sh
```
