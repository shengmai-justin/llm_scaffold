# LLM Scaffold

A deterministic experiment harness where an LLM proposes small `search/replace` edits to `train.py` and the harness handles everything else (git, execution, parsing, logging, keep/revert).

Built on top of [karpathy/autoresearch](https://github.com/karpathy/autoresearch).

## Architecture

Three pipelines sharing a common base.

**Frozen pipeline** (uses any OpenAI-compatible API):
```
main.py         — orchestration, setup, experiment loop, recovery
state.py        — persistent state (state.json), git operations, file I/O
planner.py      — LLM context assembly, proposal, search/replace editing
results.py      — execution, log parsing, result logging, keep/discard decision
prompt.md       — system prompt for the LLM (editable without touching code)
run.sh          — Slurm launch (2× B200, SGLang + main.py)
run_pro6000.sh  — Slurm launch (2× Pro 6000 Blackwell, uv venv, SDPA, no modules)
train_sdpa.py   — SDPA variant of autoresearch/train.py for Pro 6000 (SM 12.0)
clean.sh        — wipe frozen working dir + logs (parallel unlink + rm -rf fallback)
```

**RL pipeline** (`rl_pipeline/`, TTT-Discover style with PUCT tree search and entropic LOO advantages)

**ERL pipeline** (`erl_pipeline/`, Experiential RL — phased loop with one reflection per step, GRPO + RAFT distillation)

All three pipelines copy the source `autoresearch/` repo into their own working directory (`autoresearch_frozen/`, `autoresearch_rl/`, `autoresearch_erl/`) via `shutil.copytree`.

## Setup

Two supported hardware targets. Pick the one that matches your cluster.

### Target A — HiPerGator B200 (original)

**Prerequisites**
- NVIDIA B200 GPU nodes (2–8 GPUs depending on pipeline)
- Shared conda env with Python 3.10+
- SLURM + Lmod modules (`gcc/14.2.0`, `cuda/12.8.1`, `conda`)

**One-time setup**
```bash
# 1. Clone the scaffold
cd /path/to/your/project
git clone git@github.com:<your-username>/llm_scaffold.git

# 2. Install SGLang + openai in your conda env
module load gcc/14.2.0 cuda/12.8.1 conda
conda activate /path/to/your/conda/env
pip install openai "sglang[all]==0.5.10.post1" "flash-attn-4==4.0.0b4" \
    "torch==2.9.1" "torchvision==0.24.1" "torchaudio==2.9.1" "cuda-python==12.9"
```

**Run**
```bash
cd /path/to/your/project/llm_scaffold
sbatch run.sh                              # frozen pipeline (2× B200)
sbatch rl_pipeline/run_rl.sh               # RL pipeline (8× B200)
sbatch erl_pipeline/run_erl.sh             # ERL pipeline (8× B200)
```

### Target B — RTX Pro 6000 Blackwell (SM 12.0, 8 GPUs)

**Prerequisites**
- 8× RTX Pro 6000 Blackwell (96 GB each)
- NVIDIA driver + CUDA runtime ≥ 12.8
- Python 3.10+ on PATH (uv installed automatically if missing)
- No modules, no conda required

**One-time setup**
```bash
# 1. Clone the scaffold on the Pro 6000 cluster
git clone git@github.com:<your-username>/llm_scaffold.git
export PROJ_DIR=/path/to/parent_of_scaffold   # scripts read this env var
```
No manual pip install — the Pro 6000 run scripts create a uv venv at
`$SCAFFOLD_DIR/.venv_pro6000` on first run and install the pinned stack
(torch 2.9.1 + transformers + peft + accelerate + autoresearch data
deps) into it. No `flash-attn-4`, no `kernels-community`, no
`cutlass-dsl` — the SDPA `train_sdpa.py` and SGLang's Triton attention
backend need none of those.

**Run**
```bash
sbatch run_pro6000.sh                       # frozen pipeline (2× Pro 6000)
sbatch erl_pipeline/run_erl_pro6000.sh      # ERL pipeline (8× Pro 6000)
```
`train_sdpa.py` at the scaffold root is auto-copied over
`autoresearch/train.py` on every Pro 6000 run, so source stays in sync.

### What both targets do automatically

The autoresearch repo and Qwen3.5-9B model are auto-downloaded into
`$HF_HOME` on first run. First run takes ~15–30 minutes (pip install +
model download); subsequent runs reuse both caches and take < 1 min to
reach the training loop.

The script handles:
- Cloning autoresearch repo if missing
- Downloading Qwen3.5-9B model to blue storage (`HF_HOME`)
- `uv sync` for training dependencies
- Starting SGLang server (single B200, ~34GB VRAM for model + KV cache)
- Auto-resetting dirty `train.py` from interrupted runs
- Running baseline + experiment loop (unlimited until wall time)
- Graceful cleanup on exit (SIGTERM/SIGINT saves state, resets to best commit)

### Monitor

```bash
# Job status
squeue -u $USER

# Live output
tail -f autoresearch_<job_id>.log

# Experiment history
cat results.tsv

# Current best
cat state.json

# SGLang server logs
cat sglang_server.log

# Cancel
scancel <job_id>
```

### Resume after interrupt

Add `--resume` to the `python main.py` call in `run.sh`, then:

```bash
sbatch run.sh
```

Picks up from the last saved `state.json` — no work is lost.

## How it works

```
Setup:
  1. Auto-reset dirty train.py if interrupted
  2. Create experiment branch (autoresearch/<date>)
  3. Run baseline training, record initial val_bpb

Experiment loop (runs until wall time):
  1. Reset to best commit
  2. Ask LLM for one proposal (JSON: description + search/replace edits)
  3. Validate edits exist in train.py (retry once if not)
  4. Apply edits, commit
  5. Run training (uv run train.py, ~5 min)
  6. Parse val_bpb + peak_vram_mb from output
  7. Keep (val_bpb improved) or revert (git reset --hard)
  8. Log to results.tsv, save state.json
  9. Repeat
```

Only experiments that actually run training count toward `experiment_count`. Failed proposals and edit failures do not consume budget.

### Decision rules

| Status | Condition |
|---|---|
| `keep` | val_bpb strictly lower, OR equal with lower peak VRAM |
| `discard` | val_bpb equal or worse |
| `crash` | Metrics missing or timeout |
| `edit_failed` | Search/replace edits could not be applied |

### VRAM budget

**B200 (178 GB per GPU, 2 GPUs for frozen)**

| Component | VRAM |
|---|---|
| SGLang model weights (9B bf16) | ~18 GB |
| SGLang KV cache (30% of remaining) | ~48 GB |
| Available for training on the other GPU | ~112 GB |

**Pro 6000 Blackwell (96 GB per GPU, 2 GPUs for frozen)**

| Component | VRAM |
|---|---|
| SGLang model weights (9B bf16) | ~18 GB |
| SGLang KV cache (`--mem-fraction-static 0.5`, ~48 GB total for SGLang) | ~30 GB |
| Available for training on the other GPU | ~96 GB |

For ERL on 8× Pro 6000: model + LoRA on GPU 0 (~35 GB), one eval
worker per GPU on GPUs 1–7 (~5–10 GB each). No memlimit needed because
each worker owns its GPU.

## Configuration

Edit the top of `run.sh`:

```bash
MODEL="Qwen/Qwen3.5-9B"     # LLM for proposals
MAX_EXPERIMENTS=0             # 0 = unlimited (runs until wall time)
MAX_MODEL_LEN=30000           # context length for SGLang
```

Edit `prompt.md` to change the system prompt sent to the LLM — no code changes needed.

CLI flags for `main.py`:

```
--repo-path        Working repo path (default: ./autoresearch_frozen)
--source-repo      Source repo to copy from on first run (default: ./autoresearch)
--log-dir          Directory for results.tsv, run.log, state.json (default: scaffold root)
--max-experiments  Max iterations (default: 100, 0 = unlimited)
--llm-base-url     OpenAI-compatible endpoint (default: http://localhost:8000/v1)
--llm-model        Model name (default: Qwen/Qwen3.5-9B)
--resume           Resume from existing state.json
```

## Environment & Dependencies

### B200 cluster modules

```
gcc/14.2.0      — required (gcc/12.2.0 is too old, missing CXXABI_1.3.15)
cuda/12.8.1     — required for B200 GPU support
conda           — for environment management
```

### Pro 6000 cluster

No modules. Scripts install everything into a local uv venv
(`$SCAFFOLD_DIR/.venv_pro6000`). Only requirement is a working
CUDA 12.8+ driver and `python3` on PATH. `uv` is installed automatically
if missing.

### Python packages (shared)

| Package | Version | Notes |
|---|---|---|
| Python | 3.10+ | 3.10 tested, 3.13 used by autoresearch |
| `sglang[all]` | 0.5.10.post1 | LLM serving (replaces vllm due to flash-attn conflicts) |
| `openai` | >= 1.0.0 | Python client for OpenAI-compatible API |
| `uv` | latest | Package manager for autoresearch training deps |
| `flash-attn-4` | 4.0.0b4 | **B200 only**, required by SGLang's FA4 JIT kernel path on Hopper/GB100. Pro 6000 scripts skip this. |
| `torch` | 2.9.1 | Pinned for both targets (Blackwell support) |

### Autoresearch dependencies (managed by `uv sync`)

| Package | Notes |
|---|---|
| PyTorch 2.9.1 (cu128) | Installed from pytorch-cu128 index |
| flash-attn 4 (kernels) | FA4 for B200/Blackwell GPUs |
| tiktoken, rustbpe | Tokenizer |
| pyarrow | Data loading |

### Known issues

- **CuDNN check**: SGLang warns about PyTorch 2.9.1 + CuDNN < 9.15 compatibility. Safe to ignore for our use case (no Conv3d). Set `SGLANG_DISABLE_CUDNN_CHECK=1`.
- **flash-attn versions**: vllm's Qwen3.5 vision encoder imports `flash_attn.ops.triton.rotary` which was removed in FA4. This is why we use SGLang instead of vllm.
- **SGLang FA4 JIT kernel on Pro 6000**: SGLang's `sglang/jit_kernel/flash_attention_v4.py` imports `flash_attn.cute` at scheduler init. On Pro 6000 (SM 12.0) without `flash-attn-4` installed, this raises `ModuleNotFoundError` and the scheduler logs a traceback — **non-fatal**. `run_pro6000.sh` passes `--attention-backend triton` so SGLang serves via Triton kernels regardless, and the error is cosmetic log noise.
- **HuggingFace cache**: Model weights (~18GB) are cached at `$HF_HOME`. Set this to blue storage (B200) or a scratch filesystem (Pro 6000) to avoid filling home-directory quota.
- **Memory**: 64 GB system RAM required. 32 GB causes OOM during model weight loading.
- **`torch.compile` on SM 12.0**: PyTorch 2.9.1 has Blackwell support but Inductor had early edge cases for SM 12.0. If `adamw_step_fused` / `muon_step_fused` fail to compile on Pro 6000, comment out the `@torch.compile` decorators in `train_sdpa.py`.

### Environment variables (set in `run.sh`)

```bash
HF_HOME="${PROJ_DIR}/.cache/huggingface"     # model cache on blue storage
SGLANG_DISABLE_CUDNN_CHECK=1                 # skip CuDNN version warning
```

## Switching LLM backends

The scaffold talks to any OpenAI-compatible endpoint. SGLang is the default, but you can swap:

```bash
# SGLang (default, in run.sh)
python -m sglang.launch_server --model-path Qwen/Qwen3.5-9B --port 8000 ...

# vllm (commented out in run.sh)
vllm serve Qwen/Qwen3.5-9B --port 8000 --language-model-only ...

# ollama
ollama serve && ollama pull qwen3.5:9b
# endpoint: http://localhost:11434/v1
```
