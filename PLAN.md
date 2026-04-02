# Plan: Autoresearch Agent — RL + ERL Pipelines

## Overview

Three pipelines for automated train.py optimization, each a different baseline:

1. **Frozen pipeline** — LLM proposes edits via SGLang API, never learns from outcomes
2. **RL pipeline (TTT-Discover)** — LLM proposes edits, LoRA weights updated via policy gradient with entropic advantages + PUCT tree search
3. **ERL pipeline (Experiential RL)** — LLM proposes edits, reflects on batch results, retries with reflection, distills corrections back. GRPO advantages, no tree search.

---

## RL Pipeline (TTT-Discover)

### Status: Running on B200 (8 GPUs, 50 steps)

Parallel mode tested and submitted. Model on GPU 0, 7 eval workers on GPUs 1-7, batch_size=7, 50 steps.

### Design

- **Entropic LOO advantages** with adaptive beta
- **PUCT tree search** over code states (State stores full train.py, not git commits)
- **Per-token KL penalty** against base model (LoRA adapters disabled)
- **Single local model** — LoRA weights updated in-process, no server sync

### Key Files

```
rl_pipeline/
├── rl_main.py          ← entry point, sequential + parallel (Ray) modes
├── rl_model.py         ← HF model + PEFT LoRA + SDPA + gradient checkpointing
├── rl_planner.py       ← wraps planner.py, strips <think> tags, local generation
├── rl_trainer.py       ← entropic advantages, per-token KL penalty, policy gradient
├── rl_sampler.py       ← PUCT tree search
├── rl_eval.py          ← Ray parallel eval workers
├── rl_types.py         ← Rollout dataclass
├── run_rl.sh           ← SLURM 8-GPU launch script
├── smoke_test.py       ← GPU smoke test
└── debug_generate.py   ← debug: generate one proposal
```

### Rollout Flow

```
1. Parse failure (no </think>, bad JSON) → RETRY (no training signal)
2. Valid JSON but edits fail → reward=-1.0, counts as rollout
3. Edits apply, train.py crashes → reward=-1.0, counts as rollout
4. Edits apply, train.py succeeds → reward=-val_bpb, child added to PUCT
5. train.py always resets to parent code after every rollout
```

### CLI

```bash
python rl_main.py \
    --model-dir Qwen/Qwen3.5-9B --repo-path ./autoresearch_rl \
    --model-gpu 0 --eval-gpus 1,2,3,4,5,6,7 --workers-per-gpu 1 \
    --batch-size 7 --num-steps 50 \
    --lr 4e-5 --kl-coef 0.1 --puct-c 1.0 \
    --lora-rank 32 --lora-alpha 64 \
    --temperature 0.7 --max-new-tokens 8192 \
    --attn-impl sdpa --log-dir ./rl_log
```

### Bug Fixes Applied

1. Advantages/rollouts misalignment — filtered list only
2. RL results not logged — `append_result()` after each rollout
3. Baseline re-runs on resume — guarded by `resume_step is None`
4. LoRA not saved/loaded on resume
5. `ensure_results_tsv()` unconditional
6. `parent.value` truthiness — `is not None`
7. `<think>` tag stripping — split on `</think>`
8. OOM during backward — gradient checkpointing + `empty_cache()`
9. No-op edits crash git commit — skip if diff empty
10. `flash_attention_4` not supported — default `sdpa`
11. `torch_dtype` deprecated — changed to `dtype`
12. Failed proposals not retried — retry up to `batch_size * 2`
13. Ratio explosion (ratio_max=93 at step 0) — `old_logprobs` were extracted from `generate()` (autoregressive) but `new_logprobs` computed via KV-cache split forward pass. bfloat16+SDPA numerical divergence on rare tokens caused `exp(new-old)` to blow up. Fixed: recompute `old_logprobs` via `compute_response_logprobs()` (same path as training) so ratio=1.0 exactly for on-policy updates.
14. results.tsv status always "keep" — RL/ERL logged `status="keep"` for every successful run regardless of improvement. Fixed: compare against `best_bpb` to set `keep`/`discard` in logs.

---

## ERL Pipeline (Experiential RL)

### Status: Code Complete, Pending GPU Test

All files written and compile-checked. Feedback logic tested. Not yet run on GPU. Next step: 2-GPU smoke test, then full 8-GPU run.

### Design (arXiv:2602.13949 adaptation)

- **Always-reflect** — every step, not gated on failure (no binary success/fail in our task)
- **One reflection per batch** — model sees all attempt1 results, generates one reflection for all attempt2s
- **GRPO advantages** — `(reward - mean) / std` normalized within the group
- **No PUCT tree** — parent is always the current best code
- **Four training signals** — GRPO on attempt1, GRPO on reflection, GRPO on attempt2, RAFT distillation
- **Train the reflector** — reflection reward = mean(attempt2 rewards), baseline = mean(attempt1 rewards)
- **LoRA** — same as RL pipeline, easy to switch to full model later

### Key Differences from RL Pipeline

| Aspect | RL (TTT-Discover) | ERL |
|--------|-------------------|-----|
| Advantages | Entropic LOO (adaptive beta) | GRPO (normalized group) |
| Tree search | PUCT | None (always best code) |
| Reflection | None | Batch-level, one per step |
| Training signals | 1 (policy gradient) | 4 (attempt1, reflection, attempt2, distill) |
| Evals per step | batch_size | batch_size × 2 |
| Step time (est.) | ~15 min | ~25 min |

### Step Flow

```
Phase 1: Generate + eval batch_size first attempts (parallel, 7 GPUs)
Phase 2: Build batch feedback, generate ONE reflection
Phase 3: Generate + eval batch_size second attempts with shared reflection (parallel)
Phase 4: Build distillation targets (attempt2s that beat pre-step best_bpb)
Phase 5: Train (4 signals: GRPO × 3 + RAFT × 1)
Phase 6: Checkpoint (LoRA + step_log.json + best_train.py)
```

### Key Files

```
erl_pipeline/
├── erl_main.py         ← phased loop: all attempt1 → reflect → all attempt2 → train
├── erl_trainer.py      ← GRPO advantages, 4 training signals
├── erl_feedback.py     ← batch-level structured feedback
├── erl_reflect.py      ← one reflection per step, logprobs for training
├── erl_types.py        ← Episode + StepReflection dataclasses
├── run_erl.sh          ← SLURM 8-GPU launch script
└── (reuses rl_model.py, rl_eval.py, rl_planner.py, rl_types.py from rl_pipeline)
```

### CLI

```bash
python erl_main.py \
    --model-dir Qwen/Qwen3.5-9B --repo-path ./autoresearch_rl \
    --model-gpu 0 --eval-gpus 1,2,3,4,5,6,7 --workers-per-gpu 1 \
    --batch-size 7 --num-steps 50 \
    --lr 4e-5 --kl-coef 0.1 \
    --lora-rank 32 --lora-alpha 64 \
    --temperature 0.7 --max-new-tokens 8192 \
    --attn-impl sdpa --log-dir ./erl_log
```

### Bug Fixes Applied

1. Serial mode never evaluated experiments — added `run_eval_sequential()` fallback
2. Distillation threshold used post-mutation `best_bpb` — split Phase 4 into distill-first, then update
3. Resume left `step_log` undefined when `step_log.json` missing — added `step_log = []`
4. `build_distill_ids` used updated `best_val_bpb` — temporarily restore pre-step value
5. Reflection advantage baseline used filtered rollout list — changed to all episode rewards

---

## Shared Infrastructure

### Model & Hardware
- **Model**: Qwen/Qwen3.5-9B with LoRA (rank 32, alpha 64)
- **Cluster**: HiPerGator B200, 180GB VRAM per GPU, 8 GPUs per node
- **CUDA**: 12.8.1, GCC 14.2.0, SDPA attention (FA4 not yet available)
- **Each train.py**: ~74GB VRAM, ~5 min per run

### Reused Modules (never duplicated)
- `planner.py` — prompt building, edit validation/application
- `results.py` — metric parsing, results logging
- `state.py` — file I/O
- `rl_model.py` — model loading, generation with logprobs, training forward pass
- `rl_eval.py` — Ray parallel eval workers
- `rl_planner.py` — local generation wrapper
- `rl_types.py` — Rollout dataclass

### gpu_mem_limit/
LD_PRELOAD library to cap per-process GPU memory (mimics MIG). Written, not yet tested on cluster.

---

## Next Steps

1. **Check RL pipeline results** — 50-step run submitted, check `rl_log/step_log.json`
2. **Smoke test ERL** — 2 GPUs, 3 steps, batch_size=2
3. **Full ERL run** — `sbatch erl_pipeline/run_erl.sh`
4. **Compare RL vs ERL** — same model, same steps, compare best_bpb curves
5. **gpu_mem_limit testing** — run `test_memlimit.sh` on cluster, then integrate into run scripts for multi-worker-per-GPU
6. **Structured reflection template** — refine `REFLECTION_SYSTEM` prompt with domain-specific structure (currently placeholder)
