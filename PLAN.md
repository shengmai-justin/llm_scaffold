# Plan: Add RL Training to Autoresearch Scaffold (TTT-Discover Style)

## Context

The scaffold currently uses a **frozen** Qwen3.5-9B via SGLang to propose experiments. The LLM never learns from outcomes — it sees only the last 10 results in its prompt. TTT-Discover (arxiv 2601.16175) showed that training LLM weights at test time via RL dramatically improves scientific discovery. We adapt their approach: collect batches of (proposal, reward) rollouts, compute entropic advantages with per-token KL penalty, and update LoRA weights via policy gradient.

## Status: Working End-to-End

The RL pipeline is implemented and tested on B200. Single-GPU sequential mode runs successfully: baseline → proposals → train.py eval → RL update → checkpoint.

## Modular Design

**The frozen pipeline (`main.py`, `planner.py`, `state.py`, `results.py`, `run.sh`) is untouched** (except `main.py` which got a no-op edit skip fix).

All RL code lives in `rl_pipeline/`:

```
llm_scaffold/
├── main.py, planner.py, state.py, results.py, run.sh   ← frozen pipeline
└── rl_pipeline/
    ├── rl_main.py          ← entry point, RL experiment loop
    ├── rl_model.py         ← HF model + PEFT LoRA + SDPA + gradient checkpointing
    ├── rl_planner.py       ← wraps planner.py, strips <think> tags, local generation
    ├── rl_trainer.py       ← entropic advantages, per-token KL penalty, policy gradient
    ├── rl_sampler.py       ← PUCT tree search (State stores code, not git commits)
    ├── rl_types.py         ← Rollout dataclass
    ├── smoke_test.py       ← GPU smoke test (8 components)
    ├── debug_generate.py   ← debug: generate one proposal, print raw output
    ├── run_rl.sh           ← SLURM launch script (no SGLang)
    └── mps.md              ← multi-GPU parallelization plan (Ray)
```

## Key Design Decisions

1. **Single local model (no SGLang)** — TTT requires the trained model to be the inference model. `optimizer.step()` updates LoRA weights, next `model.generate()` uses them immediately. No sync needed.

2. **SDPA attention** — B200 supports FA4 but `transformers` stable doesn't yet. SDPA dispatches to efficient CUDA kernels. Switch to FA4 when available.

3. **Gradient checkpointing** — enabled to prevent OOM during RL backward pass on long sequences (~10K prompt + ~700 response tokens through a 9B model).

4. **PUCT stores code, not git commits** — `State.code` holds the full train.py text. No git operations in RL mode. PUCT tree persisted as JSON.

5. **Retry on parse failures** — if the model fails to produce valid JSON (e.g., thinking exhausts max_new_tokens), retry up to `batch_size * 2` attempts. Edit failures with valid logprobs are NOT retried — they provide useful negative reward signal for RL.

## Rollout Flow

```
1. Proposal parse fails (no </think>, bad JSON)
   → full_ids empty → RETRY (no training signal)

2. Proposal parses but edits fail (search string not found)
   → full_ids has data, status="edit_failed", reward=-1.0
   → counts as valid rollout (useful RL signal, no retry)

3. Edits apply but train.py crashes/times out
   → status="crash", reward=-1.0
   → counts as valid rollout

4. Edits apply and train.py succeeds
   → status="keep", reward=-val_bpb
   → child State added to PUCT tree

5. train.py always resets to parent code after every rollout
6. PUCT stores edited code in State.code (not git)
7. RL training only on rollouts with valid full_ids
```

## RL Training Step

```python
# For each rollout with nonzero advantage:
new_lp = compute_response_logprobs(model, full_ids, prompt_len)  # with grad
ratio = exp(new_lp - old_lp)                                     # importance sampling

# Per-token KL penalty (not per-rollout):
base_lp = compute_base_logprobs(model, full_ids, prompt_len)     # adapters disabled
shaped_adv = adv - kl_coef * (new_lp - base_lp)

loss = -(ratio * shaped_adv).mean()
loss.backward()
# Gradient accumulation across rollouts, then clip + step
```

Ported from `ttt_autoresearch/train.py:337-391`.

## Qwen3.5 Thinking Mode

Qwen3.5 outputs `<think>...</think>` before the JSON answer. The prompt template includes the opening `<think>` tag, so the model response starts inside the thinking block. `rl_planner.py` strips thinking by splitting on `</think>` and taking everything after it.

If the model exhausts `max_new_tokens` during thinking (no `</think>` in output), the parse fails and the rollout is retried.

## Bug Fixes Applied

1. **Advantages/rollouts misalignment** — advantages computed from `[r.reward for r in rollouts]` (filtered list only)
2. **RL results not logged** — `results.append_result()` after each rollout
3. **Baseline re-runs on resume** — guarded by `if args.resume_step is None`
4. **LoRA not saved/loaded on resume** — `model.save_pretrained()` each step, `PeftModel.from_pretrained()` on resume
5. **`ensure_results_tsv()` unconditional** — runs on both fresh and resume paths
6. **`parent.value` truthiness** — changed to `is not None`
7. **`<think>` tag stripping** — split on `</think>` (opening tag is in prompt, not response)
8. **OOM during RL backward** — gradient checkpointing + `torch.cuda.empty_cache()`
9. **No-op edits crash git commit** — skip commit if diff is empty (frozen pipeline fix)
10. **`flash_attention_4` not supported** — default changed to `sdpa`
11. **`torch_dtype` deprecated** — changed to `dtype`
12. **Failed proposals not retried** — retry loop fills batch up to `batch_size * 2` attempts

## CLI Reference

```bash
python rl_main.py \
    --model-dir Qwen/Qwen3.5-9B \
    --repo-path ./autoresearch_rl \
    --num-steps 50 \
    --batch-size 4 \
    --lr 4e-5 \
    --kl-coef 0.1 \
    --puct-c 1.0 \
    --lora-rank 32 \
    --lora-alpha 64 \
    --temperature 1.0 \
    --max-new-tokens 4096 \
    --max-grad-norm 1.0 \
    --attn-impl sdpa \
    --log-dir ./rl_log \
    --resume-step N           # resume from checkpoint
```

## Logs & Checkpoints

| File | Content |
|---|---|
| `rl_log/rollouts.jsonl` | Per-rollout: step, val_bpb, reward, status, description |
| `rl_log/step_log.json` | Per-step: best_bpb, rewards, loss, KL, ratio stats, timing |
| `rl_log/puct_step_XXXXXX.json` | PUCT tree checkpoint |
| `rl_log/lora_step_XXXXXX/` | LoRA adapter checkpoint |
| `rl_log/best_train.py` | Best code found so far |
| `results.tsv` | Shared with frozen pipeline format |

## Next: Multi-GPU Parallelization

See `rl_pipeline/mps.md`. Architecture: 1 GPU for model (generation + RL training), 7 GPUs for parallel train.py eval via Ray workers. Cuts step time from ~42 min to ~12 min for batch_size=7.
