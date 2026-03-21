# Plan: Add RL Training to Autoresearch Scaffold (TTT-Discover Style)

## Context

The scaffold currently uses a **frozen** Qwen3.5-9B via SGLang to propose experiments. The LLM never learns from outcomes — it sees only the last 10 results in its prompt. TTT-Discover (arxiv 2601.16175) showed that training LLM weights at test time via RL dramatically improves scientific discovery. We adapt their approach: collect batches of (proposal, reward) rollouts, compute entropic advantages with per-token KL penalty, and update LoRA weights via policy gradient.

## Modular Design Principle

**The frozen pipeline (`main.py`, `planner.py`, `state.py`, `results.py`, `run.sh`) is never modified.**

All RL code lives in new files. The RL mode is a separate entry point (`rl_main.py`) with its own launch script (`run_rl.sh`). Running `run.sh` works exactly as before.

### Key Design Decision: Single Local Model (No SGLang)

This is **test-time training** — the model that generates proposals is the same model being trained. After each RL step, the LoRA weights are updated and the next generation immediately uses the improved policy. A single in-process model guarantees this with zero sync overhead.

Why not SGLang or a hybrid approach:
- **SGLang would require two model copies** — one in the server, one for training. On a 9B model this fits in VRAM, but adds complexity for LoRA adapter syncing after every training step.
- **SGLang's `/load_lora_adapter` blocks requests** during adapter swap and doesn't support in-place weight updates.
- **Single model is simpler** — no adapter save/load/sync cycle. `optimizer.step()` updates the weights, next `model.generate()` uses them immediately.
- **Generation speed is not the bottleneck** — each rollout runs `train.py` for ~5 minutes. Proposal generation (~1-2 min with HF generate) is a small fraction of total step time.

The model uses `attn_implementation="sdpa"` by default. FA4 (`flash_attention_4`) is the ideal choice for B200 Blackwell but requires a transformers version with FA4 support (not yet in stable release). SDPA still dispatches to efficient CUDA kernels on B200. Switch to FA4 once transformers adds support.

### File Layout

```
Frozen pipeline (UNTOUCHED)        RL layer (new files)
─────────────────────────          ────────────────────
main.py                            rl_main.py        ← entry point, RL experiment loop
planner.py                         rl_planner.py     ← reuses planner.py prompt building, uses local model
state.py                           rl_model.py       ← HF model + PEFT LoRA + generation
results.py                         rl_trainer.py     ← entropic advantages, KL penalty, policy gradient
run.sh                             rl_sampler.py     ← PUCT tree search
                                   rl_types.py       ← Rollout dataclass
                                   run_rl.sh         ← launch script (NO SGLang)
```

### Import Graph (RL → frozen, never the reverse)

```
rl_main.py ──→ state.py (file I/O)
           ──→ results.py (run_experiment, parse_metrics, append_result)
           ──→ planner.py (build_planner_context, validate_planner_output, apply_edits, validate_edit_targets, preview_diff)
           ──→ rl_planner.py ──→ planner.py (prompt building)
                             ──→ rl_model.py (local generation with logprobs)
           ──→ rl_model.py (standalone: transformers + peft)
           ──→ rl_trainer.py (standalone: pure PyTorch)
           ──→ rl_sampler.py (standalone: numpy only)
           ──→ rl_types.py (standalone: dataclasses only)
```

## New Files (7)

### `rl_types.py` — Data structures

```python
@dataclass
class Rollout:
    prompt_text: str             # full prompt (after chat template)
    proposal_text: str           # raw LLM response
    full_ids: torch.Tensor       # full token sequence (prompt + response)
    old_logprobs: torch.Tensor   # per-token logprobs from generation
    prompt_len: int              # number of prompt tokens
    val_bpb: float | None
    status: str                  # keep/discard/crash/edit_failed
    reward: float
    description: str
```

### `rl_model.py` — Local model with LoRA

Single model instance for both generation and training.

- `load_model(model_dir, device, lora_rank, lora_alpha, attn_impl, lora_path) -> (model, tokenizer)`
  - `attn_implementation` defaults to `"sdpa"` (switch to `"flash_attention_4"` when transformers supports it)
  - If `lora_path` given, loads existing adapter; otherwise creates fresh LoRA
- `generate_with_logprobs(model, tokenizer, prompt, max_new_tokens, temperature) -> (text, full_ids, logprobs, prompt_len)`
  - `model.generate(output_logits=True)` returns raw logits (before temperature warper)
  - Extracts per-token logprobs, frees logit tensors immediately
  - Same model instance used for training — LoRA weights always up-to-date
- `compute_response_logprobs(model, full_ids, prompt_len, temperature) -> Tensor`
  - Forward pass with gradient (for backprop)
- `compute_base_logprobs(model, full_ids, prompt_len, temperature) -> Tensor`
  - `model.disable_adapter_layers()` → forward pass → `model.enable_adapter_layers()`
  - Returns base model logprobs for KL penalty

### `rl_planner.py` — Prompt building + local generation

Reuses `planner.py` for prompt assembly and validation, uses `rl_model.py` for generation:

- `propose_experiment_rl(model, tokenizer, agent_state, ...) -> (proposal, Rollout)`
  - Calls `planner.build_planner_context()` for prompt
  - Formats with `tokenizer.apply_chat_template()`
  - Generates via `rl_model.generate_with_logprobs()`
  - Parses JSON, validates via `planner.validate_planner_output()`

### `rl_trainer.py` — Entropic advantages + KL penalty + policy gradient

- `compute_entropic_advantages(rewards) -> Tensor` — adaptive beta binary search, LOO weighting
- `compute_reward(val_bpb, status) -> float` — `-val_bpb` for success, `-1.0` for failure
- `train_step(model, optimizer, rollouts, advantages, kl_coef, ...) -> dict`
  - Per-token KL penalty: `shaped_adv = adv - kl_coef * (new_lp - base_lp)`
  - Importance-sampled policy gradient: `loss = -(ratio * shaped_adv).mean()`
  - Returns metrics: avg_loss, num_tokens, kl_mean, ratio stats

### `rl_sampler.py` — PUCT tree search

- `State` class: stores code, value (-val_bpb), observation, parent lineage
- `PUCTSampler`: PUCT score = Q(i) + c * scale * P(i) * sqrt(1+T) / (1+n[i])
  - `sample_state()`, `update_state()`, `record_failed_rollout()`
  - JSON persistence with atomic writes

### `rl_main.py` — RL entry point

Single model serves both generation and training:
```
Setup:
  model, tokenizer = load_model(model_dir, device, lora_rank, lora_alpha)
  optimizer = AdamW(model trainable params, lr=lr)
  Run baseline, initialize PUCTSampler

For step in 1..num_steps:
  Phase 1: Collect batch_size rollouts
    parent = sampler.sample_state()              # PUCT selection
    Write parent.code to train.py
    proposal, rollout = propose_experiment_rl()   # generate with CURRENT LoRA weights
    Apply edits, run train.py, parse metrics
    Compute reward, update PUCT tree, log to results.tsv

  Phase 2: RL training step
    advantages = compute_entropic_advantages(rollout rewards)
    train_step(model, optimizer, rollouts, advantages, kl_coef)
    # LoRA weights updated — next generate() uses improved policy

  Phase 3: Checkpoint
    sampler.save(step), log metrics
```

### `run_rl.sh` — Launch script

No SGLang server. Model loads in-process.

## Existing Files — ZERO MODIFICATIONS

| File | Status |
|---|---|
| `main.py` | **Untouched** |
| `planner.py` | **Untouched** |
| `state.py` | **Untouched** |
| `results.py` | **Untouched** |
| `run.sh` | **Untouched** |

## VRAM Budget (B200 178 GB, single GPU)

Single model instance, no duplication:

| Phase | Model | train.py | Optimizer/Grads | Total |
|---|---|---|---|---|
| Generation | ~28 GB (model + KV cache) | — | — | ~28 GB |
| train.py eval | ~18 GB (model idle) | ~45 GB | — | ~63 GB |
| RL update | ~18 GB (model) | — | ~20 GB | ~38 GB |

All phases fit comfortably. Model stays loaded throughout.

## Dependencies

RL mode only (installed by `run_rl.sh`):
```
torch==2.9.1
transformers>=4.52.0
peft>=0.15.0
numpy>=2.2.6
accelerate
```

## Bug Fixes Applied

1. **Advantages/rollouts misalignment** — advantages now computed from filtered rollouts only, not all rewards
2. **RL results not logged** — `results.append_result()` called after each rollout so planner history stays current
3. **Baseline re-runs on resume** — guarded by `if args.resume_step is None`

## Design Decisions Log

1. **Single model vs hybrid (SGLang + local)** — Single model chosen because TTT requires the trained model to be the inference model. No sync overhead, simpler code.
2. **SDPA default, FA4 future** — B200 supports FA4 but transformers stable release doesn't yet. SDPA is the safe default; switch to FA4 when available.
3. **Local model over SGLang** — SGLang would need adapter sync after every training step. Blocks during loading. Adds complexity for marginal speed gain (generation is not the bottleneck).
4. **No Ray** — ttt_autoresearch uses Ray for multi-GPU eval. We're single-GPU, so removed.
