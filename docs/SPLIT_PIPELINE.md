# Split Pipeline: Ideator + Implementer

## Context

Current ERL pipeline uses a single LLM call per rollout that does both ideation ("what to change") and implementation ("produce JSON with search/replace edits"). RL reward + gradient flows through the entire combined generation, so:

- High-level idea quality and low-level JSON-formatting correctness are entangled in the same signal
- Implementation failures (`edit_failed` — malformed JSON, wrong search strings) contaminate idea-level reward
- Policy learns conservative HP tuning variants because those have both coherent ideas AND reliable JSON translation
- Observed failure mode: 2026-04-19 Pro 6000 GRPO run plateaued for 18 steps, policy stuck in HP-tuning attractor

## Goal

Decompose proposal into two sequential calls:

**Stage A — Ideator (LoRA-trained, receives RL signal):**
- Input: current `train.py` + history summary + (optional) reflection context
- Output: short natural-language change description, e.g. `"Reduce weight_decay on Muon matrix params from 0.2 to 0.1 to reduce regularization pressure during short training budget"`
- Rough size: ~100-300 tokens response
- This is the only stage whose logprobs participate in the GRPO loss and whose LoRA receives gradient updates

**Stage B — Implementer (frozen, no gradient):**
- Input: ideator description + current `train.py`
- Output: JSON with search/replace edits in existing format
- Runs under `torch.no_grad()`; logprobs not recorded
- Can be the same base model (cheaper) or a stronger code-specialist model (higher quality)

Reward from eval (`val_bpb` → reward) flows only through stage A's `full_ids` + `old_logprobs`. Stage B is treated as a deterministic translator.

## Expected benefits

1. **Smaller ideator action space** — natural language, no JSON syntax constraints. Less gradient spent on formatting.
2. **Decoupled failure modes** — implementation failures (`edit_failed`) can be handled separately from idea rewards. Options: treat as zero reward for idea, or retry stage B with different temperature before penalizing.
3. **Cleaner credit assignment** — RL signal targets ideation quality directly.
4. **Interpretable rollouts** — `results.tsv` can log the natural-language idea for human inspection without JSON clutter.
5. **Orthogonal to DDP/paper-parity** — works with any training config.

## Non-goals

- Training stage B (frozen for v1)
- Using different model families for stage A vs B (same Qwen3.5-9B base, can add later)
- Changing the phased ERL loop (attempt1 → reflection → attempt2 → train stays identical)

## Architecture

### Files to add

**`rl_pipeline/rl_planner.py` — split `propose_experiment_rl`**
- New function `propose_idea(model, tokenizer, agent_state, ...)` — generates natural-language idea, records logprobs (grad-tracked). Returns `(idea_text, full_ids, logprobs, prompt_len)`.
- New function `implement_idea(model, tokenizer, idea_text, current_code, ...)` — generates JSON edits under `no_grad`. Returns `(proposal_dict, _ignored_rollout_tensors)`.
- Keep existing `propose_experiment_rl` untouched for backward compat.
- New combined entry `propose_experiment_split(model, tokenizer, agent_state, ...)` that orchestrates A→B and returns a `Rollout` where `full_ids` / `old_logprobs` / `prompt_len` refer to stage A only.

**`rl_pipeline/prompt_ideator.md` and `rl_pipeline/prompt_implementer.md`**
- Ideator prompt: "Propose one focused change to train.py. Describe WHAT you want to change and WHY in 2-3 sentences. Do NOT produce JSON." + context.
- Implementer prompt: "Given this proposed change and the current train.py, produce the JSON edits that implement it. Output ONLY valid JSON." + idea + current code.
- Keep existing `prompt.md` for the monolithic path.

**`erl_pipeline/run_erl_pro6000_split.sh` (new)**
- Clone of `run_erl_pro6000.sh`
- Adds `--split-pipeline` flag
- Writes to `./erl_log_pro6000_split`

### Files to modify

**`erl_pipeline/erl_main.py`**
- New argparse flag: `--split-pipeline` (default False)
- In `generate_and_apply`, dispatch:
  - If `--split-pipeline`: call `propose_experiment_split(...)` instead of `propose_experiment_rl(...)`
  - Else: unchanged (current path)
- The returned `Rollout` has the same shape (`full_ids`, `old_logprobs`, etc.), so the trainer doesn't need changes.

**`rl_pipeline/rl_planner.py`**
- Add `propose_experiment_split` (~50 LOC)
- Add helper `_strip_idea_artifacts` for idea post-processing

### Files not touched

- `erl_trainer.py` — Rollout carries stage-A tensors; loss computation unchanged
- `rl_model.py` — generation primitives reused
- `erl_types.py` — Episode / StepReflection unchanged
- `rl_eval.py` — eval is downstream of implementer, unchanged
- All other launch scripts — keep default monolithic path

## Edge cases

1. **Stage B fails to produce valid JSON.** Options: retry once with different temperature/seed, then fall back to `edit_failed` status with zero reward on stage A. Decision: **retry once**, then zero reward. Log the implementer failure in rollouts.jsonl for analysis.

2. **Stage A idea is well-formed but impossible to implement** (e.g., "use adaptive depth"). Stage B fails → stage A learns this is bad. That's actually desired behavior.

3. **Think budget handling.** Stage A generates reasoning in a `<think>` block then the idea. Stage B is fast/mechanical — can run with smaller think budget (e.g., 512) or none.

4. **Context length.** Stage B receives idea + full train.py. train.py is ~400 lines (~3-4K tokens); stage A idea is short. Fits comfortably in 16K context.

5. **On-policy invariant.** Stage A's `old_logprobs` must be recomputed under the same processor-free path as the generation. Since stage A uses `generate_with_logprobs` from `rl_model.py` directly, this already holds.

## Implementation order

1. Write `propose_idea` + `implement_idea` + `propose_experiment_split` in `rl_planner.py`. Unit-test with mocks.
2. Add prompt files. Iterate on ideator wording (crucial — affects whole pipeline quality).
3. Wire `--split-pipeline` flag through `erl_main.py`.
4. Create `run_erl_pro6000_split.sh`.
5. Local mock test (no GPU): verify split path produces a valid `Rollout` with stage-A tensors.
6. Cluster smoke: 2-step run with `--split-pipeline --batch-size 2`.
7. Full run, compare against GRPO baseline and paper-parity baseline.

## Evaluation

A/B vs monolithic:
- Keep rate at matching step count
- `edit_failed` rate (should drop — stage B is specialized)
- Proposal diversity — measure overlap of ideas across rollouts within a step
- Time-to-first-structural-change — split pipeline should propose `hidden_dim` / `grad_accum` / other underexplored categories sooner if the ideator's cleaner signal breaks the HP-tuning attractor
- Plateau length — 2026-04-19 GRPO had 18-step end plateau; see if split shortens it

## Rollback

Don't pass `--split-pipeline` → defaults to existing monolithic generation. Zero change to today's runs.

## Open decisions (for implementation session)

1. **Implementer temperature.** 0.0 (deterministic) vs 0.3 (slight variation for retry robustness). Start 0.0.
2. **Implementer think budget.** 0 (no thinking — pure translation) vs 512 (light reasoning). Start 0.
3. **Retry policy on stage B failure.** 1 retry at temperature 0.7 vs immediate zero reward. Start 1 retry.
4. **Reward on stage-B-only failure.** Zero (our default) vs a small negative penalty (−0.1) to discourage stage A from producing unimplementable ideas.
5. **Ideator output format.** Free text vs structured (`CATEGORY: ... CHANGE: ... RATIONALE: ...`). Start free text for minimum intervention.
