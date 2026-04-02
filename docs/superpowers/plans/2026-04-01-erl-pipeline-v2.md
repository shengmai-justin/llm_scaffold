# ERL Pipeline v2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rewrite `erl_pipeline/` to closely match the Microsoft ERL paper (arXiv:2602.13949): always-reflect, one batch-level reflection per step, GRPO advantages, train the reflector, RAFT distillation, no PUCT tree search.

**Architecture:** Each step runs batch_size first attempts in parallel, builds batch-level feedback from all results, generates ONE reflection, then runs batch_size second attempts using that shared reflection. Four training signals: GRPO on attempt1, GRPO on reflection (rewarded by mean attempt2 outcome), GRPO on attempt2, RAFT on successful distillation targets. No PUCT — parent is always the current best code.

**Tech Stack:** Python 3.10+, PyTorch, PEFT/LoRA, Ray (parallel eval), HuggingFace Transformers. Reuses `rl_pipeline/rl_model.py`, `rl_pipeline/rl_eval.py`, `rl_pipeline/rl_planner.py`, `rl_pipeline/rl_types.py`, `planner.py`, `results.py`, `state.py` via imports.

---

## Differences from v1

| Aspect | v1 (current) | v2 (this plan) |
|--------|-------------|----------------|
| Reflection gating | Only on failure | Always — every step |
| Reflection scope | Per-episode (one per rollout) | Per-batch (one per step, sees all attempt1s) |
| Advantages | Entropic LOO (from TTT-Discover) | GRPO: `(r - mean) / std` |
| Train reflector | No | Yes — GRPO on reflection tokens |
| PUCT tree | Yes | No — parent is always current best code |
| Step flow | Per-episode sequential | Phased: all attempt1 → reflect → all attempt2 |

## File Structure

All files are rewrites of existing `erl_pipeline/` files. No new files.

```
erl_pipeline/
├── __init__.py          — unchanged (empty)
├── erl_types.py         — REWRITE: Episode + StepReflection, drop PUCT references
├── erl_feedback.py      — REWRITE: batch-level feedback builder, drop should_reflect
├── erl_reflect.py       — REWRITE: batch-level reflection, structured template
├── erl_trainer.py       — REWRITE: GRPO advantages, 4 training signals incl. reflector
├── erl_main.py          — REWRITE: phased loop, no PUCT, always-reflect
└── run_erl.sh           — MINOR UPDATE: remove --puct-c arg
```

**Reused from rl_pipeline (unchanged, imported):**
- `rl_model.py` — load_model, generate_with_logprobs, compute_response_logprobs, compute_base_logprobs
- `rl_eval.py` — EvalWorker (Ray)
- `rl_planner.py` — propose_experiment_rl
- `rl_types.py` — Rollout

**Reused from frozen pipeline (unchanged, imported):**
- `planner.py` — build_planner_context, validate_edit_targets, apply_edits, preview_diff
- `results.py` — run_experiment, parse_metrics, append_result, ensure_results_tsv
- `state.py` — read_file, write_file

---

### Task 1: Data Types

**Files:**
- Rewrite: `erl_pipeline/erl_types.py`

- [ ] **Step 1: Write erl_types.py**

```python
"""Data structures for ERL training loop."""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class Episode:
    """One ERL episode: first attempt + second attempt (after batch reflection).

    Training signals (all GRPO unless noted):
    - attempt1: GRPO (advantage from attempt1 rewards within batch)
    - attempt2: GRPO (advantage from attempt2 rewards within batch)
    - distillation: RAFT (supervised NLL on successful attempt2, original prompt)
    """
    # First attempt
    attempt1_rollout: "Rollout"
    attempt1_proposal: dict | None
    attempt1_edited_code: str | None
    attempt1_eval: dict | None

    # Second attempt (always present — reflection is not gated)
    attempt2_rollout: "Rollout | None" = None
    attempt2_proposal: dict | None = None
    attempt2_edited_code: str | None = None
    attempt2_eval: dict | None = None

    # Distillation target (attempt2 response with original prompt, no reflection)
    distill_full_ids: torch.Tensor | None = None
    distill_logprobs: torch.Tensor | None = None
    distill_prompt_len: int = 0

    # Which training signals are active
    train_attempt1: bool = True
    train_attempt2: bool = False
    train_distill: bool = False


@dataclass
class StepReflection:
    """One reflection per step, shared across all episodes.

    The reflection sees batch-level feedback (all attempt1 results)
    and generates one reflection used by all attempt2s.
    Trained with GRPO using mean(attempt2 rewards) as its reward.
    """
    feedback_text: str
    reflection_text: str
    full_ids: torch.Tensor
    old_logprobs: torch.Tensor
    prompt_len: int
    reward: float = 0.0
```

- [ ] **Step 2: Verify syntax**

Run: `cd /Users/chenshengmai/Desktop/autosearch_rl/llm_scaffold && python -m py_compile erl_pipeline/erl_types.py && echo OK`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add erl_pipeline/erl_types.py
git commit -m "feat(erl-v2): rewrite data types — Episode + StepReflection, drop PUCT"
```

---

### Task 2: Batch Feedback Builder

**Files:**
- Rewrite: `erl_pipeline/erl_feedback.py`

Replaces per-episode feedback with batch-level feedback. Removes `should_reflect` (always-reflect now).

- [ ] **Step 1: Write erl_feedback.py**

```python
"""Batch-level structured feedback from attempt1 results.

Builds a single feedback string summarizing ALL first attempts in a step,
used as input to the one-per-step reflection.
"""
from __future__ import annotations


def build_attempt_feedback(
    description: str,
    status: str,
    val_bpb: float | None,
    best_val_bpb: float,
    eval_output: str | None = None,
    edit_error: str | None = None,
) -> str:
    """Build feedback for a single attempt."""
    lines = [f"Experiment: {description}"]

    if status == "edit_failed":
        lines.append("Result: EDIT FAILED — search/replace could not be applied.")
        if edit_error:
            lines.append(f"Error: {edit_error}")
    elif status == "crash":
        lines.append("Result: CRASH — train.py failed to produce val_bpb.")
        if eval_output:
            lines.append(f"Last output:\n{eval_output.strip()[-300:]}")
    elif status == "timeout":
        lines.append("Result: TIMEOUT — exceeded time budget.")
    elif val_bpb is not None:
        delta = val_bpb - best_val_bpb
        sign = "+" if delta >= 0 else ""
        lines.append(f"Result: val_bpb={val_bpb:.6f} (best={best_val_bpb:.6f}, delta={sign}{delta:.6f})")
        if delta < 0:
            lines.append("IMPROVED over current best.")
        elif delta == 0:
            lines.append("No change from current best.")
        else:
            lines.append("REGRESSED from current best.")
    else:
        lines.append("Result: no metrics produced.")

    return "\n".join(lines)


def build_batch_feedback(
    attempts: list[dict],
    best_val_bpb: float,
) -> str:
    """Build batch-level feedback from all attempt1 results in a step.

    Args:
        attempts: list of dicts with keys:
            description, status, val_bpb, eval_output (optional), edit_error (optional)
        best_val_bpb: current best val_bpb

    Returns:
        Structured feedback string for the reflection prompt.
    """
    n_total = len(attempts)
    n_improved = sum(1 for a in attempts if a["val_bpb"] is not None and a["val_bpb"] < best_val_bpb)
    n_crashed = sum(1 for a in attempts if a["status"] in ("crash", "timeout", "edit_failed"))

    header = (
        f"## Batch Summary\n"
        f"- Total attempts: {n_total}\n"
        f"- Improved: {n_improved}\n"
        f"- Failed (crash/timeout/edit_failed): {n_crashed}\n"
        f"- Current best val_bpb: {best_val_bpb:.6f}\n"
    )

    sections = []
    for i, a in enumerate(attempts):
        fb = build_attempt_feedback(
            description=a["description"],
            status=a["status"],
            val_bpb=a["val_bpb"],
            best_val_bpb=best_val_bpb,
            eval_output=a.get("eval_output"),
            edit_error=a.get("edit_error"),
        )
        sections.append(f"### Attempt {i+1}\n{fb}")

    return header + "\n" + "\n\n".join(sections)
```

- [ ] **Step 2: Verify syntax + logic**

Run:
```bash
cd /Users/chenshengmai/Desktop/autosearch_rl/llm_scaffold && python -c "
from erl_pipeline.erl_feedback import build_batch_feedback, build_attempt_feedback

fb = build_attempt_feedback('Increase LR', 'crash', None, 0.989, eval_output='OOM')
assert 'CRASH' in fb

attempts = [
    {'description': 'Increase LR', 'status': 'crash', 'val_bpb': None},
    {'description': 'Add warmup', 'status': 'keep', 'val_bpb': 0.988},
    {'description': 'Change depth', 'status': 'keep', 'val_bpb': 0.991},
]
batch_fb = build_batch_feedback(attempts, 0.989)
assert 'Total attempts: 3' in batch_fb
assert 'Improved: 1' in batch_fb
assert 'Failed (crash/timeout/edit_failed): 1' in batch_fb
assert 'Attempt 1' in batch_fb
assert 'Attempt 2' in batch_fb
assert 'Attempt 3' in batch_fb
print('All tests pass')
"
```
Expected: `All tests pass`

- [ ] **Step 3: Commit**

```bash
git add erl_pipeline/erl_feedback.py
git commit -m "feat(erl-v2): batch-level feedback builder for all attempt1 results"
```

---

### Task 3: Batch Reflection Generator

**Files:**
- Rewrite: `erl_pipeline/erl_reflect.py`

One reflection per step. Sees all attempt1 results via batch feedback. Returns structured output. Collects logprobs for training.

- [ ] **Step 1: Write erl_reflect.py**

```python
"""Batch-level reflection generation.

One reflection per step. The model sees all attempt1 results and generates
a structured analysis used by all attempt2s. Logprobs are collected so
the reflector can be trained with GRPO.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "rl_pipeline"))

import torch
from rl_model import generate_with_logprobs


# Structured reflection template — placeholder sections for now,
# to be refined with domain-specific structure later.
REFLECTION_SYSTEM = """You are an ML researcher analyzing a batch of experiments on a GPT training script.

You are given the results of multiple experiments that modified train.py.
Analyze ALL results together to identify patterns, then produce a structured reflection.

Your reflection must contain:
1. PATTERNS: What patterns do you see across the batch? Which directions helped vs hurt?
2. DIAGNOSIS: For failed experiments, why did they fail?
3. STRATEGY: Based on all results, what specific change should be tried next?

Be concise (under 300 words). Focus on actionable insights, not summaries.
Output plain text only. No JSON, no code fences."""


def generate_batch_reflection(
    model,
    tokenizer,
    train_py: str,
    batch_feedback: str,
    best_val_bpb: float,
    temperature: float = 0.7,
    max_new_tokens: int = 1024,
) -> tuple[str, torch.Tensor, torch.Tensor, int]:
    """Generate one reflection for the entire batch of attempt1 results.

    Args:
        train_py: current parent train.py code
        batch_feedback: output of build_batch_feedback() — all attempt1 results
        best_val_bpb: current best metric

    Returns:
        (reflection_text, full_ids, logprobs, prompt_len)
    """
    user_msg = (
        f"Current train.py (first 200 lines):\n"
        f"```python\n{_abbreviate(train_py, 200)}\n```\n\n"
        f"Current best val_bpb: {best_val_bpb:.6f}\n\n"
        f"{batch_feedback}\n\n"
        f"Analyze all attempts and produce your reflection."
    )

    messages = [
        {"role": "system", "content": REFLECTION_SYSTEM},
        {"role": "user", "content": user_msg},
    ]
    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    text, full_ids, logprobs, prompt_len = generate_with_logprobs(
        model, tokenizer, prompt_text,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )

    # Strip thinking tags if present
    clean = text
    if "</think>" in clean:
        clean = clean.split("</think>", 1)[1].strip()

    return clean, full_ids, logprobs, prompt_len


def build_reflection_context(batch_feedback: str, reflection_text: str) -> str:
    """Build the context string appended to attempt2 proposal prompts.

    This is passed as error_context to propose_experiment_rl, so the model
    sees the batch results + reflection before proposing its second attempt.
    """
    return (
        f"--- Batch results from first attempts ---\n"
        f"{batch_feedback}\n\n"
        f"--- Your reflection ---\n"
        f"{reflection_text}\n\n"
        f"Based on the batch results and your reflection, propose an improved experiment."
    )


def _abbreviate(text: str, max_lines: int) -> str:
    """Truncate text to max_lines for prompt efficiency."""
    lines = text.split("\n")
    if len(lines) <= max_lines:
        return text
    return "\n".join(lines[:max_lines]) + f"\n... ({len(lines) - max_lines} more lines)"
```

- [ ] **Step 2: Verify syntax**

Run: `cd /Users/chenshengmai/Desktop/autosearch_rl/llm_scaffold && python -m py_compile erl_pipeline/erl_reflect.py && echo OK`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add erl_pipeline/erl_reflect.py
git commit -m "feat(erl-v2): batch-level reflection generation with logprob collection"
```

---

### Task 4: ERL Trainer with GRPO + Reflector Training

**Files:**
- Rewrite: `erl_pipeline/erl_trainer.py`

Four training signals: GRPO on attempt1, GRPO on reflection, GRPO on attempt2, RAFT on distillation. Advantages computed as normalized group: `(r - mean) / std`.

- [ ] **Step 1: Write erl_trainer.py**

```python
"""ERL training step: GRPO for attempts + reflector, RAFT for distillation.

Four training signals per step:
1. Attempt1 rollouts  -> GRPO
2. Reflection         -> GRPO (reward = mean of attempt2 rewards)
3. Attempt2 rollouts  -> GRPO
4. Distillation       -> RAFT (supervised NLL on successful corrections)
"""
from __future__ import annotations

import torch
from torch.nn.utils import clip_grad_norm_

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "rl_pipeline"))

from rl_model import compute_response_logprobs, compute_base_logprobs
from rl_types import Rollout
from erl_types import Episode, StepReflection


def compute_grpo_advantages(rewards: list[float]) -> torch.Tensor:
    """GRPO advantages: normalized within the group.

    advantage_i = (reward_i - mean(rewards)) / std(rewards)
    Returns zeros if std is zero (all rewards identical).
    """
    r = torch.tensor(rewards, dtype=torch.float32)
    if len(r) < 2:
        return torch.zeros_like(r)
    mean = r.mean()
    std = r.std()
    if std < 1e-8:
        return torch.zeros_like(r)
    return (r - mean) / std


def erl_train_step(
    model,
    optimizer,
    episodes: list[Episode],
    reflection: StepReflection | None,
    kl_coef: float = 0.1,
    temperature: float = 1.0,
    max_grad_norm: float = 1.0,
) -> dict:
    """One ERL training step with four signals.

    Advantages are computed internally via GRPO (not passed in).
    """
    optimizer.zero_grad()
    total_grpo_loss = 0.0
    total_distill_loss = 0.0
    total_reflect_loss = 0.0
    num_grpo_tokens = 0
    num_distill_tokens = 0
    num_reflect_tokens = 0
    all_ratios = []
    all_kls = []

    # --- Signal 1: Attempt1 rollouts (GRPO) ---
    a1_rollouts = [ep.attempt1_rollout for ep in episodes if ep.train_attempt1 and ep.attempt1_rollout.full_ids.numel() > 0]
    if len(a1_rollouts) >= 2:
        a1_advs = compute_grpo_advantages([r.reward for r in a1_rollouts])
        for rollout, adv in zip(a1_rollouts, a1_advs):
            if abs(adv.item()) < 1e-8:
                continue
            m = _grpo_loss(model, rollout, adv.item(), kl_coef, temperature, all_ratios, all_kls)
            total_grpo_loss += m["loss"] * m["tokens"]
            num_grpo_tokens += m["tokens"]

    # --- Signal 2: Reflection (GRPO, reward = mean attempt2 reward) ---
    if reflection is not None and reflection.full_ids.numel() > 0 and abs(reflection.reward) > 1e-8:
        # Reflection advantage: compare its reward to attempt1 mean reward
        a1_mean = sum(r.reward for r in a1_rollouts) / max(len(a1_rollouts), 1) if a1_rollouts else 0.0
        ref_adv = reflection.reward - a1_mean
        if abs(ref_adv) > 1e-8:
            m = _grpo_loss_from_tensors(
                model, reflection.full_ids, reflection.old_logprobs,
                reflection.prompt_len, ref_adv, kl_coef, temperature,
                all_ratios, all_kls,
            )
            total_reflect_loss += m["loss"] * m["tokens"]
            num_reflect_tokens += m["tokens"]

    # --- Signal 3: Attempt2 rollouts (GRPO) ---
    a2_rollouts = [ep.attempt2_rollout for ep in episodes if ep.train_attempt2 and ep.attempt2_rollout is not None and ep.attempt2_rollout.full_ids.numel() > 0]
    if len(a2_rollouts) >= 2:
        a2_advs = compute_grpo_advantages([r.reward for r in a2_rollouts])
        for rollout, adv in zip(a2_rollouts, a2_advs):
            if abs(adv.item()) < 1e-8:
                continue
            m = _grpo_loss(model, rollout, adv.item(), kl_coef, temperature, all_ratios, all_kls)
            total_grpo_loss += m["loss"] * m["tokens"]
            num_grpo_tokens += m["tokens"]

    # --- Signal 4: Distillation (RAFT — supervised NLL) ---
    for ep in episodes:
        if ep.train_distill and ep.distill_full_ids is not None and ep.distill_full_ids.numel() > 0:
            dl = _distill_loss(model, ep.distill_full_ids, ep.distill_prompt_len, temperature)
            total_distill_loss += dl["loss"] * dl["tokens"]
            num_distill_tokens += dl["tokens"]

    total_tokens = num_grpo_tokens + num_distill_tokens + num_reflect_tokens
    if total_tokens > 0:
        clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad],
            max_grad_norm,
        )
        optimizer.step()

    metrics = {
        "avg_grpo_loss": total_grpo_loss / max(num_grpo_tokens, 1),
        "avg_distill_loss": total_distill_loss / max(num_distill_tokens, 1),
        "avg_reflect_loss": total_reflect_loss / max(num_reflect_tokens, 1),
        "num_grpo_tokens": num_grpo_tokens,
        "num_distill_tokens": num_distill_tokens,
        "num_reflect_tokens": num_reflect_tokens,
    }
    if all_ratios:
        cat = torch.cat(all_ratios)
        metrics["ratio_mean"] = cat.mean().item()
        metrics["ratio_max"] = cat.max().item()
    if all_kls:
        metrics["kl_mean"] = sum(all_kls) / len(all_kls)
    return metrics


def _grpo_loss(
    model, rollout: Rollout, advantage: float,
    kl_coef: float, temperature: float,
    all_ratios: list, all_kls: list,
) -> dict:
    """GRPO policy gradient loss for one rollout. Calls backward."""
    return _grpo_loss_from_tensors(
        model, rollout.full_ids, rollout.old_logprobs,
        rollout.prompt_len, advantage, kl_coef, temperature,
        all_ratios, all_kls,
    )


def _grpo_loss_from_tensors(
    model, full_ids: torch.Tensor, old_logprobs: torch.Tensor,
    prompt_len: int, advantage: float,
    kl_coef: float, temperature: float,
    all_ratios: list, all_kls: list,
) -> dict:
    """GRPO loss from raw tensors (used for both rollouts and reflection)."""
    old_lp = old_logprobs.to(model.device)
    new_lp = compute_response_logprobs(model, full_ids, prompt_len, temperature=temperature)

    ratio = torch.exp(new_lp - old_lp)
    all_ratios.append(ratio.detach().cpu())

    shaped_adv = torch.tensor(advantage, device=model.device)
    if kl_coef > 0:
        base_lp = compute_base_logprobs(model, full_ids, prompt_len, temperature=temperature)
        kl_per_token = new_lp - base_lp
        all_kls.append(kl_per_token.mean().item())
        shaped_adv = shaped_adv - kl_coef * kl_per_token

    loss = -(ratio * shaped_adv).mean()
    loss.backward()
    return {"loss": loss.item(), "tokens": len(new_lp)}


def _distill_loss(
    model, full_ids: torch.Tensor, prompt_len: int, temperature: float,
) -> dict:
    """Supervised NLL loss for distillation."""
    new_lp = compute_response_logprobs(model, full_ids, prompt_len, temperature=temperature)
    loss = -new_lp.mean()
    loss.backward()
    return {"loss": loss.item(), "tokens": len(new_lp)}
```

- [ ] **Step 2: Verify syntax**

Run: `cd /Users/chenshengmai/Desktop/autosearch_rl/llm_scaffold && python -m py_compile erl_pipeline/erl_trainer.py && echo OK`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add erl_pipeline/erl_trainer.py
git commit -m "feat(erl-v2): GRPO trainer with 4 signals — attempt1, reflection, attempt2, distill"
```

---

### Task 5: ERL Main Loop (Phased, No PUCT)

**Files:**
- Rewrite: `erl_pipeline/erl_main.py`

Phased step flow:
1. Phase 1: Generate + eval all attempt1s (parallel via Ray)
2. Phase 2: Build batch feedback, generate ONE reflection
3. Phase 3: Generate + eval all attempt2s using shared reflection (parallel via Ray)
4. Phase 4: Build distillation targets for successful attempt2s
5. Phase 5: Train (4 signals)
6. Checkpoint

No PUCT. Parent = current best code. Track best code directly.

- [ ] **Step 1: Write erl_main.py**

```python
"""ERL experiment loop entry point (v2).

Phased step: all attempt1 -> batch reflection -> all attempt2 -> train.
No PUCT tree search. Parent is always the current best code.
Matches Microsoft ERL paper (arXiv:2602.13949) design.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

PIPELINE_DIR = os.path.dirname(os.path.abspath(__file__))
SCAFFOLD_DIR = os.path.dirname(PIPELINE_DIR)
RL_PIPELINE_DIR = os.path.join(SCAFFOLD_DIR, "rl_pipeline")
sys.path.insert(0, SCAFFOLD_DIR)
sys.path.insert(0, RL_PIPELINE_DIR)

import torch

import planner
import results
import state as state_mod
from rl_model import load_model
from rl_planner import propose_experiment_rl
from rl_trainer import compute_reward
from rl_types import Rollout

from erl_types import Episode, StepReflection
from erl_feedback import build_batch_feedback
from erl_reflect import generate_batch_reflection, build_reflection_context
from erl_trainer import erl_train_step

TRAIN_TIMEOUT = 600


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def generate_and_apply(
    model, tokenizer, agent_state, parent_code, train_path,
    temperature, max_new_tokens, error_context=None,
):
    """Generate proposal, apply edits. Returns (rollout, edited_code, proposal)."""
    state_mod.write_file(train_path, parent_code)

    try:
        proposal, rollout = propose_experiment_rl(
            model, tokenizer, agent_state,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            error_context=error_context,
        )
    except Exception as e:
        print(f"  Proposal failed: {e}")
        rollout = Rollout(
            prompt_text="", proposal_text="", full_ids=torch.tensor([]),
            old_logprobs=torch.tensor([]), prompt_len=0,
            val_bpb=None, status="edit_failed", reward=-1.0,
            description=f"proposal_error: {e}",
        )
        return rollout, None, None

    print(f"  >> {proposal['description']}  (risk: {proposal['risk']})")

    original_text = parent_code
    missing = planner.validate_edit_targets(train_path, proposal["edits"])
    if missing:
        print(f"  Edit failed: search strings not found")
        state_mod.write_file(train_path, original_text)
        rollout.status = "edit_failed"
        rollout.reward = compute_reward(None, "edit_failed")
        return rollout, None, proposal

    try:
        new_text = planner.apply_edits(train_path, proposal["edits"])
        diff = planner.preview_diff(original_text, new_text)
        if diff:
            print(diff)
    except ValueError as e:
        print(f"  Apply failed: {e}")
        state_mod.write_file(train_path, original_text)
        rollout.status = "edit_failed"
        rollout.reward = compute_reward(None, "edit_failed")
        return rollout, None, proposal

    if not diff:
        state_mod.write_file(train_path, original_text)
        rollout.status = "edit_failed"
        rollout.reward = compute_reward(None, "edit_failed")
        return rollout, None, proposal

    edited_code = state_mod.read_file(train_path)
    state_mod.write_file(train_path, original_text)
    return rollout, edited_code, proposal


def dispatch_eval(workers, worker_idx, parent_code, edited_code, step):
    """Dispatch eval to Ray worker, return future ref."""
    import ray
    worker = workers[worker_idx % len(workers)]
    return worker.evaluate.remote(parent_code, edited_code, step)


def collect_eval(ref, rollout):
    """Collect eval result from Ray future, update rollout."""
    import ray
    result = ray.get(ref)
    rollout.val_bpb = result["val_bpb"]
    rollout.status = "keep" if result["success"] else "crash"
    rollout.reward = compute_reward(rollout.val_bpb, rollout.status)
    return result


def build_distill_ids(model, tokenizer, agent_state, attempt2_text, temperature):
    """Build distillation target: attempt2 response with original prompt (no reflection)."""
    system_msg, user_msg = planner.build_planner_context(
        agent_state["repo_path"], agent_state["best_val_bpb"]
    )
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]
    original_prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    full_text = original_prompt + attempt2_text
    tokens = tokenizer(full_text, return_tensors="pt")
    full_ids = tokens["input_ids"][0].cpu()
    prompt_tokens = tokenizer(original_prompt, return_tensors="pt")
    prompt_len = prompt_tokens["input_ids"].shape[1]

    with torch.no_grad():
        from rl_model import compute_response_logprobs as _compute_lp
        logprobs = _compute_lp(model, full_ids, prompt_len, temperature=temperature)

    return full_ids, logprobs.cpu(), prompt_len


def run_baseline(repo_path):
    """Run train.py once, return (val_bpb, code, output). Exits on failure."""
    run_result = results.run_experiment(repo_path, timeout_seconds=TRAIN_TIMEOUT)
    if results.did_timeout(run_result):
        print("ERROR: Baseline timed out")
        sys.exit(1)
    if results.did_command_fail(run_result):
        tail = results.extract_error_tail(state_mod.read_file(results.RUN_LOG))
        print(f"ERROR: Baseline failed\n{tail}")
        sys.exit(1)
    val_bpb, peak_vram_mb = results.parse_metrics()
    if val_bpb is None:
        print("ERROR: Could not parse baseline metrics")
        sys.exit(1)
    print(f"Baseline: val_bpb={val_bpb:.6f}  peak_vram_mb={peak_vram_mb}")
    return val_bpb


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="ERL Autoresearch (Experiential RL, v2)")
    parser.add_argument("--repo-path", default=os.path.join(SCAFFOLD_DIR, "autoresearch_rl"))
    parser.add_argument("--source-repo", default=os.path.join(SCAFFOLD_DIR, "autoresearch"))
    parser.add_argument("--model-dir", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--model-gpu", type=int, default=0)
    parser.add_argument("--eval-gpus", type=str, default="")
    parser.add_argument("--workers-per-gpu", type=int, default=1)
    parser.add_argument("--num-steps", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=4e-5)
    parser.add_argument("--kl-coef", type=float, default=0.1)
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-new-tokens", type=int, default=8192)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--attn-impl", default="sdpa")
    parser.add_argument("--log-dir", default=os.path.join(PIPELINE_DIR, "erl_log"))
    parser.add_argument("--resume-step", type=int, default=None)
    args = parser.parse_args()

    repo_path = os.path.abspath(args.repo_path)
    train_path = os.path.join(repo_path, "train.py")
    os.makedirs(args.log_dir, exist_ok=True)

    parallel_mode = bool(args.eval_gpus)

    # Clone repo if needed
    if not os.path.exists(repo_path):
        source = os.path.abspath(args.source_repo)
        if not os.path.exists(source):
            print(f"ERROR: source repo not found at {source}")
            sys.exit(1)
        print(f"Cloning {source} -> {repo_path}")
        shutil.copytree(source, repo_path)

    if not os.path.exists(train_path):
        print(f"ERROR: train.py not found at {train_path}")
        sys.exit(1)

    # Init Ray workers
    workers = None
    if parallel_mode:
        import ray
        from rl_eval import EvalWorker
        ray.init(ignore_reinit_error=True)
        eval_gpu_ids = [int(g) for g in args.eval_gpus.split(",")]
        expanded = [g for g in eval_gpu_ids for _ in range(args.workers_per_gpu)]
        workers = [EvalWorker.remote(gpu, repo_path, i) for i, gpu in enumerate(expanded)]
        print(f"Parallel mode: model GPU {args.model_gpu}, eval GPUs {eval_gpu_ids}")

    # Load model
    device = f"cuda:{args.model_gpu}"
    print(f"Loading model {args.model_dir} on {device}...")
    model, tokenizer = load_model(
        args.model_dir, device=device,
        lora_rank=args.lora_rank, lora_alpha=args.lora_alpha,
        attn_impl=args.attn_impl,
    )
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, betas=(0.9, 0.95), eps=1e-8,
    )

    results.ensure_results_tsv()

    # State: best code + best bpb (no PUCT tree)
    if args.resume_step is not None:
        print(f"\n--- Resuming from step {args.resume_step} ---")
        lora_path = os.path.join(args.log_dir, f"lora_step_{args.resume_step:06d}")
        if os.path.exists(lora_path):
            from peft import PeftModel
            model = PeftModel.from_pretrained(model.base_model.model, lora_path)
            optimizer = torch.optim.AdamW(
                [p for p in model.parameters() if p.requires_grad],
                lr=args.lr, betas=(0.9, 0.95), eps=1e-8,
            )
        best_code_path = os.path.join(args.log_dir, "best_train.py")
        if os.path.exists(best_code_path):
            best_code = Path(best_code_path).read_text()
        else:
            best_code = state_mod.read_file(train_path)
        # Load best_bpb from step_log
        step_log_path = os.path.join(args.log_dir, "step_log.json")
        if os.path.exists(step_log_path):
            with open(step_log_path) as f:
                step_log = json.load(f)
            best_bpb = step_log[-1]["best_bpb"]
        else:
            best_bpb = float("inf")
    else:
        print("\n--- Running baseline ---")
        best_code = state_mod.read_file(train_path)
        best_bpb = run_baseline(repo_path)
        step_log = []

    agent_state = {"repo_path": repo_path, "best_val_bpb": best_bpb}
    if args.resume_step is None:
        step_log = []
    rollout_log_path = os.path.join(args.log_dir, "rollouts.jsonl")

    # ── Main loop ──
    for step in range(args.num_steps):
        step_start = time.time()
        print(f"\n{'='*60}")
        print(f"ERL Step {step}/{args.num_steps} | Best: {best_bpb:.6f}")
        print(f"{'='*60}")

        episodes: list[Episode] = []

        # ── Phase 1: All first attempts ──
        print("\n  --- Phase 1: First attempts ---")
        a1_refs = []  # (ray_ref, rollout, edited_code, proposal, index)

        for g in range(args.batch_size):
            print(f"\n  [Attempt1 {g+1}/{args.batch_size}]")
            rollout, edited_code, proposal = generate_and_apply(
                model, tokenizer, agent_state, best_code, train_path,
                temperature=args.temperature, max_new_tokens=args.max_new_tokens,
            )

            ep = Episode(
                attempt1_rollout=rollout,
                attempt1_proposal=proposal,
                attempt1_edited_code=edited_code,
                attempt1_eval=None,
            )
            episodes.append(ep)

            if edited_code is not None and parallel_mode:
                ref = dispatch_eval(workers, g, best_code, edited_code, step)
                a1_refs.append((ref, g))
            elif edited_code is None:
                rollout.reward = compute_reward(None, rollout.status)

        # Collect attempt1 eval results
        for ref, idx in a1_refs:
            ep = episodes[idx]
            result = collect_eval(ref, ep.attempt1_rollout)
            ep.attempt1_eval = result
            val_str = f"{ep.attempt1_rollout.val_bpb:.6f}" if ep.attempt1_rollout.val_bpb else "---"
            print(f"  Attempt1 {idx+1} result: val_bpb={val_str}  reward={ep.attempt1_rollout.reward:.4f}")

        # ── Phase 2: Batch feedback + ONE reflection ──
        print("\n  --- Phase 2: Batch reflection ---")
        attempt_summaries = []
        for ep in episodes:
            r = ep.attempt1_rollout
            attempt_summaries.append({
                "description": r.description,
                "status": r.status,
                "val_bpb": r.val_bpb,
                "eval_output": ep.attempt1_eval["output"] if ep.attempt1_eval else None,
                "edit_error": str(ep.attempt1_proposal) if r.status == "edit_failed" and ep.attempt1_proposal else None,
            })

        batch_feedback = build_batch_feedback(attempt_summaries, best_bpb)
        print(f"  Batch: {len(episodes)} attempts, "
              f"{sum(1 for a in attempt_summaries if a['val_bpb'] is not None and a['val_bpb'] < best_bpb)} improved")

        ref_text, ref_ids, ref_lp, ref_plen = generate_batch_reflection(
            model, tokenizer,
            train_py=best_code,
            batch_feedback=batch_feedback,
            best_val_bpb=best_bpb,
            temperature=args.temperature,
        )
        print(f"  Reflection: {ref_text[:120]}...")

        reflection_ctx = build_reflection_context(batch_feedback, ref_text)

        # ── Phase 3: All second attempts (using shared reflection) ──
        print("\n  --- Phase 3: Second attempts ---")
        a2_refs = []

        for g in range(args.batch_size):
            print(f"\n  [Attempt2 {g+1}/{args.batch_size}]")
            rollout2, edited2, proposal2 = generate_and_apply(
                model, tokenizer, agent_state, best_code, train_path,
                temperature=args.temperature, max_new_tokens=args.max_new_tokens,
                error_context=reflection_ctx,
            )

            ep = episodes[g]
            ep.attempt2_rollout = rollout2
            ep.attempt2_proposal = proposal2
            ep.attempt2_edited_code = edited2
            ep.train_attempt2 = rollout2.full_ids.numel() > 0

            if edited2 is not None and parallel_mode:
                ref = dispatch_eval(workers, g, best_code, edited2, step)
                a2_refs.append((ref, g))
            elif edited2 is None:
                rollout2.reward = compute_reward(None, rollout2.status)

        # Collect attempt2 eval results
        for ref, idx in a2_refs:
            ep = episodes[idx]
            result = collect_eval(ref, ep.attempt2_rollout)
            ep.attempt2_eval = result
            val_str = f"{ep.attempt2_rollout.val_bpb:.6f}" if ep.attempt2_rollout.val_bpb else "---"
            print(f"  Attempt2 {idx+1} result: val_bpb={val_str}  reward={ep.attempt2_rollout.reward:.4f}")

        # Assign reflection reward = mean of attempt2 rewards
        a2_rewards = [ep.attempt2_rollout.reward for ep in episodes if ep.attempt2_rollout is not None and ep.attempt2_rollout.full_ids.numel() > 0]
        ref_reward = sum(a2_rewards) / len(a2_rewards) if a2_rewards else 0.0

        step_reflection = StepReflection(
            feedback_text=batch_feedback,
            reflection_text=ref_text,
            full_ids=ref_ids,
            old_logprobs=ref_lp,
            prompt_len=ref_plen,
            reward=ref_reward,
        )

        # ── Phase 4: Update best + build distillation targets ──
        for ep in episodes:
            # Check both attempts for new best
            for tag, r, code in [
                ("attempt1", ep.attempt1_rollout, ep.attempt1_edited_code),
                ("attempt2", ep.attempt2_rollout, ep.attempt2_edited_code),
            ]:
                if r is None or r.val_bpb is None or code is None:
                    continue
                if r.val_bpb < best_bpb:
                    best_bpb = r.val_bpb
                    best_code = code
                    agent_state["best_val_bpb"] = best_bpb
                    print(f"  *** NEW BEST: {best_bpb:.6f} (from {tag}) ***")
                    Path(os.path.join(args.log_dir, "best_train.py")).write_text(best_code)

            # Distillation: if attempt2 beat best_bpb at time of proposal
            if (ep.attempt2_rollout is not None
                    and ep.attempt2_rollout.val_bpb is not None
                    and ep.attempt2_rollout.val_bpb < best_bpb + 1e-8  # improved or matched new best
                    and ep.attempt2_rollout.full_ids.numel() > 0):
                print(f"  [Building distillation target]")
                d_ids, d_lp, d_plen = build_distill_ids(
                    model, tokenizer, agent_state,
                    ep.attempt2_rollout.proposal_text, args.temperature,
                )
                ep.distill_full_ids = d_ids
                ep.distill_logprobs = d_lp
                ep.distill_prompt_len = d_plen
                ep.train_distill = True

        # Log rollouts
        for g, ep in enumerate(episodes):
            for tag, r in [("attempt1", ep.attempt1_rollout), ("attempt2", ep.attempt2_rollout)]:
                if r is None:
                    continue
                results.append_result("erl", r.val_bpb, None, r.status, f"[{tag}] {r.description}")
                with open(rollout_log_path, "a") as f:
                    f.write(json.dumps({
                        "step": step, "episode": g, "tag": tag,
                        "val_bpb": r.val_bpb, "reward": r.reward,
                        "status": r.status, "description": r.description,
                        "distilled": ep.train_distill,
                    }) + "\n")

        # ── Phase 5: Train ──
        torch.cuda.empty_cache()

        metrics = erl_train_step(
            model, optimizer, episodes, step_reflection,
            kl_coef=args.kl_coef,
            temperature=args.temperature,
            max_grad_norm=args.max_grad_norm,
        )
        n_distilled = sum(1 for ep in episodes if ep.train_distill)
        print(f"\n  ERL update: grpo={metrics['avg_grpo_loss']:.4f} "
              f"reflect={metrics['avg_reflect_loss']:.4f} "
              f"distill={metrics['avg_distill_loss']:.4f} "
              f"(grpo_tok={metrics['num_grpo_tokens']} "
              f"ref_tok={metrics['num_reflect_tokens']} "
              f"dist_tok={metrics['num_distill_tokens']})")

        # ── Checkpoint ──
        model.save_pretrained(os.path.join(args.log_dir, f"lora_step_{step:06d}"))
        step_time = time.time() - step_start

        step_info = {
            "step": step,
            "best_bpb": best_bpb,
            "episodes": len(episodes),
            "distilled": n_distilled,
            "reflection_reward": round(ref_reward, 4),
            "step_time_s": round(step_time, 1),
            **{k: round(v, 6) if isinstance(v, float) else v for k, v in metrics.items()},
        }
        step_log.append(step_info)
        print(f"  Step time: {step_time/60:.1f} min | distilled={n_distilled}")

        with open(os.path.join(args.log_dir, "step_log.json"), "w") as f:
            json.dump(step_log, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Done. Best val_bpb: {best_bpb:.6f}")
    Path(os.path.join(args.log_dir, "best_train.py")).write_text(best_code)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify syntax**

Run: `cd /Users/chenshengmai/Desktop/autosearch_rl/llm_scaffold && python -m py_compile erl_pipeline/erl_main.py && echo OK`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add erl_pipeline/erl_main.py
git commit -m "feat(erl-v2): phased main loop — no PUCT, GRPO, batch reflection, always-reflect"
```

---

### Task 6: SLURM Script Update

**Files:**
- Rewrite: `erl_pipeline/run_erl.sh`

Remove `--puct-c`, add note about phased eval (batch_size * 2 evals per step).

- [ ] **Step 1: Write run_erl.sh**

```bash
#!/bin/bash
#SBATCH --job-name=autoresearch-erl
#SBATCH --output=autoresearch_erl_%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=100gb
#SBATCH --time=10-12:00:00
#SBATCH --partition=hpg-b200
#SBATCH --gpus=8

# ── Configuration ─────────────────────────────────────────────
MODEL="Qwen/Qwen3.5-9B"
NUM_STEPS=50
MODEL_GPU=0
EVAL_GPUS="1,2,3,4,5,6,7"
# batch_size=7: 7 attempt1 + 7 attempt2 = 14 evals per step (2 waves on 7 GPUs)
BATCH_SIZE=7
WORKERS_PER_GPU=1
KL_COEF=0.1
LR=4e-5
LORA_RANK=32
LORA_ALPHA=64
PROJ_DIR="/blue/buyuheng/li_an.ucsb/proj_yepeng"
CONDA_ENV="${PROJ_DIR}/envs/myenv"
SCAFFOLD_DIR="${PROJ_DIR}/llm_scaffold_rl/llm_scaffold"
SOURCE_REPO="${SCAFFOLD_DIR}/autoresearch"
ERL_REPO="${SCAFFOLD_DIR}/autoresearch_rl"

cd "${SCAFFOLD_DIR}/erl_pipeline"

if [ ! -d "$SOURCE_REPO" ]; then
    echo "Cloning autoresearch repo..."
    git clone https://github.com/karpathy/autoresearch.git "$SOURCE_REPO"
fi

module load gcc/14.2.0
module load cuda/12.8.1
module load conda
conda activate "$CONDA_ENV"
export HF_HOME="${PROJ_DIR}/.cache/huggingface"

echo "Job ID:    $SLURM_JOB_ID"
echo "Node:      $(hostname)"
echo "GPUs:      $(nvidia-smi -L 2>/dev/null | wc -l)"
echo "Model:     $MODEL"
echo "Mode:      ERL v2 (phased, batch reflection, GRPO, no PUCT)"
echo "Started:   $(date)"
echo "---"

cd "$SOURCE_REPO" && uv sync && cd "${SCAFFOLD_DIR}/erl_pipeline"

echo "Starting ERL experiment loop..."
python erl_main.py \
    --repo-path "$ERL_REPO" \
    --source-repo "$SOURCE_REPO" \
    --model-dir "$MODEL" \
    --model-gpu "$MODEL_GPU" \
    --eval-gpus "$EVAL_GPUS" \
    --workers-per-gpu "$WORKERS_PER_GPU" \
    --num-steps "$NUM_STEPS" \
    --batch-size "$BATCH_SIZE" \
    --kl-coef "$KL_COEF" \
    --lr "$LR" \
    --lora-rank "$LORA_RANK" \
    --lora-alpha "$LORA_ALPHA" \
    --temperature 0.7 \
    --max-new-tokens 8192 \
    --attn-impl sdpa \
    --log-dir ./erl_log

echo "Finished. $(date)"
```

- [ ] **Step 2: Make executable + verify**

Run: `chmod +x erl_pipeline/run_erl.sh && echo OK`

- [ ] **Step 3: Commit**

```bash
git add erl_pipeline/run_erl.sh
git commit -m "feat(erl-v2): update SLURM script — remove PUCT, document phased eval"
```

---

### Task 7: Smoke Test

- [ ] **Step 1: Compile all files**

Run:
```bash
cd /Users/chenshengmai/Desktop/autosearch_rl/llm_scaffold
python -m py_compile erl_pipeline/__init__.py
python -m py_compile erl_pipeline/erl_types.py
python -m py_compile erl_pipeline/erl_feedback.py
python -m py_compile erl_pipeline/erl_reflect.py
python -m py_compile erl_pipeline/erl_trainer.py
python -m py_compile erl_pipeline/erl_main.py
echo "All compile OK"
```
Expected: `All compile OK`

- [ ] **Step 2: Test feedback logic**

Run:
```bash
cd /Users/chenshengmai/Desktop/autosearch_rl/llm_scaffold && python -c "
from erl_pipeline.erl_feedback import build_batch_feedback, build_attempt_feedback

# Single attempt
fb = build_attempt_feedback('Increase LR', 'crash', None, 0.989, eval_output='OOM')
assert 'CRASH' in fb

# Batch
attempts = [
    {'description': 'Increase LR', 'status': 'crash', 'val_bpb': None},
    {'description': 'Add warmup', 'status': 'keep', 'val_bpb': 0.988},
    {'description': 'Change depth', 'status': 'keep', 'val_bpb': 0.991},
]
batch = build_batch_feedback(attempts, 0.989)
assert 'Total attempts: 3' in batch
assert 'Improved: 1' in batch
assert 'Attempt 1' in batch
assert 'Attempt 3' in batch
print('All feedback tests pass')
"
```
Expected: `All feedback tests pass`

- [ ] **Step 3: Test GRPO advantages**

Run:
```bash
cd /Users/chenshengmai/Desktop/autosearch_rl/llm_scaffold && python -c "
import torch
# Inline test since torch may not be available
rewards = [0.5, 0.3, 0.8, 0.2]
r = torch.tensor(rewards, dtype=torch.float32)
mean = r.mean()
std = r.std()
advs = (r - mean) / std
assert advs[2] > 0  # 0.8 should have positive advantage
assert advs[3] < 0  # 0.2 should have negative advantage
assert abs(advs.mean().item()) < 1e-6  # mean advantage ~= 0

# All same rewards -> zero advantages
same = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32)
std_same = same.std()
assert std_same < 1e-8
print('All GRPO tests pass')
"
```
Expected: `All GRPO tests pass` (will only work if torch is installed locally)

---

## Summary

| Task | File | Key change from v1 |
|------|------|-------------------|
| 1 | `erl_types.py` | Add `StepReflection`, remove PUCT refs, always-reflect |
| 2 | `erl_feedback.py` | Batch-level `build_batch_feedback`, remove `should_reflect` |
| 3 | `erl_reflect.py` | `generate_batch_reflection` sees all attempt1s, one per step |
| 4 | `erl_trainer.py` | GRPO advantages, 4 signals (attempt1, reflection, attempt2, distill) |
| 5 | `erl_main.py` | Phased loop, no PUCT, always-reflect, batch reflection |
| 6 | `run_erl.sh` | Remove --puct-c, document phased eval |
| 7 | Smoke test | Compile + logic verification |
