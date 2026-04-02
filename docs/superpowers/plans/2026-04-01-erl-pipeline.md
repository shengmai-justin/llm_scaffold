# ERL Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an Experiential RL pipeline (`erl_pipeline/`) that wraps the existing `rl_pipeline/` infrastructure with a reflect-retry-internalize loop, so the model learns from structured failure feedback rather than bare scalar rewards.

**Architecture:** On each rollout, the model makes a first attempt. If it fails (crash, edit_failed, or val_bpb regression), the model receives structured feedback describing *why* it failed and generates a reflection. It then makes a second attempt conditioned on that reflection. Successful second attempts are distilled back: the model is trained to produce the corrected output from the *original* prompt alone (no reflection context). Three training signals per episode: first attempt (GRPO-style), second attempt (GRPO-style), distilled (RAFT-style supervised loss on successful corrections).

**Tech Stack:** Python 3.10+, PyTorch, PEFT/LoRA, Ray (for parallel eval), HuggingFace Transformers. Reuses `rl_pipeline/rl_model.py`, `rl_pipeline/rl_eval.py`, `rl_pipeline/rl_sampler.py`, `planner.py`, `results.py`, `state.py` directly via imports.

---

## Scope

This plan covers one subsystem: the ERL experiment loop. It does NOT modify any existing files in `rl_pipeline/` or the frozen pipeline. Everything lives in a new `erl_pipeline/` directory and imports shared modules.

## File Structure

```
llm_scaffold/
├── rl_pipeline/          # existing TTT-Discover pipeline (UNTOUCHED)
│   ├── rl_model.py       # reused: load_model, generate_with_logprobs, compute_*_logprobs
│   ├── rl_eval.py        # reused: EvalWorker, parse_metrics_from_output
│   ├── rl_sampler.py     # reused: State, PUCTSampler
│   ├── rl_trainer.py     # reused: compute_entropic_advantages (but NOT train_step)
│   └── rl_types.py       # reused: Rollout dataclass
│
└── erl_pipeline/         # NEW — all ERL-specific code
    ├── __init__.py
    ├── erl_types.py      # Episode dataclass (first attempt + reflection + second attempt + distill)
    ├── erl_feedback.py   # Structured feedback generation from eval results
    ├── erl_reflect.py    # Reflection generation (model generates diagnosis of failure)
    ├── erl_trainer.py    # ERL training step: GRPO for attempts, RAFT for distillation
    ├── erl_main.py       # Entry point, ERL experiment loop
    └── run_erl.sh        # SLURM launch script
```

**Key reuse decisions:**
- `rl_model.py` — model loading, generation, logprobs computation are identical
- `rl_eval.py` — Ray EvalWorker is identical (takes code string, returns metrics)
- `rl_sampler.py` — PUCT tree search is identical (State stores code)
- `rl_trainer.py` — `compute_entropic_advantages` reused; `train_step` replaced with ERL-specific version
- `rl_types.py` — `Rollout` reused as-is; new `Episode` wraps multiple rollouts
- `planner.py` — `build_planner_context`, `validate_planner_output`, `apply_edits`, `validate_edit_targets`, `preview_diff` all reused
- `results.py` — `run_experiment`, `parse_metrics`, `append_result`, `ensure_results_tsv` all reused

---

## Task 1: ERL Data Types

**Files:**
- Create: `erl_pipeline/__init__.py`
- Create: `erl_pipeline/erl_types.py`

- [ ] **Step 1: Create empty package**

```python
# erl_pipeline/__init__.py
```

- [ ] **Step 2: Write Episode dataclass**

```python
# erl_pipeline/erl_types.py
"""Data structures for ERL training loop."""
from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class Episode:
    """One ERL episode: first attempt, optional reflection + second attempt.

    Training signals:
    - attempt1: always trained with GRPO advantage
    - attempt2_raw: trained with GRPO advantage (reflection-conditioned)
    - attempt2_distill: trained with RAFT loss (successful corrections only,
      prompt stripped of reflection context)
    """
    # First attempt
    attempt1_rollout: "Rollout"  # from rl_types
    attempt1_proposal: dict | None
    attempt1_edited_code: str | None
    attempt1_eval: dict | None  # {"val_bpb", "success", "output", ...}

    # Feedback + reflection (None if first attempt succeeded)
    feedback_text: str | None = None
    reflection_text: str | None = None
    reflection_full_ids: torch.Tensor | None = None
    reflection_logprobs: torch.Tensor | None = None
    reflection_prompt_len: int = 0

    # Second attempt (None if first attempt succeeded or reflection failed)
    attempt2_rollout: "Rollout | None" = None
    attempt2_proposal: dict | None = None
    attempt2_edited_code: str | None = None
    attempt2_eval: dict | None = None

    # Distillation target (copy of attempt2 with original prompt, no reflection)
    distill_full_ids: torch.Tensor | None = None
    distill_logprobs: torch.Tensor | None = None
    distill_prompt_len: int = 0

    # Which training signals are active
    train_attempt1: bool = True
    train_attempt2: bool = False
    train_distill: bool = False
```

- [ ] **Step 3: Verify import**

Run: `cd /Users/chenshengmai/Desktop/autosearch_rl/llm_scaffold && python -c "from erl_pipeline.erl_types import Episode; print('OK')"`
Expected: `OK`

- [ ] **Step 4: Commit**

```bash
git add erl_pipeline/__init__.py erl_pipeline/erl_types.py
git commit -m "feat(erl): add Episode dataclass for ERL training loop"
```

---

## Task 2: Structured Feedback Generation

**Files:**
- Create: `erl_pipeline/erl_feedback.py`

The feedback module takes eval results and generates structured text describing *why* a rollout failed. This replaces the bare `-1.0` reward with actionable information.

- [ ] **Step 1: Write feedback module**

```python
# erl_pipeline/erl_feedback.py
"""Structured feedback generation from eval results.

Converts raw eval outputs (val_bpb, crash logs, edit errors) into
human-readable feedback text that the model can reflect on.
"""
from __future__ import annotations


def build_feedback(
    status: str,
    val_bpb: float | None,
    best_val_bpb: float,
    description: str,
    eval_output: str | None = None,
    edit_error: str | None = None,
) -> str:
    """Build structured feedback text from an eval result.

    Args:
        status: "keep", "crash", "edit_failed", or "regressed"
        val_bpb: achieved val_bpb (None for crash/edit_failed)
        best_val_bpb: current best val_bpb to compare against
        description: what the experiment tried
        eval_output: last 500 chars of train.py output (for crash diagnosis)
        edit_error: error message from failed edit application

    Returns:
        Structured feedback string for the reflection prompt.
    """
    lines = [f"Experiment: {description}"]

    if status == "edit_failed":
        lines.append(f"Result: EDIT FAILED — the search/replace edit could not be applied.")
        if edit_error:
            lines.append(f"Error: {edit_error}")
        lines.append("The search string does not match any text in the current train.py.")
        lines.append("Common causes: wrong whitespace, outdated code reference, or the target was already modified.")

    elif status == "crash":
        lines.append(f"Result: CRASH — train.py failed to produce val_bpb.")
        if eval_output:
            tail = eval_output.strip()[-500:]
            lines.append(f"Last output:\n{tail}")
        lines.append("The edit likely introduced a syntax error, runtime error, or caused OOM.")

    elif status == "timeout":
        lines.append(f"Result: TIMEOUT — train.py exceeded the time budget.")
        lines.append("The edit likely increased computation (larger model, more steps, etc.).")

    elif val_bpb is not None and val_bpb >= best_val_bpb:
        delta = val_bpb - best_val_bpb
        lines.append(f"Result: REGRESSED — val_bpb={val_bpb:.6f} (best={best_val_bpb:.6f}, delta=+{delta:.6f}).")
        lines.append("The edit made val_bpb worse. Consider the opposite direction or a different approach.")

    else:
        lines.append(f"Result: SUCCESS — val_bpb={val_bpb:.6f} (best={best_val_bpb:.6f}).")

    return "\n".join(lines)


def should_reflect(status: str, val_bpb: float | None, best_val_bpb: float) -> bool:
    """Reflection gating: only reflect on failures, not successes.

    ERL paper finding: reflecting on successes causes off-policy dominance
    and reward hacking. Gate reflection to failures only.
    """
    if status in ("crash", "edit_failed", "timeout"):
        return True
    if val_bpb is not None and val_bpb >= best_val_bpb:
        return True
    return False
```

- [ ] **Step 2: Verify import**

Run: `cd /Users/chenshengmai/Desktop/autosearch_rl/llm_scaffold && python -c "from erl_pipeline.erl_feedback import build_feedback, should_reflect; print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add erl_pipeline/erl_feedback.py
git commit -m "feat(erl): add structured feedback generation for failed rollouts"
```

---

## Task 3: Reflection Generation

**Files:**
- Create: `erl_pipeline/erl_reflect.py`

The model generates a structured reflection diagnosing *what went wrong* and *what to try differently*. This reflection becomes context for the second attempt.

- [ ] **Step 1: Write reflection module**

```python
# erl_pipeline/erl_reflect.py
"""Reflection generation: model diagnoses failure and proposes correction strategy.

The reflection is generated by the same model (with LoRA) and its logprobs
are collected for training. The reflection text becomes context for the
second attempt's proposal prompt.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "rl_pipeline"))

import torch
from rl_model import generate_with_logprobs


REFLECTION_SYSTEM = """You are an ML researcher reflecting on a failed experiment.

Given the experiment description, the structured feedback, and the current train.py,
diagnose what went wrong and propose a corrective strategy.

Your reflection must be concise (under 200 words) and contain:
1. DIAGNOSIS: Why did this experiment fail?
2. LESSON: What does this tell us about the model/training dynamics?
3. STRATEGY: What specific change should be tried next instead?

Output your reflection as plain text. No JSON, no code fences."""


def generate_reflection(
    model,
    tokenizer,
    train_py: str,
    description: str,
    feedback_text: str,
    best_val_bpb: float,
    temperature: float = 0.7,
    max_new_tokens: int = 1024,
) -> tuple[str, torch.Tensor, torch.Tensor, int]:
    """Generate a reflection on a failed experiment.

    Returns:
        (reflection_text, full_ids, logprobs, prompt_len)
    """
    user_msg = (
        f"Current train.py (abbreviated, first 200 lines):\n"
        f"```python\n{_abbreviate(train_py, 200)}\n```\n\n"
        f"Current best val_bpb: {best_val_bpb:.6f}\n\n"
        f"Failed experiment:\n{feedback_text}\n\n"
        f"Reflect on this failure and propose a corrective strategy."
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


def build_reflected_proposal_context(
    system_rules: str,
    user_msg: str,
    feedback_text: str,
    reflection_text: str,
) -> str:
    """Augment the standard proposal prompt with reflection context.

    Inserts feedback + reflection between the standard user message
    and the proposal request, so the model sees what went wrong
    before proposing the next edit.
    """
    reflection_block = (
        f"\n\n--- Previous attempt feedback ---\n"
        f"{feedback_text}\n\n"
        f"--- Your reflection ---\n"
        f"{reflection_text}\n\n"
        f"Based on your reflection, propose a corrected experiment."
    )
    return user_msg + reflection_block


def _abbreviate(text: str, max_lines: int) -> str:
    """Truncate text to max_lines for prompt efficiency."""
    lines = text.split("\n")
    if len(lines) <= max_lines:
        return text
    return "\n".join(lines[:max_lines]) + f"\n... ({len(lines) - max_lines} more lines)"
```

- [ ] **Step 2: Verify import**

Run: `cd /Users/chenshengmai/Desktop/autosearch_rl/llm_scaffold && python -c "from erl_pipeline.erl_reflect import generate_reflection, build_reflected_proposal_context; print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add erl_pipeline/erl_reflect.py
git commit -m "feat(erl): add reflection generation for failed experiments"
```

---

## Task 4: ERL Trainer

**Files:**
- Create: `erl_pipeline/erl_trainer.py`

Replaces `rl_pipeline/rl_trainer.py:train_step` with a version that handles three training signals per episode with different loss functions.

- [ ] **Step 1: Write ERL trainer**

```python
# erl_pipeline/erl_trainer.py
"""ERL training step: GRPO for attempts, RAFT for distillation.

Three training signals per episode:
1. First attempt  -> GRPO (policy gradient with advantage)
2. Second attempt -> GRPO (policy gradient with advantage, reflection-conditioned)
3. Distillation   -> RAFT (supervised loss on successful corrections, original prompt)
"""
from __future__ import annotations

import torch
from torch.nn.utils import clip_grad_norm_

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "rl_pipeline"))

from rl_model import compute_response_logprobs, compute_base_logprobs
from rl_types import Rollout
from erl_types import Episode


def erl_train_step(
    model,
    optimizer,
    episodes: list[Episode],
    advantages: torch.Tensor,
    kl_coef: float = 0.1,
    distill_coef: float = 1.0,
    temperature: float = 1.0,
    max_grad_norm: float = 1.0,
) -> dict:
    """One ERL training step.

    For each episode:
    - GRPO loss on attempt1 rollout (always)
    - GRPO loss on attempt2 rollout (if reflection triggered)
    - RAFT loss on distill target (if attempt2 succeeded)

    Args:
        advantages: tensor of shape [len(episodes)], computed from attempt1 rewards
                    (attempt2 advantages computed inline from improvement delta)
        distill_coef: weight for distillation loss relative to GRPO loss
    """
    optimizer.zero_grad()
    total_grpo_loss = 0.0
    total_distill_loss = 0.0
    num_grpo_tokens = 0
    num_distill_tokens = 0
    all_ratios = []
    all_kls = []

    for ei, (ep, adv) in enumerate(zip(episodes, advantages)):
        # --- Signal 1: First attempt (GRPO) ---
        if ep.train_attempt1 and abs(adv.item()) > 1e-8:
            r = ep.attempt1_rollout
            if r.full_ids.numel() > 0:
                metrics = _grpo_loss(
                    model, r, adv.item(), kl_coef, temperature,
                    all_ratios, all_kls,
                )
                total_grpo_loss += metrics["loss"] * metrics["tokens"]
                num_grpo_tokens += metrics["tokens"]

        # --- Signal 2: Second attempt (GRPO) ---
        if ep.train_attempt2 and ep.attempt2_rollout is not None:
            r2 = ep.attempt2_rollout
            if r2.full_ids.numel() > 0:
                # Advantage for attempt2: based on improvement over attempt1
                a2_reward = r2.reward
                a1_reward = ep.attempt1_rollout.reward
                a2_adv = a2_reward - a1_reward  # positive if attempt2 improved
                if abs(a2_adv) > 1e-8:
                    metrics = _grpo_loss(
                        model, r2, a2_adv, kl_coef, temperature,
                        all_ratios, all_kls,
                    )
                    total_grpo_loss += metrics["loss"] * metrics["tokens"]
                    num_grpo_tokens += metrics["tokens"]

        # --- Signal 3: Distillation (RAFT) ---
        if ep.train_distill and ep.distill_full_ids is not None:
            if ep.distill_full_ids.numel() > 0:
                dl = _distill_loss(model, ep, temperature)
                total_distill_loss += dl["loss"] * dl["tokens"]
                num_distill_tokens += dl["tokens"]

    # Combined backward already done per-rollout, now clip + step
    total_tokens = num_grpo_tokens + num_distill_tokens
    if total_tokens > 0:
        clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad],
            max_grad_norm,
        )
        optimizer.step()

    metrics = {
        "avg_grpo_loss": total_grpo_loss / max(num_grpo_tokens, 1),
        "avg_distill_loss": total_distill_loss / max(num_distill_tokens, 1),
        "num_grpo_tokens": num_grpo_tokens,
        "num_distill_tokens": num_distill_tokens,
    }
    if all_ratios:
        cat = torch.cat(all_ratios)
        metrics["ratio_mean"] = cat.mean().item()
        metrics["ratio_max"] = cat.max().item()
    if all_kls:
        metrics["kl_mean"] = sum(all_kls) / len(all_kls)
    return metrics


def _grpo_loss(
    model,
    rollout: Rollout,
    advantage: float,
    kl_coef: float,
    temperature: float,
    all_ratios: list,
    all_kls: list,
) -> dict:
    """Compute GRPO policy gradient loss for one rollout. Calls backward."""
    old_lp = rollout.old_logprobs.to(model.device)
    new_lp = compute_response_logprobs(
        model, rollout.full_ids, rollout.prompt_len,
        temperature=temperature,
    )

    ratio = torch.exp(new_lp - old_lp)
    all_ratios.append(ratio.detach().cpu())

    shaped_adv = torch.tensor(advantage, device=model.device)
    if kl_coef > 0:
        base_lp = compute_base_logprobs(
            model, rollout.full_ids, rollout.prompt_len,
            temperature=temperature,
        )
        kl_per_token = new_lp - base_lp
        all_kls.append(kl_per_token.mean().item())
        shaped_adv = shaped_adv - kl_coef * kl_per_token

    loss = -(ratio * shaped_adv).mean()
    loss.backward()

    return {"loss": loss.item(), "tokens": len(new_lp)}


def _distill_loss(
    model,
    episode: Episode,
    temperature: float,
) -> dict:
    """Supervised NLL loss on successful second attempt, conditioned on original prompt.

    This trains the model to produce the corrected behavior WITHOUT
    the reflection context — pure context distillation.
    """
    new_lp = compute_response_logprobs(
        model, episode.distill_full_ids, episode.distill_prompt_len,
        temperature=temperature,
    )

    loss = -new_lp.mean()  # standard NLL
    loss.backward()

    return {"loss": loss.item(), "tokens": len(new_lp)}
```

- [ ] **Step 2: Verify import**

Run: `cd /Users/chenshengmai/Desktop/autosearch_rl/llm_scaffold && python -c "from erl_pipeline.erl_trainer import erl_train_step; print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add erl_pipeline/erl_trainer.py
git commit -m "feat(erl): add ERL trainer with GRPO + RAFT distillation"
```

---

## Task 5: ERL Main Loop

**Files:**
- Create: `erl_pipeline/erl_main.py`

The main orchestration loop. For each step:
1. PUCT selects parent state
2. Generate batch of first attempts
3. Evaluate first attempts
4. For failures: generate reflection, then second attempt, evaluate
5. Build distillation targets for successful corrections
6. Train with three signals

- [ ] **Step 1: Write ERL main loop**

```python
# erl_pipeline/erl_main.py
"""ERL experiment loop entry point.

Experience -> Reflect -> Retry -> Internalize.

Reuses rl_pipeline modules for model loading, eval workers, PUCT, and generation.
Adds reflection generation, second attempts, and context distillation.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

# Add parent dirs for imports
PIPELINE_DIR = os.path.dirname(os.path.abspath(__file__))
SCAFFOLD_DIR = os.path.dirname(PIPELINE_DIR)
RL_PIPELINE_DIR = os.path.join(SCAFFOLD_DIR, "rl_pipeline")
sys.path.insert(0, SCAFFOLD_DIR)
sys.path.insert(0, RL_PIPELINE_DIR)

import torch

import planner
import results
import state as state_mod
from rl_model import load_model, generate_with_logprobs
from rl_planner import propose_experiment_rl
from rl_sampler import State, PUCTSampler
from rl_trainer import compute_reward, compute_entropic_advantages
from rl_types import Rollout

from erl_types import Episode
from erl_feedback import build_feedback, should_reflect
from erl_reflect import generate_reflection, build_reflected_proposal_context
from erl_trainer import erl_train_step

TRAIN_TIMEOUT = 600


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def generate_and_apply(
    model, tokenizer, agent_state, parent, train_path, temperature, max_new_tokens,
    reflection_context=None,
):
    """Generate proposal, apply edits. Returns (rollout, edited_code, proposal).

    If reflection_context is provided, it's appended to the user prompt
    so the model sees the reflection before proposing.
    """
    state_mod.write_file(train_path, parent.code)

    try:
        error_ctx = reflection_context if reflection_context else None
        proposal, rollout = propose_experiment_rl(
            model, tokenizer, agent_state,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            error_context=error_ctx,
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

    original_text = parent.code
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


def evaluate_code(workers, worker_idx, parent_code, edited_code, step):
    """Dispatch eval to Ray worker, collect result."""
    import ray
    worker = workers[worker_idx % len(workers)]
    ref = worker.evaluate.remote(parent_code, edited_code, step)
    return ray.get(ref)


def build_distill_ids(
    model, tokenizer, agent_state, attempt2_text, temperature,
):
    """Build distillation target: attempt2's response tokens with the ORIGINAL prompt.

    This strips the reflection context so the model learns to produce
    the corrected output from the standard prompt alone.
    """
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

    # Tokenize original prompt + attempt2 response together
    full_text = original_prompt + attempt2_text
    tokens = tokenizer(full_text, return_tensors="pt")
    full_ids = tokens["input_ids"][0].cpu()
    prompt_tokens = tokenizer(original_prompt, return_tensors="pt")
    prompt_len = prompt_tokens["input_ids"].shape[1]

    # Compute logprobs for the response portion (no grad needed for distill target)
    with torch.no_grad():
        from rl_model import compute_response_logprobs as _compute_lp
        logprobs = _compute_lp(model, full_ids, prompt_len, temperature=temperature)

    return full_ids, logprobs.cpu(), prompt_len


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="ERL Autoresearch (Experiential RL)")
    parser.add_argument("--repo-path", default=os.path.join(SCAFFOLD_DIR, "autoresearch_rl"))
    parser.add_argument("--source-repo", default=os.path.join(SCAFFOLD_DIR, "autoresearch"))
    parser.add_argument("--model-dir", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--model-gpu", type=int, default=0)
    parser.add_argument("--eval-gpus", type=str, default="",
                        help="Comma-separated GPU IDs for eval workers (empty = sequential)")
    parser.add_argument("--workers-per-gpu", type=int, default=1)
    parser.add_argument("--num-steps", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=4e-5)
    parser.add_argument("--kl-coef", type=float, default=0.1)
    parser.add_argument("--distill-coef", type=float, default=1.0)
    parser.add_argument("--puct-c", type=float, default=1.0)
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

    # PUCT sampler
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
        dummy = State(timestep=0, code="", value=0.0)
        sampler = PUCTSampler(
            initial_state=dummy, log_dir=args.log_dir,
            puct_c=args.puct_c, resume_step=args.resume_step,
        )
        best_state = sampler.best_state()
        best_bpb = -best_state.value if best_state and best_state.value is not None else float("inf")
    else:
        print("\n--- Running baseline ---")
        original_code = state_mod.read_file(train_path)
        from rl_main import run_baseline
        baseline_bpb, baseline_output = run_baseline(repo_path)
        initial_state = State(
            timestep=0, code=original_code,
            value=-baseline_bpb, observation=baseline_output,
        )
        sampler = PUCTSampler(
            initial_state=initial_state, log_dir=args.log_dir, puct_c=args.puct_c,
        )
        best_bpb = baseline_bpb

    agent_state = {"repo_path": repo_path, "best_val_bpb": best_bpb}
    step_log = []
    rollout_log_path = os.path.join(args.log_dir, "rollouts.jsonl")

    # ── Main loop ──
    for step in range(args.num_steps):
        step_start = time.time()
        print(f"\n{'='*60}")
        print(f"ERL Step {step}/{args.num_steps} | Best: {best_bpb:.6f} | Buffer: {sampler.buffer_size()}")
        print(f"{'='*60}")

        parent = sampler.sample_state()
        episodes: list[Episode] = []
        worker_idx = 0

        for g in range(args.batch_size):
            print(f"\n  --- Episode {g+1}/{args.batch_size} ---")

            # ── Phase 1: First attempt ──
            print("  [Attempt 1]")
            rollout1, edited1, proposal1 = generate_and_apply(
                model, tokenizer, agent_state, parent, train_path,
                temperature=args.temperature, max_new_tokens=args.max_new_tokens,
            )

            eval1 = None
            if edited1 is not None and parallel_mode:
                eval1 = evaluate_code(workers, worker_idx, parent.code, edited1, step)
                worker_idx += 1
                rollout1.val_bpb = eval1["val_bpb"]
                rollout1.status = "keep" if eval1["success"] else "crash"
                rollout1.reward = compute_reward(rollout1.val_bpb, rollout1.status)
                val_str = f"{rollout1.val_bpb:.6f}" if rollout1.val_bpb else "---"
                print(f"  Attempt 1 result: val_bpb={val_str}  reward={rollout1.reward:.4f}")

            # Determine effective status for reflection gating
            a1_status = rollout1.status
            a1_val = rollout1.val_bpb
            if a1_val is not None and a1_val >= best_bpb:
                a1_status = "regressed"

            ep = Episode(
                attempt1_rollout=rollout1,
                attempt1_proposal=proposal1,
                attempt1_edited_code=edited1,
                attempt1_eval=eval1,
            )

            # ── Phase 2: Reflection (gated on failure) ──
            if should_reflect(a1_status, a1_val, best_bpb) and rollout1.full_ids.numel() > 0:
                print("  [Reflecting...]")
                feedback = build_feedback(
                    status=a1_status,
                    val_bpb=a1_val,
                    best_val_bpb=best_bpb,
                    description=rollout1.description,
                    eval_output=eval1["output"] if eval1 else None,
                    edit_error=str(proposal1) if a1_status == "edit_failed" else None,
                )
                ep.feedback_text = feedback

                ref_text, ref_ids, ref_lp, ref_plen = generate_reflection(
                    model, tokenizer,
                    train_py=parent.code,
                    description=rollout1.description,
                    feedback_text=feedback,
                    best_val_bpb=best_bpb,
                    temperature=args.temperature,
                )
                ep.reflection_text = ref_text
                ep.reflection_full_ids = ref_ids
                ep.reflection_logprobs = ref_lp
                ep.reflection_prompt_len = ref_plen
                print(f"  Reflection: {ref_text[:100]}...")

                # ── Phase 3: Second attempt (conditioned on reflection) ──
                print("  [Attempt 2]")
                reflection_ctx = (
                    f"Previous attempt feedback:\n{feedback}\n\n"
                    f"Your reflection:\n{ref_text}\n\n"
                    f"Now propose a corrected experiment based on your reflection."
                )
                rollout2, edited2, proposal2 = generate_and_apply(
                    model, tokenizer, agent_state, parent, train_path,
                    temperature=args.temperature, max_new_tokens=args.max_new_tokens,
                    reflection_context=reflection_ctx,
                )

                eval2 = None
                if edited2 is not None and parallel_mode:
                    eval2 = evaluate_code(workers, worker_idx, parent.code, edited2, step)
                    worker_idx += 1
                    rollout2.val_bpb = eval2["val_bpb"]
                    rollout2.status = "keep" if eval2["success"] else "crash"
                    rollout2.reward = compute_reward(rollout2.val_bpb, rollout2.status)
                    val_str = f"{rollout2.val_bpb:.6f}" if rollout2.val_bpb else "---"
                    print(f"  Attempt 2 result: val_bpb={val_str}  reward={rollout2.reward:.4f}")

                ep.attempt2_rollout = rollout2
                ep.attempt2_proposal = proposal2
                ep.attempt2_edited_code = edited2
                ep.attempt2_eval = eval2
                ep.train_attempt2 = rollout2.full_ids.numel() > 0

                # ── Phase 4: Distillation target (if attempt2 succeeded) ──
                a2_val = rollout2.val_bpb
                if a2_val is not None and a2_val < best_bpb:
                    print("  [Building distillation target]")
                    d_ids, d_lp, d_plen = build_distill_ids(
                        model, tokenizer, agent_state,
                        rollout2.proposal_text, args.temperature,
                    )
                    ep.distill_full_ids = d_ids
                    ep.distill_logprobs = d_lp
                    ep.distill_prompt_len = d_plen
                    ep.train_distill = True

            # ── Update PUCT tree with best result from this episode ──
            best_rollout = rollout1
            best_code = edited1
            if ep.attempt2_rollout is not None:
                r2 = ep.attempt2_rollout
                if r2.val_bpb is not None:
                    if best_rollout.val_bpb is None or r2.val_bpb < best_rollout.val_bpb:
                        best_rollout = r2
                        best_code = ep.attempt2_edited_code

            if best_rollout.val_bpb is not None and best_code is not None:
                child = State(
                    timestep=step, code=best_code,
                    value=-best_rollout.val_bpb,
                    observation="",
                )
                sampler.update_state(child, parent)
                if best_rollout.val_bpb < best_bpb:
                    best_bpb = best_rollout.val_bpb
                    agent_state["best_val_bpb"] = best_bpb
                    print(f"  *** NEW BEST: {best_bpb:.6f} ***")
                    Path(os.path.join(args.log_dir, "best_train.py")).write_text(best_code)
            else:
                sampler.record_failed_rollout(parent)

            episodes.append(ep)

            # Log rollouts
            for tag, r in [("attempt1", rollout1), ("attempt2", ep.attempt2_rollout)]:
                if r is None:
                    continue
                results.append_result("erl", r.val_bpb, None, r.status, f"[{tag}] {r.description}")
                with open(rollout_log_path, "a") as f:
                    f.write(json.dumps({
                        "step": step, "episode": g, "tag": tag,
                        "val_bpb": r.val_bpb, "reward": r.reward,
                        "status": r.status, "description": r.description,
                        "reflected": ep.reflection_text is not None,
                        "distilled": ep.train_distill,
                    }) + "\n")

        # ── RL training step ──
        torch.cuda.empty_cache()

        valid_episodes = [ep for ep in episodes if ep.attempt1_rollout.full_ids.numel() > 0]
        if valid_episodes:
            advantages = compute_entropic_advantages(
                [ep.attempt1_rollout.reward for ep in valid_episodes]
            )
            metrics = erl_train_step(
                model, optimizer, valid_episodes, advantages,
                kl_coef=args.kl_coef,
                distill_coef=args.distill_coef,
                temperature=args.temperature,
                max_grad_norm=args.max_grad_norm,
            )
            print(f"\n  ERL update: grpo_loss={metrics['avg_grpo_loss']:.4f} "
                  f"distill_loss={metrics['avg_distill_loss']:.4f} "
                  f"grpo_tok={metrics['num_grpo_tokens']} distill_tok={metrics['num_distill_tokens']}")
        else:
            metrics = {}
            print("\n  ERL update: skipped (no valid episodes)")

        # Checkpoint
        sampler.save(step)
        model.save_pretrained(os.path.join(args.log_dir, f"lora_step_{step:06d}"))
        step_time = time.time() - step_start

        n_reflected = sum(1 for ep in episodes if ep.reflection_text is not None)
        n_distilled = sum(1 for ep in episodes if ep.train_distill)

        step_info = {
            "step": step,
            "best_bpb": best_bpb,
            "buffer_size": sampler.buffer_size(),
            "episodes": len(episodes),
            "reflected": n_reflected,
            "distilled": n_distilled,
            "step_time_s": round(step_time, 1),
            **{k: round(v, 6) if isinstance(v, float) else v for k, v in metrics.items()},
        }
        step_log.append(step_info)
        print(f"  Step time: {step_time/60:.1f} min | reflected={n_reflected} distilled={n_distilled}")

        with open(os.path.join(args.log_dir, "step_log.json"), "w") as f:
            json.dump(step_log, f, indent=2)

    # Done
    print(f"\n{'='*60}")
    print(f"Done. Best val_bpb: {best_bpb:.6f}")
    best = sampler.best_state()
    if best:
        Path(os.path.join(args.log_dir, "best_train.py")).write_text(best.code)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify import (no GPU needed)**

Run: `cd /Users/chenshengmai/Desktop/autosearch_rl/llm_scaffold && python -c "import erl_pipeline.erl_main; print('OK')"`
Expected: will fail on torch import if no GPU, but should get past the import line itself. Alternatively:
Run: `cd /Users/chenshengmai/Desktop/autosearch_rl/llm_scaffold && python -m py_compile erl_pipeline/erl_main.py && echo OK`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add erl_pipeline/erl_main.py
git commit -m "feat(erl): add ERL main loop with reflect-retry-internalize cycle"
```

---

## Task 6: SLURM Launch Script

**Files:**
- Create: `erl_pipeline/run_erl.sh`

- [ ] **Step 1: Write SLURM script**

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
BATCH_SIZE=7
WORKERS_PER_GPU=1
KL_COEF=0.1
DISTILL_COEF=1.0
PUCT_C=1.0
LR=4e-5
LORA_RANK=32
LORA_ALPHA=64
PROJ_DIR="/blue/buyuheng/li_an.ucsb/proj_yepeng"
CONDA_ENV="${PROJ_DIR}/envs/myenv"
SCAFFOLD_DIR="${PROJ_DIR}/llm_scaffold_rl/llm_scaffold"
SOURCE_REPO="${SCAFFOLD_DIR}/autoresearch"
ERL_REPO="${SCAFFOLD_DIR}/autoresearch_rl"

# ── Navigate to ERL pipeline dir ─────────────────────────────
cd "${SCAFFOLD_DIR}/erl_pipeline"

# ── Clone autoresearch if not present ─────────────────────────
if [ ! -d "$SOURCE_REPO" ]; then
    echo "Cloning autoresearch repo..."
    git clone https://github.com/karpathy/autoresearch.git "$SOURCE_REPO"
fi

# ── Load modules ──────────────────────────────────────────────
module load gcc/14.2.0
module load cuda/12.8.1
module load conda

# ── Activate conda env ────────────────────────────────────────
conda activate "$CONDA_ENV"

# ── Cache HuggingFace models on blue storage ──────────────────
export HF_HOME="${PROJ_DIR}/.cache/huggingface"

# ── Info ──────────────────────────────────────────────────────
echo "Job ID:    $SLURM_JOB_ID"
echo "Node:      $(hostname)"
echo "GPUs:      $(nvidia-smi -L 2>/dev/null | wc -l)"
echo "Model:     $MODEL"
echo "Mode:      ERL (experiential RL, no SGLang)"
echo "Started:   $(date)"
echo "---"

# ── Sync autoresearch deps ────────────────────────────────────
cd "$SOURCE_REPO" && uv sync && cd "${SCAFFOLD_DIR}/erl_pipeline"

# ── Run ERL experiment loop ───────────────────────────────────
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
    --distill-coef "$DISTILL_COEF" \
    --puct-c "$PUCT_C" \
    --lr "$LR" \
    --lora-rank "$LORA_RANK" \
    --lora-alpha "$LORA_ALPHA" \
    --temperature 0.7 \
    --max-new-tokens 8192 \
    --attn-impl sdpa \
    --log-dir ./erl_log

echo "Finished. $(date)"
```

- [ ] **Step 2: Make executable**

Run: `chmod +x /Users/chenshengmai/Desktop/autosearch_rl/llm_scaffold/erl_pipeline/run_erl.sh`

- [ ] **Step 3: Commit**

```bash
git add erl_pipeline/run_erl.sh
git commit -m "feat(erl): add SLURM launch script for 8-GPU ERL pipeline"
```

---

## Task 7: Smoke Test (Syntax + Import Validation)

**Files:** None created, just validation.

- [ ] **Step 1: Compile all ERL files**

Run:
```bash
cd /Users/chenshengmai/Desktop/autosearch_rl/llm_scaffold
python -m py_compile erl_pipeline/__init__.py
python -m py_compile erl_pipeline/erl_types.py
python -m py_compile erl_pipeline/erl_feedback.py
python -m py_compile erl_pipeline/erl_reflect.py
python -m py_compile erl_pipeline/erl_trainer.py
python -m py_compile erl_pipeline/erl_main.py
echo "All files compile OK"
```

Expected: `All files compile OK`

- [ ] **Step 2: Verify feedback module logic**

Run:
```bash
cd /Users/chenshengmai/Desktop/autosearch_rl/llm_scaffold
python -c "
from erl_pipeline.erl_feedback import build_feedback, should_reflect

# Test feedback generation
fb = build_feedback('crash', None, 0.989, 'Increase LR', eval_output='RuntimeError: CUDA OOM')
assert 'CRASH' in fb
assert 'CUDA OOM' in fb

# Test gating
assert should_reflect('crash', None, 0.989) == True
assert should_reflect('edit_failed', None, 0.989) == True
assert should_reflect('keep', 0.980, 0.989) == False  # improved = no reflect
assert should_reflect('keep', 0.995, 0.989) == True    # regressed = reflect

print('All feedback tests pass')
"
```

Expected: `All feedback tests pass`

- [ ] **Step 3: Commit (no changes, just verification)**

No commit needed — this is validation only.

---

## Summary

| Task | Module | What it does |
|------|--------|-------------|
| 1 | `erl_types.py` | Episode dataclass wrapping first attempt + reflection + second attempt + distill targets |
| 2 | `erl_feedback.py` | Structured text feedback from eval results (crash logs, regression deltas, edit errors) |
| 3 | `erl_reflect.py` | Model generates reflection diagnosing failure; builds reflected proposal context |
| 4 | `erl_trainer.py` | Three-signal training: GRPO on both attempts + RAFT distillation on successful corrections |
| 5 | `erl_main.py` | Full ERL loop: attempt → feedback → reflect → retry → internalize → PUCT update |
| 6 | `run_erl.sh` | SLURM 8-GPU launch script |
| 7 | Smoke test | Syntax + import + logic validation |

**What's reused from rl_pipeline (no duplication):**
- `rl_model.py` — model loading, generation, logprobs
- `rl_eval.py` — Ray EvalWorker
- `rl_sampler.py` — PUCT tree
- `rl_trainer.py` — `compute_entropic_advantages` only
- `rl_planner.py` — `propose_experiment_rl`
- `rl_types.py` — `Rollout`

**What's reused from frozen pipeline:**
- `planner.py` — prompt building, edit validation/application
- `results.py` — metric parsing, results logging
- `state.py` — file I/O
