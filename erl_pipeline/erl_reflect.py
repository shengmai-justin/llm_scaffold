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
REFLECTION_SYSTEM = """This is an experiment to have the LLM do its own research. The goal is to optimize a GPT training script (train.py) to minimize val_bpb on a fixed 5-minute training budget.

Your job is to analyze a batch of experiment results on the GPT training script. The goal is to extract maximum signal from these results to guide the next round of experiments.
Analyze ALL results together to identify patterns, then produce a structured reflection.

## Your reflection must contain:
1. **WHAT WORKED & WHY**: Which changes improved val_bpb? What do they have in common? Hypothesize the underlying mechanism (e.g., "both changes increased effective learning rate early in training").

2. **WHAT FAILED & WHY**: Group failures by failure mode — was it OOM, divergence, no improvement, or a bug? For each group, identify the root cause. Do NOT just list failures.

3. **DIMINISHING RETURNS CHECK**: Are recent improvements getting smaller? If yes, it's time to switch categories or try a bold combinatorial change. If no, continue the productive direction.

4. **FUTURE DIRECTION**: Based on your reflection, propose your suggestions for directions in future experiments.

## Output requirements
Be concise (under 500 words). Focus on actionable insights, not just summaries.
Output plain text only. No JSON, no code fences."""


def generate_batch_reflection(
    model,
    tokenizer,
    batch_feedback: str,
    best_val_bpb: float,
    temperature: float = 0.7,
    max_new_tokens: int = 1024,
    think_budget: int | None = 512,
) -> tuple[str, torch.Tensor, torch.Tensor, int]:
    """Generate one reflection for the entire batch of attempt1 results.

    The default `think_budget=512` reserves at least half of the 1024-token
    response for the reflection text after `</think>` is forced. Pass None
    to disable budgeting (legacy behavior).

    Args:
        batch_feedback: output of build_batch_feedback() — all attempt1 results
        best_val_bpb: current best metric

    Returns:
        (reflection_text, full_ids, logprobs, prompt_len)
    """
    user_msg = (
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
        think_budget=think_budget,
    )

    # Strip thinking tags if present
    clean = text
    if "</think>" in clean:
        clean = clean.split("</think>", 1)[1].strip()

    return clean, full_ids, logprobs, prompt_len


def build_reflection_context(batch_feedback: str, reflection_text: str) -> str:
    """Build the context string appended to attempt2 proposal prompts.

    This is passed as error_context to propose_experiment_rl, so the model
    sees the reflection before proposing its second attempt.  The raw
    batch_feedback is omitted — the reflection already distills it.
    """
    return (
        f"--- Reflection on first attempts ---\n"
        f"{reflection_text}\n\n"
        f"Based on the reflection above, propose an improved experiment."
    )
