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
    batch_feedback: str,
    best_val_bpb: float,
    temperature: float = 0.7,
    max_new_tokens: int = 1024,
) -> tuple[str, torch.Tensor, torch.Tensor, int]:
    """Generate one reflection for the entire batch of attempt1 results.

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
