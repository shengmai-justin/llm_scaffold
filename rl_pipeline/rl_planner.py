"""Prompt building + local model generation for RL mode.

Reuses planner.py for prompt assembly and validation.
Uses rl_model.py for local generation with logprobs.
"""
from __future__ import annotations

import json
import os
import re
import sys

# Add parent dir for frozen pipeline imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import planner
from rl_model import generate_with_logprobs
from rl_types import Rollout


def propose_experiment_rl(
    model,
    tokenizer,
    agent_state: dict,
    temperature: float = 1.0,
    max_new_tokens: int = 32768,
    error_context: str | None = None,
) -> tuple[dict, Rollout]:
    """Generate a proposal using the local model. Returns (proposal, rollout).

    Raises on parse/validation failure (caller handles retry).
    """
    system_msg, user_msg = planner.build_planner_context(
        agent_state["repo_path"], agent_state["best_val_bpb"]
    )
    if error_context:
        user_msg += (
            f"\n\nPrevious edit attempt failed: {error_context}\n"
            f"Please propose a corrected experiment."
        )

    messages = [
        {"role": "system", "content": system_msg},
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

    # Strip thinking tags (Qwen3.5 outputs <think>...</think> before answer)
    clean = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    # Strip markdown code fences
    clean = re.sub(r"^```[a-zA-Z]*\n?", "", clean)
    clean = re.sub(r"\n?```\s*$", "", clean)
    clean = clean.strip()
    proposal = json.loads(clean)
    planner.validate_planner_output(proposal)

    rollout = Rollout(
        prompt_text=prompt_text,
        proposal_text=text,
        full_ids=full_ids,
        old_logprobs=logprobs,
        prompt_len=prompt_len,
        val_bpb=None,
        status="pending",
        reward=0.0,
        description=proposal["description"],
    )
    return proposal, rollout
