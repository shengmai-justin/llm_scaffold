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


def _strip_wrappers(text: str) -> str:
    """Strip <think> blocks and markdown code fences from model output."""
    clean = text
    if "</think>" in clean:
        clean = clean.split("</think>", 1)[1]
    clean = re.sub(r"<think>.*?</think>", "", clean, flags=re.DOTALL)
    clean = re.sub(r"^```[a-zA-Z]*\n?", "", clean)
    clean = re.sub(r"\n?```\s*$", "", clean)
    return clean.strip()


def _extract_json(text: str) -> str:
    """Best-effort JSON extraction. Tries two stages:

    1. Strip <think>/code fences, return if `json.loads` succeeds.
    2. Walk the original text backward to find the last balanced
       top-level {...} block. Salvages `trailing_garbage` failures
       (model emitted JSON then continued generating prose).

    Returns the best candidate; caller is responsible for json.loads.
    """
    cleaned = _strip_wrappers(text)
    try:
        json.loads(cleaned)
        return cleaned
    except json.JSONDecodeError:
        pass

    # Walk backward from last '}' to find matching opening brace.
    end = text.rfind("}")
    if end == -1:
        return cleaned
    depth = 0
    for i in range(end, -1, -1):
        if text[i] == "}":
            depth += 1
        elif text[i] == "{":
            depth -= 1
            if depth == 0:
                candidate = text[i:end + 1]
                try:
                    json.loads(candidate)
                    return candidate
                except json.JSONDecodeError:
                    break
    return cleaned


def propose_experiment_rl(
    model,
    tokenizer,
    agent_state: dict,
    temperature: float = 1.0,
    max_new_tokens: int = 32768,
    error_context: str | None = None,
    history_context: str | None = None,
    think_budget: int | None = None,
) -> tuple[dict | None, Rollout]:
    """Generate a proposal using the local model. Returns (proposal, rollout).

    history_context is the cross-step dead-end summary (from
    erl_history.generate_history_summary). error_context is the
    per-batch reflection (from erl_reflect.build_reflection_context).
    Both are appended to the user message; history first so it sets
    strategic constraints before the tactical reflection.

    On parse/validation failure, returns (None, rollout). The rollout
    retains its generated full_ids and logprobs so it still contributes
    to GRPO advantage groups — K stays at batch_size, avoiding the TTT
    beta_max blowup at K=2.

    Still raises if generation itself fails (CUDA OOM, tokenizer error),
    since in that case there are no tokens to preserve.
    """
    system_msg, user_msg = planner.build_planner_context(
        agent_state["repo_path"], agent_state["best_val_bpb"]
    )
    if history_context:
        user_msg += f"\n\n{history_context}"
    if error_context:
        user_msg += f"\n\n{error_context}"

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
        think_budget=think_budget,
    )

    # Build the rollout BEFORE parsing so full_ids are preserved on failure.
    rollout = Rollout(
        prompt_text=prompt_text,
        proposal_text=text,
        full_ids=full_ids,
        old_logprobs=logprobs,
        prompt_len=prompt_len,
        val_bpb=None,
        status="pending",
        reward=0.0,
        description="",
    )

    try:
        candidate = _extract_json(text)
        proposal = json.loads(candidate)
        planner.validate_planner_output(proposal)
    except (json.JSONDecodeError, ValueError, KeyError) as e:
        rollout.status = "edit_failed"
        rollout.description = f"json_parse_error: {type(e).__name__}: {e}"
        return None, rollout

    rollout.description = proposal["description"]
    return proposal, rollout
