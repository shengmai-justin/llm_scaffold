"""Prompt building + local model generation for RL mode.

Reuses planner.py for prompt assembly and validation.
Uses rl_model.py for local generation with logprobs.
"""
from __future__ import annotations

import json
import os
import re
import sys

import torch

# Add parent dir for frozen pipeline imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import planner
import state as state_mod
from rl_model import generate_with_logprobs, underlying
from rl_types import Rollout

_PIPELINE_DIR = os.path.dirname(os.path.abspath(__file__))
PROMPT_IDEATOR_FILE = os.path.join(_PIPELINE_DIR, "prompt_ideator.md")
PROMPT_IMPLEMENTER_FILE = os.path.join(_PIPELINE_DIR, "prompt_implementer.md")

# Split-pipeline defaults (see docs/SPLIT_PIPELINE.md "Open decisions").
IDEATOR_MAX_NEW_TOKENS = 8192
IMPLEMENTER_MAX_NEW_TOKENS = 4096
IMPLEMENTER_TEMPERATURE = 0.7
IMPLEMENTER_MAX_ATTEMPTS = 3  # initial + 2 retries


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


# ---------------------------------------------------------------------------
# Split pipeline: Ideator (RL-trained) + Implementer (frozen)
# ---------------------------------------------------------------------------

def _build_ideator_messages(
    agent_state: dict,
    history_context: str | None,
    error_context: str | None,
) -> tuple[str, str]:
    train_py = state_mod.read_file(os.path.join(agent_state["repo_path"], "train.py"))
    recent = planner.summarize_recent_results()
    system_msg = state_mod.read_file(PROMPT_IDEATOR_FILE)
    user_msg = (
        f"Current train.py:\n```python\n{train_py}\n```\n\n"
        f"Current best val_bpb: {agent_state['best_val_bpb']:.6f}\n\n"
        f"{recent}"
    )
    if history_context:
        user_msg += f"\n\n{history_context}"
    if error_context:
        user_msg += f"\n\n{error_context}"
    user_msg += (
        "\n\nPropose one focused change. 2-3 sentences describing WHAT to change "
        "and WHY, specific about names / values / location. No code, no JSON."
    )
    return system_msg, user_msg


def propose_idea(
    model,
    tokenizer,
    agent_state: dict,
    temperature: float = 1.0,
    max_new_tokens: int = IDEATOR_MAX_NEW_TOKENS,
    error_context: str | None = None,
    history_context: str | None = None,
    think_budget: int | None = None,
) -> tuple[str, str, str, torch.Tensor, torch.Tensor, int]:
    """Stage A: generate a natural-language idea with tracked logprobs.

    Returns (prompt_text, raw_text, idea_text, full_ids, logprobs, prompt_len).
    The idea_text has <think> blocks and code fences stripped.
    """
    system_msg, user_msg = _build_ideator_messages(
        agent_state, history_context, error_context
    )
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]
    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    raw_text, full_ids, logprobs, prompt_len = generate_with_logprobs(
        model, tokenizer, prompt_text,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        think_budget=think_budget,
    )
    idea = _strip_wrappers(raw_text)
    return prompt_text, raw_text, idea, full_ids, logprobs, prompt_len


@torch.no_grad()
def _generate_text(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    enable_thinking: bool = False,
) -> str:
    """Forward-only generation; no logprob recompute. Used by stage B."""
    if not enable_thinking:
        # Qwen3 chat template accepts enable_thinking; if already applied upstream
        # this is a no-op. Here `prompt` is already the raw text — caller applies
        # the template. This helper just runs generate.
        pass
    inputs = tokenizer(prompt, return_tensors="pt").to(model.input_device)
    prompt_len = inputs["input_ids"].shape[1]
    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        return_dict_in_generate=True,
        do_sample=True,
        temperature=temperature,
        top_k=20,
        top_p=0.95,
    )
    outputs = underlying(model).generate(**inputs, **gen_kwargs)
    new_ids = outputs.sequences[0, prompt_len:]
    return tokenizer.decode(new_ids, skip_special_tokens=True)


def implement_idea(
    model,
    tokenizer,
    agent_state: dict,
    idea_text: str,
    temperature: float = IMPLEMENTER_TEMPERATURE,
    max_new_tokens: int = IMPLEMENTER_MAX_NEW_TOKENS,
) -> tuple[dict | None, str, str | None]:
    """Stage B: translate an idea into JSON edits. No gradient.

    Returns (proposal_or_None, raw_text, error_or_None). Thinking is disabled
    via the Qwen3 chat template so the implementer outputs JSON directly.
    """
    train_py = state_mod.read_file(os.path.join(agent_state["repo_path"], "train.py"))
    system_msg = state_mod.read_file(PROMPT_IMPLEMENTER_FILE)
    user_msg = (
        f"Proposed change:\n{idea_text}\n\n"
        f"Current train.py:\n```python\n{train_py}\n```\n\n"
        f"Produce the JSON edits."
    )
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]
    try:
        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        # Older template without enable_thinking kwarg — fall back.
        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )

    raw_text = _generate_text(
        model, tokenizer, prompt_text,
        max_new_tokens=max_new_tokens, temperature=temperature,
    )
    try:
        candidate = _extract_json(raw_text)
        proposal = json.loads(candidate)
        planner.validate_planner_output(proposal)
        return proposal, raw_text, None
    except (json.JSONDecodeError, ValueError, KeyError) as e:
        return None, raw_text, f"{type(e).__name__}: {e}"


def propose_experiment_split(
    model,
    tokenizer,
    agent_state: dict,
    temperature: float = 1.0,
    max_new_tokens: int = IDEATOR_MAX_NEW_TOKENS,
    error_context: str | None = None,
    history_context: str | None = None,
    think_budget: int | None = None,
) -> tuple[dict | None, Rollout]:
    """Split-pipeline proposal. Stage A receives RL signal; stage B is frozen.

    The returned Rollout's full_ids / old_logprobs / prompt_len refer to
    stage A (the ideator) only — stage B runs under no_grad and its tokens
    are not included in the training signal.

    On stage-B failure, we retry up to IMPLEMENTER_MAX_ATTEMPTS times before
    marking the rollout edit_failed (zero reward on the idea).
    """
    # Ideator has its own fixed budget; caller's max_new_tokens is ignored
    # so the monolithic flag doesn't accidentally resize stage A.
    del max_new_tokens
    prompt_text, raw_text, idea, full_ids, logprobs, prompt_len = propose_idea(
        model, tokenizer, agent_state,
        temperature=temperature,
        max_new_tokens=IDEATOR_MAX_NEW_TOKENS,
        error_context=error_context,
        history_context=history_context,
        think_budget=think_budget,
    )

    rollout = Rollout(
        prompt_text=prompt_text,
        proposal_text=raw_text,
        full_ids=full_ids,
        old_logprobs=logprobs,
        prompt_len=prompt_len,
        val_bpb=None,
        status="pending",
        reward=0.0,
        description=idea[:300] if idea else "",
    )

    if not idea:
        rollout.status = "edit_failed"
        rollout.description = "empty_idea"
        return None, rollout

    errors: list[str] = []
    for attempt in range(IMPLEMENTER_MAX_ATTEMPTS):
        proposal, _impl_text, err = implement_idea(
            model, tokenizer, agent_state, idea,
            temperature=IMPLEMENTER_TEMPERATURE,
            max_new_tokens=IMPLEMENTER_MAX_NEW_TOKENS,
        )
        if proposal is not None:
            rollout.description = (
                f"[idea] {idea[:200]} | [impl] {proposal['description']}"
            )
            return proposal, rollout
        errors.append(f"try{attempt + 1}:{err}")

    rollout.status = "edit_failed"
    rollout.description = (
        f"impl_failed | idea={idea[:200]} | {' ; '.join(errors)}"
    )
    return None, rollout
