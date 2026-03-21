"""Data structures for RL training loop."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class Rollout:
    """One proposal + evaluation episode, with generation artifacts for training."""

    prompt_text: str             # full prompt (after chat template)
    proposal_text: str           # raw LLM response
    full_ids: torch.Tensor       # full token sequence (prompt + response), 1D
    old_logprobs: torch.Tensor   # per-token logprobs from generation, 1D
    prompt_len: int              # number of prompt tokens
    val_bpb: float | None
    status: str                  # keep/discard/crash/edit_failed
    reward: float
    description: str
