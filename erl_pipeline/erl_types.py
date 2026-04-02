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
