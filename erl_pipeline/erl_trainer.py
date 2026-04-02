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
        # Reflection advantage: compare to mean of ALL attempt1 rewards (not just filtered)
        all_a1_rewards = [ep.attempt1_rollout.reward for ep in episodes]
        a1_mean = sum(all_a1_rewards) / len(all_a1_rewards) if all_a1_rewards else 0.0
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
