"""Entropic advantages, reward, and RL training step.

Ported from ttt_autoresearch/train.py.
"""
from __future__ import annotations

import math

import torch
from torch.nn.utils import clip_grad_norm_

from rl_model import compute_response_logprobs, compute_base_logprobs
from rl_types import Rollout


# ---------------------------------------------------------------------------
# Reward
# ---------------------------------------------------------------------------

def compute_reward(val_bpb: float | None, status: str) -> float:
    """Reward = -val_bpb for success, -1.0 for failure."""
    if status in ("crash", "edit_failed") or val_bpb is None:
        return -1.0
    return -val_bpb


# ---------------------------------------------------------------------------
# Entropic adaptive beta advantages
# ---------------------------------------------------------------------------

def compute_entropic_advantages(rewards: list[float]) -> torch.Tensor:
    """LOO entropic advantages with adaptive beta.

    Binary search for beta s.t. KL(q_beta || uniform) ~ log(2).
    Ported from ttt_autoresearch/train.py:96-147.
    """
    r = torch.tensor(rewards, dtype=torch.float32)
    k = r.shape[0]

    if k < 2:
        return torch.zeros_like(r)

    delta = math.log(2)
    beta_max = 1e6
    iters = 60
    eps = 1e-12
    logK = math.log(k)

    def kl_hat(beta_scalar: float) -> float:
        b = torch.tensor(beta_scalar, dtype=r.dtype)
        logits = b * (r - r.max())
        logq = logits - torch.logsumexp(logits, dim=0)
        q = torch.exp(logq)
        kl = (q * (logq + logK)).sum()
        return float(kl.item())

    lo, hi = 0.0, 1.0
    if kl_hat(hi) < delta:
        while hi < beta_max and kl_hat(hi) < delta:
            hi *= 2.0
        if kl_hat(hi) < delta:
            beta = hi
        else:
            beta = None
    else:
        beta = None

    if beta is None:
        for _ in range(iters):
            mid = 0.5 * (lo + hi)
            if kl_hat(mid) < delta:
                lo = mid
            else:
                hi = mid
        beta = hi

    e = torch.exp(beta * (r - r.max()))
    if k == 1:
        Z = e
    else:
        Z = (e.sum() - e) / (k - 1)
    w = e / (Z + eps)
    return w - 1.0


# ---------------------------------------------------------------------------
# RL training step
# ---------------------------------------------------------------------------

def train_step(
    model,
    optimizer,
    rollouts: list[Rollout],
    advantages: torch.Tensor,
    kl_coef: float = 0.1,
    temperature: float = 1.0,
    max_grad_norm: float = 1.0,
) -> dict:
    """One policy gradient step with per-token KL penalty.

    Ported from ttt_autoresearch/train.py:337-391.
    """
    optimizer.zero_grad()
    total_loss = 0.0
    num_tokens = 0
    all_ratios = []
    all_kls = []

    for ri, (rollout, adv) in enumerate(zip(rollouts, advantages)):
        if abs(adv.item()) < 1e-8:
            continue

        alloc = torch.cuda.memory_allocated() / 1024**3
        print(f"    [GPU] train rollout {ri}: before forward  alloc={alloc:.1f}GB  tokens={rollout.full_ids.shape[0] - rollout.prompt_len}")

        old_lp = rollout.old_logprobs.to(model.device)
        new_lp = compute_response_logprobs(
            model, rollout.full_ids, rollout.prompt_len,
            temperature=temperature,
        )

        ratio = torch.exp(new_lp - old_lp)
        all_ratios.append(ratio.detach().cpu())

        # KL penalty folded into advantage (per-token)
        shaped_adv = adv.to(model.device)
        if kl_coef > 0:
            base_lp = compute_base_logprobs(
                model, rollout.full_ids, rollout.prompt_len,
                temperature=temperature,
            )
            kl_per_token = new_lp - base_lp
            all_kls.append(kl_per_token.mean().item())
            shaped_adv = shaped_adv - kl_coef * kl_per_token

        alloc = torch.cuda.memory_allocated() / 1024**3
        print(f"    [GPU] train rollout {ri}: before backward alloc={alloc:.1f}GB")

        loss = -(ratio * shaped_adv).mean()
        loss.backward()
        total_loss += loss.item() * len(new_lp)
        num_tokens += len(new_lp)

        alloc = torch.cuda.memory_allocated() / 1024**3
        print(f"    [GPU] train rollout {ri}: after backward  alloc={alloc:.1f}GB")

    if num_tokens > 0:
        clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad],
            max_grad_norm,
        )
        optimizer.step()

    # Metrics
    metrics = {"avg_loss": total_loss / max(num_tokens, 1), "num_tokens": num_tokens}
    if all_ratios:
        cat = torch.cat(all_ratios)
        metrics["ratio_mean"] = cat.mean().item()
        metrics["ratio_max"] = cat.max().item()
    if all_kls:
        metrics["kl_mean"] = sum(all_kls) / len(all_kls)
    return metrics
