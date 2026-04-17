"""Budget-controlled thinking for Qwen3-style reasoning models.

Inserts a `LogitsProcessor` into `model.generate` that:
  - leaves logits unchanged while the model is within `think_budget`,
  - softly boosts the `</think>` logit at `soft_threshold` of the budget,
  - hard-forces `</think>` once the budget is exceeded,
  - after the first `</think>` appears, masks the `<think>` opening token
    so the model cannot re-open a second thinking block (otherwise it
    frequently burns the rest of the generation budget on another
    uncontrolled think, producing empty/truncated JSON — the dominant
    failure mode on 2026-04-16 Pro 6000 runs, ~67% of parse failures).

Logprobs are recomputed in `compute_response_logprobs` without this
processor, so the on-policy ratio stays ~1.0 — including for the forced
`</think>` token, since both old_lp and new_lp use the same processor-free
code path. The post-close `<think>` mask also doesn't affect the on-policy
invariant: the blocked token was never sampled, so it never appears in
either logprob computation.

Side effect: the GRPO loss flows gradient through the (low) natural
logprob of the forced `</think>` token, gently pressuring the policy to
self-close earlier in similar contexts over training.
"""
from __future__ import annotations

import torch
from transformers import LogitsProcessor


def _resolve_single_token_id(tokenizer, literal: str) -> int | None:
    """Return the single-token id for `literal`, or None if it doesn't map
    cleanly to one token in this tokenizer. Tries the special-token table
    first, then plain encoding."""
    tid = tokenizer.convert_tokens_to_ids(literal)
    if tid is not None and tid != tokenizer.unk_token_id:
        return tid
    ids = tokenizer.encode(literal, add_special_tokens=False)
    return ids[0] if len(ids) == 1 else None


class BudgetThinkingProcessor(LogitsProcessor):
    """Force `</think>` once the model has spent its think_budget, then
    block `<think>` to prevent re-opening a second thinking block.

    The processor is single-use: instantiate one per `model.generate` call.
    Assumes batch size 1 (we always generate one rollout at a time).
    """

    def __init__(
        self,
        tokenizer,
        prompt_len: int,
        think_budget: int,
        soft_threshold: float = 0.95,
        soft_boost: float = 5.0,
    ) -> None:
        self.prompt_len = prompt_len
        self.think_budget = think_budget
        self.soft_start = int(soft_threshold * think_budget)
        self.soft_boost = soft_boost

        end_id = _resolve_single_token_id(tokenizer, "</think>")
        if end_id is None:
            raise ValueError(
                "</think> must tokenize to a single token in this tokenizer"
            )
        self.end_think_id = end_id

        # Optional — if <think> doesn't cleanly map to one token, we skip
        # the re-opening block (some tokenizers split it). The hard-force
        # of </think> still works in that case; we just lose the post-close
        # protection.
        self.start_think_id = _resolve_single_token_id(tokenizer, "<think>")

        self.closed = False  # set True once </think> appears in the sequence

    def __call__(
        self, input_ids: torch.Tensor, scores: torch.Tensor
    ) -> torch.Tensor:
        if self.closed:
            # Block <think> to prevent the model from opening a second
            # think block after the first (forced or natural) close.
            if self.start_think_id is not None:
                scores[:, self.start_think_id] = torch.finfo(scores.dtype).min
            return scores

        # Detect that </think> was just sampled at the previous step.
        seq_len = input_ids.shape[1]
        if seq_len > self.prompt_len and input_ids[0, -1].item() == self.end_think_id:
            self.closed = True
            if self.start_think_id is not None:
                scores[:, self.start_think_id] = torch.finfo(scores.dtype).min
            return scores

        n_generated = seq_len - self.prompt_len
        if n_generated >= self.think_budget:
            # Hard force: mask everything except </think>.
            mask_value = torch.finfo(scores.dtype).min
            scores[:, :] = mask_value
            scores[:, self.end_think_id] = 0.0
        elif n_generated >= self.soft_start:
            # Soft nudge: encourage the model to wrap up.
            scores[:, self.end_think_id] = scores[:, self.end_think_id] + self.soft_boost

        return scores
