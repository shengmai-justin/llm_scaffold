"""Budget-controlled thinking for Qwen3-style reasoning models.

Inserts a `LogitsProcessor` into `model.generate` that:
  - leaves logits unchanged while the model is within `think_budget`,
  - softly boosts the `</think>` logit at `soft_threshold` of the budget,
  - hard-forces `</think>` once the budget is exceeded.

Once `</think>` is emitted (naturally or forced), the processor becomes a
no-op for the remaining JSON portion of the generation. Logprobs are
recomputed in `compute_response_logprobs` without this processor, so the
on-policy ratio stays ~1.0 — including for the forced token, since both
old_lp and new_lp use the same processor-free code path.

Side effect: the GRPO loss flows gradient through the (low) natural
logprob of the forced `</think>` token, gently pressuring the policy to
self-close earlier in similar contexts over training.
"""
from __future__ import annotations

import torch
from transformers import LogitsProcessor


class BudgetThinkingProcessor(LogitsProcessor):
    """Force `</think>` once the model has spent its think_budget.

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

        end_id = tokenizer.convert_tokens_to_ids("</think>")
        if end_id is None or end_id == tokenizer.unk_token_id:
            ids = tokenizer.encode("</think>", add_special_tokens=False)
            if len(ids) != 1:
                raise ValueError(
                    f"</think> must tokenize to a single token, got {ids}"
                )
            end_id = ids[0]
        self.end_think_id = end_id
        self.closed = False  # set True once </think> appears in the sequence

    def __call__(
        self, input_ids: torch.Tensor, scores: torch.Tensor
    ) -> torch.Tensor:
        if self.closed:
            return scores

        # Detect that </think> was just sampled at the previous step.
        seq_len = input_ids.shape[1]
        if seq_len > self.prompt_len and input_ids[0, -1].item() == self.end_think_id:
            self.closed = True
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
