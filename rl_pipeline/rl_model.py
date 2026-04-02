"""Model loading, LoRA, generation with logprobs, training utilities.

Ported from ttt_autoresearch/model.py. Single in-process model for both
generation and training — LoRA weights are always up-to-date after each
optimizer step (no sync needed).
"""
from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType, PeftModel


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(
    model_dir: str,
    device: str = "cuda:0",
    lora_rank: int = 32,
    lora_alpha: int = 64,
    lora_path: str | None = None,
    attn_impl: str = "sdpa",
) -> tuple:
    """Load base model + LoRA adapter. Returns (model, tokenizer).

    If lora_path is given, loads an existing adapter from disk.
    Otherwise creates a fresh LoRA adapter.

    attn_impl: "sdpa" (default), "flash_attention_2", or "flash_attention_4".
    """
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        dtype=torch.bfloat16,
        device_map=device,
        attn_implementation=attn_impl,
    )

    if lora_path is not None:
        model = PeftModel.from_pretrained(model, lora_path)
    else:
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=0.0,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
        )
        model = get_peft_model(model, lora_config)

    model.gradient_checkpointing_enable()
    model.print_trainable_parameters()

    return model, tokenizer


# ---------------------------------------------------------------------------
# Generation with per-token logprobs
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_with_logprobs(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 32768,
    temperature: float = 1.0,
) -> tuple[str, torch.Tensor, torch.Tensor, int]:
    """Generate a response and collect per-token logprobs.

    Logprobs are recomputed via compute_response_logprobs (KV-cache split)
    rather than extracted from generate() logits. This ensures old_logprobs
    use the same code path as new_logprobs during training, so the importance
    sampling ratio is exactly 1.0 for on-policy updates (no bfloat16
    divergence between autoregressive and parallel forward passes).

    Returns:
        (text, token_ids, logprobs, prompt_len)
        - text: decoded response string
        - token_ids: full sequence (prompt + response) as 1D tensor
        - logprobs: per-token logprobs for response tokens only (1D, len = num_new)
        - prompt_len: number of prompt tokens
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    prompt_len = inputs["input_ids"].shape[1]

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=True,
        return_dict_in_generate=True,
    )

    full_ids = outputs.sequences[0].cpu()  # [prompt_len + new_tokens]
    del outputs

    new_ids = full_ids[prompt_len:]
    text = tokenizer.decode(new_ids, skip_special_tokens=True)

    # Recompute logprobs via the same KV-cache split used in training
    logprobs = compute_response_logprobs(
        model, full_ids, prompt_len, temperature=temperature
    ).detach().cpu()

    return text, full_ids, logprobs, prompt_len


# ---------------------------------------------------------------------------
# Compute response logprobs (with gradient, for training)
# ---------------------------------------------------------------------------

def _prompt_forward(model, input_ids, prompt_len):
    """Forward prompt without grad, return KV cache + last logit."""
    with torch.no_grad():
        out = model(input_ids=input_ids[:, :prompt_len], use_cache=True)
    return out.past_key_values, out.logits[:, -1:, :]


def _response_logprobs(model, input_ids, prompt_len, past_kv, last_logit, temperature):
    """Forward response tokens with KV cache, return per-token logprobs."""
    out = model(
        input_ids=input_ids[:, prompt_len:],
        past_key_values=past_kv,
        use_cache=False,
    )
    # last_logit predicts response[0], out.logits[:-1] predict response[1:]
    response_logits = torch.cat([last_logit, out.logits[:, :-1, :]], dim=1)[0]
    response_ids = input_ids[0, prompt_len:]

    if temperature > 0:
        log_probs = torch.log_softmax(response_logits / temperature, dim=-1)
    else:
        log_probs = torch.log_softmax(response_logits, dim=-1)

    return log_probs.gather(1, response_ids.unsqueeze(1)).squeeze(1)


def compute_response_logprobs(
    model,
    full_ids: torch.Tensor,
    prompt_len: int,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Forward pass to get per-token logprobs for response tokens.

    Processes prompt without grad (builds KV cache), then computes
    logits only for response tokens with grad. Saves ~10x memory
    vs computing logits for all positions.
    """
    input_ids = full_ids.unsqueeze(0).to(model.device)
    past_kv, last_logit = _prompt_forward(model, input_ids, prompt_len)
    return _response_logprobs(model, input_ids, prompt_len, past_kv, last_logit, temperature)


# ---------------------------------------------------------------------------
# Compute base model logprobs (no gradient, adapter disabled)
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_base_logprobs(
    model,
    full_ids: torch.Tensor,
    prompt_len: int,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Forward pass with LoRA adapters disabled to get base model logprobs.

    Same KV-cache split as compute_response_logprobs but fully no_grad.
    """
    model.disable_adapter_layers()
    try:
        input_ids = full_ids.unsqueeze(0).to(model.device)
        past_kv, last_logit = _prompt_forward(model, input_ids, prompt_len)
        return _response_logprobs(model, input_ids, prompt_len, past_kv, last_logit, temperature)
    finally:
        model.enable_adapter_layers()
