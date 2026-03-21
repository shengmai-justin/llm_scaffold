"""Model loading, LoRA, generation with logprobs, training utilities.

Ported from ttt_autoresearch/model.py. Single in-process model for both
generation and training — LoRA weights are always up-to-date after each
optimizer step (no sync needed). Uses FA4 on Blackwell GPUs.
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
    attn_impl: str = "flash_attention_4",
) -> tuple:
    """Load base model + LoRA adapter. Returns (model, tokenizer).

    If lora_path is given, loads an existing adapter from disk.
    Otherwise creates a fresh LoRA adapter.

    attn_impl: "flash_attention_4" (B200), "flash_attention_2", or "sdpa".
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
        model.print_trainable_parameters()
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
        output_logits=True,
    )

    full_ids = outputs.sequences[0]  # [prompt_len + new_tokens]
    new_ids = full_ids[prompt_len:]

    # Extract per-token logprobs from logits, free GPU memory immediately
    logprobs_list = []
    for t, logits_t in enumerate(outputs.logits):
        if temperature > 0:
            log_probs = torch.log_softmax(logits_t[0] / temperature, dim=-1)
        else:
            log_probs = torch.log_softmax(logits_t[0], dim=-1)
        logprobs_list.append(log_probs[new_ids[t]].item())
    del outputs

    logprobs = torch.tensor(logprobs_list, dtype=torch.float32)
    full_ids = full_ids.cpu()
    text = tokenizer.decode(new_ids, skip_special_tokens=True)

    return text, full_ids, logprobs, prompt_len


# ---------------------------------------------------------------------------
# Compute response logprobs (with gradient, for training)
# ---------------------------------------------------------------------------

def compute_response_logprobs(
    model,
    full_ids: torch.Tensor,
    prompt_len: int,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Forward pass to get per-token logprobs for response tokens.

    Returns tensor with gradient attached (for backprop).
    """
    input_ids = full_ids.unsqueeze(0).to(model.device)
    outputs = model(input_ids=input_ids)
    logits = outputs.logits[0]  # [seq_len, vocab]

    # logits[t] predicts token at position t+1
    response_logits = logits[prompt_len - 1 : -1]  # [num_response, vocab]
    response_ids = full_ids[prompt_len:]             # [num_response]

    if temperature > 0:
        log_probs = torch.log_softmax(response_logits / temperature, dim=-1)
    else:
        log_probs = torch.log_softmax(response_logits, dim=-1)

    token_logprobs = log_probs.gather(
        1, response_ids.unsqueeze(1).to(model.device)
    ).squeeze(1)
    return token_logprobs  # [num_response], has grad


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

    Returns detached tensor (no gradient).
    """
    model.disable_adapter_layers()
    try:
        input_ids = full_ids.unsqueeze(0).to(model.device)
        outputs = model(input_ids=input_ids)
        logits = outputs.logits[0]

        response_logits = logits[prompt_len - 1 : -1]
        response_ids = full_ids[prompt_len:]

        if temperature > 0:
            log_probs = torch.log_softmax(response_logits / temperature, dim=-1)
        else:
            log_probs = torch.log_softmax(response_logits, dim=-1)

        token_logprobs = log_probs.gather(
            1, response_ids.unsqueeze(1).to(model.device)
        ).squeeze(1)
        return token_logprobs
    finally:
        model.enable_adapter_layers()
