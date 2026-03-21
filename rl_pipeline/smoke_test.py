"""Smoke test for RL training pipeline. Run on GPU.

Usage: python smoke_test.py --model-dir Qwen/Qwen3.5-9B [--gpu-id 0]

Tests each component end-to-end:
1. Model loading with LoRA + FA4
2. Generation with logprobs
3. Forward pass logprobs (with gradient)
4. Base model logprobs (adapter disabled)
5. Entropic advantages
6. Training step (policy gradient + KL penalty)
7. PUCT sampler
8. LoRA save/load roundtrip
"""
from __future__ import annotations

import argparse
import os
import tempfile
import torch

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--attn-impl", default="flash_attention_4")
    args = parser.parse_args()

    device = f"cuda:{args.gpu_id}"
    tmp_dir = tempfile.mkdtemp()

    # 1. Model loading
    print("1. Loading model...")
    from rl_model import load_model
    model, tokenizer = load_model(
        args.model_dir, device=device,
        lora_rank=16, lora_alpha=32,
        attn_impl=args.attn_impl,
    )
    print(f"   OK — {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable params")

    # 2. Generation with logprobs
    print("2. Generating with logprobs...")
    from rl_model import generate_with_logprobs
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Say hello in JSON: {\"message\": \"...\"}"}],
        tokenize=False, add_generation_prompt=True,
    )
    text, full_ids, logprobs, prompt_len = generate_with_logprobs(
        model, tokenizer, prompt, max_new_tokens=64, temperature=1.0,
    )
    num_new = full_ids.shape[0] - prompt_len
    assert logprobs.shape[0] == num_new, f"logprobs shape {logprobs.shape} != {num_new}"
    assert full_ids.device == torch.device("cpu"), "full_ids should be on CPU"
    print(f"   OK — {num_new} tokens, logprobs range [{logprobs.min():.2f}, {logprobs.max():.2f}]")
    print(f"   Response: {text[:100]}...")

    # 3. Forward pass logprobs (with gradient)
    print("3. Computing response logprobs (with grad)...")
    from rl_model import compute_response_logprobs
    new_lp = compute_response_logprobs(model, full_ids, prompt_len, temperature=1.0)
    assert new_lp.requires_grad, "new_lp should have gradient"
    assert new_lp.shape[0] == num_new, f"shape mismatch: {new_lp.shape[0]} != {num_new}"
    print(f"   OK — shape {new_lp.shape}, requires_grad={new_lp.requires_grad}")

    # 4. Base model logprobs (adapter disabled)
    print("4. Computing base model logprobs...")
    from rl_model import compute_base_logprobs
    base_lp = compute_base_logprobs(model, full_ids, prompt_len, temperature=1.0)
    assert not base_lp.requires_grad, "base_lp should NOT have gradient"
    assert base_lp.shape == new_lp.shape
    kl = (new_lp.detach() - base_lp).mean().item()
    print(f"   OK — shape {base_lp.shape}, KL(policy||base) = {kl:.4f}")

    # Clean up grad graph
    new_lp.sum().backward()
    model.zero_grad()

    # 5. Entropic advantages
    print("5. Computing entropic advantages...")
    from rl_trainer import compute_entropic_advantages, compute_reward
    rewards = [-1.5, -1.3, -1.0, -2.0]
    advantages = compute_entropic_advantages(rewards)
    assert advantages.shape[0] == 4
    print(f"   OK — rewards={rewards} -> advantages={[round(a.item(), 3) for a in advantages]}")

    # 6. Training step
    print("6. Running training step...")
    from rl_trainer import train_step
    from rl_types import Rollout

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=1e-5,
    )

    # Create a minimal rollout
    rollout = Rollout(
        prompt_text=prompt, proposal_text=text,
        full_ids=full_ids, old_logprobs=logprobs,
        prompt_len=prompt_len,
        val_bpb=1.3, status="keep", reward=-1.3,
        description="smoke test",
    )
    adv = compute_entropic_advantages([-1.3])
    metrics = train_step(
        model, optimizer, [rollout], adv,
        kl_coef=0.1, temperature=1.0, max_grad_norm=1.0,
    )
    print(f"   OK — loss={metrics['avg_loss']:.4f}, tokens={metrics['num_tokens']}")
    if "kl_mean" in metrics:
        print(f"   KL mean={metrics['kl_mean']:.4f}")

    # 7. PUCT sampler
    print("7. Testing PUCT sampler...")
    from rl_sampler import State, PUCTSampler
    baseline = State(timestep=0, code="print(1)", value=-1.5)
    sampler = PUCTSampler(initial_state=baseline, log_dir=tmp_dir, puct_c=1.0)
    parent = sampler.sample_state()
    child = State(timestep=1, code="print(2)", value=-1.3)
    sampler.update_state(child, parent)
    sampler.save(0)
    # Load roundtrip
    sampler2 = PUCTSampler(initial_state=baseline, log_dir=tmp_dir, resume_step=0)
    assert sampler2.buffer_size() == 2
    print(f"   OK — buffer_size={sampler2.buffer_size()}, save/load roundtrip passed")

    # 8. LoRA save/load
    print("8. Testing LoRA save/load...")
    adapter_path = os.path.join(tmp_dir, "test_adapter")
    model.save_pretrained(adapter_path)
    assert os.path.exists(os.path.join(adapter_path, "adapter_config.json"))
    print(f"   OK — saved to {adapter_path}")

    print(f"\nAll tests passed!")


if __name__ == "__main__":
    main()
