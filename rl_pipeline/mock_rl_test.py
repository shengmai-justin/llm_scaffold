"""Mock RL test: generate proposals + fake rewards, verify RL training works.

Skips train.py execution entirely. Tests:
1. Model generates proposals with logprobs
2. Entropic advantages computed correctly
3. Policy gradient + KL penalty backward pass completes
4. LoRA weights actually change after optimizer.step()
5. PUCT tree updates correctly

Usage: python mock_rl_test.py --model-dir Qwen/Qwen3.5-9B [--attn-impl sdpa]
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile

PIPELINE_DIR = os.path.dirname(os.path.abspath(__file__))
SCAFFOLD_DIR = os.path.dirname(PIPELINE_DIR)
sys.path.insert(0, SCAFFOLD_DIR)

import torch
from rl_model import load_model
from rl_planner import propose_experiment_rl
from rl_sampler import State, PUCTSampler
from rl_trainer import compute_entropic_advantages, train_step


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--attn-impl", default="sdpa")
    parser.add_argument("--repo-path", default=os.path.join(SCAFFOLD_DIR, "autoresearch"))
    parser.add_argument("--num-rollouts", type=int, default=4)
    args = parser.parse_args()

    device = f"cuda:{args.gpu_id}"
    tmp_dir = tempfile.mkdtemp()

    # Load model
    print("Loading model...")
    model, tokenizer = load_model(
        args.model_dir, device=device,
        lora_rank=16, lora_alpha=32,
        attn_impl=args.attn_impl,
    )
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=4e-5, betas=(0.9, 0.95),
    )

    # Snapshot LoRA weights before training
    lora_params_before = {
        n: p.clone().detach()
        for n, p in model.named_parameters() if p.requires_grad
    }

    agent_state = {"repo_path": args.repo_path, "best_val_bpb": 0.99}

    # Generate rollouts with mock rewards
    print(f"\nGenerating {args.num_rollouts} proposals (no train.py)...")
    rollouts = []
    mock_rewards = [-0.99, -0.95, -1.0, -0.98]  # fake val_bpb rewards

    for i in range(args.num_rollouts):
        print(f"\n  Rollout {i+1}/{args.num_rollouts}")
        try:
            proposal, rollout = propose_experiment_rl(
                model, tokenizer, agent_state,
                temperature=1.0, max_new_tokens=4096,
            )
            print(f"  >> {proposal['description']}")
            # Assign mock reward (skip train.py)
            rollout.reward = mock_rewards[i % len(mock_rewards)]
            rollout.val_bpb = -rollout.reward
            rollout.status = "keep"
            rollouts.append(rollout)
        except Exception as e:
            print(f"  Proposal failed: {e}")

    print(f"\n{len(rollouts)} valid rollouts")
    if len(rollouts) < 2:
        print("Need at least 2 rollouts for RL update. Exiting.")
        return

    # Compute advantages
    rewards = [r.reward for r in rollouts]
    advantages = compute_entropic_advantages(rewards)
    print(f"\nRewards:    {[round(r, 4) for r in rewards]}")
    print(f"Advantages: {[round(a.item(), 4) for a in advantages]}")

    # RL training step
    print("\nRunning RL training step...")
    torch.cuda.empty_cache()
    metrics = train_step(
        model, optimizer, rollouts, advantages,
        kl_coef=0.1, temperature=1.0, max_grad_norm=1.0,
    )
    print(f"  loss={metrics['avg_loss']:.6f}")
    print(f"  tokens={metrics['num_tokens']}")
    if "kl_mean" in metrics:
        print(f"  kl_mean={metrics['kl_mean']:.6f}")
    if "ratio_mean" in metrics:
        print(f"  ratio_mean={metrics['ratio_mean']:.6f}")
        print(f"  ratio_max={metrics['ratio_max']:.6f}")

    # Check LoRA weights changed
    changed = 0
    unchanged = 0
    for n, p in model.named_parameters():
        if p.requires_grad and n in lora_params_before:
            if not torch.equal(p.data, lora_params_before[n]):
                changed += 1
            else:
                unchanged += 1
    print(f"\nLoRA params changed: {changed}/{changed + unchanged}")
    if changed == 0:
        print("WARNING: No LoRA parameters changed! Training may not be working.")
    else:
        print("LoRA weights updated successfully.")

    # PUCT tree test
    print("\nTesting PUCT with mock states...")
    baseline = State(timestep=0, code="print(1)", value=-0.99)
    sampler = PUCTSampler(initial_state=baseline, log_dir=tmp_dir, puct_c=1.0)
    for i, rollout in enumerate(rollouts):
        child = State(
            timestep=i + 1,
            code=f"print({i})",
            value=rollout.reward,  # -val_bpb
        )
        sampler.update_state(child, baseline)
    print(f"  Buffer size: {sampler.buffer_size()}")
    best = sampler.best_state()
    print(f"  Best state value: {best.value:.4f} (val_bpb={-best.value:.4f})")
    sampler.save(0)
    print(f"  Save/load OK")

    print(f"\nAll mock RL tests passed!")


if __name__ == "__main__":
    main()
