"""ERL GPU smoke test: real model, mocked eval. Needs 1 GPU.

Tests:
1. Model loads with LoRA
2. Proposal generation with logprobs
3. Reflection generation with logprobs
4. GRPO advantages computation
5. Full erl_train_step with 4 signals (mock rewards, real forward/backward)
6. LoRA weights actually change after training
7. Checkpoint save/load roundtrip

Usage (on cluster with 1 GPU):
  srun --partition=hpg-b200 --gpus=1 --mem=100gb --time=01:00:00 \
    bash -c 'cd /path/to/llm_scaffold && python erl_pipeline/smoke_test_erl.py \
      --model-dir Qwen/Qwen3.5-9B --repo-path ./autoresearch_rl'
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile

PIPELINE_DIR = os.path.dirname(os.path.abspath(__file__))
SCAFFOLD_DIR = os.path.dirname(PIPELINE_DIR)
RL_PIPELINE_DIR = os.path.join(SCAFFOLD_DIR, "rl_pipeline")
sys.path.insert(0, SCAFFOLD_DIR)
sys.path.insert(0, RL_PIPELINE_DIR)
sys.path.insert(0, PIPELINE_DIR)

import torch
from rl_model import load_model, generate_with_logprobs
from rl_planner import propose_experiment_rl
from rl_trainer import compute_reward
from rl_types import Rollout
from erl_types import Episode, StepReflection
from erl_feedback import build_batch_feedback
from erl_reflect import generate_batch_reflection, build_reflection_context
from erl_trainer import compute_grpo_advantages, erl_train_step


def gpu_mem(label=""):
    alloc = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    print(f"  [GPU] {label:40s} alloc={alloc:.1f}GB  reserved={reserved:.1f}GB")


def main():
    parser = argparse.ArgumentParser(description="ERL GPU Smoke Test")
    parser.add_argument("--model-dir", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--attn-impl", default="sdpa")
    parser.add_argument("--repo-path", default=os.path.join(SCAFFOLD_DIR, "autoresearch_rl"))
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--num-proposals", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    args = parser.parse_args()

    device = f"cuda:{args.gpu_id}"
    tmp_dir = tempfile.mkdtemp()
    passed = 0
    total = 0

    def check(name, condition, detail=""):
        nonlocal passed, total
        total += 1
        if condition:
            print(f"  PASS: {name}")
            passed += 1
        else:
            print(f"  FAIL: {name} {detail}")

    # ── 1. Model loading ──
    print("=" * 60)
    print("ERL GPU Smoke Test")
    print("=" * 60)

    print("\n--- 1. Loading model ---")
    gpu_mem("before load")
    model, tokenizer = load_model(
        args.model_dir, device=device,
        lora_rank=args.lora_rank, lora_alpha=args.lora_alpha,
        attn_impl=args.attn_impl,
    )
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=4e-5, betas=(0.9, 0.95),
    )
    gpu_mem("after load")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Trainable: {trainable:,} / {total_params:,} ({100*trainable/total_params:.2f}%)")
    check("model loaded", model is not None)
    check("has trainable params", trainable > 0)

    # Snapshot LoRA weights
    lora_before = {
        n: p.clone().detach()
        for n, p in model.named_parameters() if p.requires_grad
    }

    # ── 2. Proposal generation ──
    print("\n--- 2. Generating proposals ---")
    agent_state = {"repo_path": args.repo_path, "best_val_bpb": 0.99}
    rollouts = []
    mock_bpbs = [0.97, 0.95, 1.02]

    for i in range(args.num_proposals):
        print(f"\n  Proposal {i+1}/{args.num_proposals}")
        try:
            proposal, rollout = propose_experiment_rl(
                model, tokenizer, agent_state,
                temperature=args.temperature,
                max_new_tokens=args.max_new_tokens,
            )
            print(f"  >> {proposal['description']}")

            # Assign mock reward (no train.py execution)
            rollout.val_bpb = mock_bpbs[i % len(mock_bpbs)]
            rollout.status = "keep"
            rollout.reward = compute_reward(rollout.val_bpb, rollout.status)
            rollouts.append(rollout)
            gpu_mem(f"after proposal {i+1}")
        except Exception as e:
            print(f"  Proposal failed: {e}")

    check("generated proposals", len(rollouts) >= 2, f"got {len(rollouts)}")

    if len(rollouts) < 2:
        print("\nNeed at least 2 proposals. Exiting.")
        sys.exit(1)

    # Check rollout structure
    r0 = rollouts[0]
    check("rollout has full_ids", r0.full_ids.numel() > 0)
    check("rollout has logprobs", r0.old_logprobs.numel() > 0)
    check("prompt_len > 0", r0.prompt_len > 0)
    check("full_ids > prompt_len", r0.full_ids.numel() > r0.prompt_len)

    # ── 3. Reflection generation ──
    print("\n--- 3. Generating batch reflection ---")
    attempt_summaries = []
    for i, r in enumerate(rollouts):
        attempt_summaries.append({
            "description": r.description,
            "status": r.status,
            "val_bpb": r.val_bpb,
            "eval_output": f"mock output for proposal {i}",
        })

    import planner
    train_path = os.path.join(args.repo_path, "train.py")
    if os.path.exists(train_path):
        train_py = open(train_path).read()
    else:
        train_py = "# mock train.py\nprint('hello')"

    batch_feedback = build_batch_feedback(attempt_summaries, 0.99)
    print(f"  Feedback length: {len(batch_feedback)} chars")

    ref_text, ref_ids, ref_lp, ref_plen = generate_batch_reflection(
        model, tokenizer,
        train_py=train_py,
        batch_feedback=batch_feedback,
        best_val_bpb=0.99,
        temperature=args.temperature,
    )
    gpu_mem("after reflection")

    check("reflection text non-empty", len(ref_text) > 0)
    check("reflection has full_ids", ref_ids.numel() > 0)
    check("reflection has logprobs", ref_lp.numel() > 0)
    check("reflection prompt_len > 0", ref_plen > 0)
    print(f"  Reflection: {ref_text[:150]}...")

    reflection_ctx = build_reflection_context(batch_feedback, ref_text)
    check("reflection context non-empty", len(reflection_ctx) > 0)

    # ── 4. Build episodes with mock data ──
    print("\n--- 4. Building episodes ---")
    episodes = []
    for i, r in enumerate(rollouts):
        ep = Episode(
            attempt1_rollout=r,
            attempt1_proposal={"description": r.description, "edits": []},
            attempt1_edited_code=f"mock_code_{i}",
            attempt1_eval={"val_bpb": r.val_bpb, "output": "mock", "success": True},
        )

        # Generate attempt2 with reflection context
        try:
            proposal2, rollout2 = propose_experiment_rl(
                model, tokenizer, agent_state,
                temperature=args.temperature,
                max_new_tokens=args.max_new_tokens,
                error_context=reflection_ctx,
            )
            rollout2.val_bpb = mock_bpbs[i % len(mock_bpbs)] - 0.02  # slightly better
            rollout2.status = "keep"
            rollout2.reward = compute_reward(rollout2.val_bpb, rollout2.status)
            ep.attempt2_rollout = rollout2
            ep.attempt2_proposal = proposal2
            ep.attempt2_edited_code = f"mock_code2_{i}"
            ep.train_attempt2 = True
        except Exception as e:
            print(f"  Attempt2 {i} failed: {e}")

        # First episode gets distillation (simulating it beat best)
        if i == 0 and ep.attempt2_rollout is not None:
            ep.distill_full_ids = ep.attempt2_rollout.full_ids.clone()
            ep.distill_logprobs = ep.attempt2_rollout.old_logprobs.clone()
            ep.distill_prompt_len = ep.attempt2_rollout.prompt_len
            ep.train_distill = True

        episodes.append(ep)

    a2_count = sum(1 for ep in episodes if ep.attempt2_rollout is not None)
    distill_count = sum(1 for ep in episodes if ep.train_distill)
    check("episodes built", len(episodes) >= 2)
    check("attempt2s generated", a2_count >= 1)
    check("distill target set", distill_count >= 1)

    # ── 5. GRPO advantages ──
    print("\n--- 5. GRPO advantages ---")
    a1_rewards = [ep.attempt1_rollout.reward for ep in episodes]
    advs = compute_grpo_advantages(a1_rewards)
    print(f"  Rewards:    {[round(r, 4) for r in a1_rewards]}")
    print(f"  Advantages: {[round(a.item(), 4) for a in advs]}")
    check("advantages computed", advs.shape[0] == len(episodes))
    check("advantages sum ~0", abs(advs.sum().item()) < 0.1)

    # ── 6. Full training step ──
    print("\n--- 6. ERL training step ---")
    a2_rewards = [ep.attempt2_rollout.reward for ep in episodes
                  if ep.attempt2_rollout is not None]
    ref_reward = sum(a2_rewards) / len(a2_rewards) if a2_rewards else 0.0

    step_reflection = StepReflection(
        feedback_text=batch_feedback,
        reflection_text=ref_text,
        full_ids=ref_ids,
        old_logprobs=ref_lp,
        prompt_len=ref_plen,
        reward=ref_reward,
    )

    torch.cuda.empty_cache()
    gpu_mem("before train step")

    metrics = erl_train_step(
        model, optimizer, episodes, step_reflection,
        kl_coef=0.1, temperature=args.temperature, max_grad_norm=1.0,
    )
    gpu_mem("after train step")

    print(f"  grpo_loss={metrics['avg_grpo_loss']:.6f}  tokens={metrics['num_grpo_tokens']}")
    print(f"  reflect_loss={metrics['avg_reflect_loss']:.6f}  tokens={metrics['num_reflect_tokens']}")
    print(f"  distill_loss={metrics['avg_distill_loss']:.6f}  tokens={metrics['num_distill_tokens']}")
    if "ratio_mean" in metrics:
        print(f"  ratio_mean={metrics['ratio_mean']:.6f}  ratio_max={metrics['ratio_max']:.6f}")
    if "kl_mean" in metrics:
        print(f"  kl_mean={metrics['kl_mean']:.6f}")

    check("train step completed", True)
    check("grpo tokens > 0", metrics["num_grpo_tokens"] > 0)

    # ── 7. LoRA weights changed ──
    print("\n--- 7. LoRA weight check ---")
    changed = 0
    unchanged = 0
    for n, p in model.named_parameters():
        if p.requires_grad and n in lora_before:
            if not torch.equal(p.data, lora_before[n]):
                changed += 1
            else:
                unchanged += 1

    print(f"  Changed: {changed}/{changed + unchanged}")
    check("LoRA weights updated", changed > 0, f"changed={changed}")

    # ── 8. Checkpoint save/load ──
    print("\n--- 8. Checkpoint roundtrip ---")
    ckpt_path = os.path.join(tmp_dir, "lora_test")
    model.save_pretrained(ckpt_path)
    check("checkpoint saved", os.path.exists(os.path.join(ckpt_path, "adapter_config.json")))

    # ── Summary ──
    print(f"\n{'=' * 60}")
    print(f"Results: {passed}/{total} passed")
    if passed == total:
        print("All ERL smoke tests passed!")
    else:
        print(f"WARNING: {total - passed} tests failed")
    print("=" * 60)

    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
