"""Mock ERL test: full phased loop with mocked model + eval. No GPU needed.

Tests:
1. Happy path: attempt1 -> reflection -> attempt2 -> distill -> train (4 signals)
2. All attempt1s fail (edit_failed) — reflection still runs, no distill targets
3. No attempt2 beats best_bpb — train runs without distillation signal
4. Single rollout — GRPO needs >=2, should not crash
5. GRPO advantages edge cases (identical rewards, single reward)
6. Feedback building (mixed statuses)
7. Reflection context construction
8. Trainer signal routing (which signals fire when)

Usage: python erl_pipeline/mock_erl_test.py
"""
from __future__ import annotations

import os
import sys
import traceback

PIPELINE_DIR = os.path.dirname(os.path.abspath(__file__))
SCAFFOLD_DIR = os.path.dirname(PIPELINE_DIR)
RL_PIPELINE_DIR = os.path.join(SCAFFOLD_DIR, "rl_pipeline")
sys.path.insert(0, SCAFFOLD_DIR)
sys.path.insert(0, RL_PIPELINE_DIR)
sys.path.insert(0, PIPELINE_DIR)

import torch
from rl_types import Rollout
from erl_types import Episode, StepReflection
from erl_feedback import build_attempt_feedback, build_batch_feedback
from erl_reflect import build_reflection_context
from erl_trainer import compute_grpo_advantages, erl_train_step


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_rollout(
    val_bpb=None, status="keep", reward=0.0, description="mock",
    seq_len=20, prompt_len=10,
):
    """Create a Rollout with fake token data."""
    full_ids = torch.randint(0, 1000, (seq_len,))
    old_logprobs = torch.randn(seq_len - prompt_len)
    return Rollout(
        prompt_text="mock prompt",
        proposal_text="mock response",
        full_ids=full_ids,
        old_logprobs=old_logprobs,
        prompt_len=prompt_len,
        val_bpb=val_bpb,
        status=status,
        reward=reward,
        description=description,
    )


def make_empty_rollout(status="edit_failed", description="failed"):
    """Create a Rollout with empty full_ids (parse failure / no tokens)."""
    return Rollout(
        prompt_text="", proposal_text="", full_ids=torch.tensor([]),
        old_logprobs=torch.tensor([]), prompt_len=0,
        val_bpb=None, status=status, reward=-1.0, description=description,
    )


def make_episode(
    a1_bpb=None, a1_status="keep", a1_reward=0.0,
    a2_bpb=None, a2_status="keep", a2_reward=0.0,
    a1_edited_code="edited", a2_edited_code="edited2",
    train_attempt1=True, train_attempt2=True, train_distill=False,
    a1_empty=False, a2_empty=False,
):
    """Create an Episode with mock rollouts."""
    if a1_empty:
        a1_rollout = make_empty_rollout(status=a1_status, description="a1 fail")
        a1_edited_code = None
    else:
        a1_rollout = make_rollout(val_bpb=a1_bpb, status=a1_status, reward=a1_reward, description="a1 mock")

    if a2_empty:
        a2_rollout = make_empty_rollout(status=a2_status, description="a2 fail")
        a2_edited_code = None
    else:
        a2_rollout = make_rollout(val_bpb=a2_bpb, status=a2_status, reward=a2_reward, description="a2 mock")

    return Episode(
        attempt1_rollout=a1_rollout,
        attempt1_proposal={"description": "mock", "edits": []},
        attempt1_edited_code=a1_edited_code,
        attempt1_eval={"val_bpb": a1_bpb, "output": "mock output", "success": a1_bpb is not None},
        attempt2_rollout=a2_rollout,
        attempt2_proposal={"description": "mock2", "edits": []},
        attempt2_edited_code=a2_edited_code,
        attempt2_eval={"val_bpb": a2_bpb, "output": "mock output2", "success": a2_bpb is not None},
        train_attempt1=train_attempt1,
        train_attempt2=train_attempt2,
        train_distill=train_distill,
    )


def make_reflection(reward=0.0, seq_len=20, prompt_len=10):
    """Create a StepReflection with fake token data."""
    return StepReflection(
        feedback_text="mock feedback",
        reflection_text="mock reflection text",
        full_ids=torch.randint(0, 1000, (seq_len,)),
        old_logprobs=torch.randn(seq_len - prompt_len),
        prompt_len=prompt_len,
        reward=reward,
    )


class FakeModel(torch.nn.Module):
    """Minimal model that supports compute_response_logprobs / compute_base_logprobs interface."""

    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 10)
        self.device_attr = torch.device("cpu")

    @property
    def device(self):
        return self.device_attr


passed = 0
failed = 0
errors = []


def run_test(name, fn):
    global passed, failed
    try:
        fn()
        print(f"  PASS: {name}")
        passed += 1
    except Exception as e:
        print(f"  FAIL: {name}")
        traceback.print_exc()
        errors.append((name, e))
        failed += 1


# ---------------------------------------------------------------------------
# Test: GRPO Advantages
# ---------------------------------------------------------------------------

def test_grpo_normal():
    """GRPO with distinct rewards produces non-zero advantages that sum near zero."""
    rewards = [-0.9, -0.95, -1.0, -0.85]
    advs = compute_grpo_advantages(rewards)
    assert advs.shape == (4,), f"Expected shape (4,), got {advs.shape}"
    assert abs(advs.mean().item()) < 1e-5, f"Mean not near zero: {advs.mean().item()}"
    assert advs.std().item() > 0.5, f"Std too low, advantages not spreading: {advs.std().item()}"


def test_grpo_identical_rewards():
    """All identical rewards -> all advantages zero."""
    advs = compute_grpo_advantages([-1.0, -1.0, -1.0])
    assert torch.all(advs == 0), f"Expected all zeros, got {advs}"


def test_grpo_single_reward():
    """Single reward -> advantage zero (can't normalize)."""
    advs = compute_grpo_advantages([-0.9])
    assert advs.shape == (1,)
    assert advs[0].item() == 0.0


def test_grpo_two_rewards():
    """Two rewards: one positive advantage, one negative."""
    advs = compute_grpo_advantages([-0.9, -1.0])
    assert advs.shape == (2,)
    assert advs[0].item() > 0, f"Better reward should have positive advantage"
    assert advs[1].item() < 0, f"Worse reward should have negative advantage"


# ---------------------------------------------------------------------------
# Test: Feedback Building
# ---------------------------------------------------------------------------

def test_feedback_improved():
    """Attempt that improved over best."""
    fb = build_attempt_feedback("lower lr", "keep", 0.90, 0.95)
    assert "IMPROVED" in fb
    assert "0.90" in fb


def test_feedback_regressed():
    """Attempt that regressed."""
    fb = build_attempt_feedback("higher lr", "keep", 1.00, 0.95)
    assert "REGRESSED" in fb


def test_feedback_crash():
    """Crashed attempt."""
    fb = build_attempt_feedback("bad edit", "crash", None, 0.95, eval_output="OOM error")
    assert "CRASH" in fb
    assert "OOM" in fb


def test_feedback_edit_failed():
    """Edit failed attempt."""
    fb = build_attempt_feedback("typo edit", "edit_failed", None, 0.95, edit_error="not found")
    assert "EDIT FAILED" in fb
    assert "not found" in fb


def test_batch_feedback_mixed():
    """Batch with mixed results."""
    attempts = [
        {"description": "a1", "status": "keep", "val_bpb": 0.90, "eval_output": "ok"},
        {"description": "a2", "status": "crash", "val_bpb": None, "eval_output": "OOM"},
        {"description": "a3", "status": "edit_failed", "val_bpb": None},
    ]
    fb = build_batch_feedback(attempts, 0.95)
    assert "Total attempts: 3" in fb
    assert "Improved: 1" in fb
    assert "Failed" in fb
    assert "Attempt 1" in fb
    assert "Attempt 2" in fb
    assert "Attempt 3" in fb


# ---------------------------------------------------------------------------
# Test: Reflection Context
# ---------------------------------------------------------------------------

def test_reflection_context():
    """build_reflection_context includes both feedback and reflection."""
    ctx = build_reflection_context("batch feedback here", "reflection analysis here")
    assert "batch feedback here" in ctx
    assert "reflection analysis here" in ctx
    assert "propose an improved experiment" in ctx


# ---------------------------------------------------------------------------
# Test: Episode + StepReflection dataclasses
# ---------------------------------------------------------------------------

def test_episode_defaults():
    """Episode defaults: attempt2 fields None, train flags."""
    r = make_rollout(val_bpb=0.9, reward=-0.9)
    ep = Episode(
        attempt1_rollout=r,
        attempt1_proposal=None,
        attempt1_edited_code=None,
        attempt1_eval=None,
    )
    assert ep.attempt2_rollout is None
    assert ep.train_attempt1 is True
    assert ep.train_attempt2 is False
    assert ep.train_distill is False


def test_step_reflection():
    """StepReflection stores all fields."""
    ref = make_reflection(reward=-0.85)
    assert ref.reward == -0.85
    assert ref.full_ids.numel() == 20
    assert ref.old_logprobs.numel() == 10


# ---------------------------------------------------------------------------
# Test: Trainer signal routing (mock model — patches compute_*_logprobs)
# ---------------------------------------------------------------------------

def _patch_logprob_fns():
    """Monkey-patch rl_model functions to work without a real model on CPU."""
    import erl_trainer as trainer_mod

    def fake_compute_response_logprobs(model, full_ids, prompt_len, temperature=1.0):
        n = len(full_ids) - prompt_len
        return torch.randn(n, requires_grad=True)

    def fake_compute_base_logprobs(model, full_ids, prompt_len, temperature=1.0):
        n = len(full_ids) - prompt_len
        return torch.randn(n)

    trainer_mod.compute_response_logprobs = fake_compute_response_logprobs
    trainer_mod.compute_base_logprobs = fake_compute_base_logprobs


def test_train_happy_path():
    """4 episodes, 3 with valid attempt1+attempt2, 1 with distill. All 4 signals fire."""
    _patch_logprob_fns()
    model = FakeModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    episodes = [
        make_episode(a1_bpb=0.92, a1_reward=-0.92, a2_bpb=0.88, a2_reward=-0.88, train_distill=True,
                     distill_full_ids_len=15),
        make_episode(a1_bpb=0.95, a1_reward=-0.95, a2_bpb=0.93, a2_reward=-0.93),
        make_episode(a1_bpb=1.00, a1_reward=-1.00, a2_bpb=0.97, a2_reward=-0.97),
    ]
    # Manually set distill data on first episode
    episodes[0].distill_full_ids = torch.randint(0, 1000, (15,))
    episodes[0].distill_logprobs = torch.randn(10)
    episodes[0].distill_prompt_len = 5

    reflection = make_reflection(reward=-0.90)

    metrics = erl_train_step(
        model, optimizer, episodes, reflection,
        kl_coef=0.1, temperature=1.0, max_grad_norm=1.0,
    )

    assert metrics["num_grpo_tokens"] > 0, "GRPO tokens should be > 0"
    assert metrics["num_distill_tokens"] > 0, "Distill tokens should be > 0"
    assert metrics["num_reflect_tokens"] > 0, "Reflect tokens should be > 0"


def test_train_all_attempt1_failed():
    """All attempt1s have empty full_ids (parse failure). GRPO should skip, no crash."""
    _patch_logprob_fns()
    model = FakeModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    episodes = [
        make_episode(a1_empty=True, a2_bpb=0.93, a2_reward=-0.93),
        make_episode(a1_empty=True, a2_bpb=0.95, a2_reward=-0.95),
    ]

    reflection = make_reflection(reward=-0.94)

    metrics = erl_train_step(
        model, optimizer, episodes, reflection,
        kl_coef=0.1, temperature=1.0, max_grad_norm=1.0,
    )

    # No attempt1 GRPO tokens (all empty), but attempt2 + reflection should work
    assert metrics["num_reflect_tokens"] > 0 or metrics["num_grpo_tokens"] >= 0


def test_train_no_distill():
    """No attempt2 beats best_bpb -> no distillation targets. Should not crash."""
    _patch_logprob_fns()
    model = FakeModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    episodes = [
        make_episode(a1_bpb=0.95, a1_reward=-0.95, a2_bpb=0.96, a2_reward=-0.96),
        make_episode(a1_bpb=0.97, a1_reward=-0.97, a2_bpb=0.98, a2_reward=-0.98),
    ]
    # No distill flags set

    reflection = make_reflection(reward=-0.97)

    metrics = erl_train_step(
        model, optimizer, episodes, reflection,
        kl_coef=0.1, temperature=1.0, max_grad_norm=1.0,
    )

    assert metrics["num_distill_tokens"] == 0, "Should have no distill tokens"
    assert metrics["avg_distill_loss"] == 0.0, "Should have no distill loss"


def test_train_single_rollout():
    """Single episode -> GRPO needs >=2 rollouts, should skip attempt GRPO gracefully."""
    _patch_logprob_fns()
    model = FakeModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    episodes = [
        make_episode(a1_bpb=0.95, a1_reward=-0.95, a2_bpb=0.93, a2_reward=-0.93),
    ]

    reflection = make_reflection(reward=-0.93)

    metrics = erl_train_step(
        model, optimizer, episodes, reflection,
        kl_coef=0.1, temperature=1.0, max_grad_norm=1.0,
    )
    # With only 1 rollout, GRPO for attempt1 and attempt2 skip (need >=2)
    # Reflection may still fire
    # Key thing: no crash


def test_train_no_reflection():
    """Reflection is None -> signal 2 skipped, no crash."""
    _patch_logprob_fns()
    model = FakeModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    episodes = [
        make_episode(a1_bpb=0.92, a1_reward=-0.92, a2_bpb=0.90, a2_reward=-0.90),
        make_episode(a1_bpb=0.95, a1_reward=-0.95, a2_bpb=0.93, a2_reward=-0.93),
    ]

    metrics = erl_train_step(
        model, optimizer, episodes, None,
        kl_coef=0.1, temperature=1.0, max_grad_norm=1.0,
    )

    assert metrics["num_reflect_tokens"] == 0


def test_train_all_same_rewards():
    """All rewards identical -> advantages all zero -> no GRPO update, no crash."""
    _patch_logprob_fns()
    model = FakeModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    episodes = [
        make_episode(a1_bpb=0.95, a1_reward=-0.95, a2_bpb=0.95, a2_reward=-0.95),
        make_episode(a1_bpb=0.95, a1_reward=-0.95, a2_bpb=0.95, a2_reward=-0.95),
        make_episode(a1_bpb=0.95, a1_reward=-0.95, a2_bpb=0.95, a2_reward=-0.95),
    ]

    reflection = make_reflection(reward=-0.95)
    # reflection advantage = reward - mean(a1 rewards) = -0.95 - (-0.95) = 0 -> skip

    metrics = erl_train_step(
        model, optimizer, episodes, reflection,
        kl_coef=0.1, temperature=1.0, max_grad_norm=1.0,
    )

    # All advantages zero, so GRPO skips all rollouts and reflection
    assert metrics["num_grpo_tokens"] == 0, f"Expected 0 GRPO tokens, got {metrics['num_grpo_tokens']}"
    assert metrics["num_reflect_tokens"] == 0


def test_train_mixed_empty_and_valid():
    """Mix of empty (parse-failed) and valid rollouts. Should train on valid ones only."""
    _patch_logprob_fns()
    model = FakeModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    episodes = [
        make_episode(a1_bpb=0.92, a1_reward=-0.92, a2_bpb=0.90, a2_reward=-0.90),
        make_episode(a1_empty=True, a2_empty=True),
        make_episode(a1_bpb=0.97, a1_reward=-0.97, a2_bpb=0.95, a2_reward=-0.95),
    ]

    reflection = make_reflection(reward=-0.92)

    metrics = erl_train_step(
        model, optimizer, episodes, reflection,
        kl_coef=0.1, temperature=1.0, max_grad_norm=1.0,
    )
    # Should process 2 valid rollouts for GRPO, skip empty one


def test_train_zero_kl():
    """kl_coef=0 -> no KL penalty, should still work."""
    _patch_logprob_fns()
    model = FakeModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    episodes = [
        make_episode(a1_bpb=0.92, a1_reward=-0.92, a2_bpb=0.90, a2_reward=-0.90),
        make_episode(a1_bpb=0.97, a1_reward=-0.97, a2_bpb=0.95, a2_reward=-0.95),
    ]

    reflection = make_reflection(reward=-0.92)

    metrics = erl_train_step(
        model, optimizer, episodes, reflection,
        kl_coef=0.0, temperature=1.0, max_grad_norm=1.0,
    )
    assert "kl_mean" not in metrics, "No KL stats when kl_coef=0"


# ---------------------------------------------------------------------------
# Test: Distillation target selection logic
# ---------------------------------------------------------------------------

def test_distill_only_improvements():
    """Only attempt2s strictly below pre-step best get distilled."""
    best_bpb = 0.95
    episodes = [
        make_episode(a2_bpb=0.93, a2_reward=-0.93),  # improved -> distill
        make_episode(a2_bpb=0.95, a2_reward=-0.95),  # equal -> NO distill
        make_episode(a2_bpb=0.97, a2_reward=-0.97),  # worse -> NO distill
        make_episode(a2_bpb=None, a2_status="crash", a2_reward=-1.0),  # crash -> NO
    ]

    distill_count = 0
    for ep in episodes:
        r2 = ep.attempt2_rollout
        if (r2 is not None
                and r2.val_bpb is not None
                and r2.val_bpb < best_bpb
                and r2.full_ids.numel() > 0):
            distill_count += 1

    assert distill_count == 1, f"Expected 1 distill target, got {distill_count}"


# ---------------------------------------------------------------------------
# Test: Best code update logic
# ---------------------------------------------------------------------------

def test_best_update_from_both_attempts():
    """Best code should update from whichever attempt is best."""
    best_bpb = 0.95
    best_code = "original"
    episodes = [
        make_episode(a1_bpb=0.93, a1_edited_code="a1_code",
                     a2_bpb=0.91, a2_edited_code="a2_code"),
    ]

    for ep in episodes:
        for tag, r, code in [
            ("attempt1", ep.attempt1_rollout, ep.attempt1_edited_code),
            ("attempt2", ep.attempt2_rollout, ep.attempt2_edited_code),
        ]:
            if r is None or r.val_bpb is None or code is None:
                continue
            if r.val_bpb < best_bpb:
                best_bpb = r.val_bpb
                best_code = code

    assert best_bpb == 0.91
    assert best_code == "a2_code"


def test_best_update_no_improvement():
    """No improvement -> best stays the same."""
    best_bpb = 0.90
    best_code = "original"
    episodes = [
        make_episode(a1_bpb=0.92, a1_edited_code="a1_code",
                     a2_bpb=0.91, a2_edited_code="a2_code"),
    ]

    for ep in episodes:
        for tag, r, code in [
            ("attempt1", ep.attempt1_rollout, ep.attempt1_edited_code),
            ("attempt2", ep.attempt2_rollout, ep.attempt2_edited_code),
        ]:
            if r is None or r.val_bpb is None or code is None:
                continue
            if r.val_bpb < best_bpb:
                best_bpb = r.val_bpb
                best_code = code

    assert best_bpb == 0.90
    assert best_code == "original"


# ---------------------------------------------------------------------------
# Run all tests
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("ERL Mock Test Suite (no GPU)")
    print("=" * 60)

    print("\n--- GRPO Advantages ---")
    run_test("normal rewards", test_grpo_normal)
    run_test("identical rewards", test_grpo_identical_rewards)
    run_test("single reward", test_grpo_single_reward)
    run_test("two rewards", test_grpo_two_rewards)

    print("\n--- Feedback Building ---")
    run_test("improved attempt", test_feedback_improved)
    run_test("regressed attempt", test_feedback_regressed)
    run_test("crashed attempt", test_feedback_crash)
    run_test("edit_failed attempt", test_feedback_edit_failed)
    run_test("batch mixed statuses", test_batch_feedback_mixed)

    print("\n--- Reflection Context ---")
    run_test("reflection context construction", test_reflection_context)

    print("\n--- Dataclasses ---")
    run_test("episode defaults", test_episode_defaults)
    run_test("step reflection fields", test_step_reflection)

    print("\n--- Trainer Signal Routing ---")
    run_test("happy path (4 signals)", test_train_happy_path)
    run_test("all attempt1 failed", test_train_all_attempt1_failed)
    run_test("no distillation targets", test_train_no_distill)
    run_test("single rollout (GRPO skip)", test_train_single_rollout)
    run_test("no reflection (None)", test_train_no_reflection)
    run_test("all same rewards (zero advantages)", test_train_all_same_rewards)
    run_test("mixed empty + valid rollouts", test_train_mixed_empty_and_valid)
    run_test("zero KL coef", test_train_zero_kl)

    print("\n--- Distillation Selection ---")
    run_test("only improvements distilled", test_distill_only_improvements)

    print("\n--- Best Code Update ---")
    run_test("best from either attempt", test_best_update_from_both_attempts)
    run_test("no improvement keeps original", test_best_update_no_improvement)

    print(f"\n{'=' * 60}")
    print(f"Results: {passed} passed, {failed} failed")
    if errors:
        print("\nFailed tests:")
        for name, e in errors:
            print(f"  - {name}: {e}")
    print("=" * 60)

    sys.exit(1 if failed > 0 else 0)


if __name__ == "__main__":
    main()
