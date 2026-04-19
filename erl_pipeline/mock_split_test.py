"""Mock test for the split pipeline (Ideator + Implementer). No GPU needed.

Covers:
1. Happy path — ideator returns idea, implementer returns valid JSON.
2. Stage B fails all 3 attempts — rollout is edit_failed, stage-A tensors preserved.
3. Stage A returns empty idea — rollout is edit_failed, skip stage B.

Usage: python erl_pipeline/mock_split_test.py
"""
from __future__ import annotations

import os
import sys
import types

PIPELINE_DIR = os.path.dirname(os.path.abspath(__file__))
SCAFFOLD_DIR = os.path.dirname(PIPELINE_DIR)
RL_PIPELINE_DIR = os.path.join(SCAFFOLD_DIR, "rl_pipeline")
sys.path.insert(0, SCAFFOLD_DIR)
sys.path.insert(0, RL_PIPELINE_DIR)

import torch

import rl_planner


# ── Fakes ─────────────────────────────────────────────────────

class FakeTokenizer:
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, **kw):
        return "PROMPT:" + "|".join(m["content"][:40] for m in messages)


class FakeModel:
    input_device = torch.device("cpu")


def make_agent_state(tmpdir: str, best_bpb: float = 1.0) -> dict:
    os.makedirs(tmpdir, exist_ok=True)
    train_py = os.path.join(tmpdir, "train.py")
    with open(train_py, "w") as f:
        f.write("learning_rate = 0.001\nhidden = 256\n")
    return {"repo_path": tmpdir, "best_val_bpb": best_bpb}


def patch_generators(monkeypatch_store: dict, ideator_text: str, implementer_sequence: list[str]):
    """Install fakes into rl_planner. Stores originals to restore after."""
    monkeypatch_store["generate_with_logprobs"] = rl_planner.generate_with_logprobs
    monkeypatch_store["_generate_text"] = rl_planner._generate_text
    monkeypatch_store["summarize_recent_results"] = rl_planner.planner.summarize_recent_results

    def fake_gen_lp(model, tokenizer, prompt, max_new_tokens, temperature, think_budget=None):
        n = 5
        full_ids = torch.zeros(len(prompt) // 4 + n, dtype=torch.long)
        logprobs = torch.zeros(n)
        return ideator_text, full_ids, logprobs, len(prompt) // 4

    seq_iter = iter(implementer_sequence)

    def fake_gen_text(model, tokenizer, prompt, max_new_tokens, temperature, enable_thinking=False):
        return next(seq_iter)

    rl_planner.generate_with_logprobs = fake_gen_lp
    rl_planner._generate_text = fake_gen_text
    rl_planner.planner.summarize_recent_results = lambda n=10: "No previous experiments."


def restore(monkeypatch_store: dict):
    rl_planner.generate_with_logprobs = monkeypatch_store["generate_with_logprobs"]
    rl_planner._generate_text = monkeypatch_store["_generate_text"]
    rl_planner.planner.summarize_recent_results = monkeypatch_store["summarize_recent_results"]


# ── Tests ─────────────────────────────────────────────────────

def test_happy_path():
    store: dict = {}
    agent = make_agent_state("/tmp/mock_split_happy")
    good_json = (
        '{"description": "lower lr", "rationale": "reduce overshoot", '
        '"risk": "low", "edits": [{"search": "learning_rate = 0.001", '
        '"replace": "learning_rate = 0.0005"}]}'
    )
    patch_generators(store, "Reduce learning_rate from 0.001 to 0.0005 to soften updates.",
                     [good_json])
    try:
        proposal, rollout = rl_planner.propose_experiment_split(
            FakeModel(), FakeTokenizer(), agent,
            temperature=1.0, max_new_tokens=16000,
        )
    finally:
        restore(store)

    assert proposal is not None, "expected valid proposal"
    assert proposal["description"] == "lower lr"
    assert rollout.status == "pending"
    assert rollout.full_ids.numel() > 0
    assert rollout.old_logprobs.numel() == 5
    assert "[idea]" in rollout.description and "[impl]" in rollout.description
    print("PASS test_happy_path")


def test_impl_fails_all_retries():
    store: dict = {}
    agent = make_agent_state("/tmp/mock_split_failimpl")
    patch_generators(
        store,
        "Change hidden dim from 256 to 384 to add capacity.",
        ["not json", "{broken", "still not json"],  # 3 attempts, all fail
    )
    try:
        proposal, rollout = rl_planner.propose_experiment_split(
            FakeModel(), FakeTokenizer(), agent,
            temperature=1.0, max_new_tokens=16000,
        )
    finally:
        restore(store)

    assert proposal is None
    assert rollout.status == "edit_failed"
    assert rollout.full_ids.numel() > 0, "stage-A tensors must be preserved"
    assert "impl_failed" in rollout.description
    print("PASS test_impl_fails_all_retries")


def test_empty_idea():
    store: dict = {}
    agent = make_agent_state("/tmp/mock_split_empty")
    patch_generators(store, "", ["{}"])  # implementer shouldn't even be called
    try:
        proposal, rollout = rl_planner.propose_experiment_split(
            FakeModel(), FakeTokenizer(), agent,
            temperature=1.0, max_new_tokens=16000,
        )
    finally:
        restore(store)

    assert proposal is None
    assert rollout.status == "edit_failed"
    assert rollout.description == "empty_idea"
    print("PASS test_empty_idea")


if __name__ == "__main__":
    test_happy_path()
    test_impl_fails_all_retries()
    test_empty_idea()
    print("\nAll split-pipeline mock tests passed.")
