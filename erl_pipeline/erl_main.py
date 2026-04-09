"""ERL experiment loop entry point (v2).

Phased step: all attempt1 -> batch reflection -> all attempt2 -> train.
No PUCT tree search. Parent is always the current best code.
Matches Microsoft ERL paper (arXiv:2602.13949) design.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

PIPELINE_DIR = os.path.dirname(os.path.abspath(__file__))
SCAFFOLD_DIR = os.path.dirname(PIPELINE_DIR)
RL_PIPELINE_DIR = os.path.join(SCAFFOLD_DIR, "rl_pipeline")
sys.path.insert(0, SCAFFOLD_DIR)
sys.path.insert(0, RL_PIPELINE_DIR)

import torch

import planner
import results
import state as state_mod
from rl_model import load_model
from rl_planner import propose_experiment_rl
from rl_trainer import compute_reward
from rl_types import Rollout

from erl_types import Episode, StepReflection
from erl_feedback import build_batch_feedback
from erl_reflect import generate_batch_reflection, build_reflection_context
from erl_trainer import erl_train_step

TRAIN_TIMEOUT = 600


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def generate_and_apply(
    model, tokenizer, agent_state, parent_code, train_path,
    temperature, max_new_tokens, error_context=None,
):
    """Generate proposal, apply edits. Returns (rollout, edited_code, proposal)."""
    state_mod.write_file(train_path, parent_code)

    try:
        proposal, rollout = propose_experiment_rl(
            model, tokenizer, agent_state,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            error_context=error_context,
        )
    except Exception as e:
        print(f"  Proposal failed: {e}")
        rollout = Rollout(
            prompt_text="", proposal_text="", full_ids=torch.tensor([]),
            old_logprobs=torch.tensor([]), prompt_len=0,
            val_bpb=None, status="edit_failed", reward=-1.0,
            description=f"proposal_error: {e}",
        )
        return rollout, None, None

    print(f"  >> {proposal['description']}  (risk: {proposal['risk']})")

    original_text = parent_code
    missing = planner.validate_edit_targets(train_path, proposal["edits"])
    if missing:
        print(f"  Edit failed: search strings not found")
        state_mod.write_file(train_path, original_text)
        rollout.status = "edit_failed"
        rollout.reward = compute_reward(None, "edit_failed")
        return rollout, None, proposal

    try:
        new_text = planner.apply_edits(train_path, proposal["edits"])
        diff = planner.preview_diff(original_text, new_text)
        if diff:
            print(diff)
    except ValueError as e:
        print(f"  Apply failed: {e}")
        state_mod.write_file(train_path, original_text)
        rollout.status = "edit_failed"
        rollout.reward = compute_reward(None, "edit_failed")
        return rollout, None, proposal

    if not diff:
        state_mod.write_file(train_path, original_text)
        rollout.status = "edit_failed"
        rollout.reward = compute_reward(None, "edit_failed")
        return rollout, None, proposal

    edited_code = state_mod.read_file(train_path)
    state_mod.write_file(train_path, original_text)
    return rollout, edited_code, proposal


def dispatch_eval(workers, worker_idx, parent_code, edited_code, step):
    """Dispatch eval to Ray worker, return future ref."""
    import ray
    worker = workers[worker_idx % len(workers)]
    return worker.evaluate.remote(parent_code, edited_code, step)


def collect_eval(ref, rollout):
    """Collect eval result from Ray future, update rollout."""
    import ray
    result = ray.get(ref)
    rollout.val_bpb = result["val_bpb"]
    rollout.status = "keep" if result["success"] else "crash"
    rollout.reward = compute_reward(rollout.val_bpb, rollout.status)
    return result


def run_eval_sequential(repo_path, parent_code, edited_code, train_path, rollout):
    """Run train.py sequentially (no Ray). Updates rollout in place, returns result dict."""
    state_mod.write_file(train_path, edited_code)
    run_result = results.run_experiment(repo_path, timeout_seconds=TRAIN_TIMEOUT)

    val_bpb, peak_vram_mb = None, None
    output_text = ""
    if results.did_timeout(run_result):
        output_text = "timeout"
    elif results.did_command_fail(run_result):
        log_text = state_mod.read_file(results.RUN_LOG)
        output_text = log_text[-2000:]
    else:
        val_bpb, peak_vram_mb = results.parse_metrics()
        if val_bpb is not None:
            output_text = state_mod.read_file(results.RUN_LOG)[-2000:]
        else:
            log_text = state_mod.read_file(results.RUN_LOG)
            output_text = log_text[-2000:]

    state_mod.write_file(train_path, parent_code)

    success = val_bpb is not None
    rollout.val_bpb = val_bpb
    rollout.status = "keep" if success else "crash"
    rollout.reward = compute_reward(val_bpb, rollout.status)
    return {"val_bpb": val_bpb, "peak_vram_mb": peak_vram_mb,
            "output": output_text, "success": success}


def build_distill_ids(model, tokenizer, agent_state, attempt2_text, temperature):
    """Build distillation target: attempt2 response with original prompt (no reflection)."""
    system_msg, user_msg = planner.build_planner_context(
        agent_state["repo_path"], agent_state["best_val_bpb"]
    )
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]
    original_prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    full_text = original_prompt + attempt2_text
    tokens = tokenizer(full_text, return_tensors="pt")
    full_ids = tokens["input_ids"][0].cpu()
    prompt_tokens = tokenizer(original_prompt, return_tensors="pt")
    prompt_len = prompt_tokens["input_ids"].shape[1]

    with torch.no_grad():
        from rl_model import compute_response_logprobs as _compute_lp
        logprobs = _compute_lp(model, full_ids, prompt_len, temperature=temperature)

    return full_ids, logprobs.cpu(), prompt_len


def run_baseline(repo_path):
    """Run train.py once, return val_bpb. Exits on failure."""
    run_result = results.run_experiment(repo_path, timeout_seconds=TRAIN_TIMEOUT)
    if results.did_timeout(run_result):
        print("ERROR: Baseline timed out")
        sys.exit(1)
    if results.did_command_fail(run_result):
        tail = results.extract_error_tail(state_mod.read_file(results.RUN_LOG))
        print(f"ERROR: Baseline failed\n{tail}")
        sys.exit(1)
    val_bpb, peak_vram_mb = results.parse_metrics()
    if val_bpb is None:
        print("ERROR: Could not parse baseline metrics")
        sys.exit(1)
    print(f"Baseline: val_bpb={val_bpb:.6f}  peak_vram_mb={peak_vram_mb}")
    return val_bpb


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="ERL Autoresearch (Experiential RL, v2)")
    parser.add_argument("--repo-path", default=os.path.join(SCAFFOLD_DIR, "autoresearch_rl"))
    parser.add_argument("--source-repo", default=os.path.join(SCAFFOLD_DIR, "autoresearch"))
    parser.add_argument("--model-dir", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--model-gpu", type=int, default=0)
    parser.add_argument("--eval-gpus", type=str, default="")
    parser.add_argument("--workers-per-gpu", type=int, default=1)
    parser.add_argument("--num-steps", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=4e-5)
    parser.add_argument("--kl-coef", type=float, default=0.1)
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-new-tokens", type=int, default=8192)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--attn-impl", default="sdpa")
    parser.add_argument("--log-dir", default=os.path.join(PIPELINE_DIR, "erl_log"))
    parser.add_argument("--resume-step", type=int, default=None)
    parser.add_argument("--gpu-mem-limit-mb", type=int, default=0,
                        help="Per-worker GPU memory cap in MB (0=disabled, e.g. 88000 for B200 2-per-GPU)")
    args = parser.parse_args()

    repo_path = os.path.abspath(args.repo_path)
    train_path = os.path.join(repo_path, "train.py")
    os.makedirs(args.log_dir, exist_ok=True)

    parallel_mode = bool(args.eval_gpus)

    # Clone repo if needed
    if not os.path.exists(repo_path):
        source = os.path.abspath(args.source_repo)
        if not os.path.exists(source):
            print(f"ERROR: source repo not found at {source}")
            sys.exit(1)
        print(f"Cloning {source} -> {repo_path}")
        shutil.copytree(source, repo_path)

    if not os.path.exists(train_path):
        print(f"ERROR: train.py not found at {train_path}")
        sys.exit(1)

    # Init Ray workers
    workers = None
    if parallel_mode:
        import ray
        runtime_env = {"env_vars": {"PYTHONPATH": RL_PIPELINE_DIR}}
        ray.init(ignore_reinit_error=True, runtime_env=runtime_env)
        from rl_eval import EvalWorker
        eval_gpu_ids = [int(g) for g in args.eval_gpus.split(",")]
        expanded = [g for g in eval_gpu_ids for _ in range(args.workers_per_gpu)]
        workers = [EvalWorker.remote(gpu, repo_path, i, args.gpu_mem_limit_mb)
                   for i, gpu in enumerate(expanded)]
        print(f"Parallel mode: model GPU {args.model_gpu}, eval GPUs {eval_gpu_ids}")

    # Load model
    device = f"cuda:{args.model_gpu}"
    print(f"Loading model {args.model_dir} on {device}...")
    model, tokenizer = load_model(
        args.model_dir, device=device,
        lora_rank=args.lora_rank, lora_alpha=args.lora_alpha,
        attn_impl=args.attn_impl,
    )
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, betas=(0.9, 0.95), eps=1e-8,
    )

    results.RESULTS_FILE = os.path.join(args.log_dir, "results.tsv")
    results.RUN_LOG = os.path.join(args.log_dir, "run.log")
    results.ensure_results_tsv()

    # State: best code + best bpb (no PUCT tree)
    if args.resume_step is not None:
        print(f"\n--- Resuming from step {args.resume_step} ---")
        lora_path = os.path.join(args.log_dir, f"lora_step_{args.resume_step:06d}")
        if os.path.exists(lora_path):
            from peft import PeftModel
            model = PeftModel.from_pretrained(model.base_model.model, lora_path)
            optimizer = torch.optim.AdamW(
                [p for p in model.parameters() if p.requires_grad],
                lr=args.lr, betas=(0.9, 0.95), eps=1e-8,
            )
        best_code_path = os.path.join(args.log_dir, "best_train.py")
        if os.path.exists(best_code_path):
            best_code = Path(best_code_path).read_text()
        else:
            best_code = state_mod.read_file(train_path)
        # Load best_bpb from step_log
        step_log_path = os.path.join(args.log_dir, "step_log.json")
        if os.path.exists(step_log_path):
            with open(step_log_path) as f:
                step_log = json.load(f)
            best_bpb = step_log[-1]["best_bpb"]
        else:
            step_log = []
            best_bpb = float("inf")
    else:
        print("\n--- Running baseline ---")
        best_code = state_mod.read_file(train_path)
        best_bpb = run_baseline(repo_path)
        step_log = []

    agent_state = {"repo_path": repo_path, "best_val_bpb": best_bpb}
    if args.resume_step is None:
        step_log = []
    rollout_log_path = os.path.join(args.log_dir, "rollouts.jsonl")

    # ── Main loop ──
    for step in range(args.num_steps):
        step_start = time.time()
        print(f"\n{'='*60}")
        print(f"ERL Step {step}/{args.num_steps} | Best: {best_bpb:.6f}")
        print(f"{'='*60}")

        episodes: list[Episode] = []

        # ── Phase 1: All first attempts ──
        print("\n  --- Phase 1: First attempts ---")
        a1_refs = []  # (ray_ref, index)

        for g in range(args.batch_size):
            print(f"\n  [Attempt1 {g+1}/{args.batch_size}]")
            rollout, edited_code, proposal = generate_and_apply(
                model, tokenizer, agent_state, best_code, train_path,
                temperature=args.temperature, max_new_tokens=args.max_new_tokens,
            )

            ep = Episode(
                attempt1_rollout=rollout,
                attempt1_proposal=proposal,
                attempt1_edited_code=edited_code,
                attempt1_eval=None,
            )
            episodes.append(ep)

            if edited_code is not None and parallel_mode:
                ref = dispatch_eval(workers, g, best_code, edited_code, step)
                a1_refs.append((ref, g))
            elif edited_code is not None:
                # Serial mode: run eval immediately
                result = run_eval_sequential(repo_path, best_code, edited_code, train_path, rollout)
                ep.attempt1_eval = result
                val_str = f"{rollout.val_bpb:.6f}" if rollout.val_bpb else "---"
                print(f"  Attempt1 {g+1} result: val_bpb={val_str}  reward={rollout.reward:.4f}")
            else:
                rollout.reward = compute_reward(None, rollout.status)

        # Collect attempt1 eval results (parallel mode only)
        for ref, idx in a1_refs:
            ep = episodes[idx]
            result = collect_eval(ref, ep.attempt1_rollout)
            ep.attempt1_eval = result
            val_str = f"{ep.attempt1_rollout.val_bpb:.6f}" if ep.attempt1_rollout.val_bpb else "---"
            print(f"  Attempt1 {idx+1} result: val_bpb={val_str}  reward={ep.attempt1_rollout.reward:.4f}")

        # ── Phase 2: Batch feedback + ONE reflection ──
        print("\n  --- Phase 2: Batch reflection ---")
        attempt_summaries = []
        for ep in episodes:
            r = ep.attempt1_rollout
            attempt_summaries.append({
                "description": r.description,
                "status": r.status,
                "val_bpb": r.val_bpb,
                "eval_output": ep.attempt1_eval["output"] if ep.attempt1_eval else None,
                "edit_error": str(ep.attempt1_proposal) if r.status == "edit_failed" and ep.attempt1_proposal else None,
            })

        batch_feedback = build_batch_feedback(attempt_summaries, best_bpb)
        print(f"  Batch: {len(episodes)} attempts, "
              f"{sum(1 for a in attempt_summaries if a['val_bpb'] is not None and a['val_bpb'] < best_bpb)} improved")

        ref_text, ref_ids, ref_lp, ref_plen = generate_batch_reflection(
            model, tokenizer,
            batch_feedback=batch_feedback,
            best_val_bpb=best_bpb,
            temperature=args.temperature,
        )
        print(f"  Reflection: {ref_text[:120]}...")

        reflection_ctx = build_reflection_context(batch_feedback, ref_text)

        # ── Phase 3: All second attempts (using shared reflection) ──
        print("\n  --- Phase 3: Second attempts ---")
        a2_refs = []

        for g in range(args.batch_size):
            print(f"\n  [Attempt2 {g+1}/{args.batch_size}]")
            rollout2, edited2, proposal2 = generate_and_apply(
                model, tokenizer, agent_state, best_code, train_path,
                temperature=args.temperature, max_new_tokens=args.max_new_tokens,
                error_context=reflection_ctx,
            )

            ep = episodes[g]
            ep.attempt2_rollout = rollout2
            ep.attempt2_proposal = proposal2
            ep.attempt2_edited_code = edited2
            ep.train_attempt2 = rollout2.full_ids.numel() > 0

            if edited2 is not None and parallel_mode:
                ref = dispatch_eval(workers, g, best_code, edited2, step)
                a2_refs.append((ref, g))
            elif edited2 is not None:
                # Serial mode
                result = run_eval_sequential(repo_path, best_code, edited2, train_path, rollout2)
                ep.attempt2_eval = result
                val_str = f"{rollout2.val_bpb:.6f}" if rollout2.val_bpb else "---"
                print(f"  Attempt2 {g+1} result: val_bpb={val_str}  reward={rollout2.reward:.4f}")
            else:
                rollout2.reward = compute_reward(None, rollout2.status)

        # Collect attempt2 eval results (parallel mode only)
        for ref, idx in a2_refs:
            ep = episodes[idx]
            result = collect_eval(ref, ep.attempt2_rollout)
            ep.attempt2_eval = result
            val_str = f"{ep.attempt2_rollout.val_bpb:.6f}" if ep.attempt2_rollout.val_bpb else "---"
            print(f"  Attempt2 {idx+1} result: val_bpb={val_str}  reward={ep.attempt2_rollout.reward:.4f}")

        # Assign reflection reward = mean of attempt2 rewards
        a2_rewards = [ep.attempt2_rollout.reward for ep in episodes if ep.attempt2_rollout is not None and ep.attempt2_rollout.full_ids.numel() > 0]
        ref_reward = sum(a2_rewards) / len(a2_rewards) if a2_rewards else 0.0

        step_reflection = StepReflection(
            feedback_text=batch_feedback,
            reflection_text=ref_text,
            full_ids=ref_ids,
            old_logprobs=ref_lp,
            prompt_len=ref_plen,
            reward=ref_reward,
        )

        # ── Phase 4: Build distillation targets + update best ──
        # Capture pre-step best for distillation threshold (Bug 2 + Bug 5 fix)
        step_best_bpb = best_bpb
        step_best_val_bpb = agent_state["best_val_bpb"]

        # First pass: build distillation targets using pre-step baseline
        for ep in episodes:
            if (ep.attempt2_rollout is not None
                    and ep.attempt2_rollout.val_bpb is not None
                    and ep.attempt2_rollout.val_bpb < step_best_bpb
                    and ep.attempt2_rollout.full_ids.numel() > 0):
                print(f"  [Building distillation target]")
                # Use pre-step best_val_bpb for prompt consistency
                saved_bpb = agent_state["best_val_bpb"]
                agent_state["best_val_bpb"] = step_best_val_bpb
                d_ids, d_lp, d_plen = build_distill_ids(
                    model, tokenizer, agent_state,
                    ep.attempt2_rollout.proposal_text, args.temperature,
                )
                agent_state["best_val_bpb"] = saved_bpb
                ep.distill_full_ids = d_ids
                ep.distill_logprobs = d_lp
                ep.distill_prompt_len = d_plen
                ep.train_distill = True

        # Second pass: update best_bpb
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
                    agent_state["best_val_bpb"] = best_bpb
                    print(f"  *** NEW BEST: {best_bpb:.6f} (from {tag}) ***")
                    Path(os.path.join(args.log_dir, "best_train.py")).write_text(best_code)

        # Log rollouts (compare against pre-step best, not post-update)
        for g, ep in enumerate(episodes):
            for tag, r in [("attempt1", ep.attempt1_rollout), ("attempt2", ep.attempt2_rollout)]:
                if r is None:
                    continue
                log_status = r.status
                if log_status == "keep" and (r.val_bpb is None or r.val_bpb >= step_best_bpb):
                    log_status = "discard"
                results.append_result("erl", r.val_bpb, None, log_status, f"[{tag}] {r.description}")
                with open(rollout_log_path, "a") as f:
                    f.write(json.dumps({
                        "step": step, "episode": g, "tag": tag,
                        "val_bpb": r.val_bpb, "reward": r.reward,
                        "status": log_status, "description": r.description,
                        "distilled": ep.train_distill,
                    }) + "\n")

        # ── Phase 5: Train ──
        torch.cuda.empty_cache()

        metrics = erl_train_step(
            model, optimizer, episodes, step_reflection,
            kl_coef=args.kl_coef,
            temperature=args.temperature,
            max_grad_norm=args.max_grad_norm,
        )
        n_distilled = sum(1 for ep in episodes if ep.train_distill)
        print(f"\n  ERL update: grpo={metrics['avg_grpo_loss']:.4f} "
              f"reflect={metrics['avg_reflect_loss']:.4f} "
              f"distill={metrics['avg_distill_loss']:.4f} "
              f"(grpo_tok={metrics['num_grpo_tokens']} "
              f"ref_tok={metrics['num_reflect_tokens']} "
              f"dist_tok={metrics['num_distill_tokens']})")

        # ── Checkpoint ──
        model.save_pretrained(os.path.join(args.log_dir, f"lora_step_{step:06d}"))
        step_time = time.time() - step_start

        step_info = {
            "step": step,
            "best_bpb": best_bpb,
            "episodes": len(episodes),
            "distilled": n_distilled,
            "reflection_reward": round(ref_reward, 4),
            "step_time_s": round(step_time, 1),
            **{k: round(v, 6) if isinstance(v, float) else v for k, v in metrics.items()},
        }
        step_log.append(step_info)
        print(f"  Step time: {step_time/60:.1f} min | distilled={n_distilled}")

        with open(os.path.join(args.log_dir, "step_log.json"), "w") as f:
            json.dump(step_log, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Done. Best val_bpb: {best_bpb:.6f}")
    Path(os.path.join(args.log_dir, "best_train.py")).write_text(best_code)


if __name__ == "__main__":
    main()
