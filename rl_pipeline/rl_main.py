"""RL experiment loop entry point.

Loads model locally with PEFT LoRA, uses PUCT for parent selection,
trains LoRA weights via policy gradient after each batch.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

# Add parent dir (llm_scaffold/) to path for frozen pipeline imports
PIPELINE_DIR = os.path.dirname(os.path.abspath(__file__))
SCAFFOLD_DIR = os.path.dirname(PIPELINE_DIR)
sys.path.insert(0, SCAFFOLD_DIR)

import torch

import planner
import results
import state as state_mod
from rl_model import load_model
from rl_planner import propose_experiment_rl
from rl_sampler import State, PUCTSampler
from rl_trainer import compute_reward, compute_entropic_advantages, train_step
from rl_types import Rollout
TRAIN_TIMEOUT = 600


# ---------------------------------------------------------------------------
# Generate + apply (phase 1 for parallel mode)
# ---------------------------------------------------------------------------

def generate_and_apply(
    model,
    tokenizer,
    agent_state: dict,
    parent: State,
    train_path: str,
    temperature: float,
    max_new_tokens: int,
) -> tuple[Rollout, str | None, dict | None]:
    """Generate proposal, validate/apply edits. Returns (rollout, edited_code, proposal).

    Does NOT run train.py. Returns edited_code=None if edits failed.
    """
    # Write parent code so planner can read it
    state_mod.write_file(train_path, parent.code)

    # Propose
    print("  Proposing...")
    try:
        proposal, rollout = propose_experiment_rl(
            model, tokenizer, agent_state,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
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

    # Apply edits — retry once with error feedback
    original_text = parent.code
    missing = planner.validate_edit_targets(train_path, proposal["edits"])
    if missing:
        error_msg = f"Search strings not found: {missing}"
        print(f"  Edit failed: {error_msg}")
        print("  Retrying with error feedback...")
        try:
            proposal2, rollout2 = propose_experiment_rl(
                model, tokenizer, agent_state,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                error_context=error_msg,
            )
            proposal = proposal2
            rollout = rollout2
            print(f"  >> {proposal['description']}  (risk: {proposal['risk']})")
            missing = planner.validate_edit_targets(train_path, proposal["edits"])
        except Exception:
            missing = ["retry failed"]

        if missing:
            print("  Edit failed after retry")
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

    edited_code = state_mod.read_file(train_path)
    # Reset train.py back to parent
    state_mod.write_file(train_path, original_text)
    return rollout, edited_code, proposal


# ---------------------------------------------------------------------------
# Baseline
# ---------------------------------------------------------------------------

def run_baseline(repo_path: str) -> tuple[float, str]:
    """Run train.py once, return (val_bpb, output_text). Exits on failure."""
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
    output = state_mod.read_file(results.RUN_LOG)
    return val_bpb, output


# ---------------------------------------------------------------------------
# Single rollout
# ---------------------------------------------------------------------------

def run_single_rollout(
    model,
    tokenizer,
    agent_state: dict,
    parent: State,
    repo_path: str,
    train_path: str,
    temperature: float,
    max_new_tokens: int,
    step: int,
) -> tuple[Rollout, State | None]:
    """Generate proposal, apply edits, run train.py. Returns (rollout, child_state)."""

    # Write parent code to train.py
    state_mod.write_file(train_path, parent.code)

    # Propose
    print("  Proposing...")
    try:
        proposal, rollout = propose_experiment_rl(
            model, tokenizer, agent_state,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
        )
    except Exception as e:
        print(f"  Proposal failed: {e}")
        rollout = Rollout(
            prompt_text="", proposal_text="", full_ids=torch.tensor([]),
            old_logprobs=torch.tensor([]), prompt_len=0,
            val_bpb=None, status="edit_failed", reward=-1.0,
            description=f"proposal_error: {e}",
        )
        return rollout, None

    print(f"  >> {proposal['description']}  (risk: {proposal['risk']})")

    # Apply edits — retry once with error feedback (same as frozen pipeline)
    original_text = parent.code
    missing = planner.validate_edit_targets(train_path, proposal["edits"])
    if missing:
        error_msg = f"Search strings not found: {missing}"
        print(f"  Edit failed: {error_msg}")
        print("  Retrying with error feedback...")
        try:
            proposal2, rollout2 = propose_experiment_rl(
                model, tokenizer, agent_state,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                error_context=error_msg,
            )
            proposal = proposal2
            rollout = rollout2
            print(f"  >> {proposal['description']}  (risk: {proposal['risk']})")
            missing = planner.validate_edit_targets(train_path, proposal["edits"])
        except Exception:
            missing = ["retry failed"]

        if missing:
            print("  Edit failed after retry")
            state_mod.write_file(train_path, original_text)
            rollout.status = "edit_failed"
            rollout.reward = compute_reward(None, "edit_failed")
            return rollout, None

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
        return rollout, None

    edited_code = state_mod.read_file(train_path)

    # Run training
    print("  Training...")
    run_result = results.run_experiment(repo_path, timeout_seconds=TRAIN_TIMEOUT)

    val_bpb, peak_vram_mb = None, None
    status = "crash"
    output_text = ""

    if results.did_timeout(run_result):
        print("  TIMEOUT")
        output_text = "timeout"
    elif results.did_command_fail(run_result):
        log_text = state_mod.read_file(results.RUN_LOG)
        output_text = log_text[-2000:]
        print(f"  CRASH\n{results.extract_error_tail(log_text)}")
    else:
        val_bpb, peak_vram_mb = results.parse_metrics()
        if val_bpb is not None:
            status = "keep"  # RL doesn't keep/discard via git, just records
            output_text = state_mod.read_file(results.RUN_LOG)[-2000:]
        else:
            log_text = state_mod.read_file(results.RUN_LOG)
            output_text = log_text[-2000:]
            print(f"  No metrics\n{results.extract_error_tail(log_text)}")

    reward = compute_reward(val_bpb, status)
    rollout.val_bpb = val_bpb
    rollout.status = status
    rollout.reward = reward

    val_str = f"{val_bpb:.6f}" if val_bpb is not None else "—"
    print(f"  Result: val_bpb={val_str}  reward={reward:.4f}")

    # Create child state for PUCT
    child = None
    if val_bpb is not None:
        child = State(
            timestep=step,
            code=edited_code,
            value=-val_bpb,  # higher = better for PUCT
            observation=output_text,
        )

    # Reset train.py
    state_mod.write_file(train_path, original_text)

    return rollout, child


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="RL Autoresearch (TTT-Discover style)")
    parser.add_argument("--repo-path", default=os.path.join(SCAFFOLD_DIR, "autoresearch_rl"))
    parser.add_argument("--source-repo", default=os.path.join(SCAFFOLD_DIR, "autoresearch"),
                        help="Source repo to clone from if repo-path doesn't exist")
    parser.add_argument("--model-dir", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--model-gpu", type=int, default=None,
                        help="GPU for model (overrides --gpu-id)")
    parser.add_argument("--eval-gpus", type=str, default="",
                        help="Comma-separated GPU IDs for eval workers (empty = sequential)")
    parser.add_argument("--workers-per-gpu", type=int, default=1,
                        help="Number of eval workers per GPU (B200: 2 fits in 180GB)")
    parser.add_argument("--no-overlap", action="store_true",
                        help="Wait for each eval before generating next")
    parser.add_argument("--num-steps", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=4e-5)
    parser.add_argument("--kl-coef", type=float, default=0.1)
    parser.add_argument("--puct-c", type=float, default=1.0)
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=32768)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--attn-impl", default="sdpa",
                        help="sdpa (default), flash_attention_2, or flash_attention_4")
    parser.add_argument("--log-dir", default=os.path.join(PIPELINE_DIR, "rl_log"))
    parser.add_argument("--resume-step", type=int, default=None)
    args = parser.parse_args()

    repo_path = os.path.abspath(args.repo_path)
    train_path = os.path.join(repo_path, "train.py")
    os.makedirs(args.log_dir, exist_ok=True)

    # Resolve model GPU (--model-gpu overrides --gpu-id)
    model_gpu = args.model_gpu if args.model_gpu is not None else args.gpu_id
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

    # Init Ray workers (parallel mode)
    workers = None
    if parallel_mode:
        import ray
        from rl_eval import EvalWorker
        ray.init(ignore_reinit_error=True)
        eval_gpu_ids = [int(g) for g in args.eval_gpus.split(",")]
        if len(eval_gpu_ids) != len(set(eval_gpu_ids)):
            print("ERROR: Duplicate GPU IDs in --eval-gpus")
            sys.exit(1)
        if model_gpu in eval_gpu_ids:
            print(f"WARNING: model GPU {model_gpu} also in eval GPUs, will contend for memory")
        expanded_gpu_ids = [g for g in eval_gpu_ids for _ in range(args.workers_per_gpu)]
        workers = [
            EvalWorker.remote(gpu, repo_path, i)
            for i, gpu in enumerate(expanded_gpu_ids)
        ]
        print(f"Parallel mode: model on GPU {model_gpu}, "
              f"eval on GPUs {eval_gpu_ids} x{args.workers_per_gpu} = {len(workers)} workers")

    # Load model
    device = f"cuda:{model_gpu}"
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

    results.ensure_results_tsv()

    # PUCT sampler — resume or fresh start
    if args.resume_step is not None:
        print(f"\n--- Resuming from step {args.resume_step} ---")
        # Load LoRA weights from checkpoint
        lora_path = os.path.join(args.log_dir, f"lora_step_{args.resume_step:06d}")
        if os.path.exists(lora_path):
            print(f"Loading LoRA from {lora_path}")
            from peft import PeftModel
            model = PeftModel.from_pretrained(model.base_model.model, lora_path)
            optimizer = torch.optim.AdamW(
                [p for p in model.parameters() if p.requires_grad],
                lr=args.lr, betas=(0.9, 0.95), eps=1e-8,
            )
        else:
            print(f"WARNING: No LoRA checkpoint at {lora_path}, using fresh adapter")
        # Load PUCT state
        dummy = State(timestep=0, code="", value=0.0)
        sampler = PUCTSampler(
            initial_state=dummy,
            log_dir=args.log_dir,
            puct_c=args.puct_c,
            resume_step=args.resume_step,
        )
        best_state = sampler.best_state()
        best_bpb = -best_state.value if best_state and best_state.value is not None else float("inf")
    else:
        print("\n--- Running baseline ---")
        original_code = state_mod.read_file(train_path)
        baseline_bpb, baseline_output = run_baseline(repo_path)

        initial_state = State(
            timestep=0,
            code=original_code,
            value=-baseline_bpb,
            observation=baseline_output,
        )
        sampler = PUCTSampler(
            initial_state=initial_state,
            log_dir=args.log_dir,
            puct_c=args.puct_c,
        )
        best_bpb = baseline_bpb

    # Agent state (for planner.build_planner_context compatibility)
    agent_state = {
        "repo_path": repo_path,
        "best_val_bpb": best_bpb,
    }
    step_log = []
    rollout_log_path = os.path.join(args.log_dir, "rollouts.jsonl")

    # Main loop
    for step in range(args.num_steps):
        step_start = time.time()
        print(f"\n{'='*60}")
        print(f"Step {step}/{args.num_steps} | Best: {best_bpb:.6f} | Buffer: {sampler.buffer_size()}")
        print(f"{'='*60}")

        parent = sampler.sample_state()
        print(f"  Parent: val_bpb={-parent.value:.6f}" if parent.value is not None else "  Parent: no value")

        rollouts: list[Rollout] = []

        if parallel_mode:
            # ── Parallel mode: generate all proposals, dispatch to Ray workers ──
            import ray

            pending = []  # (ray_ref, rollout, edited_code, attempt_num)
            max_attempts = args.batch_size * 2
            attempt = 0
            g = 0

            while g < args.batch_size and attempt < max_attempts:
                attempt += 1
                print(f"\n  Rollout {g+1}/{args.batch_size} (attempt {attempt})")

                rollout, edited_code, proposal = generate_and_apply(
                    model, tokenizer, agent_state, parent,
                    train_path,
                    temperature=args.temperature,
                    max_new_tokens=args.max_new_tokens,
                )

                if rollout.full_ids.numel() == 0:
                    # Parse failure — log and retry
                    with open(rollout_log_path, "a") as f:
                        f.write(json.dumps({
                            "step": step, "rollout": attempt,
                            "val_bpb": None, "reward": rollout.reward,
                            "status": rollout.status, "description": rollout.description,
                        }) + "\n")
                    print("  Retrying...")
                    continue

                if edited_code is not None:
                    # Dispatch to Ray worker
                    worker = workers[g % len(workers)]
                    ref = worker.evaluate.remote(parent.code, edited_code, step)
                    pending.append((ref, rollout, edited_code, attempt))
                    if args.no_overlap:
                        ray.get(ref)  # wait before generating next
                else:
                    # Edit failed — has logprobs, counts as rollout for RL
                    rollouts.append(rollout)
                    sampler.record_failed_rollout(parent)
                    results.append_result(
                        "rl", rollout.val_bpb, None, rollout.status, rollout.description
                    )
                    with open(rollout_log_path, "a") as f:
                        f.write(json.dumps({
                            "step": step, "rollout": attempt,
                            "val_bpb": rollout.val_bpb, "reward": rollout.reward,
                            "status": rollout.status, "description": rollout.description,
                        }) + "\n")

                g += 1

            # Collect eval results from Ray workers
            for ref, rollout, edited_code, attempt_num in pending:
                result = ray.get(ref)
                rollout.val_bpb = result["val_bpb"]
                rollout.status = "keep" if result["success"] else "crash"
                rollout.reward = compute_reward(rollout.val_bpb, rollout.status)

                val_str = f"{rollout.val_bpb:.6f}" if rollout.val_bpb is not None else "—"
                print(f"  Eval result: val_bpb={val_str}  reward={rollout.reward:.4f}")

                # Update PUCT tree
                if result["success"]:
                    child = State(
                        timestep=step,
                        code=edited_code,
                        value=-result["val_bpb"],
                        observation=result["output"],
                    )
                    sampler.update_state(child, parent)
                    if rollout.val_bpb < best_bpb:
                        best_bpb = rollout.val_bpb
                        agent_state["best_val_bpb"] = best_bpb
                        print(f"  *** NEW BEST: {best_bpb:.6f} ***")
                        Path(os.path.join(args.log_dir, "best_train.py")).write_text(edited_code)
                else:
                    sampler.record_failed_rollout(parent)

                rollouts.append(rollout)

                results.append_result(
                    "rl", rollout.val_bpb, None, rollout.status, rollout.description
                )
                with open(rollout_log_path, "a") as f:
                    f.write(json.dumps({
                        "step": step, "rollout": attempt_num,
                        "val_bpb": rollout.val_bpb, "reward": rollout.reward,
                        "status": rollout.status, "description": rollout.description,
                    }) + "\n")

        else:
            # ── Sequential mode: same as before ──
            max_attempts = args.batch_size * 2
            attempt = 0
            g = 0

            while g < args.batch_size and attempt < max_attempts:
                attempt += 1
                print(f"\n  Rollout {g+1}/{args.batch_size} (attempt {attempt})")
                rollout, child = run_single_rollout(
                    model, tokenizer, agent_state, parent,
                    repo_path, train_path,
                    temperature=args.temperature,
                    max_new_tokens=args.max_new_tokens,
                    step=step,
                )

                # Update PUCT tree
                if child is not None:
                    sampler.update_state(child, parent)
                    if rollout.val_bpb is not None and rollout.val_bpb < best_bpb:
                        best_bpb = rollout.val_bpb
                        agent_state["best_val_bpb"] = best_bpb
                        print(f"  *** NEW BEST: {best_bpb:.6f} ***")
                        Path(os.path.join(args.log_dir, "best_train.py")).write_text(child.code)
                else:
                    sampler.record_failed_rollout(parent)

                # Only keep rollouts with generation artifacts for training
                if rollout.full_ids.numel() > 0:
                    rollouts.append(rollout)
                    g += 1
                else:
                    print("  Retrying...")

                if rollout.full_ids.numel() > 0:
                    results.append_result(
                        "rl", rollout.val_bpb, None, rollout.status, rollout.description
                    )

                with open(rollout_log_path, "a") as f:
                    f.write(json.dumps({
                        "step": step, "rollout": attempt,
                        "val_bpb": rollout.val_bpb, "reward": rollout.reward,
                        "status": rollout.status, "description": rollout.description,
                    }) + "\n")

        # Free CUDA cache before RL training (train.py subprocess may have fragmented memory)
        torch.cuda.empty_cache()

        # RL training step — advantages computed from filtered rollouts only
        if rollouts:
            advantages = compute_entropic_advantages([r.reward for r in rollouts])
            metrics = train_step(
                model, optimizer, rollouts, advantages,
                kl_coef=args.kl_coef,
                temperature=args.temperature,
                max_grad_norm=args.max_grad_norm,
            )
            print(f"\n  RL update: loss={metrics['avg_loss']:.4f} tokens={metrics['num_tokens']}")
            if "kl_mean" in metrics:
                print(f"  KL mean: {metrics['kl_mean']:.6f}")
        else:
            metrics = {}
            print("\n  RL update: skipped (no valid rollouts)")

        # Checkpoint — save PUCT tree + LoRA weights
        sampler.save(step)
        adapter_path = os.path.join(args.log_dir, f"lora_step_{step:06d}")
        model.save_pretrained(adapter_path)
        step_time = time.time() - step_start

        step_info = {
            "step": step,
            "best_bpb": best_bpb,
            "buffer_size": sampler.buffer_size(),
            "rewards": [round(r.reward, 4) for r in rollouts],
            "step_time_s": round(step_time, 1),
            **{k: round(v, 6) if isinstance(v, float) else v for k, v in metrics.items()},
        }
        step_log.append(step_info)
        print(f"  Step time: {step_time/60:.1f} min")

        with open(os.path.join(args.log_dir, "step_log.json"), "w") as f:
            json.dump(step_log, f, indent=2)

    # Done
    print(f"\n{'='*60}")
    print(f"Done. Best val_bpb: {best_bpb:.6f}")
    best = sampler.best_state()
    if best:
        Path(os.path.join(args.log_dir, "best_train.py")).write_text(best.code)
        print(f"Best code saved to {args.log_dir}/best_train.py")


if __name__ == "__main__":
    main()
