"""Autoresearch Agent Scaffold — orchestration, setup, loop, recovery.

Usage:
    python main.py [--repo-path PATH] [--max-experiments N] [--resume]
"""

import argparse
import os
import shutil
import signal
import sys
from datetime import date

import state
import planner
import results

SCAFFOLD_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_REPO_PATH = os.path.join(SCAFFOLD_DIR, "autoresearch_frozen")
DEFAULT_SOURCE_REPO = os.path.join(SCAFFOLD_DIR, "autoresearch")
TRAIN_TIMEOUT = 600  # seconds (5 min budget + startup/compilation overhead)


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

def run_setup(repo_path, max_experiments):
    repo_path = os.path.abspath(repo_path)
    train_path = os.path.join(repo_path, "train.py")
    if not os.path.exists(train_path):
        print(f"ERROR: train.py not found at {train_path}")
        sys.exit(1)

    run_tag = date.today().isoformat()
    branch_name = f"autoresearch/{run_tag}"
    print(f"Repo:   {repo_path}")
    print(f"Tag:    {run_tag}")

    state.ensure_clean_repo(repo_path)

    try:
        state.create_experiment_branch(repo_path, branch_name)
        print(f"Branch: {branch_name}")
    except RuntimeError:
        print(f"Branch {branch_name} may already exist, continuing on current branch")

    results.ensure_results_tsv()

    agent_state = state.initialize_state(repo_path, run_tag, branch_name)
    agent_state["max_experiments"] = max_experiments

    print("\n--- Running baseline ---")
    agent_state = run_baseline(agent_state)
    return agent_state


def run_baseline(agent_state):
    repo_path = agent_state["repo_path"]
    commit = state.get_current_commit(repo_path)
    print(f"Commit: {commit}")
    print("Training...")

    run_result = results.run_experiment(repo_path, timeout_seconds=TRAIN_TIMEOUT)

    if results.did_timeout(run_result):
        print("ERROR: Baseline timed out")
        sys.exit(1)
    if results.did_command_fail(run_result):
        tail = results.extract_error_tail(state.read_file(results.RUN_LOG))
        print(f"ERROR: Baseline failed\n{tail}")
        sys.exit(1)

    val_bpb, peak_vram_mb = results.parse_metrics()
    if val_bpb is None:
        print("ERROR: Could not parse baseline metrics")
        sys.exit(1)

    print(f"Baseline: val_bpb={val_bpb:.6f}  peak_vram_mb={peak_vram_mb}")
    results.append_result(commit, val_bpb, peak_vram_mb, "keep", "Baseline (no changes)")

    agent_state["best_commit"] = commit
    agent_state["best_val_bpb"] = val_bpb
    agent_state["best_peak_vram_mb"] = peak_vram_mb
    state.save_state(agent_state)
    return agent_state


# ---------------------------------------------------------------------------
# Experiment loop
# ---------------------------------------------------------------------------

def run_experiment_loop(state_ref):
    agent_state = state_ref[0]
    max_exp = agent_state["max_experiments"]
    print(f"\n--- Experiment loop (max={max_exp}) ---\n")

    while max_exp == 0 or agent_state["experiment_count"] < max_exp:
        n = agent_state["experiment_count"] + 1
        print(f"=== Experiment {n} ===")
        agent_state = run_single_iteration(agent_state)
        state_ref[0] = agent_state  # keep signal handler in sync
        print()

    print("Max experiments reached. Done.")


def run_single_iteration(agent_state):
    repo_path = agent_state["repo_path"]
    train_path = os.path.join(repo_path, "train.py")

    # Ensure we start from the best commit
    state.reset_to_commit(repo_path, agent_state["best_commit"])

    # --- Propose ---
    print("  Proposing...")
    try:
        proposal = planner.propose_experiment(agent_state)
    except Exception as e:
        print(f"  Proposal failed: {e}")
        state.save_state(agent_state)
        return agent_state

    print(f"  >> {proposal['description']}  (risk: {proposal['risk']})")

    # --- Validate & apply edits ---
    original_text = state.read_file(train_path)
    missing = planner.validate_edit_targets(train_path, proposal["edits"])

    if missing:
        # Retry once with error feedback
        error_msg = f"Search strings not found: {missing}"
        print(f"  Edit failed: {error_msg}")
        print("  Retrying...")
        retry_desc = proposal["description"]
        try:
            proposal = planner.propose_experiment(agent_state, error_context=error_msg)
            retry_desc = proposal["description"]
            print(f"  >> {proposal['description']}  (risk: {proposal['risk']})")
            missing = planner.validate_edit_targets(train_path, proposal["edits"])
        except Exception:
            missing = ["retry failed"]

        if missing:
            print("  Edit failed after retry")
            commit = state.get_current_commit(repo_path)
            results.append_result(commit, None, None, "edit_failed", retry_desc)
            state.save_state(agent_state)
            return agent_state

    try:
        new_text = planner.apply_edits(train_path, proposal["edits"])
        diff = planner.preview_diff(original_text, new_text)
        if diff:
            print(diff)
    except ValueError as e:
        print(f"  Apply failed: {e}")
        state.write_file(train_path, original_text)
        commit = state.get_current_commit(repo_path)
        results.append_result(commit, None, None, "edit_failed", proposal["description"])
        state.save_state(agent_state)
        return agent_state

    # --- Skip if edits produced no change ---
    if not diff:
        print("  No-op edit (content unchanged), skipping")
        state.write_file(train_path, original_text)
        commit = state.get_current_commit(repo_path)
        results.append_result(commit, None, None, "edit_failed", proposal["description"])
        agent_state["experiment_count"] += 1
        state.save_state(agent_state)
        return agent_state

    # --- Commit ---
    state.commit_train_change(repo_path, f"exp: {proposal['description']}")
    commit = state.get_current_commit(repo_path)
    print(f"  Committed: {commit}")

    # --- Run training ---
    print("  Training...")
    run_result = results.run_experiment(repo_path, timeout_seconds=TRAIN_TIMEOUT)

    # --- Parse metrics ---
    val_bpb, peak_vram_mb = None, None
    status = "crash"

    if results.did_timeout(run_result):
        print("  TIMEOUT")
    elif results.did_command_fail(run_result):
        log_text = state.read_file(results.RUN_LOG)
        print(f"  CRASH\n{results.extract_error_tail(log_text)}")
    else:
        val_bpb, peak_vram_mb = results.parse_metrics()
        if val_bpb is not None:
            status = results.decide_result_status(
                val_bpb, peak_vram_mb,
                agent_state["best_val_bpb"],
                agent_state.get("best_peak_vram_mb"),
            )
        else:
            log_text = state.read_file(results.RUN_LOG)
            print(f"  No metrics\n{results.extract_error_tail(log_text)}")

    # --- Log ---
    results.append_result(commit, val_bpb, peak_vram_mb, status, proposal["description"])

    # --- Keep or revert ---
    if status == "keep":
        print(f"  KEEP  val_bpb={val_bpb:.6f} (was {agent_state['best_val_bpb']:.6f})")
        agent_state["best_commit"] = commit
        agent_state["best_val_bpb"] = val_bpb
        agent_state["best_peak_vram_mb"] = peak_vram_mb
    else:
        val_str = f"{val_bpb:.6f}" if val_bpb is not None else "—"
        print(f"  {status.upper()}  val_bpb={val_str}")
        state.reset_to_commit(repo_path, agent_state["best_commit"])

    agent_state["experiment_count"] += 1
    state.save_state(agent_state)
    return agent_state


# ---------------------------------------------------------------------------
# Shutdown
# ---------------------------------------------------------------------------

def shutdown_gracefully(agent_state):
    print("\nShutting down...")
    try:
        state.save_state(agent_state)
        if agent_state.get("best_commit"):
            state.reset_to_commit(agent_state["repo_path"], agent_state["best_commit"])
    except Exception as e:
        print(f"Warning: {e}")
    print("State saved. Exiting.")
    sys.exit(0)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Autoresearch Agent Scaffold")
    parser.add_argument("--repo-path", default=DEFAULT_REPO_PATH)
    parser.add_argument("--source-repo", default=DEFAULT_SOURCE_REPO)
    parser.add_argument("--max-experiments", type=int, default=100)
    parser.add_argument("--llm-base-url", default="http://localhost:8000/v1")
    parser.add_argument("--llm-model", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--log-dir", default=None, help="Directory for results.tsv, run.log, state.json")
    parser.add_argument("--resume", action="store_true", help="Resume from state.json")
    args = parser.parse_args()

    if args.log_dir:
        os.makedirs(args.log_dir, exist_ok=True)
        results.RESULTS_FILE = os.path.join(args.log_dir, "results.tsv")
        results.RUN_LOG = os.path.join(args.log_dir, "run.log")
        state.STATE_FILE = os.path.join(args.log_dir, "state.json")

    # Copy from source repo if working repo doesn't exist
    repo_path = os.path.abspath(args.repo_path)
    if not os.path.exists(repo_path):
        source = os.path.abspath(args.source_repo)
        if not os.path.exists(source):
            print(f"ERROR: source repo not found at {source}")
            sys.exit(1)
        print(f"Copying {source} -> {repo_path}")
        shutil.copytree(source, repo_path)

    if args.resume and os.path.exists(state.STATE_FILE):
        print("Resuming from state.json")
        agent_state = state.load_state()
        agent_state["max_experiments"] = args.max_experiments
    else:
        agent_state = run_setup(args.repo_path, args.max_experiments)

    # CLI overrides always win
    agent_state["llm_base_url"] = args.llm_base_url
    agent_state["llm_model"] = args.llm_model

    # Use a mutable container so the signal handler always sees current state
    state_ref = [agent_state]
    signal.signal(signal.SIGINT, lambda *_: shutdown_gracefully(state_ref[0]))
    signal.signal(signal.SIGTERM, lambda *_: shutdown_gracefully(state_ref[0]))

    run_experiment_loop(state_ref)


if __name__ == "__main__":
    main()
