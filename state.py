"""Persistent state, git operations, and file I/O."""

import json
import os
import subprocess

STATE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "state.json")


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

def load_state():
    with open(STATE_FILE) as f:
        return json.load(f)


def save_state(s):
    with open(STATE_FILE, "w") as f:
        json.dump(s, f, indent=2)
        f.write("\n")


def initialize_state(repo_path, run_tag, branch_name):
    return {
        "repo_path": repo_path,
        "run_tag": run_tag,
        "branch_name": branch_name,
        "best_commit": None,
        "best_val_bpb": float("inf"),
        "best_peak_vram_mb": None,
        "experiment_count": 0,
        "max_experiments": 100,
        "llm_base_url": "http://localhost:8000/v1",
        "llm_model": "Qwen/Qwen3.5-9B",
    }


# ---------------------------------------------------------------------------
# Git
# ---------------------------------------------------------------------------

def _git(args, repo_path):
    r = subprocess.run(
        ["git"] + args,
        cwd=repo_path,
        capture_output=True,
        text=True,
    )
    if r.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {r.stderr.strip()}")
    return r.stdout.strip()


def get_current_commit(repo_path):
    return _git(["rev-parse", "--short", "HEAD"], repo_path)


def create_experiment_branch(repo_path, branch_name):
    _git(["checkout", "-b", branch_name], repo_path)


def commit_train_change(repo_path, message):
    _git(["add", "train.py"], repo_path)
    _git(["commit", "-m", message], repo_path)


def reset_to_commit(repo_path, commit_hash):
    _git(["reset", "--hard", commit_hash], repo_path)


# ---------------------------------------------------------------------------
# File I/O
# ---------------------------------------------------------------------------

def read_file(path):
    with open(path) as f:
        return f.read()


def write_file(path, content):
    with open(path, "w") as f:
        f.write(content)
