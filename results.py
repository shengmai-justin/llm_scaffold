"""Execution, log parsing, result logging, and keep/discard decision."""

import os
import re
import subprocess

SCAFFOLD_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_FILE = os.path.join(SCAFFOLD_DIR, "results.tsv")
RUN_LOG = os.path.join(SCAFFOLD_DIR, "run.log")

HEADER = "commit\tval_bpb\tpeak_vram_mb\tstatus\tdescription\n"


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------

def run_experiment(repo_path, timeout_seconds=600):
    """Run uv run train.py, capture output to run.log."""
    with open(RUN_LOG, "w") as log_fh:
        try:
            r = subprocess.run(
                ["uv", "run", "train.py"],
                cwd=repo_path,
                stdout=log_fh,
                stderr=subprocess.STDOUT,
                timeout=timeout_seconds,
            )
            return {"returncode": r.returncode, "timed_out": False}
        except subprocess.TimeoutExpired:
            return {"returncode": -1, "timed_out": True}


def did_timeout(run_result):
    return run_result["timed_out"]


def did_command_fail(run_result):
    return run_result["returncode"] != 0


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def parse_metrics(run_log_path=None):
    """Extract val_bpb and peak_vram_mb from run.log. Returns (float|None, int|None)."""
    if run_log_path is None:
        run_log_path = RUN_LOG
    with open(run_log_path) as f:
        text = f.read()

    val_bpb = None
    peak_vram_mb = None

    m = re.search(r"val_bpb:\s+([\d.]+)", text)
    if m:
        val_bpb = float(m.group(1))

    m = re.search(r"peak_vram_mb:\s+([\d.]+)", text)
    if m:
        peak_vram_mb = int(float(m.group(1)))

    return val_bpb, peak_vram_mb



def extract_error_tail(log_text, n_lines=20):
    lines = log_text.strip().split("\n")
    return "\n".join(lines[-n_lines:])


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def ensure_results_tsv():
    if not os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, "w") as f:
            f.write(HEADER)


def append_result(commit, val_bpb, peak_vram_mb, status, description):
    val_str = f"{val_bpb:.6f}" if val_bpb is not None else "—"
    vram_str = str(peak_vram_mb) if peak_vram_mb is not None else "—"
    desc = description.replace("\t", " ").replace("\n", " ").strip()
    with open(RESULTS_FILE, "a") as f:
        f.write(f"{commit}\t{val_str}\t{vram_str}\t{status}\t{desc}\n")


def read_results_history():
    """Load results.tsv into a list of dicts."""
    if not os.path.exists(RESULTS_FILE):
        return []
    rows = []
    with open(RESULTS_FILE) as f:
        header = f.readline().strip().split("\t")
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) != len(header):
                continue
            row = dict(zip(header, parts))
            row["val_bpb"] = float(row["val_bpb"]) if row["val_bpb"] != "—" else None
            row["peak_vram_mb"] = int(row["peak_vram_mb"]) if row["peak_vram_mb"] != "—" else None
            rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Decision
# ---------------------------------------------------------------------------

def decide_result_status(val_bpb, peak_vram_mb, best_val_bpb, best_peak_vram_mb):
    """Returns 'keep', 'discard', or 'crash'."""
    if val_bpb is None:
        return "crash"
    if val_bpb < best_val_bpb:
        return "keep"
    if val_bpb == best_val_bpb and peak_vram_mb is not None and best_peak_vram_mb is not None:
        if peak_vram_mb < best_peak_vram_mb:
            return "keep"
    return "discard"
