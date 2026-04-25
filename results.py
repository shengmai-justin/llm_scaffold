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


_TRACEBACK_RE = re.compile(r"Traceback \(most recent call last\):", re.MULTILINE)
_EXC_LINE_RE = re.compile(r"^([A-Za-z_][\w\.]*(?:Error|Exception|Warning|Interrupt)): ?(.*)$", re.MULTILINE)
_FRAME_RE = re.compile(r'File "([^"]+)", line (\d+), in (\S+)')
_OOM_RE = re.compile(r"(CUDA out of memory|torch\.cuda\.OutOfMemoryError|OutOfMemoryError)", re.IGNORECASE)
_SEGFAULT_RE = re.compile(r"(Segmentation fault|SIGSEGV|core dumped)", re.IGNORECASE)


def extract_crash_signature_from_text(text, timed_out=False, max_msg_chars=240, tail_lines=12):
    """Parse a stdout+stderr string for a structured crash signature.

    Returns dict with keys {kind, exception_class, message, location, tail}.
    Non-crash text still yields a signature with kind='unknown' — caller
    filters by status.
    """
    tail = extract_error_tail(text, n_lines=tail_lines)

    if timed_out:
        return {"kind": "timeout", "exception_class": None,
                "message": "train.py exceeded timeout", "location": None, "tail": tail}

    # Python traceback (preferred — most informative)
    tb_matches = list(_TRACEBACK_RE.finditer(text))
    if tb_matches:
        tb_start = tb_matches[-1].start()
        tb_block = text[tb_start:]
        exc_match = None
        for m in _EXC_LINE_RE.finditer(tb_block):
            exc_match = m
        train_frames = [m for m in _FRAME_RE.finditer(tb_block) if "train.py" in m.group(1)]
        location = None
        if train_frames:
            last = train_frames[-1]
            location = f"train.py:{last.group(2)} in {last.group(3)}"
        elif exc_match:
            any_frames = list(_FRAME_RE.finditer(tb_block))
            if any_frames:
                last = any_frames[-1]
                fn = os.path.basename(last.group(1))
                location = f"{fn}:{last.group(2)} in {last.group(3)}"
        if exc_match:
            return {
                "kind": "exception",
                "exception_class": exc_match.group(1),
                "message": exc_match.group(2).strip()[:max_msg_chars],
                "location": location,
                "tail": tail,
            }

    if _OOM_RE.search(text):
        m = _OOM_RE.search(text)
        return {"kind": "oom", "exception_class": "CUDAOutOfMemoryError",
                "message": m.group(0)[:max_msg_chars], "location": None, "tail": tail}

    if _SEGFAULT_RE.search(text):
        m = _SEGFAULT_RE.search(text)
        return {"kind": "segfault", "exception_class": None,
                "message": m.group(0)[:max_msg_chars], "location": None, "tail": tail}

    return {"kind": "unknown", "exception_class": None,
            "message": None, "location": None, "tail": tail}


def retain_crash_log(run_log_path, log_dir, commit):
    """Copy run.log to <log_dir>/crashes/<commit>.log so it survives the next rollout."""
    if not os.path.exists(run_log_path):
        return None
    crashes_dir = os.path.join(log_dir, "crashes")
    os.makedirs(crashes_dir, exist_ok=True)
    dest = os.path.join(crashes_dir, f"{commit}.log")
    import shutil
    shutil.copy(run_log_path, dest)
    return dest


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
