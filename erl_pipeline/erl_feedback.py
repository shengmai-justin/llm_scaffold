"""Batch-level structured feedback from attempt1 results.

Builds a single feedback string summarizing ALL first attempts in a step,
used as input to the one-per-step reflection.
"""
from __future__ import annotations


def build_attempt_feedback(
    description: str,
    status: str,
    val_bpb: float | None,
    best_val_bpb: float,
    eval_output: str | None = None,
    edit_error: str | None = None,
    crash_signature: dict | None = None,
) -> str:
    """Build feedback for a single attempt."""
    lines = [f"Experiment: {description}"]

    if status == "edit_failed":
        lines.append("Result: EDIT FAILED — search/replace could not be applied.")
        if edit_error:
            lines.append(f"Error: {edit_error}")
    elif status == "crash":
        lines.append("Result: CRASH — train.py failed to produce val_bpb.")
        if crash_signature:
            kind = crash_signature.get("kind") or "unknown"
            cls = crash_signature.get("exception_class")
            msg = crash_signature.get("message")
            loc = crash_signature.get("location")
            label = cls if cls else kind.upper()
            line = f"Crash: {label}"
            if msg:
                line += f": {msg}"
            if loc:
                line += f"  @ {loc}"
            lines.append(line)
            tail = crash_signature.get("tail")
            if tail:
                lines.append(f"Tail:\n{tail}")
        elif eval_output:
            lines.append(f"Last output:\n{eval_output.strip()[-300:]}")
    elif status == "timeout":
        lines.append("Result: TIMEOUT — exceeded time budget.")
    elif val_bpb is not None:
        delta = val_bpb - best_val_bpb
        sign = "+" if delta >= 0 else ""
        lines.append(f"Result: val_bpb={val_bpb:.6f} (best={best_val_bpb:.6f}, delta={sign}{delta:.6f})")
        if delta < 0:
            lines.append("IMPROVED over current best.")
        elif delta == 0:
            lines.append("No change from current best.")
        else:
            lines.append("REGRESSED from current best.")
    else:
        lines.append("Result: no metrics produced.")

    return "\n".join(lines)


def build_batch_feedback(
    attempts: list[dict],
    best_val_bpb: float,
) -> str:
    """Build batch-level feedback from all attempt1 results in a step.

    Args:
        attempts: list of dicts with keys:
            description, status, val_bpb, eval_output (optional), edit_error (optional)
        best_val_bpb: current best val_bpb

    Returns:
        Structured feedback string for the reflection prompt.
    """
    n_total = len(attempts)
    n_improved = sum(1 for a in attempts if a["val_bpb"] is not None and a["val_bpb"] < best_val_bpb)
    n_crashed = sum(1 for a in attempts if a["status"] in ("crash", "timeout", "edit_failed"))

    header = (
        f"## Batch Summary\n"
        f"- Total attempts: {n_total}\n"
        f"- Improved: {n_improved}\n"
        f"- Failed (crash/timeout/edit_failed): {n_crashed}\n"
        f"- Current best val_bpb: {best_val_bpb:.6f}\n"
    )

    sections = []
    for i, a in enumerate(attempts):
        fb = build_attempt_feedback(
            description=a["description"],
            status=a["status"],
            val_bpb=a["val_bpb"],
            best_val_bpb=best_val_bpb,
            eval_output=a.get("eval_output"),
            edit_error=a.get("edit_error"),
            crash_signature=a.get("crash_signature"),
        )
        sections.append(f"### Attempt {i+1}\n{fb}")

    return header + "\n" + "\n\n".join(sections)
