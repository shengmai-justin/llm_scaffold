"""Context assembly, LLM proposal, and search/replace editing."""

import difflib
import json
import os
import re

from openai import OpenAI

import state as state_mod


# ---------------------------------------------------------------------------
# Context
# ---------------------------------------------------------------------------

PROMPT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompt.md")


def build_system_rules():
    return state_mod.read_file(PROMPT_FILE)


def summarize_recent_results(n=10):
    """Compact summary of the last n experiments from results.tsv."""
    from results import read_results_history
    rows = read_results_history()
    if not rows:
        return "No previous experiments."
    recent = rows[-n:]
    lines = []
    for r in recent:
        val = f"{r['val_bpb']:.6f}" if r["val_bpb"] is not None else "—"
        vram = str(r["peak_vram_mb"]) if r["peak_vram_mb"] is not None else "—"
        lines.append(f"  {r['status']:12s} val_bpb={val}  vram={vram}  {r['description']}")
    return "Recent experiments (oldest first):\n" + "\n".join(lines)


def build_planner_context(repo_path, best_val_bpb):
    """Construct the system + user messages for one proposal."""
    train_py = state_mod.read_file(os.path.join(repo_path, "train.py"))
    recent = summarize_recent_results()

    user_msg = (
        f"Current train.py:\n"
        f"```python\n{train_py}\n```\n\n"
        f"Current best val_bpb: {best_val_bpb:.6f}\n\n"
        f"{recent}\n\n"
        f"Propose one small experiment to improve val_bpb (lower is better)."
    )
    return build_system_rules(), user_msg


# ---------------------------------------------------------------------------
# Proposal
# ---------------------------------------------------------------------------

def propose_experiment(agent_state, error_context=None):
    """Call the LLM and return a validated proposal dict. Retries once on failure."""
    client = OpenAI(
        base_url=agent_state["llm_base_url"],
        api_key=os.environ.get("OPENAI_API_KEY", "not-needed"),
    )

    system_msg, user_msg = build_planner_context(
        agent_state["repo_path"], agent_state["best_val_bpb"]
    )
    if error_context:
        user_msg += (
            f"\n\nPrevious edit attempt failed: {error_context}\n"
            f"Please propose a corrected experiment."
        )

    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]

    for attempt in range(2):
        try:
            resp = client.chat.completions.create(
                model=agent_state["llm_model"],
                messages=messages,
                temperature=0.7,
                max_tokens=1024,
            )
            text = resp.choices[0].message.content.strip()
            # Strip markdown code fences if present
            text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
            text = re.sub(r"\n?```\s*$", "", text)
            text = text.strip()
            proposal = json.loads(text)
            validate_planner_output(proposal)
            return proposal
        except Exception as e:
            if attempt == 0:
                print(f"  LLM attempt 1 failed ({e}), retrying...")
                continue
            raise


def validate_planner_output(proposal):
    """Check required fields and edit structure."""
    for field in ("description", "rationale", "risk", "edits"):
        if field not in proposal:
            raise ValueError(f"Missing required field: {field}")

    if proposal["risk"] not in ("low", "medium", "high"):
        raise ValueError(f"Invalid risk level: {proposal['risk']}")

    if not isinstance(proposal["edits"], list) or len(proposal["edits"]) == 0:
        raise ValueError("edits must be a non-empty list")

    for edit in proposal["edits"]:
        if "search" not in edit or "replace" not in edit:
            raise ValueError("Each edit must have 'search' and 'replace' fields")
        if not edit["search"]:
            raise ValueError("search string cannot be empty")


# ---------------------------------------------------------------------------
# Editing
# ---------------------------------------------------------------------------

def validate_edit_targets(file_path, edits):
    """Returns list of search strings not found in the file."""
    content = state_mod.read_file(file_path)
    missing = [e["search"] for e in edits if e["search"] not in content]
    return missing


def apply_edits(file_path, edits):
    """Apply search/replace edits sequentially. Raises ValueError on miss."""
    content = state_mod.read_file(file_path)
    for edit in edits:
        if edit["search"] not in content:
            raise ValueError(f"Search string not found: {edit['search'][:80]!r}")
        content = content.replace(edit["search"], edit["replace"], 1)
    state_mod.write_file(file_path, content)
    return content


def preview_diff(original_text, new_text):
    """Returns a unified diff string."""
    diff = difflib.unified_diff(
        original_text.splitlines(keepends=True),
        new_text.splitlines(keepends=True),
        fromfile="train.py",
        tofile="train.py",
    )
    return "".join(diff)
