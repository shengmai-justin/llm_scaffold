"""Cross-step experiment history summarizer.

Once per step the LLM reads results.tsv and produces a compact markdown
table flagging dead-end categories (5+ attempts, 0 keeps) and what has
worked. The table is injected into proposal prompts so the model sees its
own dead-end pattern before deciding what to try next.

This is a SEPARATE call from per-batch reflection (erl_reflect.py):
- Reflection: tactical, sees only the current 4 attempts.
- History:    strategic, sees the full results.tsv across all steps.

No hardcoded category buckets — the LLM picks the categorization, so
new proposal directions are absorbed naturally without code changes.
"""
from __future__ import annotations

import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "rl_pipeline"))

from rl_model import generate_with_logprobs


# Cap context to last N rows so the prompt stays well under the model's
# context window even after a multi-day run.
MAX_ROWS_IN_CONTEXT = 300
MIN_ROWS_TO_SUMMARIZE = 20


HISTORY_SYSTEM = """You are a research assistant summarizing experiment history into a compact reference table. The proposer model keeps repeating the same dead-end ideas, and your job is to surface that pattern so the next proposal avoids it.

Read the experiment log carefully. Group attempts into categories of YOUR choosing (you decide the granularity — examples might be "LR warmup tweaks", "WEIGHT_DECAY adjustments", "WINDOW_PATTERN swaps", "depth changes", "activation swaps", or anything else you spot in the data).

Output format (markdown, no prose outside the tables):

--- EXPERIMENT HISTORY (read this carefully before proposing) ---

The following ideas have been tried REPEATEDLY without success.
DO NOT propose any of them — they are dead ends on this budget.
Synonyms count: rephrasing the same idea (e.g. "LR warmup" vs "gradient ramp at start") will be treated as a repeat.

### Dead-end categories (do not modify these parameters or propose these ideas):
| Category | Attempts | Outcome |
|----------|---------:|---------|
| ... | ... | "0 kept, N crashed, M edit_failed" |

### Directions that have worked — consider deeper variants:
- Brief description of kept change -> val_bpb improvement

### Your task:
Propose a structural or architectural change NOT in the dead-end list. If you must touch a dead-end parameter, justify in your description why your specific value is materially different from prior attempts.

## Rules:
- At most 6 rows in the dead-end table (most-attempted-with-zero-keeps first).
- At most 5 bullets in the worked-directions section (largest improvements first).
- Counts must match the raw log exactly. Do NOT estimate.
- No prose, only the structured output above."""


def _read_results_tsv(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return list(csv.DictReader(f, delimiter="\t"))


def _format_rows(rows: list[dict]) -> str:
    """One-line-per-row compact formatting for the LLM."""
    lines = []
    for i, r in enumerate(rows, 1):
        bpb = r.get("val_bpb") or "—"
        status = r.get("status", "—")
        desc = r.get("description", "").strip()
        lines.append(f"{i}. [{status:11s}] val_bpb={bpb:8s}  {desc}")
    return "\n".join(lines)


def generate_history_summary(
    model,
    tokenizer,
    results_tsv_path: str,
    temperature: float = 0.3,
    max_new_tokens: int = 1024,
    think_budget: int | None = 512,
) -> str:
    """Read results.tsv and ask the LLM to produce a dead-end + worked summary table.

    Returns "" if fewer than MIN_ROWS_TO_SUMMARIZE rows have been logged so
    far (no pattern to summarize yet).

    Temperature is kept low (0.3) to discourage hallucinated counts. The
    summary is used as prompt context only; we do not train on it, so the
    logprobs returned by generate_with_logprobs are discarded.
    """
    rows = _read_results_tsv(results_tsv_path)
    if len(rows) < MIN_ROWS_TO_SUMMARIZE:
        return ""

    if len(rows) > MAX_ROWS_IN_CONTEXT:
        rows = rows[-MAX_ROWS_IN_CONTEXT:]

    user_msg = (
        f"Total experiments in this view: {len(rows)} "
        f"(showing the most recent up to {MAX_ROWS_IN_CONTEXT}).\n\n"
        f"Raw results log:\n{_format_rows(rows)}\n\n"
        f"Produce the summary table now."
    )

    messages = [
        {"role": "system", "content": HISTORY_SYSTEM},
        {"role": "user", "content": user_msg},
    ]
    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    text, _, _, _ = generate_with_logprobs(
        model, tokenizer, prompt_text,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        think_budget=think_budget,
    )

    if "</think>" in text:
        text = text.split("</think>", 1)[1].strip()

    return text
