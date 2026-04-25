"""Probe Qwen3.5's knowledge of techniques that never appeared in ERL rollouts.

Reads `knowledge_probes.txt` line-by-line. For each probe, calls the local
OpenAI-compatible endpoint with the probe as the target technique and the
current train.py as context, then saves the raw response + a summary of
validity (JSON parsed, search strings exist in train.py, number of edits).

Usage:
    # Point at whatever SGLang / vLLM endpoint is already serving Qwen:
    python probes/run_probes.py \
        --base-url http://localhost:30000/v1 \
        --model Qwen/Qwen3.5-9B \
        --train-py autoresearch/train.py

The endpoint must be OpenAI-compatible (SGLang, vLLM, TGI, or openai.com all
work). For no-auth local servers set OPENAI_API_KEY=not-needed in env.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

from openai import OpenAI

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def slugify(s: str, maxlen: int = 60) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")[:maxlen]


def extract_json(text: str) -> dict | None:
    """Parse first valid JSON object in `text`. Tolerates <think> blocks,
    markdown fences, and trailing prose by walking back from the last '}'."""
    clean = text
    if "</think>" in clean:
        clean = clean.split("</think>", 1)[1]
    clean = re.sub(r"<think>.*?</think>", "", clean, flags=re.DOTALL)
    clean = re.sub(r"^```[a-zA-Z]*\n?", "", clean.strip())
    clean = re.sub(r"\n?```\s*$", "", clean).strip()
    try:
        return json.loads(clean)
    except json.JSONDecodeError:
        pass
    end = text.rfind("}")
    if end == -1:
        return None
    depth = 0
    for i in range(end, -1, -1):
        if text[i] == "}":
            depth += 1
        elif text[i] == "{":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[i:end + 1])
                except json.JSONDecodeError:
                    return None
    return None


def classify(valid_json: bool, num_edits: int, missing: int) -> str:
    if not valid_json:
        return "BAD_JSON"
    if num_edits == 0:
        return "ADMITTED"
    if missing == 0:
        return "OK"
    return "PARTIAL"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url",
                        default=os.environ.get("OPENAI_BASE_URL", "http://localhost:8000/v1"))
    parser.add_argument("--model", default=os.environ.get("LLM_MODEL", "Qwen/Qwen3.5-9B"))
    parser.add_argument("--train-py", default=str(ROOT / "autoresearch" / "train.py"))
    parser.add_argument("--probes", default=str(HERE / "knowledge_probes.txt"))
    parser.add_argument("--prompt", default=str(HERE / "prompt_knowledge_probe.md"))
    parser.add_argument("--output-dir", default=str(HERE / "outputs"))
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-tokens", type=int, default=8000)
    parser.add_argument("--retries", type=int, default=0,
                        help="Extra attempts when output is BAD_JSON (parse failure). "
                             "Each retry uses temperature+0.1 for variation. "
                             "ADMITTED and PARTIAL results are not retried.")
    args = parser.parse_args()

    train_py_path = Path(args.train_py)
    if not train_py_path.exists():
        sys.exit(f"train.py not found at {train_py_path}")
    train_py = train_py_path.read_text()

    probes = [ln.strip() for ln in Path(args.probes).read_text().splitlines()
              if ln.strip() and not ln.lstrip().startswith("#")]
    if not probes:
        sys.exit("no probes found")
    system_msg = Path(args.prompt).read_text()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    client = OpenAI(
        base_url=args.base_url,
        api_key=os.environ.get("OPENAI_API_KEY", "not-needed"),
    )

    user_template = (
        f"Current train.py:\n```python\n{train_py}\n```\n\n"
        f"Technique to implement: {{probe}}\n\n"
        f"Produce the JSON edits."
    )

    def call_once(probe: str, temperature: float) -> tuple[str, dict | None, int, int, int]:
        """Returns (raw_text, proposal, num_edits, valid_edits, missing_edits)."""
        try:
            resp = client.chat.completions.create(
                model=args.model,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_template.format(probe=probe)},
                ],
                temperature=temperature,
                max_tokens=args.max_tokens,
            )
            text = resp.choices[0].message.content or ""
        except Exception as e:
            text = f"__ERROR__ {type(e).__name__}: {e}"

        prop = extract_json(text) if not text.startswith("__ERROR__") else None
        valid = prop is not None and isinstance(prop.get("edits"), list)
        num = len(prop["edits"]) if valid else 0
        match = miss = 0
        if valid:
            for e in prop["edits"]:
                if not (isinstance(e, dict) and "search" in e and "replace" in e):
                    miss += 1
                    continue
                if e["search"] and e["search"] in train_py:
                    match += 1
                else:
                    miss += 1
        return text, prop, num, match, miss

    summary = []
    for i, probe in enumerate(probes):
        print(f"\n[{i + 1}/{len(probes)}] {probe}")
        attempts = 0
        text, proposal, num_edits, valid_edits, missing_edits = call_once(probe, args.temperature)
        attempts += 1
        # Retry BAD_JSON only — temperature bumps slightly each retry to shake noise.
        while (proposal is None or not isinstance(proposal.get("edits"), list)) \
                and attempts <= args.retries:
            t = args.temperature + 0.1 * attempts
            print(f"  retry {attempts}/{args.retries} at T={t:.2f} (BAD_JSON)")
            text, proposal, num_edits, valid_edits, missing_edits = call_once(probe, t)
            attempts += 1

        slug = slugify(probe)
        (out_dir / f"{i:02d}_{slug}.txt").write_text(text)

        valid_json = proposal is not None and isinstance(proposal.get("edits"), list)
        status = classify(valid_json, num_edits, missing_edits)
        summary.append({
            "probe": probe,
            "status": status,
            "valid_json": valid_json,
            "num_edits": num_edits,
            "valid_edits": valid_edits,
            "missing_edits": missing_edits,
            "attempts": attempts,
            "description": proposal.get("description", "") if valid_json else "",
        })
        print(f"  -> {status}  edits={num_edits} match={valid_edits} miss={missing_edits} attempts={attempts}")

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    print("\n=== Summary ===")
    print(f"{'probe':52s} {'status':10s} {'edits':>5} {'match':>5} {'miss':>4}")
    for s in summary:
        probe = s["probe"][:50]
        print(f"{probe:52s} {s['status']:10s} {s['num_edits']:>5} {s['valid_edits']:>5} {s['missing_edits']:>4}")

    ok = sum(1 for s in summary if s["status"] == "OK")
    adm = sum(1 for s in summary if s["status"] == "ADMITTED")
    par = sum(1 for s in summary if s["status"] == "PARTIAL")
    bad = sum(1 for s in summary if s["status"] == "BAD_JSON")
    print(f"\nTotals: OK={ok}  ADMITTED={adm}  PARTIAL={par}  BAD_JSON={bad}  total={len(summary)}")


if __name__ == "__main__":
    main()
