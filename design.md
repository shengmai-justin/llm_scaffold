# Autoresearch Agent Scaffold

> A deterministic experiment harness where the LLM proposes one small `search/replace` edit to `train.py` at a time — and the harness handles everything else.

---

## Design Principles

| Principle | Rule |
|---|---|
| **Small edits** | One hyperparameter tweak, one local architecture change, or one simplification per experiment |
| **Deterministic harness** | Git, execution, parsing, logging, and keep/revert are never delegated to the LLM |
| **External memory** | Progress lives in files (`state.json`, `results.tsv`), not the context window |
| **Structured proposals** | The planner returns JSON only — no free-form text to parse |

---

## File Structure

```
main.py      — setup, experiment loop, keep/revert, recovery
state.py     — persistent state, git operations, file I/O
planner.py   — context assembly, LLM proposal, search/replace editing
results.py   — execution, log parsing, result logging, keep/discard decision

state.json   — persistent agent state
results.tsv  — full experiment history
run.log      — latest training output
```

---

## Execution Flow

### Setup

```
1. Generate run tag and create experiment branch
2. Verify data cache
3. Read in-scope files (README.md, prepare.py, train.py)
4. Initialize results.tsv
5. Run baseline and save initial state
```

### Experiment Loop

```
1.  Load current best state
2.  Build planner context (train.py + recent results + rules)
3.  Request one experiment proposal from LLM
4.  Validate proposal structure
5.  Apply search/replace edits to train.py
    — If edits fail (search string not found), feed error back to LLM and retry once
    — If retry also fails, log as "edit_failed" and continue to next iteration
6.  Commit change
7.  Run training → run.log
8.  Parse metrics
9.  Log result row to results.tsv
10. Keep or revert
11. Update state.json
12. Repeat (up to max_experiments if set, otherwise until stopped)
```

---

## Module Reference

### `main.py` — Orchestration

Setup, the experiment loop, keep/revert decisions, and clean recovery. The only file with top-level control flow.

| Function | Description |
|---|---|
| `main()` | Entry point — runs setup then the experiment loop |
| `run_setup()` | Runs all setup steps in order |
| `run_baseline()` | Executes the unchanged baseline and records it as the starting point |
| `run_experiment_loop()` | Runs iterations indefinitely until externally stopped |
| `run_single_iteration()` | One complete cycle: propose → edit → run → log → keep/revert |
| `shutdown_gracefully()` | Saves state and avoids leaving the repo dirty on exit |

---

### `state.py` — Persistence, Git, File I/O

Everything that reads from or writes to disk: agent state, git operations, and raw file access. Mutation logic (incrementing counters, updating best) lives in `main.py` — this module only loads and saves.

**State**

| Function | Description |
|---|---|
| `load_state()` | Loads `state.json` from disk |
| `save_state(state)` | Writes the latest state to disk |
| `initialize_state(repo_path, run_tag, branch_name)` | Creates the initial state object |

**Git**

| Function | Description |
|---|---|
| `get_current_commit()` | Returns the current short commit hash |
| `create_experiment_branch(branch_name)` | Creates and switches to the experiment branch |
| `commit_train_change(message)` | Commits the current `train.py` modification |
| `reset_to_commit(commit_hash)` | Resets the repo to a known good commit (git is the backup — no separate file backup needed) |
| `verify_clean_repo()` | Checks the working tree is clean before starting |

**Files**

| Function | Description |
|---|---|
| `read_file(path)` | Reads a file and returns its contents |
| `write_file(path, content)` | Writes content to a file |

**State file:** `state.json`

```json
{
  "repo_path": "...",
  "run_tag": "2025-01-15",
  "branch_name": "autoresearch/2025-01-15",
  "best_commit": "a3f9c12",
  "best_val_bpb": 1.423,
  "experiment_count": 7,
  "max_experiments": 100,
  "llm_base_url": "http://localhost:8000/v1",
  "llm_model": "Qwen/Qwen2.5-72B-Instruct"
}
```

---

### `planner.py` — Context, Proposal, Editing

Assembles the LLM prompt, calls the model, validates the output, and applies edits to `train.py`. The only file that touches the LLM.

**LLM backend:** The planner talks to any OpenAI-compatible endpoint. This means it works with both remote APIs (OpenAI, Anthropic) and local models served via `vllm`, `ollama`, `llama.cpp`, etc. Configuration is just a base URL + model name — no provider-specific code. The planner only cares that the response is valid JSON matching the proposal schema.

**Context**

| Function | Description |
|---|---|
| `build_planner_context()` | Constructs the full prompt for one iteration |
| `summarize_recent_results(results_tsv, n)` | Compact summary of the last `n` experiments |
| `build_system_rules()` | Stable constraints: "only edit train.py", "prefer small changes" |

**Proposal**

| Function | Description |
|---|---|
| `propose_experiment(context)` | Calls the LLM and returns a structured proposal. On network/parse failure, retries once then raises. |
| `validate_planner_output(proposal)` | Checks required fields and edit structure |

**Editing**

| Function | Description |
|---|---|
| `apply_edits(file_path, edits)` | Applies a sequence of search/replace edits |
| `validate_edit_targets(file_path, edits)` | Verifies every search string exists before editing |
| `preview_diff(original_text, new_text)` | Returns a human-readable summary of what changed |

**Proposal format:**

```json
{
  "description": "Increase embedding learning rate from 0.6 to 0.8",
  "rationale": "Embeddings may benefit from faster initial learning at this model scale",
  "risk": "low",
  "edits": [
    { "search": "EMBEDDING_LR = 0.6", "replace": "EMBEDDING_LR = 0.8" }
  ]
}
```

---

### `results.py` — Execution, Parsing, Logging, Decision

Everything that happens after an edit is committed: run the training script, extract metrics, log the row, and decide keep or discard.

**Execution**

| Function | Description |
|---|---|
| `run_experiment(command, timeout_seconds)` | Runs `uv run train.py > run.log 2>&1` with timeout |
| `did_timeout(run_result)` | Returns `True` if the run exceeded the time budget |
| `did_command_fail(run_result)` | Returns `True` if the command returned an error |

**Parsing**

| Function | Description |
|---|---|
| `parse_metrics(run_log_path)` | Extracts `val_bpb` and peak VRAM from `run.log` |
| `detect_crash(log_text)` | Determines whether the run crashed or produced no metrics |
| `extract_error_tail(log_text, n_lines)` | Returns the last `n` lines of the log for crash debugging |

**Logging**

| Function | Description |
|---|---|
| `ensure_results_tsv()` | Creates `results.tsv` with the correct header if missing |
| `append_result(commit, val_bpb, memory_gb, status, description)` | Appends one row per experiment |
| `read_results_history()` | Loads previous records for use in planning |

**Decision**

| Function | Description |
|---|---|
| `decide_result_status(new_result, best_result)` | Returns `keep`, `discard`, or `crash` |
| `is_improvement(new_val_bpb, best_val_bpb)` | Returns `True` if the new result is strictly better |

**Decision rules:**

| Status | Condition |
|---|---|
| `keep` | `val_bpb` is strictly lower than the current best, OR equal `val_bpb` with lower peak VRAM |
| `discard` | `val_bpb` is equal or worse (and no VRAM improvement) |
| `crash` | Metrics are missing or the run timed out |
| `edit_failed` | Search/replace edits could not be applied (even after retry) |

**Results file:** `results.tsv`

```
commit    val_bpb    peak_vram_mb    status       description
a3f9c12   1.423      12698           keep         Increased EMBEDDING_LR 0.6→0.8
b71d3a8   1.451      12698           discard      Increased DEPTH 8→10
c90f22e   —          —               crash        Changed activation to GELU
d12e4f0   —          —               edit_failed  Search string not found in train.py
```

---

## Internal Data Objects

### `ExperimentProposal`

| Field | Type | Description |
|---|---|---|
| `description` | `str` | Human-readable summary of the change |
| `rationale` | `str` | Why this change was proposed |
| `risk` | `str` | `low`, `medium`, or `high` |
| `edits` | `list[Edit]` | Ordered search/replace operations |

### `Edit`

| Field | Type | Description |
|---|---|---|
| `search` | `str` | Exact string to find in `train.py` |
| `replace` | `str` | String to substitute in its place |

### `ExperimentResult`

| Field | Type | Description |
|---|---|---|
| `commit` | `str` | Short commit hash |
| `val_bpb` | `float \| None` | Validation bits-per-byte |
| `peak_vram_mb` | `int \| None` | Peak VRAM in MB |
| `status` | `str` | `keep`, `discard`, `crash`, or `edit_failed` |
| `description` | `str` | Sanitized experiment description |

### `AgentState`

| Field | Type | Description |
|---|---|---|
| `repo_path` | `str` | Absolute path to the repo |
| `branch_name` | `str` | Active experiment branch |
| `run_tag` | `str` | Identifier for this run series |
| `best_commit` | `str` | Short hash of the best run so far |
| `best_val_bpb` | `float` | Best validation metric seen |
| `experiment_count` | `int` | Total experiments attempted |
| `max_experiments` | `int` | Stop after this many iterations (0 = unlimited) |
| `llm_base_url` | `str` | OpenAI-compatible endpoint (local or remote) |
| `llm_model` | `str` | Model name to request from the endpoint |

---

## Summary

| File | Owns |
|---|---|
| `main.py` | Control flow, setup, loop, recovery |
| `state.py` | All I/O — state, git, files |
| `planner.py` | Context, LLM call, editing |
| `results.py` | Execution, parsing, logging, decision |

The LLM decides *what to try*. The harness decides *everything else*.
