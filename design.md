# Autoresearch Agent Scaffold

> A deterministic experiment harness where the LLM proposes one small `search/replace` edit to `train.py` at a time — and the harness handles everything else. Includes an RL pipeline (TTT-Discover style) that trains LoRA weights via policy gradient so the LLM learns from outcomes.

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
llm_scaffold/
├── main.py          — setup, experiment loop, keep/revert, recovery
├── state.py         — persistent state, git operations, file I/O
├── planner.py       — context assembly, LLM proposal, search/replace editing
├── results.py       — execution, log parsing, result logging, keep/discard decision
├── prompt.md        — system prompt template for the LLM planner
├── run.sh           — SLURM launch script (1 GPU, SGLang server + frozen pipeline)
├── serve.sh         — standalone SGLang server launch
├── setup.sh         — environment setup helper
├── pyproject.toml   — project metadata (depends on openai>=1.0.0)
│
├── state.json       — persistent agent state (generated at runtime)
├── results.tsv      — full experiment history (generated at runtime)
├── run.log          — latest training output (generated at runtime)
│
├── rl_pipeline/
│   ├── rl_main.py       — RL entry point, sequential + parallel (Ray) modes
│   ├── rl_model.py      — HF model + PEFT LoRA + SDPA + gradient checkpointing
│   ├── rl_planner.py    — wraps planner.py, local generation, strips <think> tags
│   ├── rl_trainer.py    — entropic advantages, per-token KL penalty, policy gradient
│   ├── rl_sampler.py    — PUCT tree search (State stores code, not git commits)
│   ├── rl_eval.py       — Ray EvalWorker for parallel train.py evaluation
│   ├── rl_types.py      — Rollout dataclass
│   ├── run_rl.sh        — SLURM launch script (8 GPUs, no SGLang)
│   ├── smoke_test.py    — GPU smoke test (8 components)
│   ├── debug_generate.py — debug: generate one proposal, print raw output
│   ├── mock_rl_test.py  — mock RL loop test (no GPU needed)
│   └── mps.md           — multi-GPU parallelization plan
│
└── gpu_mem_limit/
    ├── gpu_mem_limit.c  — LD_PRELOAD lib to cap per-process GPU memory
    ├── Makefile         — compile with GCC, links -ldl -lpthread
    └── test_memlimit.sh — test suite (memory reporting, OOM, two-process sharing)
```

---

## Frozen Pipeline

### Execution Flow

#### Setup

```
1. Generate run tag (YYYY-MM-DD) and create experiment branch
2. Reset dirty train.py if interrupted run left uncommitted changes
3. Initialize results.tsv
4. Run baseline and save initial state
```

#### Experiment Loop

```
1.  Reset to best commit
2.  Build planner context (train.py + recent results + rules from prompt.md)
3.  Request one experiment proposal from LLM
4.  Validate proposal structure
5.  Apply search/replace edits to train.py
    — If edits fail (search string not found), feed error back to LLM and retry once
    — If retry also fails, log as "edit_failed" and continue to next iteration
6.  Skip if edits produced no change (no-op)
7.  Commit change
8.  Run training → run.log
9.  Parse metrics
10. Log result row to results.tsv
11. Keep or revert (git reset --hard to best commit)
12. Update state.json
13. Repeat (up to max_experiments, or until stopped)
```

### Module Reference

#### `main.py` — Orchestration

Setup, the experiment loop, keep/revert decisions, and clean recovery. The only file with top-level control flow.

| Function | Description |
|---|---|
| `main()` | Entry point — parses CLI args, runs setup or resumes, then experiment loop |
| `run_setup(repo_path, max_experiments)` | Runs all setup steps in order |
| `run_baseline(agent_state)` | Executes the unchanged baseline and records it as the starting point |
| `run_experiment_loop(state_ref)` | Runs iterations up to max_experiments |
| `run_single_iteration(agent_state)` | One complete cycle: propose → edit → run → log → keep/revert |
| `shutdown_gracefully(agent_state)` | Saves state, resets to best commit, exits cleanly (SIGINT/SIGTERM handler) |

---

#### `state.py` — Persistence, Git, File I/O

Everything that reads from or writes to disk: agent state, git operations, and raw file access. Mutation logic (incrementing counters, updating best) lives in `main.py` — this module only loads and saves.

**State**

| Function | Description |
|---|---|
| `load_state()` | Loads `state.json` from disk |
| `save_state(s)` | Writes the latest state to disk |
| `initialize_state(repo_path, run_tag, branch_name)` | Creates the initial state dict |

**Git**

| Function | Description |
|---|---|
| `get_current_commit(repo_path)` | Returns the current short commit hash |
| `create_experiment_branch(repo_path, branch_name)` | Creates and switches to the experiment branch |
| `commit_train_change(repo_path, message)` | Stages and commits `train.py` |
| `reset_to_commit(repo_path, commit_hash)` | Resets the repo to a known good commit (`git reset --hard`) |
| `verify_clean_repo(repo_path)` | Checks `train.py` has no uncommitted changes |
| `ensure_clean_repo(repo_path)` | Resets dirty `train.py` if needed (e.g. from interrupted run), then verifies clean |

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
  "best_peak_vram_mb": 12698,
  "experiment_count": 7,
  "max_experiments": 100,
  "llm_base_url": "http://localhost:8000/v1",
  "llm_model": "Qwen/Qwen3.5-9B"
}
```

---

#### `planner.py` — Context, Proposal, Editing

Assembles the LLM prompt, calls the model, validates the output, and applies edits to `train.py`. The only frozen-pipeline file that touches the LLM.

**LLM backend:** The planner talks to any OpenAI-compatible endpoint. This means it works with both remote APIs (OpenAI, Anthropic) and local models served via `vllm`, `sglang`, `ollama`, `llama.cpp`, etc. Configuration is just a base URL + model name — no provider-specific code. The planner only cares that the response is valid JSON matching the proposal schema.

**Context**

| Function | Description |
|---|---|
| `build_planner_context(repo_path, best_val_bpb)` | Constructs system + user messages for one iteration |
| `summarize_recent_results(n=10)` | Compact summary of the last `n` experiments from `results.tsv` |
| `build_system_rules()` | Loads stable constraints from `prompt.md` |

**Proposal**

| Function | Description |
|---|---|
| `propose_experiment(agent_state, error_context=None)` | Calls the LLM and returns a validated proposal dict. Retries once on failure. |
| `validate_planner_output(proposal)` | Checks required fields (`description`, `rationale`, `risk`, `edits`) and edit structure |

**Editing**

| Function | Description |
|---|---|
| `apply_edits(file_path, edits)` | Applies a sequence of search/replace edits. Raises `ValueError` on miss. |
| `validate_edit_targets(file_path, edits)` | Returns list of search strings not found in the file |
| `preview_diff(original_text, new_text)` | Returns a unified diff string |

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

#### `results.py` — Execution, Parsing, Logging, Decision

Everything that happens after an edit is committed: run the training script, extract metrics, log the row, and decide keep or discard.

**Execution**

| Function | Description |
|---|---|
| `run_experiment(repo_path, timeout_seconds=600)` | Runs `uv run train.py` in `repo_path`, captures output to `run.log` |
| `did_timeout(run_result)` | Returns `True` if the run exceeded the time budget |
| `did_command_fail(run_result)` | Returns `True` if the command returned a non-zero exit code |

**Parsing**

| Function | Description |
|---|---|
| `parse_metrics(run_log_path=None)` | Extracts `val_bpb` and `peak_vram_mb` from `run.log`. Returns `(float|None, int|None)` |
| `extract_error_tail(log_text, n_lines=20)` | Returns the last `n` lines of a log string for crash debugging |

**Logging**

| Function | Description |
|---|---|
| `ensure_results_tsv()` | Creates `results.tsv` with the correct header if missing |
| `append_result(commit, val_bpb, peak_vram_mb, status, description)` | Appends one tab-separated row per experiment |
| `read_results_history()` | Loads `results.tsv` into a list of dicts |

**Decision**

| Function | Description |
|---|---|
| `decide_result_status(val_bpb, peak_vram_mb, best_val_bpb, best_peak_vram_mb)` | Returns `keep` or `discard` |

**Decision rules:**

| Status | Condition |
|---|---|
| `keep` | `val_bpb` is strictly lower than the current best, OR equal `val_bpb` with lower peak VRAM |
| `discard` | `val_bpb` is equal or worse (and no VRAM improvement) |
| `crash` | Metrics are missing or the run timed out (determined in `main.py`, not by this function) |
| `edit_failed` | Search/replace edits could not be applied even after retry (determined in `main.py`) |

**Results file:** `results.tsv`

```
commit    val_bpb    peak_vram_mb    status       description
a3f9c12   1.423000   12698           keep         Increased EMBEDDING_LR 0.6→0.8
b71d3a8   1.451000   12698           discard      Increased DEPTH 8→10
c90f22e   —          —               crash        Changed activation to GELU
d12e4f0   —          —               edit_failed  Search string not found in train.py
```

---

## RL Pipeline

Adds RL training (TTT-Discover style) to the scaffold. The LLM (Qwen3.5-9B) is loaded locally with LoRA, proposes experiments, and its weights are updated via policy gradient after each batch. The frozen pipeline files are reused but not modified.

### RL Execution Flow

```
1. Load model with PEFT LoRA on model GPU
2. Run baseline (reuses frozen pipeline's run_experiment)
3. Initialize PUCT tree with baseline state
4. For each step:
   a. PUCT selects parent state to expand
   b. Generate batch_size proposals (local model, with logprobs)
   c. Apply edits, run train.py (sequential or parallel via Ray)
   d. Compute rewards: -val_bpb for success, -1.0 for crash/edit_failed
   e. Compute entropic LOO advantages with adaptive beta
   f. Policy gradient update with per-token KL penalty
   g. Save PUCT tree + LoRA checkpoint
```

### Module Reference

#### `rl_main.py` — RL Entry Point

Supports two modes:
- **Sequential** (default): model generates + train.py runs on same GPU
- **Parallel**: model on GPU 0, train.py eval on GPUs 1-7 via Ray workers

| Function | Description |
|---|---|
| `main()` | CLI parsing, model loading, baseline, main RL loop |
| `run_baseline(repo_path)` | Run train.py once, return `(val_bpb, output_text)` |
| `run_single_rollout(...)` | Generate proposal, apply edits, run train.py, return `(rollout, child_state)` |
| `generate_and_apply(...)` | Generate proposal + apply edits only (no train.py). Used in parallel mode. |

#### `rl_model.py` — Model Loading & Generation

Single in-process model for both generation and training. LoRA weights are always up-to-date after each optimizer step.

| Function | Description |
|---|---|
| `load_model(model_dir, device, lora_rank, lora_alpha, lora_path, attn_impl)` | Load base model + LoRA adapter. Returns `(model, tokenizer)` |
| `generate_with_logprobs(model, tokenizer, prompt, max_new_tokens, temperature)` | Generate response + collect per-token logprobs. Returns `(text, full_ids, logprobs, prompt_len)` |
| `compute_response_logprobs(model, full_ids, prompt_len, temperature)` | Forward pass with grad for response tokens (KV-cache split: prompt no-grad, response with-grad) |
| `compute_base_logprobs(model, full_ids, prompt_len, temperature)` | Forward pass with LoRA adapters disabled to get base model logprobs |

#### `rl_planner.py` — Local Generation Wrapper

Wraps `planner.py` for RL mode: uses `generate_with_logprobs` instead of the OpenAI API, strips `<think>...</think>` tags from Qwen3.5 responses.

| Function | Description |
|---|---|
| `propose_experiment_rl(model, tokenizer, agent_state, temperature, max_new_tokens, error_context)` | Generate a proposal locally, return `(proposal, rollout)`. Raises on parse failure. |

#### `rl_trainer.py` — RL Training

| Function | Description |
|---|---|
| `compute_reward(val_bpb, status)` | `-val_bpb` for success, `-1.0` for crash/edit_failed |
| `compute_entropic_advantages(rewards)` | LOO entropic advantages with adaptive beta (binary search for beta s.t. KL ~ log(2)) |
| `train_step(model, optimizer, rollouts, advantages, kl_coef, temperature, max_grad_norm)` | One policy gradient step: importance-weighted loss with per-token KL penalty, gradient accumulation across rollouts |

#### `rl_sampler.py` — PUCT Tree Search

| Class | Description |
|---|---|
| `State` | Node in the search tree. `value = -val_bpb` (higher is better). Stores full `code` string (not git commits). |
| `PUCTSampler` | PUCT tree with rank-based priors. `score(i) = Q(i) + c * scale * P(i) * sqrt(1+T) / (1+n[i])` |

PUCTSampler methods:

| Method | Description |
|---|---|
| `sample_state()` | Pick the highest-PUCT-score state to expand |
| `update_state(child, parent)` | Add child to tree, update visit counts and best-reachable values |
| `record_failed_rollout(parent)` | Increment visits without adding a child |
| `best_state()` | Return the state with the highest value |
| `save(step)` / `load(step)` | Persist/restore PUCT tree as JSON |

#### `rl_eval.py` — Parallel Evaluation (Ray)

| Class | Description |
|---|---|
| `EvalWorker` | Ray remote actor. Owns an isolated repo copy + GPU. Receives edited code, runs `uv run train.py` with `CUDA_VISIBLE_DEVICES`, parses metrics, resets code. |

#### `rl_types.py` — Data Structures

| Field | Type | Description |
|---|---|
| `prompt_text` | `str` | Full prompt after chat template |
| `proposal_text` | `str` | Raw LLM response |
| `full_ids` | `torch.Tensor` | Full token sequence (prompt + response), 1D |
| `old_logprobs` | `torch.Tensor` | Per-token logprobs from generation, 1D |
| `prompt_len` | `int` | Number of prompt tokens |
| `val_bpb` | `float \| None` | Validation bits-per-byte |
| `status` | `str` | `keep` / `crash` / `edit_failed` / `pending` |
| `reward` | `float` | Computed reward |
| `description` | `str` | Experiment description |

### Rollout Flow

```
1. Parse failure (no </think>, bad JSON, exhausted max_new_tokens)
   → full_ids empty → RETRY (no training signal)

2. Proposal parses but edits fail (search string not found)
   → full_ids has data, status="edit_failed", reward=-1.0
   → counts as valid rollout (useful RL signal, not retried)

3. Edits apply but train.py crashes/times out
   → status="crash", reward=-1.0

4. Edits apply and train.py succeeds
   → status="keep", reward=-val_bpb
   → child State added to PUCT tree

5. train.py always resets to parent code after every rollout
```

---

## GPU Memory Limiter

`LD_PRELOAD` library to enforce per-process GPU memory limits. Needed for running 2 train.py workers per B200 GPU (each uses ~74GB, B200 has 180GB).

Intercepts CUDA runtime API calls (`cudaMalloc`, `cudaFree`, `cudaMemGetInfo`, and async variants) to cap each process at `GPU_MEM_LIMIT_MB` megabytes.

```bash
GPU_MEM_LIMIT_MB=88000 LD_PRELOAD=./libgpumemlimit.so python train.py
```

Tracks allocations in a fixed-size array (64K entries). Thread-safe via `pthread_mutex`. Fakes `cudaMemGetInfo` to report the limit as total memory.

---

## Internal Data Objects

All data objects are plain dicts (frozen pipeline) or dataclasses (RL pipeline). No formal class hierarchy.

### ExperimentProposal (dict)

| Field | Type | Description |
|---|---|
| `description` | `str` | Human-readable summary of the change |
| `rationale` | `str` | Why this change was proposed |
| `risk` | `str` | `low`, `medium`, or `high` |
| `edits` | `list[dict]` | Ordered search/replace operations, each with `search` and `replace` keys |

### AgentState (dict)

| Field | Type | Description |
|---|---|
| `repo_path` | `str` | Absolute path to the repo |
| `branch_name` | `str` | Active experiment branch |
| `run_tag` | `str` | Identifier for this run series |
| `best_commit` | `str \| None` | Short hash of the best run so far |
| `best_val_bpb` | `float` | Best validation metric seen (`inf` initially) |
| `best_peak_vram_mb` | `int \| None` | Peak VRAM of the best run |
| `experiment_count` | `int` | Total experiments attempted |
| `max_experiments` | `int` | Stop after this many iterations (0 = unlimited) |
| `llm_base_url` | `str` | OpenAI-compatible endpoint (local or remote) |
| `llm_model` | `str` | Model name to request from the endpoint |

### State (class, `rl_sampler.py`)

| Field | Type | Description |
|---|---|
| `id` | `str` | UUID |
| `timestep` | `int` | Step when this state was created |
| `code` | `str` | Full `train.py` content |
| `value` | `float \| None` | `-val_bpb` (higher is better for PUCT) |
| `parent_values` | `list[float]` | Ancestor values for lineage tracking |
| `parents` | `list[dict]` | Ancestor `{id, timestep}` pairs |
| `observation` | `str` | Training output text |

### Rollout (dataclass, `rl_types.py`)

See `rl_types.py` section above.

---

## Summary

| File | Owns |
|---|---|
| `main.py` | Control flow, setup, loop, recovery |
| `state.py` | All I/O — state, git, files |
| `planner.py` | Context, LLM call, editing |
| `results.py` | Execution, parsing, logging, decision |
| `rl_pipeline/rl_main.py` | RL orchestration, sequential + parallel modes |
| `rl_pipeline/rl_model.py` | Model loading, generation with logprobs, training forward pass |
| `rl_pipeline/rl_planner.py` | Local generation wrapper for RL mode |
| `rl_pipeline/rl_trainer.py` | Reward, advantages, policy gradient step |
| `rl_pipeline/rl_sampler.py` | PUCT tree search over code states |
| `rl_pipeline/rl_eval.py` | Ray parallel evaluation workers |
| `gpu_mem_limit/` | Per-process GPU memory capping via LD_PRELOAD |

The LLM decides *what to try*. The harness decides *everything else*. In RL mode, the LLM also *learns from what it tried*.
