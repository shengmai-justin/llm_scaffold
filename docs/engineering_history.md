# Engineering History

Dated notes on infrastructure setup, bug fixes, and design decisions. Historical context — read on demand, not needed for day-to-day work.

## Pro 6000 one-time setup (2026-04-10)

The Pro 6000 box ships only the NVIDIA driver — **no CUDA toolkit, no `python3-dev` headers, no `module load`**. Setting up the scaffold requires:

1. **Local CUDA toolkit install (no sudo needed):**
   ```bash
   wget https://developer.download.nvidia.com/compute/cuda/12.8.0/local_installers/cuda_12.8.0_570.86.10_linux.run
   sh cuda_12.8.0_570.86.10_linux.run --silent --toolkit --toolkitpath=$HOME/cuda-12.8 --no-opengl-libs --override
   # Add to ~/.bashrc:
   export CUDA_HOME=$HOME/cuda-12.8
   export PATH=$CUDA_HOME/bin:$PATH
   export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
   ```
   Without this, `deep_gemm` (transitively imported by sglang) asserts on a missing CUDA_HOME at import time and SGLang dies before loading any model.

2. **`run_pro6000.sh` auto-exports the triton gcc JIT linker paths:**
   - `LIBRARY_PATH=$CUDA_HOME/lib64/stubs` (so `gcc -lcuda` finds the unversioned stub — the driver only ships `libcuda.so.1`)
   - `CPATH=$CUDA_HOME/include` (so gcc finds `cuda.h`)

3. **uv-managed Python (required, not optional):** System Python 3.10 on itml-1 has no dev headers (`Python.h` missing), and triton JIT-compiles a C extension that needs them. `run_pro6000.sh` sets `UV_PYTHON_PREFERENCE=only-managed` and calls `uv python install 3.10` to pull in a python-build-standalone distribution with headers baked in. The script also auto-wipes `autoresearch_frozen/.venv/` if it was built against system Python (detects via `readlink -f .venv/bin/python`).

4. **`flash-attn-4` is pulled in transitively** by `sglang==0.5.10.post1`'s metadata, so `uv pip install` in the Pro 6000 script passes `--prerelease=allow`. It installs but is never loaded at runtime — `--attention-backend triton` keeps SGLang off the FA4 codepath entirely.

## ERL worker venv sharing (2026-04-11)

`rl_eval.py:create_worker_repo` used to call `shutil.copytree(base_repo, worker_dir)`, which cloned the full ~15 GB venv per worker — devastating on `/blue/buyuheng`'s 4 TB group quota. It now copies the repo **without** `.venv/` (via `shutil.ignore_patterns(".venv", ...)`) and symlinks the worker's `.venv` at the parent's. Disk cost drops from `~15 GB × N` to `~20 MB × N`. Workers are still fully isolated for parallel `train.py` writes; the venv is read-only during training so the symlink is safe.

**Cleanup order matters:** `erl_pipeline/clean.sh` deletes worker dirs *before* the source, so dangling symlinks never cause a real venv to be traversed. Don't `rm -rf autoresearch_erl/` while workers still exist.

## TTT vs GRPO namespacing

`--adv-type ttt` via `run_erl_4gpu_ttt.sh` uses `ERL_REPO=autoresearch_erl_ttt` (not `autoresearch_erl`) so TTT and GRPO runs don't share the same working tree or worker dirs. The log dir is also namespaced (`erl_log_ttt` vs `erl_log`). `clean_ttt.sh` is a separate script because `clean.sh` hard-codes the GRPO paths. If you add new `--adv-type` variants, follow the same pattern in the run script and add a matching clean script.

## Multi-GPU model sharding (2026-04-13)

`rl_model.py:load_model` accepts `model_gpus=[0,1,...]` to shard the model across multiple GPUs via HuggingFace `device_map="auto"` + `max_memory`. All B200 memlimit scripts and the Pro 6000 ERL script use this to halve (or third) per-GPU activation memory during training. The CLI flag is `--model-gpus "0,1"` (plural, comma-separated). The old `--model-gpu 0` still works for the single-GPU path. `model.input_device` (set at load time) replaces `model.device` everywhere — with `device_map` there's no single `.device`, so all tensor placement goes through `model.input_device` (the device where the embedding layer lives).

## Memory optimizations in the trainer (2026-04-13)

Two changes in `erl_trainer.py` / `rl_trainer.py` / `rl_model.py`:

1. **Reordered `base_lp` before `new_lp`** in the GRPO loss. Base model's forward (no-grad) runs first and its KV cache is freed before the grad-tracked `new_lp` forward builds its activation graph. Peak memory drop: ~15-20 GB per rollout.
2. **Selective log-softmax** in `_response_logprobs`. Replaced `log_softmax(logits, dim=-1).gather(...)` with `target_logit - logsumexp(logits)`, avoiding a `(seq_len, vocab_size)` materialization. Saves ~4.6 GB per forward for Qwen3.5-9B at 16k tokens.

Both changes are mathematically identical to the original code — same values, same gradients.

## Allocation-hold pattern (2026-04-13)

All ERL run scripts end with `echo "Python exited..." && sleep infinity` instead of exiting cleanly. This keeps the SLURM allocation alive if the python process dies (OOM, kill, crash), so you can SSH in and restart without queue wait. Important: `srun --overlap --pty bash` does NOT work for this — the overlap job step is tied to the main script's lifecycle and dies when it exits. Use plain `ssh <node>` instead, which is independent of SLURM job steps.

## Reflection context for attempt2 (2026-04-12)

`erl_reflect.py:build_reflection_context` used to include both `batch_feedback` (raw attempt1 summaries) AND the model's reflection text in the attempt2 prompt. The reflection was generated FROM `batch_feedback`, so including both was redundant and bloated the prompt. Now only the reflection text is passed through. Also removed the "Previous edit attempt failed:" / "propose a corrected experiment" framing that `rl_planner.py` used to wrap the `error_context` — that framing confused the model into thinking attempt2 was a retry of a failed error, causing frequent non-JSON output. Attempt2 JSON failure rate dropped but didn't go to zero — see the think-budget fix below (2026-04-15).

## Think-budget enforcement + history summarizer + reward consistency (2026-04-15)

Three coordinated changes that fixed the top pathologies from the 2026-04-14 B200 runs (26 % JSON-parse rate, 0 keeps in dead-end categories like warmup/weight_decay despite 27+ re-attempts, reflection baseline skewed by hardcoded `-1.0` edit_failed reward).

1. **`rl_pipeline/budget_processor.py` — `BudgetThinkingProcessor`.** A `LogitsProcessor` attached to `model.generate` when `generate_with_logprobs(..., think_budget=N)` is passed. Soft-boosts the `</think>` logit at 95 % of budget, hard-forces it at 100 %. `compute_response_logprobs` is NOT run with the processor, so `old_lp` and `new_lp` stay on-policy (ratio ≈ 1.0). GRPO gradient flows through the natural logprob of the forced `</think>`, gently pressuring the policy to self-close earlier. Default: `--think-budget 6000` in `erl_main.py`, `--max-new-tokens` reduced 32768 → 16000 to match.

2. **`erl_pipeline/erl_history.py` — `generate_history_summary`.** One extra LLM call per step before Phase 1. Reads the last 300 rows of `results.tsv`, asks the model (low temperature 0.3) to produce a markdown table: dead-end categories (5+ attempts, 0 keeps) with "DO NOT propose" instruction + worked-direction bullets. No hardcoded category buckets — the LLM chooses the granularity. Injected via new `history_context` parameter on `propose_experiment_rl`, passed to BOTH attempt1 and attempt2 prompts. Separate from `erl_reflect.py`: reflection is tactical (per-batch), history is strategic (cross-step).

3. **`edit_failed` reward consistency.** `erl_main.py` previously hardcoded `reward=-1.0` in the `Rollout` constructor for JSON-parse failures, while every other `edit_failed` path used `compute_reward(None, "edit_failed") = 0.0`. The rollout itself was filtered from GRPO (empty `full_ids`), but it was included in `all_a1_rewards` for reflection's baseline, artificially inflating `ref_adv`. Now uses `compute_reward` for consistency.

**Observed impact (2026-04-15 Pro 6000 run, 25/100 steps in):** JSON-parse failures 63 → 15 (4.2× fewer), wasted steps 1/30 → 0/25, distillation rate 20 % → 50 %, keep rate 3.75 % → 11.5 %. Proposal volume for yesterday's dead-end categories dropped 48–78 % (weight_decay 34→12, warmup 27→14, depth 23→5). See `results/2026-04-15_erl_pro6000.md` for full analysis.

## Pro 6000 8-GPU layout (2026-04-15)

`run_erl_pro6000.sh` upgraded from 6 GPUs (3 model + 3 eval) to 8 GPUs (4 model + 4 eval), `BATCH_SIZE 3→4`, `NUM_STEPS 50→100`. Per-model-GPU memory drops from ~12 GB (9B/3) to ~9 GB (9B/4). Batch size matched to eval worker count so every attempt runs on its own worker with no queueing. Also `RAY_TMPDIR=/tmp/raye_$USER` (was `$PROJ_DIR/ray_tmp_erl`) to fit under AF_UNIX 107-byte socket-path limit.

## Results logging order (2026-04-12)

`erl_main.py` used to interleave `[attempt1] episode 0, [attempt2] episode 0, [attempt1] episode 1, ...` in `results.tsv`. Now logs all attempt1s first, then all attempt2s, matching the phased execution order. Easier to read chronologically.

## Phase 1 robustness pass — parse preservation, Qwen sampling, think re-opening block (2026-04-16)

Six coordinated changes addressing the 11.2 % JSON-parse rate and bottom-quartile keep rate observed in the 2026-04-16 Pro 6000 run (see `results/2026-04-16_erl_pro6000.md`). Each change targets a different failure path:

1. **Parse-failure `full_ids` preservation** (`rl_planner.py:propose_experiment_rl`). Previously, JSON-parse failures raised from `rl_planner`, caught in `erl_main.py`, and a fresh `Rollout` with `full_ids=torch.tensor([])` was created — discarding the actual generation tokens. Now `propose_experiment_rl` builds the rollout **before** parsing and returns `(None, rollout_with_full_ids)` on parse/validation failure. Callers (`erl_main.py`, `rl_main.py`) branch on `proposal is None` and keep the rollout in the batch with `reward = compute_reward(None, "edit_failed") = 0.0`. **Critical consequence for TTT:** K no longer drops to 2 when two parse failures coincide, so the `compute_entropic_advantages` corner case at K=2 (target `log(2)` = max-achievable KL, β escapes to `beta_max` → `w=1e12` → inf loss) is no longer reachable from parse failures.

2. **Brace-match JSON extractor** (`rl_planner.py:_extract_json`). Two-stage: first strip `<think>...</think>` + markdown fences and try `json.loads`; on failure, walk the raw text backward to find the last balanced top-level `{...}` block and try parsing that. Salvages (a) "prose then JSON" outputs and (b) "JSON then trailing prose" outputs. Does NOT save empty outputs (model ran out of tokens still in think) or mid-string truncation — those are still unrecoverable but at least don't drop K thanks to fix #1.

3. **`<think>` re-opening block after forced close** (`budget_processor.py`). The existing `BudgetThinkingProcessor` hard-forces `</think>` at budget exhaustion, then went no-op. Observed: the model frequently generated a second `<think>` block, consumed the remaining `max_new_tokens`, and emitted no JSON. `_resolve_single_token_id(tokenizer, "<think>")` now caches the opening-tag token id in `__init__`, and the processor masks it (`-inf`) whenever `self.closed = True`. Applies both on the step that detects the close (natural or forced) and every subsequent step. If `<think>` doesn't tokenize to one token (unlikely for Qwen3), `start_think_id = None` and we silently skip the block. On-policy invariant preserved: `compute_response_logprobs` runs without any processor, so `old_lp` / `new_lp` see the natural distribution and the blocked token never appears in either.

4. **Qwen3.5-9B sampling defaults** (`rl_model.py:generate_with_logprobs`). Added `top_k=20, top_p=0.95` arguments with defaults matching Qwen team's precise-coding profile (from the official HF model card). Previously: `top_k` inherited from `generation_config.json` (often 50), `top_p=1.0`. Presence-penalty kept at 0 by user decision (the paired value in the precise-coding profile). Tighter distribution → fewer pathological long-tail continuations. No CLI plumbing required; all callers pick up new defaults automatically.

5. **Short-search-string rule** (`prompt.md`). Added rule: *"Use the shortest unique search string (1-5 lines). Do NOT quote entire functions."* Targets the root cause of 67 % of empty-output parse failures — the model quotes multi-dozen-line code blocks verbatim in the `search` field, bloats JSON past the remaining-budget ceiling, and truncates. Prompt-level fix is cheaper than any decoding hack.

6. **Reflection budget bump** (`erl_reflect.py:generate_batch_reflection` + `erl_main.py`). Defaults lifted `max_new_tokens 1024 → 2048, think_budget 512 → 1024` to handle richer `batch_feedback` at batch_size=4 (feedback alone often >500 tokens of attempt traces). Cap in `erl_main.py:414` also lifted `min(think_budget, 512) → min(think_budget, 1024)`. 50/50 think-vs-visible ratio preserved. Wall-clock cost ≈ +25 s per step (≈ +40 min on a 100-step run).

**TTT still has a latent root-cause bug** in `compute_entropic_advantages`: K=2 with distinct rewards, or K≥3 with near-identical rewards, pushes β to `beta_max=1e6` and overflows. Not patched pending empirical evidence that fix #1 eliminates the path to K=2 in practice. One-line mitigation (`beta = min(float(beta), 20.0)` after binary search) is ready but unshipped.

## Ensue swarm analysis — community benchmark (2026-04-16)

Pulled the full public `@autoresearch-at-home/` namespace via Ensue's public JSON-RPC endpoint (`https://api.ensue-network.ai/public/autoresearch-at-home`, no auth). 3,206 experiments from 56 agents. Full analysis in `results/2026-04-16_ensue_swarm_analysis.md`. Key findings relevant to our scaffold:

- **Network HP:structural split is 37.9 : 31.6 %** — near parity, not HP-dominated as we initially hypothesized from `collab.md` examples. HP bias in our ERL is search-strategy/maturation, not protocol-level.
- **Network keep-rates by category essentially tie** (HP 17.1 %, STRUCT 17.5 %). Neither direction is inherently higher-yield.
- **MLP width is criminally underexplored** (9 runs, 44.4 % keep-rate — highest in the whole swarm). Our ERL has never proposed a `hidden_dim` change.
- **Our 5.7 % keep-rate is bottom-quartile.** Top agents (herman, m5max, phoenix, cipher) land 22-28 %. Worst tier (spectre, janus, sparrow) lands 0-4 %.
- **Our HP proposals tune the wrong knobs.** Network winners: `final_lr` (22.0 %), `embed_lr` (25.0 %), `grad_accum` (25.0 %), `warmdown` (19.2 %). Our ERL over-indexes on `warmup`, `weight_decay`, generic `LR`.

The raw data dump lives at `/tmp/ens_all_results.json` (~2 MB, schema documented in the analysis file). Can be pulled into our history summarizer as an oracle corpus if we want to seed the model with swarm-derived structural exemplars.

## Deviations from original ERL paper (arXiv 2602.13949) — 2026-04-19

Audit of our pipeline vs Microsoft ERL reference.

### Shared reflection (intentional divergence)

Paper generates **per-rollout reflection** gated by `r(1) < τ=1` — one reflection per failed attempt1, matching attempt2. We generate **one shared reflection per batch** from aggregated attempt1 feedback, used by all K attempt2 prompts.

Rationale: paper's tasks (Sokoban, FrozenLake, HotpotQA) batch *different* problem instances — each rollout attacks its own puzzle, so a shared reflection would average across unrelated failures. Our task batches K attempts at the *same* train.py at the *same* step, so failures are correlated (policy drift → all 4 propose similar weight_decay variants). Shared meta-reflection surfaces cross-rollout patterns ("all 4 tried HP tuning — try structural change") that per-rollout reflection cannot. Keeping shared; K× cheaper and better signal for our problem shape.

### Other deviations

Revised after reading `microsoft/experiential_rl/train_scripts/train_erl_sokoban.sh`:

- **Loss aggregation:** paper uses `seq-mean-token-sum` (sum tokens within rollout, mean across rollouts). Ours is `seq-sum-token-mean` (opposite on both axes).
- **KL coefficient:** paper 0.001, ours 0.1 (100×).
- **Learning rate:** paper 1e-6, ours 4e-5 (40×). Our LR-to-aggregation ratio roughly offsets the paper's, so effective step size is similar.
- **Clip ratio:** paper uses DAPO-style asymmetric upper 0.28; ours had no clipping.
- **Max tokens:** paper 10240, ours 16000 (long-CoT task justifies ours; paper-parity script caps at 10240).
- **Distributed strategy:** paper uses FSDP + CPU offload (full-param training). Ours uses `device_map="auto"` pipeline parallel. Not comparable because our LoRA economics favor DDP, not FSDP.
- **Reflection always-on vs gated:** paper skips reflection when attempt1 succeeded (`r(1) ≥ τ`). We always generate. Wastes compute when attempt1 already kept; noise in the reflection loss when there's nothing to reflect on.
- **Distillation gate:** paper `I(r(2) > 0)` binary; ours `val_bpb < step_best_bpb` (domain-appropriate for continuous reward).

### Paper-parity flags landed (2026-04-19)

Additive flags in `erl_main.py` + dispatch in `erl_trainer.py`:
- `--loss-agg-mode {seq-sum-token-mean (default), seq-mean-token-mean, seq-mean-token-sum, seq-mean-token-sum-norm}`
- `--clip-ratio-high <eps>` — PPO-style asymmetric clip (min-trick), disabled by default

New launch script `run_erl_pro6000_paper.sh` opts into paper values (`lr=1e-6`, `kl_coef=0.001`, `loss_agg_mode=seq-mean-token-sum`, `clip_ratio_high=0.28`, `max_new_tokens=10240`). All existing scripts unchanged — defaults preserve prior behavior bit-for-bit.

### DDP infrastructure landed, rollout restructuring deferred (2026-04-19)

In `rl_model.py`: `use_ddp=True` branch wraps with `DistributedDataParallel`; `underlying()` helper strips DDP for `.generate()` / `.save_pretrained()` / `.disable_adapter_layers()` call-sites. Non-reentrant gradient checkpointing enabled.

In `erl_main.py`: `torch.distributed.init_process_group` runs when launched via torchrun (triggered by `LOCAL_RANK` env); rank-0 guards on Ray init and checkpoint save; DDP-aware resume path.

**Not done (deferred):** phase-1/3 rollout loops still monolithic (all-rollouts-per-rank). Needs per-rank slice + reward all_gather + reflection broadcast + Ray coordination + barriers. Scoped at ~300 LOC with many failure modes that need cluster testing.

Next direction instead: `docs/SPLIT_PIPELINE.md` — decompose proposal into ideator (RL-trained) + implementer (frozen), smaller action space, cleaner credit assignment. Much simpler than DDP and orthogonal to other changes.
