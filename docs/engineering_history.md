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

## Split pipeline landed, TTT+split becomes reference config — 2026-04-20

Split-pipeline implementation shipped. Three new functions in `rl_pipeline/rl_planner.py` (`propose_idea`, `implement_idea`, `propose_experiment_split`), two prompt files (`prompt_ideator.md`, `prompt_implementer.md`), `--split-pipeline` flag in `erl_main.py`, four new launch scripts (+ matching clean scripts) fully namespaced so split runs don't collide with monolithic runs.

**Implementation decisions (resolved during the session):**
- Ideator: 8192 max_new_tokens hardcoded (ignores caller's flag), `--think-budget 6000` retained, single shot per rollout (no stage-A retry — retrying would break the one-action-per-rollout RL invariant).
- Implementer: temp 0.7, max 4096 tokens, 3 attempts total (initial + 2 retries at same temp), `enable_thinking=False` via Qwen3 chat template.
- Zero reward on stage-B failure (no −0.1 penalty).
- Free-text ideator output.

**Pro 6000 launch bug + fix.** First non-smoke launch failed at baseline with `uv run train.py` complaining about a broken `.venv` inside `autoresearch_erl_split/`. Root cause: the first `uv run` inside that dir auto-created a `./.venv` with the default Python (no torch), and uv prefers project-local `.venv` over `VIRTUAL_ENV`. Fix landed in `run_erl_pro6000_split.sh` (and its ttt variant): pre-clone `$ERL_REPO` from shell, then `ln -sfn $VENV_DIR $ERL_REPO/.venv` before invoking Python. Self-healing on every launch. The monolithic `run_erl_pro6000.sh` was left unchanged — it works on the cluster because `autoresearch/.venv` is already a working uv-managed venv that `shutil.copytree` duplicates.

**First full runs — split pipeline on B200 (2026-04-20):**

| Config | Steps | Keeps | `edit_failed` | `crash` | Baseline → best | Δ |
|---|---|---|---|---|---|---|
| GRPO + split | 14 | 0 | 3.6% | 8.9% | 0.988775 → 0.988775 | 0 |
| **TTT + split** | **22** | **15 (8.5%)** | **1.7%** | 44.3% | **0.985769 → 0.969023** | **−0.0168** |

TTT+split is the best configuration observed on this scaffold. 10/15 keeps from attempt2 — first direct evidence the reflection → refinement channel produces improvements (0/K in all prior runs). Distillation fired on 9/22 steps. Coherent research arc visible in `results/2026-04-20_erl_ttt_split.md`: depth → batch size → EMBEDDING_LR → SCALAR_LR / WEIGHT_DECAY → VE_GATE_CHANNELS binary search (32→16→8→4) → schedule tuning.

CoT drift re-emerging after step 16 (`num_grpo_tokens` 13K → 42K by step 17, 46K by step 21). Same warning signal that preceded the 2026-04-19 monolithic TTT OOM. Split's 8K ideator cap bounds stage A, but stage A can still drift past 8K via repeated think tokens if the budget processor misses. Worth watching on resumes.

GRPO+split's 0 keeps isn't a pipeline failure — the baseline (0.9888) is the tightest on this scaffold to date, median rollout val_bpb was 1.036, closest miss was 0.988979 (0.02% above baseline). Likely converts to keeps with more wallclock; the 10 h slot only got 14 steps.

## Forge XL-tier leaderboard analysis — 2026-04-23

Pulled Ensue public MCP (`@autoresearch-at-home/best/tier/xl/train_py`) and compared forge's global-best train.py (val_bpb=0.926381, H200 XL tier, WSD 30/70 sqrt + FINAL_LR_FRAC 0.02) against our 2026-04-20 TTT+split best (val_bpb=0.969023). Gap = −0.043 BPB.

Disaggregated by category:

| Gap source | Est. contribution | Notes |
|---|---|---|
| Compiler/kernel pipeline | ~−0.013 (remaining after our baseline FA4) | Inductor flags (`coordinate_descent_tuning`, `epilogue_fusion`, `aggressive_fusion`, `shape_padding`, `max_autotune_pointwise`), `torch.library.custom_op` wrapping FA4 fwd + bwd so compile doesn't graph-break, `max-autotune` mode + CUDA graphs. We have raw FA4 kernel (`from flash_attn.cute import flash_attn_func`) and basic `torch.compile(model)`, capturing ~−0.010 of the full Day-4 stack's −0.021. Remaining is pure engineering port, no RL discovery needed. |
| Architecture | ~−0.012 | All-layer VE (forge: `has_ve=True` always, we: alternating), `ve_gate_channels=64` (we went wrong way: 32→4), `ve_gate` multiplier 4× (we: 2×), QK scale ×1.15 (we: none), skip-2 residual with learnable `skip2_lambdas` (we: absent), softcap 13 (we: 15), short_window = `long_window//16` (we: `//2`), rotary base 50000 (we: 10000), DEPTH 12 (we: 10). |
| HP refinements | ~−0.008 | `TOTAL_BATCH_SIZE` 2^17 (we: 2^18, forge halved again for more steps), `EMBEDDING_LR` 1.2 (we: 0.4 — opposite direction), `WARMDOWN_RATIO` 0.7 / WSD 30/70 sqrt (we: 0.5), `FINAL_LR_FRAC` 0.02 (we: 0.01), `ADAM_BETAS` (0.8, 0.99) (we: (0.8, 0.95)). |
| Init + optimizer internals | ~−0.002 | `wte` std 0.8 (we: 1.0), matrix init √2 (we: √3), `resid_lambdas` init 0.9 (we: 1.0), Muon `ns_steps=7` (we: 5), Muon `beta2=0.90` (we: 0.95), decoupled `value_embeds` LR at 0.3, AdamW weight_decay 0.01 on lm_head. |

**Forge's VRAM budget:** 48.8 GB measured on H200 (out of 178 GB). Comfortably fits our 88 GB B200 per-worker cap with ~40 GB headroom. DEPTH 12 + all-layer VE pays for itself in memory by halving `DEVICE_BATCH_SIZE` to 64 and shrinking short-window attention (`//16` vs `//2`). The package is VRAM-neutral vs our current setup.

**Qwen3.5-9B cannot write the FA4 `custom_op` wrapper from scratch.** Git history on `github.com/mikeapedia/autoresearch-at-home` branch `autoresearch/mar15` shows the wrapper took **7 commits** to land (Claude + human-in-loop): started with FlexAttention (failed), fell back to `torch._dynamo.allow_in_graph` (insufficient fusion), escalated to full `@torch.library.custom_op`, hit a saved-LSE bug, return-type bug, then extended to wrap backward. Even Claude (200B-class) iterated through compile errors to land this. Our Qwen3.5-9B has no runtime feedback channel in ERL and would almost certainly crash-loop on it. Recommendation: **port those ~40 lines manually** rather than wait for RL to re-derive a known recipe.

## Probes directory — 2026-04-23

Added `probes/` — diagnostic tooling that tests whether Qwen3.5-9B knows techniques our ERL has zero coverage on (derived from the Ensue swarm category data). Not part of any pipeline.

- `prompt_knowledge_probe.md` — "implement technique X in this train.py or honestly admit the gap" system prompt (admits-gap is a valid response, not a failure).
- `knowledge_probes.txt` — 18 techniques grouped by category: Muon internals (Newton-Schulz coefficients + step count), optimizer swaps (Sophia / Lion / MuP), stability (SWA / EMA / stochastic depth / QK-norm / tied Q=K), loss shape (z-loss / label smoothing), architecture (2× MLP stacking), schedule (FINAL_LR_FRAC, sqrt warmdown), compilation (torch.compile max-autotune), training-budget knobs (DEVICE_BATCH_SIZE restructuring, knowledge distillation).
- `run_probes.py` — iterates probes against an OpenAI-compatible endpoint, saves raw outputs + `summary.json` classifying each as `OK` / `ADMITTED` / `PARTIAL` / `BAD_JSON`. `--retries N` retries BAD_JSON only (not ADMITTED or PARTIAL — those are genuine signals worth preserving). `OK` = valid JSON + all search strings match train.py; `PARTIAL` = valid JSON but some search strings don't match (knows concept, wrong names); `ADMITTED` = empty `edits` list with honest "I don't know" description.
- `run_probes_pro6000.sh` — one-command launcher mirroring `run_pro6000.sh` serve-then-query pattern (SGLang on port 8000, triton backend, auto-picks 1 free GPU, trap-cleans server on exit).

Does **not** actually execute `train.py` with the proposed edits — measures knowledge + localization only, not correctness. An `OK` classification means edits apply cleanly, not that the resulting train.py runs or converges. Adding `--eval-apply` to route OK outputs through `results.run_experiment` is a ~30 LOC extension if needed.

**Method-coverage baseline (scan of 2,432 ERL rollouts across 8 runs, 2026-04-23):**
- **Genuine zero-coverage Qwen blind spots vs swarm:** MLP stacking (swarm 10 runs, 20% keep; us 0), Newton-Schulz coefficients (part of swarm's 232 muon-internals runs; us 0 on coefficients specifically — we touch MATRIX_LR, ns_steps, Muon momentum, beta2 etc. but never the polynomial 3-tuple), `training_duration` / num_iterations changes (swarm 63 runs, 20.6% keep; us ~0 real hits).
- **Day-5 novel techniques we've never tried:** temporal time-mixing (Ember's causal carry on MLP inputs), QK scale tuning, VE gate warmup, sqrt warmdown, multi-seed variance baselining.
- **Zero-coverage on both sides (swarm backlog):** curriculum learning, quality filtering, domain weighting. Called out in Day-5 blog as "the last major unexplored axis."

## Variance floor awareness — 2026-04-23

Day-5 Ensue blog reported **seed variance ~0.007 BPB on B200** — larger than most single-parameter "improvements" (0.001–0.003). Implication for our TTT+split run: the cumulative Δ=−0.0168 is safely above the floor, but the individual step-by-step keeps (0.973→0.972→0.971) are within noise. The overall research arc is real; the fine-grained "which step was the important one" question is not well-posed under this noise level. If future runs care about per-step attribution, add a multi-seed baseline first (3–5 runs of unchanged baseline) to establish the floor on our exact hardware.
