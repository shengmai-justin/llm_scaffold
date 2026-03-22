# Plan: Multi-GPU Parallel Evaluation with Ray

## Context

The RL pipeline currently runs rollouts sequentially: each train.py takes ~5 min, so batch_size=7 takes ~35 min per step. With 8 GPUs (1 model + 7 eval workers), we can run all 7 train.py evaluations in parallel (~5 min total), cutting step time from ~42 min to ~12 min.

We use Ray (not raw subprocess) because:
- **Proven pattern**: ttt_autoresearch already works with Ray `EvalWorker` — direct port
- **Fault isolation**: crashed train.py doesn't affect model or other workers
- **Gen/eval overlap**: generate next proposal while previous eval runs
- **Future scaling**: easy to extend to multi-node

## Current vs Proposed Flow

### Current (sequential, ~42 min per step with batch_size=7)
```
for g in batch_size:
  generate proposal (~1 min)  →  run train.py (~5 min)  →  next
```

### Proposed (parallel with overlap, ~12 min per step with batch_size=7)
```
Gen 0 → dispatch to worker 0 (GPU 1)     ─┐
Gen 1 → dispatch to worker 1 (GPU 2)      │ train.py runs in parallel
Gen 2 → dispatch to worker 2 (GPU 3)      │ on 7 GPUs while model
Gen 3 → dispatch to worker 3 (GPU 4)      │ generates next proposals
Gen 4 → dispatch to worker 4 (GPU 5)      │
Gen 5 → dispatch to worker 5 (GPU 6)      │
Gen 6 → dispatch to worker 6 (GPU 7)     ─┘
Collect all results (~5 min wall time)
RL training step (~30s)
```

## GPU Assignment

```
GPU 0: Model (generation + RL training)    ~20GB
GPU 1: eval_worker_0 (train.py)            ~45GB
GPU 2: eval_worker_1 (train.py)            ~45GB
GPU 3: eval_worker_2 (train.py)            ~45GB
GPU 4: eval_worker_3 (train.py)            ~45GB
GPU 5: eval_worker_4 (train.py)            ~45GB
GPU 6: eval_worker_5 (train.py)            ~45GB
GPU 7: eval_worker_6 (train.py)            ~45GB
```

## Files to Create / Modify

### New: `rl_pipeline/rl_eval.py` — Ray eval worker

Port from `ttt_autoresearch/env.py` + `ttt_autoresearch/train.py:EvalWorker`.

```python
@ray.remote
class EvalWorker:
    """Each worker owns an isolated repo copy and a GPU."""

    def __init__(self, gpu_id: int, base_repo: str, worker_id: int):
        self.gpu_id = gpu_id
        self.repo_path = create_worker_repo(base_repo, worker_id)

    def evaluate(self, parent_code: str, edited_code: str, step: int) -> dict:
        """Write code, run train.py, parse metrics, reset."""
        train_path = os.path.join(self.repo_path, "train.py")

        # Write edited code
        Path(train_path).write_text(edited_code)

        # Run train.py on assigned GPU
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(self.gpu_id)
        r = subprocess.run(
            ["uv", "run", "train.py"],
            cwd=self.repo_path,
            capture_output=True, text=True,
            timeout=600, env=env,
        )

        # Parse metrics from output
        ...

        # Reset train.py
        Path(train_path).write_text(parent_code)
        return {"val_bpb": ..., "peak_vram_mb": ..., "output": ..., "success": ...}


def create_worker_repo(base_repo: str, worker_id: int) -> str:
    """Create isolated repo copy for a worker."""
    worker_dir = os.path.join(os.path.dirname(base_repo), f"eval_worker_{worker_id}")
    if not os.path.exists(worker_dir):
        shutil.copytree(base_repo, worker_dir)
    return worker_dir
```

Key differences from ttt_autoresearch:
- No State serialization needed (we pass code strings directly)
- Parse metrics inline (reuse `results.parse_metrics` logic, but from captured output not shared log file)
- Each worker has its own repo copy (same as ttt_autoresearch)

### Modify: `rl_pipeline/rl_main.py` — Parallel main loop

**New CLI args:**
```python
parser.add_argument("--model-gpu", type=int, default=0)
parser.add_argument("--eval-gpus", type=str, default="1,2,3,4,5,6,7",
                    help="Comma-separated GPU IDs for eval workers")
parser.add_argument("--overlap", action="store_true", default=True,
                    help="Overlap generation with evaluation")
parser.add_argument("--no-overlap", dest="overlap", action="store_false")
```

**Startup:**
```python
ray.init(ignore_reinit_error=True)
eval_gpu_ids = [int(g) for g in args.eval_gpus.split(",")]
workers = [
    EvalWorker.remote(gpu, repo_path, i)
    for i, gpu in enumerate(eval_gpu_ids)
]
```

**Restructured main loop:**
```python
for step in range(args.num_steps):
    parent = sampler.sample_state()
    rollouts = []
    eval_refs = []

    # Phase 1: Generate proposals + dispatch evals
    for g in range(args.batch_size):
        # Generate on model GPU
        proposal, rollout = propose_experiment_rl(...)
        if proposal valid and edits apply:
            edited_code = apply_edits(...)
            # Dispatch eval to Ray worker (non-blocking)
            worker = workers[g % len(workers)]
            ref = worker.evaluate.remote(parent.code, edited_code, step)
            eval_refs.append((ref, rollout, edited_code))
        else:
            # Failed proposal — no eval needed
            rollouts.append(rollout)

    # Phase 2: Collect eval results
    for ref, rollout, edited_code in eval_refs:
        result = ray.get(ref)
        rollout.val_bpb = result["val_bpb"]
        rollout.status = "keep" if result["success"] else "crash"
        rollout.reward = compute_reward(rollout.val_bpb, rollout.status)
        # Update PUCT, log results
        ...
        rollouts.append(rollout)

    # Phase 3: RL training step
    advantages = compute_entropic_advantages([r.reward for r in valid_rollouts])
    train_step(model, optimizer, valid_rollouts, advantages, ...)
```

**Gen/eval overlap**: With `overlap=True` (default), generation of proposal g+1 happens while eval of proposal g is running on its worker. The model GPU is generating while eval GPUs are training. Without overlap, we `ray.get(ref)` immediately after dispatch.

### Modify: `rl_pipeline/run_rl.sh` — Multi-GPU SLURM

```bash
#SBATCH --gpus=8

MODEL_GPU=0
EVAL_GPUS="1,2,3,4,5,6,7"
BATCH_SIZE=7  # match number of eval GPUs

pip install "ray[default]>=2.44.0" --quiet

python rl_main.py \
    --model-gpu "$MODEL_GPU" \
    --eval-gpus "$EVAL_GPUS" \
    --batch-size "$BATCH_SIZE" \
    --model-dir "$MODEL" \
    ...
```

### No changes needed:
- `rl_model.py` — unchanged, model stays on model GPU
- `rl_trainer.py` — unchanged, RL training on model GPU
- `rl_sampler.py` — unchanged
- `rl_planner.py` — unchanged
- `rl_types.py` — unchanged
- Frozen pipeline files — unchanged

## Worker Repo Layout

```
llm_scaffold/
├── autoresearch_rl/          ← base repo (used by rl_main for baseline)
├── eval_worker_0/            ← worker 0 repo copy (GPU 1)
├── eval_worker_1/            ← worker 1 repo copy (GPU 2)
├── ...
└── eval_worker_6/            ← worker 6 repo copy (GPU 7)
```

Created at startup via `shutil.copytree`. Each worker writes its own train.py independently.

## Dependencies

Add to `run_rl.sh`:
```bash
pip install "ray[default]>=2.44.0" --quiet
```

Matches `ttt_autoresearch/pyproject.toml` which pins `ray[default]>=2.44.0`.

## Backward Compatibility

Single-GPU mode still works:
```bash
python rl_main.py --model-gpu 0 --eval-gpus 0 --batch-size 2
```
When `eval_gpus == [model_gpu]`, falls back to sequential execution (no Ray workers, uses current `run_single_rollout` path).

## Implementation Steps

1. Create `rl_pipeline/rl_eval.py` (EvalWorker + create_worker_repo + metric parsing)
2. Add `--model-gpu`, `--eval-gpus`, `--overlap` args to `rl_main.py`
3. Add Ray init + worker creation to `rl_main.py` startup
4. Restructure main loop for generate → dispatch → collect → train pattern
5. Keep single-GPU fallback path
6. Update `run_rl.sh` for 8-GPU SLURM config
7. Test: single-GPU (backward compat), multi-GPU (parallel evals)

## Verification

1. **Single GPU**: `--eval-gpus 0` — same behavior as current sequential pipeline
2. **8 GPUs**: `--model-gpu 0 --eval-gpus 1,2,3,4,5,6,7 --batch-size 7`
   - Check `nvidia-smi`: model on GPU 0, train.py on GPUs 1-7
   - All 7 evals start within seconds of each other (check timestamps)
   - Step time ~12 min (vs ~42 min sequential)
   - RL update uses all 7 rollouts correctly
3. **Fault tolerance**: kill one train.py mid-run — other workers + model unaffected
4. **Overlap**: with `--overlap`, generation time overlaps eval time

## Reference Code

| Component | Source |
|-----------|--------|
| EvalWorker | `ttt_autoresearch/train.py:67-89` |
| create_worker_repo | `ttt_autoresearch/env.py:177-183` |
| run_training with GPU | `ttt_autoresearch/env.py:125-156` |
| Ray dispatch + collect | `ttt_autoresearch/train.py:253-326` |
| Overlap pattern | `ttt_autoresearch/train.py:280-281` |
