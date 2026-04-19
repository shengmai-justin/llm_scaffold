# DDP Refactor + ERL Paper Parity — Pro 6000 ERL

**Status (2026-04-19):** Part 2 (paper-parity flags) **landed**. Part 1 (DDP) **deprioritized** — infrastructure in place but rollout-loop restructuring not done; too complex relative to expected gain. Preferred direction for next work: `docs/SPLIT_PIPELINE.md` (ideator + implementer decomposition) + cheap wins (FA2 probe, shorter max_tokens, inter-rollout `empty_cache`).

Two additive changes. Neither breaks existing scripts.

## Problem

Pro 6000 ERL OOMs every ~15-17 steps. Root cause: `device_map="auto"` is designed for inference; at training time it forces sequential layer execution across 4 GPUs with activation pileup on middle-pipeline GPUs (1, 2). Qwen3.5-9B bf16 is ~18 GB and fits easily on one 96 GB card — the right parallelism is DDP, not pipeline.

Separately, our trainer deviates from the ERL paper on several axes (aggregation, LR, KL, clip). Worth making configurable while we're refactoring.

## Goal

1. **DDP path** for Pro 6000 ERL — 4 full model replicas, parallel rollout generation, gradient all-reduce.
2. **ERL-paper-parity flags** — opt-in aggregation mode / clip / LR / KL matching `microsoft/experiential_rl/train_scripts/train_erl_sokoban.sh`.

Both additive. B200 + RL + frozen pipelines untouched.

---

# Part 1: DDP

## Per-GPU memory

Current pipeline: GPU 1 hits 94 GB / 96 GB due to activation pileup across ~9 backward passes per step.

DDP: each GPU holds 18 GB weights + ~30 GB peak activations + ~1 GB LoRA grads/optimizer ≈ **50 GB per GPU**. Comfortable fit, 46 GB headroom. Throughput 4× from parallel rollout generation.

## File changes

### `rl_pipeline/rl_model.py` (+50 LOC)

Add `use_ddp: bool = False, local_rank: int = 0` to `load_model`. New branch:

```python
if use_ddp:
    device = torch.device(f"cuda:{local_rank}")
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, dtype=torch.bfloat16, attn_implementation=attn_impl,
    ).to(device)
    # ... LoRA + gradient_checkpointing_enable(kwargs={"use_reentrant": False}) ...
    model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[local_rank], find_unused_parameters=True,
    )
    model.module.input_device = device
```

Existing `model_gpus` (sharded) and single-GPU paths stay as-is — B200 runs unaffected.

Add helper for `.generate()`, `.save_pretrained()`, `.parameters()` access:

```python
def _underlying(model):
    return model.module if isinstance(model, DDP) else model
```

### `erl_pipeline/erl_main.py` (+150 LOC)

**Distributed init at `main()` start:**

```python
is_distributed = "LOCAL_RANK" in os.environ
if is_distributed:
    torch.distributed.init_process_group("nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = torch.distributed.get_world_size()
    torch.cuda.set_device(local_rank)
else:
    local_rank, world_size = 0, 1
is_rank0 = local_rank == 0
```

**Per-phase changes:**

- **Phase 0 (history):** rank 0 generates, `broadcast_object_list` to others.
- **Phase 1/3 (attempts):** rank `i` generates rollout `i`. Assumes `batch_size == world_size` for v1. Eval dispatched from rank 0 (gather edited_code, scatter results) — simpler than multi-rank Ray client.
- **Phase 2 (reflection):** rank 0 generates, broadcasts `full_ids / old_logprobs / prompt_len` tensors to all ranks. Each rank computes identical reflection loss → DDP averages → equivalent to single-GPU math.
- **Phase 4 (train):** each rank computes local loss on its rollout + reflection + distill (if applicable), `.backward()` triggers all-reduce automatically. One `optimizer.step()` per rank updates local replica.

**Rewards → advantages (group-wide):**

```python
local_reward = torch.tensor([rollout.reward], device=device)
all_rewards = [torch.zeros_like(local_reward) for _ in range(world_size)]
torch.distributed.all_gather(all_rewards, local_reward)
local_adv = compute_attempt_advantages(torch.cat(all_rewards).tolist())[local_rank]
```

**Rank-0-only:** Ray init, `results.tsv` / `rollouts.jsonl` writes, `save_pretrained`, `print`.

**Barriers** at phase boundaries to avoid races on shared state.

### `erl_pipeline/run_erl_pro6000_ddp.sh` (new, +80 LOC)

Clone of `run_erl_pro6000.sh`, replace the python invocation with:

```bash
torchrun --standalone --nproc_per_node=4 --master_port=29500 \
    erl_main.py \
    --eval-gpus "$EVAL_GPUS" \
    --workers-per-gpu 1 \
    ...
```

Drop `--model-gpus`. `LOCAL_RANK` env triggers distributed code path.

## Testing sequence

No local multi-GPU machine — test on cluster:

1. **Sanity:** existing `run_erl_pro6000.sh` unchanged → identical behavior to today.
2. **Smoke:** `torchrun --nproc_per_node=2 erl_main.py --num-steps 1 --batch-size 2 ...` → rank 0/1 each generate 1 rollout, train step completes.
3. **Short run:** `run_erl_pro6000_ddp.sh --num-steps 5` → compare `results.tsv` vs reference.
4. **Memory check:** per-GPU usage should stay ~50 GB at step 30+.

---

# Part 2: ERL paper parity (additive flags)

Based on `microsoft/experiential_rl/train_scripts/train_erl_sokoban.sh`.

| param | current | ERL paper | flag |
|-------|---------|-----------|------|
| `loss_agg_mode` | seq-sum-token-mean (implicit) | `seq-mean-token-sum` | `--loss-agg-mode` |
| `clip_ratio_high` | PPO default (symmetric) | 0.28 (DAPO asymmetric) | `--clip-ratio-high` |
| `kl_coef` | 0.1 | 0.001 | `--kl-coef` (exists) |
| `lr` | 4e-5 | 1e-6 | `--lr` (exists) |
| `max_new_tokens` | 16000 | 10240 | `--max-new-tokens` (exists) |

### File changes

**`erl_pipeline/erl_main.py`:** add two argparse flags
```
--loss-agg-mode  default: "seq-sum-token-mean"   (current behavior)
--clip-ratio-high  default: None                 (disabled)
```

**`erl_pipeline/erl_trainer.py` (+10 LOC):** replace `loss = -(ratio * shaped_adv).mean()` with a dispatch on `loss_agg_mode`. For asymmetric clip, clamp `ratio` upper before loss. Pass `K` (group size) through the per-rollout loop.

```python
adv_weighted = ratio * shaped_adv
if mode == "seq-sum-token-mean":
    loss = -adv_weighted.mean()                    # current
elif mode == "seq-mean-token-mean":
    loss = -adv_weighted.mean() / K
elif mode == "seq-mean-token-sum":
    loss = -adv_weighted.sum() / K                  # ERL paper
elif mode == "seq-mean-token-sum-norm":
    loss = -adv_weighted.sum() / (K * MAX_LEN)      # Dr. GRPO
```

**New launch script `run_erl_pro6000_paper.sh`:** identical to Pro 6000 base but with paper values:
```
--lr 1e-6
--kl-coef 0.001
--loss-agg-mode seq-mean-token-sum
--clip-ratio-high 0.28
--max-new-tokens 10240
```

### Rollback

All flags default to current values. Existing scripts unaffected.

---

## Combined estimated LOC

| file | LOC |
|------|-----|
| `rl_pipeline/rl_model.py` | +50 (DDP branch) |
| `erl_pipeline/erl_main.py` | +150 (dist init, gather/broadcast, rank-0 guards, agg-mode flag) |
| `erl_pipeline/erl_trainer.py` | +15 (agg-mode dispatch, asymmetric clip) |
| `run_erl_pro6000_ddp.sh` | +80 (new) |
| `run_erl_pro6000_paper.sh` | +80 (new) |
| **Total** | **~375 LOC** |

## Order of operations

1. Land Part 2 (paper-parity flags) first — small, no cluster coordination risk. Validate in a single-GPU ERL run.
2. Land Part 1 (DDP) — bigger, cluster-only testing. Combine with Part 2 flags for the flagship run.

## Open decisions

1. **v1 constraint:** `batch_size == world_size`. Extend to `k × world_size` later (each rank handles k rollouts).
2. **Ray coordination:** rank-0-only dispatch first (simpler). Multi-rank Ray client if we need throughput.
3. **After DDP works, tune LR:** DDP's default averaging divides gradients by 4 vs current sum → effective LR 4× smaller. Either retune or multiply loss by `world_size`.

## Rollback plan

Existing `run_erl_pro6000.sh` untouched. If DDP breaks, use that. If paper-parity flags break, omit them (defaults match current).

## Implementation status (2026-04-19)

**Part 2 (paper-parity flags): DONE.**
- `erl_trainer.py` — 4-way loss aggregation dispatch + PPO-style asymmetric clip.
- `erl_main.py` — `--loss-agg-mode` + `--clip-ratio-high` flags plumbed through.
- `run_erl_pro6000_paper.sh` — new launch script with ERL-paper values.

**Part 1 (DDP): infrastructure landed, rollout-parallel restructuring TODO.**

Landed:
- `rl_model.py` — `use_ddp=True` branch wraps with `DistributedDataParallel`; `underlying()` helper strips DDP for `.generate()` / `.save_pretrained()` / `.disable_adapter_layers()` callers. Non-reentrant gradient checkpointing.
- `erl_main.py` — distributed init at `main()` start (triggered by `LOCAL_RANK`); rank-0 guards on Ray init, save_pretrained; DDP-aware resume path. Raises if `batch_size != world_size` when distributed.

Still TODO (cluster-testing required):
- Phase 1/3 rollout loop restructuring: rank `i` generates rollout `i`, rank 0 dispatches eval via Ray, eval results broadcast back.
- Phase 2 reflection: rank 0 generates, broadcast `full_ids` + `old_logprobs` + `prompt_len` tensors to all ranks.
- Phase 4 distill + logging: all writes (results.tsv, rollouts.jsonl) rank-0 only; gather rewards/bpb before writing.
- `torch.distributed.barrier()` at phase boundaries.
- `run_erl_pro6000_ddp.sh` — new torchrun launch script.

Current DDP infrastructure launches cleanly but requires the rollout-loop work before it produces a correct training run. Paper-parity flags (Part 2) are independently usable today.
