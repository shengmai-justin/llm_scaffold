# The `custom_op` Gap — FA4 + `torch.compile` Compiler Stack

Report on the single largest engineering gap between our best config (val_bpb=0.969023) and forge's global-best (val_bpb=0.926381). Estimated contribution: **~−0.013 BPB** of the total −0.043 gap. Dated 2026-04-23.

## What `custom_op` is

`torch.library.custom_op` is PyTorch's sanctioned mechanism for registering an opaque operator with Inductor's graph. When `torch.compile` traces the model, it needs every op in the forward and backward path to be either:

1. A known ATen op (trivially traceable), or
2. A `custom_op` with registered `meta` (shape inference), `setup_context`, and backward implementations.

Anything else causes a **graph break** — Inductor splits the compiled region, dropping back to eager between the break points. Graph breaks kill the two largest wins of compile: cross-op fusion (e.g., attention+residual+RMSNorm in one kernel) and CUDA-graph capture (removing per-launch overhead).

Flash-Attention-4 (`flash_attn.cute.flash_attn_func`) is a CUTLASS-backed extension kernel. It is **not** an ATen op, so naive `torch.compile(model)` breaks the graph at every attention call. The `custom_op` wrapper tells Inductor: "this is a black-box op with these input/output shapes and this backward — fuse around it, schedule it, capture it in a CUDA graph, but don't trace into it."

Without the wrapper, `torch.compile` "works" but buys you almost nothing on an FA4-based transformer: the model is a sequence of Inductor islands separated by eager FA4 calls.

## What forge did

Forge's `github.com/mikeapedia/autoresearch-at-home` branch `autoresearch/mar15` landed the wrapper over **seven commits** (Claude + human-in-loop, all with runtime-error feedback):

1. First attempt: FlexAttention — failed (wrong algorithm for their attention pattern).
2. Fallback: `torch._dynamo.allow_in_graph(flash_attn_func)` — partial success, but insufficient fusion (Inductor still treated the call as opaque-in-eager, no cross-op scheduling).
3. Escalation: full `@torch.library.custom_op` registration — baseline working version.
4. Bug fix: saved-LSE tensor (log-sum-exp from forward needed in backward) was being recomputed instead of saved, causing numerical drift.
5. Bug fix: return-type mismatch (forward returned a tuple, Inductor expected single tensor).
6. Backward wrap: registered `torch.library.register_autograd` so the compiled graph includes FA4 backward.
7. Final polish: `setup_context` tightened to only save tensors actually needed for backward, cutting activation memory.

On top of the wrapper, forge enables an Inductor config stack:

```python
import torch._inductor.config as ic
ic.coordinate_descent_tuning = True
ic.epilogue_fusion = True
ic.aggressive_fusion = True
ic.shape_padding = True
ic.max_autotune_pointwise = True
```

And compiles with the strongest mode plus CUDA graph capture:

```python
model = torch.compile(model, mode="max-autotune", fullgraph=True, dynamic=False)
```

Combined payoff on their H200 XL tier: **~−0.021 BPB** vs eager FA4. The mechanism is not a model change — it's pure throughput. More tokens processed per second at the same val budget → more training steps within the 5-minute wallclock → better val_bpb.

## What ours did

Our best train.py (`results/2026-04-20_erl_ttt_split_best_train.py`) uses:

```python
from flash_attn.cute import flash_attn_func as fa4    # line 24 — raw FA4 kernel
...
@torch.compile(dynamic=False, fullgraph=True)         # lines 305, 316 — two helper fns only
def norm(x): ...

model = torch.compile(model, dynamic=False)           # line 508 — default mode, no custom_op
```

We have **two thirds** of the stack:

- ✓ FA4 kernel imported and called directly.
- ✓ `torch.compile` applied to the model and two helpers with `fullgraph=True`.
- ✗ **No `custom_op` wrapper** — every `fa4(...)` call is a graph break.
- ✗ No Inductor flags tuned — `coordinate_descent_tuning`, `epilogue_fusion`, `aggressive_fusion`, `shape_padding`, `max_autotune_pointwise` all default.
- ✗ Default compile mode (not `"max-autotune"`), no CUDA graph capture.

Concretely, `fullgraph=True` on the model *should* raise on a graph break — which means either our FA4 call is being allow-listed implicitly somewhere in our environment, or the compile is silently falling back. Either way, we are not getting cross-attention-layer fusion or CUDA graphs, and the Inductor config is at upstream defaults.

Estimated captured value: **~−0.010 of forge's −0.021 stack**. Remaining **~−0.011** is sitting in the compiler gap alone, with another **~−0.002** plausibly available from the aggressive Inductor flags on top.

## Why RL won't close this gap

The wrapper code is ~40 lines, most of them mechanical (`setup_context`, `backward`, `register_fake` for meta shape inference). Qwen3.5-9B, our ideator model, has never seen this API at pretraining scale — `torch.library.custom_op` is a 2024 addition, underrepresented in code corpora. Forge's Claude-assisted trajectory shows that even a 200B-class model with full runtime-error feedback needed seven iterations to land it. Our ERL pipeline gives the ideator:

- **No runtime feedback.** Status is binary (`crash` vs `keep`); the traceback never reaches the next rollout.
- **One shot per rollout.** A partial wrapper that compiles but crashes on backward (forge commits 4–6) is indistinguishable from random noise to the GRPO/TTT reward signal.
- **Dead-end suppression.** After 3 crashes on "wrap FA4 in custom_op", `erl_history.py` will bucket it as "DO NOT propose" and extinguish the direction permanently.

This is the horizon mismatch described in `docs/engineering_history.md` (2026-04-23) and the dilemma discussed in the 2026-04-23 session: forge's fix is a horizon-7 trajectory; our scaffold evaluates at horizon-1.

## Recommendation

**Port the wrapper manually**, not via RL. The contract is:

1. Copy forge's final `custom_op` wrapper (post-commit-7) verbatim into our `autoresearch/train.py` and `train_sdpa.py`. ~40 lines.
2. Add the five Inductor config flags at the top of `train.py`.
3. Change `torch.compile(model, dynamic=False)` → `torch.compile(model, mode="max-autotune", fullgraph=True, dynamic=False)`.
4. Baseline the new `train.py` (single run, no RL) to confirm val_bpb drop of roughly −0.010 to −0.013 on B200.
5. Resume ERL from the new baseline. RL now operates on a stronger compiler-layer foundation and can spend its horizon-1 budget on single-edit architecture/HP wins where it already performs well.

Cost: ~1 hour of manual porting + 1 baseline run (~30 min B200). Expected value: ~−0.013 BPB, roughly **30% of the total remaining gap to forge**, at zero RL compute.

Trying to RL-discover this is negative-expected-value: every crashed attempt poisons the history summarizer against the correct direction, and the probability of Qwen emitting the complete seven-step trajectory in one rollout is effectively zero.
