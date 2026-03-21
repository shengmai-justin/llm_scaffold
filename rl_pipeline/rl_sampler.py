"""PUCT tree search over train.py edits.

Ported from ttt_autoresearch/puct.py.
"""
from __future__ import annotations

import json
import os
import uuid

import numpy as np


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class State:
    """A node in the search tree. value = -val_bpb (higher is better for PUCT)."""

    __slots__ = ("id", "timestep", "value", "code", "parent_values", "parents", "observation")

    def __init__(
        self,
        timestep: int = 0,
        code: str = "",
        value: float | None = None,
        parent_values: list[float] | None = None,
        parents: list[dict] | None = None,
        id: str | None = None,
        observation: str = "",
    ):
        self.id = id or str(uuid.uuid4())
        self.timestep = timestep
        self.value = value
        self.code = code
        self.parent_values = parent_values or []
        self.parents = parents or []
        self.observation = observation

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "timestep": self.timestep,
            "value": self.value,
            "code": self.code,
            "parent_values": self.parent_values,
            "parents": self.parents,
            "observation": self.observation,
        }

    @classmethod
    def from_dict(cls, d: dict) -> State:
        return cls(
            timestep=d["timestep"],
            code=d.get("code", ""),
            value=d.get("value"),
            parent_values=d.get("parent_values", []),
            parents=d.get("parents", []),
            id=d.get("id"),
            observation=d.get("observation", ""),
        )


# ---------------------------------------------------------------------------
# PUCTSampler
# ---------------------------------------------------------------------------

class PUCTSampler:
    """PUCT tree search. score(i) = Q(i) + c * scale * P(i) * sqrt(1+T) / (1+n[i])"""

    def __init__(
        self,
        initial_state: State,
        log_dir: str = "./rl_log",
        puct_c: float = 1.0,
        max_buffer: int = 500,
        topk_children: int = 2,
        resume_step: int | None = None,
    ):
        self.log_dir = log_dir
        self.puct_c = puct_c
        self.max_buffer = max_buffer
        self.topk_children = topk_children

        self._states: list[State] = []
        self._initial_ids: set[str] = set()
        self._n: dict[str, int] = {}       # visit counts
        self._m: dict[str, float] = {}     # best reachable value
        self._T: int = 0                   # total visits

        if resume_step is not None:
            self.load(resume_step)
        else:
            self._states.append(initial_state)
            self._initial_ids.add(initial_state.id)

    # -- persistence ---------------------------------------------------------

    def _path(self, step: int) -> str:
        os.makedirs(self.log_dir, exist_ok=True)
        return os.path.join(self.log_dir, f"puct_step_{step:06d}.json")

    def save(self, step: int):
        store = {
            "step": step,
            "states": [s.to_dict() for s in self._states],
            "initial_ids": list(self._initial_ids),
            "puct_n": self._n,
            "puct_m": {k: float(v) for k, v in self._m.items()},
            "puct_T": self._T,
        }
        tmp = f"{self._path(step)}.tmp.{os.getpid()}"
        os.makedirs(os.path.dirname(tmp) or ".", exist_ok=True)
        with open(tmp, "w") as f:
            json.dump(store, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, self._path(step))

    def load(self, step: int):
        path = self._path(step)
        if not os.path.exists(path):
            raise FileNotFoundError(f"No sampler checkpoint at {path}")
        with open(path) as f:
            store = json.load(f)
        self._states = [State.from_dict(s) for s in store["states"]]
        self._initial_ids = set(store.get("initial_ids", []))
        self._n = store.get("puct_n", {})
        self._m = {k: float(v) for k, v in store.get("puct_m", {}).items()}
        self._T = int(store.get("puct_T", 0))

    # -- PUCT scoring --------------------------------------------------------

    def sample_state(self) -> State:
        """Pick the highest-PUCT-score state to expand."""
        if not self._states:
            raise RuntimeError("No states in buffer")

        vals = np.array([
            float(s.value if s.value is not None else float("-inf"))
            for s in self._states
        ])
        rng = max(float(np.max(vals) - np.min(vals)), 1e-6)

        # Rank-based prior: best value gets highest weight
        N = len(vals)
        ranks = np.argsort(np.argsort(-vals))
        weights = (N - ranks).astype(np.float64)
        P = weights / weights.sum()

        sqrtT = np.sqrt(1.0 + self._T)

        best_idx, best_score = 0, float("-inf")
        for i, s in enumerate(self._states):
            n = self._n.get(s.id, 0)
            m = self._m.get(s.id, vals[i])
            Q = m if n > 0 else vals[i]
            bonus = self.puct_c * rng * P[i] * sqrtT / (1.0 + n)
            score = Q + bonus
            if score > best_score:
                best_score = score
                best_idx = i

        return self._states[best_idx]

    # -- update tree ---------------------------------------------------------

    def update_state(self, child: State, parent: State):
        """Add child to tree, update visit counts and best-reachable values."""
        child.parent_values = (
            [parent.value] + parent.parent_values if parent.value is not None else []
        )
        child.parents = [{"id": parent.id, "timestep": parent.timestep}] + parent.parents

        if child.value is not None:
            pid = parent.id
            self._m[pid] = max(self._m.get(pid, child.value), child.value)
            anc_ids = [pid] + [str(p["id"]) for p in (parent.parents or []) if p.get("id")]
            for aid in anc_ids:
                self._n[aid] = self._n.get(aid, 0) + 1
            self._T += 1

        # Add child (dedup by code)
        if child.value is not None:
            existing_codes = {s.code for s in self._states if s.code}
            if child.code not in existing_codes:
                self._states.append(child)

        self._apply_topk_filter()
        self._trim_buffer()

    def record_failed_rollout(self, parent: State):
        """Increment visits without adding a child."""
        anc_ids = [parent.id] + [str(p["id"]) for p in (parent.parents or []) if p.get("id")]
        for aid in anc_ids:
            self._n[aid] = self._n.get(aid, 0) + 1
        self._T += 1

    def _apply_topk_filter(self):
        if self.topk_children <= 0:
            return
        by_parent: dict[str, list[State]] = {}
        no_parent: list[State] = []
        for s in self._states:
            if s.parents:
                pid = s.parents[0]["id"]
                by_parent.setdefault(pid, []).append(s)
            else:
                no_parent.append(s)
        filtered = list(no_parent)
        for children in by_parent.values():
            children.sort(
                key=lambda x: x.value if x.value is not None else float("-inf"),
                reverse=True,
            )
            filtered.extend(children[: self.topk_children])
        self._states = filtered

    def _trim_buffer(self):
        if len(self._states) <= self.max_buffer:
            return
        scored = sorted(
            enumerate(self._states),
            key=lambda x: x[1].value if x[1].value is not None else float("-inf"),
            reverse=True,
        )
        keep = {i for i, s in enumerate(self._states) if s.id in self._initial_ids}
        for idx, _ in scored:
            if len(keep) >= self.max_buffer:
                break
            keep.add(idx)
        self._states = [self._states[i] for i in sorted(keep)]

    # -- info ----------------------------------------------------------------

    def best_state(self) -> State | None:
        if not self._states:
            return None
        return max(self._states, key=lambda s: s.value if s.value is not None else float("-inf"))

    def buffer_size(self) -> int:
        return len(self._states)
