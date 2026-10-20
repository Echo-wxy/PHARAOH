"""Simulated scenario libraries and per-query candidate pools.

A fixed library of complete scenarios (Control nodes) is drawn per reservoir,
stratified by region; every scenario in it is run with the full-physics
simulator. The library is then split into a surrogate-training library and an
evaluation library (see `pharaoh.splits.make.split_library`).

Validation and test queries draw their pools from the evaluation library; when
the pool size equals the library size, every evaluation query is answered
against the whole library. Training queries draw pools from the
training library. For queries with region or pattern-type constraints, a
fraction of a sampled pool (`focus_fraction`) comes from the matching subtree.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np

from pharaoh.petrograph.constraints import satisfies, scenario_summary, structural_constraints
from pharaoh.utils.seed import derive_rng


def load_scenarios(nodes) -> list[dict[str, Any]]:
    """Summaries of all Control nodes from a node stream."""
    return [scenario_summary(n) for n in nodes if n["type"] == "Control"]


def draw_library(scenarios: list[dict[str, Any]], size: int, seed: int, reservoir: str) -> list[dict[str, Any]]:
    """Region-stratified library of exactly `size` scenarios."""
    if size > len(scenarios):
        raise ValueError(f"Library size {size} exceeds the {len(scenarios)} Control nodes of the graph")
    rng = derive_rng(seed, "library", reservoir)
    by_region: dict[str, list[int]] = defaultdict(list)
    for i, s in enumerate(scenarios):
        by_region[s["region"]].append(i)
    quota = {r: size * len(idx) / len(scenarios) for r, idx in by_region.items()}
    take = {r: int(np.floor(q)) for r, q in quota.items()}
    for r in sorted(quota, key=lambda r: quota[r] - take[r], reverse=True)[: size - sum(take.values())]:
        take[r] += 1
    chosen: list[int] = []
    for r, idx in sorted(by_region.items()):
        chosen.extend(rng.choice(idx, size=take[r], replace=False).tolist())
    return [scenarios[i] for i in sorted(chosen)]


class StructureIndex:
    """Positions in a scenario list by region, pattern type, and (region, pattern type)."""

    def __init__(self, items: list[dict[str, Any]]) -> None:
        idx: dict[tuple, list[int]] = defaultdict(list)
        for i, s in enumerate(items):
            idx[("region", s["region"])].append(i)
            idx[("pattern_type", s["pattern_type"])].append(i)
            idx[("both", s["region"], s["pattern_type"])].append(i)
        self._idx = {k: np.asarray(v, dtype=int) for k, v in idx.items()}
        self._items = items

    def matches(self, struct: list[dict[str, Any]]) -> np.ndarray:
        """Positions satisfying region / pattern-type equality constraints."""
        d = {c["field"]: c["value"] for c in struct if c["op"] == "=="}
        if set(d) == {"region", "pattern_type"}:
            key: tuple = ("both", d["region"], d["pattern_type"])
        elif set(d) == {"region"}:
            key = ("region", d["region"])
        elif set(d) == {"pattern_type"}:
            key = ("pattern_type", d["pattern_type"])
        else:
            return np.nonzero([satisfies(s, struct) for s in self._items])[0]
        return self._idx.get(key, np.zeros(0, dtype=int))

    def count(self, region: str, pattern_type: str) -> int:
        return len(self._idx.get(("both", region, pattern_type), ()))


def build_pools(queries: list[dict[str, Any]], library: list[dict[str, Any]], pool_size: int,
                focus_fraction: float, seed: int) -> dict[str, list[str]]:
    """Return {qid: sorted scenario ids} for every query."""
    ids = np.array(sorted(s["id"] for s in library))
    if pool_size >= len(library):
        whole = ids.tolist()
        return {q["qid"]: whole for q in queries}
    order = np.argsort([s["id"] for s in library])
    lib = [library[i] for i in order]
    index = StructureIndex(lib)
    pools: dict[str, list[str]] = {}
    for q in queries:
        rng = derive_rng(seed, "pool", q["qid"])
        struct = structural_constraints(q)
        if struct:
            focus_idx = index.matches(struct)
            n_focus = min(int(round(focus_fraction * pool_size)), len(focus_idx))
            a = rng.choice(focus_idx, size=n_focus, replace=False) if n_focus else np.array([], int)
            rest = np.setdiff1d(np.arange(len(lib)), a, assume_unique=True)
            b = rng.choice(rest, size=pool_size - n_focus, replace=False)
            idx = np.concatenate([a, b])
        else:
            idx = rng.choice(len(lib), size=pool_size, replace=False)
        pools[q["qid"]] = sorted(ids[idx].tolist())
    return pools
