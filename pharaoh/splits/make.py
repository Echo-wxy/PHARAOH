"""Split construction.

Scenario libraries: the simulated scenario library of each reservoir is split
by scenario id (or, optionally, by a coarser hierarchy group) into a
surrogate-training library and an evaluation candidate library. The
surrogate and every normalization statistic are fitted on the training
library only; validation and test queries are answered against the
evaluation library.

Query split: train / val / test over query groups. Queries sharing a
normalized signature (template, objective senses, weights, constraints with
units) or the same normalized text are in one group; a group never crosses
splits.

Ablation subset: a fixed, category-stratified fraction of the test split,
written to a file so every ablation is evaluated on the same queries.
"""
from __future__ import annotations

import re
from collections import defaultdict
from typing import Any

from pharaoh.petrograph.schema import ancestor_at
from pharaoh.utils.seed import derive_rng

SPLITS = ("train", "val", "test")


def normalize_text(text: str) -> str:
    """Lower-case, whitespace-collapsed query text."""
    return re.sub(r"\s+", " ", text.lower()).strip()


def query_groups(queries: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Connected components of queries linked by equal signature or equal normalized text."""
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        while parent.setdefault(x, x) != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    for q in queries:
        union("s:" + q["signature"], "t:" + normalize_text(q["text"]))
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for q in queries:
        groups[find("s:" + q["signature"])].append(q)
    return groups


def split_queries(queries: list[dict[str, Any]], n_queries: dict[str, int], seed: int) -> dict[str, list[dict[str, Any]]]:
    """Group-aware split; groups are filled into test, then val, and the rest go to train."""
    groups = query_groups(queries)
    keys = sorted(groups)
    rng = derive_rng(seed, "query_split")
    rng.shuffle(keys)
    out: dict[str, list[dict[str, Any]]] = {s: [] for s in SPLITS}
    cap = {"test": int(n_queries["test"]), "val": int(n_queries["val"])}
    for k in keys:
        g = groups[k]
        for s in ("test", "val"):
            if len(out[s]) + len(g) <= cap[s]:
                out[s].extend(g)
                break
        else:
            out["train"].extend(g)
    for s in SPLITS:
        out[s].sort(key=lambda q: q["qid"])
    return out


def ablation_subset(test: list[dict[str, Any]], fraction: float, seed: int) -> list[str]:
    """Category-stratified random subset of test query ids."""
    rng = derive_rng(seed, "ablation_subset")
    by_cat: dict[str, list[str]] = defaultdict(list)
    for q in test:
        by_cat[q["category"]].append(q["qid"])
    target = int(round(fraction * len(test)))
    chosen: list[str] = []
    cats = sorted(by_cat)
    for i, cat in enumerate(cats):
        ids = sorted(by_cat[cat])
        k = target - len(chosen) if i == len(cats) - 1 else int(round(fraction * len(ids)))
        chosen.extend(rng.choice(ids, size=min(max(k, 0), len(ids)), replace=False).tolist())
    return sorted(chosen)


def split_library(library: list[str], eval_size: int, unit: str, seed: int) -> tuple[list[str], list[str]]:
    """Split simulated scenario ids into (surrogate-training library, evaluation library).

    `unit` is the hierarchy level that is kept whole: "Control" splits by
    scenario id; "Well" keeps all control variants of a placement on one side.
    With a coarser unit the evaluation library size is matched as closely as
    whole groups allow.
    """
    groups: dict[str, list[str]] = defaultdict(list)
    for sid in library:
        groups[ancestor_at(sid, unit)].append(sid)
    keys = sorted(groups)
    rng = derive_rng(seed, "library_split", unit)
    rng.shuffle(keys)
    eval_ids: list[str] = []
    train_ids: list[str] = []
    for k in keys:
        if len(eval_ids) + len(groups[k]) <= eval_size:
            eval_ids.extend(groups[k])
        else:
            train_ids.extend(groups[k])
    return sorted(train_ids), sorted(eval_ids)
