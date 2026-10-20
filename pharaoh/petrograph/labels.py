"""Ground-truth Pareto sets from full-physics simulation results.

For each query, the candidate pool is restricted to scenarios that satisfy
the query's attribute constraints (on d_V) and outcome constraints (on the
simulated outcomes); the non-dominated subset of the remaining scenarios
under the query's objectives is the ground-truth plan set. Scenarios
without a simulation result are counted and treated as infeasible.
"""
from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from pharaoh.petrograph.constraints import satisfies, scenario_summary
from pharaoh.petrograph.pareto import non_dominated_mask


def read_results(path: str | Path, id_column: str = "scenario_id") -> dict[str, dict[str, float]]:
    """Read outcome records, rejecting duplicate ids and malformed numeric values.

    A missing optional breakthrough time is permitted (no breakthrough observed).
    Missing required objectives are detected by the labeling CLI before release.
    """
    out: dict[str, dict[str, float]] = {}
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or id_column not in reader.fieldnames:
            raise ValueError(f"Missing scenario id column {id_column!r} in {path}")
        for row_num, row in enumerate(reader, start=2):
            sid = (row.get(id_column) or "").strip()
            if not sid:
                raise ValueError(f"Empty scenario id in {path}, row {row_num}")
            if sid in out:
                raise ValueError(f"Duplicate scenario id {sid!r} in {path}, row {row_num}")
            vals: dict[str, float] = {}
            for k, raw in row.items():
                if k == id_column:
                    continue
                if k is None:
                    raise ValueError(f"Unexpected extra CSV column in {path}, row {row_num}")
                if raw is None or not raw.strip() or raw.strip().lower() == "nan":
                    if k == "water_breakthrough_days":
                        continue
                    raise ValueError(f"Missing {k!r} for {sid!r} in {path}, row {row_num}")
                try:
                    val = float(raw)
                except ValueError as exc:
                    raise ValueError(f"Invalid {k!r} for {sid!r} in {path}, row {row_num}") from exc
                if not math.isfinite(val):
                    raise ValueError(f"Non-finite {k!r} for {sid!r} in {path}, row {row_num}")
                vals[k] = val
            out[sid] = vals
    return out


def label_query(query: dict[str, Any], pool: list[str], results: dict[str, dict[str, float]],
                summaries: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Ground-truth record for one query."""
    names = [o["name"] for o in query["objectives"]]
    senses = [o["sense"] for o in query["objectives"]]
    feasible, missing = [], 0
    for sid in pool:
        r = results.get(sid)
        if r is None:
            missing += 1
            continue
        if satisfies(summaries[sid], query["attribute_constraints"]) and satisfies(r, query["outcome_constraints"]):
            if all(n in r for n in names):
                feasible.append(sid)
    if feasible:
        vals = np.array([[results[s][n] for n in names] for s in feasible])
        mask = non_dominated_mask(vals, senses)
        pareto = sorted(np.asarray(feasible)[mask].tolist())
    else:
        pareto = []
    return {"qid": query["qid"], "pareto_ids": pareto, "n_pool": len(pool),
            "n_feasible": len(feasible), "n_missing_results": missing}


def label_split(queries: Iterable[dict[str, Any]], pools: dict[str, list[str]],
                results: dict[str, dict[str, float]], nodes: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Label every query of a split."""
    summaries = {n["id"]: scenario_summary(n) for n in nodes if n["type"] == "Control"}
    return [label_query(q, pools[q["qid"]], results, summaries) for q in queries]


def label_stats(labels: list[dict[str, Any]]) -> dict[str, float]:
    """Summary used to compare against reported benchmark statistics."""
    sizes = [len(r["pareto_ids"]) for r in labels]
    return {
        "n_queries": len(labels),
        "mean_pareto_size": float(np.mean(sizes)) if sizes else 0.0,
        "queries_with_empty_ground_truth": int(sum(s == 0 for s in sizes)),
        "mean_feasible_fraction": float(np.mean([r["n_feasible"] / max(r["n_pool"], 1) for r in labels])) if labels else 0.0,
        "total_missing_results": int(sum(r["n_missing_results"] for r in labels)),
    }
