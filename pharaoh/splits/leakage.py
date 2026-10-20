"""Leakage audit over a split directory.

Hard checks (any failure makes the audit fail):
  1. query ids are disjoint across train / val / test
  2. no normalized query text appears in more than one split
  3. no normalized query signature appears in more than one split
  4. the ablation subset lies inside the test split
  5. the surrogate-training and evaluation libraries are disjoint
     (and, if the split unit is coarser than a scenario, no unit straddles them)
  6. no val/test pool contains a surrogate-training scenario
  7. no train pool contains an evaluation-library scenario
  8. every file hash matches the manifest
Reported for information: well placements shared between the two libraries
(control variants of one placement on both sides) when splitting by scenario id.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from pharaoh.petrograph.schema import ancestor_at
from pharaoh.splits.make import SPLITS, normalize_text
from pharaoh.utils.io import read_json, read_jsonl, read_lines, sha256_file


def _cross_split(values: dict[str, set[str]]) -> int:
    seen: dict[str, set[str]] = defaultdict(set)
    for sp, vals in values.items():
        for v in vals:
            seen[v].add(sp)
    return sum(1 for sps in seen.values() if len(sps) > 1)


def audit(split_dir: str | Path) -> dict[str, Any]:
    """Run all checks; returns a JSON-serializable report."""
    d = Path(split_dir)
    man = read_json(d / "manifest.json")
    queries = {sp: list(read_jsonl(d / f"queries_{sp}.jsonl")) for sp in SPLITS}
    pools = {sp: {r["qid"]: r["scenario_ids"] for r in read_jsonl(d / f"pools_{sp}.jsonl.gz")} for sp in SPLITS}
    c: dict[str, Any] = {}
    c["qid_cross_split"] = _cross_split({sp: {q["qid"] for q in qs} for sp, qs in queries.items()})
    c["text_cross_split"] = _cross_split({sp: {normalize_text(q["text"]) for q in qs} for sp, qs in queries.items()})
    c["signature_cross_split"] = _cross_split({sp: {q["signature"] for q in qs} for sp, qs in queries.items()})
    subset = set(read_lines(d / man["files"]["test_subset_qids"]))
    c["subset_outside_test"] = len(subset - {q["qid"] for q in queries["test"]})

    tr = set(read_lines(d / man["files"]["surrogate_train"]))
    ev = set(read_lines(d / man["files"]["eval_candidates"]))
    unit = man["policies"]["library_split_unit"]
    c["library_overlap"] = len(tr & ev)
    c["library_unit_straddle"] = len({ancestor_at(s, unit) for s in tr} & {ancestor_at(s, unit) for s in ev})
    eval_pool = {sid for sp in ("val", "test") for p in pools[sp].values() for sid in p}
    train_pool = {sid for p in pools["train"].values() for sid in p}
    c["surrogate_train_in_val_test_pools"] = len(eval_pool & tr)
    c["eval_library_in_train_pools"] = len(train_pool & ev)
    c["val_test_pools_outside_eval_library"] = len(eval_pool - ev)
    c["well_placements_in_both_libraries"] = len({ancestor_at(s, "Well") for s in tr} & {ancestor_at(s, "Well") for s in ev})
    c["hash_mismatch"] = [k for k, v in man["sha256"].items() if sha256_file(d / man["files"][k]) != v]

    hard = ["qid_cross_split", "text_cross_split", "signature_cross_split", "subset_outside_test", "library_overlap",
            "library_unit_straddle", "surrogate_train_in_val_test_pools", "eval_library_in_train_pools",
            "val_test_pools_outside_eval_library", "hash_mismatch"]
    failures = [k for k in hard if c[k]]
    return {"reservoir": man["reservoir"], "ok": not failures, "failures": failures,
            "library_split_unit": unit, "checks": c}


def format_report(rep: dict[str, Any]) -> str:
    """Plain-text rendering of an audit report."""
    lines = [f"Leakage audit: {rep['reservoir']}  ->  {'PASS' if rep['ok'] else 'FAIL'}"]
    for k, v in rep["checks"].items():
        tag = "  (info)" if k == "well_placements_in_both_libraries" else ""
        lines.append(f"  {k:40s} {v}{tag}")
    if rep["failures"]:
        lines.append("  failed: " + ", ".join(rep["failures"]))
    return "\n".join(lines)
