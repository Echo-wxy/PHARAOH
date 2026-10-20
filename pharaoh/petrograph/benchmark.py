"""End-to-end benchmark assembly for one reservoir: libraries, queries, pools, splits, manifest.

Order of operations (each step uses only what the previous steps produced):
    1. draw the simulated scenario library (region-stratified Control nodes)
    2. split it by scenario id into the surrogate-training library and the
       evaluation candidate library
    3. generate queries; region / pattern-type constraints are restricted to
       combinations present in both libraries
    4. split queries into train / val / test by query group; draw the fixed
       ablation subset of the test split
    5. assign pools: val/test queries -> evaluation library,
       train queries -> training library

Outputs (in the split directory):
    queries_{train,val,test}.jsonl       structured queries
    pools_{train,val,test}.jsonl.gz      {qid, scenario_ids}
    test_subset_qids.txt                 fixed ablation subset of the test split
    surrogate_train_scenarios.txt.gz     training library (surrogate fitting and normalization statistics)
    eval_candidate_scenarios.txt.gz      evaluation library (val/test pools; surrogate held-out accuracy)
    scenarios_to_simulate.txt.gz         full-physics workload (both libraries)
    manifest.json                        counts, policies, hashes
"""
from __future__ import annotations

import logging
from collections import Counter
from pathlib import Path
from typing import Any

from pharaoh import __version__
from pharaoh.config import config_hash, require
from pharaoh.petrograph.constraints import structural_constraints
from pharaoh.petrograph.generate import iter_nodes
from pharaoh.petrograph.pools import StructureIndex, build_pools, draw_library, load_scenarios
from pharaoh.petrograph.queries import generate_queries, load_templates
from pharaoh.petrograph.schema import ancestor_at
from pharaoh.splits.make import SPLITS, ablation_subset, split_library, split_queries
from pharaoh.utils.io import read_json, sha256_file, write_json, write_jsonl, write_lines

log = logging.getLogger(__name__)


def build_benchmark(cfg: dict[str, Any], graph_dir: str | Path, out_dir: str | Path, root: str | Path = ".") -> dict[str, Any]:
    """Build libraries, queries, pools, and all split files for one reservoir; returns the manifest."""
    root, graph_dir, out_dir = Path(root), Path(graph_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    b = require(cfg, "benchmark")
    seed = int(require(cfg, "benchmark.seed"))
    meta = read_json(graph_dir / "meta.json")
    lib_cfg = require(cfg, "benchmark.library")
    unit = lib_cfg["split_unit"]

    # 1-2. scenario library and its split
    scenarios = load_scenarios(iter_nodes(graph_dir))
    library = draw_library(scenarios, int(lib_cfg["size"]), seed, meta["reservoir"])
    train_ids, eval_ids = split_library([s["id"] for s in library], int(lib_cfg["eval_size"]), unit, seed)
    by_id = {s["id"]: s for s in library}
    train_lib = [by_id[i] for i in train_ids]
    eval_lib = [by_id[i] for i in eval_ids]

    # 3. queries
    min_m = int(lib_cfg["min_structural_matches"])
    ti, ei = StructureIndex(train_lib), StructureIndex(eval_lib)
    allowed = [(r, p) for r, ps in sorted(meta["region_patterns"].items()) for p in ps
               if ti.count(r, p) >= min_m and ei.count(r, p) >= min_m]
    templates = load_templates(root / require(cfg, "benchmark.query_templates"))
    queries = generate_queries(b, {**meta, "allowed_structures": allowed}, templates)

    # 4. query split and ablation subset
    split = split_queries(queries, b["n_queries"], seed)
    subset = ablation_subset(split["test"], float(require(cfg, "benchmark.split.test_subset_fraction")), seed)

    # 5. pools
    p = require(cfg, "benchmark.pools")
    focus = float(p["focus_fraction"])
    pools = build_pools(split["train"], train_lib, int(p["train_pool_size"]), focus, seed)
    pools.update(build_pools(split["val"] + split["test"], eval_lib, int(p["pool_size"]), focus, seed))

    files: dict[str, str] = {}
    for sp in SPLITS:
        write_jsonl(out_dir / f"queries_{sp}.jsonl", split[sp])
        write_jsonl(out_dir / f"pools_{sp}.jsonl.gz", ({"qid": q["qid"], "scenario_ids": pools[q["qid"]]} for q in split[sp]))
        files[f"queries_{sp}"] = f"queries_{sp}.jsonl"
        files[f"pools_{sp}"] = f"pools_{sp}.jsonl.gz"
    write_lines(out_dir / "test_subset_qids.txt", subset)
    write_lines(out_dir / "surrogate_train_scenarios.txt.gz", train_ids)
    write_lines(out_dir / "eval_candidate_scenarios.txt.gz", eval_ids)
    write_lines(out_dir / "scenarios_to_simulate.txt.gz", sorted(by_id))
    files.update({
        "test_subset_qids": "test_subset_qids.txt",
        "surrogate_train": "surrogate_train_scenarios.txt.gz",
        "eval_candidates": "eval_candidate_scenarios.txt.gz",
        "scenarios_to_simulate": "scenarios_to_simulate.txt.gz",
    })

    def unmatched(qs: list[dict[str, Any]], index: StructureIndex) -> int:
        return sum(1 for q in qs if structural_constraints(q) and len(index.matches(structural_constraints(q))) < min_m)

    shared_wells = len({ancestor_at(i, "Well") for i in train_ids} & {ancestor_at(i, "Well") for i in eval_ids})
    groups = Counter(q["signature"] for q in queries)
    manifest = {
        "reservoir": meta["reservoir"],
        "builder_version": __version__,
        "benchmark_config_hash": config_hash(cfg),
        "graph": {"config_hash": meta["config_hash"], "sha256": meta["sha256"],
                  "n_nodes": meta["n_nodes"], "n_edges": meta["n_edges"]},
        "seed": seed,
        "policies": {
            "library_split_unit": unit,
            "surrogate_fit_on": "surrogate_train",
            "normalization_fit_on": "surrogate_train",
            "surrogate_heldout": "eval_candidates",
            "val_test_pools_from": "eval_candidates",
            "train_pools_from": "surrogate_train",
            "query_split_unit": "normalized signature / text group",
            "pool_size": int(p["pool_size"]),
            "train_pool_size": int(p["train_pool_size"]),
            "focus_fraction": focus,
        },
        "counts": {
            "library": len(library),
            "surrogate_train": len(train_ids),
            "eval_candidates": len(eval_ids),
            "well_placements_in_both_libraries": shared_wells,
            "queries": {sp: len(split[sp]) for sp in SPLITS},
            "queries_by_category": {sp: dict(Counter(q["category"] for q in split[sp])) for sp in SPLITS},
            "distinct_signatures": {sp: len({q["signature"] for q in split[sp]}) for sp in SPLITS},
            "largest_signature_group": max(groups.values()),
            "test_subset": len(subset),
            "test_subset_by_category": dict(Counter(q["category"] for q in split["test"] if q["qid"] in set(subset))),
            "allowed_region_pattern_combinations": len(allowed),
            "queries_below_structural_minimum": {"train": unmatched(split["train"], ti),
                                                 "val": unmatched(split["val"], ei), "test": unmatched(split["test"], ei)},
            "full_physics_runs": len(library),
        },
        "files": files,
    }
    manifest["sha256"] = {k: sha256_file(out_dir / v) for k, v in files.items()}
    write_json(out_dir / "manifest.json", manifest)
    log.info("%s: %s", meta["reservoir"], manifest["counts"])
    return manifest
