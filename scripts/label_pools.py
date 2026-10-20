"""Compute ground-truth Pareto sets from full-physics results for every split.

The results CSV needs a `scenario_id` column plus one column per objective
and `water_cut` (final field water cut, fraction).

Example:
    python scripts/label_pools.py --config configs/benchmark/egg.yaml --results data/simulations/egg_results.csv
"""
from __future__ import annotations

import argparse
import json

from _common import ROOT, setup_logging

from pharaoh.config import load_config, require
from pharaoh.petrograph.generate import iter_nodes
from pharaoh.petrograph.labels import label_split, label_stats, read_results
from pharaoh.splits.make import SPLITS
from pharaoh.utils.io import read_jsonl, read_lines, write_json, write_jsonl



def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--results", required=True)
    args = ap.parse_args()
    setup_logging()
    cfg = load_config(args.config)
    split_dir = ROOT / require(cfg, "benchmark.split_dir")
    graph_dir = ROOT / require(cfg, "benchmark.graph_dir")
    results = read_results(ROOT / args.results, require(cfg, "benchmark.labels.id_column"))
    expected = set(read_lines(split_dir / "scenarios_to_simulate.txt.gz"))
    required_fields = {"eur_mmbbl", "npv_musd", "capex_musd", "cum_water_mmbbl", "water_cut"}
    missing = expected - results.keys()
    extra = results.keys() - expected
    incomplete = [sid for sid in expected & results.keys() if not required_fields <= results[sid].keys()]
    if missing or extra or incomplete:
        raise ValueError(
            f"Refusing to publish incomplete/misaligned labels: "
            f"{len(missing)} missing, {len(extra)} unexpected, {len(incomplete)} incomplete scenarios. "
            "Check the full-physics run summaries and scenario library."
        )
    nodes = [n for n in iter_nodes(graph_dir) if n["type"] == "Control"]
    stats = {}
    for sp in SPLITS:
        queries = list(read_jsonl(split_dir / f"queries_{sp}.jsonl"))
        pools = {r["qid"]: r["scenario_ids"] for r in read_jsonl(split_dir / f"pools_{sp}.jsonl.gz")}
        labels = label_split(queries, pools, results, nodes)
        write_jsonl(split_dir / f"labels_{sp}.jsonl", labels)
        stats[sp] = label_stats(labels)
    write_json(split_dir / "label_stats.json", stats)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
