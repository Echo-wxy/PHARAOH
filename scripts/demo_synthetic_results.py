"""Write synthetic stand-in run summaries for every scenario of the demo split (demo only).

Stands in for the full-physics simulator in the quick demo: one CSV per
scenario with DAYS, FOPT, FWPT, FWIT, in the layout read by
scripts/summarize_runs.py. It must not be used for real benchmark labels.
"""
from __future__ import annotations

import argparse
import csv

from _common import ROOT

from pharaoh.config import load_config, require
from pharaoh.petrograph.generate import iter_nodes
from pharaoh.petrograph.schedule import file_name
from pharaoh.synthetic import DEMO_NOTICE, synthetic_run
from pharaoh.utils.io import read_lines



def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reservoir-config", required=True)
    ap.add_argument("--benchmark-config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rcfg, bcfg = load_config(args.reservoir_config), load_config(args.benchmark_config)
    fp = require(rcfg, "full_physics")
    needed = set(read_lines(ROOT / require(bcfg, "benchmark.split_dir") / "scenarios_to_simulate.txt.gz"))
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    for n in iter_nodes(ROOT / require(bcfg, "benchmark.graph_dir")):
        if n["id"] not in needed:
            continue
        ts = synthetic_run(n, float(fp["horizon_years"]), float(fp["report_step_days"]), args.seed)
        with open(out / (file_name(n["id"])[:-4] + ".csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(list(ts))
            for row in zip(*ts.values()):
                w.writerow([f"{v:.6g}" for v in row])
    print(DEMO_NOTICE)


if __name__ == "__main__":
    main()
