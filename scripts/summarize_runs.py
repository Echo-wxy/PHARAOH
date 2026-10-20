"""Turn full-physics summary output into the results CSV used for labels.

Each run is read either from a CSV with columns DAYS, FOPT, FWPT, FWIT
(one file per scenario, named like the schedule file with a .csv suffix) or,
if the `resdata` package is installed and --format unsmry is given, directly
from the simulator's <name>.UNSMRY/.SMSPEC files.

Example:
    python scripts/summarize_runs.py --reservoir-config configs/reservoirs/egg.yaml \
        --benchmark-config configs/benchmark/egg.yaml --runs data/simulations/egg/runs \
        --out data/simulations/egg_results.csv
"""
from __future__ import annotations

import argparse
import csv
import logging
import os
import tempfile
from pathlib import Path

import numpy as np

from _common import ROOT, setup_logging

from pharaoh.config import load_config, require
from pharaoh.petrograph.economics import capex_musd, objectives_from_totals
from pharaoh.petrograph.generate import iter_nodes
from pharaoh.petrograph.schedule import file_name
from pharaoh.utils.io import read_lines


FIELDS = ["scenario_id", "eur_mmbbl", "npv_musd", "capex_musd", "cum_water_mmbbl", "water_cut", "water_breakthrough_days"]
KEYS = ("DAYS", "FOPT", "FWPT", "FWIT")
log = logging.getLogger("summarize_runs")


def read_csv_run(path: Path) -> dict[str, np.ndarray]:
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return {k: np.array([float(r[k]) for r in rows]) for k in KEYS}


def read_unsmry_run(case: Path) -> dict[str, np.ndarray]:
    from resdata.summary import Summary  # optional dependency

    s = Summary(str(case))
    return {"DAYS": np.asarray(s.numpy_vector("TIME")), **{k: np.asarray(s.numpy_vector(k)) for k in KEYS[1:]}}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reservoir-config", required=True)
    ap.add_argument("--benchmark-config", required=True)
    ap.add_argument("--runs", required=True, help="directory with one run per scenario")
    ap.add_argument("--format", choices=["csv", "unsmry"], default="csv")
    ap.add_argument("--out", required=True)
    ap.add_argument("--allow-incomplete", action="store_true",
                    help="write partial results only for diagnostics; partial results cannot be labeled")
    args = ap.parse_args()
    setup_logging()
    rcfg, bcfg = load_config(args.reservoir_config), load_config(args.benchmark_config)
    fp = require(rcfg, "full_physics")
    wanted = set(read_lines(ROOT / require(bcfg, "benchmark.split_dir") / "scenarios_to_simulate.txt.gz"))
    runs = ROOT / args.runs
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    done = missing = 0
    seen: set[str] = set()
    # Only publish a complete results CSV. A failed run must not leave a file
    # that could be mistaken for full-physics ground truth.
    with tempfile.TemporaryDirectory(prefix=".pharaoh-results-", dir=out.parent) as tmp:
        temporary = Path(tmp) / "results.csv"
        with open(temporary, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            w.writeheader()
            for node in iter_nodes(ROOT / require(bcfg, "benchmark.graph_dir")):
                if node["id"] not in wanted:
                    continue
                seen.add(node["id"])
                stem = file_name(node["id"])[:-4]
                path = runs / (stem + ".csv") if args.format == "csv" else runs / stem / stem
                try:
                    ts = read_csv_run(path) if args.format == "csv" else read_unsmry_run(path)
                except (FileNotFoundError, OSError):
                    missing += 1
                    continue
                obj = objectives_from_totals(ts["DAYS"], ts["FOPT"], ts["FWPT"], ts["FWIT"],
                                             capex_musd(node, fp["economics"]["well_cost_model"]), fp["economics"],
                                             float(fp["water_breakthrough_threshold"]))
                w.writerow({"scenario_id": node["id"], **obj})
                done += 1
        missing += len(wanted - seen)
        if missing and not args.allow_incomplete:
            raise RuntimeError(f"{missing} of {len(wanted)} requested full-physics runs are missing. "
                               "No results file written. Use --allow-incomplete for diagnostics only.")
        os.replace(temporary, out)
    log.info("summarized %d runs, %d missing (of %d scenarios)", done, missing, len(wanted))
    print(f"wrote {out.relative_to(ROOT)}: {done} scenarios, {missing} missing")


if __name__ == "__main__":
    main()
