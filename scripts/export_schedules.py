"""Write simulator SCHEDULE includes for every scenario that needs a full-physics run.

Example:
    python scripts/export_schedules.py --reservoir-config configs/reservoirs/egg.yaml \
        --benchmark-config configs/benchmark/egg.yaml --out data/simulations/egg/schedules
"""
from __future__ import annotations

import argparse

from _common import ROOT, setup_logging

from pharaoh.config import load_config, require
from pharaoh.petrograph.generate import iter_nodes
from pharaoh.petrograph.schedule import export_schedules
from pharaoh.utils.io import read_lines



def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reservoir-config", required=True)
    ap.add_argument("--benchmark-config", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    setup_logging()
    rcfg, bcfg = load_config(args.reservoir_config), load_config(args.benchmark_config)
    split_dir = ROOT / require(bcfg, "benchmark.split_dir")
    wanted = set(read_lines(split_dir / "scenarios_to_simulate.txt.gz"))
    n = export_schedules(iter_nodes(ROOT / require(bcfg, "benchmark.graph_dir")), wanted,
                         require(rcfg, "full_physics"), ROOT / args.out)
    if n != len(wanted):
        raise ValueError(f"Exported {n} schedules but expected {len(wanted)}; missing Control nodes")
    print(f"wrote {n} schedule files to {args.out} (expected {len(wanted)})")


if __name__ == "__main__":
    main()
