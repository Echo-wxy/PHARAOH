"""Build queries, candidate pools, and all split files for one reservoir.

Example:
    python scripts/build_benchmark.py --config configs/benchmark/egg.yaml
"""
from __future__ import annotations

import argparse
import json

from _common import ROOT, setup_logging

from pharaoh.config import load_config, require
from pharaoh.petrograph.benchmark import build_benchmark



def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    setup_logging()
    cfg = load_config(args.config)
    man = build_benchmark(cfg, ROOT / require(cfg, "benchmark.graph_dir"), ROOT / require(cfg, "benchmark.split_dir"), root=ROOT)
    print(json.dumps({"policies": man["policies"], "counts": man["counts"]}, indent=2))


if __name__ == "__main__":
    main()
