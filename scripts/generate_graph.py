"""Generate one PetroGraph subgraph from a reservoir config.

Example:
    python scripts/generate_graph.py --config configs/reservoirs/egg.yaml --out data/petrograph/egg
"""
from __future__ import annotations

import argparse
import json

from _common import ROOT, setup_logging

from pharaoh.config import load_config
from pharaoh.petrograph.generate import generate_graph



def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    setup_logging()
    meta = generate_graph(load_config(args.config), ROOT / args.out, root=ROOT)
    print(json.dumps({k: meta[k] for k in ("reservoir", "node_counts", "n_nodes", "edge_counts", "n_edges")}, indent=2))


if __name__ == "__main__":
    main()
