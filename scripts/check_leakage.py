"""Audit a split directory for query and scenario leakage; exits non-zero on failure.

Example:
    python scripts/check_leakage.py --split-dir splits/egg --json outputs/leakage_egg.json
"""
from __future__ import annotations

import argparse
import sys

from _common import ROOT

from pharaoh.splits.leakage import audit, format_report
from pharaoh.utils.io import write_json



def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split-dir", required=True, nargs="+")
    ap.add_argument("--json", default=None, help="optional path for a machine-readable report")
    args = ap.parse_args()
    reports = [audit(ROOT / d) for d in args.split_dir]
    for r in reports:
        print(format_report(r))
    if args.json:
        write_json(ROOT / args.json, reports)
    sys.exit(0 if all(r["ok"] for r in reports) else 1)


if __name__ == "__main__":
    main()
