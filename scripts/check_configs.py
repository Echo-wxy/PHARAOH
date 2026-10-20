"""Load every config, list settings that are still null, and print the baseline resource table.

Null values in fully resolved configurations may indicate incomplete
experiment settings. This is a structural completeness check; it does not
validate published metrics or execute model-training configurations.

Example:
    python scripts/check_configs.py --resources
"""
from __future__ import annotations

import argparse

from _common import ROOT

from pharaoh.config import find_unset, load_config


COLUMNS = ["physical_attributes", "surrogate_scores", "feasibility_rules",
           "llm_backbone", "step_budget", "trained_on_target_graph"]


def _cell(v) -> str:
    if v is True:
        return "yes"
    if v is False:
        return "no"
    return "?" if v is None else str(v)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--resources", action="store_true", help="print the baseline resource table")
    args = ap.parse_args()
    files = sorted(p for p in (ROOT / "configs").rglob("*.yaml") if not p.name.startswith("_"))
    total = 0
    for p in files:
        cfg = load_config(p)
        unset = find_unset(cfg)
        total += len(unset)
        rel = p.relative_to(ROOT)
        print(f"{rel}: {'all set' if not unset else f'{len(unset)} unset'}")
        for k in unset:
            print(f"    - {k}")
    print(f"\n{len(files)} configs loaded, {total} unset values in total.")
    if args.resources:
        rows = []
        for p in sorted((ROOT / "configs" / "baselines").glob("*.yaml")):
            if p.name.startswith("_"):
                continue
            b = load_config(p)["baseline"]
            rows.append([b["name"], b["regime"], b["family"]] + [_cell(b["resources"].get(c)) for c in COLUMNS])
        head = ["Method", "Regime", "Family"] + COLUMNS
        print("\n| " + " | ".join(head) + " |\n|" + "---|" * len(head))
        for r in rows:
            print("| " + " | ".join(r) + " |")


if __name__ == "__main__":
    main()
