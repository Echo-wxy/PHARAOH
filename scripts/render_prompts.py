"""Render the agent prompts for one benchmark query, for inspection or archiving.

Example:
    python scripts/render_prompts.py --benchmark-config configs/benchmark/egg.yaml \
        --agent-config configs/agent/pharaoh.yaml --split test --index 0 --out outputs/prompts_egg.md
"""
from __future__ import annotations

import argparse

from _common import ROOT

from pharaoh.config import load_config, require
from pharaoh.prompting import PromptSet
from pharaoh.utils.io import read_json, read_jsonl



def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--benchmark-config", required=True)
    ap.add_argument("--agent-config", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    bcfg, acfg = load_config(args.benchmark_config), load_config(args.agent_config)
    split_dir = ROOT / require(bcfg, "benchmark.split_dir")
    reservoir = read_json(split_dir / "manifest.json")["reservoir"]
    q = list(read_jsonl(split_dir / f"queries_{args.split}.jsonl"))[args.index]
    ps = PromptSet(ROOT / "prompts")
    objectives = require(bcfg, "benchmark.objectives")
    system = ps.agent_system(reservoir, objectives)
    user = ps.agent_user(q["text"], int(require(acfg, "agent.max_steps")))
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(f"# System\n\n{system}\n\n# User\n\n{user}\n", encoding="utf-8")
    print(f"wrote {out.relative_to(ROOT)} ({len(system) + len(user)} characters)")


if __name__ == "__main__":
    main()
