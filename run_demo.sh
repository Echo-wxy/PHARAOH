#!/usr/bin/env bash
# Quick demo on a small synthetic grid: graph -> benchmark/splits -> leakage audit
# -> simulator schedules -> synthetic run summaries -> objectives -> ground-truth labels -> rendered prompts -> config check.
set -euo pipefail
cd "$(dirname "$0")"
OUT=outputs/demo
rm -rf "$OUT"
python scripts/generate_graph.py --config configs/reservoirs/demo.yaml --out "$OUT/graph"
python scripts/build_benchmark.py --config configs/benchmark/demo.yaml
python scripts/check_leakage.py --split-dir "$OUT/splits" --json "$OUT/leakage_report.json"
python scripts/export_schedules.py --reservoir-config configs/reservoirs/demo.yaml \
    --benchmark-config configs/benchmark/demo.yaml --out "$OUT/schedules"
python scripts/demo_synthetic_results.py --reservoir-config configs/reservoirs/demo.yaml \
    --benchmark-config configs/benchmark/demo.yaml --out "$OUT/runs"
python scripts/summarize_runs.py --reservoir-config configs/reservoirs/demo.yaml \
    --benchmark-config configs/benchmark/demo.yaml --runs "$OUT/runs" --out "$OUT/results.csv"
python scripts/label_pools.py --config configs/benchmark/demo.yaml --results "$OUT/results.csv"
python scripts/render_prompts.py --benchmark-config configs/benchmark/demo.yaml \
    --agent-config configs/agent/pharaoh.yaml --split test --index 0 --out "$OUT/prompt_example.md"
python scripts/check_configs.py --resources > "$OUT/config_check.txt"
echo "Synthetic demo data; numbers are illustrative only."
echo "Outputs in $OUT/"
