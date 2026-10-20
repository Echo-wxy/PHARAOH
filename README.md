# PHARAOH

Research-code companion for PHARAOH (physics-informed hierarchical adaptive retrieval for multi-objective reservoir development planning) and the PetroGraph benchmark.

**Release scope (important).** This repository implements graph construction, query/pool generation, scenario and query splits, leakage checks, SCHEDULE export, summary-to-objective conversion, Pareto labeling, feature extraction, and prompt/configuration utilities. **It does not contain a complete executable PHARAOH inference agent, surrogate trainer, baseline implementations, distillation trainer, or a script that reproduces the paper's reported retrieval metrics.** Configurations and prompts describe those experiments; they are not substitutes for their implementations or model checkpoints. See [Reproducibility and provenance](#reproducibility-and-provenance) before citing experimental results.

## Overview

PHARAOH lets a language-model agent search a multi-resolution development knowledge graph. Nodes are scored by a surrogate model that estimates EUR, NPV, CAPEX, and cumulative water production under a Chebyshev multi-objective relevance that excludes infeasible configurations. The agent alternates between global anchoring (`ProxySearch`) and type-filtered hierarchical refinement (`ProxyExpand`), keeps a non-dominated archive of visited plans, and the archives of parallel agents are fused by non-dominated sorting. The research method also describes distillation into a compact student model with a physics-consistency term; that training procedure is not implemented in this release.

PetroGraph turns three public reservoir models into heterogeneous knowledge graphs with the hierarchy Region → Pattern → Well → Control, where every Control node is a complete development plan. The intended method searches a full graph; the provided benchmark builder constructs bounded candidate pools, whose ground-truth Pareto sets require external full-physics simulator runs. Graph construction and benchmark preparation can be run using this code, but the retrieval agent itself is not supplied.

| Component | Location |
|---|---|
| Development knowledge graph G = (V, E, φ_V, φ_E, d_V) | `pharaoh/petrograph/generate.py`, `schema.py`, `regions.py`, `patterns.py` |
| Node attribute vector d_V(v) and node text | `pharaoh/petrograph/schema.py`, `text.py` |
| Surrogate input features (64 per scenario) | `pharaoh/features.py` |
| Multi-objective queries | `prompts/benchmark/query_templates.yaml`, `pharaoh/petrograph/queries.py` |
| Candidate pools and simulation workload | `pharaoh/petrograph/pools.py`, `benchmark.py` |
| Full-physics schedules and objectives (EUR, NPV, CAPEX, water) | `pharaoh/petrograph/schedule.py`, `economics.py` |
| Ground-truth Pareto sets | `pharaoh/petrograph/labels.py`, `pareto.py` |
| Train / validation / test splits, ablation subset, surrogate splits | `pharaoh/splits/make.py` |
| Leakage audit | `pharaoh/splits/leakage.py` |
| Agent prompts and tool schema (`ProxySearch`, `ProxyExpand`, `finish`) | `prompts/agent/`, `pharaoh/prompting.py` |
| Baseline prompts (KAR, Think-on-Graph) | `prompts/baselines/` |
| Surrogate, agent, ablation, distillation, and baseline settings | `configs/` |

## PetroGraph

The following figures are **manuscript/release-description values supplied with the original repository**, not independently reproduced or validated by the files in this archive. The generator writes actual graph and split counts to its output manifests; do not treat these table values as verified outputs until the corresponding manifests, simulator summaries, and evaluation logs are available.

| Reservoir | Source model | Entities | Relations | Simulated scenarios (surrogate training / evaluation candidates) | Candidate pool per query | Mean Pareto set size | Queries (train / val / test) |
|---|---|---|---|---|---|---|---|
| Olympus | TNO OLYMPUS field optimization benchmark | ≈900 K | 8.5 M | 4,800 (3,840 / 960) | 960 | 11.8 | 6,000 / 1,000 / 1,000 |
| Egg | Egg model (channelized, waterflooding) | 1.2 M | 11.2 M | 6,400 (5,120 / 1,280) | 1,280 | 18.6 | 6,000 / 1,000 / 1,000 |
| PUNQ-S3 | PUNQ-S3 (heterogeneous) | 185 K | 9.8 M | 3,600 (2,880 / 720) | 720 | 14.3 | 6,000 / 1,000 / 1,000 |

The scripts record the counts they actually produce in `meta.json` (graph, including the total pore volume of the model) and `manifest.json` (splits). Neither the source reservoir decks nor generated graphs, full-physics results, trained weights, or paper evaluation logs are bundled here.

### Graph

- **Region**: k-means partition of the active grid footprint, with aggregated permeability, porosity, net-to-gross, area, and centroid.
- **Pattern**: a well-pattern type (five-spot, inverted five-spot, nine-spot, inverted nine-spot, direct and staggered line drive, peripheral flood) at a given spacing that fits inside the region.
- **Well**: a concrete injector/producer placement on active columns, with a completion interval (all, upper, or lower layers) and skin.
- **Control**: injection rate per injector, producer bottom-hole pressure, and a constant, ramp-up, or ramp-down schedule.
- Edges: `refines` (parent to child), `combines` (adjacent regions; the same pattern in adjacent regions), `transforms` (alternative completions of a placement; controls differing in one setting).

### Queries and ground truth

Queries are instantiated from property-centric and relation-centric templates. Each query stores its objectives with explicit senses (EUR and NPV maximized; CAPEX and cumulative water minimized), outcome constraints (NPV floor, CAPEX ceiling, water-cut ceiling), and attribute constraints (well spacing, well depth, well count, region, pattern type). For each query, every scenario in its candidate pool is simulated; scenarios violating any constraint are removed, and the non-dominated remainder under the query's objectives is the ground-truth plan set. Hit@1, Recall@20, and MRR are computed against these sets.

Full-physics runs use OPM Flow with 30-day report steps over 20 years (Olympus), 10 years (Egg), and 16.5 years (PUNQ-S3). EUR, cumulative water, and NPV come from the field production and injection totals; CAPEX from a per-well and per-meter drilling cost; water cut from the oil and water production rates of each report interval, WC(t) = q_w(t) / (q_o(t) + q_w(t)), with the final interval used for constraints.

### Splits and leakage control

- **Scenarios.** Before query generation, the configured scenario library is split into a surrogate-training library (80%) and an evaluation candidate library (20%). Scenario ids never overlap, and all control variants of one well placement stay on the same side, so no evaluation scenario has a sibling in the training library that differs only in its injection rate, BHP, or schedule. The **intended training protocol** fits the surrogate and normalisation statistics on the training library only, then reports accuracy on the evaluation library; the actual training/evaluation implementation is not bundled. Validation/test query pools draw from the evaluation library; training pools draw from the training library.
- **Queries.** Queries are grouped by a normalized signature (template, objective senses, weights, constraint fields, thresholds, and units) and by normalized text; a group never crosses train / validation / test. Repeats within a split are allowed.
- **Ablations.** The provided Table 2 configurations refer to a fixed 10% subset of the test split (100 queries per reservoir, `test_subset_qids.txt`); configurations for Tables S1–S3 refer to the full test split. **No corresponding execution/evaluation code is bundled.**
- `scripts/check_leakage.py` checks query/scenario split isolation and the file hashes recorded in `manifest.json`. It does **not** establish whether external simulator outputs are valid or whether the reported paper metrics are reproduced.
- `scripts/label_pools.py` rejects incomplete or mismatched full-physics results instead of silently creating labels from missing scenarios.

## Data

The three reservoir models are obtained from their original providers; see [`data/README.md`](data/README.md) for sources, terms of use, and where to place the files. A small explicitly synthetic grid in the same format is used **only for the quick demonstration and automated tests**, not as evidence for the paper results.

## Repository layout

```
configs/
  reservoirs/     graph generation and full-physics settings per reservoir
  benchmark/      queries, pools, splits, surrogate split per reservoir
  surrogate.yaml  surrogate model
  agent/          full model, ablations (Table 2, S1, S2), compute sweep (S3)
  distill/        student models and budgets (Table 3, S4)
  baselines/      every baseline, with the resources it can access
prompts/
  agent/          system, user, observation, format-error prompts and tool schema
  benchmark/      query templates and paraphrase prompt
  baselines/      KAR expansion and Think-on-Graph prompts
pharaoh/          library code (graph, queries, pools, schedules, labels, splits, prompts)
scripts/          command-line entry points
tests/            unit and end-to-end tests
data/             data sources and placement
```

## Usage

### Quick demo (synthetic fixtures; not a paper reproduction)

```bash
pip install -e .
bash run_demo.sh
```

Builds a graph on a synthetic grid, generates queries, pools, and splits, runs a leakage audit, writes example simulator schedules, produces synthetic run summaries **in place of the external physics simulator**, computes illustrative objectives and labels, renders an agent prompt, and checks configs. Everything in `outputs/demo/` is demonstration output and **must not be reported as physical or measured benchmark evidence**. No real-model claims are tested by this demo.

### Building PetroGraph (Egg shown; Olympus and PUNQ-S3 are analogous)

```bash
# 1. Knowledge graph
python scripts/generate_graph.py --config configs/reservoirs/egg.yaml --out data/petrograph/egg

# 2. Queries, candidate pools, splits, surrogate splits, simulation workload
python scripts/build_benchmark.py --config configs/benchmark/egg.yaml
python scripts/check_leakage.py --split-dir splits/egg

# 3. Simulator schedules for every scenario in splits/egg/scenarios_to_simulate.txt.gz
python scripts/export_schedules.py --reservoir-config configs/reservoirs/egg.yaml \
    --benchmark-config configs/benchmark/egg.yaml --out data/simulations/egg/schedules

# 4. Run each schedule with the independently obtained reservoir base deck and OPM Flow
#    (deck preparation, simulator execution, and compatibility validation are external to this repo).
#    Supply one complete run output for every required scenario; then collect objectives.
python scripts/summarize_runs.py --reservoir-config configs/reservoirs/egg.yaml \
    --benchmark-config configs/benchmark/egg.yaml --runs data/simulations/egg/runs \
    --format unsmry --out data/simulations/egg_results.csv

# 5. Ground-truth Pareto sets
python scripts/label_pools.py --config configs/benchmark/egg.yaml --results data/simulations/egg_results.csv
```

The generated schedule include is intended to be used in the reservoir base deck after its `SCHEDULE` keyword, replacing the original well definitions; this **requires manual adaptation and simulator validation for the specific realization/deck**. This repository does not run or validate OPM Flow jobs. Incomplete runs fail by default; `scripts/summarize_runs.py --allow-incomplete` is for diagnostics only, and the labeling script rejects partial outcomes. `--format unsmry` requires the optional `resdata` package; with `--format csv`, provide one CSV per run with columns `DAYS, FOPT, FWPT, FWIT`. The results file also records the water breakthrough time (first report at which water cut reaches 5%).

### Split files

| File | Content |
|---|---|
| `queries_{train,val,test}.jsonl` | `qid`, `text`, `category`, `template_id`, `objectives` (name, sense), `outcome_constraints`, `attribute_constraints`, `reference_weights`, `signature` |
| `pools_{train,val,test}.jsonl.gz` | `qid`, `scenario_ids` (Control node ids) |
| `labels_{train,val,test}.jsonl` | `qid`, `pareto_ids`, `n_pool`, `n_feasible`, `n_missing_results` |
| `test_subset_qids.txt` | fixed 10% test subset used for Table 2 |
| `surrogate_train_scenarios.txt.gz` | training library: surrogate fitting and normalization statistics, training-query pools |
| `eval_candidate_scenarios.txt.gz` | evaluation library: validation/test pools and surrogate accuracy |
| `scenarios_to_simulate.txt.gz` | every scenario that needs a full-physics run (both libraries) |
| `manifest.json` | policies, counts, and SHA-256 of every file |

### Prompts

`prompts/agent/system.md` defines the graph, the objectives, the subquery format (weights, reference point, constraints), the tools, and the output format (`<think>…</think>` followed by one `<tool_call>{…}</tool_call>`). The tool schema is `prompts/agent/tools.json`. `pharaoh/prompting.py` renders the prompts and validates every tool call. To inspect the exact prompt for a query:

```bash
python scripts/render_prompts.py --benchmark-config configs/benchmark/egg.yaml \
    --agent-config configs/agent/pharaoh.yaml --split test --index 0 --out outputs/prompt_egg.md
```

### Configurations

| Experiment | Configs |
|---|---|
| Main comparison (Table 1) | `configs/agent/pharaoh.yaml`, `configs/baselines/*.yaml`, `configs/distill/qwen2.5-7b_6000q.yaml` |
| Toolset ablation (Table 2) | `configs/agent/ablation_full_on_subset.yaml`, `ablation_no_proxyexpand.yaml`, `ablation_no_query_ranking.yaml`, `ablation_no_type_filter.yaml` |
| Relevance function (Table S1, full test split) | `configs/agent/relevance_bm25.yaml` |
| Aggregation strategy (Table S2, full test split) | `configs/agent/aggregation_{voting,ordering,random}.yaml` |
| Compute–performance trade-off (Table S3, full test split) | `configs/agent/compute_grid.yaml` |
| Distillation budget and student size (Table 3) | `configs/distill/qwen2.5-{7b,14b}_{600q,6000q}.yaml` |
| Distillation dynamics (Table S4) | `configs/distill/budget_sweep.yaml`, `qwen2.5-7b_6000q_lambda0.yaml` |
| Surrogate accuracy (Table S6) | `configs/surrogate.yaml` (fitted on the training library, evaluated on the evaluation library) |

The experiment **configurations specify** three independent run seeds and per-query logging fields (LLM calls, prompt and completion tokens, surrogate calls, GPU time, API time, and wall-clock time). No agent runner or logging pipeline to produce those records is included.

The **surrogate architecture is specified in configuration** as an MLP (64 → 256 → 256 → 128 → 64) with four main heads (EUR, NPV, CAPEX, cumulative water) and an auxiliary water-cut head, but no model training or inference code is provided. Its intended 64 inputs (`pharaoh/features.py`) describe a complete scenario: region rock statistics and size (10), pattern spacing and type (8), well counts, geometry, and completion (15), permeability, porosity, and kh over the completed interval at injectors and producers (13), injection rate, producer BHP, and schedule stages (12), and pattern volumetrics such as drainage area, pattern pore volume, and injected pore volumes per year (6). The paper describes standardising inputs/outputs with training-library statistics, resolving node scores to scenarios without duplicates, summing the objectives of distinct combined scenarios, and Pareto fusion (Eq. 7). **These agent-side operations are not implemented in this code release.**

Configs inherit from a `_base_` file and override only what differs. `python scripts/check_configs.py --resources` loads every config and prints which resources each method can access (node text, physical attributes, surrogate scores, feasibility rules, LLM backbone, step budget, training on the target graph).

## Values mentioned in the supplied manuscript description (not reproduced here)

The following figures were included in the source README as manuscript results (averages over Olympus, Egg, and PUNQ-S3). **Their underlying run logs, external solver results, and executable agent/distillation implementations are not present, so this repository cannot verify them.** They should be checked against the final paper and its archived experiment artifacts before public claims of exact reproducibility.

| Method | Hit@1 | R@20 | MRR |
|---|---|---|---|
| PHARAOH (GPT-4.1, T_max = 20, n = 3) | 60.7 | 76.3 | 68.1 |
| PHARAOH-distilled (Qwen2.5-7B, 6,000 queries) | 51.7 | 72.0 | 60.3 |

## Dependencies

Python ≥ 3.9 with `numpy` and `pyyaml` (`python -m pip install -e .`). Tests: `python -m pip install -e ".[dev]" && python -m pytest -q`. Reading simulator summary files directly needs optional `resdata`. Producing physical production histories requires independently installed and validated OPM Flow (or another compatible simulator), plus the original reservoir model files. The supplied **Python utilities** run on a CPU; the external simulator and the implementations required to generate the paper's results are **not included**. GitHub Actions runs unit tests and the isolated synthetic-fixture smoke test on Python 3.9, 3.11 and 3.12.

## Reproducibility and provenance

- **Published models, not measured field data:** OLYMPUS, Egg and PUNQ-S3 are published numerical reservoir-model benchmarks. Inputs originate from their providers; predicted production histories are **simulator outputs**, not physically measured field histories. This distinction must be preserved in research reporting.
- **Synthetic fixtures:** `pharaoh/synthetic.py`, `scripts/demo_synthetic_results.py`, and `configs/*/demo.yaml` are intentionally labelled as demonstration/test fixtures. They do not supply labels for the paper's real-model experiments.
- **Not bundled:** provider reservoir decks, reservoir simulator and deck-specific integration, generated graphs/splits, full-physics runs, trained surrogate and student checkpoints, agent runtime, baseline/distillation implementations and raw paper evaluation logs. Configurations alone cannot reproduce Tables 1–3 and S1–S6.
- **Approximate geometry:** `pharaoh/grdecl.py` reads a subset of Eclipse-style property keywords; it does not interpret `COORD`/`ZCORN`, `BOX`, or `EDIT` modifiers. Cell sizes and datum depth are supplied by configuration and may be approximations. Independently verify these assumptions against each source reservoir model before publishing physical conclusions.
- **Checks and limits:** Run `python -m pytest -q` and `python scripts/check_configs.py --resources`; when using external models, compare actual `meta.json` and `manifest.json` to expected counts, confirm solver-version and deck compatibility, run every planned scenario, and audit final evaluation logs. Passing the included synthetic-fixture test does not verify solver physics or an absent training/inference pipeline.
- **Citation metadata:** `CITATION.cff` contains a generic contributor name because the archive does not identify the paper's authors. Replace it with the actual paper author list and DOI, if applicable, before the official publication release.

## License

MIT; see [`LICENSE`](LICENSE). The reservoir models are subject to their providers' terms (see `data/README.md`).

## Citation

See [`CITATION.cff`](CITATION.cff).
