# Data

PetroGraph is designed to be generated from three published reservoir simulation benchmark models. These are numerical reservoir-model inputs, **not measured oil-field production records**; the derived production histories require an external numerical simulator. The model files are not redistributed here; download them from the original sources and place them as shown. Each file is subject to its provider's terms of use.

| Model | Original source | Place at | Config |
|---|---|---|---|
| Olympus | Fonseca, R. M., Della Rossa, E., Emerick, A. A., Hanea, R. G., Jansen, J. D. (2020). Introduction to the special issue: Overview of OLYMPUS Optimization Benchmark Challenge. *Computational Geosciences* 24, 1933–1941. https://doi.org/10.1007/s10596-020-10003-4 (model files are distributed through the OLYMPUS challenge) | `data/raw/olympus/` (point `reservoir.grid.path` at the simulation deck of the realization used) | `configs/reservoirs/olympus.yaml` |
| Egg | Jansen, J. D. et al. (2014). The egg model – a geological ensemble for reservoir simulation. *Geoscience Data Journal* 1(2), 192–195. Data files: https://doi.org/10.4121/uuid:916c86cd-3558-4672-829a-105c62985ab2 (4TU General Terms of Use) | `data/raw/egg/` (point `reservoir.grid.path` at the Eclipse deck) | `configs/reservoirs/egg.yaml` |
| PUNQ-S3 | Floris, F. J. T., Bush, M. D., Cuypers, M., Roggero, F., Syversveen, A.-R. (2001). Methods for quantifying the uncertainty of production forecasts: a comparative study. *Petroleum Geoscience* 7, S87–S96. | `data/raw/punq_s3/` (point `reservoir.grid.path` at the Eclipse deck) | `configs/reservoirs/punq_s3.yaml` |

The grid reader follows `INCLUDE` statements, so `reservoir.grid.path` may point at the main `.DATA` deck. It reads `SPECGRID`/`DIMENS`, `PERMX`, `PORO`, `NTG`, and `ACTNUM`. It **does not** parse corner-point geometry (`COORD`/`ZCORN`), `BOX`/`EDIT` changes or all deck semantics; cell sizes and top depth are provided by the config (some are marked approximate). It is essential to cross-check extracted properties and numerical-grid geometry against the original provider files before making physical claims.

## Generated files

| Path | Produced by | Content |
|---|---|---|
| `data/petrograph/<reservoir>/` | `scripts/generate_graph.py` | `nodes.jsonl.gz`, `edges.tsv.gz`, `control_ids.txt.gz`, `meta.json` |
| `splits/<reservoir>/` | `scripts/build_benchmark.py` | queries, candidate pools, ablation subset, surrogate scenario splits, simulation workload, `manifest.json` |
| `splits/<reservoir>/labels_*.jsonl` | `scripts/label_pools.py` | ground-truth Pareto sets from full-physics results |

### Full-physics results format

`scripts/summarize_runs.py` writes, and `scripts/label_pools.py` reads, one CSV per reservoir with a `scenario_id` column (Control node id) and the columns `eur_mmbbl`, `npv_musd`, `capex_musd`, `cum_water_mmbbl`, `water_cut` (water cut of the last report interval, from production rates), and `water_breakthrough_days`. Every id listed in `splits/<reservoir>/scenarios_to_simulate.txt.gz` should have one row.

## Data provenance and validation

Generated graphs, scenario IDs and label CSVs must be traceable to the exact provider deck, realization, numerical-simulator version, run settings, and outputs. For each published experiment, retain source-deck checksum, solver/version, run exit status and corresponding `.UNSMRY` or CSV histories. The quick demonstration uses synthetic data and is **not** a substitute for these artifacts. Incomplete or corrupt physical outputs must not be silently classified as infeasible configurations.
