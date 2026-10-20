"""Export the SCHEDULE section for each scenario (Control node) to run full-physics labels.

Each file defines the scenario's wells (WELSPECS), completions (COMPDAT),
producer bottom-hole pressure (WCONPROD), injector rates per control stage
(WCONINJE), and report steps (TSTEP). It is meant to be INCLUDEd in the
reservoir's base deck after its own SCHEDULE keyword, replacing the original
well definitions. Grid indices are 1-based, as in the simulator input.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def stage_days(horizon_years: float, stage_years: list[float], n_stages: int) -> list[float]:
    """Stage lengths in days; a single-stage schedule spans the whole horizon."""
    total = horizon_years * 365.25
    if n_stages == 1:
        return [total]
    if len(stage_years) != n_stages:
        raise ValueError(f"control_stage_years has {len(stage_years)} entries for a {n_stages}-stage schedule")
    w = np.asarray(stage_years, dtype=float)
    return (total * w / w.sum()).tolist()


def _tstep(days: float, report: float) -> str:
    n = int(days // report)
    rest = round(days - n * report, 3)
    parts = [f"{n}*{report:g}"] if n else []
    if rest > 0:
        parts.append(f"{rest:g}")
    return "TSTEP\n " + " ".join(parts) + " /\n"


def schedule_text(node: dict[str, Any], fp: dict[str, Any]) -> str:
    """SCHEDULE include for one Control node."""
    if node["type"] != "Control":
        raise ValueError(f"{node['id']} is a {node['type']} node; only Control nodes are complete scenarios")
    a = node["attributes"]
    k0, k1 = a["completion_top_layer"] + 1, a["completion_bottom_layer"] + 1
    inj, prod = [], []
    for w in a["wells"]:
        (inj if w["role"] == "I" else prod).append(w)
    names = [(f"INJ{i + 1}", w) for i, w in enumerate(inj)] + [(f"PROD{i + 1}", w) for i, w in enumerate(prod)]
    lines = [f"-- Scenario {node['id']}", "WELSPECS"]
    for name, w in names:
        group, phase = ("INJ", "WATER") if name.startswith("INJ") else ("PROD", "OIL")
        lines.append(f" '{name}' '{group}' {w['i'] + 1} {w['j'] + 1} 1* '{phase}' /")
    lines += ["/", "COMPDAT"]
    for name, w in names:
        lines.append(f" '{name}' {w['i'] + 1} {w['j'] + 1} {k0} {k1} 'OPEN' 1* 1* 0.2 1* {a['skin']:.3f} /")
    lines += ["/", "WCONPROD"]
    for name, _ in names:
        if name.startswith("PROD"):
            lines.append(f" '{name}' 'OPEN' 'BHP' 5* {a['prod_bhp_bar']:g} /")
    lines.append("/")
    out = "\n".join(lines) + "\n"
    days = stage_days(float(fp["horizon_years"]), list(fp["control_stage_years"]), int(a["n_control_stages"]))
    bhp_limit = float(fp["injector_bhp_limit_bar"])
    for mult, d in zip(a["stage_rate_multipliers"], days):
        rate = a["inj_rate_m3d"] * float(mult)
        out += "WCONINJE\n" + "".join(
            f" '{name}' 'WATER' 'OPEN' 'RATE' {rate:.2f} 1* {bhp_limit:g} /\n" for name, _ in names if name.startswith("INJ")
        ) + "/\n"
        out += _tstep(d, float(fp["report_step_days"]))
    return out


def file_name(scenario_id: str) -> str:
    """File-system-safe name for a scenario id."""
    return scenario_id.replace(":", "_") + ".SCH"


def export_schedules(nodes, wanted: set[str], fp: dict[str, Any], out_dir: str | Path) -> int:
    """Write one schedule include per wanted scenario; returns the number written."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(out_dir / "index.tsv", "w", encoding="utf-8") as idx:
        idx.write("scenario_id\tfile\n")
        for node in nodes:
            if node["id"] in wanted:
                (out_dir / file_name(node["id"])).write_text(schedule_text(node, fp), encoding="utf-8")
                idx.write(f"{node['id']}\t{file_name(node['id'])}\n")
                n += 1
    return n
