"""Natural-language node descriptions used by lexical and dense retrievers."""
from __future__ import annotations

from typing import Any

from pharaoh.petrograph.patterns import PATTERN_LABELS


def _fmt(v: Any, nd: int = 1) -> str:
    return f"{v:.{nd}f}" if isinstance(v, float) else str(v)


def node_text(node: dict[str, Any]) -> str:
    """Concatenate type and physical attributes into a short description.

    Objective values (EUR, NPV, CAPEX, water) are never included, so text
    retrievers see the same information as the surrogate's inputs.
    """
    a = node["attributes"]
    t = node["type"]
    parts = [f"{t} node {node['id']} in {node['region_name']}."]
    parts.append(
        f"Mean permeability {_fmt(a['perm_mean_md'])} mD (log10 std {_fmt(a['perm_log10_std'], 2)}), "
        f"porosity {_fmt(a['poro_mean'], 3)}, net-to-gross {_fmt(a['ntg_mean'], 2)}."
    )
    if t == "Region":
        parts.append(f"{a['active_cells']} active cells over {_fmt(a['area_km2'], 2)} km2.")
    if t in ("Pattern", "Well", "Control"):
        parts.append(f"{PATTERN_LABELS[a['pattern_type']].capitalize()} pattern with {_fmt(a['pattern_spacing_m'], 0)} m spacing.")
    if t in ("Well", "Control"):
        parts.append(
            f"{a['n_injectors']} injectors and {a['n_producers']} producers, minimum well spacing "
            f"{_fmt(a['well_spacing_m'], 0)} m, completed in layers {a['completion_top_layer'] + 1}-"
            f"{a['completion_bottom_layer'] + 1} ({a['completion']}), total depth {_fmt(a['well_depth_m'], 0)} m, skin {_fmt(a['skin'], 1)}."
        )
    if t == "Control":
        parts.append(
            f"Injection {_fmt(a['inj_rate_m3d'], 0)} m3/d per injector, producer BHP {_fmt(a['prod_bhp_bar'], 0)} bar, "
            f"{a['n_control_stages']}-stage {a['schedule_shape'].replace('_', '-')} schedule."
        )
    return " ".join(parts)
