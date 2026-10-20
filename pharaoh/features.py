"""The 64-dimensional surrogate input vector of a complete scenario (Control node).

Groups (count):
    region rock and size (10)        static properties of the development region
    pattern (8)                      spacing and one-hot pattern type
    placement and completion (15)    well counts and geometry, completion interval, skin, one-hot completion
    completed-interval rock (13)     permeability, porosity, and kh at the well columns, overall and per role
    controls (12)                    injection rate, producer BHP, schedule stages, one-hot schedule
    volumetrics (6)                  pattern pore volume, injected pore volumes, offsets
Permeabilities enter as log10(mD); lengths in meters; areas in km2; volumes in 10^6 m3.
All features are computed from node attributes only (no simulation output).
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

PATTERN_TYPES = ("five_spot", "inverted_five_spot", "nine_spot", "inverted_nine_spot",
                 "line_drive", "staggered_line_drive", "peripheral")
COMPLETIONS = ("all", "upper", "lower")
SCHEDULES = ("constant", "ramp_up", "ramp_down")
MAX_STAGES = 3


def _onehot(prefix: str, value: str, vocab: tuple[str, ...]) -> dict[str, float]:
    if value not in vocab:
        raise ValueError(f"{prefix}: unknown value {value!r}")
    return {f"{prefix}={v}": float(v == value) for v in vocab}


def feature_dict(node: dict[str, Any]) -> dict[str, float]:
    """Named features of one Control node, in the canonical order."""
    if node["type"] != "Control":
        raise ValueError("surrogate features are defined for complete scenarios (Control nodes)")
    a = node["attributes"]
    n_inj, n_prod = a["n_injectors"], a["n_producers"]
    mult = list(a["stage_rate_multipliers"])
    mult = (mult + [mult[-1]] * MAX_STAGES)[:MAX_STAGES]
    total_rate = a["inj_rate_m3d"] * n_inj
    pattern_pv = a["drainage_area_km2"] * 1e6 * a["completed_thickness_m"] * a["poro_mean"] * a["ntg_mean"] / 1e6
    f: dict[str, float] = {}
    # region rock and size (10)
    f.update({
        "region_perm_log10_mean": a["region_perm_log10_mean"], "region_perm_log10_std": a["region_perm_log10_std"],
        "region_perm_log10_p10": a["region_perm_log10_p10"], "region_perm_log10_p90": a["region_perm_log10_p90"],
        "region_poro_mean": a["region_poro_mean"], "region_ntg_mean": a["region_ntg_mean"],
        "region_area_km2": a["area_km2"], "region_pore_volume_mm3": a["region_pore_volume_mm3"],
        "region_centroid_x_km": a["region_centroid_x_m"] / 1e3, "region_centroid_y_km": a["region_centroid_y_m"] / 1e3,
    })
    # pattern (8)
    f["pattern_spacing_m"] = a["pattern_spacing_m"]
    f.update(_onehot("pattern", a["pattern_type"], PATTERN_TYPES))
    # placement and completion (15)
    f.update({
        "n_injectors": n_inj, "n_producers": n_prod, "n_wells": n_inj + n_prod,
        "injector_producer_ratio": n_inj / max(n_prod, 1),
        "well_spacing_min_m": a["well_spacing_m"], "inj_prod_distance_mean_m": a["inj_prod_distance_mean_m"],
        "well_depth_m": a["well_depth_m"], "completed_layers": a["completed_layers"],
        "completed_thickness_m": a["completed_thickness_m"],
        "completion_top_frac": a["completion_top_layer"] / max(a["n_grid_layers"] - 1, 1),
        "completion_bottom_frac": a["completion_bottom_layer"] / max(a["n_grid_layers"] - 1, 1),
        "skin": a["skin"],
    })
    f.update(_onehot("completion", a["completion"], COMPLETIONS))
    # completed-interval rock (13)
    f.update({
        "perm_log10_mean": a["perm_log10_mean"], "perm_log10_std": a["perm_log10_std"],
        "perm_log10_min": a["perm_log10_min"], "perm_log10_max": a["perm_log10_max"],
        "poro_mean": a["poro_mean"], "poro_std": a["poro_std"], "ntg_mean": a["ntg_mean"],
        "injector_perm_log10_mean": a["injector_perm_log10_mean"], "producer_perm_log10_mean": a["producer_perm_log10_mean"],
        "injector_poro_mean": a["injector_poro_mean"], "producer_poro_mean": a["producer_poro_mean"],
        "injector_kh_log10": math.log10(max(a["injector_kh_md_m"], 1e-3)),
        "producer_kh_log10": math.log10(max(a["producer_kh_md_m"], 1e-3)),
    })
    # controls (12)
    f.update({
        "inj_rate_m3d": a["inj_rate_m3d"], "total_inj_rate_m3d": total_rate,
        "inj_rate_per_producer_m3d": total_rate / max(n_prod, 1), "prod_bhp_bar": a["prod_bhp_bar"],
        "n_control_stages": a["n_control_stages"],
        **{f"stage_multiplier_{i + 1}": float(m) for i, m in enumerate(mult)},
        "stage_multiplier_mean": float(np.mean(a["stage_rate_multipliers"])),
    })
    f.update(_onehot("schedule", a["schedule_shape"], SCHEDULES))
    # volumetrics (6)
    f.update({
        "drainage_area_km2": a["drainage_area_km2"], "pattern_pore_volume_mm3": pattern_pv,
        "injected_pattern_pv_per_year": total_rate * 365.25 / max(pattern_pv * 1e6, 1.0),
        "injected_region_pv_per_year": total_rate * 365.25 / max(a["region_pore_volume_mm3"] * 1e6, 1.0),
        "depth_to_top_m": a["depth_to_top_m"], "centroid_offset_m": a["centroid_offset_m"],
    })
    return f


FEATURE_NAMES: tuple[str, ...] = ()


def feature_vector(node: dict[str, Any]) -> np.ndarray:
    """Feature vector in FEATURE_NAMES order (float64, length INPUT_DIM)."""
    f = feature_dict(node)
    return np.array([f[k] for k in FEATURE_NAMES], dtype=float)


def _names() -> tuple[str, ...]:
    a = {k: 1.0 for k in (
        "region_perm_log10_mean", "region_perm_log10_std", "region_perm_log10_p10", "region_perm_log10_p90",
        "region_poro_mean", "region_ntg_mean", "area_km2", "region_pore_volume_mm3", "region_centroid_x_m",
        "region_centroid_y_m", "pattern_spacing_m", "n_injectors", "n_producers", "inj_prod_distance_mean_m",
        "well_spacing_m", "well_depth_m", "completed_layers", "completed_thickness_m", "completion_top_layer",
        "completion_bottom_layer", "n_grid_layers", "skin", "perm_log10_mean", "perm_log10_std", "perm_log10_min",
        "perm_log10_max", "poro_mean", "poro_std", "ntg_mean", "injector_perm_log10_mean", "producer_perm_log10_mean",
        "injector_poro_mean", "producer_poro_mean", "injector_kh_md_m", "producer_kh_md_m", "inj_rate_m3d",
        "prod_bhp_bar", "n_control_stages", "drainage_area_km2", "depth_to_top_m", "centroid_offset_m")}
    a.update({"pattern_type": PATTERN_TYPES[0], "completion": COMPLETIONS[0], "schedule_shape": SCHEDULES[0],
              "stage_rate_multipliers": [1.0]})
    return tuple(feature_dict({"type": "Control", "attributes": a}))


FEATURE_NAMES = _names()
INPUT_DIM = len(FEATURE_NAMES)
