"""Objective values from full-physics field totals.

Inputs are cumulative field totals at report times (FOPT, FWPT, FWIT in sm3),
as written by Eclipse-type simulators (OPM Flow, Eclipse) in the summary
output. Water cut is computed from the production rates of each report
interval, WC(t) = q_w(t) / (q_o(t) + q_w(t)); the reported water cut is the
value of the last interval, and the breakthrough time is the first report time
at which WC reaches the configured threshold.
"""
from __future__ import annotations

from typing import Any

import numpy as np

SM3_TO_BBL = 6.289811


def capex_musd(node: dict[str, Any], well_cost: dict[str, float]) -> float:
    """Drilling and completion cost: per-well fixed cost plus a cost per drilled meter."""
    a = node["attributes"]
    n_wells = a["n_injectors"] + a["n_producers"]
    return n_wells * (float(well_cost["fixed_musd_per_well"]) + float(well_cost["usd_per_meter"]) * a["well_depth_m"] / 1e6)


def water_cut_series(days: np.ndarray, fopt: np.ndarray, fwpt: np.ndarray) -> np.ndarray:
    """WC per report interval from interval-average oil and water production rates."""
    dt = np.diff(np.asarray(days, float), prepend=0.0)
    qo = np.diff(np.asarray(fopt, float), prepend=0.0) / dt
    qw = np.diff(np.asarray(fwpt, float), prepend=0.0) / dt
    total = qo + qw
    return np.divide(qw, total, out=np.zeros_like(total), where=total > 0)


def objectives_from_totals(days: np.ndarray, fopt: np.ndarray, fwpt: np.ndarray, fwit: np.ndarray,
                           capex: float, econ: dict[str, float], breakthrough_wc: float = 0.05) -> dict[str, float]:
    """EUR, discounted NPV (net of CAPEX at time zero), cumulative water, final water cut, breakthrough time."""
    days = np.asarray(days, dtype=float)
    if days.ndim != 1 or len(days) < 2 or days[0] <= 0 or np.any(np.diff(days) <= 0):
        raise ValueError("days must be positive, strictly increasing, with at least two report times")
    for name, series in (("FOPT", fopt), ("FWPT", fwpt), ("FWIT", fwit)):
        vec = np.asarray(series, dtype=float)
        if vec.ndim != 1 or vec.shape != days.shape or not np.all(np.isfinite(vec)):
            raise ValueError(f"{name} must have the same shape as DAYS and contain finite values")
        if np.any(vec < 0) or np.any(np.diff(vec) < -1e-7):
            raise ValueError(f"{name} must be nonnegative and nondecreasing (cumulative field total)")
    if not np.all(np.isfinite(days)) or not np.isfinite(capex) or capex < 0:
        raise ValueError("DAYS and CAPEX must be finite, and CAPEX must be nonnegative")
    if not 0 <= breakthrough_wc <= 1 or float(econ["discount_rate"]) <= -1:
        raise ValueError("Invalid water-cut threshold or discount rate")
    d_oil = np.diff(np.asarray(fopt, float), prepend=0.0) * SM3_TO_BBL
    d_wp = np.diff(np.asarray(fwpt, float), prepend=0.0) * SM3_TO_BBL
    d_wi = np.diff(np.asarray(fwit, float), prepend=0.0) * SM3_TO_BBL
    cash = (d_oil * econ["oil_price_usd_bbl"] - d_wp * econ["water_handling_usd_bbl"]
            - d_wi * econ["water_injection_usd_bbl"])
    disc = (1.0 + float(econ["discount_rate"])) ** (days / 365.25)
    wc = water_cut_series(days, fopt, fwpt)
    hit = np.nonzero(wc >= breakthrough_wc)[0]
    return {
        "eur_mmbbl": float(fopt[-1] * SM3_TO_BBL / 1e6),
        "npv_musd": float((cash / disc).sum() / 1e6 - capex),
        "capex_musd": float(capex),
        "cum_water_mmbbl": float(fwpt[-1] * SM3_TO_BBL / 1e6),
        "water_cut": float(wc[-1]),
        "water_breakthrough_days": float(days[hit[0]]) if len(hit) else float("nan"),
    }
