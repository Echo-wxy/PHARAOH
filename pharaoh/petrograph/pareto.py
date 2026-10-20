"""Pareto dominance with explicit objective senses."""
from __future__ import annotations

import numpy as np


def to_maximization(values: np.ndarray, senses: list[str]) -> np.ndarray:
    """Flip 'min' columns so that larger is better in every column."""
    sign = np.array([1.0 if s == "max" else -1.0 for s in senses])
    if any(s not in ("max", "min") for s in senses):
        raise ValueError(f"senses must be 'max' or 'min', got {senses}")
    return values * sign


def non_dominated_mask(values: np.ndarray, senses: list[str]) -> np.ndarray:
    """Boolean mask of rows not dominated by any other row.

    Row u dominates row v if u is no worse in every objective and strictly
    better in at least one (after converting all objectives to maximization).
    Duplicate rows do not dominate each other.
    """
    v = to_maximization(np.asarray(values, dtype=float), senses)
    n = len(v)
    if n == 0:
        return np.zeros(0, dtype=bool)
    ge = (v[:, None, :] >= v[None, :, :]).all(-1)
    gt = (v[:, None, :] > v[None, :, :]).any(-1)
    dominated = (ge & gt).any(0)
    return ~dominated
