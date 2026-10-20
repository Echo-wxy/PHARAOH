"""Synthetic stand-ins used only by the quick demo and the tests.

`synthetic_grid` produces a channelized or Gaussian heterogeneous grid in the
same GRDECL format as the real reservoir models. `synthetic_run`
produces demo field totals (FOPT, FWPT, FWIT over time) for a scenario
so that summarizing and ground-truth labeling can be exercised end to end; it
is not a reservoir simulator and its numbers are illustrative only.
"""
from __future__ import annotations

import numpy as np

from pharaoh.grdecl import GridProperties
from pharaoh.utils.seed import derive_rng

DEMO_NOTICE = "Synthetic demo data; numbers are illustrative only."


def _smooth_field(rng: np.random.Generator, nx: int, ny: int, corr: float) -> np.ndarray:
    """Gaussian random field via spectral filtering (unit variance)."""
    white = rng.standard_normal((nx, ny))
    kx = np.fft.fftfreq(nx)[:, None]
    ky = np.fft.fftfreq(ny)[None, :]
    filt = np.exp(-0.5 * (kx**2 + ky**2) * (2 * np.pi * corr) ** 2)
    f = np.real(np.fft.ifft2(np.fft.fft2(white) * filt))
    return (f - f.mean()) / (f.std() + 1e-12)


def synthetic_grid(nx: int, ny: int, nz: int, seed: int, style: str = "channelized") -> GridProperties:
    """Heterogeneous permeability/porosity grid with an irregular active boundary."""
    rng = derive_rng(seed, "synthetic_grid", style)
    logk = np.zeros((nx, ny, nz))
    for k in range(nz):
        base = _smooth_field(rng, nx, ny, corr=max(nx, ny) / 8)
        if style == "channelized":
            ii = np.arange(nx)[:, None] / nx
            jj = np.arange(ny)[None, :] / ny
            phase = rng.uniform(0, 2 * np.pi)
            channel = np.cos(2 * np.pi * (2.5 * jj + 0.6 * np.sin(2 * np.pi * ii + phase)))
            logk[:, :, k] = np.where(channel > 0.55, 3.4, 1.3) + 0.3 * base
        else:
            logk[:, :, k] = 2.0 + 0.6 * base
    permx = 10.0 ** logk
    poro = np.clip(0.08 + 0.05 * (logk - 1.0) + 0.01 * rng.standard_normal(logk.shape), 0.03, 0.35)
    edge = _smooth_field(rng, nx, ny, corr=max(nx, ny) / 5)
    ii, jj = np.meshgrid(np.linspace(-1, 1, nx), np.linspace(-1, 1, ny), indexing="ij")
    active2d = (ii**2 + jj**2) < 0.85 + 0.12 * edge
    actnum = np.repeat(active2d[:, :, None], nz, axis=2)
    ntg = np.clip(0.85 + 0.1 * rng.standard_normal(logk.shape), 0.4, 1.0)
    return GridProperties(nx=nx, ny=ny, nz=nz, permx=permx, poro=poro, actnum=actnum, ntg=ntg)


def synthetic_outcomes(node: dict, seed: int) -> dict[str, float]:
    """Demo objective values for one Control (scenario) node.

    Uses simple monotone relations between attributes and outcomes with
    multiplicative noise so that Pareto sets are non-trivial.
    """
    a = node["attributes"]
    rng = derive_rng(seed, "synthetic_outcomes", node["id"])
    n_wells = a["n_injectors"] + a["n_producers"]
    kh = a["perm_mean_md"] * a["ntg_mean"] * a["completed_layers"] / 1000.0
    rate = a["inj_rate_m3d"] * a["n_injectors"] / 1000.0
    sweep = 1.0 - np.exp(-a["well_spacing_m"] / 150.0)
    eur = 8.0 * (kh ** 0.3) * (rate ** 0.4) * sweep * (1.0 + 0.04 * (a["n_control_stages"] - 1))
    eur *= np.exp(0.10 * rng.standard_normal())
    capex = n_wells * (4.0 + 0.003 * a["well_depth_m"]) * np.exp(0.04 * rng.standard_normal())
    water_cut = float(np.clip(0.08 + 0.15 * rate ** 0.7 / max(sweep, 0.1) + 0.04 * rng.standard_normal(), 0.02, 0.95))
    cum_water = eur * water_cut / max(1.0 - water_cut, 0.05)
    npv = 45.0 * eur - capex - 3.0 * cum_water
    return {
        "eur_mmbbl": float(eur),
        "npv_musd": float(npv),
        "capex_musd": float(capex),
        "cum_water_mmbbl": float(cum_water),
        "water_cut": water_cut,
    }


def synthetic_run(node: dict, horizon_years: float, report_step_days: float, seed: int) -> dict[str, np.ndarray]:
    """Demo field totals in sm3 at report times, shaped like simulator summary vectors."""
    o = synthetic_outcomes(node, seed)
    days = np.arange(report_step_days, horizon_years * 365.25 + 1e-9, report_step_days)
    tau = days[-1] / 3.0
    shape = (1.0 - np.exp(-days / tau)) / (1.0 - np.exp(-days[-1] / tau))
    fopt = o["eur_mmbbl"] * 1e6 / 6.289811 * shape
    t_bt = 0.25 * days[-1]
    fwct = np.clip(o["water_cut"] * (days - t_bt) / (days[-1] - t_bt), 0.0, None)
    oil_rate = np.diff(fopt, prepend=0.0)
    wat_rate = oil_rate * fwct / np.maximum(1.0 - fwct, 0.05)
    fwpt = np.cumsum(wat_rate)
    fwit = np.cumsum(1.05 * (oil_rate + wat_rate))
    return {"DAYS": days, "FOPT": fopt, "FWPT": fwpt, "FWIT": fwit}
