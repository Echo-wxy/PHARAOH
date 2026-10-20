"""Build a hierarchical development knowledge graph G = (V, E, phi_V, phi_E, d_V).

Hierarchy (refines edges, parent -> child):
    Region   - a k-means partition of the active grid footprint
    Pattern  - a well-pattern type at a given spacing that fits in the region
    Well     - a concrete injector/producer placement plus completion interval
    Control  - an injection-rate / producer-BHP / schedule setting; every
               Control node is a complete development scenario
Lateral edges:
    combines   - adjacent regions; same pattern type and spacing in adjacent regions
    transforms - Well placements differing only in completion interval;
                 Controls under the same placement differing in exactly one control factor

Nodes are streamed to `nodes.jsonl.gz`, edges to `edges.tsv.gz`
(src, dst, type; refines is directed, the others are stored once with src < dst).
"""
from __future__ import annotations

import csv
import gzip
import itertools
import json
import logging
import math
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from pharaoh import __version__
from pharaoh.config import config_hash, require
from pharaoh.grdecl import GridProperties, read_grdecl, write_grdecl
from pharaoh.petrograph.patterns import PATTERN_TEMPLATES, rotate
from pharaoh.petrograph.regions import kmeans_columns, region_adjacency, region_names
from pharaoh.petrograph.text import node_text
from pharaoh.synthetic import synthetic_grid
from pharaoh.utils.io import open_gz_text_write, sha256_file, write_json, write_lines
from pharaoh.utils.seed import derive_rng

log = logging.getLogger(__name__)


def load_grid(cfg: dict[str, Any], root: Path) -> tuple[GridProperties, str]:
    """Load the reservoir grid named in the config; returns (grid, provenance)."""
    g = require(cfg, "reservoir.grid")
    if g["source"] == "grdecl":
        path = root / require(cfg, "reservoir.grid.path")
        return read_grdecl(path), f"grdecl:{path.name}:sha256={sha256_file(path)}"
    if g["source"] == "synthetic":
        s = g["synthetic"]
        grid = synthetic_grid(s["nx"], s["ny"], s["nz"], seed=s["seed"], style=s.get("style", "channelized"))
        return grid, f"synthetic:{s['nx']}x{s['ny']}x{s['nz']}:seed={s['seed']}"
    raise ValueError(f"Unknown grid source {g['source']!r}")


def _static_stats(grid: GridProperties, cols_i: np.ndarray, cols_j: np.ndarray, k0: int, k1: int) -> dict[str, float]:
    act = grid.actnum[cols_i, cols_j, k0:k1 + 1]
    if not act.any():
        return {"perm_mean_md": math.nan, "perm_log10_std": math.nan, "poro_mean": math.nan, "ntg_mean": math.nan}
    perm = grid.permx[cols_i, cols_j, k0:k1 + 1][act]
    return {
        "perm_mean_md": float(perm.mean()),
        "perm_log10_std": float(np.log10(np.maximum(perm, 1e-6)).std()),
        "poro_mean": float(grid.poro[cols_i, cols_j, k0:k1 + 1][act].mean()),
        "ntg_mean": float(grid.ntg[cols_i, cols_j, k0:k1 + 1][act].mean()),
    }


def _log10(x: np.ndarray) -> np.ndarray:
    return np.log10(np.maximum(x, 1e-6))


def _region_stats(grid: GridProperties, ri: np.ndarray, rj: np.ndarray, cell_volume: float) -> dict[str, float]:
    """Region-level rock statistics over all active cells of the region's columns."""
    act = grid.actnum[ri, rj, :]
    lk = _log10(grid.permx[ri, rj, :][act])
    poro, ntg = grid.poro[ri, rj, :][act], grid.ntg[ri, rj, :][act]
    return {
        "region_perm_log10_mean": float(lk.mean()),
        "region_perm_log10_std": float(lk.std()),
        "region_perm_log10_p10": float(np.quantile(lk, 0.1)),
        "region_perm_log10_p90": float(np.quantile(lk, 0.9)),
        "region_poro_mean": float(poro.mean()),
        "region_ntg_mean": float(ntg.mean()),
        "region_pore_volume_mm3": float((poro * ntg).sum() * cell_volume / 1e6),
    }


def _hull_area(xy: np.ndarray) -> float:
    """Area of the convex hull of 2-D points (monotone chain); 0 for degenerate sets."""
    pts = sorted(map(tuple, xy.tolist()))
    if len(pts) < 3:
        return 0.0

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: list = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper: list = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    h = lower[:-1] + upper[:-1]
    return 0.5 * abs(sum(h[i][0] * h[(i + 1) % len(h)][1] - h[(i + 1) % len(h)][0] * h[i][1] for i in range(len(h))))


def _well_stats(grid: GridProperties, wells: list[tuple[int, int, str]], k0: int, k1: int, dz: float) -> dict[str, float]:
    """Completed-interval statistics at the well columns, overall and per role."""
    out: dict[str, float] = {}
    allk: list[float] = []
    allp: list[float] = []
    for role, tag in (("I", "injector"), ("P", "producer")):
        lk_r, poro_r, kh_r = [], [], []
        for wi, wj, rr in wells:
            if rr != role:
                continue
            act = grid.actnum[wi, wj, k0:k1 + 1]
            perm = grid.permx[wi, wj, k0:k1 + 1][act]
            if perm.size:
                lk_r.extend(_log10(perm).tolist())
                poro_r.extend(grid.poro[wi, wj, k0:k1 + 1][act].tolist())
                kh_r.append(float((perm * grid.ntg[wi, wj, k0:k1 + 1][act]).sum() * dz))
            else:
                kh_r.append(0.0)
        out[f"{tag}_perm_log10_mean"] = float(np.mean(lk_r)) if lk_r else math.nan
        out[f"{tag}_poro_mean"] = float(np.mean(poro_r)) if poro_r else math.nan
        out[f"{tag}_kh_md_m"] = float(np.mean(kh_r)) if kh_r else 0.0
        allk += lk_r
        allp += poro_r
    out["perm_log10_mean"] = float(np.mean(allk)) if allk else math.nan
    out["perm_log10_min"] = float(np.min(allk)) if allk else math.nan
    out["perm_log10_max"] = float(np.max(allk)) if allk else math.nan
    out["poro_std"] = float(np.std(allp)) if allp else math.nan
    return out


def _completion_intervals(nz: int, options: list[str]) -> list[tuple[str, int, int]]:
    out = []
    half = max(nz // 2, 1)
    for opt in options:
        if opt == "all":
            out.append(("all", 0, nz - 1))
        elif opt == "upper" and nz > 1:
            out.append(("upper", 0, half - 1))
        elif opt == "lower" and nz > 1:
            out.append(("lower", half, nz - 1))
    return out


class _EdgeWriter:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._f = open_gz_text_write(path, newline="")
        self._w = csv.writer(self._f, delimiter="\t")
        self._w.writerow(["src", "dst", "type"])
        self.counts = {"refines": 0, "combines": 0, "transforms": 0}

    def add(self, src: str, dst: str, etype: str) -> None:
        if etype != "refines" and dst < src:
            src, dst = dst, src
        self._w.writerow([src, dst, etype])
        self.counts[etype] += 1

    def close(self) -> None:
        self._f.close()


def generate_graph(cfg: dict[str, Any], out_dir: str | Path, root: str | Path = ".") -> dict[str, Any]:
    """Generate one PetroGraph subgraph from a reservoir config; returns the metadata dict."""
    root, out_dir = Path(root), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    res = require(cfg, "reservoir")
    gcfg = require(cfg, "graph")
    seed = int(require(cfg, "graph.seed"))
    prefix = res["prefix"]
    cell = res["grid"]["cell_size_m"]
    dx, dy, dz = float(cell["dx"]), float(cell["dy"]), float(cell["dz"])
    top = float(res["grid"]["top_depth_m"])

    grid, provenance = load_grid(cfg, root)
    if res["grid"]["source"] == "synthetic":
        write_grdecl(out_dir / "grid.grdecl", grid)
    active = grid.active_columns
    labels = kmeans_columns(active, int(gcfg["regions"]["n_regions"]), seed, int(gcfg["regions"].get("kmeans_iters", 50)))
    names = region_names(labels)
    adjacency = region_adjacency(labels)

    pat_types: list[str] = list(gcfg["patterns"]["types"])
    spacings: list[float] = [float(s) for s in gcfg["patterns"]["spacing_m"]]
    n_place = int(gcfg["wells"]["placements_per_pattern"])
    max_attempts = int(gcfg["wells"].get("max_placement_attempts", 200))
    completions = _completion_intervals(grid.nz, list(gcfg["wells"]["completion_options"]))
    skin_lo, skin_hi = gcfg["wells"]["skin_range"]
    rates = [float(x) for x in gcfg["controls"]["inj_rate_m3d"]]
    bhps = [float(x) for x in gcfg["controls"]["prod_bhp_bar"]]
    schedules = list(gcfg["controls"]["schedules"])

    nodes_path = out_dir / "nodes.jsonl.gz"
    edges = _EdgeWriter(out_dir / "edges.tsv.gz")
    counts = {t: 0 for t in ("Region", "Pattern", "Well", "Control")}
    control_ids: list[str] = []
    pattern_keys: dict[int, dict[tuple[str, float], str]] = {}
    quant: dict[str, list[float]] = {"well_spacing_m": [], "well_depth_m": [], "n_wells": [], "inj_rate_m3d": [], "prod_bhp_bar": []}


    fnodes = open_gz_text_write(nodes_path)

    def emit(node: dict[str, Any]) -> None:
        node["text"] = node_text(node)
        fnodes.write(json.dumps(node, sort_keys=True) + "\n")
        counts[node["type"]] += 1

    for r in range(int(labels.max()) + 1):
        ri, rj = np.nonzero(labels == r)
        rid = f"{prefix}:R{r:02d}"
        rstat = _static_stats(grid, ri, rj, 0, grid.nz - 1)
        r_attr = {
            **rstat,
            **_region_stats(grid, ri, rj, dx * dy * dz),
            "active_cells": int(grid.actnum[ri, rj, :].sum()),
            "area_km2": float(len(ri) * dx * dy / 1e6),
            "centroid_x_m": float((ri.mean() + 0.5) * dx),
            "centroid_y_m": float((rj.mean() + 0.5) * dy),
        }
        r_attr["region_centroid_x_m"] = r_attr["centroid_x_m"]
        r_attr["region_centroid_y_m"] = r_attr["centroid_y_m"]
        emit({"id": rid, "type": "Region", "level": 0, "region": rid, "region_name": names[r], "attributes": r_attr})
        region_mask = labels == r
        pattern_keys[r] = {}
        p_idx = 0
        for ptype, spacing in itertools.product(pat_types, spacings):
            rng = derive_rng(seed, "placement", rid, ptype, str(spacing))
            step_i, step_j = spacing / dx, spacing / dy
            placements: list[list[tuple[int, int, str]]] = []
            seen: set[tuple] = set()
            for _ in range(max_attempts):
                if len(placements) >= n_place:
                    break
                a = rng.integers(len(ri))
                ai, aj = int(ri[a]), int(rj[a])
                wells = []
                ok = True
                for ox, oy, role in rotate(PATTERN_TEMPLATES[ptype], int(rng.integers(4))):
                    wi, wj = int(round(ai + ox * step_i)), int(round(aj + oy * step_j))
                    if not (0 <= wi < grid.nx and 0 <= wj < grid.ny) or not active[wi, wj] or not region_mask[wi, wj]:
                        ok = False
                        break
                    wells.append((wi, wj, role))
                key = tuple(sorted(wells))
                if ok and len({(w[0], w[1]) for w in wells}) == len(wells) and key not in seen:
                    seen.add(key)
                    placements.append(wells)
            if not placements:
                continue
            pid = f"{rid}:P{p_idx:02d}"
            p_idx += 1
            p_attr = {**r_attr, "pattern_type": ptype, "pattern_spacing_m": spacing}
            emit({"id": pid, "type": "Pattern", "level": 1, "region": rid, "region_name": names[r], "attributes": p_attr})
            edges.add(rid, pid, "refines")
            pattern_keys[r][(ptype, spacing)] = pid
            w_idx = 0
            for pl_i, wells in enumerate(placements):
                wi = np.array([w[0] for w in wells])
                wj = np.array([w[1] for w in wells])
                xy = np.stack([(wi + 0.5) * dx, (wj + 0.5) * dy], 1)
                dmin = float(min(np.hypot(*(xy[a] - xy[b])) for a, b in itertools.combinations(range(len(xy)), 2)))
                roles = np.array([w[2] for w in wells])
                d_ip = [float(np.hypot(*(xy[a] - xy[b]))) for a in np.nonzero(roles == "I")[0] for b in np.nonzero(roles == "P")[0]]
                hull_km2 = max(_hull_area(xy), spacing * spacing) / 1e6
                c_off = float(np.hypot(xy[:, 0].mean() - r_attr["centroid_x_m"], xy[:, 1].mean() - r_attr["centroid_y_m"]))
                skin = float(derive_rng(seed, "skin", pid, str(pl_i)).uniform(skin_lo, skin_hi))
                sibling_wells: list[str] = []
                for comp, k0, k1 in completions:
                    wid = f"{pid}:W{w_idx:03d}"
                    w_idx += 1
                    w_attr = {
                        **p_attr,
                        **_static_stats(grid, wi, wj, k0, k1),
                        "area_km2": r_attr["area_km2"], "active_cells": r_attr["active_cells"],
                        "centroid_x_m": float(xy[:, 0].mean()), "centroid_y_m": float(xy[:, 1].mean()),
                        "n_injectors": sum(w[2] == "I" for w in wells),
                        "n_producers": sum(w[2] == "P" for w in wells),
                        "well_spacing_m": dmin,
                        "well_depth_m": top + (k1 + 1) * dz,
                        "completed_layers": k1 - k0 + 1,
                        "completion_top_layer": k0, "completion_bottom_layer": k1,
                        "completion": comp, "skin": skin,
                        **_well_stats(grid, wells, k0, k1, dz),
                        "completed_thickness_m": (k1 - k0 + 1) * dz,
                        "depth_to_top_m": top + k0 * dz,
                        "n_grid_layers": grid.nz,
                        "inj_prod_distance_mean_m": float(np.mean(d_ip)) if d_ip else dmin,
                        "drainage_area_km2": hull_km2,
                        "centroid_offset_m": c_off,
                        "wells": [{"i": int(w[0]), "j": int(w[1]), "role": w[2]} for w in wells],
                    }
                    emit({"id": wid, "type": "Well", "level": 2, "region": rid, "region_name": names[r], "attributes": w_attr})
                    edges.add(pid, wid, "refines")
                    for other in sibling_wells:
                        edges.add(other, wid, "transforms")
                    sibling_wells.append(wid)
                    # Controls: full factorial over rate x BHP x schedule.
                    grid_ids: dict[tuple[int, int, int], str] = {}
                    for c_idx, (a, b, s) in enumerate(itertools.product(range(len(rates)), range(len(bhps)), range(len(schedules)))):
                        cid = f"{wid}:C{c_idx:03d}"
                        sched = schedules[s]
                        c_attr = {
                            **w_attr,
                            "inj_rate_m3d": rates[a], "prod_bhp_bar": bhps[b],
                            "schedule_shape": sched["shape"], "n_control_stages": int(sched["stages"]),
                            "stage_rate_multipliers": list(sched["multipliers"]),
                        }
                        emit({"id": cid, "type": "Control", "level": 3, "region": rid, "region_name": names[r], "attributes": c_attr})
                        edges.add(wid, cid, "refines")
                        grid_ids[(a, b, s)] = cid
                        control_ids.append(cid)
                        quant["inj_rate_m3d"].append(rates[a])
                        quant["prod_bhp_bar"].append(bhps[b])
                    if gcfg["edges"].get("transforms_controls", True):
                        for (a, b, s), cid in grid_ids.items():
                            for nb in ((a + 1, b, s), (a, b + 1, s), (a, b, s + 1)):
                                if nb in grid_ids:
                                    edges.add(cid, grid_ids[nb], "transforms")
                    quant["well_spacing_m"].append(dmin)
                    quant["well_depth_m"].append(top + (k1 + 1) * dz)
                    quant["n_wells"].append(len(wells))

    if gcfg["edges"].get("combines_adjacent_regions", True):
        for a, b in sorted(adjacency):
            edges.add(f"{prefix}:R{a:02d}", f"{prefix}:R{b:02d}", "combines")
            for key, pa in pattern_keys[a].items():
                pb = pattern_keys[b].get(key)
                if pb:
                    edges.add(pa, pb, "combines")
    fnodes.close()
    edges.close()
    write_lines(out_dir / "control_ids.txt.gz", control_ids)

    qs = (0.1, 0.25, 0.5, 0.75, 0.9)
    meta = {
        "reservoir": res["name"],
        "prefix": prefix,
        "generator_version": __version__,
        "config_hash": config_hash(cfg),
        "grid": {"nx": grid.nx, "ny": grid.ny, "nz": grid.nz, "provenance": provenance,
                 "active_cells": int(grid.actnum.sum()),
                 "pore_volume_mm3": float((grid.poro * grid.ntg)[grid.actnum].sum() * dx * dy * dz / 1e6),
                 "pore_volume_mmbbl": float((grid.poro * grid.ntg)[grid.actnum].sum() * dx * dy * dz * 6.289811 / 1e6)},
        "regions": [{"id": f"{prefix}:R{r:02d}", "name": n} for r, n in enumerate(names)],
        "region_patterns": {f"{prefix}:R{r:02d}": sorted({k[0] for k in pattern_keys[r]}) for r in pattern_keys},
        "node_counts": counts,
        "n_nodes": int(sum(counts.values())),
        "edge_counts": edges.counts,
        "n_edges": int(sum(edges.counts.values())),
        "attribute_quantiles": {
            k: {str(q): float(np.quantile(v, q)) for q in qs} for k, v in quant.items() if v
        },
        "files": {"nodes": "nodes.jsonl.gz", "edges": "edges.tsv.gz", "control_ids": "control_ids.txt.gz"},
    }
    meta["sha256"] = {k: sha256_file(out_dir / v) for k, v in meta["files"].items()}
    write_json(out_dir / "meta.json", meta)
    log.info("%s: %d nodes, %d edges", res["name"], meta["n_nodes"], meta["n_edges"])
    return meta


def iter_nodes(graph_dir: str | Path) -> Iterator[dict[str, Any]]:
    """Stream nodes of a generated graph."""
    from pharaoh.utils.io import read_jsonl

    yield from read_jsonl(Path(graph_dir) / "nodes.jsonl.gz")


def iter_edges(graph_dir: str | Path) -> Iterator[tuple[str, str, str]]:
    """Stream (src, dst, type) edges of a generated graph."""
    with gzip.open(Path(graph_dir) / "edges.tsv.gz", "rt", encoding="utf-8", newline="") as f:
        rd = csv.reader(f, delimiter="\t")
        next(rd)
        for src, dst, etype in rd:
            yield src, dst, etype
