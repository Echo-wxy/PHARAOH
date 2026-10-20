"""Sample natural-language multi-objective queries from templates.

Every query carries its structured form (objectives with senses, outcome
constraints, attribute constraints) so that ground truth can be computed
without parsing text, and a `signature` that identifies queries with the
same template and the same constraint values. Splits are made over
signature groups, so two queries that differ only in wording can never
fall into different splits.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import yaml

from pharaoh.config import require
from pharaoh.petrograph.patterns import PATTERN_LABELS
from pharaoh.utils.io import stable_hash
from pharaoh.utils.seed import derive_rng

# slot name -> (constraint field, operator, kind, unit)
_SLOT_SPEC: dict[str, tuple[str, str, str, str]] = {
    "npv_min": ("npv_musd", ">=", "outcome", "MUSD"),
    "capex_max": ("capex_musd", "<=", "outcome", "MUSD"),
    "water_cut_max": ("water_cut", "<=", "outcome", "fraction"),
    "spacing_min": ("well_spacing_m", ">=", "attribute", "m"),
    "depth_max": ("well_depth_m", "<=", "attribute", "m"),
    "n_wells_max": ("n_wells", "<=", "attribute", "count"),
    "region": ("region", "==", "attribute", "id"),
    "pattern_type": ("pattern_type", "==", "attribute", "category"),
}


def query_signature(template_id: str, objectives: list[dict[str, Any]], weights: list[float],
                    constraints: list[dict[str, Any]]) -> str:
    """Normalized signature: template, objective senses, weights, and every constraint with its unit.

    Two queries with the same signature ask for the same thing, whatever their wording.
    """
    canon = {
        "template": template_id,
        "objectives": [[o["name"], o["sense"]] for o in objectives],
        "weights": [round(float(w), 6) for w in weights],
        "constraints": sorted([c["field"], c["op"], c["value"], c["unit"]] for c in constraints),
    }
    return stable_hash(canon)


def load_templates(path: str | Path) -> dict[str, list[dict[str, Any]]]:
    """Load property-centric and relation-centric templates."""
    with open(path, encoding="utf-8") as f:
        t = yaml.safe_load(f)
    for cat in ("property_centric", "relation_centric"):
        if not t.get(cat):
            raise ValueError(f"{path}: no {cat} templates")
    return t


def _round_to(x: float, step: float) -> float:
    return round(float(np.round(x / step) * step), 6)


def _sample_slot(slot: str, rng: np.random.Generator, bcfg: dict[str, Any], meta: dict[str, Any]) -> Any:
    oc = bcfg["outcome_constraint_ranges"]
    aq = bcfg["attribute_constraint_quantiles"]
    q = meta["attribute_quantiles"]
    step = bcfg["rounding"]

    def from_quantile(key: str, lo_hi: list[float], step: float) -> float:
        lo, hi = lo_hi
        u = rng.uniform(lo, hi)
        qs = sorted((float(k), v) for k, v in q[key].items())
        xs, ys = zip(*qs)
        return _round_to(float(np.interp(u, xs, ys)), step)

    if slot == "npv_min":
        return _round_to(rng.uniform(*oc["npv_musd_min"]), step["musd"])
    if slot == "capex_max":
        return _round_to(rng.uniform(*oc["capex_musd_max"]), step["musd"])
    if slot == "water_cut_max":
        return _round_to(rng.uniform(*oc["water_cut_max"]), step["water_cut"])
    if slot == "spacing_min":
        return from_quantile("well_spacing_m", aq["spacing_min"], step["length_m"])
    if slot == "depth_max":
        return from_quantile("well_depth_m", aq["depth_max"], step["length_m"])
    if slot == "n_wells_max":
        return int(from_quantile("n_wells", aq["n_wells_max"], 1))
    raise KeyError(slot)


def _sample_structure(slots: list[str], rng: np.random.Generator, bcfg: dict[str, Any],
                      meta: dict[str, Any]) -> dict[str, Any]:
    """Draw region / pattern-type slots from combinations that exist in the graph."""
    want_r, want_p = "region" in slots, "pattern_type" in slots
    if not (want_r or want_p):
        return {}
    allowed = set(bcfg["pattern_types"])
    combos = [(r, p) for r, p in meta.get("allowed_structures") or
              [(r, p) for r, ps in sorted(meta["region_patterns"].items()) for p in ps] if p in allowed]
    if not combos:
        raise ValueError("No region/pattern combinations available for structural constraints")
    if want_r and want_p:
        r, p = combos[int(rng.integers(len(combos)))]
        return {"region": r, "pattern_type": p}
    if want_p:
        types = sorted({p for _, p in combos})
        return {"pattern_type": types[int(rng.integers(len(types)))]}
    regions = sorted({r for r, _ in combos})
    return {"region": regions[int(rng.integers(len(regions)))]}


def _weights(rng: np.random.Generator, m: int, step: float) -> list[float]:
    """Dirichlet weights snapped to a grid of `step` that still sum to one (no zero weights)."""
    units = int(round(1.0 / step))
    raw = rng.dirichlet(np.ones(m) * 2.0) * units
    k = np.maximum(np.floor(raw).astype(int), 1)
    while k.sum() > units:
        k[np.argmax(k)] -= 1
    while k.sum() < units:
        k[np.argmax(raw - k)] += 1
    return [round(float(x) * step, 6) for x in k]


def make_query(qid: str, template: dict[str, Any], category: str, rng: np.random.Generator,
               bcfg: dict[str, Any], meta: dict[str, Any], senses: dict[str, str]) -> dict[str, Any]:
    """Instantiate one template into a structured query record."""
    slots = list(template.get("outcome_constraints", [])) + list(template.get("attribute_constraints", []))
    values = {s: _sample_slot(s, rng, bcfg, meta) for s in slots if s not in ("region", "pattern_type")}
    values.update(_sample_structure(slots, rng, bcfg, meta))
    region_names = {r["id"]: r["name"] for r in meta["regions"]}
    fill = {
        "npv_min": f"{values.get('npv_min', 0):g}",
        "capex_max": f"{values.get('capex_max', 0):g}",
        "water_cut_max_pct": f"{100 * values.get('water_cut_max', 0):g}",
        "spacing_min": f"{values.get('spacing_min', 0):g}",
        "depth_max": f"{values.get('depth_max', 0):g}",
        "n_wells_max": f"{values.get('n_wells_max', 0)}",
        "region_name": region_names.get(values.get("region", ""), ""),
        "pattern_label": PATTERN_LABELS.get(values.get("pattern_type", ""), ""),
    }
    text = template["text"].format(**fill)
    outcome, attribute = [], []
    for s, v in values.items():
        field, op, kind, unit = _SLOT_SPEC[s]
        (outcome if kind == "outcome" else attribute).append({"field": field, "op": op, "value": v, "unit": unit})
    objectives = [{"name": o, "sense": senses[o]} for o in template["objectives"]]
    w = _weights(rng, len(objectives), float(bcfg["rounding"]["weight"]))
    signature = query_signature(template["id"], objectives, w, outcome + attribute)
    return {
        "qid": qid,
        "reservoir": meta["reservoir"],
        "category": category,
        "template_id": template["id"],
        "text": text,
        "canonical_text": text,
        "objectives": objectives,
        "reference_weights": w,
        "outcome_constraints": outcome,
        "attribute_constraints": attribute,
        "signature": signature,
    }


def generate_queries(bcfg: dict[str, Any], meta: dict[str, Any], templates: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    """Sample the full query set for one reservoir (before splitting)."""
    seed = int(require({"b": bcfg}, "b.seed"))
    n_total = int(sum(bcfg["n_queries"].values()))
    frac_rel = float(bcfg["relation_centric_fraction"])
    senses = {o["name"]: o["sense"] for o in bcfg["objectives"]}
    unique = bool(bcfg.get("unique_signatures", False))
    max_tries = int(bcfg.get("max_resample_tries", 50))
    rng = derive_rng(seed, "queries", meta["reservoir"])
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for n in range(n_total):
        qid = f"{meta['prefix']}-Q{n:05d}"
        for _ in range(max_tries):
            category = "relation_centric" if rng.uniform() < frac_rel else "property_centric"
            pool = templates[category]
            tpl = pool[int(rng.integers(len(pool)))]
            q = make_query(qid, tpl, category, rng, bcfg, meta, senses)
            if not unique or q["signature"] not in seen:
                break
        else:
            raise RuntimeError(
                f"Could not draw a query with a new signature after {max_tries} tries; "
                "widen the constraint ranges or add templates."
            )
        seen.add(q["signature"])
        out.append(q)
    return out
