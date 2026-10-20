"""Feasibility checks shared by pool construction and ground-truth labeling."""
from __future__ import annotations

import operator
from typing import Any

_OPS = {">=": operator.ge, "<=": operator.le, "==": operator.eq, ">": operator.gt, "<": operator.lt}


def scenario_summary(node: dict[str, Any]) -> dict[str, Any]:
    """Fields of a Control node that attribute constraints may refer to."""
    a = node["attributes"]
    return {
        "id": node["id"],
        "region": node["region"],
        "pattern_type": a["pattern_type"],
        "well_spacing_m": a["well_spacing_m"],
        "well_depth_m": a["well_depth_m"],
        "n_wells": a["n_injectors"] + a["n_producers"],
    }


def satisfies(values: dict[str, Any], constraints: list[dict[str, Any]]) -> bool:
    """True if every constraint {field, op, value, ...} holds on `values`.

    A constraint on a field that is absent from `values` counts as violated,
    so missing simulation outputs never make a scenario look feasible.
    """
    for c in constraints:
        v = values.get(c["field"])
        if v is None or not _OPS[c["op"]](v, c["value"]):
            return False
    return True


def structural_constraints(query: dict[str, Any]) -> list[dict[str, Any]]:
    """Attribute constraints that select a subtree (region / pattern type)."""
    return [c for c in query["attribute_constraints"] if c["field"] in ("region", "pattern_type")]
