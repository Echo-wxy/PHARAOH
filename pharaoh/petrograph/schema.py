"""Node/edge type vocabulary and the fixed attribute layout d_V(v)."""
from __future__ import annotations

import math
from typing import Any

#: Ordered hierarchy H = {Region, Pattern, Well, Control}; index = level l.
LEVELS: tuple[str, ...] = ("Region", "Pattern", "Well", "Control")
LEVEL_INDEX: dict[str, int] = {t: i for i, t in enumerate(LEVELS)}

#: Edge types phi_E.
EDGE_TYPES: tuple[str, ...] = ("refines", "combines", "transforms")

#: Numeric attribute vector d_V(v), in a fixed order. Fields that do not apply
#: to a node's level are stored as null and become NaN in the vector.
ATTRIBUTE_FIELDS: tuple[str, ...] = (
    # static rock properties (aggregated over the node's footprint)
    "perm_mean_md", "perm_log10_std", "poro_mean", "ntg_mean", "active_cells", "area_km2",
    "centroid_x_m", "centroid_y_m",
    # pattern
    "pattern_spacing_m",
    # well placement and completion
    "n_injectors", "n_producers", "well_spacing_m", "well_depth_m", "completed_layers",
    "completion_top_layer", "completion_bottom_layer", "skin",
    # production-injection controls
    "inj_rate_m3d", "prod_bhp_bar", "n_control_stages",
)

#: Categorical attributes (kept separately from the numeric vector).
CATEGORICAL_FIELDS: tuple[str, ...] = ("pattern_type", "completion", "schedule_shape")


def attribute_vector(node: dict[str, Any]) -> list[float]:
    """Numeric attribute vector in ATTRIBUTE_FIELDS order (NaN where not applicable)."""
    a = node.get("attributes", {})
    out: list[float] = []
    for f in ATTRIBUTE_FIELDS:
        v = a.get(f)
        out.append(float(v) if v is not None else math.nan)
    return out


def parent_id(node_id: str) -> str | None:
    """Hierarchical parent id (ids are colon-separated paths)."""
    parts = node_id.split(":")
    return ":".join(parts[:-1]) if len(parts) > 2 else None


def ancestor_at(node_id: str, level: str) -> str:
    """Id of the ancestor (or self) at the given hierarchy level."""
    parts = node_id.split(":")
    return ":".join(parts[: 2 + LEVEL_INDEX[level]])
