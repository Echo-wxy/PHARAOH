"""Partition the active grid footprint into development regions."""
from __future__ import annotations

import numpy as np

from pharaoh.utils.seed import derive_rng

_COMPASS = ("east", "north_east", "north", "north_west", "west", "south_west", "south", "south_east")


def kmeans_columns(active: np.ndarray, n_regions: int, seed: int, iters: int = 50) -> np.ndarray:
    """Lloyd's k-means on active column coordinates.

    Returns an (nx, ny) integer array with region index per active column and
    -1 for inactive columns. Clusters are relabeled by centroid (j, then i)
    so labels are stable under seed changes that only permute clusters.
    """
    ii, jj = np.nonzero(active)
    pts = np.stack([ii, jj], axis=1).astype(float)
    if len(pts) < n_regions:
        raise ValueError(f"Only {len(pts)} active columns for {n_regions} regions")
    rng = derive_rng(seed, "kmeans")
    centers = pts[rng.choice(len(pts), n_regions, replace=False)]
    labels = np.zeros(len(pts), dtype=int)
    for _ in range(iters):
        d = ((pts[:, None, :] - centers[None, :, :]) ** 2).sum(-1)
        new = d.argmin(1)
        for c in range(n_regions):
            m = new == c
            centers[c] = pts[m].mean(0) if m.any() else pts[rng.integers(len(pts))]
        if np.array_equal(new, labels):
            break
        labels = new
    order = np.lexsort((centers[:, 0], centers[:, 1]))
    remap = np.empty(n_regions, dtype=int)
    remap[order] = np.arange(n_regions)
    out = np.full(active.shape, -1, dtype=int)
    out[ii, jj] = remap[labels]
    return out


def region_adjacency(labels: np.ndarray) -> set[tuple[int, int]]:
    """Pairs (a, b), a < b, of regions sharing a 4-neighbor column boundary."""
    pairs: set[tuple[int, int]] = set()
    for a, b in ((labels[1:, :], labels[:-1, :]), (labels[:, 1:], labels[:, :-1])):
        m = (a >= 0) & (b >= 0) & (a != b)
        for x, y in zip(a[m].tolist(), b[m].tolist()):
            pairs.add((min(x, y), max(x, y)))
    return pairs


def region_names(labels: np.ndarray, central_radius: float = 0.25) -> list[str]:
    """Human-readable names (Region_central, Region_north_east, ...) by centroid position."""
    nx, ny = labels.shape
    n = int(labels.max()) + 1
    names: list[str] = []
    used: dict[str, int] = {}
    for r in range(n):
        ii, jj = np.nonzero(labels == r)
        x = (ii.mean() - (nx - 1) / 2) / (nx / 2)
        y = (jj.mean() - (ny - 1) / 2) / (ny / 2)
        if np.hypot(x, y) < central_radius:
            base = "central"
        else:
            sector = int(((np.degrees(np.arctan2(y, x)) + 22.5) % 360) // 45)
            base = _COMPASS[sector]
        used[base] = used.get(base, 0) + 1
        names.append(f"Region_{base}" if used[base] == 1 else f"Region_{base}_{used[base]}")
    return names
