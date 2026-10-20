"""Central place for random number generators so every stage is reproducible."""
from __future__ import annotations

import hashlib
import random

import numpy as np


def set_global_seed(seed: int) -> None:
    """Seed Python's and NumPy's global generators."""
    random.seed(seed)
    np.random.seed(seed % (2**32))


def derive_rng(seed: int, *tags: str) -> np.random.Generator:
    """Return an independent generator derived from a base seed and string tags.

    Deriving sub-streams by name keeps one stage's randomness unaffected when
    another stage changes the number of draws it makes.
    """
    h = hashlib.sha256(("|".join([str(seed), *tags])).encode("utf-8")).digest()
    return np.random.default_rng(int.from_bytes(h[:8], "little"))
