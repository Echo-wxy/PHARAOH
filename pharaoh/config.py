"""YAML configuration loading with `_base_` inheritance and release checks.

A config may declare `_base_: path/to/base.yaml` (or a list of paths, relative
to the file). Bases are merged first, then the file's own keys override them
recursively. Keys whose value is `null` mark settings that must be filled with
the values used in the actual experiments; `find_unset` lists them.
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

from pharaoh.utils.io import stable_hash


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge `override` into a copy of `base`."""
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(path: str | Path, _seen: tuple[Path, ...] = ()) -> dict[str, Any]:
    """Load a YAML config, resolving `_base_` chains (cycles are rejected)."""
    path = Path(path).resolve()
    if path in _seen:
        raise ValueError(f"Cyclic _base_ chain at {path}")
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    bases = raw.pop("_base_", [])
    if isinstance(bases, str):
        bases = [bases]
    merged: dict[str, Any] = {}
    for b in bases:
        merged = deep_merge(merged, load_config(path.parent / b, _seen + (path,)))
    merged = deep_merge(merged, raw)
    merged.setdefault("_meta", {})
    merged["_meta"] = {"path": str(path)}
    return merged


def find_unset(cfg: Any, prefix: str = "") -> list[str]:
    """Return dotted keys whose value is None (settings still to be filled in)."""
    out: list[str] = []
    if isinstance(cfg, dict):
        for k, v in cfg.items():
            if k == "_meta":
                continue
            key = f"{prefix}.{k}" if prefix else str(k)
            if v is None:
                out.append(key)
            else:
                out.extend(find_unset(v, key))
    elif isinstance(cfg, list):
        for i, v in enumerate(cfg):
            out.extend(find_unset(v, f"{prefix}[{i}]"))
    return out


def config_hash(cfg: dict[str, Any]) -> str:
    """Hash of a resolved config, excluding file-location metadata."""
    clean = {k: v for k, v in cfg.items() if k != "_meta"}
    return stable_hash(clean)


def require(cfg: dict[str, Any], dotted: str) -> Any:
    """Fetch a nested key and fail loudly if it is missing or null."""
    cur: Any = cfg
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            raise KeyError(f"Missing config key: {dotted}")
        cur = cur[part]
    if cur is None:
        raise ValueError(f"Config key {dotted} is null; set it to the value used in the experiments.")
    return cur
