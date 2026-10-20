"""Minimal reader/writer for Eclipse-style GRDECL property files.

Only the keywords needed to derive node attributes are handled: grid
dimensions (SPECGRID or DIMENS), PERMX, PORO, NTG, and ACTNUM. INCLUDE
statements are followed relative to the including file, so the path may point
at a full simulation deck (.DATA). Values follow the Eclipse convention: i
varies fastest, then j, then k, and `n*value` expands to n repetitions.
Corner-point geometry (COORD/ZCORN) and BOX/EDIT-section modifiers are not
parsed; cell sizes come from the reservoir config instead.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

_DIM_KEYWORDS = ("SPECGRID", "DIMENS")
_ARRAY_KEYWORDS = ("PERMX", "PORO", "NTG", "ACTNUM")


@dataclass
class GridProperties:
    """Cell properties on an (nx, ny, nz) grid, indexed [i, j, k]."""

    nx: int
    ny: int
    nz: int
    permx: np.ndarray
    poro: np.ndarray
    actnum: np.ndarray
    ntg: np.ndarray = field(default=None)  # type: ignore[assignment]

    def __post_init__(self) -> None:
        shape = (self.nx, self.ny, self.nz)
        if self.ntg is None:
            self.ntg = np.ones(shape)
        for name in ("permx", "poro", "actnum", "ntg"):
            arr = getattr(self, name)
            if arr.shape != shape:
                raise ValueError(f"{name} has shape {arr.shape}, expected {shape}")
        self.actnum = self.actnum.astype(bool)

    @property
    def active_columns(self) -> np.ndarray:
        """Boolean (nx, ny) mask of columns with at least one active cell."""
        return self.actnum.any(axis=2)


def _tokens(text: str) -> list[str]:
    out: list[str] = []
    for line in text.splitlines():
        line = line.split("--", 1)[0].strip()
        if line:
            out.extend(line.replace("/", " / ").split())
    return out


def _expand(values: list[str]) -> list[float]:
    out: list[float] = []
    for v in values:
        if "*" in v:
            n, val = v.split("*", 1)
            out.extend([float(val)] * int(n))
        else:
            out.append(float(v))
    return out


def _deck_tokens(path: Path, depth: int = 0) -> list[str]:
    """Tokens of a deck with INCLUDE files spliced in place."""
    if depth > 8:
        raise ValueError(f"INCLUDE nesting too deep at {path}")
    toks = _tokens(path.read_text(encoding="utf-8", errors="replace"))
    out: list[str] = []
    i = 0
    while i < len(toks):
        if toks[i].upper() == "INCLUDE" and i + 1 < len(toks):
            inc = toks[i + 1].strip("'\"")
            inc_path = (path.parent / inc).resolve()
            if not inc_path.exists():
                raise FileNotFoundError(f"{path}: INCLUDE file not found: {inc_path}")
            out.extend(_deck_tokens(inc_path, depth + 1))
            i += 2
            if i < len(toks) and toks[i] == "/":
                i += 1
        else:
            out.append(toks[i])
            i += 1
    return out


def read_grdecl(path: str | Path) -> GridProperties:
    """Parse dimensions and the supported property arrays from a GRDECL file or deck."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Grid file not found: {path}. See data/README.md for how to obtain "
            "the reservoir model files and where to place them."
        )
    toks = _deck_tokens(path)
    dims: tuple[int, int, int] | None = None
    arrays: dict[str, list[float]] = {}
    i = 0
    while i < len(toks):
        kw = toks[i].upper()
        if kw in _DIM_KEYWORDS or kw in _ARRAY_KEYWORDS:
            j = i + 1
            vals: list[str] = []
            while j < len(toks) and toks[j] != "/":
                vals.append(toks[j])
                j += 1
            if kw in _DIM_KEYWORDS:
                dims = (int(float(vals[0])), int(float(vals[1])), int(float(vals[2])))
            else:
                arrays[kw] = _expand(vals)
            i = j + 1
        else:
            i += 1
    if dims is None:
        raise ValueError(f"{path}: no SPECGRID or DIMENS keyword found")
    nx, ny, nz = dims
    n = nx * ny * nz

    def grid(name: str, default: float | None) -> np.ndarray:
        if name not in arrays:
            if default is None:
                raise ValueError(f"{path}: required keyword {name} is missing")
            return np.full((nx, ny, nz), default, dtype=float)
        vals = np.asarray(arrays[name], dtype=float)
        if vals.size != n:
            raise ValueError(f"{path}: {name} has {vals.size} values, expected {n}")
        return vals.reshape((nz, ny, nx)).transpose(2, 1, 0)

    return GridProperties(
        nx=nx, ny=ny, nz=nz,
        permx=grid("PERMX", None),
        poro=grid("PORO", None),
        actnum=grid("ACTNUM", 1.0),
        ntg=grid("NTG", 1.0),
    )


def write_grdecl(path: str | Path, props: GridProperties) -> None:
    """Write dimensions and properties in GRDECL format (used for demo grids)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    def block(name: str, arr: np.ndarray, fmt: str) -> str:
        flat = arr.transpose(2, 1, 0).reshape(-1)
        lines = [" ".join(fmt.format(v) for v in flat[s:s + 8]) for s in range(0, flat.size, 8)]
        return f"{name}\n" + "\n".join(lines) + "\n/\n"

    with open(path, "w", encoding="utf-8") as f:
        f.write(f"SPECGRID\n{props.nx} {props.ny} {props.nz} 1 F /\n")
        f.write(block("ACTNUM", props.actnum.astype(int), "{:d}"))
        f.write(block("PERMX", props.permx, "{:.4f}"))
        f.write(block("PORO", props.poro, "{:.5f}"))
        f.write(block("NTG", props.ntg, "{:.4f}"))
