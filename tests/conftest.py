from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pharaoh.config import load_config  # noqa: E402
from pharaoh.petrograph.benchmark import build_benchmark  # noqa: E402
from pharaoh.petrograph.generate import generate_graph  # noqa: E402


@pytest.fixture(scope="session")
def root() -> Path:
    return ROOT


@pytest.fixture(scope="session")
def demo_graph(tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("graph")
    generate_graph(load_config(ROOT / "configs/reservoirs/demo.yaml"), out, root=ROOT)
    return out


@pytest.fixture(scope="session")
def demo_splits(demo_graph, tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("splits")
    build_benchmark(load_config(ROOT / "configs/benchmark/demo.yaml"), demo_graph, out, root=ROOT)
    return out
