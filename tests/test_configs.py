from pathlib import Path

import pytest

from pharaoh.config import deep_merge, find_unset, load_config, require

ROOT = Path(__file__).resolve().parents[1]
ALL = sorted(p for p in (ROOT / "configs").rglob("*.yaml") if not p.name.startswith("_"))


@pytest.mark.parametrize("path", ALL, ids=lambda p: str(p.relative_to(ROOT)))
def test_every_config_loads(path):
    cfg = load_config(path)
    assert isinstance(cfg, dict) and len(cfg) > 1


@pytest.mark.parametrize("path", ALL, ids=lambda p: str(p.relative_to(ROOT)))
def test_no_unset_values(path):
    assert find_unset(load_config(path)) == []


def test_libraries_and_pool_sizes():
    sizes = {"olympus": (4800, 960), "egg": (6400, 1280), "punq_s3": (3600, 720)}
    for res, (lib, ev) in sizes.items():
        cfg = load_config(ROOT / f"configs/benchmark/{res}.yaml")
        assert require(cfg, "benchmark.library.size") == lib and require(cfg, "benchmark.library.eval_size") == ev
        assert ev == 0.2 * lib
        assert require(cfg, "benchmark.pools.pool_size") == ev
        assert cfg["benchmark"]["library"]["split_unit"] == "Well"
        assert cfg["benchmark"]["n_queries"] == {"train": 6000, "val": 1000, "test": 1000}


def test_table2_vs_supplementary_subsets():
    for f in ("ablation_full_on_subset", "ablation_no_proxyexpand", "ablation_no_query_ranking", "ablation_no_type_filter"):
        assert load_config(ROOT / f"configs/agent/{f}.yaml")["evaluation"]["query_subset"] == "test_subset_qids.txt"
    for f in ("relevance_bm25", "aggregation_voting", "aggregation_ordering", "aggregation_random", "compute_grid"):
        assert load_config(ROOT / f"configs/agent/{f}.yaml")["evaluation"]["query_subset"] == "all"


def _diff(a, b, prefix=""):
    keys = set(a) | set(b)
    out = []
    for k in keys:
        if k == "_meta":
            continue
        x, y = a.get(k), b.get(k)
        p = f"{prefix}.{k}" if prefix else k
        if isinstance(x, dict) and isinstance(y, dict):
            out += _diff(x, y, p)
        elif x != y:
            out.append(p)
    return sorted(out)


@pytest.mark.parametrize("name,expected", [
    ("ablation_full_on_subset", ["evaluation.query_subset"]),
    ("ablation_no_proxyexpand", ["agent.tools.proxy_expand.enabled", "evaluation.query_subset"]),
    ("ablation_no_query_ranking", ["agent.tools.proxy_expand.query_ranking", "evaluation.query_subset"]),
    ("ablation_no_type_filter", ["agent.tools.proxy_expand.type_filter", "evaluation.query_subset"]),
    ("relevance_bm25", ["agent.relevance.scorer"]),
    ("aggregation_voting", ["agent.aggregation.method"]),
    ("aggregation_random", ["agent.aggregation.method"]),
])
def test_ablations_change_one_thing(name, expected):
    base = load_config(ROOT / "configs/agent/pharaoh.yaml")
    abl = load_config(ROOT / f"configs/agent/{name}.yaml")
    assert _diff(base, abl) == expected


def test_main_settings():
    a = load_config(ROOT / "configs/agent/pharaoh.yaml")["agent"]
    assert (a["backbone"], a["temperature"], a["max_steps"], a["n_parallel"]) == ("gpt-4.1", 0.7, 20, 3)
    assert a["aggregation"]["count_sense"] == "max"
    assert a["aggregation"]["objective_senses"] == {"eur_mmbbl": "max", "npv_musd": "max", "capex_musd": "min", "cum_water_mmbbl": "min"}
    assert (a["tools"]["proxy_search"]["k_default"], a["tools"]["proxy_expand"]["k_default"]) == (3, 5)
    assert a["run_seeds"] == [0, 1, 2]
    s = load_config(ROOT / "configs/surrogate.yaml")["surrogate"]
    assert s["outputs"] == ["eur_mmbbl", "npv_musd", "capex_musd", "cum_water_mmbbl"]
    assert s["architecture"]["layers"] == [64, 256, 256, 128, 64]
    d = load_config(ROOT / "configs/distill/qwen2.5-7b_6000q_lambda0.yaml")["distill"]
    assert d["loss"]["phys"]["lambda"] == 0.0 and d["optim"]["lr"] == 1e-5 and d["context_length"] == 16384


def test_deep_merge_and_cycle(tmp_path):
    assert deep_merge({"a": {"b": 1, "c": 2}}, {"a": {"b": 3}}) == {"a": {"b": 3, "c": 2}}
    (tmp_path / "x.yaml").write_text("_base_: y.yaml\nk: 1\n")
    (tmp_path / "y.yaml").write_text("_base_: x.yaml\nk: 2\n")
    with pytest.raises(ValueError):
        load_config(tmp_path / "x.yaml")
