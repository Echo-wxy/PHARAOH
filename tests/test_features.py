import numpy as np
import pytest

from pharaoh.config import load_config
from pharaoh.features import FEATURE_NAMES, INPUT_DIM, feature_dict, feature_vector
from pharaoh.petrograph.generate import iter_nodes
from pharaoh.utils.io import read_json


def test_dimension_matches_surrogate_config(root):
    assert INPUT_DIM == 64 == len(set(FEATURE_NAMES))
    assert load_config(root / "configs/surrogate.yaml")["surrogate"]["input_dim"] == INPUT_DIM


def test_features_finite_on_all_scenarios(demo_graph):
    X = np.array([feature_vector(n) for n in iter_nodes(demo_graph) if n["type"] == "Control"])
    assert X.shape[1] == 64 and np.isfinite(X).all()
    onehots = [i for i, k in enumerate(FEATURE_NAMES) if "=" in k]
    assert set(np.unique(X[:, onehots])) <= {0.0, 1.0}
    assert X[:, [FEATURE_NAMES.index(k) for k in FEATURE_NAMES if k.startswith("pattern=")]].sum(1).tolist() == [1.0] * len(X)


def test_physical_consistency(demo_graph):
    n = next(x for x in iter_nodes(demo_graph) if x["type"] == "Control" and x["attributes"]["n_control_stages"] == 3)
    f = feature_dict(n)
    a = n["attributes"]
    assert f["total_inj_rate_m3d"] == pytest.approx(a["inj_rate_m3d"] * a["n_injectors"])
    assert f["stage_multiplier_mean"] == pytest.approx(np.mean(a["stage_rate_multipliers"]))
    assert 0 <= f["completion_top_frac"] <= f["completion_bottom_frac"] <= 1
    assert f["perm_log10_min"] <= f["perm_log10_mean"] <= f["perm_log10_max"]
    assert f["drainage_area_km2"] > 0 and f["pattern_pore_volume_mm3"] > 0


def test_pore_volume_in_meta(demo_graph):
    g = read_json(demo_graph / "meta.json")["grid"]
    assert g["pore_volume_mmbbl"] == pytest.approx(g["pore_volume_mm3"] * 6.289811)


def test_rejects_non_scenario(demo_graph):
    with pytest.raises(ValueError):
        feature_dict(next(iter_nodes(demo_graph)))
