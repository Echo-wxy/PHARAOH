import numpy as np

from pharaoh.petrograph.labels import label_query, label_stats
from pharaoh.petrograph.pareto import non_dominated_mask


def test_pareto_senses():
    v = np.array([[10, 5], [8, 3], [10, 6], [12, 9], [7, 1]])
    # maximize column 0, minimize column 1
    m = non_dominated_mask(v, ["max", "min"])
    assert m.tolist() == [True, True, False, True, True]
    # maximize both
    m2 = non_dominated_mask(v, ["max", "max"])
    assert m2.tolist() == [False, False, False, True, False]


def test_duplicates_do_not_dominate():
    v = np.array([[1, 1], [1, 1]])
    assert non_dominated_mask(v, ["max", "max"]).all()


def test_label_query_constraints_and_missing():
    q = {"qid": "q", "objectives": [{"name": "eur", "sense": "max"}, {"name": "capex", "sense": "min"}],
         "outcome_constraints": [{"field": "npv", "op": ">=", "value": 100}],
         "attribute_constraints": [{"field": "well_spacing_m", "op": ">=", "value": 200}]}
    summ = {s: {"well_spacing_m": sp} for s, sp in {"a": 250, "b": 250, "c": 150, "d": 300, "e": 300}.items()}
    res = {"a": {"eur": 5, "capex": 50, "npv": 120}, "b": {"eur": 4, "capex": 60, "npv": 130},
           "c": {"eur": 9, "capex": 10, "npv": 200}, "d": {"eur": 6, "capex": 70, "npv": 90}}
    r = label_query(q, ["a", "b", "c", "d", "e"], res, summ)
    assert r["pareto_ids"] == ["a"]
    assert r["n_feasible"] == 2 and r["n_missing_results"] == 1
    st = label_stats([r])
    assert st["mean_pareto_size"] == 1.0 and st["total_missing_results"] == 1
