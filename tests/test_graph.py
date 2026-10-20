from collections import Counter, defaultdict

from pharaoh.config import load_config
from pharaoh.petrograph.generate import generate_graph, iter_edges, iter_nodes
from pharaoh.petrograph.schema import ATTRIBUTE_FIELDS, LEVEL_INDEX, attribute_vector, parent_id
from pharaoh.utils.io import read_json


def test_hierarchy_invariants(demo_graph):
    nodes = {n["id"]: n for n in iter_nodes(demo_graph)}
    parents = defaultdict(list)
    by_type = Counter()
    for src, dst, et in iter_edges(demo_graph):
        assert src in nodes and dst in nodes
        ls, ld = LEVEL_INDEX[nodes[src]["type"]], LEVEL_INDEX[nodes[dst]["type"]]
        if et == "refines":
            assert ld == ls + 1
            parents[dst].append(src)
        elif et == "transforms":
            assert ls == ld and parent_id(src) == parent_id(dst) and ls >= 2
        else:
            assert et == "combines" and ls == ld and ls <= 1
        by_type[et] += 1
    for nid, n in nodes.items():
        if n["type"] == "Region":
            assert nid not in parents
        else:
            assert parents[nid] == [parent_id(nid)]
    meta = read_json(demo_graph / "meta.json")
    assert meta["n_nodes"] == len(nodes)
    assert meta["edge_counts"] == dict(by_type)
    assert sum(1 for n in nodes.values() if n["type"] == "Control") > 0


def test_control_nodes_are_complete_scenarios(demo_graph):
    for n in iter_nodes(demo_graph):
        if n["type"] == "Control":
            vec = attribute_vector(n)
            assert len(vec) == len(ATTRIBUTE_FIELDS)
            assert all(v == v for v in vec), n["id"]  # no NaN on Control nodes
            assert "EUR" not in n["text"] and "NPV" not in n["text"]


def test_deterministic(tmp_path, root):
    cfg = load_config(root / "configs/reservoirs/demo.yaml")
    a = generate_graph(cfg, tmp_path / "a", root=root)
    b = generate_graph(cfg, tmp_path / "b", root=root)
    assert a["sha256"] == b["sha256"]
