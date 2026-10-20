import numpy as np

from pharaoh.grdecl import read_grdecl, write_grdecl
from pharaoh.synthetic import synthetic_grid


def test_roundtrip(tmp_path):
    g = synthetic_grid(12, 9, 3, seed=1)
    write_grdecl(tmp_path / "g.grdecl", g)
    h = read_grdecl(tmp_path / "g.grdecl")
    assert (h.nx, h.ny, h.nz) == (12, 9, 3)
    assert np.allclose(g.permx, h.permx, rtol=1e-4)
    assert np.array_equal(g.actnum, h.actnum)


def test_repeat_counts_include_and_ordering(tmp_path):
    (tmp_path / "perm.inc").write_text("PERMX\n2*10 2*20 -- comment\n4*30 /\n")
    (tmp_path / "deck.DATA").write_text(
        "RUNSPEC\nDIMENS\n2 2 2 /\nGRID\nINCLUDE\n'perm.inc' /\nPORO\n8*0.2 /\n"
    )
    g = read_grdecl(tmp_path / "deck.DATA")
    # i fastest, then j, then k
    assert g.permx[0, 0, 0] == 10 and g.permx[1, 0, 0] == 10
    assert g.permx[0, 1, 0] == 20 and g.permx[0, 0, 1] == 30
    assert g.actnum.all() and np.allclose(g.ntg, 1.0)


def test_missing_file_message(tmp_path):
    try:
        read_grdecl(tmp_path / "nope.DATA")
    except FileNotFoundError as e:
        assert "data/README.md" in str(e)
    else:
        raise AssertionError("expected FileNotFoundError")
