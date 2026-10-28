import numpy as np
import pytest

from onsagerdecomposer import iter_hitpoly_frames, read_hitpoly_topology
from onsagerdecomposer.io import residue_kind


def test_residue_kinds_include_copolymer_blocks():
    assert [residue_kind(t) for t in ["PL1", "P1A", "P2B", "AN1", "CA1"]] == \
        ["solvent", "solvent", "solvent", "anion", "cation"]
    with pytest.raises(ValueError, match="XX1"):
        residue_kind("XX1")


def test_copolymer_folder(tmp_path):
    # two copolymer blocks (P1, P2), one anion atom, one cation; 2 frames
    (tmp_path / "atom_names_rdf_all.txt").write_text("C,P1A,1\nO,P2B,1\nN,AN1,2\nLi,CA1,3\n")
    (tmp_path / "box_len.txt").write_text("20.0")
    frames = np.array([[[0, 0, 0], [2, 0, 0], [5, 5, 5], [8, 8, 8]],
                       [[1, 0, 0], [3, 0, 0], [5, 5, 5], [8, 8, 8]]], dtype=float)
    np.savetxt(tmp_path / "xyz_unwrapped_rdf_all.txt", frames.reshape(-1, 3), delimiter=",")
    top = read_hitpoly_topology(tmp_path)
    assert list(top.kinds) == ["solvent", "solvent", "anion", "cation"]
    xyz = np.concatenate(list(iter_hitpoly_frames(tmp_path, top)))
    com = (12.0107 * xyz[:, 0] + 15.9994 * xyz[:, 1]) / (12.0107 + 15.9994)     # both blocks are solvent
    assert np.allclose(com, 0)
