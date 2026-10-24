import numpy as np

from onsagerdecomposer import Shells, transitions
from onsagerdecomposer.crystal import activated_states, crystal_events
from onsagerdecomposer.shells import KINDS


def toy_crystal():
    """Three ions (atoms 0, 1, 2) over 3 frames; ion 0 and 1 are neighbours throughout."""
    sites = np.array([[0, 5, 9],      # frame 0
                      [1, 6, 9],      # frame 1: ions 0 and 1 hop, ion 2 stays
                      [1, 6, 8]])     # frame 2: only ion 2 hops
    pos = np.zeros((3, 3, 3))
    pos[1, 0] = [1, 0, 0]
    pos[1, 1] = [-1, 0.1, 0]          # ion 1 hops opposite to ion 0 -> unaligned
    pos[2, 2] = [0, 0, 1]
    rows = [(f, 0, 1, 1) for f in range(3)] + [(f, 1, 0, 0) for f in range(3)]
    rows = np.array(rows)
    rows = rows[np.lexsort((rows[:, 3], rows[:, 2], rows[:, 0], rows[:, 1]))]
    kind = np.full(3, KINDS.index("cation"))
    return Shells(rows, kind, pos, np.zeros((3, 0, 3)), 1.0, np.arange(3), sites)


def test_crystal_events():
    ev = crystal_events(toy_crystal())
    got = [[ev.names[k] for k in row] for row in ev.labels]
    assert got == [["unaligned_concerted_hop", "local_vibration"],
                   ["unaligned_concerted_hop", "local_vibration"],
                   ["local_vibration", "single_ion_hop"]]
    assert crystal_events(toy_crystal(), aligned_cos=-1.0).labels[0, 0] == ev.names.index("aligned_concerted_hop")


def test_activated_states():
    states, names = activated_states(toy_crystal())
    assert states.tolist() == [[1, 1, 1], [1, 1, 1], [0, 0, 0]]
    tr = transitions(states, names)
    assert tr.names[tr.labels[2, 0]] == "non-A->non-A"
