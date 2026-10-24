import numpy as np

from onsagerdecomposer import Shells, liquid_events, polymer_events, solvation_states, transitions
from onsagerdecomposer.shells import KINDS


def toy_shells(shells_per_cation):
    """Build Shells from {cation: [ {residue: [atoms]} per frame ]}; residues 1-2 solvent, 3-4 anion."""
    rows = []
    for c, frames in shells_per_cation.items():
        for f, shell in enumerate(frames):
            rows += [(f, c, r, a) for r, atoms in shell.items() for a in atoms]
    rows = np.array(rows)
    rows = rows[np.lexsort((rows[:, 3], rows[:, 2], rows[:, 0], rows[:, 1]))]
    kind = np.full(5, -1)
    kind[[1, 2]] = KINDS.index("solvent")
    kind[[3, 4]] = KINDS.index("anion")
    n_frames = len(next(iter(shells_per_cation.values())))
    pos = np.zeros((n_frames, len(shells_per_cation), 3))
    return Shells(rows, kind, pos, np.zeros((n_frames, 1, 3)), 1.0)


SHELLS = toy_shells({
    0: [{1: [10, 11]}, {1: [10, 11]}, {1: [11, 12]}, {1: [11, 12], 2: [20]}],
    1: [{3: [30]}, {3: [30], 1: [13]}, {1: [13]}, {1: [13, 14]}],
})


def names_of(events):
    return [[events.names[k] for k in row] for row in events.labels]


def test_polymer_events():
    assert names_of(polymer_events(SHELLS)) == [
        ["vehicular", "complete_intrachain_hop", "interchain_hop"],
        ["interchain_hop", "anion_exchange", "partial_intrachain_hop"],
    ]


def test_liquid_events():
    assert names_of(liquid_events(SHELLS)) == [
        ["vehicular", "vehicular", "solvent_exchange"],
        ["solvent_exchange", "anion_exchange", "vehicular"],
    ]


def test_majority_chain_breaks_ties_by_lowest_residue():
    # chains 1 and 2 coordinate equally; chain 1 (lower id) rolls, chain 2 is static -> complete hop
    shells = toy_shells({0: [{1: [10, 11], 2: [20, 21]}, {1: [11, 12], 2: [20, 21]}]})
    assert names_of(polymer_events(shells)) == [["complete_intrachain_hop"]]


def test_solvation_states_and_transitions():
    shells = toy_shells({
        0: [{1: [10]}, {1: [10], 3: [30]}, {3: [30], 4: [40]}],   # SSIP, AGG (anion 3 shared), AGG (2 anions)
        1: [{3: [31]}, {3: [31]}, {1: [12]}],                     # CIP, AGG (shares anion 3), SSIP
    })
    states, names = solvation_states(shells)
    assert [[names[s] for s in row] for row in states] == [["SSIP", "AGG", "AGG"], ["CIP", "AGG", "SSIP"]]
    tr = transitions(states, names)
    assert names_of(tr) == [["SSIP->AGG", "AGG->AGG"], ["CIP->AGG", "AGG->SSIP"]]
    assert np.isclose(tr.probabilities().sum(), 1)
