"""Macrostates: coarse-grained labels of each cation's shell in each frame.

A macrostate rule is any function ``shells -> (states, names)`` with states of
shape (n_cation, n_frames). ``transitions`` turns it into one event per ordered
pair of macrostates (X->Y), which can be decomposed like any other event set.
"""
import numpy as np

from .events import Events
from .shells import KINDS


def solvation_states(shells):
    """SSIP (no anion), CIP (one anion, not shared), AGG (two or more anions, or an
    anion that also coordinates another cation in the same frame)."""
    r = shells.rows[shells.row_kinds() == KINDS.index("anion"), :3]
    r = r[np.r_[True, np.any(r[1:] != r[:-1], axis=1)]] if len(r) else r   # one row per (frame, cation, anion)
    frame, cation, anion = r.T
    n_anions = np.zeros((shells.n_cations, shells.n_frames), dtype=int)
    np.add.at(n_anions, (cation, frame), 1)
    _, inverse, counts = np.unique(np.column_stack([frame, anion]), axis=0, return_inverse=True, return_counts=True)
    shared = np.zeros_like(n_anions, dtype=bool)
    is_shared = counts[inverse.ravel()] >= 2
    shared[cation[is_shared], frame[is_shared]] = True
    states = np.select([n_anions == 0, (n_anions >= 2) | shared], [0, 2], default=1)
    return states, ["SSIP", "CIP", "AGG"]


def transitions(states, names):
    """Events X->Y for consecutive frames, ordered X-major (SSIP->SSIP, SSIP->CIP, ...)."""
    n = len(names)
    labels = states[:, :-1] * n + states[:, 1:]
    return Events(labels, [f"{a}->{b}" for a in names for b in names])
