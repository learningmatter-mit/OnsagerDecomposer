"""Event (transition) classification of every cation window.

An event rule is any function ``shells -> Events`` that gives each window
[t, t + dt] of each cation exactly one label. ``residue_entries_exits`` and
``atom_entries_exits`` are the building blocks: they report, for every window,
whether a residue (molecule, chain, anion) or a coordinating atom entered or
left the shell. The rules below are the ones used in the paper.
"""
from dataclasses import dataclass

import numpy as np

from .shells import KINDS


@dataclass
class Events:
    labels: np.ndarray   # (n_cation, n_windows) index into names
    names: list

    def probabilities(self):
        """Fraction of all windows (over all cations) assigned to each event."""
        return np.bincount(self.labels.ravel(), minlength=len(self.names)) / self.labels.size


def _entries_exits(cation, frame, item, n_cations, n_frames):
    """Entered/left masks for unique (cation, frame, item) occurrences."""
    n_items = int(item.max()) + 1 if len(item) else 1
    key = (cation * n_frames + frame) * n_items + item
    entered = np.zeros((n_cations, n_frames - 1), dtype=bool)
    left = np.zeros_like(entered)
    new = (frame > 0) & ~np.isin(key - n_items, key)            # absent in the previous frame
    entered[cation[new], frame[new] - 1] = True
    gone = (frame < n_frames - 1) & ~np.isin(key + n_items, key)  # absent in the next frame
    left[cation[gone], frame[gone]] = True
    return entered, left


def residue_entries_exits(shells, kind):
    """(entered, left), each (n_cation, n_windows) bool: a residue of ``kind`` entered / left the shell."""
    r = shells.rows[shells.row_kinds() == KINDS.index(kind), :3]
    r = r[np.r_[True, np.any(r[1:] != r[:-1], axis=1)]] if len(r) else r   # rows are sorted: drop repeated residues
    frame, cation, residue = r.T
    return _entries_exits(cation, frame, residue, shells.n_cations, shells.n_frames)


def residue_changes(shells, kind):
    """(n_cation, n_windows) bool: some residue of ``kind`` entered or left the shell."""
    entered, left = residue_entries_exits(shells, kind)
    return entered | left


def atom_entries_exits(shells, rows):
    """(entered, left) for coordinating atoms, restricted to the given subset of shell rows."""
    return _entries_exits(rows[:, 1], rows[:, 0], rows[:, 3], shells.n_cations, shells.n_frames)


def liquid_events(shells):
    """Vehicular motion, solvent exchange, anion exchange, or both exchanges in one window."""
    solvent = residue_changes(shells, "solvent")
    anion = residue_changes(shells, "anion")
    return Events(solvent.astype(int) + 2 * anion,
                  ["vehicular", "solvent_exchange", "anion_exchange", "solvent_anion_exchange"])


def _runs(mask):
    """Run id (>= 0) of each True window, consecutive True windows of a cation sharing an id; -1 elsewhere."""
    start = mask & ~np.pad(mask, ((0, 0), (1, 0)))[:, :-1]
    run = np.cumsum(start.ravel()).reshape(mask.shape) - 1
    return np.where(mask, run, -1)


def majority_chain_rows(shells, steady):
    """Shell rows of the majority chain during each steady stretch of windows.

    steady: (n_cation, n_windows) bool, windows in which no residue entered or left.
    Consecutive steady windows form a stretch spanning frames t0 .. t1 + 1. The
    majority chain of a stretch is the polymer residue with the most coordinating
    atoms summed over its frames (ties: lowest residue id).
    """
    run_w = _runs(steady)
    run_f = np.full((shells.n_cations, shells.n_frames), -1)
    run_f[:, :-1] = run_w
    c, t = np.nonzero(run_w >= 0)                               # each steady window also claims its end frame
    run_f[c, t + 1] = run_w[c, t]

    rows = shells.rows[shells.row_kinds() == KINDS.index("solvent")]
    run = run_f[rows[:, 1], rows[:, 0]]
    rows, run = rows[run >= 0], run[run >= 0]
    pairs, inverse = np.unique(np.column_stack([run, rows[:, 2]]), axis=0, return_inverse=True)
    counts = np.bincount(inverse.ravel())
    best = np.lexsort((pairs[:, 1], -counts, pairs[:, 0]))          # per run: most atoms, then lowest id
    first = np.r_[True, pairs[best[1:], 0] != pairs[best[:-1], 0]]
    majority = np.full(run_w.max() + 1, -1)
    majority[pairs[best[first], 0]] = pairs[best[first], 1]
    return rows[majority[run] == rows[:, 2]]


def polymer_events(shells):
    """Six polymer-electrolyte events (paper Fig. 4a).

    A chain entering/leaving the shell is an interchain hop; an anion entering/leaving
    is anion exchange; both together is a coupled event. Otherwise, the majority
    chain gaining and losing coordinating atoms is a complete intrachain hop, only
    gaining or only losing is a partial intrachain hop, and no change is vehicular.
    """
    chain = residue_changes(shells, "solvent")
    anion = residue_changes(shells, "anion")
    steady = ~chain & ~anion
    entered, left = atom_entries_exits(shells, majority_chain_rows(shells, steady))
    intrachain = (entered & steady).astype(int) + (left & steady)
    labels = np.select([chain & anion, anion, chain], [5, 4, 3], default=intrachain)
    return Events(labels, ["vehicular", "partial_intrachain_hop", "complete_intrachain_hop",
                           "interchain_hop", "anion_exchange", "anion_exchange_interchain_hop"])
