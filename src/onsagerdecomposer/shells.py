"""First coordination shells of every cation in every frame.

The shell of a cation is the set of coordinating atoms (e.g. O, N, F) around it,
found either with a hard distance cutoff or with the parameter-free convex-hull
criterion used in the paper. Shells are stored as a flat table so that any
event or macrostate rule can be written with a few vectorised numpy operations.
"""
from dataclasses import dataclass
from multiprocessing import Pool

import numpy as np
from scipy.spatial import ConvexHull, cKDTree

KINDS = ("solvent", "anion", "cation")


@dataclass
class Shells:
    """Coordination shells and positions on a regular grid of frames.

    rows: (n_rows, 4) int array with columns [frame, cation, residue, atom]; one row
        per coordinating atom, sorted by (cation, frame, residue, atom).
    residue_kind: (max_residue + 1,) int array, index into ``KINDS`` (-1 if unused).
    cation_positions: (n_frames, n_cation, 3) unwrapped positions.
    anion_positions: (n_frames, n_anion, 3) unwrapped positions of the anion centre atoms.
    timestep: time between stored frames (ps).
    cation_atoms: (n_cation,) atom index of each cation.
    sites: optional (n_frames, n_cation) crystal site occupied by each cation (crystals only).
    """

    rows: np.ndarray
    residue_kind: np.ndarray
    cation_positions: np.ndarray
    anion_positions: np.ndarray
    timestep: float
    cation_atoms: np.ndarray = None
    sites: np.ndarray = None

    @property
    def n_frames(self):
        return self.cation_positions.shape[0]

    @property
    def n_cations(self):
        return self.cation_positions.shape[1]

    def subsample(self, k):
        """Shells at every k-th stored frame (window length k * timestep)."""
        keep = self.rows[:, 0] % k == 0
        rows = self.rows[keep].copy()
        rows[:, 0] //= k
        return Shells(rows, self.residue_kind, self.cation_positions[::k], self.anion_positions[::k],
                      self.timestep * k, self.cation_atoms, None if self.sites is None else self.sites[::k])

    def cation_displacements(self):
        """(n_cation, n_frames - 1, 3) displacement of each cation in each window."""
        return np.diff(self.cation_positions, axis=0).transpose(1, 0, 2)

    def anion_displacements(self):
        return np.diff(self.anion_positions, axis=0).transpose(1, 0, 2)

    def row_kinds(self):
        """Kind index (into KINDS) of the residue of every row."""
        return self.residue_kind[self.rows[:, 2]]

    def save(self, path):
        extra = {k: v for k, v in (("cation_atoms", self.cation_atoms), ("sites", self.sites)) if v is not None}
        np.savez_compressed(path, rows=self.rows, residue_kind=self.residue_kind,
                            cation_positions=self.cation_positions, anion_positions=self.anion_positions,
                            timestep=self.timestep, **extra)

    @classmethod
    def load(cls, path):
        f = np.load(path)
        return cls(f["rows"], f["residue_kind"], f["cation_positions"], f["anion_positions"], float(f["timestep"]),
                   f["cation_atoms"] if "cation_atoms" in f else None, f["sites"] if "sites" in f else None)


def convex_hull_shell_size(vectors, distances, nearest_other, max_distance_ratio=1.75):
    """Number of coordinators (taken in order of distance) inside the convex-hull shell.

    Start from the 4 nearest coordinators and keep adding the next one while
    (i) it adds exactly one hull vertex, (ii) the nearest non-coordinating heavy
    atom would still sit on the hull, (iii) adding that atom would change the
    hull, and (iv) it is closer than ``max_distance_ratio`` times the nearest one.
    """
    n = min(4, len(vectors))
    if len(vectors) > 4:
        n_old = len(ConvexHull(vectors[:4]).vertices)
    for j in range(4, len(vectors)):
        shell = vectors[:n]
        n_new = len(ConvexHull(np.vstack([shell, vectors[j]])).vertices)
        with_other = ConvexHull(np.vstack([shell, nearest_other, vectors[j]]))
        if (n_new == n_old + 1 and n in with_other.vertices and n_new != len(with_other.vertices)
                and distances[j] < max_distance_ratio * distances[0]):
            n, n_old = n + 1, n_new
        else:
            break
    return n


# Per-process constants for the worker pool (set once by _init_worker).
_W = {}


def _init_worker(box, cation_local, is_coordinator, method, cutoff, max_distance_ratio):
    _W.update(box=box, cation_local=cation_local, is_coordinator=is_coordinator, method=method,
              cutoff=cutoff, max_distance_ratio=max_distance_ratio)


def _frame_shells(positions):
    """Shell rows [cation, local_atom] for one frame of the heavy-atom positions."""
    box, cat = _W["box"], _W["cation_local"]
    wrapped = np.mod(positions, box)
    wrapped[wrapped >= box] = 0.0
    tree = cKDTree(wrapped, boxsize=box)
    out = []
    for c, neighbours in enumerate(tree.query_ball_point(wrapped[cat], _W["cutoff"])):
        idx = np.array([i for i in neighbours if i != cat[c]], dtype=np.int64)
        vec = positions[idx] - positions[cat[c]]
        vec -= box * np.round(vec / box)
        dist = np.linalg.norm(vec, axis=1)
        order = np.argsort(dist, kind="stable")
        idx, vec, dist = idx[order], vec[order], dist[order]
        coord = _W["is_coordinator"][idx]
        if _W["method"] == "convex_hull":
            other = np.flatnonzero(~coord)[0]
            n = convex_hull_shell_size(vec[coord], dist[coord], vec[other], _W["max_distance_ratio"])
            members = idx[coord][:n]
        else:
            members = idx[coord]
        out.append(np.column_stack([np.full(len(members), c), members]))
    return np.concatenate(out)


def _rebatch(blocks, size):
    """Concatenate frame blocks into batches of at least ``size`` frames (keeps all workers busy)."""
    buffer = []
    for block in blocks:
        buffer.append(block)
        if sum(len(b) for b in buffer) >= size:
            yield np.concatenate(buffer)
            buffer = []
    if buffer:
        yield np.concatenate(buffer)


def compute_shells(topology, frames, timestep, coordinators=("O", "N", "F"), anion_center="N",
                   method="convex_hull", cutoff=6.0, max_distance_ratio=1.75, n_workers=1):
    """Coordination shells of all cations for a stream of frames.

    topology: ``io.Topology``. frames: iterable of (k, n_atoms, 3) position blocks
    (e.g. ``io.iter_hitpoly_frames``). timestep: ps between yielded frames.
    method: "convex_hull" (``cutoff`` is the neighbour search radius) or "cutoff"
    (every coordinator within ``cutoff``). anion_center: element tracked as the
    anion position (one per anion residue, e.g. "N" for TFSI/FSI, "P" for PF6).
    """
    if method not in ("convex_hull", "cutoff"):
        raise ValueError(f"unknown method: {method!r}")
    kinds = topology.kinds
    heavy = np.flatnonzero(topology.elements != "H")         # H never coordinates nor blocks a shell
    cation_atoms = np.flatnonzero(kinds == "cation")
    anion_atoms = np.flatnonzero((kinds == "anion") & (topology.elements == anion_center))
    anion_res = topology.residues[anion_atoms]
    assert len(np.unique(anion_res)) == len(anion_res), "anion_center must appear once per anion"
    local = np.full(topology.n_atoms, -1)
    local[heavy] = np.arange(len(heavy))
    is_coordinator = np.isin(topology.elements[heavy], coordinators) & (kinds[heavy] != "cation")

    residue_kind = np.full(topology.residues.max() + 1, -1)
    residue_kind[topology.residues] = [KINDS.index(k) for k in kinds]

    init = (topology.box, local[cation_atoms], is_coordinator, method, cutoff, max_distance_ratio)
    rows, cat_pos, an_pos = [], [], []
    frame = 0
    with Pool(n_workers, initializer=_init_worker, initargs=init) as pool:
        for block in _rebatch(frames, 8 * n_workers):
            for shell in pool.imap(_frame_shells, block[:, heavy], chunksize=4):
                atoms = heavy[shell[:, 1]]
                rows.append(np.column_stack([np.full(len(shell), frame), shell[:, 0],
                                             topology.residues[atoms], atoms]))
                frame += 1
            cat_pos.append(block[:, cation_atoms])
            an_pos.append(block[:, anion_atoms[np.argsort(anion_res)]])
    rows = np.concatenate(rows)
    rows = rows[np.lexsort((rows[:, 3], rows[:, 2], rows[:, 0], rows[:, 1]))]
    return Shells(rows, residue_kind, np.concatenate(cat_pos), np.concatenate(an_pos), timestep, cation_atoms)
