"""Crystalline conductors: shells from crystallographic sites (paper Fig. 3, LATP).

Each mobile ion (Li) is assigned to the site of a precomputed site network whose
polyhedron of framework anions (e.g. the O of a 6b/18e/36f site) contains it. Two
ions are neighbours when they are closer than ``neighbor_cutoff`` and their sites
are at most ``neighbor_edges`` edges apart in the network. The shell of an ion is
its site's framework anions plus its neighbouring ions.

The site network is a networkx graph (as in ``supercell_network_s*_wyckoff.gpickle``):
nodes 0..n-1 with attribute ``neigh`` = framework-anion indices (counted from the
first framework anion) of the site polyhedron; edges join face-sharing sites.
Requires ``ase`` (trajectory reading) and ``networkx`` (site network).
"""
import pickle
from multiprocessing import Pool

import numpy as np
from scipy.spatial import ConvexHull

from .events import Events
from .io import MASSES
from .shells import KINDS, Shells


def read_ase_topology(path):
    """(elements, cell) of the first frame of an ASE trajectory."""
    from ase.io.trajectory import Trajectory
    atoms = Trajectory(path)[0]
    return np.array(atoms.get_chemical_symbols()), np.array(atoms.cell[:])


def iter_ase_frames(path, stride=1, com="framework", mobile="Li", chunk_frames=1000):
    """Yield (k, n_atoms, 3) blocks of unwrapped positions from an ASE trajectory (fixed cell).

    Frames are unwrapped against the previous frame in fractional coordinates; with
    ``com="framework"`` the mass-weighted centre of all non-``mobile`` atoms is held at the origin.
    """
    from ase.io.trajectory import Trajectory
    traj = Trajectory(path)
    elements, cell = read_ase_topology(path)
    inv = np.linalg.inv(cell)
    if com == "framework":
        frame_atoms = elements != mobile
        masses = np.array([MASSES[e] for e in elements])[frame_atoms]
    elif com is not None:
        raise ValueError(f"unknown com correction: {com!r}")
    prev = None
    for start in range(0, len(traj), chunk_frames):
        frac = np.stack([traj[i].get_positions() for i in range(start, min(start + chunk_frames, len(traj)))]) @ inv
        first = frac[0] if prev is None else prev
        steps = np.diff(frac, axis=0, prepend=first[None])
        steps -= np.round(steps)
        frac = first + np.cumsum(steps, axis=0)
        prev = frac[-1]
        xyz = frac[(np.arange(start, start + len(frac)) % stride) == 0] @ cell
        if len(xyz) == 0:
            continue
        if com == "framework":
            xyz = xyz - (np.einsum("a,kad->kd", masses, xyz[:, frame_atoms]) / masses.sum())[:, None, :]
        yield xyz


def load_site_network(path):
    with open(path, "rb") as f:
        return pickle.load(f)


_C = {}


def _init_worker(constants):
    _C.update(constants)


def _hull_vertices(points):
    return len(ConvexHull(points).vertices)


def _assign_site(vec, rel):
    """Site of one ion from its framework anions (vectors from the ion, sorted by distance; relative ids)."""
    neigh, o_to_nodes = _C["neigh"], _C["o_to_nodes"]
    candidates = sorted(set().union(*(o_to_nodes.get(o, ()) for o in rel[:4])))
    origin = np.zeros((1, 3))
    best, best_dv = None, np.inf
    for node in candidates:                                     # site whose polyhedron contains the ion
        mask = np.isin(rel, neigh[node])
        pts = vec[mask]
        if len(pts) not in (4, 6):
            continue
        if _hull_vertices(pts) == _hull_vertices(np.vstack([pts, origin])):
            return node, mask
        dv = ConvexHull(np.vstack([pts, origin])).volume - ConvexHull(pts).volume
        if dv < best_dv:
            best, best_dv = node, dv
    if best is None or best_dv >= _C["volume_tolerance"]:
        raise ValueError("ion is not inside any site polyhedron; the framework may have changed "
                         f"(closest site volume mismatch {best_dv:.2f} A^3)")
    return best, np.isin(rel, neigh[best])                     # otherwise: smallest volume increase


def _frame_sites(positions):
    cell, inv = _C["cell"], _C["inv_cell"]
    ions, anions = _C["ions"], _C["anions"]

    def min_image(d):
        f = d @ inv
        return (f - np.round(f)) @ cell

    sites = np.empty(len(ions), dtype=np.int64)
    rows = []
    for c, ion in enumerate(ions):
        vec = min_image(positions[anions] - positions[ion])
        dist = np.linalg.norm(vec, axis=1)
        near = np.flatnonzero(dist <= _C["search_radius"])
        near = near[np.argsort(dist[near], kind="stable")]
        sites[c], mask = _assign_site(vec[near], anions[near] - _C["first_anion"])
        rows.append(np.column_stack([np.full(mask.sum(), c), anions[near][mask]]))
    for c, ion in enumerate(ions):                              # neighbouring ions (needs all sites first)
        d = np.linalg.norm(min_image(positions[ions] - positions[ion]), axis=1)
        nb = np.flatnonzero((d <= _C["neighbor_cutoff"]) & _C["within"][sites[c], sites])
        nb = nb[nb != c]
        rows.append(np.column_stack([np.full(len(nb), c), ions[nb]]))
    return sites, np.concatenate(rows)


def compute_site_shells(elements, cell, frames, site_network, timestep, mobile="Li", framework_anion="O",
                        neighbor_edges=2, neighbor_cutoff=4.0, search_radius=4.8, volume_tolerance=10.0,
                        n_workers=1):
    """Site-based shells for a crystalline conductor; ``Shells.sites`` holds the occupied site per frame.

    elements/cell: from ``read_ase_topology``; frames: iterable of (k, n_atoms, 3) blocks
    (e.g. ``iter_ase_frames``); site_network: networkx graph (``load_site_network``).
    """
    import networkx as nx
    ions = np.flatnonzero(elements == mobile)
    anions = np.flatnonzero(elements == framework_anion)
    nodes = list(site_network.nodes)
    assert nodes == list(range(len(nodes))), "site network nodes must be 0..n-1"
    neigh = [np.array(sorted(site_network.nodes[n]["neigh"])) for n in nodes]
    o_to_nodes = {}
    for n in nodes:
        for o in neigh[n]:
            o_to_nodes.setdefault(o, set()).add(n)
    within = np.zeros((len(nodes), len(nodes)), dtype=bool)
    for n in nodes:
        within[n, list(nx.single_source_shortest_path_length(site_network, n, cutoff=neighbor_edges))] = True

    init = dict(cell=cell, inv_cell=np.linalg.inv(cell), ions=ions, anions=anions, first_anion=anions.min(),
                neigh=neigh, o_to_nodes=o_to_nodes, within=within, search_radius=search_radius,
                neighbor_cutoff=neighbor_cutoff, volume_tolerance=volume_tolerance)
    residue_kind = np.full(len(elements), -1)
    residue_kind[anions] = KINDS.index("anion")
    residue_kind[ions] = KINDS.index("cation")
    rows, sites, positions = [], [], []
    frame = 0
    with Pool(n_workers, initializer=_init_worker, initargs=(init,)) as pool:
        for block in frames:
            for s, r in pool.imap(_frame_sites, block, chunksize=8):
                sites.append(s)
                rows.append(np.column_stack([np.full(len(r), frame), r[:, 0], r[:, 1], r[:, 1]]))
                frame += 1
            positions.append(block[:, ions])
    rows = np.concatenate(rows)
    rows = rows[np.lexsort((rows[:, 3], rows[:, 2], rows[:, 0], rows[:, 1]))]
    cat = np.concatenate(positions)
    return Shells(rows, residue_kind, cat, np.zeros((len(cat), 0, 3)), timestep, ions, np.stack(sites))


def crystal_events(shells, aligned_cos=0.0):
    """Local vibration, vibration while a neighbour hops, single-ion hop, and concerted hops.

    A hop is a change of occupied site. A concerted hop is a hop while at least one
    neighbouring ion (at the start or end of the window) also hops; it is aligned if the
    largest cosine between the two hop vectors is >= ``aligned_cos`` (0: angle <= 90 deg).
    """
    hop = (shells.sites[1:] != shells.sites[:-1]).T                      # (n_ion, n_windows)
    disp = shells.cation_displacements()
    local = np.full(shells.residue_kind.shape, -1)
    local[shells.cation_atoms] = np.arange(shells.n_cations)
    nb = shells.rows[shells.row_kinds() == KINDS.index("cation")]
    f, c, n = nb[:, 0], nb[:, 1], local[nb[:, 3]]
    last = shells.n_frames - 1
    c = np.concatenate([c[f < last], c[f > 0]])                          # neighbours at start or end
    n = np.concatenate([n[f < last], n[f > 0]])
    t = np.concatenate([f[f < last], f[f > 0] - 1])
    moving = hop[n, t]
    c, n, t = c[moving], n[moving], t[moving]
    a, b = disp[c, t], disp[n, t]
    cos = np.einsum("kd,kd->k", a, b) / np.linalg.norm(a, axis=1) / np.linalg.norm(b, axis=1)
    neighbor_hop = np.zeros_like(hop)
    neighbor_hop[c, t] = True
    best = np.full(hop.shape, -np.inf)
    np.maximum.at(best, (c, t), cos)
    labels = np.select([~hop & ~neighbor_hop, ~hop, ~neighbor_hop, best < aligned_cos], [0, 1, 2, 3], 4)
    return Events(labels, ["local_vibration", "local_vibration_neighbor_hop", "single_ion_hop",
                           "unaligned_concerted_hop", "aligned_concerted_hop"])


def activated_states(shells):
    """A (another ion within the neighbour range) or non-A, per ion and frame."""
    nb = shells.rows[shells.row_kinds() == KINDS.index("cation")]
    states = np.zeros((shells.n_cations, shells.n_frames), dtype=int)
    states[nb[:, 1], nb[:, 0]] = 1
    return states, ["non-A", "A"]
