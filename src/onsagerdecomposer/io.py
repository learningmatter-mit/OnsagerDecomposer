"""Reading HiTPoly-style text trajectories (liquid and polymer electrolytes).

A HiTPoly analysis folder contains:
  atom_names_rdf_all.txt      one line per atom: element,type,residue_id  (e.g. "O,PL1,3")
  xyz_unwrapped_rdf_all.txt   one line per atom per frame: x,y,z  (unwrapped, Angstrom)
  box_len.txt                 cubic box length (Angstrom)

The type label prefix sets the residue kind: PL* -> "solvent" (solvent molecule or
polymer chain), AN* -> "anion", CA* -> "cation".
"""
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

KIND_PREFIXES = {"PL": "solvent", "AN": "anion", "CA": "cation"}

# Atomic masses (pymatgen values) used for the centre-of-mass correction.
MASSES = {"H": 1.00794, "Li": 6.941, "B": 10.811, "C": 12.0107, "N": 14.0067, "O": 15.9994, "F": 18.9984032,
          "Na": 22.98976928, "Mg": 24.305, "Al": 26.9815386, "Si": 28.0855, "P": 30.973762, "S": 32.065,
          "Cl": 35.453, "K": 39.0983, "Ti": 47.867, "Zn": 65.409, "Ga": 69.723, "Ge": 72.64, "Se": 78.96,
          "Br": 79.904, "Y": 88.90585, "Zr": 91.224, "Nb": 92.90638, "I": 126.90447, "La": 138.90547,
          "Ta": 180.94788}


@dataclass
class Topology:
    elements: np.ndarray   # (n_atoms,) element symbols
    types: np.ndarray      # (n_atoms,) HiTPoly type labels, e.g. "PL1"
    residues: np.ndarray   # (n_atoms,) residue ids
    box: float             # cubic box length in Angstrom

    @property
    def kinds(self):
        """(n_atoms,) "solvent", "anion" or "cation" from the type prefix."""
        return np.array([KIND_PREFIXES[t[:2]] for t in self.types])

    @property
    def n_atoms(self):
        return len(self.elements)


def read_hitpoly_topology(folder):
    folder = Path(folder)
    table = pd.read_csv(folder / "atom_names_rdf_all.txt", header=None, names=["element", "type", "residue"],
                        skipinitialspace=True)
    box = float((folder / "box_len.txt").read_text())
    return Topology(table["element"].str.strip().to_numpy(), table["type"].str.strip().to_numpy(),
                    table["residue"].to_numpy(dtype=np.int64), box)


def iter_hitpoly_frames(folder, topology, stride=1, com="solvent", chunk_frames=100):
    """Yield (frame_positions, ...) blocks of shape (k, n_atoms, 3), keeping every ``stride``-th frame.

    Frames are unwrapped against the previous frame (no jump larger than half a box)
    and, with ``com="solvent"``, shifted so the mass-weighted centre of the solvent
    atoms stays at the origin. ``com=None`` leaves positions as stored.
    """
    n = topology.n_atoms
    box = topology.box
    if com == "solvent":
        solvent = topology.kinds == "solvent"
        masses = np.array([MASSES[e] for e in topology.elements])[solvent]
    elif com is not None:
        raise ValueError(f"unknown com correction: {com!r}")

    prev = None      # last (unwrapped) frame read, used to unwrap the next one
    offset = 0       # global index of the first frame in the current block
    reader = pd.read_csv(Path(folder) / "xyz_unwrapped_rdf_all.txt", header=None, dtype=np.float64,
                         engine="c", chunksize=n * chunk_frames)
    for block in reader:
        xyz = block.to_numpy()
        if len(xyz) % n:
            warnings.warn(f"trajectory ends with an incomplete frame ({len(xyz) % n} of {n} atoms); dropping it")
            xyz = xyz[: len(xyz) - len(xyz) % n]
        xyz = xyz.reshape(-1, n, 3)
        if len(xyz) == 0:
            continue
        first = xyz[0] if prev is None else prev
        steps = np.diff(xyz, axis=0, prepend=first[None])
        steps -= box * np.round(steps / box)
        xyz = first + np.cumsum(steps, axis=0)
        prev = xyz[-1]
        keep = (np.arange(offset, offset + len(xyz)) % stride) == 0
        offset += len(xyz)
        xyz = xyz[keep]
        if len(xyz) == 0:
            continue
        if com == "solvent":
            centre = np.einsum("a,kad->kd", masses, xyz[:, solvent]) / masses.sum()
            xyz = xyz - centre[:, None, :]
        yield xyz
