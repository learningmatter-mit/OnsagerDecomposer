"""Compare new shell detection with an archived legacy raw pickle (polymer or liquid).

python benchmarks/polymer_shells.py <hitpoly_folder> <legacy_raw_pickle> <stride> <anion_center> <out.npz> [n_workers]

The legacy pickle must hold a PolymerElectrolyteAnalyzer / LiquidElectrolyteAnalyzer
computed with stepskip == stride. Writes the new Shells to out.npz.
"""
import pickle
import sys
import time
from pathlib import Path

import numpy as np

from onsagerdecomposer.io import iter_hitpoly_frames, read_hitpoly_topology
from onsagerdecomposer.shells import compute_shells


def legacy_rows(old):
    """[frame, cation, residue, atom] rows from a legacy analyzer (atom = true atom index)."""
    cations = sorted(old.complete_coord_info)
    out = []
    back = getattr(old, "anion_indices_back", None)
    polymer = type(old).__name__ == "PolymerElectrolyteAnalyzer"   # polymer stored sequential coordinator ids
    for c, li in enumerate(cations):
        f = old.complete_coord_info[li]["flattened_O_providers"]
        atoms = np.array([back[a] for a in f[:, 2]]) if polymer else f[:, 2]
        out.append(np.column_stack([f[:, 0], np.full(len(f), c), f[:, 1], atoms]))
    rows = np.concatenate(out).astype(np.int64)
    return rows[np.lexsort((rows[:, 3], rows[:, 2], rows[:, 0], rows[:, 1]))], cations


def main():
    folder, pkl, stride, anion_center, out = sys.argv[1:6]
    n_workers = int(sys.argv[6]) if len(sys.argv) > 6 else 8
    stride = int(stride)
    sys.path.insert(0, str(Path(pkl).parent))      # legacy module stored next to the pickle
    old = pickle.load(open(pkl, "rb"))
    old_rows, cations = legacy_rows(old)

    top = read_hitpoly_topology(folder)
    t = time.time()
    new = compute_shells(top, iter_hitpoly_frames(folder, top, stride=stride), timestep=1.0 * stride,
                         anion_center=anion_center, n_workers=n_workers)
    print(f"new shells: {time.time() - t:.0f}s for {new.n_frames} frames, {new.n_cations} cations")
    new.save(out)

    n_frames = min(new.n_frames, old_rows[:, 0].max() + 1)
    a = {tuple(r) for r in new.rows[new.rows[:, 0] < n_frames]}
    b = {tuple(r) for r in old_rows}
    print(f"frames compared {n_frames}; rows new {len(a)}, legacy {len(b)}, only-new {len(a - b)}, only-legacy {len(b - a)}")
    diff_frames = {(r[0], r[1]) for r in (a ^ b)}
    print(f"(frame, cation) shells that differ: {len(diff_frames)} of {n_frames * len(cations)}")

    pos_old = np.stack([old.complete_coord_info[li]["cartesian_positions"] for li in cations], axis=1)
    print("max |cation position diff| (A):", np.abs(new.cation_positions[:n_frames] - pos_old[:n_frames]).max())
    an_old = np.stack([old.anion_displacements[k]["disp"] for k in sorted(old.anion_displacements)])
    m = n_frames - 1
    print("max |anion displacement diff| (A):", np.abs(new.anion_displacements()[:, :m] - an_old[:, :m]).max())


if __name__ == "__main__":
    main()
