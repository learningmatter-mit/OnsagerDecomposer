"""Reproduce the archived LATP (x = 1/3, 800 K, skip 10) results from the original trajectory.

python benchmarks/crystal_latp.py <crystal_old_data folder> [n_workers]
"""
import pickle
import sys
import time
from pathlib import Path

import numpy as np

from onsagerdecomposer import decompose, self_diffusion_table, transitions
from onsagerdecomposer.crystal import (activated_states, compute_site_shells, crystal_events, iter_ase_frames,
                                       load_site_network, read_ase_topology)


def compare(name, new, old):
    scale = np.abs(old["self_values"]).max()
    err = max(np.abs(new.self_terms - old["self_values"]).max(),
              np.abs(new.distinct_terms - old["distinct_values"]).max()) / scale
    print(f"  {name:26s} max rel. diff {err:.1e}")
    return err


def main():
    folder = Path(sys.argv[1])
    n_workers = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    tag = "latp_skip10_neighbor2_800K_neighbormode_both_alignment_0.0_concentration_2"
    traj = folder / "stuff2_800K.traj"
    elements, cell = read_ase_topology(traj)
    network = load_site_network(folder / "supercell_network_s2_wyckoff.gpickle")
    t = time.time()
    shells = compute_site_shells(elements, cell, iter_ase_frames(traj, stride=10), network, timestep=1.0,
                                 n_workers=n_workers)
    print(f"site shells: {time.time() - t:.0f}s for {shells.n_frames} frames, {shells.n_cations} Li")
    shells.save(folder / "new_shells_skip10.npz")

    occ = pickle.load(open(folder / f"occurences_raw_{tag}.pickle", "rb"))
    old = pickle.load(open(folder / f"onsager_{tag}.pickle", "rb"))
    events = crystal_events(shells)
    counts = np.bincount(events.labels.ravel(), minlength=5)
    print("  event counts new   :", dict(zip(events.names, counts)))
    print("  event counts legacy:", [occ["simplified_data"][i] for i in range(5)])
    states, names = activated_states(shells)
    tr = transitions(states, names)
    print("  A/non-A counts new :", np.bincount(tr.labels.ravel(), minlength=4), " legacy:",
          [occ["markov_1"][i] for i in range(4)])

    cd, zeros = shells.cation_displacements(), np.zeros((shells.n_cations, shells.n_frames - 1, 3))
    errs = [compare("events (event mode)", decompose(cd, events.labels, events.names, zeros),
                    old["simplified_transitions"]["event"]),
            compare("events (particle mode)", decompose(cd, np.zeros_like(events.labels), ["Li"], zeros),
                    old["simplified_transitions"]["particle"]),
            compare("A/non-A (event mode)", decompose(cd, tr.labels, tr.names, zeros),
                    old["markov_transitions"][1]["event"])]
    # transport table with the paper settings (0.1 ps frames x 10, fit over 8-10 % of the lags)
    shells.timestep = 1.0
    from ase.io.trajectory import Trajectory
    volume = Trajectory(traj)[0].get_volume()
    corr = decompose(cd, events.labels, events.names, zeros, distinct=False)
    table = self_diffusion_table(corr, events, shells.timestep, volume, 800, fit_range=(0.08, 0.1))
    print(table)
    old_self = old["simplified_transitions"]["event"]["self_values"].sum(axis=0)[:5]
    lo, hi = int(0.08 * old_self.shape[1]), int(0.1 * old_self.shape[1])
    lags = np.arange(old_self.shape[1])[lo:hi]
    old_D = np.array([np.polyfit(lags, c[lo:hi], 1)[0] for c in old_self]) / 6 / 32 * 1e-4
    errs.append(np.abs(table["D_self"].iloc[:5].to_numpy() / old_D - 1).max())
    print(f"  D_self vs legacy fit          max rel. diff {errs[-1]:.1e}")
    assert max(errs) < 1e-8, errs


if __name__ == "__main__":
    main()
