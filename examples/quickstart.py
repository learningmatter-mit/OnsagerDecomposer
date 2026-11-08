"""Run OnsagerDecomposer on the example data in examples/data/ (excerpts of the paper's MD runs).

python examples/quickstart.py

1. Input format: coordination shells from 10 frames of FAN/LiFSI HiTPoly input files.
2. Liquid (1.3 M LiFSI in FAN, 298 K, 2 ns every 2 ps): events and SSIP/CIP/AGG states.
3. Polymer (LiTFSI in PPM, 443 K, 500 ns every 500 ps): polymer events at dt = 0.5, 1, 2 ns.
"""
from pathlib import Path

import numpy as np
import pandas as pd

import onsagerdecomposer as od

DATA = Path(__file__).parent / "data"
pd.set_option("display.width", 160, "display.max_columns", None, "display.precision", 3)


def load(name):
    """Shells plus the box (A) and temperature (K) stored with them."""
    f = np.load(DATA / f"{name}_shells.npz")
    return od.Shells.load(DATA / f"{name}_shells.npz"), float(f["box"]), float(f["temperature"])


def table(shells, rule, k, box, temperature, fit_range):
    s = shells.subsample(k)
    events = rule(s)
    corr = od.decompose(s.cation_displacements(), events.labels, events.names, s.anion_displacements(),
                        distinct=False)
    return od.self_diffusion_table(corr, events, s.timestep, box**3, temperature, fit_range), s


def main():
    # 1. From HiTPoly input files to shells (the step that needs the full trajectory)
    folder = DATA / "fan_lifsi_input"
    top = od.read_hitpoly_topology(folder)
    shells = od.compute_shells(top, od.iter_hitpoly_frames(folder, top), timestep=2.0, anion_center="N",
                               n_workers=2)
    print(f"1. {shells.n_frames} frames of {top.n_atoms} atoms -> {len(shells.rows)} shell rows "
          f"({len(shells.rows) / shells.n_frames / shells.n_cations:.1f} coordinating atoms per Li)\n")

    # 2. Liquid electrolyte
    fan, box, temperature = load("fan_lifsi")
    for k in (1, 5, 10):                                        # dt = 2, 10, 20 ps
        t, s = table(fan, od.liquid_events, k, box, temperature, (0.05, 0.10))
        print(f"2. FAN/LiFSI, dt = {s.timestep:g} ps\n{t}\n")
    s = fan.subsample(5)
    states, names = od.solvation_states(s)
    tr = od.transitions(states, names)
    corr = od.decompose(s.cation_displacements(), tr.labels, tr.names, s.anion_displacements(), distinct=False)
    print(f"2. FAN/LiFSI macrostates, dt = {s.timestep:g} ps\n"
          f"{od.per_macrostate(od.self_diffusion_table(corr, tr, s.timestep, box**3, temperature), names)}\n")

    # 3. Polymer electrolyte
    ppm, box, temperature = load("ppm_litfsi")
    for k in (1, 2, 4):                                         # dt = 0.5, 1, 2 ns
        t, s = table(ppm, od.polymer_events, k, box, temperature, (0.06, 0.12))
        print(f"3. PPM/LiTFSI, dt = {s.timestep / 1000:g} ns\n{t}\n")


if __name__ == "__main__":
    main()
