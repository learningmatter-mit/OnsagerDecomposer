from pathlib import Path

import numpy as np

import onsagerdecomposer as od

DATA = Path(__file__).resolve().parents[1] / "examples" / "data"


def test_input_files_give_the_shipped_shells():
    folder = DATA / "fan_lifsi_input"
    top = od.read_hitpoly_topology(folder)
    new = od.compute_shells(top, od.iter_hitpoly_frames(folder, top), timestep=2.0, anion_center="N", n_workers=1)
    ref = od.Shells.load(DATA / "fan_lifsi_shells.npz")
    assert np.array_equal(new.rows, ref.rows[ref.rows[:, 0] < new.n_frames])
    assert np.allclose(new.cation_positions, ref.cation_positions[:new.n_frames], atol=1e-3)


def test_polymer_excerpt_reproduces_paper_value():
    f = np.load(DATA / "ppm_litfsi_shells.npz")
    s = od.Shells.load(DATA / "ppm_litfsi_shells.npz")
    events = od.polymer_events(s)
    corr = od.decompose(s.cation_displacements(), events.labels, events.names, s.anion_displacements(),
                        distinct=False)
    t = od.self_diffusion_table(corr, events, s.timestep, float(f["box"]) ** 3, float(f["temperature"]), (0.06, 0.12))
    assert np.isclose(t.loc["total", "D_self"], 7.616e-8, rtol=1e-3)        # paper run, dt = 0.5 ns
