# Example data

Small excerpts of the MD runs in the paper ([arXiv:2602.16636](https://arxiv.org/abs/2602.16636)), enough to
run every step of OnsagerDecomposer in seconds. `python examples/quickstart.py` uses all of them.

| File | System | Content |
|---|---|---|
| `fan_lifsi_input/` | 1.3 M LiFSI in fluoroacetonitrile (FAN), 298 K | HiTPoly input files, first 10 frames (2 ps apart), 11,610 atoms |
| `fan_lifsi_shells.npz` | same run | coordination shells, first 2 ns every 2 ps (1000 frames, 141 Li) |
| `ppm_litfsi_shells.npz` | LiTFSI in PPM, 443 K | coordination shells, the full 500 ns every 500 ps (1000 frames, 288 Li) |

Shells were computed as in the paper: convex-hull first shells, O, N and F as coordinating atoms, TFSI/FSI
tracked by their N. Load them with `od.Shells.load(path)`; each file also holds `box` (cubic box length, Å),
`temperature` (K) and a one-line `note`. Positions are stored as float32.

Event labels compare the shells at the two ends of each window, so shells sampled every 500 ps give the
same labels and coefficients as the full PPM run at Δt = 0.5, 1 and 2 ns (checked: identical labels, D within
2e-7 relative; `tests/test_example_data.py`). Shorter windows need the full trajectory.

Excerpts this short are for learning the workflow; the liquid's 2 ns gives usable values for Δt up to tens of ps.
If you use these data, please cite the paper.
