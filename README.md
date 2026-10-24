<p align="center">
  <img src="docs/logo.svg" width="150" alt="OnsagerDecomposer logo">
</p>

<h1 align="center">OnsagerDecomposer</h1>

<p align="center">
  Which microscopic events carry the ions, how often they happen, and how effective each one is —<br>
  an exact, additive decomposition of transport coefficients from molecular dynamics.
</p>

<p align="center">
  <a href="https://github.com/learningmatter-mit/OnsagerDecomposer/actions/workflows/tests.yml"><img src="https://github.com/learningmatter-mit/OnsagerDecomposer/actions/workflows/tests.yml/badge.svg" alt="tests"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="MIT license"></a>
  <img src="https://img.shields.io/badge/python-3.9%2B-blue.svg" alt="Python 3.9+">
</p>

---

OnsagerDecomposer splits the self-diffusion coefficient, the Nernst–Einstein conductivity and the
full Onsager matrix of an MD trajectory into contributions of physically defined events — vehicular
motion, solvent or anion exchange, intra- and interchain hops, single-ion and concerted hops, or
transitions between solvation states such as SSIP, CIP and AGG. The contributions sum **exactly** to
the conventional Einstein / Green–Kubo values, and scanning the window length Δt shows on which
timescales each mechanism emerges and dominates.

It works for **liquid**, **polymer** and **crystalline** ion conductors.

## How it works

| Step | What happens | API |
|---|---|---|
| 1. Shells | First coordination shell of every cation in every frame (convex hull, distance cutoff, or crystal sites) | `compute_shells`, `crystal.compute_site_shells` |
| 2. Events | Every window [t, t+Δt] of every cation gets exactly one event label | `liquid_events`, `polymer_events`, `crystal.crystal_events`, `transitions` |
| 3. Decomposition | Each cation is split into one virtual particle per event; event-resolved self and distinct correlations from FFTs | `decompose` |
| 4. Transport | D_self, σ_NE, share and effectiveness (D^m / p_m) of every event | `self_diffusion_table`, `per_macrostate` |

## Installation

```bash
git clone https://github.com/learningmatter-mit/OnsagerDecomposer.git
cd OnsagerDecomposer
pip install -e .              # liquids and polymers
pip install -e ".[crystal]"   # + crystals (ase, networkx)
```

## Quick start: liquid or polymer electrolyte

Input is a HiTPoly analysis folder with `atom_names_rdf_all.txt` (element, type, residue id per
atom), `xyz_unwrapped_rdf_all.txt` (unwrapped coordinates) and `box_len.txt`.

```python
import onsagerdecomposer as od

folder = "path/to/simulation"
top = od.read_hitpoly_topology(folder)
frames = od.iter_hitpoly_frames(folder, top, stride=1)        # unwrapped, solvent-COM frame
shells = od.compute_shells(top, frames, timestep=2.0,           # ps between frames
                           anion_center="N", n_workers=8)       # N for TFSI/FSI, P for PF6
shells.save("shells.npz")                                       # compute once, reuse for every Δt

s = shells.subsample(10)                                        # Δt = 10 frames = 20 ps
events = od.liquid_events(s)                                    # or od.polymer_events(s)
corr = od.decompose(s.cation_displacements(), events.labels, events.names, s.anion_displacements())
table = od.self_diffusion_table(corr, events, s.timestep, volume=top.box**3, temperature=298,
                                fit_range=(0.05, 0.10))
print(table)   # probability, D_self, sigma_NE, share, effective_D_self, effective_sigma_NE per event
```

SSIP / CIP / AGG macrostates:

```python
states, names = od.solvation_states(s)
tr = od.transitions(states, names)                              # SSIP->SSIP, SSIP->CIP, ...
corr = od.decompose(s.cation_displacements(), tr.labels, tr.names, s.anion_displacements())
print(od.per_macrostate(od.self_diffusion_table(corr, tr, s.timestep, top.box**3, 298), names))
```

`examples/decompose_hitpoly.py` runs the full Δt scan from the command line.

## Quick start: crystalline conductor

Input is an ASE trajectory and a network of crystallographic sites (a networkx graph whose nodes
carry the framework-anion indices of each site polyhedron, e.g. the 6b/18e/36f sites of NASICON).

```python
from onsagerdecomposer import crystal, decompose, self_diffusion_table

elements, cell = crystal.read_ase_topology("latp_800K.traj")
network = crystal.load_site_network("supercell_network_wyckoff.gpickle")
frames = crystal.iter_ase_frames("latp_800K.traj", stride=10)
shells = crystal.compute_site_shells(elements, cell, frames, network, timestep=1.0, n_workers=8)

events = crystal.crystal_events(shells)    # vibration, single-ion hop, aligned / unaligned concerted hop
states, names = crystal.activated_states(shells)   # activated (A) / non-activated macrostates
```

## Fitting window

D_self is the slope of each event's displacement correlation versus lag time, divided by 6 and by
the number of cations. The slope is fitted over a window given as a **fraction of the available lag
range** (`fit_range`), so the same fraction corresponds to different absolute times for different
trajectory lengths. The windows used in the paper:

| System | Frame interval | Shell definition | `fit_range` |
|---|---|---|---|
| LATP (crystal) | 0.1 ps | crystal sites; Li neighbours ≤ 4 Å and ≤ 2 site edges | `(0.08, 0.10)` |
| PEO / PPM + LiTFSI (polymer) | 10 ps | convex hull; O, N, F coordinators | `(0.06, 0.12)` |
| EC / LiPF6, FAN / LiFSI (liquid) | 2 ps | convex hull; O, N, F coordinators | `(0.05, 0.10)` |

Check that the correlations are linear inside the window before trusting a decomposed coefficient;
rare events converge last.

## Defining your own events

An event rule is any function that gives every cation window exactly one label. Helpers report,
for each window, whether a residue (solvent molecule, chain, anion) or a coordinating atom entered or
left the shell:

```python
import numpy as np
import onsagerdecomposer as od

def anion_gain_or_loss(shells):
    entered, left = od.residue_entries_exits(shells, "anion")
    labels = np.select([entered & left, entered, left], [3, 1, 2], default=0)
    return od.Events(labels, ["no_change", "anion_gained", "anion_lost", "anion_swapped"])
```

A macrostate rule returns one state per cation and frame; `od.transitions(states, names)` turns it
into X→Y events that are decomposed the same way.

## Validation

`pytest` runs the unit tests. The scripts in `benchmarks/` reproduce the results of the original
research code (PPM/LiTFSI, FAN/LiFSI and LATP) from the raw trajectories: coordination shells and
event labels are identical, and correlation matrices and transport coefficients agree to better
than 1e-9 relative. Pair sums are replaced by per-event collective sums, so the full self + distinct
decomposition takes seconds rather than cluster-days.

## Citation

If you use OnsagerDecomposer, please cite:

> K. Jun, P. A. Leon, J. Ruža, J. Nam, B. D. McCloskey, R. Gómez-Bombarelli,
> *Environment-mediated transport as a general principle of fast ion conduction* (2026).

## Contact

KyuJung Jun — School of Mechanical Engineering, Korea University
([kyujung@korea.ac.kr](mailto:kyujung@korea.ac.kr), [kyujung@mit.edu](mailto:kyujung@mit.edu)).
Questions, bug reports and new event definitions are also welcome as
[GitHub issues](https://github.com/learningmatter-mit/OnsagerDecomposer/issues).

## License

MIT — see [LICENSE](LICENSE).
