# Changelog

## Unreleased

### Fixed
- Copolymers: HiTPoly type labels P1*, P2*, ... (one per monomer block) are read as polymer, as in the
  original analysis code. Before, any label other than PL*, AN* or CA* stopped `compute_shells` with a KeyError.

## 1.0.0 — 2026-10-25

First public release: a rewrite of the research code (v0.5.3) used for the paper
*Environment-mediated transport as a general principle of fast ion conduction* ([arXiv:2602.16636](https://arxiv.org/abs/2602.16636)).

### Added
- Installable package (`pip install -e .`) with one pipeline for liquids, polymers and crystals:
  shells → events → `decompose` → `self_diffusion_table`.
- `compute_shells`: convex-hull or cutoff coordination shells, computed once at the finest stride
  and subsampled for every window length (`Shells.subsample`), in parallel over frames.
- Event rules as small functions: `liquid_events`, `polymer_events`, `crystal.crystal_events`;
  macrostates `solvation_states` (SSIP/CIP/AGG), `crystal.activated_states` (A/non-A) and
  `transitions`. Helpers `residue_entries_exits` / `atom_entries_exits` for writing new rules.
- `self_diffusion_table` and `per_macrostate`: D_self, σ_NE, share and effectiveness per event,
  with the fit window as a parameter (paper values listed in the README).
- Crystal support: ASE trajectory reader and site-network shells (`onsagerdecomposer.crystal`).
- Unit tests and CI workflow; validated against archived results of the legacy code.

### Changed
- Correlations are computed from per-event collective sums instead of all particle pairs; the full
  self + distinct decomposition takes seconds instead of hours to days.
- SSIP/CIP/AGG follows the paper definition (AGG: two or more anions, or an anion shared with
  another cation), as in v0.5.3. Results produced with v0.4.2 counted only shared anions.

### Fixed (relative to the legacy code)
- Homopolymer shells no longer fail on the copolymer-only `solvating_types` key.
- Undecomposed (particle-mode) results no longer assume equal numbers of cations and anions.
- Trajectories ending in an incomplete frame are read with a warning instead of an error.

### Removed
- The legacy module is kept in `legacy/` for reference only and is not installed.
