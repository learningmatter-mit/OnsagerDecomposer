# Future ideas (archived, not implemented)

## 1. Learned solvation-environment representation → automatic event discovery

Goal: find which kinds of shell changes exist in a system without hand-written rules,
then map them to labels people can interpret.

Sketch:
- Represent each cation shell per frame as a fixed-size vector, e.g. a local-environment
  descriptor (SOAP/ACE) of the coordinating atoms, or a small graph of coordinating residues
  (kinds, number of contacts, residue identities kept as a set).
- Learn an embedding z (autoencoder or contrastive model; frames close in time = positives).
- A window [t, t+Δt] becomes the pair (z_t, z_{t+Δt}) or the difference Δz. Cluster these
  transition vectors; each cluster is a candidate event.
- Interpret clusters with the features already computed by `events.py` (did a solvent/anion/chain
  enter or leave, did the majority chain change atoms, …) — e.g. report, per cluster, the
  fraction of windows with each feature.

Fit with the current code: any clustering gives one integer label per window, so the result is
an `Events(labels, names)` object and plugs straight into `decompose` and `self_diffusion_table`.
Additivity is kept automatically as long as every window gets exactly one label.

Open questions: Δt dependence of the clusters (discover at one Δt or per Δt?); permutation
invariance of identical molecules; whether rare but effective events (interchain hops in PEO)
survive clustering, since they are a small share of windows.

## 2. Distance-limited distinct correlations to speed up convergence

Observation: distinct terms Σ_{i≠j} ⟨d_i·d_j⟩ sum over all pairs in the box. Pairs far apart
add mostly noise (variance grows with the number of pairs) while the physical correlation
likely decays with distance.

Idea: only count pairs closer than r_c at the start of the window,

    C_distinct(τ; r_c) = ⟨ Σ_{i≠j, r_ij(t) < r_c} d_i(t, τ) · d_j(t, τ) ⟩_t ,

and study convergence in r_c (look for a plateau).

Notes to keep in mind:
- The pair set changes with the time origin t, so the FFT trick used for full sums no longer
  applies directly. A neighbour-list sum over origins costs O(T · n_τ · pairs within r_c), which
  is still far below all pairs if r_c is small. Restricting τ to the range used for the slope
  fit keeps it cheap.
- Truncation changes the estimator: it equals the full distinct term only if correlations beyond
  r_c vanish. In periodic boxes with a fixed reference frame (e.g. solvent centre of mass),
  momentum conservation adds a small long-range, system-size-dependent contribution spread over
  all pairs. This must be checked (e.g. compare r_c → L/2 with the full FFT result, and repeat for
  two box sizes).
- A distance-resolved version C_distinct(τ; r) in shells [r, r + dr] would show directly where the
  correlation lives and how much of the noise comes from far pairs.
- The legacy code had an early attempt (`binmethod="DISTANCE"/"NUMPARTICLE"` in
  `legacy/onsagerdecomposer_v053.py`) that masked pairs by distance at both ends of each step; it
  was not used for the paper.
- Event decomposition carries over unchanged: replace d_i by the event virtual-particle
  displacement d_i^m.
