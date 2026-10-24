"""Reproduce archived legacy Onsager results from the legacy per-frame shells.

python benchmarks/compare_archived.py polymer <mechanism_dir> <skip>
python benchmarks/compare_archived.py liquid  <analysis_dir> <skip>

Shells are rebuilt from the legacy raw pickle (so no trajectory is read); events,
macrostates and correlations are then computed with the new code and compared
with the legacy onsager_*.pickle files of the same skip.
"""
import glob
import pickle
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from polymer_shells import legacy_rows  # noqa: E402

from onsagerdecomposer import Events, Shells, decompose, liquid_events, polymer_events  # noqa: E402
from onsagerdecomposer import solvation_states, transitions  # noqa: E402
from onsagerdecomposer.shells import KINDS  # noqa: E402

# legacy event key -> new event name
POLYMER_KEYS = {"0-1": "anion_exchange", "1-0": "interchain_hop", "1-1": "anion_exchange_interchain_hop",
                "Roll": "complete_intrachain_hop", "LOSE_OR_GAIN": "partial_intrachain_hop",
                "Segmental": "vehicular"}
LIQUID_KEYS = {"0-0": "vehicular", "0-1": "anion_exchange", "1-0": "solvent_exchange",
               "1-1": "solvent_anion_exchange"}


def shells_from_legacy(old, sampling):
    rows, cations = legacy_rows(old)
    kinds = {"polymer": "solvent", "solvent": "solvent", "anion": "anion", "cation": "cation"}
    residue_kind = np.full(max(old.res_id_mapper) + 1, -1)
    for res, k in old.res_id_mapper.items():
        residue_kind[res] = KINDS.index(kinds[k])
    cat = np.stack([old.complete_coord_info[li]["cartesian_positions"] for li in cations], axis=1)
    an_d = np.stack([old.anion_displacements[k]["disp"] for k in old.anion_displacements], axis=1)
    an = np.concatenate([np.zeros((1,) + an_d.shape[1:]), np.cumsum(an_d, axis=0)])
    return Shells(rows, residue_kind, cat, an, sampling)


def polymer_chain_anion_states(shells):
    """Legacy polymer macrostate 'method 1': only chain, chain + anion, only anion, none."""
    has = {}
    for kind in ("solvent", "anion"):
        r = shells.rows[shells.row_kinds() == KINDS.index(kind)]
        m = np.zeros((shells.n_cations, shells.n_frames), dtype=bool)
        m[r[:, 1], r[:, 0]] = True
        has[kind] = m
    states = np.select([has["solvent"] & ~has["anion"], has["solvent"] & has["anion"], has["anion"]], [0, 1, 2], 3)
    return states, ["chain", "chain+anion", "anion", "none"]


def compare(name, new, old_part, order):
    """Max relative difference after reordering new event axes to the legacy order."""
    idx = [new.names.index(n) for n in order] + [len(new.names)]
    s = new.self_terms[np.ix_(idx, idx)]
    d = new.distinct_terms[np.ix_(idx, idx)]
    scale = np.abs(old_part["self_values"]).max()
    err = max(np.abs(s - old_part["self_values"]).max(), np.abs(d - old_part["distinct_values"]).max()) / scale
    print(f"  {name:28s} max rel. diff {err:.1e}")
    return err


def main():
    system, folder, skip = sys.argv[1], Path(sys.argv[2]), int(sys.argv[3])
    sys.path.insert(0, str(folder))
    raw = glob.glob(str(folder / f"raw_*_skip-{skip}_*.pickle"))[0]
    tag = Path(raw).name[len("raw_"):].replace("_combined", "").replace(".pickle", "")
    t = time.time()
    old = pickle.load(open(raw, "rb"))
    print(f"loaded {Path(raw).name} in {time.time() - t:.0f}s")
    shells = shells_from_legacy(old, old.sampling_freq)
    cd, ad = shells.cation_displacements(), shells.anion_displacements()

    simplified = pickle.load(open(folder / f"onsager_simplified_{tag}.pickle", "rb"))
    markov_file = folder / f"onsager_markov_1_{tag}.pickle"
    markov = pickle.load(open(markov_file, "rb")) if markov_file.exists() else None
    distinct = "selfonly-False" in tag
    errs = []
    t = time.time()
    if system == "polymer":
        ev = polymer_events(shells)
        keys = POLYMER_KEYS
        states, names = polymer_chain_anion_states(shells)
    else:
        ev = liquid_events(shells)
        keys = LIQUID_KEYS
        # archived liquid results used legacy v0.4.2 states; reproduce the engine with those states
        states = np.stack([np.array(old.complete_states[li]["method_dicts"][1]) for li in sorted(old.complete_states)])
        names = ["SSIP", "CIP", "AGG"]
    res = decompose(cd, ev.labels, ev.names, ad, distinct=distinct)
    errs.append(compare("events (event mode)", res, simplified["event"], list(keys.values())))
    tot = decompose(cd, np.zeros_like(ev.labels), ["all"], ad, distinct=distinct)
    errs.append(compare("events (particle mode)", tot, simplified["particle"], ["all"]))
    if markov is not None:
        tr = transitions(states, names)
        res = decompose(cd, tr.labels, tr.names, ad, distinct=False)   # legacy markov runs were self-only
        errs.append(compare("macrostates (event mode)", res, markov["event"], tr.names))
    print(f"new events + correlations: {time.time() - t:.1f}s for {shells.n_cations} cations x {shells.n_frames} frames")
    print("  event probabilities:", dict(zip(ev.names, np.round(ev.probabilities(), 4))))

    if system == "liquid":
        # paper / v0.5.3 SSIP-CIP-AGG definition vs the legacy v0.5.3 implementation
        import contextlib
        import io
        sys.modules.pop("utils", None)                     # the archive folder ships its own utils.py
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "legacy"))
        import onsagerdecomposer_v053 as legacy
        # the legacy state code scales as n_frames^2, so check the definition on the first frames only
        n = min(500, shells.n_frames)
        for li, info in old.complete_coord_info.items():
            f = info["flattened_O_providers"]
            old.complete_coord_info[li] = dict(info, flattened_O_providers=f[f[:, 0] < n],
                                               global_times=info["global_times"][:n],
                                               cartesian_positions=info["cartesian_positions"][:n])
        old.anion_coordinating_Li = {k: v[v[:, 0] < n] if len(v) else v for k, v in old.anion_coordinating_Li.items()}
        with contextlib.redirect_stdout(io.StringIO()):
            ref = legacy.LiquidElectrolyteAnalyzer.simplify_complete_coord_info_to_states(old, methods=[1], parallel=False)
        ref_states = np.stack([np.array(ref[li]["method_dicts"][1]) for li in sorted(ref)])
        new_states = solvation_states(shells)[0][:, :n]
        states = states[:, :n]
        print(f"  SSIP/CIP/AGG states differing from legacy v0.5.3: {(new_states != ref_states).sum()} of {new_states.size}")
        print("  populations v0.4.2 (archived):", np.round(np.bincount(states.ravel(), minlength=3) / states.size, 4),
              " paper definition:", np.round(np.bincount(new_states.ravel(), minlength=3) / new_states.size, 4))
    assert max(errs) < 1e-8, errs


if __name__ == "__main__":
    main()
