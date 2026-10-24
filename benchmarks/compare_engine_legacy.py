"""Check the new Onsager engine against the legacy pairwise engine (v0.5.3) on synthetic data.

Run from the repository root:  python benchmarks/compare_engine_legacy.py
"""
import contextlib
import io
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "legacy"))
import onsagerdecomposer_v053 as legacy  # noqa: E402
from onsagerdecomposer.onsager import decompose  # noqa: E402


def legacy_inputs(d, labels, an, names):
    li = {}
    for i in range(d.shape[0]):
        li[i] = {e: {"global_time": np.where(labels[i] == k)[0], "disp": d[i][labels[i] == k]}
                 for k, e in enumerate(names)}
    anion = {1000 + k: {"global_times": np.arange(d.shape[1]), "disp": an[k]} for k in range(an.shape[0])}
    return li, anion


def run_legacy(li, anion, mode, self_only):
    with contextlib.redirect_stdout(io.StringIO()):
        oa = legacy.OnsagerAnalyzer(1, li, anion, mode=mode)
        return oa.compute_fft_onsager(self_only=self_only)


def main():
    rng = np.random.default_rng(0)
    n_cat, n_an, n, names = 12, 12, 2000, ["vehicular", "solvent", "anion", "both"]
    d = rng.normal(size=(n_cat, n, 3))
    labels = rng.integers(0, len(names), size=(n_cat, n))
    an = rng.normal(size=(n_an, n, 3))
    li, anion = legacy_inputs(d, labels, an, names)

    worst = 0.0
    for mode, self_only in [("EVENT", True), ("EVENT", False), ("PARTICLE", True), ("PARTICLE", False)]:
        t = time.time()
        old_self, old_dist = run_legacy(li, anion, mode, self_only)
        t_old = time.time() - t
        t = time.time()
        if mode == "EVENT":
            new = decompose(d, labels, names, an, distinct=not self_only)
        else:
            new = decompose(d, np.zeros_like(labels), ["Li"], an, distinct=not self_only)
        t_new = time.time() - t
        scale = np.abs(old_self).max()
        err = max(np.abs(new.self_terms - old_self).max(), np.abs(new.distinct_terms - old_dist).max()) / scale
        worst = max(worst, err)
        print(f"{mode:8s} self_only={self_only!s:5s}  max rel. diff {err:.1e}   legacy {t_old:6.2f}s   new {t_new:6.3f}s")
    assert worst < 1e-10, worst
    print("OK: new engine reproduces the legacy engine")


if __name__ == "__main__":
    main()
