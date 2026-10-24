"""Recompute the PPM (443 K, r = 0.08) D_self / frequency / effectiveness used for the paper figures.

python benchmarks/polymer_paper_numbers.py <mechanism_dir> <df_6_12.pkl> [skip ...]

Shells come from the legacy raw pickles; everything after (events, correlations, fits)
is the new code with the paper settings: 10 ps frames, fit over 6-12 % of the lags.
"""
import glob
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare_archived import POLYMER_KEYS, shells_from_legacy  # noqa: E402

from onsagerdecomposer import decompose, polymer_events, self_diffusion_table  # noqa: E402


def main():
    folder, df_path = Path(sys.argv[1]), sys.argv[2]
    skips = [int(s) for s in sys.argv[3:]] or [200, 100, 50, 20, 10, 5]
    sys.path.insert(0, str(folder))
    ref = pd.read_pickle(df_path)
    ref = ref[(ref.system == "ppm") & (ref.analysis_type == "simplified") & (ref.concentration == 0.08)]
    worst = 0.0
    for skip in skips:
        old = pickle.load(open(glob.glob(str(folder / f"raw_*_skip-{skip}_*.pickle"))[0], "rb"))
        shells = shells_from_legacy(old, 10.0 * skip)
        events = polymer_events(shells)
        corr = decompose(shells.cation_displacements(), events.labels, events.names,
                         shells.anion_displacements(), distinct=False)
        new = self_diffusion_table(corr, events, shells.timestep, old.structure.volume, 443, fit_range=(0.06, 0.12))
        fit = pd.DataFrame(ref[ref.skip == skip].iloc[0].fit_values)
        old_D = fit.self_diffusivity.to_numpy()
        names = list(POLYMER_KEYS.values())                      # legacy event order
        new_D = np.r_[new.loc[names, "D_self"].to_numpy(), new.loc["anion", "D_self"], new.loc["total", "D_self"]]
        new_p = new.loc[names, "probability"].to_numpy()
        err = max(np.abs(new_D / old_D - 1).max(), np.abs(new_p - fit.frequency.to_numpy()[:6]).max())
        worst = max(worst, err)
        print(f"dt = {shells.timestep / 1000:5.2f} ns  total D_self {new.loc['total', 'D_self']:.3e} cm2/s"
              f"  anion {new.loc['anion', 'D_self']:.3e}  max rel. diff vs paper table {err:.1e}")
    assert worst < 1e-8, worst


if __name__ == "__main__":
    main()
