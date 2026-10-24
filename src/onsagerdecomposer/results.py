"""Transport coefficients from decomposed correlations (same conventions as the paper).

D^m is the slope of the event-m self correlation over a lag window (by default
5-10 % of the available lags), divided by 6 and by the number of cations.
sigma_NE^m = F^2 z^2 c D^m / RT, and the effectiveness of event m is D^m / p_m.
"""
import numpy as np
import pandas as pd
import scipy.constants as const
from scipy import stats

A2_PER_PS_TO_CM2_PER_S = 1e-4


def fit_slopes(curves, window_time, fit_range=(0.05, 0.1)):
    """Slope (A^2/ps) of each curve (rows of ``curves``, indexed by lag in windows) over a fraction of lags."""
    n = curves.shape[-1]
    lo, hi = int(np.floor(n * fit_range[0])), int(np.floor(n * fit_range[1]))
    lags = np.arange(n)[lo:hi] * window_time
    return np.array([stats.linregress(lags, c[lo:hi]).slope for c in np.atleast_2d(curves)])


def sigma_per_D(volume, n_ions, temperature, charge=1):
    """Factor converting D (cm^2/s) of ``n_ions`` ions in ``volume`` (A^3) to sigma_NE (mS/cm)."""
    concentration = n_ions / (volume * 1e-24 * const.N_A)        # mol/cm^3
    return 1000 * concentration * charge**2 * const.physical_constants["Faraday constant"][0] ** 2 / (
        const.R * temperature)


def self_diffusion_table(corr, events, window_time, volume, temperature, fit_range=(0.05, 0.1)):
    """Per-event probability, D_self (cm^2/s), sigma_NE (mS/cm), share of the total, and effectiveness.

    corr: ``Correlations`` from ``decompose``; events: the ``Events`` used to compute it.
    window_time: ps per window; volume: A^3; temperature: K.
    """
    slopes = fit_slopes(corr.cation_self_by_event, window_time, fit_range)
    D = slopes / 6 / corr.n_cations * A2_PER_PS_TO_CM2_PER_S
    factor = sigma_per_D(volume, corr.n_cations, temperature)
    p = events.probabilities()
    with np.errstate(divide="ignore", invalid="ignore"):
        table = pd.DataFrame({"probability": p, "D_self": D, "sigma_NE": factor * D, "share": D / D.sum(),
                              "effective_D_self": D / p, "effective_sigma_NE": factor * D / p},
                             index=pd.Index(events.names, name="event"))
    table.loc["total"] = [1.0, D.sum(), factor * D.sum(), 1.0, D.sum(), factor * D.sum()]
    if corr.n_anions:
        anion_D = fit_slopes(corr.self_terms[-1, -1], window_time, fit_range)[0] / 6 / corr.n_anions
        table.loc["anion"] = [np.nan, anion_D * A2_PER_PS_TO_CM2_PER_S, np.nan, np.nan, np.nan, np.nan]
    return table


def per_macrostate(table, state_names):
    """Probability, D_self and effective D_self of each macrostate from a transition table.

    Each X->Y transition (X != Y) is assigned half to X and half to Y (paper Fig. 6).
    ``table`` is ``self_diffusion_table`` for ``macrostates.transitions`` events.
    """
    rows = []
    for x in state_names:
        events = table.drop(index=["total", "anion"], errors="ignore")
        w = np.array([(a == x) + (b == x) for a, b in (n.split("->") for n in events.index)]) / 2
        p = (w * events["probability"]).sum()
        D = (w * events["D_self"]).sum()
        rows.append({"state": x, "probability": p, "D_self": D, "effective_D_self": D / p})
    return pd.DataFrame(rows).set_index("state")
