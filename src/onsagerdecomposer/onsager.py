"""Event-decomposed displacement correlations (the core of OnsagerDecomposer).

Every cation displacement over a window is assigned to exactly one event m.
The virtual-particle trajectory r_i^m is the cumulative sum of the windows of
ion i that were labelled m. Because correlations are bilinear, every quantity
reduces to correlations of per-event sums, so no loop over particle pairs is
needed:

    self[a, b]      = sum_i      C(r_i^a, r_i^b)
    distinct[a, b]  = sum_{i!=j} C(r_i^a, r_j^b) = C(R^a, R^b) - self[a, b]
    distinct[a, X]  = sum_{i,k}  C(r_i^a, r_k)   = C(R^a, R_X)

with R^a = sum_i r_i^a and X the anions (last row/column). C(x, y)(tau) is the
time-window-averaged displacement correlation
<(x(t+tau) - x(t)) . (y(t+tau) - y(t))>_t, evaluated with FFTs.
"""
from dataclasses import dataclass

import numpy as np


@dataclass
class Correlations:
    """Summed displacement correlations in A^2, indexed by lag (in windows).

    ``self_terms`` and ``distinct_terms`` have shape (n_events + 1, n_events + 1, n_lags);
    index -1 is the anion. Summing ``self_terms[m, :-1]`` over the second axis
    gives the event-m contribution sum_i <d_i^m . d_i>, and these contributions
    add up to the total cation MSD sum.
    """

    names: list
    self_terms: np.ndarray
    distinct_terms: np.ndarray
    n_cations: int
    n_anions: int

    @property
    def cation_self_by_event(self):
        """(n_events, n_lags): contribution of each event to sum_i MSD_i."""
        return self.self_terms[:-1, :-1].sum(axis=1)

    @property
    def cation_distinct_by_event(self):
        """(n_events, n_lags): contribution of each event to sum_{i!=j} <d_i . d_j>."""
        return self.distinct_terms[:-1, :-1].sum(axis=1)


def _positions(displacements):
    """Cumulative positions from per-window displacements (axis -2 is time)."""
    return np.cumsum(displacements, axis=-2)


def _window_correlation(spec_sum, prod_sum, n):
    """Combine FFT cross-spectrum and same-time products into C(tau), tau = 0..n-1.

    spec_sum: rfft-domain sum over particles/dims of X * conj(Y) + Y * conj(X).
    prod_sum: (n,) sum over particles/dims of x(k) * y(k).
    """
    s2 = np.fft.irfft(spec_sum, n=2 * n)[:n]
    cs = np.concatenate([[0.0], np.cumsum(prod_sum)])
    tau = np.arange(n)
    s1 = (cs[n] - cs[tau]) + cs[n - tau]
    return (s1 - s2) / (n - tau)


def _spectra(pos, nfft):
    """rfft over time of positions shaped (..., n, 3) -> (..., 3, nfft//2+1)."""
    return np.fft.rfft(np.moveaxis(pos, -2, -1), n=nfft, axis=-1)


def correlation(x, y):
    """C(x, y)(tau) for position series shaped (n, 3), summed over dimensions."""
    n = x.shape[0]
    fx, fy = _spectra(x, 2 * n), _spectra(y, 2 * n)
    spec = (fx * fy.conj() + fy * fx.conj()).real.sum(axis=0)
    return _window_correlation(spec, np.einsum("kd,kd->k", x, y), n)


def decompose(cation_displacements, labels, names, anion_displacements, distinct=True, chunk=32):
    """Compute event-decomposed self and distinct displacement correlations.

    cation_displacements: (n_cation, n_windows, 3) displacement of each cation in each window.
    labels: (n_cation, n_windows) integer event index into ``names`` for each window.
    names: event names; use a single name (and all-zero labels) for the undecomposed result.
    anion_displacements: (n_anion, n_windows, 3).
    distinct: if False, only self terms are computed (distinct terms are zero).
    chunk: cations processed at once (bounds memory).
    """
    disp = np.asarray(cation_displacements, dtype=float)
    labels = np.asarray(labels)
    n_cat, n = labels.shape
    nfft = 2 * n
    nf = nfft // 2 + 1
    E = len(names)

    self_spec = np.zeros((E, E, nf))          # sum over cations of 2 Re(F^a conj F^b)
    self_prod = np.zeros((E, E, n))
    coll_pos = np.zeros((E, n, 3))            # R^a(t)
    for start in range(0, n_cat, chunk):
        d = disp[start:start + chunk]
        lab = labels[start:start + chunk]
        onehot = lab[None, :, :] == np.arange(E)[:, None, None]       # (E, c, n)
        pos = _positions(d[None] * onehot[..., None])                  # (E, c, n, 3)
        coll_pos += pos.sum(axis=1)
        spec = _spectra(pos, nfft)                                     # (E, c, 3, nf)
        self_spec += 2 * np.einsum("acdf,bcdf->abf", spec, spec.conj()).real
        self_prod += np.einsum("acnd,bcnd->abn", pos, pos)

    an_pos = _positions(np.asarray(anion_displacements, dtype=float))  # (n_an, n, 3)
    an_spec = _spectra(an_pos, nfft)
    an_self = _window_correlation(2 * (an_spec * an_spec.conj()).real.sum(axis=(0, 1)),
                                  np.einsum("knd,knd->n", an_pos, an_pos), n)

    self_terms = np.zeros((E + 1, E + 1, n))
    for a in range(E):
        for b in range(E):
            self_terms[a, b] = _window_correlation(self_spec[a, b], self_prod[a, b], n)
    self_terms[-1, -1] = an_self

    distinct_terms = np.zeros_like(self_terms)
    if distinct:
        R_an = an_pos.sum(axis=0)
        for a in range(E):
            for b in range(E):
                distinct_terms[a, b] = correlation(coll_pos[a], coll_pos[b]) - self_terms[a, b]
            distinct_terms[a, -1] = distinct_terms[-1, a] = correlation(coll_pos[a], R_an)
        distinct_terms[-1, -1] = correlation(R_an, R_an) - an_self
    return Correlations(list(names), self_terms, distinct_terms, n_cat, an_pos.shape[0])
