import numpy as np

from onsagerdecomposer.onsager import correlation, decompose


def brute_correlation(x, y, tau):
    if tau == 0:
        return 0.0
    return np.mean(np.sum((x[tau:] - x[:-tau]) * (y[tau:] - y[:-tau]), axis=1))


def make_system(seed=0, n_cat=4, n_an=3, n=120, n_events=3):
    rng = np.random.default_rng(seed)
    d = rng.normal(size=(n_cat, n, 3))
    labels = rng.integers(0, n_events, size=(n_cat, n))
    an = rng.normal(size=(n_an, n, 3))
    return d, labels, an


def test_correlation_matches_brute_force():
    rng = np.random.default_rng(1)
    x, y = rng.normal(size=(2, 50, 3)).cumsum(axis=1)
    c = correlation(x, y)
    for tau in [0, 1, 7, 49]:
        assert np.isclose(c[tau], brute_correlation(x, y, tau))


def test_decomposition_matches_pairwise_definition():
    d, labels, an = make_system()
    names = ["a", "b", "c"]
    res = decompose(d, labels, names, an)
    r = np.cumsum(d[:, None] * (labels[:, None] == np.arange(3)[None, :, None])[..., None], axis=2)  # (cat, E, n, 3)
    ra = np.cumsum(an, axis=1)
    for tau in [1, 5, 60]:
        for a in range(3):
            for b in range(3):
                s = sum(brute_correlation(r[i, a], r[i, b], tau) for i in range(4))
                dd = sum(brute_correlation(r[i, a], r[j, b], tau) for i in range(4) for j in range(4) if i != j)
                assert np.isclose(res.self_terms[a, b, tau], s)
                assert np.isclose(res.distinct_terms[a, b, tau], dd)
            x = sum(brute_correlation(r[i, a], ra[k], tau) for i in range(4) for k in range(3))
            assert np.isclose(res.distinct_terms[a, -1, tau], x)
        assert np.isclose(res.self_terms[-1, -1, tau], sum(brute_correlation(ra[k], ra[k], tau) for k in range(3)))


def test_events_sum_to_undecomposed():
    d, labels, an = make_system(seed=2)
    split = decompose(d, labels, ["a", "b", "c"], an)
    total = decompose(d, np.zeros_like(labels), ["all"], an)
    assert np.allclose(split.cation_self_by_event.sum(axis=0), total.self_terms[0, 0])
    assert np.allclose(split.cation_distinct_by_event.sum(axis=0), total.distinct_terms[0, 0])
    assert np.allclose(split.distinct_terms[:-1, -1].sum(axis=0), total.distinct_terms[0, 1])


def test_chunking_does_not_change_result():
    d, labels, an = make_system(seed=3, n_cat=7)
    a = decompose(d, labels, ["a", "b", "c"], an, chunk=2)
    b = decompose(d, labels, ["a", "b", "c"], an, chunk=100)
    assert np.allclose(a.self_terms, b.self_terms)
    assert np.allclose(a.distinct_terms, b.distinct_terms)
