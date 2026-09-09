"""The batched forms must equal the scalar forms they replaced, exactly.

Both `measure_windows` and `_nearest_level` exist only to remove numpy's per-call overhead
from a deep record; neither is allowed to change a single value. These assert that, including
the cases real data never reaches -- exact ties, windows off the ends of the record, and
segments too short to have a slope.
"""
import numpy as np
import pytest

from wfmsynth.events import measure_window, measure_windows
from wfmsynth.measure import _cluster_levels, _nearest_level


def _signals(n=4096, seed=7):
    rng = np.random.default_rng(seed)
    return {
        "random": rng.standard_normal(n),
        "nrz": np.repeat(rng.choice([-1.0, 1.0], n // 16), 16) + 0.01 * rng.standard_normal(n),
        "flat": np.zeros(n),                      # ptp 0 -> the deadband is 1e-12, not 0
        "ramp": np.linspace(-1, 1, n),
        "ringing": np.sin(np.arange(n) / 3.0) * np.exp(-np.arange(n) / 900.0),
        "ripple": np.repeat(rng.choice([-1.0, 1.0], n // 16), 16) + 1e-9 * rng.standard_normal(n),
        "empty": np.zeros(0),
    }


@pytest.mark.parametrize("tag", sorted(_signals()))
def test_measure_windows_equals_measure_window(tag):
    x = _signals()[tag]
    n = len(x)
    rng = np.random.default_rng(3)
    wins = [(c - 16, c + 16, float(c)) for c in range(8, max(n - 8, 9), 16)]
    wins += [(-50, 10, None), (n - 5, n + 99, None), (0, 0, None), (5, 6, None), (7, 9, 8.0),
             (-10, -1, None), (n + 5, n + 9, None), (0, n, None), (3, 4, 3.5), (3, 20, 1e9)]
    wins += [(int(a), int(a + b), None)
             for a, b in zip(rng.integers(-20, max(n, 1) + 20, 200), rng.integers(0, 80, 200))]
    got = measure_windows(x, [w[0] for w in wins], [w[1] for w in wins], [w[2] for w in wins])
    for w, g in zip(wins, got):
        want = measure_window(x, w[0], w[1], center=w[2])
        assert want == g, f"{tag} {w}: {want} != {g}"
        assert {k: type(v) for k, v in want.items()} == {k: type(v) for k, v in g.items()}


def test_window_entirely_before_the_record_is_empty():
    """`stop <= 0` used to slice `x[0:-1]` -- the whole record -- through Python's negative
    indexing, so a window off the front reported the record's own peak-to-peak."""
    x = np.linspace(-1.0, 1.0, 512)
    m = measure_window(x, -50, -10)
    assert m["ptp"] == 0.0 and m["peak"] == 0.0 and m["trough"] == 0.0


@pytest.mark.parametrize("k", [2, 3, 4, 8])
def test_nearest_level_equals_argmin(k):
    rng = np.random.default_rng(11)
    c = np.sort(rng.standard_normal(k))
    mid = 0.5 * (c[:-1] + c[1:])
    for s in (rng.standard_normal(50_000),
              np.concatenate([mid, np.nextafter(mid, -np.inf), np.nextafter(mid, np.inf),
                              c, np.nextafter(c, np.inf)]),      # on and astride every boundary
              np.repeat(c, 100)):
        want = np.abs(s[:, None] - c[None, :]).argmin(1)
        assert np.array_equal(want, _nearest_level(s, c))


def test_nearest_level_breaks_ties_to_the_lower_index():
    """argmin's own rule. Real records never produce an exact tie, so nothing else covers it."""
    assert np.array_equal(_nearest_level(np.zeros(5), np.array([-1.0, 1.0])), np.zeros(5, int))


def test_cluster_levels_still_returns_sorted_centers():
    rng = np.random.default_rng(5)
    s = np.concatenate([rng.normal(-1, 0.05, 5000), rng.normal(1, 0.05, 5000)])
    c, lab = _cluster_levels(s, 2)
    assert np.all(np.diff(c) > 0) and set(np.unique(lab)) <= {0, 1}
