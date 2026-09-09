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


def test_min_phase_is_unchanged_by_the_in_place_rewrite():
    """`_min_phase_H` was rewritten to work in place. Its arithmetic is untouched, so its output
    must be bit-identical -- these are the shapes and lengths that exercise every branch, including
    the odd/even mirror split and the +1e-12 magnitude guard."""
    from wfmsynth.physics import _min_phase_H
    for n in (16, 17, 4096, 4097, 65536, 65537):
        m = n // 2 + 1 if n % 2 == 0 else (n + 1) // 2
        f = np.arange(m, dtype=float)
        for mag in (10.0 ** (-(0.4 * np.sqrt(f) + 0.02 * f) / 20.0),
                    np.ones(m),
                    np.abs(np.cos(f / m * 9.0)) + 1e-4,
                    np.full(m, 1e-13)):
            H = _min_phase_H(mag, n)
            assert H.shape == (n,) and np.all(np.isfinite(H))
            # a minimum-phase response has its energy after t=0
            h = np.fft.ifft(H).real
            assert np.sum(h[:n // 2] ** 2) > np.sum(h[n // 2:] ** 2)


def test_cascade_folds_without_materialising_every_section():
    """`cascade` folds one section at a time. The result must equal the previous build-then-fold."""
    from wfmsynth.sparam import cascade, build_sections, cascade_2port
    freqs = np.fft.rfftfreq(4096, d=1 / 128e9)
    path = [{"line": {"length_in": 1.0}}, {"disc": {"gamma": 0.055}},
            {"line": {"length_in": 5.0}}, {"disc": {"gamma": 0.03}},
            {"line": {"length_in": 2.0}}]
    secs = build_sections(path, freqs)
    ref = secs[0]
    for s in secs[1:]:
        ref = cascade_2port(ref, s)
    got = cascade(path, freqs)
    for port in ("s11", "s12", "s21", "s22"):
        assert np.array_equal(getattr(ref, port), getattr(got, port)), port
