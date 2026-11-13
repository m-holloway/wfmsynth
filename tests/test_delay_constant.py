"""`delay(by=<constant>)` is one fixed FIR (GitHub #104).

The per-sample gather it replaces is kept here as `old_delay`, the reference. The two do not agree
bit for bit and cannot: the old path rebuilt the kernel at every sample from `pos - floor(pos)`,
which carries the rounding of `arange(n) - d*fs` (it grows with the sample index, to ~n*eps), and
its `einsum` sums in its own order. The new kernel is computed once from the exact fraction, so
the difference is the OLD path's noise: BOUND_EPS units of eps times the record's peak.
"""
from __future__ import annotations

import time

import numpy as np
import pytest

import wfmsynth as ws
from wfmsynth import Grid, Signal, joins as JN

EPS = np.finfo(float).eps
FS = 10e9
BOUND_EPS = 4.0e5          # measured worst 1.8e5 eps (n=70000, 1/3 sample); 2x margin


def old_delay(x, s):
    """The 0.48.0 path for a constant delay of `s` samples: per-sample positions and kernels."""
    n = len(x)
    pos = np.arange(n) - s
    base = np.floor(pos)
    mu = pos - base
    base = base.astype(np.int64)
    k = np.arange(-JN.DELAY_TAPS // 2 + 1, JN.DELAY_TAPS // 2 + 1)
    y = np.empty(n, x.dtype)
    for a in range(0, n, JN._CHUNK):
        b = min(a + JN._CHUNK, n)
        taps = x[np.clip(base[a:b, None] + k[None, :], 0, n - 1)]
        y[a:b] = np.einsum("ij,ij->i", taps, JN._kernel(mu[a:b]))
    return y


def record(n, seed=1):
    x = np.random.default_rng(seed).normal(size=n) + 3.0
    x[:10] = 2.0                  # both ends idle at a non-zero level, different ones
    x[-10:] = -4.0
    return x


@pytest.mark.parametrize("n", [100, 5000, 70000])
@pytest.mark.parametrize("s", [0.25, 5.25, 0.3, 17.7, 1 / 3, 0.999, 1e-9,
                               "near_end", "past_end", "far_past"])
def test_constant_delay_matches_the_per_sample_path(n, s):
    s = {"near_end": n - 3.3, "past_end": n + 50.4, "far_past": 3 * n + 0.1}.get(s, s)
    x = record(n)
    got = JN.delay(x, s / FS, FS)
    want = old_delay(x, s)
    assert np.abs(got - want).max() <= BOUND_EPS * EPS * np.abs(x).max()


def test_dyadic_fraction_differs_only_by_summation_order():
    x = record(5000)
    for s in (0.25, 5.5, 40.75):
        assert np.abs(JN.delay(x, s / FS, FS) - old_delay(x, s)).max() <= 64 * EPS * np.abs(x).max()


def test_a_whole_sample_delay_is_still_an_exact_shift():
    x = record(500)
    y = JN.delay(x, 7 / FS, FS)
    assert np.array_equal(y[7:], x[:-7]) and np.all(y[:7] == x[0])


def test_ends_are_held_not_zero_padded():
    x = np.full(300, 3.3)
    x[-100:] = 1.1
    y = JN.delay(x, 4.4 / FS, FS)
    assert np.abs(y[:5] - 3.3).max() < 1e-9               # no false edge at the head
    assert np.abs(y[-3:] - 1.1).max() < 1e-9              # and the tail's level survives
    assert np.abs(JN.delay(np.full(50, 2.5), 600.3 / FS, FS) - 2.5).max() < 1e-12


def test_a_negative_delay_is_still_refused_by_name():
    with pytest.raises(ValueError, match="negative delay"):
        JN.delay(record(100), -2.3 / FS, FS)


def test_complex_envelope_matches():
    rng = np.random.default_rng(3)
    x = rng.normal(size=2000) + 1j * rng.normal(size=2000)
    got = JN.delay(x, 3.3 / FS, FS, fc_hz=1e9)
    want = old_delay(x, 3.3) * np.exp(-2j * np.pi * np.mod(1e9 * 3.3 / FS, 1.0))
    assert np.abs(got - want).max() <= BOUND_EPS * EPS * 4


def test_signal_valued_by_still_takes_the_per_sample_path():
    g = Grid(fs=1e6, n=4096)
    wob = Signal(grid=g).carrier("sine", f_hz=300.0, amp=2e-6, offset=4.3e-6)
    t = np.arange(g.n) / g.fs
    x = Signal(grid=g).carrier("sine", f_hz=20e3).waveform()
    y = Signal(grid=g).carrier("sine", f_hz=20e3).delay(by=wob).waveform()
    d = 4.3e-6 + 2e-6 * np.sin(2 * np.pi * 300.0 * t)
    assert np.abs(y - np.sin(2 * np.pi * 20e3 * (t - d)))[24:-16].max() < 1e-4
    assert not np.array_equal(y[:50], x[:50])


def test_constant_fractional_delay_is_not_dearer_than_the_chain_it_delays():
    g = Grid(fs=FS, n=200_000)
    s = Signal(seed=1, grid=g).carrier("nrz", n_ui=9000, pattern="prbs9", seed=5,
                                       tr_frac=0.17).timing(rj_ps=2.0).gain(by=0.33)

    def best(sig, reps=3):
        out = []
        for _ in range(reps):
            c = sig.copy()
            t = time.perf_counter()
            c.waveform()
            out.append(time.perf_counter() - t)
        return min(out)

    base = best(s)
    frac = best(s.copy().delay(by=525e-12))
    assert frac < 10 * base, f"{frac:.3f} s against {base:.3f} s undelayed"
