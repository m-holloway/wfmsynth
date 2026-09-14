"""The rise time asked for is the rise time delivered.

`physics._shape_edges` designed its Bessel with scipy's default ``norm='phase'``, whose realised
-3 dB corner sits at about 0.66x the frequency requested. Every shaped edge in this library came
out 1.55x slower than asked on the causal path and 2.10x on the zero-phase one, at every grid
density -- so a record was graded on an edge nobody chose, and asking for a finer grid did not
help because the error is in the filter design and not in the sampling.

The same defect had already been found and fixed in `instrument.scope_bandwidth`, whose docstring
records it and whose design passes ``norm='mag'``. The edge shaper never got the fix.

These tests re-solve the two corner constants from scratch and fail if either moves, so the
constants cannot drift away from the behaviour they are supposed to produce.
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy import optimize, signal

import wfmsynth as ws
from wfmsynth import physics as P

BAUD = 16e9
TR_S = 20e-12


def _tr_1090(y, dt):
    """10-90 % rise time by interpolated crossings, so the answer is not quantised to a sample."""
    y = np.asarray(y, float)
    lo, hi = y.min(), y.max()

    def cross(frac):
        lvl = lo + frac * (hi - lo)
        i = int(np.argmax(y > lvl))
        assert i > 0, "the record starts above the crossing level"
        return (i - 1) + (lvl - y[i - 1]) / (y[i] - y[i - 1])

    return (cross(0.9) - cross(0.1)) * dt


def _step(tr_samples, causal, scale):
    """One isolated step through the shaper, at a chosen corner scale."""
    n = 1 << 14
    x = np.concatenate([np.full(n // 2, -1.0), np.full(n // 2, 1.0)])
    wn = min(scale * 0.7 / tr_samples, 0.98)
    sos = signal.bessel(4, wn, output="sos", norm="mag")
    if not causal:
        return signal.sosfiltfilt(sos, x)
    zi = signal.sosfilt_zi(sos) * x[0]
    y, _ = signal.sosfilt(sos, x, zi=zi)
    return y


@pytest.mark.parametrize("causal, constant", [
    (True, P.EDGE_CORNER_CAUSAL),
    (False, P.EDGE_CORNER_ZEROPHASE),
])
def test_the_corner_constant_is_the_one_that_delivers_the_rise_time(causal, constant):
    """Re-solve it. The constant is not a magic number: it is the corner scale at which the
    delivered 10-90 % rise time equals the one requested."""
    solved = optimize.brentq(
        lambda s: _tr_1090(_step(200.0, causal, s), 1.0) / 200.0 - 1.0, 0.3, 4.0, xtol=1e-10)
    assert solved == pytest.approx(constant, rel=1e-4), (
        f"the corner that delivers the rise time is {solved:.6f}, the constant says {constant}")


def test_the_two_paths_need_different_corners():
    """The zero-phase path runs the filter twice, so it needs a wider corner for the same rise
    time. Matching the pair's -3 dB point instead gives 1.384 and is 4.5 % wrong, because two
    passes are a different filter shape with a different bandwidth-rise-time product."""
    assert P.EDGE_CORNER_ZEROPHASE > P.EDGE_CORNER_CAUSAL * 1.3
    assert P.EDGE_CORNER_ZEROPHASE == pytest.approx(1.390, abs=0.002)
    assert P.EDGE_CORNER_ZEROPHASE != pytest.approx(1.384, abs=0.001)


@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("k", [8, 10, 16, 32])
def test_a_carrier_delivers_the_rise_time_it_was_asked_for(causal, k):
    """Through the real op, not the filter in isolation. `k` is samples across the transition,
    which is what the sizing rule in the docs sets."""
    fs = k / TR_S
    n = int(round(16 * fs / BAUD))
    grid = ws.Grid(fs=fs, baud=BAUD, n=n, v_full=2.0)
    y = ws.Signal(seed=1, grid=grid).symbols(
        [-1.0] * 8 + [1.0] * 8, tr_frac=TR_S * BAUD, causal=causal).waveform()
    got = _tr_1090(y, 1.0 / fs)
    # 2 % at the coarsest grid the sizing rule allows, tightening as the grid resolves the edge.
    # It was 55 % and 110 % out before, and did not improve with k at all.
    assert got == pytest.approx(TR_S, rel=0.02), (
        f"k={k} causal={causal}: asked {TR_S*1e12:.1f} ps, delivered {got*1e12:.2f} ps "
        f"({got/TR_S:.4f}x)")


@pytest.mark.parametrize("causal", [True, False])
def test_the_error_falls_as_the_grid_resolves_the_edge(causal):
    """The residual has to be the GRID's, not the model's. A model error does not care how finely
    you sample; this one has to shrink when you do, or the constants are wrong."""
    err = {}
    for k in (8, 32, 128):
        fs = k / TR_S
        n = int(round(16 * fs / BAUD))
        grid = ws.Grid(fs=fs, baud=BAUD, n=n, v_full=2.0)
        y = ws.Signal(seed=1, grid=grid).symbols(
            [-1.0] * 8 + [1.0] * 8, tr_frac=TR_S * BAUD, causal=causal).waveform()
        err[k] = abs(_tr_1090(y, 1.0 / fs) / TR_S - 1.0)
    assert err[32] < err[8], f"error did not fall from k=8 ({err[8]:.4f}) to k=32 ({err[32]:.4f})"
    assert err[128] < 2e-3, f"asymptotic error {err[128]:.5f} is not a grid effect"


def test_the_bandwidth_rise_time_product_matches_the_documented_constant():
    """`BW x tr = 0.3497` for 10-90 %, which is the number the sizing rule is built on. Checked
    against the library's own front end rather than against first principles, because that is the
    filter a record actually meets."""
    from wfmsynth import instrument as INST
    fs, bw = 1024e9, 20e9
    n = 1 << 15
    grid = ws.Grid(fs=fs, baud=BAUD, n=n, v_full=2.0)
    x = np.concatenate([np.full(n // 2, -1.0), np.full(n // 2, 1.0)])
    y = INST.scope_bandwidth(x, grid, bw)
    tr = _tr_1090(y, 1.0 / fs)
    assert bw * tr == pytest.approx(0.3497, rel=0.02), f"BW*tr = {bw*tr:.4f}"
