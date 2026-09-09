"""wfmsynth.eye — the recorded eye, and the two things about it that must not move.

These tests do not need the demo checkout: they build their own records and grade
the answer against arithmetic. The A/B against the shipping reduction lives in
spikes/eye3m/verify.py, which needs one.
"""
import numpy as np
import pytest

import wfmsynth as ws
from wfmsynth import cdr as C
from wfmsynth import eye as E
from wfmsynth import physics as P
from wfmsynth.grid import Grid


def _record(n_ui=4000, spb=16, pattern="prbs15", noise=0.0, seed=1):
    n = int(n_ui * spb)
    rng = np.random.default_rng(seed)
    x = np.asarray(P.nrz(n_ui=n_ui, n=n, pattern=pattern, tr_frac=0.35), float)
    if noise:
        x = x + noise * rng.standard_normal(n)
    return x, Grid(fs=1e9 * spb, baud=1e9, n=n)


def test_recovers_the_period_the_record_actually_has():
    """The period comes from the EDGES, so a grid that states the wrong rate must not
    move it — and must not move the picture either."""
    x, g = _record()
    img, meta = ws.eye_density(x, g)
    assert meta["clock"]["source"] == "recovered"
    assert abs(meta["clock"]["samplesPerUi"] - 16.0) < 1e-6
    for mult in (1.7, 0.5):
        wrong = Grid(fs=g.fs, baud=g.baud * mult, n=g.n)
        img2, meta2 = ws.eye_density(x, wrong)
        assert np.array_equal(img, img2), "the grid's rate moved a recovered picture"


def test_threads_do_not_change_a_single_count():
    """Every parallel path here is a partition of independent work recombined in a
    fixed order. One thread and eight must agree bit for bit, everywhere."""
    x, g = _record(n_ui=6000, noise=0.01)
    a, ma = ws.eye_density(x, g, max_traces=6000, threads=1)
    b, mb = ws.eye_density(x, g, max_traces=6000, threads=8)
    assert np.array_equal(a, b)
    assert ma["clock"] == mb["clock"]
    c1, d1 = E.crossings(x, float(np.mean(x)), float(np.ptp(x)) * 0.1, threads=1)
    c8, d8 = E.crossings(x, float(np.mean(x)), float(np.ptp(x)) * 0.1, threads=8)
    assert np.array_equal(c1, c8) and np.array_equal(d1, d8)


def test_crossings_match_the_exhaustive_definition():
    """The edge finder skips 97 % of the samples the definition looks at. It has to
    land on the same edges as the definition it is short-cutting."""
    x, g = _record(n_ui=400, spb=8, noise=0.02)
    level = float(np.mean(x))
    hyst = float(np.ptp(x)) * 0.1
    got, dirs = E.crossings(x, level, hyst)
    # the definition, written out: every armed sample, alternation-filtered, then a
    # running walk back to the straddling pair
    n = x.size
    up = x > level + hyst
    dn = x < level - hyst
    ev = np.flatnonzero(up | dn)
    ev_up = up[ev]
    keep = np.empty(ev.size, bool)
    keep[0] = (1 if ev_up[0] else -1) != (1 if x[0] >= level else -1)
    keep[1:] = ev_up[1:] != ev_up[:-1]
    ev, ev_up = ev[keep], ev_up[keep]
    idx = np.arange(n)
    last_le = np.maximum.accumulate(np.where(x <= level, idx, -1))
    last_ge = np.maximum.accumulate(np.where(x >= level, idx, -1))
    j0 = np.where(ev_up, last_le[ev], last_ge[ev])
    good = (j0 >= 0) & (j0 + 1 < n)
    j0, ev_up = j0[good], ev_up[good]
    a = x[j0] - level
    b = x[j0 + 1] - level
    den = a - b
    frac = np.where(den == 0.0, 0.0, a / np.where(den == 0.0, 1.0, den))
    assert np.array_equal(got, j0 + frac)
    assert np.array_equal(dirs, np.where(ev_up, 1, -1).astype(np.int8))


def test_the_cdr_is_the_library_cdr():
    """The eye's loop must be `wfmsynth.cdr.recover_clock` and nothing that merely
    resembles it: same transfer, to the last bit, threaded or not."""
    rng = np.random.default_rng(0)
    ph = np.cumsum(rng.standard_normal(50000)) * 1e-3
    baud, bw = 1e9, 1e9 / E.LOOP_DIVISOR
    ref = C.recover_clock(ph, baud, bw, order=2, damping=0.707)
    for nt in (1, 2, 8):
        got = E._recover_clock(ph, baud, bw, 2, 0.707, nt)
        assert np.array_equal(got[0], ref[0]) and np.array_equal(got[1], ref[1])


def test_jitter_below_the_corner_is_tracked_and_above_it_is_not():
    """The measurement the eye exists to make. Sinusoidal timing jitter well inside
    the loop bandwidth must come out of the residual; well outside it must survive."""
    K, spb = 20000, 8
    baud = 1e9
    bw = baud / E.LOOP_DIVISOR
    k = np.arange(K, dtype=float)
    out = {}
    for cycles in (0.2, 40.0):
        j = 0.20 * np.sin(2 * np.pi * cycles * k / K)
        res = C.jitter_transfer(j, baud, bw, order=2, damping=0.707)
        out[cycles] = float(np.std(res[K // 4:]))
    assert out[0.2] < 0.01, "the loop failed to track jitter inside its bandwidth"
    assert out[40.0] > 0.9 * (0.20 / np.sqrt(2)), "the loop swallowed jitter it must pass"


def test_the_picture_lands_where_the_arithmetic_says():
    """A clean two-level record: the centre column must hold two rows, one per level,
    and the mid-level row must be busiest a quarter of the span either side."""
    x, g = _record(n_ui=3000, spb=16, pattern="prbs15")
    img, meta = ws.eye_density(x, g, w=128, h=96, ui_span=2.0, max_traces=2000)
    vmin, vmax = meta["vMin"], meta["vMax"]
    col = img[:, 64].astype(float)
    occupied = np.flatnonzero(col)
    opening = int(np.diff(occupied).max()) - 1
    assert opening > 0.7 * 96, "the decision column should be open, not a smear"
    assert occupied.min() < 8 and occupied.max() > 88, "the levels are not at the rails"
    mid = img[int(round((1.0 - (0.0 - vmin) / (vmax - vmin)) * 95)), :].astype(float)
    peaks = sorted(np.argsort(mid)[-2:].tolist())
    assert abs(peaks[0] - 0.25 * 127) < 4 and abs(peaks[1] - 0.75 * 127) < 4


def test_a_record_with_no_clock_in_it_says_so():
    """No edges, no recovery — and the answer is a stated reason, not a confident
    picture folded on a rate nobody measured."""
    x = np.zeros(4096)
    img, meta = ws.eye_density(x, Grid(fs=16e9, baud=1e9, n=4096))
    assert meta["clock"]["source"] == "nominal"
    assert meta["clock"]["reason"] in ("edges", "nolock", "short")
    assert img.sum() == 0


@pytest.mark.parametrize("spb", [4, 9.7, 16])
def test_non_integer_and_small_samples_per_ui(spb):
    n_ui = 3000
    n = int(n_ui * spb)
    x = np.asarray(P.nrz(n_ui=n_ui, n=n, pattern="prbs15", tr_frac=0.4), float)
    g = Grid(fs=1e9 * spb, baud=1e9, n=n)
    img, meta = ws.eye_density(x, g)
    assert meta["clock"]["source"] == "recovered"
    assert abs(meta["clock"]["samplesPerUi"] - spb) < 1e-4 * spb
    assert img.sum() == meta["traces"] * 128
