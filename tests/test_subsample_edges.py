"""GitHub #98: a carrier's symbol edges are at their own times, not on whole samples.

`physics._place_symbols` used to put every edge on a sample, so a samples-per-UI that is not whole
walked as a sawtooth of up to one sample (14.4 ps rms at 20 GSa/s), and `jitter=dict(rj=...)` under
a sample had a floor of about half a sample (25 ps rms). Each edge is now the hard step at its own
`floor(time)` delayed by the fraction, which is what `square` and `cmos` have done since 0.46.0.
"""
import numpy as np
import pytest

import wfmsynth as ws
from wfmsynth import physics as P

FS, BAUD = 20e9, 500e6
PS_PER_SAMPLE = 1e12 / FS


def _tie_rms_ps(x):
    """rms timing error of the rising mid-level crossings against a constant-period fit (the
    issue's measure; linear interpolation of the crossing is its own, small, floor)."""
    i = np.flatnonzero((x[:-1] < 0) & (x[1:] >= 0))
    t = (i + x[i] / (x[i] - x[i + 1])) / FS
    k = np.r_[0, np.cumsum(np.round(np.diff(t) / np.median(np.diff(t))))]
    A = np.c_[np.ones_like(k), k]
    c, *_ = np.linalg.lstsq(A, t, rcond=None)
    return float((t - A @ c).std() * 1e12)


def _clock(n, n_ui, **kw):
    g = ws.Grid(fs=FS, baud=BAUD, n=n)
    return (ws.Signal(seed=1, grid=g).carrier("nrz", n_ui=n_ui, pattern="clock", **kw)
            .scope(bw_hz=6e9))


def test_a_non_whole_samples_per_ui_adds_no_timing_error_of_the_grids():
    # 40.00061 samples per UI; this measured 14.43 ps rms when edges were on samples
    x = _clock(1 << 21, 52428).waveform()
    assert _tie_rms_ps(x) < 0.1
    # and whole is as it was: nothing to measure
    assert _tie_rms_ps(_clock(52428 * 40, 52428).waveform()) < 1e-3


@pytest.mark.parametrize("r", [0.01, 0.1, 0.4, 1.0, 2.0])
def test_carrier_jitter_under_a_sample_measures_what_was_asked(r):
    """The band-limiting of the 6 GHz scope after it is what the tolerance is for: a 2 percent
    band holds from 0.5 ps to 100 ps rms here (0.5 % at 20 ps)."""
    x = _clock(20000 * 40, 20000, jitter=dict(rj=r)).waveform()
    assert _tie_rms_ps(x) == pytest.approx(r * PS_PER_SAMPLE, rel=0.02)


def test_carrier_jitter_matches_timing_at_the_same_per_edge_rms():
    """`timing(rj_ps=)` draws a fresh offset per SAMPLE, so what an edge shows depends on the
    filtering after it (0.5 ps asked, about 0.32 ps per edge through this scope). Asked for that
    per-edge figure, the carrier's own jitter lands on the same number."""
    g = ws.Grid(fs=FS, baud=BAUD, n=20000 * 40)
    via_timing = _tie_rms_ps(ws.Signal(seed=1, grid=g).carrier("nrz", n_ui=20000, pattern="clock")
                             .timing(rj_ps=0.5).scope(bw_hz=6e9).waveform())
    assert 0.25 < via_timing < 0.40
    via_carrier = _tie_rms_ps(
        _clock(20000 * 40, 20000, jitter=dict(rj=via_timing / PS_PER_SAMPLE)).waveform())
    assert via_carrier == pytest.approx(via_timing, rel=0.03)


def test_sub_sample_dcd_and_pj_are_realised_and_no_longer_quantised():
    a = _clock(20000 * 40, 20000, jitter=dict(dcd=0.01)).waveform()
    b = _clock(20000 * 40, 20000, jitter=dict(dcd=0.04)).waveform()
    assert not np.array_equal(a, b)
    p = _clock(20000 * 40, 20000, jitter=dict(pj=0.05, f_pj=50.0)).waveform()
    # 0.05 samples of sine is 2.5 ps peak, 1.77 ps rms
    assert _tie_rms_ps(p) == pytest.approx(0.05 * PS_PER_SAMPLE / np.sqrt(2), rel=0.03)


@pytest.mark.parametrize("kind", ["nrz", "pam4", "pam5"])
def test_whole_samples_per_ui_with_no_jitter_is_the_plain_stream(kind):
    """Every edge is on a sample, so there is nothing to place: the old stream, bit for bit."""
    n, n_ui = 4096, 256
    lv = P.carrier_symbols(kind, n_ui)
    old = lv[np.clip((np.arange(n) / (n / n_ui)).astype(int), 0, n_ui - 1)]
    assert np.array_equal(P._place_symbols(lv, n, n / n_ui), old)


def test_jitter_that_lands_every_edge_on_a_sample_is_the_plain_stream():
    n, n_ui = 4096, 256
    lv = P.carrier_symbols("nrz", n_ui)
    old = lv[np.clip((np.arange(n) / (n / n_ui)).astype(int), 0, n_ui - 1)]
    x = P._place_symbols(lv, n, n / n_ui, P.Jitter(dcd=0.0), np.random.default_rng(0))
    assert np.array_equal(x, old)
    shifted = P._place_symbols(lv, n, n / n_ui, P.Jitter(dcd=2.0), np.random.default_rng(0))
    assert set(np.unique(shifted)) <= set(np.unique(lv))      # a whole-sample shift adds no levels


def test_an_edge_between_samples_keeps_its_step_and_its_moment():
    """The stream's first moment about an edge is the edge's time: sum of (final - x) over the
    samples is te to rounding, wherever the fraction falls."""
    for te in (50.0, 50.1, 50.5, 50.9):
        edge = np.zeros(100)
        m = int(np.floor(te))
        edge[m:] = 1.0
        out = P._delay_steps(edge, np.array([m]), np.array([te - m]), np.array([1.0]))
        assert out[:40].max() < 1e-12 and abs(out[-1] - 1.0) < 1e-12
        assert (1.0 - out).sum() == pytest.approx(te, abs=1e-9)


def _prbs7_peak(spb, k):
    n_ui = 500
    return float(P.nrz(n_ui=n_ui, n=int(round(spb * n_ui)), tr_frac=k / spb,
                       tr_floor_samples=2.0).max())


def test_the_ring_of_a_short_edge_between_samples_is_what_the_docs_say():
    """Measured, not wished away: at the 2-sample floor a non-whole spb peaks up to ~3 points
    above a whole one (the 8-tap Lagrange step is not flat near Nyquist); from a 4-sample edge
    it is about a point, the size of the scatter between sample phases. `nrz`'s docstring and the reference quote
    these numbers."""
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        floor = [_prbs7_peak(s, 2.0) for s in (7.37, 9.3, 12.3)]
        whole = _prbs7_peak(12.0, 2.0)
        assert all(whole < p < whole + 0.04 for p in floor)
        assert _prbs7_peak(4.4, 2.0) < 1.16
        for s in (8.4, 12.3):
            assert abs(_prbs7_peak(s, 4.0) - _prbs7_peak(round(s), 4.0)) < 0.015
