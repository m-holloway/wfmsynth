"""Sub-sample displacement is interpolated with a bandlimited kernel, not linearly.

Jitter, duty-cycle distortion, intra-pair skew, a free-running sample clock and a fractional
reflection delay are all the same mechanism: move the signal by a fraction of a sample. How that
is interpolated is physics, not convenience. MEASURED cost of linear interpolation for a
half-sample shift of band-limited content:

    samples per symbol      4        10       32
    linear error        1.52 %    0.30 %   0.024 %      of peak-to-peak

A link graded on a 6 mV eye out of 800 mV is graded at 0.75 %, so the coarse case is twice the
quantity being measured.

Two places still interpolate linearly, deliberately, and the last test bounds what that costs.
"""
from __future__ import annotations

import numpy as np
import pytest

import wfmsynth as ws
from wfmsynth import physics as P
from wfmsynth import resample as RS


def _bandlimited(n=1 << 13, sps=10, seed=0):
    """Content confined below fs/(2*sps), i.e. what `sps` samples per symbol carries."""
    rng = np.random.default_rng(seed)
    f = np.fft.rfftfreq(n)
    X = (rng.normal(size=f.size) + 1j * rng.normal(size=f.size)) * (f < 0.5 / sps)
    x = np.fft.irfft(X, n)
    return x / np.ptp(x)


# ------------------------------------------------------------------ the kernel itself
def test_a_whole_sample_shift_is_exact():
    """An integer request needs no interpolation and must not pick up any."""
    x = _bandlimited()
    for k in (1, 7, -3):
        y = RS.shift(x, k)
        if k > 0:
            assert np.array_equal(y[k:], x[:-k])
        else:
            assert np.array_equal(y[:k], x[-k:])


def test_zero_shift_and_zero_displacement_are_identity():
    x = _bandlimited()
    assert np.array_equal(RS.shift(x, 0.0), x)
    assert np.array_equal(RS.displace(x, np.zeros(x.size)), x)


def test_a_fractional_shift_is_far_better_than_linear():
    x = _bandlimited(sps=10)
    idx = np.arange(x.size, dtype=float)
    sinc = RS.resample_at(x, idx - 0.5)
    lin = np.interp(idx - 0.5, idx, x)
    interior = slice(RS.HALF_WIDTH * 2, -RS.HALF_WIDTH * 2)
    # the exact answer for a half-sample shift of band-limited content, by construction:
    # shifting twice by 0.5 must equal shifting once by 1, which is exact
    twice = RS.resample_at(sinc, idx - 0.5)
    assert np.max(np.abs(twice[interior] - RS.shift(x, 1.0)[interior])) < 2e-3
    assert np.max(np.abs(lin[interior] - sinc[interior])) > 1e-3, "linear was not measurably worse"


def test_the_zero_fill_is_what_an_echo_needs():
    """`fill=0.0` says nothing is there yet. Holding the first sample instead would invent a
    precursor on a line that had not carried one."""
    x = np.ones(512)
    held = RS.shift(x, 10.5, fill="hold")
    zeroed = RS.shift(x, 10.5, fill=0.0)
    assert held[0] == pytest.approx(1.0, abs=1e-6)
    # the head is a bandlimited step ON, so it rings about zero rather than sitting at it
    assert abs(zeroed[0]) < 0.02
    assert zeroed[:8].mean() < 0.01 < held[:8].mean()
    # EXACTLY identical once past the kernel: the fill is one-sided, so it cannot roll off the
    # tail. Padding both ends made the whole record differ by 6.4e-3.
    assert np.array_equal(held[64:], zeroed[64:])


# ------------------------------------------------------------------ through the ops
@pytest.mark.parametrize("td_ps", [100.0, 100.25, 100.5, 100.75, 101.4])
def test_a_reflection_delay_is_not_quantised_to_a_sample(td_ps):
    """This op used to round the delay, so a fine sweep did not move the echo: at 500 GSa/s,
    100.0, 100.5 and 101.0 ps all landed at 100.0 ps."""
    fs, n = 500e9, 4096
    grid = ws.Grid(fs=fs, baud=16e9, n=n, v_full=2.0)
    imp = np.zeros(n)
    imp[n // 4] = 1.0
    y = P.multi_reflection(imp, grid=grid, td_ps=td_ps, gamma_s=0.3, gamma_l=0.5,
                           n_bounce=1, node="source")
    echo = np.asarray(y, float) - imp
    w = echo ** 2
    centroid = float((np.arange(n) * w).sum() / w.sum()) - n // 4
    want = 2.0 * td_ps * 1e-12 * fs                 # the round trip, in samples
    assert centroid == pytest.approx(want, abs=0.05), (
        f"asked {want:.4f} samples, echo centroid at {centroid:.4f}")


def test_a_sub_sample_change_in_the_delay_moves_the_echo():
    """The staircase, asserted directly: a quarter-sample change used to produce an identical
    record."""
    fs, n = 500e9, 4096
    grid = ws.Grid(fs=fs, baud=16e9, n=n, v_full=2.0)
    imp = np.zeros(n)
    imp[n // 4] = 1.0
    kw = dict(grid=grid, gamma_s=0.3, gamma_l=0.5, n_bounce=1, node="source")
    a = P.multi_reflection(imp, td_ps=100.0, **kw)
    b = P.multi_reflection(imp, td_ps=100.25, **kw)
    assert not np.allclose(a, b, atol=1e-9), "a quarter-sample change did nothing"


def test_the_acquisition_resample_is_bandlimited():
    """`acquire` lands on the STORED grid, which can be as coarse as 4 samples/UI -- the one
    resample where the oversample is low and linear interpolation is expensive."""
    baud, tr = 16e9, 20e-12
    fs_synth, fs_store, n_ui = 10 / tr, 64e9, 1024
    n = int(round(n_ui * fs_synth / baud))
    grid = ws.Grid(fs=fs_synth, baud=baud, n=n, v_full=0.8)
    prof = ws.AcquisitionProfile(sample_rate_hz=fs_store,
                                 record_length=int(round(n * fs_store / fs_synth)))
    def chain():
        # a fresh chain each time: the builder appends to itself and returns self, so reusing one
        # Signal would put the acquire op on the record it is being compared against
        return ws.Signal(seed=3, grid=grid).carrier(
            "nrz", pattern="prbs13", n_ui=n_ui, tr_frac=0.35, causal=True, seed=5)

    got = np.asarray(chain().acquire(prof).waveform(), float)
    conditioned = np.asarray(chain().waveform(), float)
    t = np.arange(prof.record_length) / fs_store * fs_synth
    assert np.allclose(got, RS.resample_at(conditioned, t), atol=1e-12)
    # and it is measurably not the linear answer
    assert np.max(np.abs(got - np.interp(t, np.arange(n), conditioned))) > 1e-4


def test_the_linear_sites_that_remain_are_bounded_by_the_sizing_rule():
    """Two places still interpolate linearly: the FFE's fractional tap spacing, and the
    interleaved-converter channel skew. Both act on the SYNTHESIS grid, which the sizing rule
    (`fs = k/tr`, k >= 8) keeps at 20+ samples per symbol -- and that is what makes it tolerable.
    This bounds it, so a future change that coarsens the grid does not quietly make it matter.
    """
    idx = np.arange(1 << 13, dtype=float)
    for sps, bound in ((8, 5e-3), (20, 1e-3), (32, 5e-4)):
        x = _bandlimited(sps=sps)
        lin = np.interp(idx - 0.5, idx, x)
        sinc = RS.resample_at(x, idx - 0.5)
        k = slice(RS.HALF_WIDTH * 2, -RS.HALF_WIDTH * 2)
        err = float(np.max(np.abs(lin[k] - sinc[k])))
        assert err < bound, f"{sps} samples/UI: linear costs {err:.2e}, bound {bound:.0e}"


# ------------------------------------------------------- the displacement ops use the kernel
def _skewable(n=1 << 13, sps=10, seed=2):
    return _bandlimited(n=n, sps=sps, seed=seed)


def test_intra_pair_skew_uses_the_kernel_and_not_linear_interpolation():
    """A stated skew in picoseconds is a sub-sample number, so the interpolator's error is an
    error in the impairment. Pinned both ways: it matches the bandlimited answer, and it is
    measurably not the linear one."""
    x = _skewable()
    grid = ws.Grid(fs=256e9, baud=16e9, n=x.size, v_full=2.0)
    skew_ps = 1.45                                     # 0.371 samples at 256 GSa/s
    d = skew_ps * 1e-12 * grid.fs
    _, nn = P.differential_pair(x, grid=grid, skew_ps=skew_ps)
    # n is -0.5 * the delayed arm
    got = np.asarray(nn, float) / -0.5
    idx = np.arange(x.size, dtype=float)
    k = slice(256, x.size - 256)
    assert np.allclose(got[k], RS.shift(x, d)[k], atol=1e-9)
    lin = np.interp(idx - d, idx, x, left=x[0], right=x[-1])
    assert np.max(np.abs(got[k] - lin[k])) > 1e-4, "indistinguishable from linear interpolation"


def test_supply_induced_timing_jitter_uses_the_kernel():
    x = _skewable(seed=4)
    grid = ws.Grid(fs=256e9, baud=16e9, n=x.size, v_full=2.0)
    y = P.supply_coupling(x, grid, f_ripple_hz=1e6, am_depth=0.0, psij_ps=2.0)
    s = np.sin(2 * np.pi * 1e6 * (np.arange(x.size) / grid.fs))
    dev = (2.0 * 1e-12 * grid.fs) * s
    idx = np.arange(x.size, dtype=float)
    k = slice(256, x.size - 256)
    assert np.allclose(np.asarray(y, float)[k], RS.displace(x, dev)[k], atol=1e-9)
    lin = np.interp(idx - dev, idx, x, left=x[0], right=x[-1])
    assert np.max(np.abs(np.asarray(y, float)[k] - lin[k])) > 1e-6


def test_a_displacement_applied_and_removed_comes_back():
    """The sharpest statement of what the kernel buys. Linear interpolation low-passes the signal
    on the way out and again on the way back; MEASURED 5.0e-3 against 8.1e-8, a factor of 61,000.
    """
    x = _skewable(seed=6)
    there = RS.shift(x, 0.37)
    back = RS.shift(there, -0.37)
    k = slice(256, x.size - 256)
    assert np.max(np.abs(back[k] - x[k])) < 1e-6

    idx = np.arange(x.size, dtype=float)
    lin_there = np.interp(idx - 0.37, idx, x, left=x[0], right=x[-1])
    lin_back = np.interp(idx + 0.37, idx, lin_there, left=lin_there[0], right=lin_there[-1])
    assert np.max(np.abs(lin_back[k] - x[k])) > 1e-4
