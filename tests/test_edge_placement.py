"""`square` and `cmos` place every edge at its own time, to a fraction of a sample.

Until 0.46.0 both built their ideal train as ``where(phase(t) < duty)`` on the samples, so an
edge between two samples went to the next one: each edge late by a different amount in [0, 1)
sample, which on the record is one sample peak to peak of deterministic jitter (15.6 ps at
64 GSa/s) that is the grid's and reads as the link's. Separately, an edge exactly ON a sample
was decided by the rounding of ``f * t % 1.0``, so a commensurate clock had the odd edge one
sample late too.

Every figure here is read with an instrument that does not share the generator's method: the
crossing time from a local polynomial through the samples about each crossing, the duty from
the record's mean. `_legacy` is the old construction, kept here as the known answer that
calibrates the instrument -- it must read one sample peak to peak, or the instrument is blind.
"""
import warnings

import numpy as np
import pytest

import wfmsynth.physics as P
from wfmsynth import Grid, Signal


def _legacy(n, fs, f, duty=0.5, phase_rad=0.0):
    """The ideal train as it was built before 0.46.0."""
    t = np.arange(n, dtype=float) / fs
    return np.where((f * t + phase_rad / (2 * np.pi)) % 1.0 < duty, 1.0, -1.0)


def _crossings(x, level):
    """Crossing times in samples (a degree-7 polynomial through the 8 samples about each),
    and the direction of each."""
    s = np.asarray(x, float) - level
    i = np.flatnonzero((s[:-1] < 0) != (s[1:] < 0))
    i = i[(i > 8) & (i < s.size - 9)]
    t = np.empty(i.size)
    for a, k in enumerate(i):
        w = np.arange(-3, 5)
        r = np.polynomial.Polynomial.fit(w, s[k + w], 7).convert().roots()
        r = r[np.abs(r.imag) < 1e-9].real
        t[a] = k + r[(r >= -1e-6) & (r <= 1 + 1e-6)][0]
    return t, np.sign(s[i + 1] - s[i])


def _tie_pp(x, level, spp):
    """Worst peak-to-peak edge-time error in samples, rising and falling edges separately,
    against a perfectly periodic train of `spp` samples. The record's ends are left out."""
    t, d = _crossings(x, level)
    worst = 0.0
    for sign in (1, -1):
        tt = t[d == sign]
        tt = tt[(tt > 0.1 * len(x)) & (tt < 0.9 * len(x))]
        e = tt - np.round((tt - tt[0]) / spp) * spp
        worst = max(worst, float(np.ptp(e)))
    return worst


N = 1 << 14


@pytest.mark.parametrize("cycles,k,bound", [(257, 8.0, 1e-5), (333, 16.0, 1e-6),
                                            (1001, 4.0, 1e-3), (257, 2.0, 1e-2)])
def test_an_edge_between_samples_is_at_its_own_time(cycles, k, bound):
    """`k` is the edge in samples. The known answer first: the old construction, shaped by
    the same filter, reads one sample peak to peak on this instrument."""
    spp = N / cycles
    old = P._shape_edges(_legacy(N, 1.0, cycles / N), k)
    assert 0.9 < _tie_pp(old, 0.0, spp) < 1.05
    sq = P.square(cycles=cycles, tr_frac=k / spp, n=N)
    cm = P.cmos(v_lo=0.0, v_hi=3.3, cycles=cycles, tr_s=k, n=N, fs=1.0)
    assert _tie_pp(sq, 0.0, spp) < bound
    assert _tie_pp(cm, 1.65, spp) < bound


def test_on_a_64_gsa_grid_the_clock_no_longer_carries_a_sample_of_jitter():
    """The figure the gap was reported with: 0.997 GHz at 64 GSa/s, 15.6 ps peak to peak."""
    g = Grid(fs=64e9, n=1 << 15)
    spp = 64e9 / 0.997e9
    for kind, level, kw in (("square", 0.0, dict(tr_frac=0.15)),
                            ("cmos", 1.65, dict(v_lo=0.0, v_hi=3.3, tr_s=150e-12))):
        x = Signal(seed=1, grid=g).carrier(kind, f_hz=0.997e9, **kw).waveform()
        assert _tie_pp(x, level, spp) / 64e9 * 1e12 < 1e-3          # ps


@pytest.mark.parametrize("duty", [0.3, 0.5, 0.123456])
def test_the_duty_is_the_one_asked_for(duty):
    """The mean of a +/-1 train over whole periods is 2*duty - 1. With 76.8 samples high the
    old train was high for 77 of 256: -0.3984 where -0.4 was asked. (The record starts three
    quarters of the way through a period, so no edge sits against its ends.)"""
    kw = dict(f_hz=1e9, duty=duty, phase_rad=1.5 * np.pi, n=8192, fs=256e9)   # 32 periods
    x = P.square(tr_frac=0.05, **kw)
    assert abs(x.mean() - (2 * duty - 1)) < 1e-9
    v = P.cmos(v_lo=0.0, v_hi=3.3, tr_s=20e-12, **kw)
    assert abs(v.mean() - 3.3 * duty) < 1e-9
    if duty == 0.3:
        old = P._shape_edges(_legacy(8192, 256e9, 1e9, duty, 1.5 * np.pi), 0.05 * 256)
        assert abs(old.mean() - (2 * duty - 1)) > 1e-3


@pytest.mark.parametrize("kw", [
    dict(fs=10e9, f_hz=10e6, n=4096),
    dict(fs=1e8, f_hz=1e6, n=4096),
    dict(fs=2.0 ** 30, f_hz=2.0 ** 20, n=8192),
    dict(fs=1e9, cycles=16, n=4096, duty=0.25),
    dict(fs=1e9, cycles=8, n=4096, phase_rad=np.pi / 2),
    dict(cycles=16, n=4096),
])
def test_edges_that_were_on_samples_and_rightly_placed_are_bit_identical(kw):
    """Where every edge is on a sample and the old phase test put it there, nothing moved:
    the ideal train is the same array, so the shaped record is the same bits. The old train
    is rebuilt here exactly as `square` built it, from the library's own timebase."""
    duty, phase = kw.get("duty", 0.5), kw.get("phase_rad", 0.0)
    t, f, _ = P._cycles_and_t(kw.get("f_hz"), kw.get("cycles"), kw["n"], kw.get("fs"), None)
    old = np.where((f * t + phase / (2 * np.pi)) % 1.0 < duty, 1.0, -1.0)
    spp = (1.0 / f) / (t[1] - t[0])
    assert abs(spp - round(spp)) < 1e-9 and np.ptp(np.diff(np.flatnonzero(np.diff(old)))) <= \
        abs(round(spp) * (1 - 2 * duty))                    # on samples, and evenly spaced
    assert np.array_equal(P.square(tr_frac=0.05, **kw),
                          P._shape_edges(old, max(0.05 * spp, 2.0)))
    if kw.get("fs"):
        assert np.array_equal(P.cmos(v_lo=0.0, v_hi=3.3, tr_s=8 / kw["fs"], **kw),
                              0.0 + 3.3 * (0.5 * (1.0 + P._shape_edges(old, 8.0))))


def test_a_commensurate_clock_has_one_period():
    """1 GHz on 64 GSa/s is 64 samples a period, every edge on a sample. The old phase test
    rounded some of them one sample late (periods of 63 and 65 among the 64s); every period
    is 64 now. This is the case that is NOT bit-identical, and the old record was wrong."""
    def periods(ideal):
        hi = ideal > 0
        rise = np.flatnonzero(hi[1:] & ~hi[:-1])
        return set(np.diff(rise).tolist())

    assert periods(_legacy(1 << 15, 64e9, 1e9)) == {63, 64, 65}
    new = P._edge_train(1 << 15, 64e9 / 1e9, 0.5, 0.0)
    assert periods(new) == {64} and set(np.unique(new).tolist()) == {-1.0, 1.0}
    x = Signal(seed=1, grid=Grid(fs=64e9, n=1 << 15)).carrier("square", f_hz=1e9).waveform()
    assert _tie_pp(x, 0.0, 64.0) < 1e-9


def test_a_constant_duty_and_a_phase_offset_are_what_they_were():
    assert np.array_equal(P._edge_train(64, 16.0, 1.0, 0.0), np.ones(64))
    assert np.array_equal(P._edge_train(64, 16.0, 0.0, 0.0), -np.ones(64))
    # a quarter period late is the same train moved by a quarter period, fraction and all
    spp = N / 257
    a = P.square(cycles=257, tr_frac=8 / spp, n=N)
    b = P.square(cycles=257, tr_frac=8 / spp, n=N, phase_rad=-np.pi / 2)
    ta, da = _crossings(a, 0.0)
    tb, db = _crossings(b, 0.0)
    ta, tb = ta[da > 0], tb[db > 0]
    ta, tb = ta[(ta > 2000) & (ta < 12000)], tb[(tb > 2000) & (tb < 12000)]
    lag = (tb[:20] - ta[:20]) % spp
    assert np.allclose(lag, spp / 4, atol=1e-4)


def test_the_consumers_of_these_carriers_still_render_without_warning():
    """`open_drain`, `pass_fet` and a `net` driver threshold or scale the carrier; the
    interpolated ideal train never reaches them, only the shaped edge does."""
    g = Grid(fs=1e8, n=4096)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        od = (Signal(seed=2, grid=g).carrier("cmos", v_lo=0.0, v_hi=3.3, f_hz=3.3e5, tr_s=8e-8)
              .open_drain(r_pullup_ohm=1e3, c_bus_f=2e-9).waveform())
        fet = (Signal(seed=2, grid=g).carrier("sine", f_hz=1e6)
               .pass_fet(gate=dict(f_hz=3.3e5)).waveform())
    assert np.isfinite(od).all() and np.isfinite(fet).all()
    assert od.min() > -0.01 and od.max() < 3.31


def test_interpolating_in_blocks_bounds_the_temporaries_and_not_the_result(monkeypatch):
    """A dense train (4.67 samples a period) has an edge every other sample. The per-edge
    temporaries are made a block of edges at a time: the same bits as one pass, and a peak
    near the six records the old construction held (measured 6.3; one pass over all edges was 21)."""
    import tracemalloc
    n = 1 << 19
    blocked = P.square(cycles=n / 4.67, n=n)
    monkeypatch.setattr(P, "EDGE_BLOCK", 1 << 30)
    assert np.array_equal(P.square(cycles=n / 4.67, n=n), blocked)
    monkeypatch.undo()
    tracemalloc.start()
    P.square(cycles=n / 4.67, n=n)
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    assert peak < 7.0 * blocked.nbytes, f"{peak / blocked.nbytes:.1f} records"
