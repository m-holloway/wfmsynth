"""Fading and multipath as COMPOSITIONS (GitHub #89): `wfmsynth.recipes.fading` and
`.multipath` return chains of existing ops, and the two things that were missing to write
them are one filter kind (`filter("doppler")`) and `awgn` as the first op of a chain.

What is asserted, each against a closed form or an independent method:

    the Doppler spectrum   the power below each frequency against arcsin (Jakes) / a ramp
                           (flat), on the filter and on a rendered record's periodogram; the
                           autocorrelation against J0
    the envelope           Kolmogorov-Smirnov against Rayleigh and Rice, at significance 0.001
    the power delay profile  least-squares taps of the rendered record against the stated
                           gains and delays: exactly when static, on average when fading

THE STATISTICAL TESTS. A KS test needs independent samples, and a fading process is
correlated by construction, so the samples are taken where the correlation is known:

  * "flat": the autocorrelation of a flat spectrum over 2*f_d is a sinc with zeros every
    1/(2*f_d), and the records here hold a whole number of those, so samples that far apart
    are EXACTLY uncorrelated -- jointly Gaussian, hence independent. The test is exact.
  * "jakes": J0 has no evenly spaced zeros. Samples 4 Doppler periods apart have |J0| under
    0.12, so |h|**2 values are correlated under 1.3 %: stated, and negligible against the
    threshold. Rice is tested on "flat" only, because with a line of sight |h| is correlated
    as J0 itself rather than its square.

The seeds are fixed, so each test is deterministic: a p-value above 0.001 is what a correct
process gives 999 times in 1000, and the wrong law is shown failing the same test.
"""
from __future__ import annotations

import json

import numpy as np
import pytest
from scipy import special, stats

import wfmsynth as ws
from wfmsynth import stages as ST
from wfmsynth.compose import Signal
from wfmsynth.grid import Grid

FS, N = 1e6, 1 << 18
FD = FS / 32                                   # 8192 Doppler periods in the record
G = Grid(fs=FS, n=N, fc_hz=2.4e9)
ALPHA = 1e-3


def process(shape="jakes", k_db=None, seed=3):
    """A fading gain rendered as a record of its own."""
    s = ws.recipes.fading(FD, k_db=k_db, shape=shape)
    s.seed, s.grid = seed, G
    return s.waveform()


def cdf(shape, f):
    u = np.clip(f / FD, -1, 1)
    return 0.5 + (np.arcsin(u) / np.pi if shape == "jakes" else 0.5 * u)


# ------------------------------------------------------------------ the two mechanisms
@pytest.mark.parametrize("shape", ST.DOPPLER_SHAPES)
def test_the_doppler_filter_is_its_stated_spectrum_with_unit_mean_power_gain(shape):
    H2 = ST.doppler_gain(N, FS, FD, shape) ** 2
    f = np.fft.fftfreq(N, 1 / FS)
    df = FS / N
    assert np.all(H2[np.abs(f) > FD + df] == 0.0)                  # nothing outside +/-f_d
    assert H2.sum() == pytest.approx(2 * FD / df, rel=1e-12)       # mean gain 1 over the band
    order = np.argsort(f)
    below = np.cumsum(H2[order]) / H2.sum()                        # power below each bin's top
    assert np.abs(below - cdf(shape, f[order] + df / 2)).max() < 1e-12
    if shape == "jakes":                # the U: 1/sqrt(1 - (f/fd)^2), read off the middle bins
        mid = np.abs(f) < 0.9 * FD
        want = (2 / np.pi) / np.sqrt(1 - (f[mid] / FD) ** 2)
        assert np.allclose(H2[mid], want, rtol=1e-3)


def test_awgn_first_in_a_chain_is_a_complex_gaussian_source_of_stated_power():
    s = Signal(seed=5, grid=G).awgn(snr_db=0.0, bw_hz=FS / 4, ref_power=0.5)
    x = s.waveform()
    assert x.dtype == np.complex128 and s.grid.domain == "iq" and s.roles() == ["awgn/0"]
    # 0.5 in a quarter of the band is 2.0 over all of it, half on each rail, uncorrelated
    assert np.mean(np.abs(x) ** 2) == pytest.approx(2.0, rel=1e-12)
    assert np.var(x.real) == pytest.approx(1.0, rel=1e-3) and np.var(x.imag) == pytest.approx(1.0, rel=1e-3)
    assert abs(np.mean(x.real * x.imag)) < 4 / np.sqrt(N)
    assert stats.kstest(x.real[::8], "norm").pvalue > ALPHA
    assert not np.array_equal(s.contrast("awgn/0", seed=1), x)
    assert np.array_equal(Signal.from_recipe(json.loads(s.to_json())).waveform(), x)
    assert s.rf_report()[0]["noise_power"] == pytest.approx(2.0)
    with pytest.raises(ValueError, match="starts the chain.*ref_power"):
        Signal(seed=5, grid=G).awgn(snr_db=0.0, bw_hz=FS / 4).waveform()


# ------------------------------------------------------------------ the Doppler spectrum
@pytest.mark.parametrize("shape,other", [("jakes", "flat"), ("flat", "jakes")])
def test_a_fading_gains_spectrum_is_the_stated_doppler_shape(shape, other):
    h = process(shape)
    P = np.abs(np.fft.fft(h)) ** 2
    f = np.fft.fftfreq(N, 1 / FS)
    order = np.argsort(f)
    below = np.cumsum(P[order]) / P.sum()
    top = f[order] + 0.5 * FS / N
    d_own = float(np.abs(below - cdf(shape, top)).max())
    d_other = float(np.abs(below - cdf(other, top)).max())
    outside = float(P[np.abs(f) > FD + FS / N].sum() / P.sum())
    print(f"{shape}: power below f against its own law, worst {d_own:.4f} (limit 0.03); "
          f"against {other}: {d_other:.3f}; power outside +/-f_d: {outside:.1e}")
    assert d_own < 0.03 < 0.08 < d_other
    assert outside < 1e-20
    assert np.mean(np.abs(h) ** 2) == pytest.approx(1.0, abs=0.03)


def test_a_jakes_gains_autocorrelation_is_the_bessel_function():
    h = process("jakes")
    R = np.fft.ifft(np.abs(np.fft.fft(h)) ** 2).real
    lags = np.arange(0, 6 * 32)                            # six Doppler periods
    got = R[lags] / R[0]
    want = special.j0(2 * np.pi * FD * lags / FS)
    err = float(np.abs(got - want).max())
    print(f"jakes autocorrelation against J0(2 pi f_d tau) over 6 periods: worst {err:.4f}")
    assert err < 0.03
    assert got[12] < 0.1 and want[12] < 0.1                # J0's first zero, 0.383 / f_d


# ------------------------------------------------------------------ the envelope's statistics
def test_a_fading_gains_envelope_is_rayleigh():
    sigma = np.sqrt(0.5)                                   # per rail, for unit mean power
    for shape, step in (("flat", 16), ("jakes", 4 * 32)):
        r = np.abs(process(shape))[::step]
        ks = stats.kstest(r, "rayleigh", args=(0, sigma))
        print(f"{shape}: {len(r)} samples {step} apart, KS against Rayleigh: "
              f"D = {ks.statistic:.4f}, p = {ks.pvalue:.3f} (needs > {ALPHA})")
        assert ks.pvalue > ALPHA
    # the test can fail: a Rician record against the Rayleigh law
    r = np.abs(process("flat", k_db=3.0))[::16]
    assert stats.kstest(r, "rayleigh", args=(0, sigma)).pvalue < 1e-12


@pytest.mark.parametrize("k_db", [0.0, 6.0, 12.0])
def test_a_fading_gain_with_a_line_of_sight_is_rician_at_the_stated_k(k_db):
    k = 10 ** (k_db / 10)
    sigma, los = np.sqrt(0.5 / (1 + k)), np.sqrt(k / (1 + k))
    h = process("flat", k_db=k_db)
    r = np.abs(h)[::16]
    ks = stats.kstest(r, "rice", args=(los / sigma, 0, sigma))
    wrong = stats.kstest(r, "rice", args=(np.sqrt(2 * k * 2) , 0, np.sqrt(0.5 / (1 + 2 * k))))
    k_meas = abs(h.mean()) ** 2 / np.var(h)                # the steady part over the scatter
    print(f"K = {k_db} dB: {len(r)} independent samples, KS against Rice: D = "
          f"{ks.statistic:.4f}, p = {ks.pvalue:.3f}; against K + 3 dB: p = {wrong.pvalue:.1e}; "
          f"K measured {10 * np.log10(k_meas):.2f} dB")
    assert ks.pvalue > ALPHA > wrong.pvalue
    assert 10 * np.log10(k_meas) == pytest.approx(k_db, abs=0.2)
    assert np.mean(np.abs(h) ** 2) == pytest.approx(1.0, abs=0.03)


# ------------------------------------------------------------------ a fade is a visible chain
def test_fading_is_three_visible_ops_with_one_role_and_a_fade_is_a_gain():
    g = Grid(fs=FS, baud=FS / 8, n=1 << 14, fc_hz=2.4e9)
    fade = ws.recipes.fading(2e3, k_db=6.0, los_hz=500.0)
    assert [o["op"] for o in fade.ops] == ["awgn", "filter", "spur"]
    assert fade.ops[1] == {"op": "filter", "kind": "doppler", "f_hz": 2e3, "shape": "jakes"}
    assert fade.ops[0]["ref_power"] + fade.ops[2]["ref_power"] == pytest.approx(1.0)
    sig = Signal(seed=1, grid=g).iq("qpsk", prbs=11).gain(by=fade)
    y = sig.waveform()
    assert sig.roles() == ["gain/1/by/awgn/0"]
    assert "doppler" in sig.summary() and "[gain/1/by]" in sig.summary()
    assert np.array_equal(y, Signal(seed=1, grid=g).iq("qpsk", prbs=11).waveform()
                          * sig.subchain("gain/1/by"))
    assert not np.array_equal(sig.contrast("gain/1/by/awgn/0", seed=9), y)   # another fade
    assert np.array_equal(Signal.from_recipe(json.loads(sig.to_json())).waveform(), y)
    # the line of sight is a tone at its own Doppler shift
    los = ws.recipes.fading(2e3, k_db=60.0, los_hz=500.0)
    los.seed, los.grid = 1, g
    ph = np.unwrap(np.angle(los.waveform()))
    assert np.polyfit(np.arange(g.n) / FS, ph, 1)[0] / (2 * np.pi) == pytest.approx(500.0, abs=1.0)


# ------------------------------------------------------------------ multipath
DELAYS = [0.0, 8e-6, 22.5e-6]                 # 0, 8 and 22.5 samples
GAINS = [1.0, 0.7, 0.4]


def taps(x, y, delays, fs, block=None):
    """Least-squares complex taps of `y` on copies of `x` delayed by `delays` (shifted in the
    frequency domain: not the op's interpolator). One set per `block` samples, or one in all."""
    k = np.fft.fftfreq(len(x))
    X = np.fft.fft(x)
    A = np.stack([np.fft.ifft(X * np.exp(-2j * np.pi * k * d * fs)) for d in delays], axis=1)
    keep = slice(256, len(x) - 256)                        # clear of the shift's wrap
    A, y = A[keep], y[keep]
    if block is None:
        return np.linalg.lstsq(A, y, rcond=None)[0]
    m = len(y) // block
    A, y = A[:m * block].reshape(m, block, -1), y[:m * block].reshape(m, block, 1)
    Ah = np.conj(np.swapaxes(A, 1, 2))
    return np.linalg.solve(Ah @ A, Ah @ y)[..., 0]


def test_static_multipath_has_the_stated_taps_each_turned_by_its_delays_carrier_phase():
    fc = 2.4e9 + 13e3                                      # so the delays are not whole turns
    g = Grid(fs=FS, baud=FS / 4, n=1 << 14, fc_hz=fc)
    tx = Signal(seed=1, grid=g).iq("qpsk", prbs=11)
    ch = ws.recipes.multipath(tx, DELAYS, GAINS)
    assert [o["op"] for o in ch.ops] == ["iq", "combine"]
    assert [[o["op"] for o in p["ops"]] for p in ch.ops[1]["others"]] == [
        ["iq", "delay", "gain"], ["iq", "delay", "gain"]]
    assert ch.roles() == []
    c = taps(tx.waveform(), ch.waveform(), DELAYS, FS)
    want = np.array(GAINS) * np.exp(-2j * np.pi * fc * np.array(DELAYS))
    print("static multipath, power delay profile: "
          + ", ".join(f"{d * 1e6:g} us {abs(v) ** 2:.4f} (stated {w ** 2:.4f})"
                      for d, v, w in zip(DELAYS, c, GAINS))
          + f"; worst complex tap error {np.abs(c - want).max():.1e}")
    assert np.abs(c - want).max() < 1e-4
    assert np.allclose(np.abs(c) ** 2, np.square(GAINS), rtol=1e-4)
    # a complex gain states a phase on top of the delay's
    flipped = ws.recipes.multipath(tx, DELAYS, [1.0, -0.7, 0.4j])
    c2 = taps(tx.waveform(), flipped.waveform(), DELAYS, FS)
    assert np.abs(c2 - want * np.array([1, -1, 1j])).max() < 1e-4
    assert flipped.ops[1]["others"][1]["ops"][-1] == {"op": "gain", "by": [0.0, 0.4]}


def test_fading_multipath_has_the_stated_power_delay_profile_and_independent_taps():
    """A sounder's measurement: taps fitted block by block (256 samples, 1/20 of a Doppler
    period), their mean power against the stated profile. The mean power of a Jakes process
    over the 210 Doppler periods here wanders by 5.6 % rms from record to record (measured over
    24 seeds, 11 % at worst), so the bound is 20 %."""
    g = Grid(fs=FS, baud=FS / 4, n=1 << 20, fc_hz=2.4e9)
    tx = Signal(seed=1, grid=g).iq("qpsk", prbs=11)
    ch = ws.recipes.multipath(tx, DELAYS, GAINS, doppler_hz=200.0)
    assert ch.roles() == ["gain/1/by/awgn/0", "combine/2/others/0/gain/2/by/awgn/0",
                          "combine/2/others/1/gain/2/by/awgn/0"]
    c = taps(tx.waveform(), ch.waveform(), DELAYS, FS, block=256)
    pdp = np.mean(np.abs(c) ** 2, axis=0)
    rho = np.corrcoef(c.T)
    cross = float(np.abs(rho[np.triu_indices(3, 1)]).max())
    print("fading multipath, power delay profile over "
          f"{len(c)} blocks: " + ", ".join(f"{p:.3f} (stated {w ** 2:.3f})"
                                           for p, w in zip(pdp, GAINS))
          + f"; worst correlation between two taps {cross:.3f}")
    assert np.allclose(pdp, np.square(GAINS), rtol=0.20)
    assert cross < 0.2
    for k in range(3):                                     # each tap is Rayleigh
        r = np.abs(c[::64, k]) / GAINS[k]                  # 1.6 Doppler periods apart
        assert stats.kstest(r / np.sqrt(np.mean(r ** 2)), "rayleigh",
                            args=(0, np.sqrt(0.5))).pvalue > ALPHA
    # re-rolling one path's fade moves that tap and leaves the others' fits alone
    c2 = taps(tx.waveform(), ch.contrast("combine/2/others/0/gain/2/by/awgn/0", seed=4),
              DELAYS, FS, block=256)
    moved = np.abs(c2 - c).mean(axis=0)        # a block's fit leaks a little between taps
    print(f"re-rolling path 1's fade moves the fitted taps by {moved.round(4)} on average")
    assert moved[0] < 0.02 and moved[2] < 0.02 < 0.3 < moved[1]


def test_multipath_needs_a_transmitter_that_is_the_same_wherever_it_is_rendered():
    g = Grid(fs=FS, baud=FS / 4, n=1 << 13, fc_hz=2.4e9)
    noisy = Signal().iq("qpsk", prbs=11).awgn(es_n0_db=20)
    with pytest.raises(ValueError, match=r"tx draws random roles \['awgn/1'\].*use\("):
        ws.recipes.multipath(noisy, DELAYS, GAINS)
    # ... which is what a named chain is: the same noisy transmitter down every path
    whole = [0.0, 8e-6, 22e-6]                 # whole samples and whole carrier turns: exact
    x = ws.Channels({"tx": noisy, "rx": ws.recipes.multipath(Signal().use("tx"), whole, GAINS)},
                    seed=1, grid=g).waveforms()
    want = x["tx"][22:] + 0.7 * x["tx"][14:-8] + 0.4 * x["tx"][:-22]
    assert np.allclose(x["rx"][22:], want, atol=1e-9)        # noise and all, three times
    with pytest.raises(ValueError, match="one delay and one gain per path"):
        ws.recipes.multipath(Signal().iq("qpsk", prbs=11), DELAYS, GAINS[:2])
    with pytest.raises(ValueError, match="already has its source"):
        ws.recipes.multipath(Signal(), DELAYS, GAINS)


# ------------------------------------------------------------------ refusals
def test_what_a_fade_cannot_be_is_refused_by_name():
    src = lambda n=1 << 12: Signal(seed=1, grid=Grid(fs=FS, n=n, fc_hz=2.4e9)).awgn(
        snr_db=0.0, bw_hz=1e3, ref_power=1.0)
    with pytest.raises(ValueError, match=r"0.41 Doppler periods.*gain\(by=<complex>\)"):
        src().filter("doppler", f_hz=100.0).waveform()
    with pytest.raises(ValueError, match="between 0 and the Nyquist frequency"):
        src().filter("doppler", f_hz=FS)
    with pytest.raises(ValueError, match="shape is 'jakes'"):
        src().filter("doppler", f_hz=1e4, shape="gauss")
    with pytest.raises(TypeError, match="always applied with zero phase"):
        src().filter("doppler", f_hz=1e4, zero_phase=True)
    with pytest.raises(ValueError, match="doppler_hz.*must be positive"):
        ws.recipes.fading(0.0)
    with pytest.raises(ValueError, match="los_hz.*without k_db"):
        ws.recipes.fading(100.0, los_hz=20.0)
    d = src().filter("doppler", f_hz=1e4).filter_delays()[0]
    assert d["zero_phase"] and d["group_delay_s"] == 0.0
    # a real noise through the same filter stays real
    real = (Signal(seed=1, grid=Grid(fs=FS, n=1 << 12)).carrier("noise", rms=1.0)
            .filter("doppler", f_hz=1e4))
    assert real.waveform().dtype == np.float64
