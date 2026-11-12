"""The compositions of GitHub #91 in `wfmsynth.recipes`: hum, ripple, pickup, bounce,
intermittent, oscillator_startup, multitone and buck.

Each is tested twice. FIRST against the chain it claims to be, written out by hand here from
ops alone: the recipe and the samples must be the same, bit for bit, so the function adds
nothing a caller could not have typed. THEN for the property its docstring states, against a
closed form or an independent measurement, with the measured number printed:

    hum                  line amplitudes; every harmonic's frequency under a wandering line
    ripple               the triangle's peak-to-peak and Fourier fundamental; the ring's
                         amplitude, frequency and decay at each edge; the swept clock's range
    pickup               carrier and AM sidebands; FM lines against Bessel functions
    bounce               closure count, instants and gaps, from the gate's own events and
                         from the samples
    intermittent         dwell means and the open fraction
    oscillator_startup   the logistic envelope, time to 90 %, 10-90 % time, the frequency
                         error's start, time constant and settled value
    multitone            every line's amplitude and phase, an empty notch, the crest factor
                         per phase policy, on a voltage and on an envelope
    buck                 mean, ripple amplitude and frequency against the textbook expression

`spur` as the first op of a chain (a tone with no signal under it) is the one thing this
issue changed outside `recipes.py`; it is tested at the end.
"""
from __future__ import annotations

import json
import warnings

import numpy as np
import pytest
from scipy import signal as sps
from scipy import special

import wfmsynth as ws
from wfmsynth.compose import Signal
from wfmsynth.grid import Grid

R = ws.recipes


def record(sub, grid, seed=1):
    """A sub-chain alone on a record: added to nothing."""
    return Signal(seed=seed, grid=grid).carrier("dc", level=0.0).combine(sub)


def same(a, b):
    """Two Signals are the same recipe and the same samples."""
    assert json.dumps(a.recipe(), sort_keys=True) == json.dumps(b.recipe(), sort_keys=True)
    xa, xb = a.waveform(), b.waveform()
    assert xa.dtype == xb.dtype and np.array_equal(xa, xb)
    return xa


def line(x, fs, f_hz):
    """The complex amplitude of the component at `f_hz` (a whole number of cycles)."""
    t = np.arange(len(x)) / fs
    c = np.mean(x * np.exp(-2j * np.pi * f_hz * t))
    return c if np.iscomplexobj(x) else 2.0 * c


def inst_freq(x, fs, lo_hz, hi_hz):
    """Instantaneous frequency of the part of `x` between two frequencies."""
    f = np.fft.rfftfreq(len(x), 1.0 / fs)
    y = np.fft.irfft(np.where((f > lo_hz) & (f < hi_hz), np.fft.rfft(x), 0.0), len(x))
    return np.gradient(np.unwrap(np.angle(sps.hilbert(y)))) * fs / (2.0 * np.pi)


# ------------------------------------------------------------------ hum
HG = Grid(fs=16384.0, n=1 << 16)                       # 4 s: 200 line periods


def test_hum_is_the_chain_it_says_and_its_lines_are_the_stated_ones():
    got = record(R.hum(50.0, amp=0.02, harmonics={3: 0.3, 2: 0.5}, phase_rad=0.4), HG)
    hand = record(
        Signal().carrier("sine", f_hz=50.0, amp=0.02, phase_rad=0.4)
        .combine([Signal().carrier("sine", f_hz=100.0, amp=0.01, phase_rad=0.8),
                  Signal().carrier("sine", f_hz=150.0, amp=0.02 * 0.3, phase_rad=3 * 0.4)]), HG)
    x = same(got, hand)
    amps = [abs(line(x, HG.fs, 50.0 * k)) for k in (1, 2, 3, 4, 5)]
    print(f"hum lines at 50/100/150/200/250 Hz: {amps}")
    assert np.allclose(amps[:3], [0.02, 0.01, 0.006], rtol=1e-12)
    assert max(amps[3:]) < 1e-15
    # phase_rad slides the waveform in time: harmonic k turns by k times it
    turned = [np.angle(line(x, HG.fs, 50.0 * k) * 1j) for k in (1, 2, 3)]
    assert np.allclose(turned, [0.4, 0.8, 1.2], atol=1e-12)
    assert got.roles() == []


def test_a_wandering_line_moves_every_harmonic_by_the_same_fraction():
    got = record(R.hum(50.0, amp=0.02, harmonics={3: 0.3}, f_drift_hz=0.5, f_wander_hz=1.0), HG)
    a = 0.5 / (2 * np.pi * 1.0 * 50.0)
    hand = record(
        Signal().carrier("sine", f_hz=50.0, amp=0.02, phase_rad=0.0)
        .combine([Signal().carrier("sine", f_hz=150.0, amp=0.02 * 0.3, phase_rad=0.0)])
        .delay(by=Signal().carrier("sine", f_hz=1.0, amp=a, offset=a, phase_rad=-0.5 * np.pi)),
        HG)
    x = same(got, hand)
    t = np.arange(HG.n) / HG.fs
    keep = slice(HG.n // 8, -HG.n // 8)
    for k in (1, 3):
        f = inst_freq(x, HG.fs, 50.0 * k - 20.0, 50.0 * k + 20.0)[keep]
        want = k * (50.0 - 0.5 * np.sin(2 * np.pi * 1.0 * t))[keep]
        err = float(np.abs(f - want).max())
        print(f"harmonic {k}: swings {f.min() - 50 * k:+.5f} .. {f.max() - 50 * k:+.5f} Hz, "
              f"largest error against k*(f - f_drift*sin) {err:.1e} Hz")
        assert err < 1e-4 * k
    assert abs(x[0]) < 1e-15                        # the delay starts at zero: nothing held


# ------------------------------------------------------------------ ripple
RG = Grid(fs=1e9, n=1 << 18)
F_SW = RG.fs / RG.n * 128                              # 488.28 kHz, 2048 samples a period
DUTY = 0.3


def test_ripple_is_a_triangle_with_the_stated_swing_and_fundamental():
    got = record(R.ripple(F_SW, amp=0.01, duty=DUTY), RG)
    x = same(got, record(Signal().carrier("triangle", f_hz=F_SW, amp=0.01, symmetry=DUTY), RG))
    pp = float(x.max() - x.min())
    fund = abs(line(x, RG.fs, F_SW))
    want = 2 * 0.01 * np.sin(np.pi * DUTY) / (np.pi ** 2 * DUTY * (1 - DUTY))
    print(f"ripple: {pp:.6f} peak to peak (2*amp = 0.02); fundamental {fund:.8f} against "
          f"the triangle's Fourier term {want:.8f}")
    assert abs(pp / 0.02 - 1) < 1e-3                # a vertex between two samples
    assert abs(fund / want - 1) < 1e-6
    i_peak = int(np.argmax(x[:2048]))
    assert abs(i_peak - DUTY * 2048) <= 1           # it rises for `duty` of the period


@pytest.mark.parametrize("per,q", [(16, 2.0), (16, 4.0), (32, 4.0), (32, 6.0), (64, 2.0)])
def test_each_switching_edge_starts_the_stated_ring(per, q):
    f_ring = RG.fs / per
    got = record(R.ripple(F_SW, amp=0.01, duty=DUTY, spike=0.05, f_ring_hz=f_ring, q=q), RG)
    half = 0.5 / q
    root, damped = np.sqrt(1 + half * half), np.sqrt(1 - half * half)
    hand = record(
        Signal().carrier("triangle", f_hz=F_SW, amp=0.01, symmetry=DUTY)
        .combine(Signal().carrier("square", f_hz=F_SW, duty=DUTY,
                                  amp=0.5 * 0.05 * q * damped, tr_frac=0.125 * F_SW / f_ring)
                 .filter("butter", btype="bandpass", order=1,
                         f_hz=[f_ring * (root - half), f_ring * (root + half)])), RG)
    y = same(got, hand) - record(R.ripple(F_SW, amp=0.01, duty=DUTY), RG).waveform()
    f_d, alpha = f_ring * damped, np.pi * f_ring / q
    for edge, sign in ((3.0, 1.0), (3.0 + DUTY, -1.0)):   # up at the trough, down at the peak
        t_edge = edge / F_SW
        i0 = int(np.ceil(t_edge * RG.fs))
        tt = np.arange(i0, i0 + int(2.5 * q * per)) / RG.fs - t_edge
        # the stated ring with a free amplitude and start phase: exp(-alpha t) sin(2 pi f_d t + p)
        basis = np.exp((-alpha + 2j * np.pi * f_d) * tt)
        A = np.column_stack([basis.imag, basis.real])
        c = np.linalg.lstsq(A, y[i0:i0 + len(tt)], rcond=None)[0]
        amp = float(np.hypot(*c))
        late = float(np.arctan2(sign * c[1], sign * c[0]) / (2 * np.pi) * per)
        left = float(np.abs(y[i0:i0 + len(tt)] - A @ c)[per:].max())
        print(f"{per} samples a ring period, q={q:g}, edge at {edge:g} periods: amplitude "
              f"{amp / 0.05:.4f} of spike, starts {late:+.3f} samples from the edge, "
              f"{left / 0.05:.4f} of spike not that shape after one period")
        assert 0.92 < amp / 0.05 < 0.98             # the square's own edge takes 3 to 7 % off
        assert abs(late) < 0.75                     # the sign and the instant are the edge's
        assert left < 0.01 * 0.05                   # frequency and decay are the stated ones


def test_a_spread_clock_sweeps_the_stated_range_and_turns_the_line_into_a_band():
    g = Grid(fs=250e6, n=1 << 20)
    df = g.fs / g.n
    f_sw, f_spread = 2048 * df, 4 * df
    got = record(R.ripple(f_sw, amp=0.01, spread_pct=6.0, f_spread_hz=f_spread), g)
    x = same(got, record(Signal().carrier("triangle", f_hz=f_sw, amp=0.01, symmetry=0.5)
                         .ssc(f_ssc=f_spread, spread=0.06, profile="center"), g))
    f = inst_freq(x, g.fs, 0.8 * f_sw, 1.2 * f_sw)[g.n // 16:-g.n // 16] / f_sw
    freqs = np.fft.rfftfreq(g.n, 1 / g.fs)
    near = (freqs > 0.8 * f_sw) & (freqs < 1.2 * f_sw)
    p1 = np.abs(np.fft.rfft(x)[near]) ** 2
    p0 = np.abs(np.fft.rfft(record(R.ripple(f_sw, amp=0.01), g).waveform())[near]) ** 2
    drop = 10 * np.log10(p1.max() / p0.max())
    print(f"spread 6 %: switching frequency {f.min():.4f} .. {f.max():.4f} of f_sw; the "
          f"fundamental's strongest bin falls {drop:.1f} dB and its power is "
          f"{p1.sum() / p0.sum():.5f} of the unspread line's")
    assert abs(f.min() - 0.94) < 1e-3 and abs(f.max() - 1.06) < 1e-3
    assert drop < -10.0 and abs(p1.sum() / p0.sum() - 1) < 1e-3


# ------------------------------------------------------------------ pickup
PG = Grid(fs=1e7, n=1 << 16)
PDF = PG.fs / PG.n
FC, FM = 4096 * PDF, 64 * PDF


def test_pickup_plain_and_am_have_the_stated_lines():
    plain = same(record(R.pickup(FC, amp=0.01), PG),
                 record(Signal().carrier("sine", f_hz=FC, amp=0.01), PG))
    assert abs(abs(line(plain, PG.fs, FC)) - 0.01) < 1e-15
    msg = Signal().carrier("sine", f_hz=FM)
    x = same(record(R.pickup(FC, amp=0.01, message=msg, depth=0.4), PG),
             record(Signal().carrier("sine", f_hz=FC, amp=0.01)
                    .modulate(kind="am", depth=0.4, message=msg), PG))
    got = [abs(line(x, PG.fs, f)) for f in (FC - FM, FC, FC + FM, FC + 2 * FM)]
    print(f"AM pickup, depth 0.4: lower/carrier/upper/next {got}")
    assert np.allclose(got[:3], [0.002, 0.01, 0.002], rtol=1e-12) and got[3] < 1e-15


def test_pickup_fm_has_bessel_lines():
    msg = Signal().carrier("sine", f_hz=FM, phase_rad=0.5 * np.pi)       # a cosine message
    x = same(record(R.pickup(FC, amp=0.01, message=msg, dev_hz=2 * FM), PG),
             record(Signal().modulate(kind="fm", fc_hz=FC, dev_hz=2 * FM, amp=0.01,
                                      message=msg), PG))
    got = np.array([abs(line(x, PG.fs, FC + k * FM)) / 0.01 for k in range(5)])
    want = np.abs(special.jv(np.arange(5), 2.0))
    print(f"FM pickup, index 2: lines {got.round(6)} against |J_k(2)| {want.round(6)}")
    assert np.abs(got - want).max() < 1e-4
    noisy = record(R.pickup(FC, amp=0.01, dev_hz=5e3,
                            message=Signal().carrier("noise", rms=0.3)), PG)
    assert noisy.roles() == ["combine/1/others/0/modulate/0/message/carrier_noise/0"]


# ------------------------------------------------------------------ bounce
BG = Grid(fs=1e6, n=1 << 13)


def test_bounce_closes_n_plus_one_times_at_the_stated_instants():
    got = Signal(seed=1, grid=BG).carrier("dc", level=1.0).gate(
        by=R.bounce(t_s=2e-3, gap_s=400e-6, n=3, ratio=0.5))

    def level(v, d=None):
        seg = {"chain": Signal().carrier("dc", level=v)}
        return seg if d is None else dict(seg, duration_s=d)

    hand = Signal(seed=1, grid=BG).carrier("dc", level=1.0).gate(by=Signal().sequence([
        level(0.0, 2e-3), level(1.0, 100e-6), level(0.0, 400e-6), level(1.0, 100e-6),
        level(0.0, 200e-6), level(1.0, 100e-6), level(0.0, 100e-6), level(1.0)]))
    x = same(got, hand)
    closes, opens = R.bounce_times(2e-3, 400e-6, n=3, ratio=0.5)
    assert np.allclose(closes, [2e-3, 2.5e-3, 2.8e-3, 3.0e-3], atol=1e-15)
    assert abs(closes[-1] - (2e-3 + 3 * 100e-6 + 400e-6 * (1 - 0.5 ** 3) / 0.5)) < 1e-15
    # the gate's own ground truth
    ev = [e for e in got.event_list() if e.kind == "transition"]
    on = np.array([e.t_s for e in ev if e.params["state"] == "on"])
    off = np.array([e.t_s for e in ev if e.params["state"] == "off"])
    # and the samples, independently
    d = np.diff(x)
    rise, fall = (np.flatnonzero(d > 0.5) + 1) / BG.fs, (np.flatnonzero(d < -0.5) + 1) / BG.fs
    gaps = rise[1:] - fall
    print(f"bounce: closes at {rise * 1e3} ms, opens at {fall * 1e3} ms, gaps {gaps * 1e6} us")
    assert len(on) == len(rise) == 4 and len(off) == len(fall) == 3
    assert np.abs(rise - closes).max() < 1e-12 and np.abs(fall - opens).max() < 1e-12
    assert np.abs(on - closes).max() <= 0.5 / BG.fs + 1e-12      # a crossing between samples
    assert np.abs(off - opens).max() <= 0.5 / BG.fs + 1e-12
    assert np.allclose(gaps, [400e-6, 200e-6, 100e-6], atol=1e-12)
    assert np.allclose(gaps[1:] / gaps[:-1], 0.5)
    assert got.roles() == []
    assert x[int(3.5e-3 * BG.fs):].min() == 1.0     # and then it stays closed


def test_bounce_from_the_first_sample_has_no_open_segment_before_it():
    s = R.bounce(t_s=0.0, gap_s=100e-6, n=1, touch_s=50e-6)
    assert len(s.ops[0]["segments"]) == 3
    x = record(s, BG).waveform()
    assert x[0] == 1.0 and x[75] == 0.0 and x[200] == 1.0


# ------------------------------------------------------------------ intermittent
def test_intermittent_is_a_two_state_source_with_the_stated_dwells():
    g = Grid(fs=1e6, n=1 << 21)                        # 2.1 s
    got = Signal(seed=4, grid=g).carrier("dc", level=1.0).gate(
        by=R.intermittent(closed_s=5e-3, open_s=50e-6))
    hand = Signal(seed=4, grid=g).carrier("dc", level=1.0).gate(
        by=Signal().carrier("states", levels=[1.0, 0.0], dwell=[5e-3, 50e-6], start=0))
    x = same(got, hand)
    assert got.roles() == ["gate/1/by/states/0"]
    iv = [i for i in ws.states.intervals(got.event_list(), at="gate/1/by")]
    assert iv[0]["state"] == 0 and iv[0]["t_start_s"] == 0.0 and x[0] == 1.0   # starts closed
    dwell = {s: np.array([i["t_stop_s"] - i["t_start_s"] for i in iv
                          if i["state"] == s and i["complete"]]) for s in (0, 1)}
    n = len(dwell[1])
    print(f"intermittent over {g.n / g.fs:.2f} s: {n} opens, mean {dwell[1].mean() * 1e6:.2f} us "
          f"(50), mean closed {dwell[0].mean() * 1e3:.3f} ms (5), open "
          f"{1 - x.mean():.5f} of the time ({50e-6 / 5.05e-3:.5f})")
    assert n > 300
    for s, mean in ((0, 5e-3), (1, 50e-6)):         # an exponential's mean, 4 standard errors
        assert abs(dwell[s].mean() / mean - 1) < 4 / np.sqrt(len(dwell[s]))
    assert abs((1 - x.mean()) / (50e-6 / 5.05e-3) - 1) < 0.15
    soft = R.intermittent(5e-3, {"dist": "lognormal", "median_s": 30e-6, "sigma": 1.0},
                          tr_s=10e-6)
    assert soft.ops[0]["tr_s"] == 10e-6 and soft.ops[0]["dwell"][1]["dist"] == "lognormal"


# ------------------------------------------------------------------ oscillator_startup
OG = Grid(fs=160e6, n=1 << 17)                         # 819 us
F0, T90, PPM, TAU_F = 10e6, 200e-6, 300.0, 100e-6


def test_oscillator_startup_is_the_chain_it_says():
    got = R.oscillator_startup(F0, startup_s=T90, ppm=PPM, settle_s=TAU_F,
                               sig=Signal(seed=1, grid=OG))
    tau = 100e-6 / (2 * np.log(9.0))
    hand = (Signal(seed=1, grid=OG)
            .modulate(kind="fm", fc_hz=F0, dev_hz=F0 * PPM * 1e-6, amp=1.0,
                      message=Signal().carrier("exp", decay=True, tau_s=TAU_F, t_start_s=0.0))
            .gain(by=Signal().carrier("dc", level=1.0).filter("integrate", tau_s=tau)
                  .offset(by=-(T90 - 0.5 * 100e-6) / tau).law("tanh", gain=0.25, sat=0.5).offset(by=0.5)))
    same(got, hand)
    assert got.roles() == []
    plain = R.oscillator_startup(F0, startup_s=T90, sig=Signal(seed=1, grid=OG))
    assert [o["op"] for o in plain.ops] == ["carrier", "gain"] and plain.ops[0]["kind"] == "sine"
    sub = record(R.oscillator_startup(F0, startup_s=T90), OG).waveform()   # and as a sub-chain
    assert np.array_equal(sub, plain.waveform())


def test_the_amplitude_is_the_logistic_and_reaches_90_percent_at_startup_s():
    osc = R.oscillator_startup(F0, startup_s=T90, rise_s=80e-6, sig=Signal(seed=1, grid=OG))
    env = osc.subchain("gain/1/by")
    t = np.arange(OG.n) / OG.fs
    tau = 80e-6 / (2 * np.log(9.0))
    want = 1.0 / (1.0 + np.exp(-(t - (T90 - 40e-6)) / tau))

    def crossing(level):
        i = int(np.argmax(env >= level))
        return t[i - 1] + (level - env[i - 1]) / (env[i] - env[i - 1]) / OG.fs

    t90, rise = crossing(0.9), crossing(0.9) - crossing(0.1)
    # the record itself: its cycle peaks follow the envelope
    x = osc.waveform()
    peaks = sps.find_peaks(x)[0]
    peaks = peaks[t[peaks] > 150e-6]
    worst = float(np.abs(x[peaks] / want[peaks] - 1).max())
    print(f"oscillator: envelope within {np.abs(env - want).max():.1e} of the logistic; 90 % "
          f"at {t90 * 1e6:.4f} us (200), 10-90 % in {rise * 1e6:.4f} us (80); cycle peaks "
          f"within {worst:.1e} of it; first sample {20 * np.log10(env[0]):.1f} dB")
    assert np.abs(env - want).max() < 1e-12
    assert abs(t90 - T90) < 1e-10 and abs(rise - 80e-6) < 1e-10
    assert worst < 5e-3                             # 16 samples a cycle: a peak between samples
    # early on it is exponential growth: the log of the envelope is a straight line of slope 1/tau
    early = t < 60e-6
    slope = np.polyfit(t[early], np.log(env[early]), 1)[0]
    assert abs(slope * tau - 1) < 1e-2
    default = R.oscillator_startup(F0, startup_s=T90).ops[1]["by"]
    start = record({"ops": default["ops"]}, OG).waveform()[0]
    assert abs(20 * np.log10(start) + 57.27) < 0.01


def test_the_frequency_starts_off_by_ppm_and_settles_with_the_stated_time_constant():
    osc = R.oscillator_startup(F0, startup_s=T90, ppm=PPM, settle_s=TAU_F,
                               sig=Signal(seed=1, grid=OG))
    x = osc.waveform() / osc.subchain("gain/1/by")       # the tone, its growth divided out
    t = np.arange(OG.n) / OG.fs
    excess = np.unwrap(np.angle(sps.hilbert(x))) - 2 * np.pi * F0 * t
    want = 2 * np.pi * F0 * PPM * 1e-6 * TAU_F * (1 - np.exp(-t / TAU_F))
    inner = slice(2000, -2000)
    d = (excess - want)[inner]
    w = int(10e-6 * OG.fs)                               # frequency over 20 us windows
    tc = np.arange(20e-6, 500e-6, 10e-6)
    f_err = np.array([(excess[int(c * OG.fs) + w] - excess[int(c * OG.fs) - w])
                      / (2 * np.pi * 2 * w / OG.fs) for c in tc])
    slope, at0 = np.polyfit(tc, np.log(f_err), 1)
    end = (excess[-3000] - excess[-19000]) / (2 * np.pi * 16000 / OG.fs)
    print(f"oscillator frequency: phase within {d.max() - d.min():.1e} rad of the integral of "
          f"f*(1 + ppm*exp(-t/tau)); starts {np.exp(at0) / F0 * 1e6:.2f} ppm high (300), time "
          f"constant {-1e6 / slope:.3f} us (100), {end:.2f} Hz high over the last 100 us "
          f"({end / F0 * 1e6:.3f} ppm)")
    assert d.max() - d.min() < 5e-3
    assert abs(np.exp(at0) / (F0 * PPM * 1e-6) - 1) < 5e-3
    assert abs(-1 / slope / TAU_F - 1) < 2e-3
    assert abs(end / F0) < 0.3e-6                         # settled: 300 ppm * exp(-7.5)


# ------------------------------------------------------------------ multitone
MG = Grid(fs=64e6, n=1 << 16)
MI = Grid(fs=64e6, n=1 << 16, fc_hz=1e9)
MDF = MG.fs / MG.n


def crest(x):
    return float(np.abs(x).max() / np.sqrt(np.mean(np.abs(x) ** 2)))


def comb(n, **kw):
    return R.multitone(1024 * MDF, (1024 + 64 * (n - 1)) * MDF, n, sig=Signal(seed=1, grid=MG),
                       **kw)


def comb_iq(n, **kw):
    return R.multitone(-32 * (n - 1) * MDF, 32 * (n - 1) * MDF, n, domain="iq",
                       sig=Signal(seed=1, grid=MI), **kw)


def test_multitone_is_the_chain_it_says_on_a_voltage_and_on_an_envelope():
    f = [1024 * MDF, 1088 * MDF, 1152 * MDF]
    ph = [0.0, np.pi / 3, 4 * np.pi / 3]                  # newman, 3 tones: pi k^2 / 3
    a = np.sqrt(2.0 / 3)
    hand = (Signal(seed=1, grid=MG).carrier("sine", f_hz=f[0], amp=a, phase_rad=ph[0] + np.pi / 2)
            .combine([Signal().carrier("sine", f_hz=f[k], amp=a, phase_rad=ph[k] + np.pi / 2)
                      for k in (1, 2)]))
    x = same(comb(3), hand)
    t = np.arange(MG.n) / MG.fs
    assert np.abs(x - a * sum(np.cos(2 * np.pi * fk * t + pk) for fk, pk in zip(f, ph))).max() < 1e-9
    hand = Signal(seed=1, grid=MI)
    for fk, pk in zip([-64 * MDF, 0.0, 64 * MDF], ph):
        hand = hand.spur(f_hz=fk, dbc=0.0, ref_power=1.0 / 3, phase_rad=pk)
    z = same(comb_iq(3), hand)
    assert np.iscomplexobj(z)
    want = sum(np.exp(1j * (2 * np.pi * fk * t + pk)) for fk, pk in zip([-64 * MDF, 0.0, 64 * MDF], ph))
    assert np.abs(z - want / np.sqrt(3)).max() < 1e-9
    assert comb(3).roles() == [] and comb_iq(3).roles() == []


@pytest.mark.parametrize("policy", ["newman", "schroeder", "random", "zero"])
def test_every_line_has_the_stated_level_and_phase(policy):
    seed = 5 if policy == "random" else None
    f, ph, kept = R.multitone_plan(1024 * MDF, (1024 + 64 * 32) * MDF, 33, policy, seed=seed)
    assert kept.all() and len(f) == 33
    x = comb(33, phase=policy, seed=seed).waveform()
    z = comb_iq(33, phase=policy, seed=seed).waveform()
    got = np.array([line(x, MG.fs, fk) for fk in f])
    got_iq = np.array([line(z, MI.fs, fk) for fk in np.linspace(-32 * 32 * MDF, 32 * 32 * MDF, 33)])
    assert np.abs(got - np.sqrt(2.0 / 33) * np.exp(1j * ph)).max() < 1e-12
    assert np.abs(got_iq - np.sqrt(1.0 / 33) * np.exp(1j * ph)).max() < 1e-12
    assert abs(np.sqrt(np.mean(x ** 2)) - 1) < 1e-12 and abs(np.mean(np.abs(z) ** 2) - 1) < 1e-12


def test_the_crest_factor_is_the_phase_policys():
    rows = {}
    for n in (9, 33, 129):
        rows[n] = {p: (crest(comb(n, phase=p, seed=3 if p == "random" else None).waveform()),
                       crest(comb_iq(n, phase=p, seed=3 if p == "random" else None).waveform()))
                   for p in R.PHASES}
        print(f"{n} tones, crest factor on a voltage / an envelope: "
              + ", ".join(f"{p} {v:.3f} / {e:.3f}" for p, (v, e) in rows[n].items()))
        v, e = rows[n]["zero"]
        assert abs(v / np.sqrt(2 * n) - 1) < 1e-3 and abs(e / np.sqrt(n) - 1) < 1e-9
        for p in ("newman", "schroeder"):
            assert 1.75 < rows[n][p][0] < 2.0 and 1.3 < rows[n][p][1] < 1.42
        assert rows[n]["newman"][0] < rows[n]["random"][0] < rows[n]["zero"][0]
        assert rows[n]["newman"][1] < rows[n]["random"][1] < rows[n]["zero"][1]
    # the quadratic policies do not grow with the number of tones; in-phase tones do
    assert rows[129]["newman"][0] < rows[9]["newman"][0] and rows[129]["zero"][0] > 3.7 * rows[9]["zero"][0]


def test_a_notch_removes_its_tones_and_touches_nothing_else():
    lo, hi = (1024 + 64 * 14) * MDF, (1024 + 64 * 18) * MDF
    full, cut = comb(33).waveform(), comb(33, notch=(lo, hi))
    f, ph, kept = R.multitone_plan(1024 * MDF, (1024 + 64 * 32) * MDF, 33, notch=(lo, hi))
    assert (~kept).sum() == 5 and len(cut.ops[1]["others"]) == 27
    x = cut.waveform()
    amp = np.array([abs(line(x, MG.fs, fk)) for fk in f])
    same_as_full = np.array([abs(line(x, MG.fs, fk) - line(full, MG.fs, fk)) for fk in f[kept]])
    print(f"notch of 5 of 33 tones: kept {amp[kept].min():.6f} .. {amp[kept].max():.6f}, in "
          f"the notch {amp[~kept].max():.1e}; rms {np.sqrt(np.mean(x ** 2)):.6f} "
          f"(sqrt(28/33) = {np.sqrt(28 / 33):.6f})")
    assert amp[~kept].max() < 1e-13 and same_as_full.max() < 1e-13
    assert abs(np.sqrt(np.mean(x ** 2)) - np.sqrt(28 / 33)) < 1e-12
    # what the notch is for: a compressing amplifier fills it, and the fill is the measure
    y = cut.copy().law("tanh", gain=1.0, sat=3.0).waveform()
    out = np.array([abs(line(y, MG.fs, fk)) for fk in f])
    npr = 20 * np.log10(out[kept].mean() / np.sqrt(np.mean(out[~kept] ** 2)))
    print(f"through tanh with rails at 3 rms: noise power ratio {npr:.1f} dB")
    assert 25 < npr < 45
    zi = comb_iq(33, notch=(-2 * 64 * MDF, 2 * 64 * MDF))
    assert len(zi.ops) == 28


def test_random_phases_are_stated_in_the_recipe_and_follow_seed_not_the_record():
    a = comb(17, phase="random", seed=1)
    b = R.multitone(1024 * MDF, (1024 + 64 * 16) * MDF, 17, phase="random", seed=1,
                    sig=Signal(seed=99, grid=MG))
    c = comb(17, phase="random", seed=2)
    assert np.array_equal(a.waveform(), b.waveform())
    assert not np.array_equal(a.waveform(), c.waveform())
    assert a.roles() == []
    again = Signal.from_recipe(json.loads(a.to_json()))
    assert np.array_equal(again.waveform(), a.waveform())


# ------------------------------------------------------------------ buck
KG = Grid(fs=400e6, n=1 << 19)                         # 1.3 ms; 8 samples on the 20 ns edge
CONVERTERS = [dict(v_in=12.0, duty=0.275, f_sw_hz=500e3, l_h=10e-6, c_f=22e-6, r_load_ohm=0.5),
              dict(v_in=12.0, duty=0.5, f_sw_hz=500e3, l_h=10e-6, c_f=22e-6, r_load_ohm=0.5),
              dict(v_in=5.0, duty=0.66, f_sw_hz=500e3, l_h=4.7e-6, c_f=10e-6, r_load_ohm=1.0)]


@pytest.mark.parametrize("conv", CONVERTERS)
def test_buck_ripple_matches_the_textbook_expression(conv):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        x = R.buck(sig=Signal(seed=1, grid=KG), **conv).waveform()
    # the switch node resolves on this grid, so `net` has nothing to say. (Only its own
    # warnings are looked at: numpy 2.0 on macOS raises spurious floating-point flags from
    # matmul inside the solve, with the same numbers out.)
    assert [str(w.message) for w in caught if str(w.message).startswith("net:")] == []
    book = R.buck_ripple(**conv)
    assert book["settle_s"] * 20 < KG.n / KG.fs        # the start-up has died: 20 time constants
    per = KG.fs / conv["f_sw_hz"]
    tail = x[-int(20 * per):]
    pp, mean = float(tail.max() - tail.min()), float(tail.mean())
    spec = np.abs(np.fft.rfft(tail - mean))
    f_ripple = float(np.fft.rfftfreq(len(tail), 1 / KG.fs)[np.argmax(spec)])
    want = conv["v_in"] * conv["duty"] * (1 - conv["duty"]) / (
        8 * conv["l_h"] * conv["c_f"] * conv["f_sw_hz"] ** 2)
    print(f"buck {conv['v_in']:g} V, duty {conv['duty']}: mean {mean:.6f} V "
          f"({book['v_out_v']:.6f}), ripple {pp * 1e3:.4f} mV peak to peak "
          f"({want * 1e3:.4f} textbook, {100 * (pp / want - 1):+.3f} %) at {f_ripple:.0f} Hz; "
          f"f_sw / f0 = {conv['f_sw_hz'] / book['f0_hz']:.1f}")
    assert abs(book["v_ripple_pp_v"] / want - 1) < 1e-12
    assert abs(mean / book["v_out_v"] - 1) < 1e-5
    assert abs(pp / want - 1) < 5e-3
    assert abs(f_ripple / conv["f_sw_hz"] - 1) < 1e-9
    assert abs(x[0]) < 1e-9 and x.max() > book["v_out_v"] * 1.01   # at rest, then overshoot


def test_buck_is_one_net_whose_driver_is_a_pwm_chain():
    conv = CONVERTERS[0]
    got = R.buck(sig=Signal(seed=1, grid=KG), **conv)
    pwm = Signal().carrier("cmos", v_lo=0.0, v_hi=12.0, duty=0.275, f_hz=500e3, tr_s=20e-9,
                           phase_rad=np.pi * 1.275)
    hand = Signal(seed=1, grid=KG).net(
        nodes=["sw", "out"],
        elements=[{"id": "c_sw", "kind": "c", "a": "sw", "b": "gnd", "farad": 20e-9 / (2.2 * 0.05)},
                  {"id": "l", "kind": "l", "a": "sw", "b": "out", "henry": 10e-6},
                  {"id": "c", "kind": "c", "a": "out", "b": "gnd", "farad": 22e-6},
                  {"id": "load", "kind": "r", "a": "out", "b": "gnd", "ohm": 0.5}],
        drivers=[{"id": "switch", "node": "sw", "ohm": 0.05, "state": "driving",
                  "source": {"ops": pwm, "volts": True}}],
        probe="out")
    same(got, hand)
    assert [o["op"] for o in got.ops] == ["net"] and got.roles() == []
    assert "kind='cmos'" in got.summary()


# ------------------------------------------------------------------ the rule, for all of them
ALL = {
    "hum": lambda: R.hum(50.0, 0.02, harmonics={3: 0.3}, f_drift_hz=0.1, f_wander_hz=1.0),
    "ripple": lambda: R.ripple(1e3, 0.01, spike=0.02, f_ring_hz=16e3 * 4, spread_pct=2.0,
                               f_spread_hz=50.0),
    "pickup": lambda: R.pickup(2e3, 0.01, message=Signal().carrier("sine", f_hz=100.0)),
    "bounce": lambda: R.bounce(t_s=0.1, gap_s=0.02),
    "intermittent": lambda: R.intermittent(0.2, 0.01),
    "oscillator_startup": lambda: R.oscillator_startup(1e3, startup_s=0.2),
    "multitone": lambda: R.multitone(1e3, 2e3, 5),
}


@pytest.mark.parametrize("name", sorted(ALL))
def test_each_returns_a_visible_sub_chain_that_replays_from_json_alone(name):
    g = Grid(fs=1e6, n=1 << 20)
    sub = ALL[name]()
    assert isinstance(sub, Signal) and sub.seed == 0 and sub.grid is None
    rec = record(sub, g, seed=7)
    text = rec.summary()
    assert f"0  {sub.ops[0]['op']}" in text and "[combine/1/others/0]" in text
    again = Signal.from_recipe(json.loads(rec.to_json()))
    assert again.sha256() == rec.sha256()
    assert np.array_equal(again.waveform(), rec.waveform())


REFUSED = [
    (lambda: R.hum(50.0, 0.02, harmonics={1: 0.5}), "harmonic number"),
    (lambda: R.hum(50.0, 0.02, f_drift_hz=0.1), "both f_drift_hz= and f_wander_hz="),
    (lambda: R.hum(-50.0, 0.02), "f_hz must be a positive number"),
    (lambda: R.ripple(1e5, 0.01, duty=1.0), "duty is the fraction"),
    (lambda: R.ripple(1e5, 0.01, spike=0.1), "both spike= and f_ring_hz="),
    (lambda: R.ripple(1e5, 0.01, spike=0.1, f_ring_hz=2e5), "runs into the next"),
    (lambda: R.ripple(1e5, 0.01, spike=0.1, f_ring_hz=1e7, q=0.4), "does not ring"),
    (lambda: R.ripple(1e5, 0.01, spread_pct=5.0), "both spread_pct= and f_spread_hz="),
    (lambda: R.pickup(1e6, 0.01, dev_hz=1e3), "there is no message="),
    (lambda: R.bounce(1e-3, 1e-4, n=0), "carrier\\('step'\\)"),
    (lambda: R.bounce(-1e-3, 1e-4), "first touches"),
    (lambda: R.oscillator_startup(1e6, 1e-4, rise_s=3e-4), "under 2 \\* startup_s"),
    (lambda: R.oscillator_startup(1e6, 1e-4, ppm=100.0), "both ppm= and settle_s="),
    (lambda: R.oscillator_startup(1e6, 1e-4, sig=Signal().carrier("dc")), "EMPTY Signal"),
    (lambda: R.multitone(1e6, 2e6, 1), "at least 2"),
    (lambda: R.multitone(2e6, 1e6, 4), "f_lo_hz < f_hi_hz"),
    (lambda: R.multitone(1e6, 2e6, 4, phase="random"), "needs seed="),
    (lambda: R.multitone(1e6, 2e6, 4, seed=3), "has none to pick"),
    (lambda: R.multitone(1e6, 2e6, 4, phase="flat"), "phase is one of"),
    (lambda: R.multitone(1e6, 2e6, 4, notch=(3e6, 4e6)), "at least one"),
    (lambda: R.multitone(1e6, 2e6, 4, notch=1.5e6), "notch is \\(lo_hz, hi_hz\\)"),
    (lambda: R.multitone(-1e6, 2e6, 4), "domain='iq'"),
    (lambda: R.multitone(1e6, 2e6, 4, domain="optical"), "domain is 'real'"),
    (lambda: R.buck(12.0, 1.2, 5e5, 1e-5, 2e-5, 1.0), "duty is the fraction"),
    (lambda: R.buck(12.0, 0.5, 5e5, 0.0, 2e-5, 1.0), "l_h must be a positive number"),
]


@pytest.mark.parametrize("call,says", REFUSED, ids=[s for _, s in REFUSED])
def test_what_cannot_be_built_is_refused_by_name(call, says):
    with pytest.raises(ValueError, match=says):
        call()


# ------------------------------------------------------------------ spur as a source
def test_a_spur_may_start_a_chain_as_an_unmodulated_carrier():
    g = Grid(fs=1e6, n=1 << 12, fc_hz=1e9)
    f = 64 * g.fs / g.n
    s = Signal(seed=1, grid=g).spur(f_hz=f, dbc=-6.0, ref_power=4.0, phase_rad=0.3)
    x = s.waveform()
    c = line(x, g.fs, f)
    assert np.iscomplexobj(x) and s.roles() == []
    assert abs(abs(c) - np.sqrt(4.0 * 10 ** -0.6)) < 1e-12 and abs(np.angle(c) - 0.3) < 1e-12
    assert abs(np.mean(np.abs(x) ** 2) - 4.0 * 10 ** -0.6) < 1e-12        # nothing else in it
    assert s.rf_report()[0]["amplitude"] == pytest.approx(np.sqrt(4.0 * 10 ** -0.6))
    assert s.stored_grid(x).domain == "iq"
    # as a sub-chain it is a complex gain: a frequency shift of the record it multiplies
    tx = Signal(seed=1, grid=g).spur(f_hz=0.0, dbc=0.0, ref_power=1.0)
    y = tx.gain(by=Signal().spur(f_hz=f, dbc=0.0, ref_power=1.0)).waveform()
    assert abs(line(y, g.fs, f) - 1.0) < 1e-12


def test_a_spur_that_starts_a_chain_needs_its_power_stated_and_a_record_length():
    g = Grid(fs=1e6, n=1 << 12, fc_hz=1e9)
    with pytest.raises(ValueError, match="spur: this op starts the chain.*ref_power=1.0\\) is a tone"):
        Signal(seed=1, grid=g).spur(f_hz=1e4, dbc=0.0).waveform()
    with pytest.raises(ValueError, match="awgn: this op starts the chain"):
        Signal(seed=1, grid=g).awgn(snr_db=0.0, bw_hz=1e5).waveform()
    with pytest.raises(ValueError):                       # and still not on a voltage
        Signal(seed=1, grid=Grid(fs=1e6, n=1 << 12)).carrier("sine", f_hz=1e4).spur(
            f_hz=1e4, dbc=0.0, ref_power=1.0)
