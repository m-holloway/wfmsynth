"""A delay and a complex gain on an I/Q record (GitHub #89).

`delay` on an envelope is the envelope of the DELAYED PASSBAND SIGNAL: shifted, and turned by
-2*pi*fc*delay. The reference is the real op on the real record, between the two mixers. Both
routes are given the envelope a passband record holds (through `upconvert` -> `downconvert`),
so the mixers' own low-pass is common to both and what is compared is the delay.

`gain(by=)` takes a complex number or an I/Q sub-chain on an I/Q record.

Receiver-side I/Q imbalance needs no op: `iq_imbalance` placed after the channel is it, and
the last test here says exactly in what sense.
"""
from __future__ import annotations

import json

import numpy as np
import pytest
from scipy.signal import fftconvolve

import wfmsynth as ws
from wfmsynth import iq as IQ
from wfmsynth import joins as JN
from wfmsynth.compose import Signal
from wfmsynth.grid import Grid

FS, FC, BAUD, N = 8e9, 1e9, 50e6, 1 << 14
G = Grid(fs=FS, baud=BAUD, n=N, fc_hz=FC)
LPF = dict(pass_hz=300e6, stop_hz=900e6, atten_db=100)
EDGE = 300                       # samples at each end where the mixers' low-pass sees the edge


def env():
    return Signal(seed=1, grid=G).iq("qam16", prbs=11)


def looped():
    """The envelope a passband record holds: through both mixers, with nothing between."""
    return env().upconvert().downconvert(fc_hz=FC, lpf=LPF)


def worst(a, b):
    return float(np.abs(a - b)[EDGE:-EDGE].max() / np.abs(a).max())


WOBBLE = dict(f_hz=2e6, amp=0.2e-9, offset=1e-9)       # a 1 ns delay moving by +/-0.2 ns


@pytest.mark.parametrize("by,bound", [
    (1e-9, 1e-9),                 # 8 whole samples: an exact shift, and exactly one turn
    (0.33e-9, None),              # 2.64 samples, 0.33 of a carrier turn
    (7.77e-9, None),              # 62.16 samples, 7.77 turns
    ("wobble", None),             # a delay that moves
])
def test_delay_on_an_envelope_is_the_envelope_of_the_delayed_passband_record(by, bound):
    how = (lambda: Signal().carrier("sine", **WOBBLE)) if by == "wobble" else (lambda: by)
    ref = env().upconvert().delay(by=how()).downconvert(fc_hz=FC, lpf=LPF).waveform()
    got = looped().delay(by=how()).waveform()
    if bound is None:             # the interpolator's own stated error, on each route
        bound = 2 * JN.describe_delay(FS)["error"]
    err = worst(ref, got)
    print(f"delay by {by}: envelope route against upconvert -> delay -> downconvert, worst "
          f"{err:.2e} of peak (bound {bound:.1e})")
    assert err < bound
    # ... and the rotation is most of it: the same envelope merely shifted is a different record
    if by == 0.33e-9:
        m = by * FS
        k = np.fft.fftfreq(N)
        shifted = np.fft.ifft(np.fft.fft(looped().waveform()) * np.exp(-2j * np.pi * k * m))
        turn = np.exp(-2j * np.pi * FC * by)
        assert worst(ref, shifted * turn) < 1e-3          # shift AND turn: the same thing
        assert worst(ref, shifted) > 1.0                  # shift alone: 119 degrees off


def test_a_constant_delay_is_a_shift_and_a_carrier_phase_exactly():
    x = env().waveform()
    for m, fc in ((8, 1e9), (5, 1.03e9), (40, 2.4e9)):
        g = Grid(fs=FS, baud=BAUD, n=N, fc_hz=fc)
        y = Signal(seed=1, grid=g).iq("qam16", prbs=11).delay(by=m / FS).waveform()
        want = x[:-m] * np.exp(-2j * np.pi * fc * m / FS)
        assert np.allclose(y[m:], want, rtol=0, atol=1e-12)
        assert np.array_equal(y[:m], np.full(m, y[0]))    # the line held its first sample


def test_a_moving_delay_is_a_doppler_shift_of_minus_fc_times_its_rate():
    """A path lengthening at v seconds per second moves the carrier by -fc * v: the delay is
    the integral of a constant, and the envelope is a constant, so the output is one tone."""
    fs, fc, v = 1e6, 2.4e9, 1e-5                      # 3 km/s away: -24 kHz at 2.4 GHz
    g = Grid(fs=fs, n=1 << 14, fc_hz=fc)
    ramp = Signal().carrier("dc", level=v).filter("integrate", tau_s=1.0)
    y = Signal(seed=1, grid=g).capture(values=np.ones(g.n, complex)).delay(by=ramp).waveform()
    assert np.allclose(np.abs(y[64:-64]), 1.0, atol=1e-9)
    f_meas = np.polyfit(np.arange(g.n)[64:-64] / fs, np.unwrap(np.angle(y[64:-64])), 1)[0] / (2 * np.pi)
    print(f"Doppler: measured {f_meas:.3f} Hz, -fc*v = {-fc * v:.3f} Hz")
    assert abs(f_meas + fc * v) < 1e-3 * fc * v
    # a shift that would leave the grid is refused, naming the shift
    fast = Signal().carrier("dc", level=3e-4).filter("integrate", tau_s=1.0)
    with pytest.raises(ValueError, match=r"moves a carrier at 2.4e\+09 Hz by 7.2e\+05 Hz"):
        Signal(seed=1, grid=g).capture(values=np.ones(g.n, complex)).delay(by=fast).waveform()


def test_a_delay_needs_the_centre_frequency_and_keeps_its_promises_on_an_envelope():
    bare = Grid(fs=FS, baud=BAUD, n=N)                 # an I/Q record with no stated centre
    with pytest.raises(ValueError, match=r"centre frequency was never stated.*Grid\(.*fc_hz"):
        Signal(seed=1, grid=bare).iq("qpsk", prbs=9).delay(by=1e-9).waveform()
    with pytest.raises(ValueError, match="negative delay is an advance"):
        env().delay(by=-1e-9).waveform()
    assert env().delay(by=1e-9).delay_errors()[0]["error"] == JN.describe_delay(FS)["error"]
    # a real record's delay is the samples it always was
    real = Signal(seed=1, grid=Grid(fs=FS, n=N)).carrier("sine", f_hz=37e6)
    assert real.delay(by=0.33e-9).waveform().dtype == np.float64


# ------------------------------------------------------------------ a complex gain
def test_a_complex_gain_scales_and_turns_an_envelope_and_is_data_in_the_recipe():
    x = env().waveform()
    g = 0.5 * np.exp(1j * 1.2)
    s = env().gain(by=g)
    assert s.ops[-1] == {"op": "gain", "by": [g.real, g.imag]}
    y = s.waveform()
    assert np.array_equal(y, x * g)
    assert np.array_equal(Signal.from_recipe(json.loads(s.to_json())).waveform(), y)
    assert s.sha256() != env().gain(by=0.5).sha256()
    assert np.array_equal(env().gain(by=0.5).waveform(), 0.5 * x)      # a real one still is


def test_gain_by_an_iq_sub_chain_is_the_product_of_the_two_envelopes_samples():
    other = Signal().iq("qpsk", prbs=9, prbs_seed=5)
    s = env().gain(by=other)
    y = s.waveform()
    by = s.subchain("gain/1/by")
    assert np.iscomplexobj(by) and np.array_equal(y, env().waveform() * by)
    # a real sub-chain is still a real factor on both rails
    ripple = Signal().carrier("sine", f_hz=1e6, amp=0.1, offset=1.0)
    r = env().gain(by=ripple)
    assert r.subchain("gain/1/by").dtype == np.float64
    assert np.array_equal(r.waveform(), env().waveform() * r.subchain("gain/1/by"))


def test_a_complex_gain_on_a_voltage_is_refused_by_name():
    real = lambda: Signal(seed=1, grid=Grid(fs=FS, baud=BAUD, n=N)).carrier("sine", f_hz=1e6)
    with pytest.raises(ValueError, match="complex and the record is a real voltage"):
        real().gain(by=1j)
    r = real().recipe()
    r["ops"].append({"op": "gain", "by": [0.0, 1.0]})
    with pytest.raises(ValueError, match="complex and the record is a real voltage"):
        Signal.from_recipe(r).waveform()
    with pytest.raises(ValueError, match="COMPLEX"):
        real().gain(by=Signal().iq("qpsk", prbs=9)).waveform()
    with pytest.raises(ValueError, match="not a finite number"):
        env().gain(by=complex(np.nan, 0.0))


# ------------------------------------------------------------------ receiver-side imbalance
def _demodulate(x, gain_db, phase_deg):
    """An imbalanced DEMODULATOR, written out: the I mixer is cos(wt), the Q mixer has gain g
    and is `phase_deg` from quadrature. Independent of `rf.iq_imbalance`."""
    g, phi = 10 ** (gain_db / 20), np.deg2rad(phase_deg)
    wt = 2 * np.pi * FC * np.arange(N) / FS
    r = IQ.upconvert(x, FC, FS)
    h = IQ.lowpass(IQ.lowpass_spec(LPF), FS)
    return (fftconvolve(2 * r * np.cos(wt), h, mode="same")
            + 1j * fftconvolve(-2 * g * r * np.sin(wt + phi), h, mode="same"))


def test_receiver_side_imbalance_is_iq_imbalance_after_the_channel():
    """`iq_imbalance` is stated at the modulator. Placed after the channel it is the
    DEMODULATOR's imbalance: exactly for a gain error, and for a phase error with the sign of
    `phase_deg` reversed and the record's carrier phase referred to a different rail -- a
    fixed turn `a` before the op and back after it, which is three ops and adds no mechanism.
    The image it makes is the same size either way."""
    x = env().waveform()
    # a gain error alone: the same op, as it stands
    ref = _demodulate(x, 0.8, 0.0)
    got = looped().iq_imbalance(gain_db=0.8).waveform()
    e_gain = worst(ref, got)
    # gain and phase: phase_deg reversed, between a turn and its inverse
    gdb, pdeg = 0.8, 5.0
    nu = (1 - 10 ** (gdb / 20) * np.exp(1j * np.deg2rad(pdeg))) / 2
    a = np.exp(-1j * np.angle(nu))
    ref = _demodulate(x, gdb, pdeg)
    got = (looped().gain(by=a).iq_imbalance(gain_db=gdb, phase_deg=-pdeg)
           .gain(by=np.conj(a)).waveform())
    e_both = worst(ref, got)
    plain = looped().iq_imbalance(gain_db=gdb, phase_deg=-pdeg).waveform()
    e_plain = worst(ref, plain)
    print(f"receiver imbalance: gain only {e_gain:.1e}; gain+phase with the turn {e_both:.1e}, "
          f"without it {e_plain:.2e} (the image's phase, not its size)")
    assert e_gain < 1e-9 and e_both < 1e-9
    assert 1e-3 < e_plain < 0.2
    assert ws.rf.image_rejection_db(gdb, -pdeg) == pytest.approx(ws.rf.image_rejection_db(gdb, pdeg))
