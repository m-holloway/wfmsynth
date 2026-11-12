"""`lossy` and `sparam` on an I/Q record: the baseband equivalent, H(fc + f) (GitHub #89).

THE REFERENCE is the passband route: `upconvert` -> the op on the real record -> `downconvert`.
The envelope route is handed the envelope a passband record holds (through both mixers, with
nothing between), so the mixers' low-pass is common to both routes and what is compared is the
channel. The passband route runs at 16 GSa/s and decimates to 8 GSa/s; the envelope route then
runs at 8 GSa/s about the 3 GHz the `downconvert` stated -- so `baseband.passband_top` is
8 GHz, the passband record's own Nyquist, and a causal `lossy` builds its minimum phase over
the same band on both routes.

THE BOUND. Interior samples agree to 3e-4 of the record's peak (measured: 0.7e-4 to 1.0e-4).
What is left is the real path's own discretisation -- its response is sampled on a transform
whose length follows the record's (`physics.apply_transfer` states 7.7e-5 for that alone).
The first and last 400 samples see the mixers' low-pass at the record's edge and are excluded.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pytest

import wfmsynth as ws
from wfmsynth import baseband as BB
from wfmsynth import physics as P
from wfmsynth.compose import Signal
from wfmsynth.grid import Grid

F, FC, BAUD, N = 16e9, 3e9, 250e6, 1 << 15
G = Grid(fs=F, baud=BAUD, n=N, fc_hz=FC)
LPF = dict(pass_hz=1.0e9, stop_hz=2.5e9, atten_db=100)
EDGE, BOUND = 400, 3e-4


def passband():
    return Signal(seed=1, grid=G).iq("qam16", prbs=11).upconvert()


def down(sig):
    return sig.downconvert(fc_hz=FC, lpf=LPF, fs_out_hz=F / 2)


def agree(op, **kw):
    """``(worst difference, how much the op changed the record)``, both against the peak."""
    ref = down(getattr(passband(), op)(**kw)).waveform()
    got = getattr(down(passband()), op)(**kw).waveform()
    untouched = down(passband()).waveform()
    assert np.iscomplexobj(got) and got.shape == ref.shape
    peak = np.abs(ref).max()
    return (float(np.abs(ref - got)[EDGE:-EDGE].max() / peak),
            float(np.abs(got - untouched)[EDGE:-EDGE].max() / np.abs(untouched).max()))


@pytest.mark.parametrize("kw", [
    dict(loss_db=12.0, loss_at_ghz=3.0),                        # zero-phase: a magnitude
    dict(loss_db=12.0, loss_at_ghz=3.0, causal=True),           # minimum phase
    dict(length_in=10.0, tand=0.02, causal=True),               # the per-inch law
    dict(trend=(-2.0, -1.5, -0.2), causal=True),                # a fitted |S21| curve
], ids=["anchored", "anchored-causal", "per-inch-causal", "trend-causal"])
def test_lossy_on_an_envelope_is_the_passband_route(kw):
    err, effect = agree("lossy", **kw)
    print(f"lossy({kw}): envelope route against upconvert -> lossy -> downconvert, worst "
          f"{err:.2e} of peak (bound {BOUND:.0e}); the op itself moved the record by {effect:.2f}")
    assert err < BOUND
    assert effect > 0.5              # ... and it is a real channel that was compared


def test_the_magnitude_is_the_loss_law_at_fc_plus_f_on_a_grid_that_cannot_hold_the_carrier():
    """2.4 GHz on a 20 MSa/s record. A tone 3 MHz above the centre comes out at the law's
    loss at 2.403 GHz: the stated 6 dB at 2.4 GHz, exactly, at the centre itself."""
    g = Grid(fs=20e6, n=4000, fc_hz=2.4e9)
    kw = dict(loss_db=6.0, loss_at_ghz=2.4, linear=False)     # circular: a tone is an eigenvector

    def tone(f_hz, **more):
        s = Signal(seed=1, grid=g).capture(values=np.ones(g.n, complex))
        return (s.freq_offset(f_hz=f_hz) if f_hz else s).lossy(**kw, **more).waveform()

    centre = np.abs(tone(0.0))
    assert np.allclose(centre, 10 ** (-6.0 / 20), rtol=1e-12)
    for causal in (False, True):
        got = np.abs(tone(3e6, causal=causal))
        want = 10 ** (-P.insertion_loss_db(2.403, loss_db=6.0, loss_at_ghz=2.4) / 20)
        print(f"|H| at fc + 3 MHz, causal={causal}: {got.mean():.9f}, the law {want:.9f}")
        assert np.allclose(got, want, rtol=1e-6 if causal else 1e-12)
    # the same op read about DC -- what applying it to the samples as they stand would be --
    # loses 0.2 dB at 3 MHz, not 6: a different channel
    about_dc = 10 ** (-P.insertion_loss_db(0.003, loss_db=6.0, loss_at_ghz=2.4) / 20)
    assert about_dc > 0.97 and abs(about_dc - want) > 0.4


def test_a_causal_envelope_response_is_the_passband_minimum_phase_read_at_fc_plus_f():
    """Bin for bin: with 3 GHz on a bin of the 8 GHz axis, the envelope's response at offset
    f is the passband minimum-phase response at 3 GHz + f, and its conjugate below 0 Hz."""
    fs, fc, n = 8e9, 3e9, 1 << 14
    f = np.fft.fftfreq(n, 1 / fs)
    law = dict(loss_db=12.0, loss_at_ghz=3.0)
    H = BB.lossy_response(f, fs, fc, causal=True, **law)
    f_pass = np.fft.rfftfreq(2 * n, 1 / (2 * fs))                 # 0 .. 8 GHz, the same spacing
    Hp = P._min_phase_H(10 ** (-P.insertion_loss_db(f_pass / 1e9, **law) / 20), 2 * n, half=True)
    k = np.rint((fc + f) / (fs / n)).astype(int)                  # -1 GHz .. 7 GHz, in bins
    want = np.where(k < 0, np.conj(Hp[np.abs(k)]), Hp[np.abs(k)])
    err = float(np.abs(H - want).max())
    print(f"causal envelope response against the passband minimum phase: worst {err:.1e}")
    assert err < 1e-9
    zero = BB.lossy_response(f, fs, fc, causal=False, **law)
    assert np.allclose(np.abs(H), np.abs(zero), rtol=1e-6)        # the same magnitude
    assert np.all(zero.imag == 0) and np.abs(H.imag).max() > 0.1  # ... with a phase
    assert BB.passband_top(8e9, 3e9) == 8e9 and BB.passband_top(20e6, 2.4e9) == 2.41e9


def test_the_causal_phase_far_above_the_sample_rate_is_a_stated_choice_with_a_stated_cost():
    """2.4 GHz on 20 MSa/s: the envelope's band is at the very top of the axis the minimum
    phase is built on. Against the same law's minimum phase on an axis 64 times wider (its
    cepstrum written out here) the response differs by a constant phase and a bulk delay --
    the group delay is a small ADVANCE -- and by little else."""
    fs, fc, law = 20e6, 2.4e9, dict(loss_db=20.0, loss_at_ghz=2.4)
    f = np.linspace(-0.4 * fs, 0.4 * fs, 801)
    got = np.unwrap(np.angle(BB.lossy_response(f, fs, fc, causal=True, **law)))
    n, top = 1 << 23, 64 * BB.passband_top(fs, fc)
    fa = np.fft.rfftfreq(n) * 2 * top
    la = np.log(10 ** (-P.insertion_loss_db(fa / 1e9, **law) / 20))
    c = np.fft.ifft(np.concatenate([la, la[-2:0:-1]])).real
    w = np.zeros(n)
    w[0], w[1:n // 2], w[n // 2] = 1, 2, 1
    wide = np.interp(fc + f, fa, np.fft.fft(c * w).imag[:n // 2 + 1])
    delay = lambda p: float(np.mean(-np.gradient(p, f)) / (2 * np.pi))       # noqa: E731
    d = got - wide
    rest = float(np.abs(d - np.polyval(np.polyfit(f, d, 1), f)).max())
    print(f"phase at fc {np.interp(0, f, got):+.3f} rad (wide axis {np.interp(0, f, wide):+.3f}); "
          f"group delay {delay(got) * 1e9:+.3f} ns (wide {delay(wide) * 1e9:+.3f}); apart from "
          f"a constant and a delay they differ by {rest:.1e} rad")
    assert abs(np.interp(0, f, got) + 0.025) < 0.005 and abs(np.interp(0, f, wide) + 3.50) < 0.02
    assert abs(delay(got) * 1e9 + 0.35) < 0.02 and abs(delay(wide) * 1e9 - 0.15) < 0.02
    assert rest < 2e-3                                           # measured 1.2e-3


# ------------------------------------------------------------------ sparam
FREQS = np.linspace(1.5e9, 4.5e9, 601)
S21 = 0.8 * np.exp(-2j * np.pi * FREQS * 0.4e-9) * (1 + 0.3 * np.exp(-2j * np.pi * FREQS * 1.3e-9))


def test_sparam_on_an_envelope_is_the_passband_route():
    """A measured transfer with a 1.3 ns echo in it: ripple every 770 MHz across the band."""
    with pytest.warns(RuntimeWarning, match="band='zero'"):
        err, effect = agree("sparam", freqs=FREQS.tolist(), s21=S21.tolist(), band="zero")
    print(f"sparam: envelope route against the passband route, worst {err:.2e} of peak; "
          f"the op itself moved the record by {effect:.2f}")
    assert err < BOUND and effect > 0.5


def test_sparam_reads_the_file_at_fc_plus_f_and_refuses_a_file_that_is_not_there(tmp_path):
    g = Grid(fs=400e6, baud=25e6, n=1 << 12, fc_hz=3e9)
    src = lambda grid=g: Signal(seed=1, grid=grid).iq("qpsk", prbs=9)
    y = src().sparam(freqs=FREQS.tolist(), s21=S21.tolist()).waveform()
    # one tone: the interpolated point at fc + f, as a complex number
    x = src().waveform()
    k = np.fft.fftfreq(g.n, 1 / g.fs)
    Hk = np.interp(3e9 + k, FREQS, S21.real) + 1j * np.interp(3e9 + k, FREQS, S21.imag)
    circ = src().sparam(freqs=FREQS.tolist(), s21=S21.tolist(), linear=False).waveform()
    assert np.allclose(circ, np.fft.ifft(np.fft.fft(x) * Hk), atol=1e-12)
    # a Touchstone file is the same transfer
    S = np.zeros((len(FREQS), 2, 2), complex)
    S[:, 1, 0] = S[:, 0, 1] = S21
    path = os.path.join(str(tmp_path), "line.s2p")
    ws.write_touchstone(path, FREQS, S)
    assert np.allclose(src().sparam(path=path).waveform(), y, atol=1e-9)
    # a file that stops below the signal, and a record with no centre, are refused by name
    with pytest.raises(ValueError, match="lies above the measured band.*band-limit the envelope"):
        src(Grid(fs=400e6, baud=25e6, n=1 << 12, fc_hz=4.6e9)).sparam(
            freqs=FREQS.tolist(), s21=S21.tolist()).waveform()
    with pytest.raises(ValueError, match=r"sparam.*centre frequency was never stated"):
        src(Grid(fs=400e6, baud=25e6, n=1 << 12)).sparam(
            freqs=FREQS.tolist(), s21=S21.tolist()).waveform()


# ------------------------------------------------------------------ the recipe, and refusals
def test_an_envelope_channel_replays_and_every_parameter_is_in_its_digest():
    g = Grid(fs=40e6, baud=10e6, n=1 << 12, fc_hz=2.4e9)
    base = dict(loss_db=20.0, loss_at_ghz=2.4, causal=True)
    s = Signal(seed=1, grid=g).iq("qam16", prbs=9).lossy(**base)
    y = s.waveform()
    assert np.array_equal(Signal.from_recipe(json.loads(s.to_json())).waveform(), y)
    for k, v in dict(loss_db=21.0, loss_at_ghz=2.5, causal=False).items():
        other = Signal(seed=1, grid=g).iq("qam16", prbs=9).lossy(**{**base, k: v})
        assert other.sha256() != s.sha256() and not np.array_equal(other.waveform(), y)
    # the centre is part of the record: another fc_hz is another channel
    g2 = Grid(fs=40e6, baud=10e6, n=1 << 12, fc_hz=5.8e9)
    moved = Signal(seed=1, grid=g2).iq("qam16", prbs=9).lossy(**base)
    assert moved.sha256() != s.sha256() and not np.array_equal(moved.waveform(), y)
    assert s.stage_kinds(domains=True) == [("source", "iq"), ("channel", "iq")]


def test_what_the_envelope_path_cannot_do_is_refused_by_name():
    src = lambda grid: Signal(seed=1, grid=grid).iq("qpsk", prbs=9)
    g = Grid(fs=40e6, baud=10e6, n=1 << 12, fc_hz=2.4e9)
    with pytest.raises(ValueError, match="lossy: method='overlap' on an I/Q record"):
        src(g).lossy(loss_db=3.0, loss_at_ghz=2.4, method="overlap").waveform()
    with pytest.raises(ValueError, match=r"lossy.*centre frequency was never stated.*fc_hz"):
        src(Grid(fs=40e6, baud=10e6, n=1 << 12)).lossy(loss_db=3.0, loss_at_ghz=2.4).waveform()
    # the channel ops built on a delay or a second signal are not opted in
    for op, kw in (("reflect", dict(td_ps=100.0, gamma_s=0.2, gamma_l=0.2)),
                   ("cascade", dict(path=[{"line": {"length_in": 2.0}}])),
                   ("crosstalk", dict(coupling=0.1))):
        with pytest.raises(ValueError, match=f"{op}.*complex baseband"):
            getattr(src(g), op)(**kw)


def test_a_shared_line_serves_an_iq_channel_and_a_voltage_channel_of_one_record():
    g = Grid(fs=F, baud=BAUD, n=1 << 13, fc_hz=FC)
    line = dict(loss_db=6.0, loss_at_ghz=3.0)
    rec = ws.Channels(
        {"rf": Signal().iq("qpsk", prbs=9).use("line"),
         "scope": Signal().iq("qpsk", prbs=9).upconvert().use("line")},
        shared={"line": Signal().lossy(**line)}, seed=1, grid=g)
    x = rec.waveforms()
    assert np.iscomplexobj(x["rf"]) and x["scope"].dtype == np.float64
    alone = Signal(seed=1, grid=g).iq("qpsk", prbs=9).lossy(**line).waveform()
    assert np.array_equal(x["rf"], alone)
