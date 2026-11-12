"""GitHub #81: complex baseband records, and `upconvert` / `downconvert` between the domains.

A record used to be a real voltage and nothing else. Radio and coherent signals are defined,
generated and analysed as COMPLEX BASEBAND -- I + jQ about a centre frequency -- and the
library could neither hold one nor hand one to an analysis tool.

The design has four claims, and each is checked here against something that is not the code
under test:

  * THE MIXERS ARE THE TEXTBOOK IDENTITIES. A single complex tone at +f0 up-converts to
    exactly ``cos(2*pi*(fc + f0)*t)`` -- one sideband, the upper one -- which is a closed form
    and also the test that the imaginary part is used with the right sign. And
    ``downconvert(upconvert(x)) == x`` to the error of the STATED low-pass, a bound computed
    from the filter's own measured response rather than from the round trip.
  * THE SOURCE HAS GROUND TRUTH. The pulse is checked against the analytic root-raised-cosine
    spectrum and against its defining property (its autocorrelation is zero at every other
    symbol), and a matched filter returns the symbols `Signal.transmitted()` reports.
  * NOTHING DROPS AN IMAGINARY PART. Every op is real-only unless `wfmsynth.domain.ACCEPTS`
    says otherwise; the refusal names the op, the domain and the op that converts; and an op
    that opts in and then returns a real array is stopped by the composer.
  * A REAL RECORD IS UNTOUCHED. Its recipe has no new key, so its content address is the one
    it had; `stage_kinds()` returns what it returned.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

import wfmsynth as ws
from wfmsynth import AcquisitionProfile, Grid, Signal
from wfmsynth import compose as C
from wfmsynth import domain as DOMAIN
from wfmsynth import iq as IQ

FS, BAUD, FC = 8e9, 25e6, 1e9
N = 1 << 16
G = Grid(fs=FS, baud=BAUD, n=N, domain="iq", fc_hz=FC)
LPF = dict(pass_hz=20e6, stop_hz=60e6)

TONES = ((-19.7e6, 0.30, 0.1), (-7.1e6, 0.20, 1.0), (0.9e6, 0.25, 2.0), (4.3e6, 0.15, -1.0),
         (18.9e6, 0.10, 0.5))


def _tones(n=1 << 15, fs=FS):
    """A complex envelope that is band-limited to the low-pass's passband BY CONSTRUCTION:
    five off-bin tones, on both sides of DC, none at a symmetric frequency."""
    t = np.arange(n) / fs
    return sum(a * np.exp(1j * (2 * np.pi * f * t + p)) for f, a, p in TONES)


def _src(con="qam16", **kw):
    return Signal(seed=1, grid=G).iq(con, prbs=15, prbs_phase=100, **kw)


# ------------------------------------------------------------------ the mixers
def test_a_complex_tone_upconverts_to_the_upper_sideband_and_nothing_else():
    """exp(j*2*pi*f0*t) -> cos(2*pi*(fc + f0)*t), exactly. A mixer that used only I would give
    both sidebands at half amplitude; one with the sign of Q flipped would give fc - f0."""
    n, f0 = 1 << 12, 3.3e6
    t = np.arange(n) / FS
    x = 0.7 * np.exp(1j * (2 * np.pi * f0 * t + 0.4))
    s = Signal(seed=0, grid=Grid(fs=FS, n=n, domain="iq", fc_hz=FC)).capture(values=x).upconvert()
    y = s.waveform()
    assert y.dtype == np.float64
    err = float(np.abs(y - 0.7 * np.cos(2 * np.pi * (FC + f0) * t + 0.4)).max())
    lower = float(np.abs(y - 0.7 * np.cos(2 * np.pi * (FC - f0) * t + 0.4)).max())
    print(f"upper sideband: max |y - closed form| = {err:.2e}; against the LOWER one {lower:.2f}")
    assert err < 1e-11 and lower > 0.5


@pytest.mark.parametrize("atten_db", [60.0, 80.0, 100.0])
def test_downconvert_of_upconvert_is_the_input_within_the_stated_filters_error(atten_db):
    """The bound is not fitted to the round trip. `lowpass_error` measures the taps: the
    passband's worst |H - 1| scales the wanted tones, and the stopband's worst |H| scales
    their images at 2*fc. Both multiply the sum of the tone amplitudes, the worst alignment."""
    x = _tones()
    lpf = dict(LPF, atten_db=atten_db)
    g = Grid(fs=FS, n=len(x), domain="iq", fc_hz=FC)
    y = Signal(seed=0, grid=g).capture(values=x).upconvert().downconvert(fc_hz=FC, lpf=lpf).waveform()
    assert y.dtype == np.complex128 and len(y) == len(x)
    half = len(IQ.lowpass(lpf, FS)) // 2
    pass_err, stop_db = IQ.lowpass_error(lpf, FS)
    bound = (pass_err + 10 ** (stop_db / 20)) * sum(a for _, a, _ in TONES)
    err = float(np.abs(y - x)[half:-half].max())
    print(f"atten_db={atten_db:g}: {2 * half + 1} taps, passband |H-1| <= {pass_err:.2e}, "
          f"stopband {stop_db:.1f} dB; round trip {err:.2e} against a bound of {bound:.2e}")
    assert err < bound
    assert err > 1e-3 * bound, "the bound is so loose it would not notice a broken filter"
    # the stated attenuation is what the taps deliver, to the accuracy of Kaiser's formula
    assert abs(stop_db + atten_db) < 2.0 and pass_err < 2.5 * 10 ** (-atten_db / 20)


def test_the_mixer_phases_are_the_carriers_and_they_cancel():
    """`phase_rad` on the way up rotates the passband carrier; the same phase on the way down
    removes it, and a different one leaves exactly the difference as a rotation of the I/Q."""
    x = _tones()
    g = Grid(fs=FS, n=len(x), domain="iq", fc_hz=FC)
    half = len(IQ.lowpass(LPF, FS)) // 2

    def rt(up, down):
        return (Signal(seed=0, grid=g).capture(values=x).upconvert(phase_rad=up)
                .downconvert(fc_hz=FC, lpf=LPF, phase_rad=down).waveform()[half:-half])

    ref = x[half:-half]
    same = float(np.abs(rt(0.9, 0.9) - ref).max())
    rot = float(np.abs(rt(0.9, 0.2) - ref * np.exp(0.7j)).max())
    print(f"matched phases {same:.2e}; a 0.7 rad mismatch is a 0.7 rad rotation to {rot:.2e}")
    assert same < 3e-4 and rot < 3e-4
    assert float(np.abs(rt(0.9, 0.2) - ref).max()) > 0.1


def test_decimating_downconvert_keeps_every_dth_sample_of_the_same_filter():
    x = _tones()
    g = Grid(fs=FS, n=len(x), domain="iq", fc_hz=FC)
    full = Signal(seed=0, grid=g).capture(values=x).upconvert().downconvert(fc_hz=FC, lpf=LPF)
    dec = Signal(seed=0, grid=g).capture(values=x).upconvert().downconvert(
        fc_hz=FC, lpf=LPF, fs_out=FS / 32)
    y = dec.waveform()
    assert np.array_equal(y, full.waveform()[::32])
    sg = dec.stored_grid(y)
    assert (sg.fs, sg.n, sg.domain, sg.fc_hz) == (FS / 32, len(x) // 32, "iq", FC)


# ------------------------------------------------------------------ the source
@pytest.mark.parametrize("beta", [0.2, 0.35, 0.5, 1.0])
def test_the_pulse_is_the_analytic_root_raised_cosine(beta):
    """Three properties, none of them the formula in `iq.rrc`: unit energy per symbol, an
    autocorrelation that vanishes at every other symbol centre (which is what "root raised
    cosine" MEANS), and the textbook frequency response."""
    over, half = 16, 400
    t = np.arange(-half * over, half * over + 1) / over
    h = IQ.rrc(t, beta)
    energy = float(np.sum(h * h) / over)
    isi = max(abs(float(np.sum(h[:len(h) - k * over] * h[k * over:]) / over)) for k in (1, 2, 3, 7))
    nfft = 1 << 18
    H = np.abs(np.fft.rfft(h, nfft)) / over
    f = np.fft.rfftfreq(nfft, 1.0 / over)
    lo, hi = (1 - beta) / 2, (1 + beta) / 2
    want = np.where(f <= lo, 1.0, np.where(
        f >= hi, 0.0, np.sqrt(0.5 * (1 + np.cos(np.pi / beta * (np.clip(f, lo, hi) - lo))))))
    spec = float(np.abs(H - want).max())
    print(f"beta={beta:g}: energy {energy:.10f}, ISI at other symbols {isi:.1e}, "
          f"|H - analytic| {spec:.1e}")
    assert abs(energy - 1.0) < 1e-8 and isi < 1e-8 and spec < 2e-3


def test_the_pulse_is_continuous_through_its_removable_singularity():
    b = 0.25                                  # the formula is 0/0 at t = 1/(4*beta) = 1.0
    a, at, c = IQ.rrc(np.array([1.0 - 1e-6, 1.0, 1.0 + 1e-6]), b)
    assert abs(at - 0.5 * (a + c)) < 1e-9


@pytest.mark.parametrize("con,points", [("bpsk", 2), ("qpsk", 4), ("qam16", 16), ("qam64", 64)])
def test_a_constellation_has_unit_power_and_is_gray_mapped(con, points):
    k = IQ.CONSTELLATIONS[con]
    words = np.array([[(w >> (k - 1 - j)) & 1 for j in range(k)] for w in range(points)])
    a = IQ.map_bits(con, words.ravel())
    assert len(np.unique(np.round(a, 9))) == points
    assert abs(float(np.mean(np.abs(a) ** 2)) - 1.0) < 1e-12
    dmin = min(abs(a[i] - a[j]) for i in range(points) for j in range(i))
    for i in range(points):                   # nearest neighbours differ in exactly one bit
        for j in range(i):
            if abs(a[i] - a[j]) < dmin * 1.001:
                assert int(np.sum(words[i] != words[j])) == 1


def test_qpsk_puts_a_one_bit_on_the_positive_rail():
    a = IQ.map_bits("qpsk", [0, 0, 0, 1, 1, 0, 1, 1])
    assert np.allclose(a * np.sqrt(2), [-1 - 1j, -1 + 1j, 1 - 1j, 1 + 1j])


@pytest.mark.parametrize("con", ["bpsk", "qpsk", "qam16", "qam64"])
def test_a_matched_filter_returns_the_transmitted_symbols(con):
    """The ground truth is usable: what `transmitted()` reports is what the record holds. The
    residual is the pulse's truncation and nothing else -- see the span test below."""
    s = _src(con)
    x = s.waveform()
    sent, t_s = s.transmitted()
    assert x.dtype == np.complex128 and len(x) == N
    assert len(sent) == int(N * BAUD / FS) and np.allclose(t_s, (np.arange(len(sent)) + 0.5) / BAUD)
    got = IQ.demodulate(x, G)
    e = IQ.evm(got[8:-8], sent[8:-8])
    print(f"{con}: {len(sent)} symbols, EVM {100 * e:.3f} % at span_ui=16")
    assert e < 0.006


def test_the_only_error_is_the_pulses_truncation_and_it_falls_with_the_span():
    out = []
    for span in (8, 16, 32):
        s = _src("qpsk", span_ui=span)
        sent, _ = s.transmitted()
        got = IQ.demodulate(s.waveform(), G, span_ui=span)
        out.append(IQ.evm(got[16:-16], sent[16:-16]))
    print("EVM at span_ui = 8 / 16 / 32: " + " / ".join(f"{100 * e:.3f} %" for e in out))
    assert out[0] > 2 * out[1] > 2 * out[2] and out[2] < 0.002


def test_samples_per_symbol_need_not_be_an_integer():
    g = Grid(fs=FS, baud=FS / 37.3, n=N, domain="iq", fc_hz=FC)
    s = Signal(seed=1, grid=g).iq("qam16", prbs=15, prbs_phase=100)
    sent, _ = s.transmitted()
    e = IQ.evm(IQ.demodulate(s.waveform(), g)[8:-8], sent[8:-8])
    assert len(sent) == int(N / 37.3) and e < 0.006


def test_literal_bits_and_literal_symbols_are_the_ground_truth_they_were_given_as():
    bits = [0, 0, 1, 1, 0, 1, 1, 0] * 8
    s = Signal(seed=1, grid=G).iq("qpsk", bits=bits)
    assert np.array_equal(s.transmitted()[0], IQ.map_bits("qpsk", bits))
    psk8 = np.exp(2j * np.pi * np.arange(40) / 8)             # not a built-in constellation
    s = Signal(seed=1, grid=G).iq(symbols=psk8)
    sent, _ = s.transmitted()
    assert np.array_equal(sent, psk8)
    assert s.recipe()["ops"][0]["symbols"][1] == [float(psk8[1].real), float(psk8[1].imag)]
    got = IQ.demodulate(s.waveform(), G, n_sym=40)
    assert IQ.evm(got[8:-8], sent[8:-8]) < 0.006
    # a burst: the record past the last symbol's pulse is silent
    x = s.waveform()
    assert float(np.abs(x[int((40 + 8.5) * FS / BAUD) + 1:]).max()) == 0.0


@pytest.mark.parametrize("knob,alt", [("beta", 0.2), ("span_ui", 8), ("amp", 0.5),
                                      ("prbs_seed", 5), ("prbs_phase", 7)])
def test_every_source_knob_moves_the_record_and_its_address(knob, alt):
    a, b = _src("qpsk"), Signal(seed=1, grid=G).iq("qpsk", **{"prbs": 15, "prbs_phase": 100,
                                                              knob: alt})
    assert a.sha256() != b.sha256()
    assert float(np.abs(a.waveform() - b.waveform()).max()) > 1e-3


@pytest.mark.parametrize("build,knob,alt", [
    (lambda **k: _src().upconvert(**k), "fc_hz", 1.2e9),
    (lambda **k: _src().upconvert(**k), "phase_rad", 0.5),
    (lambda **k: _src().upconvert().downconvert(**{"fc_hz": FC, "lpf": LPF, **k}), "fc_hz", FC + 1e6),
    (lambda **k: _src().upconvert().downconvert(**{"fc_hz": FC, "lpf": LPF, **k}), "phase_rad", 0.5),
    (lambda **k: _src().upconvert().downconvert(**{"fc_hz": FC, "lpf": LPF, **k}), "lpf", 5e6),
])
def test_every_mixer_knob_moves_the_record_and_its_address(build, knob, alt):
    a, b = build(), build(**{knob: alt})
    assert a.sha256() != b.sha256()
    assert float(np.abs(a.waveform() - b.waveform()).max()) > 1e-3


# ------------------------------------------------------------------ an instrument sampling RF
def test_a_scope_sampling_rf_then_downconverting_returns_the_symbols():
    """The reason for `upconvert`: after it the record is a voltage, so the existing front
    end, sampler and converter apply -- here an 8-bit acquisition at a QUARTER of the
    synthesis rate. `downconvert` then has to work at the rate the record was STORED at; with
    the synthesis rate it would mix by the wrong frequency and return noise."""
    g = Grid(fs=16e9, baud=BAUD, n=1 << 18, domain="iq", fc_hz=FC)
    prof = AcquisitionProfile(sample_rate_hz=4e9, record_length=1 << 16,
                              input_bandwidth_hz=3e9, bits=8)
    s = (Signal(seed=1, grid=g).iq("qam16", prbs=15, prbs_phase=100).upconvert().acquire(prof)
         .downconvert(fc_hz=FC, lpf=LPF, fs_out=250e6))
    y = s.waveform()
    sg = s.stored_grid(y)
    assert (sg.fs, sg.n, sg.domain, sg.fc_hz) == (250e6, 4096, "iq", FC)
    sent, _ = s.transmitted()
    got = IQ.demodulate(y, sg)
    raw, fit = IQ.evm(got[16:-16], sent[16:-16]), IQ.evm(got[16:-16], sent[16:-16], fit=True)
    gain = IQ.complex_gain(got[16:-16], sent[16:-16])
    print(f"8-bit, 4 GSa/s acquisition of a 1 GHz carrier: EVM {100 * fit:.2f} % once the "
          f"front end's carrier phase ({np.angle(gain):+.3f} rad, gain {abs(gain):.4f}) is "
          f"taken out; {100 * raw:.1f} % before")
    assert fit < 0.01 < raw
    assert s.stage_kinds(domains=True) == [("source", "iq"), ("shape", "real"),
                                           ("instrument", "real"), ("instrument", "iq")]


# ------------------------------------------------------------------ the domain
def test_waveform_is_complex_and_the_recipe_replays_it_bit_for_bit():
    s = _src().upconvert().probe(bw_hz=3e9).downconvert(fc_hz=FC, lpf=LPF)
    x = s.waveform()
    r = json.loads(s.to_json())
    assert r["grid"]["domain"] == "iq" and r["grid"]["fc_hz"] == FC
    t = Signal.from_recipe(r)
    y = t.waveform()
    assert x.dtype == y.dtype == np.complex128 and np.array_equal(x, y)
    assert t.sha256() == s.sha256() and t.grid == G


def test_the_domain_and_the_centre_frequency_are_part_of_the_content_address():
    def addr(**kw):
        return Signal(seed=1, grid=Grid(fs=FS, baud=BAUD, n=N, **kw)).capture(
            values=[0.0] * N).sha256()
    assert len({addr(), addr(domain="iq"), addr(domain="iq", fc_hz=1e9),
                addr(domain="iq", fc_hz=2e9)}) == 4


def test_a_real_record_is_untouched():
    """No new key in a real record's recipe, so its sha256 is the one it had; `stage_kinds()`
    returns the list it returned. (Waveform and recipe digests of fourteen real chains were
    compared against branch `main` when this landed: identical.)"""
    g = Grid(fs=80e9, baud=10e9, n=4096)
    s = Signal(seed=1, grid=g).carrier("nrz", n_ui=512).lossy(loss_db=6, loss_at_ghz=5).probe(
        bw_hz=20e9)
    r = s.recipe()
    assert r["grid"] == {"fs": 80e9, "baud": 10e9, "n": 4096, "v_full": 1.0}
    assert s.stage_kinds() == ["source", "channel", "probe"]
    assert s.stage_kinds(domains=True) == [("source", "real"), ("channel", "real"),
                                           ("probe", "real")]
    x = s.waveform()
    assert x.dtype == np.float64 and s.stored_grid(x) == g
    assert g == Grid(80e9, 10e9, 4096) and (g.domain, g.fc_hz) == ("real", None)


@pytest.mark.parametrize("op,kw", [
    ("lossy", dict(loss_db=3.0, loss_at_ghz=1.0)), ("probe", dict(bw_hz=1e9)),
    ("digitize", dict(bits=8)), ("rx_noise", dict(rms=0.01)), ("scope", dict(bw_hz=1e9)),
    ("ac_couple", dict(fc_hz=1e5)), ("events", dict(kind="glitch", on="times", times=[1e-6])),
    ("eo", {}), ("modulate", dict(message=dict(f_hz=1e6))), ("carrier", dict(kind="nrz")),
])
def test_a_real_only_op_refuses_an_iq_record_naming_the_domain_and_the_converter(op, kw):
    s = _src()
    with pytest.raises(ValueError) as e:
        getattr(s, op)(**kw)
    msg = str(e.value)
    assert msg.startswith(op) and "'iq'" in msg and "'real'" in msg and "upconvert(" in msg
    assert len(s.ops) == 1, "a refused op must not be left in the chain"


def test_every_op_is_real_only_unless_it_is_listed():
    """The default that makes the design safe, as an inventory: this fails when an op is
    opted in, so the opting-in is a decision somebody wrote down next to a test of it."""
    both = {op for op in C._EXEC if "iq" in DOMAIN.accepts(op)}
    assert both == {"capture", "iq", "upconvert", "filter",
                    # defined on an envelope and nowhere else; tests/test_iq_impairments.py
                    "iq_imbalance", "iq_offset", "freq_offset", "phase_noise", "awgn", "spur",
                    "dac",
                    # the general stages that keep their meaning on an envelope, and a named
                    # chain of a several-channel record; each way in tests/test_iq_seams.py
                    "gain", "gate", "combine", "sequence", "switch", "use"}
    assert set(DOMAIN.ACCEPTS) | set(DOMAIN.EMITS) <= set(C._EXEC)


# ------------------------------------------------------------------ `filter` on an envelope
def test_filter_rrc_on_an_iq_record_is_the_matched_filter():
    """The `iq` source shapes with a root-raised-cosine and `filter("rrc")` is the same pulse,
    so the pair is a raised cosine: sampled at the symbol centres the record IS the sent
    symbols, with no gain or phase fitted. The source alone is not (an RRC has ISI), and a
    filter with another roll-off is not matched."""
    sent, t_s = _src().transmitted()
    k = np.rint(t_s * FS).astype(int)
    assert np.abs(t_s * FS - k).max() < 1e-6            # 320 samples/symbol: centres on samples
    core = slice(10, len(sent) - 10)                    # the ends lack their neighbours' tails

    def err(sig, delay=0, fit=False):
        return IQ.evm(sig.waveform()[k[core] + delay], sent[core], fit=fit)

    matched = _src().filter("rrc", beta=0.35, zero_phase=True)
    assert matched.waveform().dtype == np.complex128
    assert matched.stage_kinds(domains=True)[-1] == ("general", "iq")
    assert err(matched) < 0.006                         # 0.36 % measured, nothing fitted
    assert err(_src(), fit=True) > 0.10                 # 13.7 %: the source alone has ISI
    assert err(_src().filter("rrc", beta=0.6, zero_phase=True), fit=True) > 2 * err(matched)
    # the causal form is the same record late by exactly the delay `filter_delays()` reports
    causal = _src().filter("rrc", beta=0.35)
    late = causal.filter_delays()[0]["group_delay_s"] * FS
    assert late == pytest.approx(round(late), abs=1e-6) and late > 0
    assert err(causal, delay=int(round(late))) < 0.006
    # and the independent matched-filter measure agrees symbol by symbol
    ref = IQ.demodulate(_src().waveform(), G)[core]
    assert np.abs(matched.waveform()[k[core]] - ref).max() < 0.005      # 1.8e-3 measured


def test_a_filter_on_an_envelope_acts_about_the_centre_frequency():
    """A complex tone at +f0 and one at -f0 leave a Butterworth low-pass with the SAME constant
    magnitude, the design's |H(f0)|: the response is about DC of the envelope, which is the
    carrier. Filtering only I (or I and Q with different responses) would ripple."""
    f0, fc, order = 30e6, 20e6, 4
    want = 1.0 / np.sqrt(1.0 + (f0 / fc) ** (2 * order))
    t = np.arange(N) / FS
    for sign in (+1, -1):
        y = (Signal(seed=1, grid=G).capture(values=np.exp(sign * 2j * np.pi * f0 * t))
             .filter("butter", f_hz=fc, order=order).waveform())
        mag = np.abs(y[N // 2:])                        # past the turn-on
        assert mag.mean() == pytest.approx(want, rel=1e-3)
        assert np.ptp(mag) < 1e-6 * want


def test_the_converters_refuse_the_domain_they_do_not_convert():
    with pytest.raises(ValueError, match=r"upconvert.*'real'.*downconvert\("):
        Signal(seed=1, grid=Grid(fs=FS, n=N)).carrier("sine", f_hz=FC).upconvert(fc_hz=FC)
    with pytest.raises(ValueError, match=r"downconvert.*'iq'.*upconvert\("):
        _src().downconvert(fc_hz=FC, lpf=LPF)
    with pytest.raises(ValueError, match=r"iq.*STATES domain='real'.*Leave domain= off"):
        Signal(seed=1, grid=Grid(fs=FS, baud=BAUD, n=N, domain="real")).iq("qpsk", prbs=9)


def test_the_refusal_also_guards_a_recipe_and_a_hand_edited_op_list():
    r = _src().recipe()
    r["ops"].append({"op": "lossy", "loss_db": 3.0, "loss_at_ghz": 1.0})
    with pytest.raises(ValueError, match="lossy.*'iq'"):
        Signal.from_recipe(r).waveform()


def test_an_op_that_opts_in_and_then_drops_the_imaginary_part_is_stopped(monkeypatch):
    """Opting in is one line, and the composer holds the op to it: a real array coming back
    for an I/Q record is the imaginary part gone."""
    monkeypatch.setitem(DOMAIN.ACCEPTS, "drift", DOMAIN.BOTH)          # the one line
    monkeypatch.setitem(C._EXEC, "drift", lambda x, p, st, g, i: np.real(x))
    s = _src().drift(amount=0.1)
    assert s.stage_kinds(domains=True)[-1] == ("supply", "iq")
    with pytest.raises(ValueError, match="drift.*dropped the imaginary part"):
        s.waveform()
    monkeypatch.setitem(C._EXEC, "drift", lambda x, p, st, g, i: 2 * x)  # an honest one runs
    assert np.array_equal(s.waveform(), 2 * _src().waveform())


def test_a_grid_states_its_domain_or_is_refused():
    with pytest.raises(ValueError, match="unknown domain"):
        Grid(fs=1e9, domain="complex")
    with pytest.raises(ValueError, match="fc_hz.*domain='iq'"):
        Grid(fs=1e9, fc_hz=1e8, domain="real")
    assert Grid(fs=1e9, fc_hz=1e8).domain == "iq"          # only an envelope has a centre
    assert Grid(fs=1e9, domain="iq").fc_hz is None                     # a centre is optional
    with pytest.raises(ValueError, match="upconvert needs the carrier"):
        Signal(seed=0, grid=Grid(fs=1e9, n=64, domain="iq")).capture(
            values=np.ones(64, complex)).upconvert().waveform()


# ------------------------------------------------------------------ what the mixers refuse
def test_upconvert_refuses_a_grid_too_slow_for_the_carrier_or_the_envelope():
    with pytest.raises(ValueError, match="not fast enough for the carrier.*does not resample"):
        _src().upconvert(fc_hz=4.5e9).waveform()
    # the carrier fits; the envelope does not -- 12 MHz of carrier under a +/-17 MHz envelope
    with pytest.raises(ValueError, match=r"dB of the envelope's power.*fold_db=-40"):
        _src().upconvert(fc_hz=12e6).waveform()
    low = IQ.folded_db(_src().waveform(), 12e6, FS)
    ok = IQ.folded_db(_src().waveform(), FC, FS)
    print(f"folded power: {low:.1f} dB at fc = 12 MHz, {ok:.1f} dB at 1 GHz (limit -40)")
    assert low > -20 and ok < -60
    y = _src().upconvert(fc_hz=12e6, fold_db=-5.0).waveform()           # stated, so accepted
    assert y.dtype == np.float64


def test_downconvert_refuses_a_filter_it_cannot_honour():
    up = lambda: _src().upconvert()                                     # noqa: E731
    with pytest.raises(ValueError, match="low-pass STATED"):
        up().downconvert(fc_hz=FC).waveform()
    with pytest.raises(ValueError, match="image.*would come through"):
        up().downconvert(fc_hz=FC, lpf=dict(pass_hz=20e6, stop_hz=1.99e9)).waveform()
    with pytest.raises(ValueError, match="not fs=.*divided by a whole number"):
        up().downconvert(fc_hz=FC, lpf=LPF, fs_out=3e9).waveform()
    with pytest.raises(ValueError, match="above the output Nyquist.*would alias"):
        up().downconvert(fc_hz=FC, lpf=LPF, fs_out=100e6).waveform()
    with pytest.raises(ValueError, match="taps.*and the record has"):
        up().downconvert(fc_hz=FC, lpf=dict(pass_hz=20e6, stop_hz=20.2e6)).waveform()
    with pytest.raises(ValueError, match="does not take .*'cutoff_hz'"):
        up().downconvert(fc_hz=FC, lpf=dict(cutoff_hz=20e6)).waveform()
    with pytest.raises(ValueError, match="not below Nyquist"):
        up().downconvert(fc_hz=5e9, lpf=LPF).waveform()
    # the carrier the grid states is the one a round trip comes back about, and is recorded
    back = up().downconvert(lpf=LPF)
    assert back.ops[-1]["fc_hz"] == FC and back.stored_grid(back.waveform()).fc_hz == FC
    assert np.array_equal(back.waveform(), up().downconvert(fc_hz=FC, lpf=LPF).waveform())
    with pytest.raises(ValueError, match="downconvert needs fc_hz.*states no centre"):
        Signal(seed=1, grid=Grid(fs=FS, n=1 << 12)).carrier("sine", f_hz=FC).downconvert(lpf=LPF)
    with pytest.raises(ValueError, match="low-pass STATED"):
        up().downconvert()                           # refused when the op is written


def test_the_source_refuses_what_it_cannot_render_honestly():
    with pytest.raises(ValueError, match="would alias"):                # 1.2 samples/symbol
        Signal(seed=1, grid=Grid(fs=30e6, baud=BAUD, n=4096, domain="iq")).iq(
            "qpsk", prbs=9).waveform()
    with pytest.raises(ValueError, match="do not fit.*nothing is truncated"):
        Signal(seed=1, grid=G).iq("qpsk", bits=[0, 1] * 300).waveform()
    with pytest.raises(ValueError, match="not a whole number of qam16 symbols"):
        Signal(seed=1, grid=G).iq("qam16", bits=[0, 1, 1]).waveform()
    with pytest.raises(ValueError, match="unknown constellation '8psk'.*symbols="):
        Signal(seed=1, grid=G).iq("8psk", prbs=9)
    with pytest.raises(ValueError, match="needs its symbols"):
        Signal(seed=1, grid=G).iq("qpsk").waveform()
    with pytest.raises(ValueError, match="not both"):
        Signal(seed=1, grid=G).iq("qpsk", symbols=[1 + 0j])
    with pytest.raises(ValueError, match="unknown pulse 'rect'"):
        Signal(seed=1, grid=G).iq("qpsk", prbs=9, pulse="rect").waveform()
    with pytest.raises(ValueError, match=r"Grid\(fs=..., baud=..."):
        Signal(seed=1, grid=Grid(fs=FS, n=N, domain="iq")).iq("qpsk", prbs=9).waveform()
    with pytest.raises(TypeError, match="does not take 'rolloff'"):
        Signal(seed=1, grid=G).iq("qpsk", prbs=9, rolloff=0.3)


def test_transmitted_is_refused_where_there_is_nothing_to_report():
    s = Signal(seed=1, grid=Grid(fs=80e9, baud=10e9, n=4096)).carrier("nrz")
    with pytest.raises(ValueError, match="starts with `carrier`.*carrier_symbols"):
        s.transmitted()


def test_a_lead_in_is_refused_rather_than_moving_the_carrier_phase():
    for op in ("iq", "upconvert", "downconvert"):
        assert op in C.OP_KIND
        assert ((op in C._LEAD_LTI) + (op in C._LEAD_SKIP) + (op in C._LEAD_REJECT)
                + (op in C._LEAD_ANALYTIC)) == 1 and op in C._LEAD_REJECT
    g = Grid(fs=FS, baud=FS / 16, n=1 << 14)
    s = Signal(seed=1, grid=g, lead_in=True).carrier("nrz", n_ui=1 << 10).downconvert(
        fc_hz=FC, lpf=LPF)
    with pytest.raises(ValueError, match="downconvert.*local oscillator's phase"):
        s.waveform()


def test_the_voltage_exporter_refuses_an_iq_record_rather_than_writing_its_real_part():
    with pytest.raises(ValueError, match=r"complex baseband.*x.real and x.imag"):
        ws.hdf5.counts_for(_tones(64))


# ------------------------------------------------------------------ capture, in I/Q
def test_capture_starts_an_iq_chain_from_complex_values_and_replays_them():
    x = _tones(256)
    g = Grid(fs=FS, n=256, domain="iq", fc_hz=FC)
    s = Signal(seed=0, grid=g).capture(values=x)
    assert s.recipe()["ops"][0]["values"][3] == [float(x[3].real), float(x[3].imag)]
    y = Signal.from_recipe(json.loads(s.to_json())).waveform()
    assert y.dtype == np.complex128 and np.array_equal(y, x)
    # real samples are a legitimate I/Q record with Q = 0, and say so in the dtype
    z = Signal(seed=0, grid=g).capture(values=list(x.real)).waveform()
    assert z.dtype == np.complex128 and np.array_equal(z, x.real + 0j)


def test_complex_samples_offered_to_a_real_record_are_refused_not_cast(tmp_path):
    x = _tones(256)
    g = Grid(fs=FS, n=256, domain="real")
    with pytest.raises(ValueError, match=r"complex \(I/Q\) samples.*domain='iq'"):
        Signal(seed=0, grid=g).capture(values=x)
    with pytest.raises(ValueError, match=r"complex \(I/Q\) samples.*it has no Grid"):
        Signal(seed=0).capture(values=x)
    np.save(tmp_path / "iq.npy", x)
    with pytest.raises(ValueError, match=r"iq.npy.*complex \(I/Q\) samples"):
        Signal(seed=0, grid=g).capture(path=str(tmp_path / "iq.npy"))
    r = Signal(seed=0, grid=Grid(fs=FS, n=256, domain="iq")).capture(values=x).recipe()
    r["grid"]["domain"] = "real"                           # an I/Q recipe relabelled as real
    del r["grid"]["fc_hz"]
    with pytest.raises(ValueError, match=r"complex \(I/Q\) samples"):
        Signal.from_recipe(r).waveform()


def test_a_complex_capture_file_is_read_digested_and_guarded(tmp_path):
    x = _tones(256)
    g = Grid(fs=FS, n=256, domain="iq", fc_hz=FC)
    np.save(tmp_path / "a.npy", x)
    np.savez(tmp_path / "a.npz", values=x)
    np.savetxt(tmp_path / "a.csv", np.c_[np.arange(256) / FS, x.real, x.imag], delimiter=",")
    for name in ("a.npy", "a.npz", "a.csv"):
        s = Signal(seed=0, grid=g).capture(path=str(tmp_path / name))
        assert np.allclose(s.waveform(), x, rtol=0, atol=1e-15), name
    by_path = Signal(seed=0, grid=g).capture(path=str(tmp_path / "a.npy"))
    embedded = Signal(seed=0, grid=g).capture(path=str(tmp_path / "a.npy"), embed=True)
    assert np.array_equal(by_path.waveform(), embedded.waveform())
    assert by_path.ops[0]["sha256"] == embedded.ops[0]["sha256"] == ws.capture.digest(x)
    # the digest covers Q: the same I with a different Q is a different capture
    assert ws.capture.digest(x) != ws.capture.digest(x.real + 0j) != ws.capture.digest(x.real)
    np.save(tmp_path / "a.npy", x.conj())
    with pytest.raises(ValueError, match="does not match the recipe"):
        by_path.waveform()
    np.savetxt(tmp_path / "two.csv", np.c_[np.arange(4), np.arange(4)], delimiter=",")
    with pytest.raises(ValueError, match="three -- time, I, Q"):
        Signal(seed=0, grid=Grid(fs=FS, n=4, domain="iq")).capture(path=str(tmp_path / "two.csv"))
