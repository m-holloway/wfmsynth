"""An amplifier on an I/Q record: AM/AM and AM/PM as laws on the envelope's magnitude (#89).

THE QUESTION THE ISSUE ASKED: does a memoryless law on the passband record, followed by
`downconvert`, already give AM/PM? No, and the last test measures it: `law("tanh")` between
the mixers turns the carrier's phase by under 10 microradians at every drive level (measured:
3e-8 rad), from the linear region to 28 dB into saturation. A memoryless law's output follows its input's sign,
so the fundamental it returns is in phase with the one it was given. AM/PM is a separate
mechanism (the device's delay moving with drive), and `law("saleh")` is where it lives.

What the passband route DOES give is AM/AM, and the envelope law `rapp` reproduces it: the
same test states to what accuracy.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

import wfmsynth as ws
from wfmsynth import stages as ST
from wfmsynth.compose import Signal
from wfmsynth.grid import Grid

SALEH = dict(alpha_a=2.1587, beta_a=1.1517, alpha_phi_rad=4.0033, beta_phi=9.1040)
G = Grid(fs=8e6, baud=1e6, n=4096, fc_hz=1e9)


def two_level(r_lo=0.3, r_hi=0.9, n=4096):
    """An envelope at two magnitudes, each with its own phase: a burst over a pilot."""
    x = np.where(np.arange(n) < n // 2, r_lo * np.exp(0.4j), r_hi * np.exp(-1.1j))
    return Signal(seed=1, grid=G).capture(values=x), x


def test_saleh_on_a_two_level_envelope_is_its_am_am_and_am_pm_curves():
    sig, x = two_level()
    y = sig.law("saleh", **SALEH).waveform()
    for r in (0.3, 0.9):
        at = np.abs(x) == r
        amp = SALEH["alpha_a"] * r / (1 + SALEH["beta_a"] * r ** 2)             # Saleh (1981)
        phi = SALEH["alpha_phi_rad"] * r ** 2 / (1 + SALEH["beta_phi"] * r ** 2)
        got_amp, got_phi = np.abs(y[at]), np.angle(y[at] * np.conj(x[at]))
        print(f"saleh at r = {r}: |y| {got_amp[0]:.6f} (law {amp:.6f}), phase added "
              f"{np.degrees(got_phi[0]):.3f} deg (law {np.degrees(phi):.3f})")
        assert np.allclose(got_amp, amp, rtol=1e-12) and np.allclose(got_phi, phi, atol=1e-12)
    # the level-dependent phase is the point: the two levels were turned by different amounts
    A, phi = ST.am_am_pm("saleh", np.array([0.3, 0.9]), **SALEH)
    assert np.degrees(phi[1] - phi[0]) > 5.0
    assert np.allclose(A, [SALEH["alpha_a"] * r / (1 + SALEH["beta_a"] * r ** 2) for r in (0.3, 0.9)])


def test_the_saleh_curve_has_the_landmarks_its_parameters_state():
    r = np.linspace(0.0, 30.0, 300001)
    A, phi = ST.am_am_pm("saleh", r, **SALEH)
    peak_at, peak = r[np.argmax(A)], A.max()
    assert peak_at == pytest.approx(1 / np.sqrt(SALEH["beta_a"]), abs=1e-3)
    assert peak == pytest.approx(SALEH["alpha_a"] / (2 * np.sqrt(SALEH["beta_a"])), rel=1e-6)
    assert A[1] / r[1] == pytest.approx(SALEH["alpha_a"], rel=1e-6)          # small-signal gain
    assert phi[-1] == pytest.approx(SALEH["alpha_phi_rad"] / SALEH["beta_phi"], rel=2e-4)


def test_rapp_is_linear_then_flat_and_leaves_the_phase_alone():
    sig, x = two_level(r_lo=0.05, r_hi=20.0)
    y = sig.law("rapp", gain=2.0, sat=1.5, smoothness=3.0).waveform()
    lo, hi = y[0], y[-1]
    want = lambda r: 2.0 * r / (1 + (2.0 * r / 1.5) ** 6) ** (1 / 6)            # Rapp (1991)
    assert abs(lo) == pytest.approx(want(0.05), rel=1e-12)
    assert abs(hi) == pytest.approx(want(20.0), rel=1e-12)
    assert abs(lo) == pytest.approx(0.1, rel=1e-6) and abs(hi) == pytest.approx(1.5, rel=1e-6)
    assert np.allclose(np.angle(y * np.conj(x)), 0.0, atol=1e-15)
    # the knee: a smaller `smoothness` is a softer one
    knee = [abs(Signal(seed=1, grid=G).capture(values=np.full(G.n, 0.75 + 0j))
                .law("rapp", gain=2.0, sat=1.5, smoothness=p).waveform()[0]) for p in (1, 3)]
    assert knee[0] < knee[1] < 1.5


# ------------------------------------------------------------------ against the passband route
# A limiter makes every odd harmonic of the carrier, and on a sampled record they alias. With
# fc = fs * 8/77 they land on multiples of fs/77 = 13 MHz -- clear of the low-pass below --
# and the first to land ON the carrier is the 153rd, which a tanh driven 28 dB past its knee
# no longer makes (its harmonics fall off exponentially past about the 25th).
FS = 1e9
FC = FS * 8 / 77
LEVELS = np.logspace(-1.5, 1.0, 13)
STEP = 1024
LPF = dict(pass_hz=3e6, stop_hz=12e6, atten_db=90)


def staircase():
    """13 drive levels from 22 dB below the knee to 28 dB past it, 1024 samples each, with
    raised-cosine steps between them so the envelope stays narrow-band."""
    n = len(LEVELS) * STEP
    r = np.repeat(LEVELS, STEP)
    w = np.hanning(129)
    r = np.convolve(np.pad(r, 64, mode="edge"), w / w.sum(), mode="valid")
    return Signal(seed=1, grid=Grid(fs=FS, n=n, fc_hz=FC)).capture(values=r.astype(complex))


def plateaus(y):
    mid = y.reshape(len(LEVELS), STEP)[:, STEP // 2 - 128: STEP // 2 + 128]
    return np.abs(mid).mean(axis=1), np.angle(mid.mean(axis=1))


def test_passband_tanh_gives_am_am_only_and_rapp_is_its_envelope_equivalent():
    gain, sat = 2.0, 0.8
    amp, phase = plateaus(staircase().upconvert().law("tanh", gain=gain, sat=sat)
                          .downconvert(fc_hz=FC, lpf=LPF).waveform())
    # the measurement itself, against the tanh's describing function (the fundamental of
    # tanh driven by a cosine), integrated independently
    th = np.linspace(0, 2 * np.pi, 4096, endpoint=False)
    describing = np.array([2 * np.mean(sat * np.tanh(gain * r * np.cos(th) / sat) * np.cos(th))
                           for r in LEVELS])
    e_meas = np.abs(amp / describing - 1).max()
    # AM/PM: none, at any drive
    worst_phase = np.abs(phase).max()
    # AM/AM: the envelope law rapp(gain, 4*sat/pi, smoothness=1.12), at every level
    env_amp, env_phase = plateaus(staircase().law("rapp", gain=gain, sat=4 * sat / np.pi,
                                                  smoothness=1.12).waveform())
    e_rapp = np.abs(env_amp / amp - 1)
    print(f"passband tanh: AM/AM within {e_meas:.1e} of its describing function; AM/PM "
          f"{worst_phase:.1e} rad at worst over {20 * np.log10(LEVELS[-1] / LEVELS[0]):.0f} dB "
          f"of drive. Envelope rapp(sat=4*sat/pi, smoothness=1.12) against it: worst "
          f"{100 * e_rapp.max():.2f} % (at the knee), {100 * e_rapp[0]:.3f} % small-signal, "
          f"{100 * e_rapp[-1]:.3f} % saturated")
    assert e_meas < 1e-4
    assert worst_phase < 1e-5                       # a memoryless passband law: AM/AM only
    assert e_rapp.max() < 0.012                     # 1.2 % at the knee
    assert e_rapp[0] < 2e-3 and e_rapp[-1] < 2e-3   # where they SHOULD agree, they do
    assert amp[-1] == pytest.approx(4 * sat / np.pi, rel=2e-3)     # a square wave's fundamental
    assert np.abs(env_phase).max() < 1e-12
    # ... and the same staircase through saleh is turned by the law's phase, level by level
    _, sal = plateaus(staircase().law("saleh", **SALEH).waveform())
    assert np.allclose(sal, ST.am_am_pm("saleh", LEVELS, **SALEH)[1], atol=1e-6)
    assert np.degrees(sal.max()) > 20.0


# ------------------------------------------------------------------ the recipe, and refusals
ENVELOPE = {
    "saleh": (SALEH, dict(alpha_a=1.9, beta_a=1.0, alpha_phi_rad=3.0, beta_phi=8.0)),
    "rapp": (dict(gain=1.0, sat=1.0, smoothness=2.0), dict(gain=1.5, sat=0.8, smoothness=3.0)),
}


def test_the_sweep_names_every_envelope_law_and_every_parameter_of_each():
    assert set(ENVELOPE) == set(ST.ENVELOPE_LAWS) <= set(ws.LAW_KINDS)
    for kind, (base, other) in ENVELOPE.items():
        assert set(base) == set(other) == set(ST._LAW[kind]), kind


@pytest.mark.parametrize("kind,key", [(k, p) for k, (b, _) in ENVELOPE.items() for p in b])
def test_every_envelope_law_parameter_moves_the_samples_and_the_digest(kind, key):
    base, other = ENVELOPE[kind]
    src = lambda: Signal(seed=1, grid=G).iq("qam16", prbs=9)
    a = src().law(kind, **base)
    b = src().law(kind, **{**base, key: other[key]})
    assert a.sha256() != b.sha256() and not np.array_equal(a.waveform(), b.waveform())
    assert np.array_equal(Signal.from_recipe(json.loads(a.to_json())).waveform(), a.waveform())
    assert a.roles() == []                       # a law draws nothing


def test_a_law_handed_the_other_kind_of_record_is_refused_where_it_is_written():
    env = lambda: Signal(seed=1, grid=G).iq("qpsk", prbs=9)
    real = lambda: Signal(seed=1, grid=Grid(fs=8e6, n=4096)).carrier("sine", f_hz=1e5)
    with pytest.raises(ValueError, match=r"law\('tanh'\).*complex baseband.*saleh and rapp"):
        env().law("tanh", gain=2.0)
    with pytest.raises(ValueError, match=r"law\('rapp'\).*real voltage.*ENVELOPE law.*tanh"):
        real().law("rapp")
    env().upconvert().law("tanh", gain=2.0)          # each is accepted on its own kind
    # ... and in a hand-written recipe, where the builder never saw it
    r = real().recipe()
    r["ops"].append({"op": "law", "kind": "saleh", **SALEH})
    with pytest.raises(ValueError, match="ENVELOPE law"):
        Signal.from_recipe(r).waveform()
    with pytest.raises(TypeError, match=r"law\('saleh'\) needs alpha_a, beta_a"):
        env().law("saleh")
    with pytest.raises(ValueError, match="cannot be negative"):
        env().law("saleh", **{**SALEH, "beta_a": -1.0})
    with pytest.raises(ValueError, match="smoothness must be positive"):
        env().law("rapp", smoothness=0.0)
    with pytest.raises(ValueError, match="no AM/AM or AM/PM curve"):
        ST.am_am_pm("tanh", 1.0)


def test_an_envelope_law_on_a_chain_that_cannot_yet_know_its_record_waits_for_the_render():
    """`Signal().use("tx")` and a stage of filters do not say what they act on, so the law is
    checked when it renders: accepted on I/Q, refused by name on a voltage."""
    src = Signal().iq("qpsk", prbs=9)
    for rx in (Signal().use("tx").law("rapp", sat=0.8),
               Signal().use("tx").filter("butter", f_hz=2e6).law("saleh", **SALEH),
               ws.recipes.multipath(Signal().use("tx"), [0.0, 1e-6], [1.0, 0.5]).law("rapp")):
        x = ws.Channels({"tx": src, "rx": rx}, seed=1, grid=G).waveforms()
        assert x["rx"].dtype == np.complex128
    # ... and it is the law itself: the same op written on the transmitter's own chain
    named = ws.Channels({"tx": src, "rx": Signal().use("tx").law("rapp", sat=0.8)},
                        seed=1, grid=G).waveforms()["rx"]
    direct = Signal(seed=1, grid=G).iq("qpsk", prbs=9).law("rapp", sat=0.8).waveform()
    assert np.allclose(named, direct, rtol=0, atol=1e-12)
    stage = Signal().filter("butter", f_hz=2e6).law("rapp")         # a shared stage builds
    assert [o["op"] for o in stage.ops] == ["filter", "law"]
    volts = ws.Channels({"tx": Signal().carrier("sine", f_hz=1e5),
                         "rx": Signal().use("tx").law("rapp")},
                        seed=1, grid=Grid(fs=8e6, n=4096))
    with pytest.raises(ValueError, match=r"law\('rapp'\).*real voltage.*ENVELOPE law"):
        volts.waveforms()
    with pytest.raises(ValueError, match=r"law\('rapp'\).*real voltage"):
        Signal().carrier("sine", f_hz=1e5).law("rapp")              # a source says: at once
