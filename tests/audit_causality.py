"""Assertions against a CLASS of defect: a stage that is not causal, or whose phase does not
belong to its magnitude.

`sosfiltfilt` filters forward AND backward. It is a two-line convenience with two consequences
that are invisible unless you look for them: the realised magnitude is the design SQUARED, and
the group delay is exactly zero because the impulse response is symmetric about t=0. A stage
built that way responds BEFORE its input arrives. Nothing in a magnitude-only test sees it, and
an eye diagram made from it looks better than the link it claims to model, because half of the
pre-cursor ISI has been moved into a pre-shoot that a real receiver never has to equalise.

WHAT THIS FILE ASSERTS, AND WHY EACH INSTRUMENT EXISTS
-----------------------------------------------------
Four instruments, in the order they were needed. Each was calibrated against a CONSTRUCTED
answer before any number it produced was believed (`test_the_instruments_recover_constructed_answers`),
and each is here because the one before it could not see something.

  1. **pre-energy fraction** — the fraction of impulse-response energy at t < 0. For a
     zero-phase Bessel it is ~35 %, which is the signature. MEASURED here: 0.0e+00 for a
     single-pass causal Bessel, 35.28 % for the same design through `sosfiltfilt`.

  2. **symmetry residual** — ``||h[-k] - h[+k]|| / ||h||``. Zero-phase means h is EXACTLY
     symmetric, so this collapses to the float epsilon (measured 2.3e-17) and is the sharpest
     possible detector of the defect. It is ONE-SIDED: a small residual proves zero phase, a
     large one proves nothing.

  3. **group delay** — the phase slope, which must accumulate across a chain. MEASURED: a
     five-stage chain delays 179.8002 ps and its parts sum to 179.8002 ps, either order.

  4. **the Kramers-Kronig / minimum-phase gate** — the measured phase must equal
     ``-Hilbert{log|H|}``. This is the only instrument that catches a stage whose magnitude was
     changed while its phase was left alone, WITHOUT knowing the stage's closed form. MEASURED:
     0.29 deg max error for the causal dispersive channel, 179.99 deg for its zero-phase form.

WHY INSTRUMENT 2 EXISTS AT ALL — a check that was never observed failing
-----------------------------------------------------------------------
Pre-energy is BLIND to a zero-phase high-pass. A first-order high-pass impulse response is a
delta minus a long shallow tail, so almost all of its energy sits in the one sample at t=0 and
the anticausal half carries only what the tail carries. MEASURED on a CONSTRUCTED zero-phase
high-pass at 2 GHz: pre-energy 6.06e-03 — below any threshold that separates a real stage from
rounding — while its symmetry residual is 6.4e-17 and its group delay is +0.0000 ps. A
pre-energy-only audit passes `ac_couple` and reports the file clean. That is exactly the failure
mode this project has been bitten by before, so the blind spot is asserted here as a test in its
own right, with the constructed filter that exhibits it.

WHERE ZERO PHASE IS CORRECT, STATED SO THE AUDIT CANNOT CONFUSE IT WITH A DEFECT
-------------------------------------------------------------------------------
A band limit sits in one of two places and they are not the same operator.

  * ANALOG, BEFORE the converter — front end, probe roll-off, RC load, series coupling cap,
    the transmitter's own edge shaping, the channel. A physical network: causal, single-pass,
    and it DELAYS. Nothing in it knows the future. Zero phase there is a DEFECT.
  * DIGITAL, AFTER the converter — the instrument's selected-bandwidth filter, a receiver's
    FFE, any DSP on a stored record. It can look forwards, and it does. Zero phase there is
    NOT a defect, it is what the instrument does, and
    `test_the_digital_selected_bandwidth_filter_is_zero_phase_and_that_is_correct` asserts it
    POSITIVELY so nobody "fixes" it.

The consequence for how the audit is written: **the gate is applied to the STAGE, never to a
record.** A record-level "there must be no pre-edge activity" gate is invalid, and
`test_a_record_level_causality_gate_would_false_positive_on_a_legitimate_record` measures why.
A record built entirely from causal stages shows 1e-4 of swing before an isolated transition;
put the instrument's own DIGITAL brickwall after the converter, as a real capture has, and the
same record shows 0.053 % of swing at 2.00 x baud, 0.413 % at 1.25 x, 2.090 % at 0.75 x and
4.290 % at 0.38 x — all of it legitimate. And a REAL export (a 2.7 Gb/s serial capture at
40 GSa/s, 556 isolated rising edges averaged) carries -7.14 % / +6.93 % of swing of pre-edge
excursion, with an antisymmetry residual of 0.946 about the 50 % crossing. That is what a real
record has that a constructed fixture does not: the instrument's own zero-phase DSP, plus
reflections, sitting on top of a causal link. Gate the operator, not the waveform.

Run it directly (``python3 wfmsynth/tests/audit_causality.py``) to see every number without a
harness; the numbers below are what it printed on a 256 GSa/s / 16 GBd grid, n = 16384.
"""
from __future__ import annotations

import os
import sys
import warnings

if __package__ in (None, "") and __name__ == "__main__":   # `python3 tests/audit_causality.py`
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
from scipy import signal as sg

from wfmsynth import instrument as INST
from wfmsynth import physics as P
from wfmsynth import rx as RX
from wfmsynth import sparam as SP
from wfmsynth.compose import Signal
from wfmsynth.grid import Grid

FS, BAUD, N = 256e9, 16e9, 1 << 14
SPB = int(FS / BAUD)
GRID = Grid(fs=FS, baud=BAUD, n=N)

# A causal stage's impulse response must have NEGLIGIBLE energy before t=0. "Negligible" is not
# a taste: the causal stages here measure 0.0e+00 (Bessel, edge shaping, CTLE), 7.5e-22
# (minimum-phase Gaussian) and 2.6e-12 (minimum-phase dispersive channel, the cepstral fold's
# own float noise), and the defect measures 0.35-0.46. Nine orders of margin either way.
CAUSAL_PRE_ENERGY = 1e-9
# Zero phase means h is symmetric to the float epsilon. Measured 2.3e-17 (Bessel through
# sosfiltfilt), 6.4e-17 (ac_couple), 1.9e-16 (zero-phase channel), 1.8e-16 (brickwall).
ZERO_PHASE_SYMMETRY = 1e-12


# ------------------------------------------------------------------ instruments
def _impulse_response(stage, n=N, m=None):
    """h[k] with the impulse at index `m` (the record's middle by default), so a symmetric
    response has room to reach backwards. Returns (h, m)."""
    m = n // 2 if m is None else m
    x = np.zeros(n)
    x[m] = 1.0
    return np.asarray(stage(x), float), m


def pre_energy_fraction(stage, n=N):
    """The fraction of impulse-response ENERGY at negative time. ~0 for a causal stage,
    ~0.35 for a zero-phase Bessel, 0.5 for an antisymmetric (centred-difference) kernel."""
    h, m = _impulse_response(stage, n)
    e = h ** 2
    return float(e[:m].sum() / (e.sum() + 1e-300))


def symmetry_residual(stage, n=N):
    """``||h[m-k] - h[m+k]|| / ||h||``. ONE-SIDED: below ZERO_PHASE_SYMMETRY the stage is
    zero-phase and that is proof; above it, nothing is proved (a causal high-pass whose delta
    dominates its tail measures 0.155)."""
    h, m = _impulse_response(stage, n)
    pre, post = h[:m][::-1], h[m + 1:]
    k = min(len(pre), len(post))
    return float(np.linalg.norm(pre[:k] - post[:k]) / (np.linalg.norm(h) + 1e-300))


def _H(stage, n=N):
    """(f_Hz, H) with the probe impulse's own m-sample offset divided out, so H is the
    stage's response and not the fixture's."""
    h, m = _impulse_response(stage, n)
    f = np.fft.rfftfreq(len(h), d=1.0 / FS)
    return f, np.fft.rfft(h) * np.exp(2j * np.pi * f * m / FS)


def group_delay_s(stage, band, n=N):
    """-d(phase)/d(omega), least-squares over `band`. Recovers a constructed 37-sample delay
    as 144.5313 ps against a truth of 144.5312."""
    f, H = _H(stage, n)
    k = (f >= band[0]) & (f <= band[1]) & (np.abs(H) > 1e-9 * np.abs(H).max())
    slope = np.polyfit(2 * np.pi * f[k], np.unwrap(np.angle(H[k])), 1)[0]
    return float(-slope)


def gain_phase_at(stage, f0_hz, n=N):
    """(|H|, phase_deg, f_actual) at the rfft bin nearest f0."""
    f, H = _H(stage, n)
    i = int(np.argmin(np.abs(f - f0_hz)))
    return float(abs(H[i])), float(np.degrees(np.angle(H[i]))), float(f[i])


def min_phase_error_deg(stage, band, n=N):
    """(max, rms) phase error against ``-Hilbert{log|H|}`` in degrees — the Kramers-Kronig
    gate. Deliberately NOT built on `physics._min_phase_H`: it folds the cepstrum here so a
    defect in that function cannot hide inside its own check. Applies only to a stage claiming
    a MINIMUM-phase response: a pure delay is causal and fails this gate by construction."""
    f, H = _H(stage, n)
    mag = np.abs(H)
    lm = np.log(np.maximum(mag, 1e-30))
    full = np.concatenate([lm, lm[-2:0:-1]])
    c = np.fft.ifft(full).real
    m = len(full)
    w = np.zeros(m)
    w[0] = 1.0
    w[1:m // 2] = 2.0
    w[m // 2] = 1.0
    hmin = np.exp(np.fft.fft(c * w))[:len(mag)]
    k = (mag > 1e-8 * mag.max()) & (f >= band[0]) & (f <= band[1])
    d = np.angle(H[k] * np.conj(hmin[k]))
    return float(np.degrees(np.abs(d).max())), float(np.degrees(np.sqrt((d ** 2).mean())))


def edge_lead_ps(causal, n=1 << 12, thresh=0.01):
    """How far BEFORE the symbol boundary the transmitter's edge starts moving, and the
    waveform's value AT that boundary. The record-level view of the source's edge shaping,
    on a record with exactly one transition so nothing else can be blamed."""
    g = Grid(fs=FS, baud=BAUD, n=n)
    n_ui = n // SPB
    k = n_ui // 2
    sym = [-1.0] * k + [1.0] * (n_ui - k)
    y = Signal(seed=1, grid=g).symbols(sym, tr_frac=0.25, causal=causal).waveform()
    k0 = k * SPB
    base = y[:SPB * 4].mean()
    swing = y[-SPB * 4:].mean() - base
    dev = np.abs(y - base)
    j = int(np.argmax(dev > thresh * abs(swing)))
    return (k0 - j) / FS * 1e12, float(y[k0])


# ------------------------------------------------------------------ stages under audit
def _scope(kind, bw=40e9, **kw):
    return lambda x: INST.scope_bandwidth(x, GRID, bw, kind=kind, **kw)


def _lossy(**kw):
    return lambda x: P.lossy_channel(x, grid=GRID, loss_db=10.0, loss_at_ghz=8.0, **kw)


def _ac_couple(fc_hz):
    def stage(x):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return P.ac_couple(x, fc_hz=fc_hz, grid=GRID)
    return stage


def _line(td_ps):
    f = np.fft.rfftfreq(N, d=1.0 / FS)
    s21 = SP.line(f, td_ps=td_ps, causal=False, loss_length_in=0.0).s21
    return lambda x: SP.sparam_channel(x, f, s21, grid=GRID)


def _ctle():
    return lambda x: RX.ctle(x, GRID, 2.0, 8.0, 16.0, dc_gain=0.65)


def _probe():
    return lambda x: INST.probe(x, GRID, c_load_f=0.45e-12)


# Every ANALOG stage the library exposes, i.e. every stage that models a physical network
# sitting before the converter. Each one must be causal. The registry is explicit rather than
# discovered, because a stage that has not been CLASSIFIED as analog or digital is exactly the
# stage this audit exists to catch.
ANALOG_STAGES = {
    "scope_bandwidth(kind='bessel')": _scope("bessel"),
    "scope_bandwidth(kind='gaussian')": _scope("gaussian"),
    "probe_loading(causal=True)": lambda x: INST.probe_loading(x, GRID, 0.45e-12, 50.0,
                                                               causal=True),
    "probe()": _probe(),
    "lossy_channel(causal=True)": _lossy(causal=True),
    "resonant_reflection": lambda x: P.resonant_reflection(x, grid=GRID, td_ps=90.0,
                                                           f0_ghz=10.7, q=9.0, gamma0=0.2),
    "multi_reflection": lambda x: P.multi_reflection(x, grid=GRID, td_ps=40.0, gamma_s=0.06,
                                                     gamma_l=0.06, n_bounce=3, node="source"),
    "sparam line(td=125ps)": _line(125.0),
    "_shape_edges(causal=True)": lambda x: P._shape_edges(x, 8.0, causal=True),
    "ctle": _ctle(),
}

# The per-stage pre-energy budget. `CAUSAL_PRE_ENERGY` (1e-9) is the default and holds for
# seven of the ten; the three exceptions are BAND TRUNCATION, not zero phase, and each is
# characterised rather than tolerated. A stage whose closed-form response is applied by
# sampling H on the rfft grid has its spectrum cut at Nyquist, and the cut tail reconstructs as
# symmetric ringing. The signature that distinguishes it from the defect is that it falls with
# fc/Nyquist and does NOT depend on record length:
#   probe loading  4.965e-03 at fc/Nyq = 0.0553, 1.245e-03 at 0.0138, 3.11e-04 at 0.0035;
#                  4.962e-03 / 4.965e-03 / 4.966e-03 at n = 4k / 16k / 64k.
#   resonant       1.477e-08 at 256 GSa/s, 1.774e-10 at 1024 GSa/s; 1.45e-08 / 1.48e-08 /
#                  1.48e-08 at n = 4k / 16k / 64k; and it falls with Q (1.34e-07 at Q=3,
#                  1.48e-08 at Q=9, 1.33e-09 at Q=30) because a narrower resonance has less
#                  skirt left above Nyquist.
#   sparam cascade 1.564e-03, from interpolating a measured band onto the FFT grid (a bare
#                  linear-phase line needs no exception at all: 8.5e-30).
# All of them are 2-8 orders below the 0.35-0.46 signature. `test_the_probe_loading_pole_is_
# band_truncation_and_not_a_zero_phase_stage` asserts the fs-scaling, so the exception cannot
# be used to hide a real acausality.
ANALOG_PRE_ENERGY = {
    "probe_loading(causal=True)": 1e-2,
    "probe()": 1e-2,
    "resonant_reflection": 1e-6,
}

# Stages whose zero phase is a DESIGN CHOICE, not a defect: DSP on a stored record.
DIGITAL_STAGES = {
    "scope_bandwidth(kind='brickwall')": _scope("brickwall"),
}


# ================================================================== 0. calibration
def test_the_instruments_recover_constructed_answers():
    """GROUND TRUTH FIRST. Nothing below is believed until each instrument reproduces an answer
    that is known in closed form, and until the KK gate is OBSERVED FAILING something.

    MEASURED, and asserted here:
      * a 37-sample pure delay: pre-energy 0.0e+00, group delay 144.5313 ps (truth 144.5312).
      * a symmetric 5-tap FIR [.1 .2 .4 .2 .1]: pre-energy 0.1923 (truth 0.1923 exactly, from
        the taps), group delay 0.0000 ps, symmetry residual 0.0e+00.
      * a bilinear single pole at 6.366 GHz: |H(fc)| 0.7068 (truth 0.70711), phase -45.028 deg
        (truth -45), pre-energy 0.0e+00, group delay at DC 24.7056 ps against the pole's own
        RC = 25.0008 ps (the 1.2 % is the bilinear frequency warp, not the instrument).
      * the KK gate returns 1.11 deg max on that same min-phase pole and 78.58 deg on the
        symmetric FIR. The second half is the point: a gate never observed failing is not a
        gate.
    """
    d = 37
    delay = lambda x: np.concatenate([np.zeros(d), np.asarray(x, float)[:-d]])
    assert pre_energy_fraction(delay) == 0.0
    gd = group_delay_s(delay, (1e9, 20e9))
    assert abs(gd - d / FS) < 1e-15, f"delay instrument: {gd * 1e12:.4f} ps for {d} samples"

    taps = np.array([0.1, 0.2, 0.4, 0.2, 0.1])
    fir = lambda x: np.convolve(np.asarray(x, float), taps, mode="same")
    truth = (taps[:2] ** 2).sum() / (taps ** 2).sum()
    assert abs(pre_energy_fraction(fir) - truth) < 1e-12
    assert abs(group_delay_s(fir, (1e9, 20e9))) < 1e-15
    assert symmetry_residual(fir) < ZERO_PHASE_SYMMETRY

    fc = 6.366e9
    b, a = sg.bilinear([1.0], [1.0 / (2 * np.pi * fc), 1.0], fs=FS)
    pole = lambda x: sg.lfilter(b, a, np.asarray(x, float))
    mag, ph, _f = gain_phase_at(pole, fc)
    assert abs(mag - 0.70711) < 5e-3, f"constructed pole |H(fc)| {mag:.4f}"
    assert abs(ph + 45.0) < 0.5, f"constructed pole phase {ph:.3f} deg"
    assert pre_energy_fraction(pole) < CAUSAL_PRE_ENERGY
    tau = 1.0 / (2 * np.pi * fc)
    assert abs(group_delay_s(pole, (1e6, 0.2 * fc)) / tau - 1.0) < 0.02

    assert min_phase_error_deg(pole, (1e9, 100e9))[0] < 3.0, "KK gate rejects a min-phase pole"
    assert min_phase_error_deg(fir, (1e9, 100e9))[0] > 30.0, (
        "KK gate does NOT flag a zero-phase FIR — it is not measuring anything")


def test_pre_energy_alone_cannot_see_a_zero_phase_high_pass():
    """The blind spot that made the symmetry residual necessary, asserted on a CONSTRUCTED
    filter so it can never quietly stop being true.

    A first-order high-pass impulse response is a delta minus a long shallow tail. Run it
    zero-phase and MEASURED at a 2 GHz corner: pre-energy 6.06e-03, i.e. 0.6 % — three orders
    below the 35 % signature and below any threshold that separates a stage from rounding —
    while the symmetry residual is 6.4e-17 (exactly symmetric) and the group delay is
    +0.0000 ps. The same filter run forwards only: pre-energy 0.0e+00, symmetry residual
    0.155, group delay +2.9203 ps.

    So: pre-energy is necessary and NOT sufficient, and any stage that is high-pass-shaped has
    to be gated on phase.
    """
    sos = sg.butter(1, 2e9 / (FS / 2.0), btype="high", output="sos")
    zero_phase = lambda x: sg.sosfiltfilt(sos, np.asarray(x, float))
    causal = lambda x: sg.sosfilt(sos, np.asarray(x, float))
    pe = pre_energy_fraction(zero_phase)
    assert pe < 0.02, (f"the constructed zero-phase high-pass shows {pe:.3e} of pre-energy; "
                       f"this test is the record that a pre-energy gate would pass it")
    assert symmetry_residual(zero_phase) < ZERO_PHASE_SYMMETRY
    assert abs(group_delay_s(zero_phase, (4e9, 20e9))) < 1e-15
    assert pre_energy_fraction(causal) < CAUSAL_PRE_ENERGY
    assert group_delay_s(causal, (4e9, 20e9)) > 1e-12


# ================================================================== 1. the analog stages
@pytest.mark.parametrize("name", sorted(ANALOG_STAGES))
def test_an_analog_stage_puts_no_energy_before_t_equals_zero(name):
    """THE class assertion. Every stage that models a physical network before the converter
    gets an impulse in the middle of a quiescent record and must leave the first half of it
    alone.

    MEASURED, all ten analog stages: 0.0e+00 (bessel front end, causal edge shaping, CTLE,
    source-node echo train, the s-parameter line at 8.5e-30), 7.5e-22 (minimum-phase Gaussian
    front end), 2.6e-12
    (minimum-phase dispersive channel — the cepstral fold's own float noise), 1.5e-08
    (resonant reflection), 4.9e-03 (probe loading). The last two are band truncation, budgeted
    and characterised in `ANALOG_PRE_ENERGY`; the defect signature is 0.35-0.46, i.e. 5 to 8
    orders above the largest of them and 8 orders above the default budget.
    """
    pe = pre_energy_fraction(ANALOG_STAGES[name])
    limit = ANALOG_PRE_ENERGY.get(name, CAUSAL_PRE_ENERGY)
    assert pe < limit, (f"{name}: {pe:.4e} of the impulse-response energy is at t < 0 "
                        f"(budget {limit:g}). An analog stage cannot respond before its input "
                        f"arrives; symmetry residual "
                        f"{symmetry_residual(ANALOG_STAGES[name]):.2e} "
                        f"(< {ZERO_PHASE_SYMMETRY:g} would mean exactly zero phase).")


def test_the_probe_loading_pole_is_band_truncation_and_not_a_zero_phase_stage():
    """`probe_loading(causal=True)` applies the exact continuous-time pole ``1/(1+jf/fc)`` on
    the rfft grid, and MEASURES 0.4965 % of pre-energy rather than zero. That is NOT the
    zero-phase defect and this test is what separates the two.

    The residual is BAND TRUNCATION, established by how it scales. It is independent of record
    length — 0.4962 % at n = 4096, 0.4965 % at 16384, 0.4966 % at 65536, so it is not a
    circular wrap — and it falls linearly with fc/Nyquist: 0.4965 % at fc/Nyq = 0.0553
    (256 GSa/s), 0.1245 % at 0.0138 (1024 GSa/s), 0.0311 % at 0.0035 (4096 GSa/s). An ideal
    single pole's spectrum does not end at Nyquist; a sampled record's does, and the truncated
    tail reconstructs as symmetric ringing. That is the price of the exact closed form, it is
    70x below the zero-phase signature, and the phase is the pole's:

    MEASURED at R = 50 ohm, C = 0.45 pF (fc = 7.0736 GHz): |H(fc)| = 0.7069 against the
    closed form 0.70711, phase -45.019 deg against -45, group delay 22.2087 ps against
    RC = 22.5079 ps. Its zero-phase opt-out measures |H(fc)| the closed form SQUARED and
    -45.019 deg of phase at a doubled magnitude, which is the whole reason the flag exists.
    """
    fc = INST.rc_pole_hz(50.0, 0.45e-12)
    stage = lambda x: INST.probe_loading(x, GRID, 0.45e-12, 50.0, causal=True)
    mag, ph, fa = gain_phase_at(stage, fc)
    assert abs(mag - 0.70711) < 5e-3, f"|H(fc)| = {mag:.4f} at {fa / 1e9:.4f} GHz"
    assert abs(ph + 45.0) < 1.0, (f"a stage claiming ONE RC pole must be -45 deg at its "
                                  f"corner; measured {ph:.3f} deg")
    tau = 1.0 / (2 * np.pi * fc)
    gd = group_delay_s(stage, (1e8, 0.2 * fc))
    assert abs(gd / tau - 1.0) < 0.05, (f"group delay {gd * 1e12:.4f} ps against the pole's "
                                        f"own RC = {tau * 1e12:.4f} ps")
    assert symmetry_residual(stage) > ZERO_PHASE_SYMMETRY, "the pole came out zero-phase"
    coarse = pre_energy_fraction(stage)
    fine_grid = Grid(fs=4 * FS, baud=BAUD, n=N)
    fine = pre_energy_fraction(lambda x: INST.probe_loading(x, fine_grid, 0.45e-12, 50.0,
                                                            causal=True))
    assert fine < 0.4 * coarse, (f"the pre-energy residual does not fall with fc/Nyquist "
                                 f"({coarse:.3e} -> {fine:.3e} at 4x the sample rate), so it "
                                 f"is not band truncation and needs a different explanation")


def test_an_analog_stage_delays_and_the_chain_delay_is_the_sum_of_its_parts():
    """Group delay must ACCUMULATE. Five stages, four different delay mechanisms — a linear
    phase (the s-parameter line), a minimum-phase loss (Kramers-Kronig), a pole pair (CTLE), a
    single-pass Bessel and an RC load — and the composite must be their sum, in either order,
    because they are all LTI.

    MEASURED over 1-10 GHz: line 125.0000, CTLE 9.5285, front end 7.7645, probe 14.2749,
    causal channel 23.2323 ps; sum 179.8002 ps; chain 179.8002 ps forwards and 179.8002 ps
    reversed. Every part is strictly positive, which is the other half of the claim: a
    physical network in the signal path delays.
    """
    parts = {"line 125 ps": _line(125.0), "ctle": _ctle(),
             "front end bessel 40 GHz": _scope("bessel"), "probe 0.45 pF": _probe(),
             "channel 10 dB @ 8 GHz causal": _lossy(causal=True)}
    band = (1e9, 10e9)
    each = {k: group_delay_s(v, band) for k, v in parts.items()}
    for k, v in each.items():
        assert v > 1e-13, f"{k} has no group delay ({v * 1e12:.4f} ps)"

    def run(order):
        def chain(x):
            for k in order:
                x = parts[k](x)
            return x
        return group_delay_s(chain, band)

    fwd, rev = run(list(parts)), run(list(parts)[::-1])
    total = sum(each.values())
    detail = ", ".join(f"{k} {v * 1e12:.4f}" for k, v in each.items())
    assert abs(fwd - total) < 0.05e-12, (f"chain {fwd * 1e12:.4f} ps, parts sum to "
                                         f"{total * 1e12:.4f} ps ({detail})")
    assert abs(rev - fwd) < 0.05e-12, f"order changed the delay: {rev * 1e12:.4f} ps"


def test_a_transmission_line_delays_by_exactly_the_td_it_was_given():
    """The one stage whose delay is a closed form with no approximation in it: a pure linear
    phase. MEASURED 50.0000 / 125.0000 / 400.0000 ps for td_ps = 50 / 125 / 400, and a
    three-section cascade of a 150 ps line, a 0.12 discontinuity and a 90 ps line delays
    240.0000 ps — its lines' sum — with 0.16 % pre-energy (the measured-band interpolation's
    truncation, the same mechanism as the probe pole's)."""
    for td in (50.0, 125.0, 400.0):
        gd = group_delay_s(_line(td), (1e9, 10e9)) * 1e12
        assert abs(gd - td) < 1e-3, f"td_ps={td}: measured {gd:.4f} ps"
    path = [{"line": {"td_ps": 150.0, "loss_db": 6.0, "loss_at_ghz": 8.0, "causal": True}},
            {"disc": {"gamma": 0.12}},
            {"line": {"td_ps": 90.0, "loss_db": 2.0, "loss_at_ghz": 8.0, "causal": True}}]
    casc = lambda x: SP.cascade_channel(x, path, grid=GRID)
    gd = group_delay_s(casc, (1e9, 10e9)) * 1e12
    assert abs(gd - 240.0) < 0.05, f"cascade delayed {gd:.4f} ps, its lines sum to 240"
    assert pre_energy_fraction(casc) < 1e-2


def test_the_causal_dispersive_channel_satisfies_kramers_kronig():
    """A stage that claims a causal dispersive channel must have the phase its own magnitude
    implies: ``phase = -Hilbert{log|H|}``. Folded here independently of `physics._min_phase_H`
    so the check cannot be satisfied by the function it is checking.

    MEASURED over 1-100 GHz on a 10 dB @ 8 GHz channel: causal=True gives 0.29 deg max /
    0.06 deg rms; the same channel's magnitude-only form gives 179.99 deg max / 151.43 deg rms,
    which is what a real magnitude with no phase at all looks like to this gate. The
    minimum-phase Gaussian front end measures 1.1e-07 deg and the single-pass Bessel
    0.019 deg.
    """
    band = (1e9, 100e9)
    mx, rms = min_phase_error_deg(_lossy(causal=True), band)
    assert mx < 5.0 and rms < 1.0, (f"the causal channel's phase is not its magnitude's: "
                                    f"{mx:.3f} deg max, {rms:.3f} deg rms against "
                                    f"-Hilbert(log|H|)")
    bad, _ = min_phase_error_deg(_lossy(), band)
    assert bad > 90.0, (f"the magnitude-only channel measures only {bad:.3f} deg of phase "
                        f"error, so this gate is not seeing the difference it exists for")
    assert min_phase_error_deg(_scope("gaussian"), (1e9, 60e9))[0] < 1.0
    assert min_phase_error_deg(_scope("bessel"), (1e9, 60e9))[0] < 1.0


# ================================================================== 2. where zero phase is right
@pytest.mark.parametrize("name", sorted(DIGITAL_STAGES))
def test_the_digital_selected_bandwidth_filter_is_zero_phase_and_that_is_correct(name):
    """ASSERTED POSITIVELY, so the audit cannot be read as condemning it and nobody "fixes" it.

    The instrument's selected-bandwidth filter sits AFTER the converter, on a stored record, and
    is DSP: it can look forwards and it does. MEASURED: 34.37 % of its impulse-response energy
    is at t < 0, its symmetry residual is 1.8e-16 (exactly symmetric) and its group delay is
    +0.0000 ps — the same three numbers that condemn an analog stage, and here they are the
    specification. The corroboration is a real export: the record's noise floor drops 38 dB
    across 2 GHz at the corner with no group delay to be found.

    The stage also has to REFUSE a causal form rather than quietly oblige, because a flag that
    is accepted and ignored is worse than no flag.
    """
    stage = DIGITAL_STAGES[name]
    assert pre_energy_fraction(stage) > 0.1, (
        f"{name} has become causal. That is not this stage's job — if the intent is an analog "
        f"band limit, use an analog kind; if this is a deliberate redesign, this test is the "
        f"place to record it.")
    assert symmetry_residual(stage) < ZERO_PHASE_SYMMETRY
    assert abs(group_delay_s(stage, (1e9, 10e9))) < 1e-15
    with pytest.raises(ValueError):
        INST.scope_bandwidth(np.zeros(64), GRID, 40e9, kind="brickwall", causal=True)


def test_a_digital_transmit_ffe_may_look_forward_and_its_delay_is_a_closed_form():
    """A Tx FFE is digital: the transmitter has the whole symbol stream and a pre-cursor tap is
    the standard preset, so a NEGATIVE group delay there is a modelling convention and not an
    acausality. What must not drift is the tap INDEXING, so it is pinned to a closed form
    instead: a single unit tap at ``pre=p`` has group delay exactly ``-p * spb / fs``.

    MEASURED at 16 samples/UI: +0.0000 / -62.5000 / -125.0000 ps for p = 0 / 1 / 2, against a
    closed form of -0.0000 / -62.5000 / -125.0000.

    The trap this pins, stated because it costs a whole UI: `de_emphasis_taps(db)` returns
    ``[main, post]`` and `tx_ffe`'s DEFAULT is ``pre=1``, so calling them together puts the MAIN
    cursor one UI in the past — MEASURED -58.0449 ps of group delay and 97.16 % of the impulse
    response before t=0. `compose._op_de_emphasis` passes ``pre=0`` and measures +4.4551 ps and
    0.00 %, which is the correct pairing; this test asserts both, so the composer cannot quietly
    acquire the default.
    """
    for p in (0, 1, 2):
        gd = group_delay_s(lambda x, p=p: P.tx_ffe(x, [1.0], SPB, pre=p), (1e9, 20e9))
        want = -p * SPB / FS
        assert abs(gd - want) < 1e-15, (f"tx_ffe(pre={p}) delayed {gd * 1e12:+.4f} ps, closed "
                                        f"form {want * 1e12:+.4f} ps")
    de = P.de_emphasis_taps(3.0)
    assert pre_energy_fraction(lambda x: P.tx_ffe(x, de, SPB, pre=0)) < CAUSAL_PRE_ENERGY, (
        "de_emphasis_taps are [main, post] and must be applied with pre=0")
    assert pre_energy_fraction(lambda x: P.tx_ffe(x, de, SPB, pre=1)) > 0.5, (
        "tx_ffe(pre=1) no longer advances a [main, post] pair by a UI — the tap indexing "
        "moved, and the closed-form check above is the one to trust")


def test_a_causal_stage_may_still_have_negative_band_group_delay():
    """The discriminator that keeps this audit from reporting a false defect, and the reason
    "group delay must be non-negative" is asserted for SINGLE stages with a stated delay and
    not for arbitrary ones.

    A passive echo sum is ``x + gamma * x(t - td) + ...``: every term is causal, and the sum's
    group delay over a band can be NEGATIVE. MEASURED over 1-10 GHz: multi_reflection at the
    source node -2.5054 ps, resonant_reflection -0.9376 ps — and over 2-4 GHz the same two
    stages read +0.6479 and +0.6649 ps, so the sign is a property of the band, not of the
    stage. Pre-energy: 0.0e+00 and 1.5e-08.

    The gate is therefore pre-energy, which is a statement about the operator, not band group
    delay, which is a statement about a band.
    """
    for name in ("multi_reflection", "resonant_reflection"):
        stage = ANALOG_STAGES[name]
        pe = pre_energy_fraction(stage)
        assert pe < ANALOG_PRE_ENERGY.get(name, CAUSAL_PRE_ENERGY), f"{name}: pre-energy {pe:.3e}"
    bands = ((1e9, 10e9), (2e9, 4e9), (6e9, 12e9), (1e9, 20e9))
    seen = {name: [group_delay_s(ANALOG_STAGES[name], b) * 1e12 for b in bands]
            for name in ("multi_reflection", "resonant_reflection")}
    assert any(v < 0.0 for vs in seen.values() for v in vs), (
        f"neither causal echo stage shows a negative band group delay any more "
        f"({seen}), so the discriminator this test records can no longer be demonstrated "
        f"and a non-negativity gate could be applied to these stages after all")


def test_a_record_level_causality_gate_would_false_positive_on_a_legitimate_record():
    """WHY THE GATE IS ON THE STAGE AND NEVER ON A RECORD. This is the "what does a REAL record
    have that the fixture does not" half, measured rather than asserted.

    One isolated transition, a causal transmitter and a causal channel: MEASURED 0.01 % of
    swing of motion in the 2 UI before the symbol boundary. Now add the instrument's own
    DIGITAL brickwall, which every real capture has been through: 0.053 % of swing at
    2.00 x baud, 0.413 % at 1.25 x, 2.090 % at 0.75 x, 3.353 % at 0.50 x, 4.290 % at 0.38 x.
    None of that is a defect — it is the scope's DSP, doing what it does.

    A REAL export agrees and goes further: a 2.7 Gb/s serial capture at 40 GSa/s, 556 isolated
    rising edges averaged, carries -7.14 % / +6.93 % of swing of pre-edge excursion and an
    antisymmetry residual of 0.946 about its 50 % crossing. So a record-level "no pre-edge
    activity" gate would fail three real captures and pass nothing useful, while the
    stage-level gate above separates 0.0e+00 from 0.35 with nine orders of margin.
    """
    n = 1 << 13
    g = Grid(fs=FS, baud=BAUD, n=n)
    n_ui, k = n // SPB, (n // SPB) // 2
    sym = [-1.0] * k + [1.0] * (n_ui - k)
    k0 = k * SPB

    def pre_edge(bw_hz=None):
        s = (Signal(seed=1, grid=g).symbols(sym, tr_frac=0.25, causal=True)
             .lossy(loss_db=8.0, loss_at_ghz=8.0, causal=True))
        if bw_hz is not None:
            s = s.scope(bw_hz=bw_hz, kind="brickwall")
        y = s.waveform()
        base = y[k0 - 8 * SPB:k0 - 2 * SPB].mean()
        swing = y[-4 * SPB:].mean() - base
        return float(np.abs((y[k0 - 2 * SPB:k0] - base) / swing).max())

    clean = pre_edge()
    assert clean < 1e-3, f"the all-causal record shows {clean * 100:.4f} % of swing pre-edge"
    ring = {bw: pre_edge(bw) for bw in (32e9, 12e9, 6e9)}
    assert ring[6e9] > 20 * clean, (
        f"the DIGITAL brickwall no longer puts pre-edge ringing on a record "
        f"({ring}); either it became causal (see the digital-stage test) or a record-level "
        f"gate has stopped being wrong, and this file's reasoning needs revisiting")
    assert ring[6e9] > ring[12e9] > ring[32e9], f"pre-ringing not monotone in the corner: {ring}"


# ================================================================== 3. the live defects
@pytest.mark.xfail(strict=True, reason=(
    "MEASURED DEFECT, asserted as physics: `physics.ac_couple` is a series COUPLING CAPACITOR, "
    "an analog network, and it is applied with `sosfiltfilt` — forwards and backwards. Two "
    "independent errors at once, both measured on a 256 GSa/s grid and both against a "
    "first-order high-pass's closed form. MAGNITUDE: |H(fc)| = 0.5000 where the closed form is "
    "0.70711, i.e. exactly the closed form SQUARED (at fc/10 it reads 0.01021 against 0.10104, "
    "which is 0.10104**2; at 10*fc, 0.99049 against 0.99504 = 0.99504**2). The realised corner "
    "is therefore 1.55x the requested one at every setting. PHASE: -0.000 deg at the corner "
    "where a first-order high-pass is +45, and -0.000 deg at fc/10 where it is +84.20 — the "
    "phase is zero at EVERY frequency, the symmetry residual is 6.45e-17 (h is exactly "
    "symmetric) and the group delay is +0.0000 ps. Against the Kramers-Kronig gate it reads "
    "179.24 deg max / 29.88 deg rms. What a record sees: on a single isolated transition the "
    "baseline droops to 100 % of the post-edge droop's peak BEFORE the edge, and the first "
    "motion is 367.2 ps before it at fc = 2 GHz and 3668.0 ps before it at fc = 0.2 GHz — the "
    "record sags in anticipation of a transition that has not happened. Note the pre-energy "
    "fraction is only 0.61 % here, which is why `test_pre_energy_alone_cannot_see_a_zero_phase_"
    "high_pass` exists: this defect is invisible to the headline instrument and has to be "
    "caught on phase. The fix is one forward pass of `butter(1, fc, 'high')` with "
    "`sosfilt_zi`-initialised state; remove this marker then."))
def test_ac_coupling_is_a_causal_first_order_high_pass():
    """A series cap is a physical network. Its magnitude and its phase are the same pole."""
    for fc in (0.5e9, 2e9, 8e9):
        stage = _ac_couple(fc)
        mag, ph, fa = gain_phase_at(stage, fc)
        assert abs(mag - 0.70711) < 0.02, (f"|H(fc)| = {mag:.4f} at fc = {fa / 1e9:.3f} GHz; "
                                           f"the closed form is 0.70711 and its SQUARE is "
                                           f"{0.70711 ** 2:.4f}")
        assert abs(ph - 45.0) < 2.0, (f"a first-order high-pass is +45 deg at its corner; "
                                      f"measured {ph:+.3f} deg")
        _m10, ph10, f10 = gain_phase_at(stage, fc / 10.0)
        want = np.degrees(np.arctan(fc / f10))
        assert abs(ph10 - want) < 3.0, (f"phase at fc/10 = {ph10:+.3f} deg, closed form "
                                        f"{want:+.3f}")
        assert symmetry_residual(stage) > ZERO_PHASE_SYMMETRY, "h is exactly symmetric"
        assert group_delay_s(stage, (2 * fc, 10 * fc)) > 0.0


@pytest.mark.xfail(strict=True, reason=(
    "MEASURED DEFECT IN A DEFAULT: `physics.lossy_channel`'s `causal` parameter defaults to "
    "False, and `compose._op_lossy` forwards it only when a recipe names it — so every recipe "
    "that writes `.lossy(loss_db=..., loss_at_ghz=...)` and does not know to add `causal=True` "
    "gets a magnitude-only, zero-phase, NON-CAUSAL interconnect. MEASURED on a 10 dB @ 8 GHz "
    "channel at 256 GSa/s: 42.21 % of the impulse-response energy is at t < 0, the symmetry "
    "residual is 1.93e-16 (h is exactly symmetric to the float epsilon), the group delay is "
    "+0.0000 ps where the same channel's causal form delays 23.2323 ps, and the "
    "Kramers-Kronig gate reads 179.99 deg max / 151.43 deg rms — a real magnitude with no "
    "phase at all. What a record sees: on an isolated step the output starts moving 4007.81 ps "
    "before the edge and reaches 47.80 % of the step height before the edge arrives. Half of "
    "the channel's dispersion is delivered as pre-cursor ISI that no real link has and no real "
    "receiver has to equalise, which is the direction that makes a synthesised eye look better "
    "than the board. `causal=True` measures 2.6e-12 pre-energy and 0.044 % pre-step. The fix "
    "is to default the flag to True (the legacy form stays reachable as an explicit opt-out, "
    "exactly as `scope_bandwidth` and `probe_loading` now do); remove this marker then."))
def test_the_channel_a_recipe_gets_without_asking_is_causal():
    """A dispersive interconnect is the textbook Kramers-Kronig system: state the loss and the
    phase is no longer free. A recipe that does not mention phase must still get the right one."""
    stage = _lossy()
    pe = pre_energy_fraction(stage)
    assert pe < CAUSAL_PRE_ENERGY, (f"the DEFAULT channel puts {pe * 100:.2f} % of its "
                                    f"impulse-response energy before t=0 (symmetry residual "
                                    f"{symmetry_residual(stage):.2e})")
    assert group_delay_s(stage, (1e9, 10e9)) > 1e-13
    assert min_phase_error_deg(stage, (1e9, 100e9))[0] < 5.0


@pytest.mark.xfail(strict=True, reason=(
    "MEASURED DEFECT IN A DEFAULT: `physics._shape_edges` — the TRANSMITTER's own edge "
    "shaping, before any channel — is `sosfiltfilt` unless `causal=True` is passed, and "
    "`compose._carrier`/`_op_symbols` default it to False. A transmitter's output stage is an "
    "analog network; this one switches before the bit arrives. MEASURED at 16 samples/UI, "
    "tr_frac = 0.25: 45.81 % of the shaping kernel's energy is at t < 0, symmetry residual "
    "1.09e-15, group delay +0.0000 ps against +44.9499 ps for the causal form. On a record "
    "with exactly ONE transition, at 16 GBd on a 256 GSa/s grid: the edge starts moving "
    "27.34 ps (0.44 UI) BEFORE the symbol boundary that causes it, and the waveform AT the "
    "boundary is already +0.1213 — 10.6 % of the way up a +/-1 swing — where the causal form "
    "reads -0.9945 and starts 3.91 ps after. Every eye diagram from a default carrier is "
    "therefore built on edges that anticipate their own data, and the jitter that `Jitter` "
    "applies to edge TIMES is applied to edges that are already 0.44 UI early. The fix is to "
    "default `causal=True` through `nrz`/`pam4`/`from_symbols` and the composer's carrier; "
    "remove this marker then."))
def test_the_transmitter_edge_a_recipe_gets_without_asking_is_causal():
    """The source's own edges, which no channel-side fix reaches."""
    kernel = lambda x: P._shape_edges(x, 8.0)
    pe = pre_energy_fraction(kernel)
    lead_ps, at_boundary = edge_lead_ps(causal=False)
    assert pe < CAUSAL_PRE_ENERGY, (
        f"the DEFAULT edge shaping puts {pe * 100:.2f} % of its energy before t=0 (symmetry "
        f"residual {symmetry_residual(kernel):.2e}); on a single transition the edge starts "
        f"{lead_ps:.2f} ps before its symbol boundary and the record is already at "
        f"{at_boundary:+.4f} there")
    assert lead_ps <= 0.0, f"the edge starts {lead_ps:.2f} ps before its own symbol boundary"
    assert group_delay_s(kernel, (1e9, 10e9)) > 1e-13


@pytest.mark.xfail(strict=True, reason=(
    "MEASURED DEFECT: `physics.crosstalk(kind='fext')` couples the aggressor's FUTURE. FEXT is "
    "proportional to dV/dt of the neighbouring line and the model takes that derivative with "
    "`np.gradient`, a CENTRED difference — MEASURED, the coupled term's kernel is "
    "[0, +1.2e-10, 0, -1.2e-10, 0] about t=0, exactly 50.00 % of its energy is at t < 0 and "
    "its group delay is +0.0000 ps. Half of the crosstalk a victim receives therefore arrives "
    "before the aggressor's transition that caused it. The scale is one sample: a one-sided "
    "difference measures 0.00 % pre-energy and +1.9531 ps of group delay, exactly half a "
    "sample at 256 GSa/s, so the model runs 1.95 ps = 3.1 % of a 16 GBd UI early and spreads "
    "the coupled pulse symmetrically instead of one-sidedly. `kind='next'` is already causal "
    "(0.00 % pre-energy) — it is a delayed copy, not a derivative. The fix is a backward "
    "difference (and a half-sample delay if the two kinds must stay aligned); remove this "
    "marker then."))
def test_forward_crosstalk_does_not_couple_the_aggressor_s_future():
    """FEXT is a derivative, and every causal discrete derivative is one-sided."""
    fext = lambda a: P.crosstalk(np.zeros(len(a)), a, coupling=0.12, kind="fext")
    nxt = lambda a: P.crosstalk(np.zeros(len(a)), a, coupling=0.12, kind="next", td_frac=0.05)
    assert pre_energy_fraction(nxt) < CAUSAL_PRE_ENERGY, "NEXT was causal and is not any more"
    pe = pre_energy_fraction(fext)
    assert pe < 1e-2, (f"FEXT delivers {pe * 100:.2f} % of its coupled energy before the "
                       f"aggressor's transition; its kernel is a centred difference "
                       f"(group delay {group_delay_s(fext, (1e9, 20e9)) * 1e12:+.4f} ps, "
                       f"a one-sided one measures {0.5 / FS * 1e12:+.4f} ps)")


# ------------------------------------------------------------------ direct run
def _report():
    """Print every measurement, then run every test in this file and say what happened."""
    import inspect

    warnings.simplefilter("ignore")
    print(f"grid: fs = {FS / 1e9:.0f} GSa/s, baud = {BAUD / 1e9:.0f} GBd, n = {N}, "
          f"{SPB} samples/UI\n")
    hdr = f"{'stage':40s} {'pre-energy':>12s} {'symmetry':>11s} {'gd 1-10 GHz':>13s}"
    print(hdr)
    print("-" * len(hdr))
    for label, table in (("ANALOG (must be causal)", ANALOG_STAGES),
                         ("DIGITAL (zero phase is correct)", DIGITAL_STAGES),
                         ("LIVE DEFECTS", {
                             "ac_couple(fc=2 GHz)": _ac_couple(2e9),
                             "lossy_channel() [default]": _lossy(),
                             "_shape_edges() [default]": lambda x: P._shape_edges(x, 8.0),
                             "crosstalk(kind='fext')": (
                                 lambda a: P.crosstalk(np.zeros(len(a)), a, kind="fext")),
                             "scope_bandwidth(bessel, causal=False)": _scope("bessel",
                                                                             causal=False),
                             "probe_loading(causal=False)": (
                                 lambda x: INST.probe_loading(x, GRID, 0.45e-12, 50.0,
                                                              causal=False)),
                         })):
        print(f"\n-- {label}")
        for name, stage in table.items():
            print(f"{name:40s} {pre_energy_fraction(stage):12.4e} "
                  f"{symmetry_residual(stage):11.2e} "
                  f"{group_delay_s(stage, (1e9, 10e9)) * 1e12:10.4f} ps")

    print("\n-- closed forms")
    fc = INST.rc_pole_hz(50.0, 0.45e-12)
    for name, stage, f0, want_mag, want_ph in (
            ("probe_loading(causal=True) @ fc", ANALOG_STAGES["probe_loading(causal=True)"],
             fc, 0.70711, -45.0),
            ("probe_loading(causal=False) @ fc",
             lambda x: INST.probe_loading(x, GRID, 0.45e-12, 50.0, causal=False), fc,
             0.70711, -45.0),
            ("ac_couple(2 GHz) @ fc", _ac_couple(2e9), 2e9, 0.70711, 45.0)):
        mag, ph, fa = gain_phase_at(stage, f0)
        print(f"{name:40s} |H| {mag:.4f} (closed form {want_mag:.5f}, squared "
              f"{want_mag ** 2:.5f})   phase {ph:+8.3f} deg (closed form {want_ph:+.0f})")

    print("\n-- Kramers-Kronig gate (max, rms deg vs -Hilbert(log|H|))")
    for name, stage, band in (("lossy(causal=True)", _lossy(causal=True), (1e9, 100e9)),
                              ("lossy() [default]", _lossy(), (1e9, 100e9)),
                              ("ac_couple(2 GHz)", _ac_couple(2e9), (0.2e9, 60e9)),
                              ("scope bessel (causal)", _scope("bessel"), (1e9, 60e9)),
                              ("scope gaussian (causal)", _scope("gaussian"), (1e9, 60e9))):
        mx, rms = min_phase_error_deg(stage, band)
        print(f"{name:40s} {mx:9.3f} {rms:9.3f}")

    print("\n-- the record-level view of the source's edge shaping")
    for causal in (False, True):
        lead, at_b = edge_lead_ps(causal)
        print(f"  causal={causal!s:5s}: the edge starts {lead:8.2f} ps before its symbol "
              f"boundary; value AT the boundary {at_b:+.4f}")

    print("\n" + "=" * 78)
    tests = [(n, f) for n, f in sorted(vars(sys.modules[__name__]).items())
             if n.startswith("test_") and inspect.isfunction(f)]
    fails = 0
    for name, fn in tests:
        marks = getattr(fn, "pytestmark", [])
        xfail = any(m.name == "xfail" for m in marks)
        params = [m for m in marks if m.name == "parametrize"]
        cases = [{}]
        if params:
            key = params[0].args[0]
            cases = [{key: v} for v in params[0].args[1]]
        for kw in cases:
            tag = f"{name}{'[' + str(list(kw.values())[0]) + ']' if kw else ''}"
            try:
                fn(**kw)
                verdict = "XPASS (a fix landed -- remove the marker)" if xfail else "pass"
                fails += 1 if xfail else 0
            except AssertionError as exc:
                first = str(exc).strip().splitlines()[0]
                verdict = f"xfail: {first}" if xfail else f"FAIL: {first}"
                fails += 0 if xfail else 1
            print(f"  {verdict.split(':')[0]:6s} {tag}")
            if verdict.startswith(("xfail", "FAIL", "XPASS")):
                print(f"         {verdict.split(':', 1)[-1].strip()[:300]}")
    print(f"\n{len(tests)} test functions; {fails} unexpected result(s).")
    return fails


if __name__ == "__main__":
    raise SystemExit(1 if _report() else 0)
