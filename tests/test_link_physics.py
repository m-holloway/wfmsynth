"""The link physics an engineer looking at a modelled record would miss — three mechanisms that
had NO representation, each asserted against a CLOSED FORM rather than against a previous run.

U-09  an INDEPENDENT sampling clock. `compose._op_digitize`'s `n_out=` reaches the acquisition
      rate through `resample_poly`, a RATIONAL and therefore phase-locked rate change: the
      record's timebase is a divided copy of the link's and the two never drift apart. A real
      instrument's timebase is a different oscillator.
U-12  receiver AGC and the RECEIVER's own input-referred noise. A receiver normalises before it
      equalises, so a modelled chain with no AGC carries an absolute level that means something
      it would not mean on hardware; and the library modelled the SCOPE's noise but not the
      receiver's, which enters at a different point in the chain.
U-13  duty-cycle distortion as a direct knob, in ps, with the standard definition.

Every number below is a closed form or a measurement on the op's own output. The two gates
observed FAILING are `test_a_rational_rate_change_cannot_slip` and
`test_the_indirect_dcd_route_still_fails_this`.
"""
import numpy as np
import pytest

from wfmsynth import instrument as INST
from wfmsynth import rx as RX
from wfmsynth.compose import Signal, _EXEC, _dfe_core, dfe_decisions
from wfmsynth.grid import Grid
from wfmsynth.streams import Streams

FS, BAUD = 256e9, 16e9
SPB = int(FS / BAUD)
UI_S = 1.0 / BAUD


def _grid(n_ui):
    return Grid(fs=FS, baud=BAUD, n=n_ui * SPB, v_full=0.8)


def _tone(n, f_hz, phase=0.371):
    return np.cos(2 * np.pi * f_hz * np.arange(n) / FS + phase)


def _tone_omega(y, guard=256):
    """The EXACT radian frequency of a noiseless real sinusoid, with no FFT bin quantisation:
    every pure tone obeys ``y[k+1] + y[k-1] = 2*cos(w)*y[k]``, so one least-squares solve for
    ``2*cos(w)`` recovers w to machine precision. Bin interpolation could not resolve a ppm."""
    y = np.asarray(y, float)[guard:len(y) - guard]
    a, b = y[1:-1], y[2:] + y[:-2]
    return float(np.arccos(np.clip(float(np.dot(a, b) / np.dot(a, a)) / 2.0, -1.0, 1.0)))


def _realised_ppm(x_in, y_out):
    """The sample-rate offset a record was taken on, read off the record itself."""
    return (_tone_omega(x_in) / _tone_omega(y_out) - 1.0) * 1e6


def _crossings(x, thresh=0.0):
    """Sub-sample threshold crossings and their polarity. Two estimator choices, both of which
    cost a factor of ten if made the other way:

      * THE THRESHOLD IS THE CONSTRUCTED ONE (0.0), not an estimate off the record. Every record
        measured in this file is a symmetric +/-1 NRZ source whose mid-reference level is
        therefore exactly 0. Estimating it makes the measurement, not the impairment, dominate:
        a 1st/99th-percentile midpoint is not invariant under a time warp (the warp changes how
        many samples sit on each edge, so the percentiles move) and a min/max midpoint reads the
        overshoot rather than the level -- MEASURED, the DCD gate's error goes from 0.048 % on
        the constructed threshold to 0.59 % on the percentile one and 0.49 % on min/max.
      * the position comes from a local CUBIC through four samples (t as a cubic in level), not
        a straight line, whose curvature bias differs between the two records being compared."""
    x = np.asarray(x, float)
    s = x - float(thresh)
    up = (s[:-1] <= 0) & (s[1:] > 0)
    dn = (s[:-1] >= 0) & (s[1:] < 0)
    k = np.nonzero(up | dn)[0]
    k = k[(k >= 2) & (k < len(s) - 2)]
    pos = np.array([np.polyval(np.polyfit(s[i - 1:i + 3], np.arange(i - 1, i + 3, dtype=float), 3), 0.0)
                    for i in k])
    return pos, np.where(up[k], 1.0, -1.0)


def _ui_widths(x, thresh=0.0):
    """(mean high pulse width, mean low pulse width) in seconds — the two quantities DCD is
    defined as the difference of."""
    c, sg = _crossings(x, thresh)
    w = np.diff(c) / FS
    return float(w[sg[:-1] > 0].mean()), float(w[sg[:-1] < 0].mean())


def _clock(n_ui, seed=1, **kw):
    """An alternating 1010 source. DCD and slip are both measured against UI BOUNDARIES, and a
    PRBS has unequal mean high/low RUN lengths — PRBS7 on this grid carries a +1.216 ps
    high-minus-low width of its own, purely from pattern statistics — so the reference pattern
    has to be the one with no runs at all."""
    g = _grid(n_ui)
    return Signal(seed=seed, grid=g).carrier("nrz", n_ui=n_ui, pattern="clock", tr_frac=0.35,
                                             causal=True, **kw).waveform(), g


# ==========================================================================================
# U-09  an independent sampling clock
# ==========================================================================================

# f/fs -> the max absolute error `resample_at` makes reproducing a unit tone at an arbitrary
# fractional offset. MEASURED. The 0.386 row is the Kaiser passband edge for a 64-tap kernel and
# the 0.420 row is what is WRONG above it: 4.4 % of full scale, which is why `sample_clock`
# checks the input's out-of-band energy instead of assuming it away.
TONE_ACCURACY = {0.02: 3e-7, 0.10: 3e-7, 0.20: 4e-7, 0.30: 3e-7, 0.386: 2e-6}


@pytest.mark.parametrize("f_frac", sorted(TONE_ACCURACY))
def test_the_interpolator_reproduces_the_analytic_signal_at_arbitrary_phase(f_frac):
    """THE CONSTRUCTED ANSWER. A pure tone's value at a non-integer sample position is known in
    closed form, so bandlimited interpolation can be checked against it rather than against
    another resampler. This is the property `resample_poly` cannot offer at all: it can only
    place output samples on a rational sub-grid."""
    n = 1 << 15
    x = _tone(n, f_frac * FS)
    for delta in (0.5, 0.137, 0.9999):
        src = np.arange(n, dtype=float) + delta
        y = INST.resample_at(x, src)
        ref = np.cos(2 * np.pi * f_frac * FS * (np.arange(n) + delta) / FS + 0.371)
        err = float(np.max(np.abs(y[200:n - 200] - ref[200:n - 200])))
        assert err < TONE_ACCURACY[f_frac], f"f/fs={f_frac} delta={delta}: max err {err:.3e}"


def test_the_interpolator_is_wrong_above_its_passband_and_says_so():
    """A check never observed failing is not a check, and this one bounds the tool's honesty:
    ABOVE 0.386*fs the same interpolation is off by 4.4 % of full scale. `sample_clock` measures
    the input's out-of-band energy and refuses rather than obliging quietly."""
    n = 1 << 15
    x = _tone(n, 0.42 * FS)
    y = INST.resample_at(x, np.arange(n, dtype=float) + 0.5)
    ref = np.cos(2 * np.pi * 0.42 * FS * (np.arange(n) + 0.5) / FS + 0.371)
    err = float(np.max(np.abs(y[200:n - 200] - ref[200:n - 200])))
    assert err == pytest.approx(0.044, abs=0.004), f"{err:.4f}"
    assert INST.out_of_band_fraction(x) > 0.99
    with pytest.raises(ValueError, match="passband"):
        INST.sample_clock(x, Grid(fs=FS, n=n), ppm=25.0)
    INST.sample_clock(x, Grid(fs=FS, n=n), ppm=25.0, band_tol=None)     # the documented opt-out


PPM_CASES = [300.0, 25.0, -50.0, 1.0]


@pytest.mark.parametrize("ppm", PPM_CASES)
def test_a_stated_ppm_offset_produces_exactly_that_rate(ppm):
    """A STATED ppm OFFSET IS THE REALISED ONE, read back off the record. A tone at f goes in;
    a clock running ppm fast records it as a tone at f/(1+ppm*1e-6), and that ratio is
    recoverable to machine precision from the record alone. MEASURED errors, in ppm:
    +1.4e-06 (300), +1.6e-06 (25), -1.7e-06 (-50), +1.0e-06 (1)."""
    n = 1 << 17
    x = _tone(n, 0.05 * FS)
    y = INST.sample_clock(x, Grid(fs=FS, n=n), ppm=ppm)
    got = _realised_ppm(x, y)
    assert got == pytest.approx(ppm, abs=1e-4), f"asked {ppm} ppm, realised {got:.6f} ppm"


@pytest.mark.parametrize("ppm", PPM_CASES)
def test_a_stated_ppm_offset_slips_exactly_the_closed_form(ppm):
    """The same thing in the time domain, which is how it is felt: a feature at LINK sample i
    lands at RECORD index i + slip(i), slip(i) = ppm*1e-6 * i exactly
    (`instrument.clock_slip_samples`). Every one of ~2000 crossings of a clock pattern is
    checked, not just the last. MEASURED residual 3.2e-04 samples (1.2 fs at 256 GSa/s) at every
    ppm alike, i.e. it is the cubic crossing estimator's own bias and not the clock's."""
    x, g = _clock(2048)
    ci, _ = _crossings(x)
    y = INST.sample_clock(x, g, ppm=ppm)
    co, _ = _crossings(y)
    m = min(len(ci), len(co))
    slip = co[:m] - ci[:m]
    want = INST.clock_slip_samples(ci[:m], ppm=ppm)
    assert np.max(np.abs(slip - want)) < 2e-3, (
        f"ppm={ppm}: last slip {slip[-1]:.6f} vs closed form {want[-1]:.6f}, "
        f"max err {np.max(np.abs(slip - want)):.3e} samples over {m} crossings")


def test_the_stated_scale_of_the_missing_mechanism():
    """The number this unit exists for, in closed form: 300 ppm over 3 M UI at 16 samples/UI is
    14,400 samples of slip. The modelled scope's `n_out` path slips 0 (next test)."""
    n_samples = 3_000_000 * SPB
    assert INST.clock_slip_samples(n_samples, ppm=300.0) == pytest.approx(14400.0, abs=1e-6)


@pytest.mark.parametrize("drift", [5.0e7, -3.0e7])
def test_a_drifting_clock_slips_quadratically(drift):
    """An oscillator that is warming up has a ppm that MOVES, so the slip picks up a term
    quadratic in time: slip(i) = p*i + r*i**2/(2*fs). One integer rate ratio cannot express
    this at all — that is the structural half of why `resample_poly` is the wrong primitive,
    independent of any quantisation.

    THE DRIFT HERE IS DELIBERATELY UNPHYSICAL and the reason is worth stating: the quadratic
    term is ``r*N*T/2`` for a record of N samples and T seconds, so on this 4096 UI / 256 ns
    record a realistic few-ppm-per-second oscillator accumulates 1e-9 of a sample and cannot be
    observed at all. Drift is a mechanism of MILLISECOND captures. 5e7 ppm/s is what makes the
    term 0.42 samples here, big enough to separate from the linear part."""
    x, g = _clock(4096)
    ci, _ = _crossings(x)
    y = INST.sample_clock(x, g, ppm=10.0, drift_ppm_per_s=drift)
    co, _ = _crossings(y)
    m = min(len(ci), len(co))
    slip = co[:m] - ci[:m]
    want = INST.clock_slip_samples(ci[:m], ppm=10.0, drift_ppm_per_s=drift, fs=FS)
    lin = INST.clock_slip_samples(ci[:m], ppm=10.0)
    assert np.max(np.abs(slip - want)) < 2e-3, (
        f"drift={drift:g} ppm/s: max err {np.max(np.abs(slip - want)):.3e} samples")
    # and the quadratic term is actually present, i.e. the test is not passing on the linear part
    assert abs(want[-1] - lin[-1]) > 0.1, "the drift term is too small to be observed here"


def test_a_stated_starting_phase_shifts_every_feature_and_slips_nothing():
    """A real timebase has no reason to line up with the link's sample grid, and an INTEGER
    output length cannot express a fraction of a sample. A stated phase0 displaces every feature
    by exactly that and accumulates nothing."""
    x, g = _clock(2048)
    ci, _ = _crossings(x)
    for ph_samples in (0.37, -0.62):
        y = INST.sample_clock(x, g, ppm=0.0, phase0_s=ph_samples / FS, span="fit")
        co, _ = _crossings(y)
        m = min(len(ci), len(co))
        shift = co[:m] - ci[:m]
        assert np.allclose(shift, -ph_samples, atol=2e-3), (
            f"phase0={ph_samples} samples: realised {shift.mean():.6f} +/- {shift.std():.1e}")
        assert shift.std() < 1e-3, "a starting phase must not accumulate"


def test_the_default_span_never_extrapolates_and_strict_refuses_to_fabricate():
    """A clock offset changes the span of link time a record covers, so a record of the input's
    length is not always available. The default fits the record to what the input supports;
    `span='strict'` keeps the length and RAISES rather than holding the last sample to fill it."""
    x, g = _clock(512)
    n = len(x)
    short = INST.sample_clock(x, g, ppm=-200.0)
    assert len(short) == int(np.floor((n - 1) * (1 - 200e-6))) + 1 < n
    with pytest.raises(ValueError, match="short"):
        INST.sample_clock(x, g, ppm=-200.0, span="strict")
    assert len(INST.sample_clock(x, g, ppm=+200.0)) == n     # a fast clock needs less input


# ---------------------------------------------------------------- THE GATE OBSERVED FAILING
# What the modelled scope realises, MEASURED, on a 1<<17-sample record with a 0.05*fs tone.
# `n_out` is an integer, so the realisable offsets are quantised to 1/n = 7.6294 ppm here.
RESAMPLE_POLY_PPM = {300.0: 297.544884, 25.0: 22.888045}


@pytest.mark.parametrize("ppm", sorted(RESAMPLE_POLY_PPM))
def test_a_rational_rate_change_cannot_slip(ppm):
    """THE GATE OBSERVED FAILING. `_op_digitize` is the only rate change in the library and it
    is `resample_poly`, a RATIONAL one. Two measured consequences:

      * with no `n_out` — the default, and what every chain in the library does — the op does
        not call it at all, so the record is BIT-IDENTICAL to the link's own samples and the
        realised offset is exactly 0.000000 ppm. There is no independent clock.
      * with `n_out=round(n*(1+ppm*1e-6))` the offset exists but is QUANTISED to whole output
        samples: 1/131072 = 7.6294 ppm on this record. 300 ppm comes back as 297.5449 and 25 as
        22.8880 — errors of 2.455 and 2.112 ppm, i.e. 0.82 % and 8.4 % of the request, and the
        error changes if you change the record length.

    A drifting clock and a sub-sample starting phase are not expressible at either setting."""
    n = 1 << 17
    x = _tone(n, 0.05 * FS)
    g = Grid(fs=FS, n=n)
    st = Streams(0)
    locked = _EXEC["digitize"](x, {"op": "digitize"}, st, g, 0)
    assert np.array_equal(np.asarray(locked, float), x), "the default path must be the identity"
    assert _realised_ppm(x, locked) == pytest.approx(0.0, abs=1e-6)

    n_out = int(round(n * (1 + ppm * 1e-6)))
    rational = _EXEC["digitize"](x, {"op": "digitize", "n_out": n_out}, st, g, 0)
    got = _realised_ppm(x, rational)
    assert got == pytest.approx(RESAMPLE_POLY_PPM[ppm], abs=5e-4), f"{got:.6f}"
    assert abs(got - ppm) > 2.0, "the quantisation error is the point"
    with pytest.raises(AssertionError):                 # the same assertion, on that path
        assert got == pytest.approx(ppm, abs=1e-4)
    # and `sample_clock` gets it right on the same record, to 1e-4 ppm
    assert _realised_ppm(x, INST.sample_clock(x, g, ppm=ppm)) == pytest.approx(ppm, abs=1e-4)


# ==========================================================================================
# U-12  receiver AGC and input-referred noise
# ==========================================================================================

@pytest.mark.parametrize("metric", RX.AGC_METRICS)
@pytest.mark.parametrize("in_level", [0.004, 1.0, 37.0])
@pytest.mark.parametrize("target", [0.5, 1.0])
def test_the_agc_brings_a_known_input_level_to_a_known_output_level(metric, in_level, target):
    """AN AGC BRINGS A KNOWN INPUT LEVEL TO A KNOWN OUTPUT LEVEL — the whole analytic content of
    the stage, over four orders of magnitude of input level and all four detector metrics.
    MEASURED exact to 1.1e-16 (a float rounding) on every combination."""
    rng = np.random.default_rng(4)
    x = (rng.standard_normal(1 << 14) * 0.37 + 0.1) * in_level
    y = RX.agc(x, target=target, metric=metric)
    assert RX.agc_level(y, metric) == pytest.approx(target, rel=1e-12)


def test_the_agc_is_what_makes_a_stated_full_scale_downstream_mean_anything():
    """WHY THE STAGE EXISTS. `dfe(scale=...)` is a stated full scale; without an AGC the level
    reaching it is whatever the chain happened to produce, so a change anywhere upstream
    silently redefines every decision threshold. MEASURED on a 2048 UI PAM4 record through 8 dB
    of loss, comparing the DFE's decisions at nominal drive against 10x and 0.1x drive:

      no AGC, scale=1.0    symbol disagreement 86.57 % (10x) and 32.54 % (0.1x)
      AGC then scale=1.0   symbol disagreement  0.00 % and  0.00 %

    THE CONSTELLATION HAS TO HAVE MORE THAN TWO LEVELS FOR THIS TO BE VISIBLE, and that is worth
    recording because the first version of this test measured nothing. On NRZ with levels
    [-1, +1] the slicer is a sign test: scale the record by 10 and every decision is unchanged,
    0.00 % disagreement with no AGC at all. A two-level slicer is level-INDEPENDENT, so it hides
    exactly the defect. PAM4's inner levels at +/-1/3 are what a wrong full scale destroys.

    An AGC's job is to delete the absolute level, and this is that deletion being observed."""
    n_ui = 2048
    g = _grid(n_ui)
    dfe_p = {"taps": [0.18, 0.06], "scale": 1.0}          # levels default to PAM4

    def decisions(drive, with_agc):
        x = (Signal(seed=1, grid=g).carrier("pam4", n_ui=n_ui, tr_frac=0.35, causal=True)
             .lossy(causal=True, loss_db=8.0, loss_at_ghz=8.0).waveform()) * drive
        if with_agc:
            x = RX.agc(x, target=1.0, metric="p99")
        return dfe_decisions(x, dfe_p, g)[2]

    for with_agc in (False, True):
        ref = decisions(1.0, with_agc)
        for drive in (10.0, 0.1):
            bad = float(np.mean(decisions(drive, with_agc) != ref))
            # re-measured after the edge shaper was corrected; the shape of the claim is
            # unchanged -- without AGC most decisions move, with it none do
            want = {(False, 10.0): 0.857841, (False, 0.1): 0.338544}.get((with_agc, drive), 0.0)
            assert bad == pytest.approx(want, abs=5e-4), (
                f"{'AGC' if with_agc else 'no-AGC'} chain, drive {drive}x: {bad:.4%} "
                f"disagreement, expected {want:.4%}")


def test_a_tracking_agc_settles_to_the_target_and_follows_a_level_that_moves():
    """A block AGC hits the target exactly because it sees the whole record; real hardware runs
    a loop and therefore lags. The assertion is the one a loop can actually make: after 8 time
    constants the local level is the target, and a level STEP is followed rather than averaged
    away (which is the reason to want the loop at all)."""
    g = Grid(fs=FS, baud=BAUD, n=1 << 18, v_full=0.8)
    rng = np.random.default_rng(7)
    x = rng.standard_normal(g.n) * 0.05                       # a small input: needs gain 10
    tau = 200e-12
    y = RX.agc(x, target=0.5, metric="rms", tau_s=tau, grid=g)
    k = int(8 * tau * FS)
    settled = float(np.sqrt(np.mean(y[k:] ** 2)))
    assert settled == pytest.approx(0.5, rel=0.05), f"settled rms {settled:.5f}"
    x2 = x.copy(); x2[g.n // 2:] *= 8.0                       # +18 dB step mid-record
    y2 = RX.agc(x2, target=0.5, metric="rms", tau_s=tau, grid=g)
    tail = float(np.sqrt(np.mean(y2[g.n // 2 + k:] ** 2)))
    assert tail == pytest.approx(0.5, rel=0.05), f"after an 18 dB step: {tail:.5f}"
    frozen = x2 * RX.agc_gain(x, target=0.5, metric="rms", tau_s=tau, grid=g)
    assert float(np.sqrt(np.mean(frozen[g.n // 2 + k:] ** 2))) > 3.0   # what NOT following does


def test_the_agc_refuses_a_target_its_stated_range_cannot_reach():
    """A finite AGC range that cannot reach the target is the `ac_couple` defect again: clamp
    silently and the record sits below its stated level with nothing said. It raises, and names
    the gain it would have needed; `on_limit='clip'` is the explicit opt-out."""
    x = np.random.default_rng(1).standard_normal(4096) * 1e-3
    with pytest.raises(ValueError, match="needs gain"):
        RX.agc(x, target=1.0, metric="rms", gain_limits=(0.1, 10.0))
    y = RX.agc(x, target=1.0, metric="rms", gain_limits=(0.1, 10.0), on_limit="clip")
    assert RX.agc_level(y, "rms") == pytest.approx(1e-2, rel=0.05)      # saturated, not target
    assert RX.agc_level(RX.agc(x, target=1.0, gain_limits=(0.1, 1e6)), "rms") == \
        pytest.approx(1.0, rel=1e-12)


@pytest.mark.parametrize("rms", [1e-4, 3.7e-3])
def test_the_receiver_noise_realises_the_rms_it_states(rms):
    """A STATED rms IS THE REALISED rms, exactly — the draw is renormalised, the convention
    `physics.inject_jitter` already uses for its Rj, so the level is assertable rather than
    approximately right."""
    g = Grid(fs=FS, n=1 << 16)
    y = RX.input_noise(np.zeros(g.n), grid=g, rms=rms, rng=np.random.default_rng(2))
    assert float(np.sqrt(np.mean(y ** 2))) == pytest.approx(rms, rel=1e-12)


def test_the_receiver_noise_realises_the_spectral_density_it_states():
    """Input-referred noise is quoted as a DENSITY over the receiver's noise bandwidth, not as a
    number of volts, and a receiver front end is not white to the sampler's Nyquist. Asked for
    2 nV/sqrt(Hz) over 20 GHz: realised rms 0.282843 (closed form 2e-6*sqrt(20e9) = 0.2828427),
    realised in-band one-sided PSD 4.008e-12 against 4.000e-12 asked, and 1.5e-42 above the
    band — zero."""
    g = Grid(fs=FS, n=1 << 16)
    dens, bw = 2e-6, 20e9
    y = RX.input_noise(np.zeros(g.n), grid=g, density=dens, bw_hz=bw,
                       rng=np.random.default_rng(1))
    assert float(np.sqrt(np.mean(y ** 2))) == pytest.approx(dens * np.sqrt(bw), rel=1e-12)
    psd = np.abs(np.fft.rfft(y)) ** 2 * 2.0 / (g.n * g.fs)
    f = np.fft.rfftfreq(g.n, d=1.0 / g.fs)
    assert float(psd[(f > 0.05 * bw) & (f < 0.95 * bw)].mean()) == pytest.approx(dens ** 2, rel=0.02)
    assert float(psd[f > 1.05 * bw].max()) < (dens ** 2) * 1e-12
    with pytest.raises(ValueError):
        RX.input_noise(np.zeros(64), grid=g, rms=1e-3, density=1e-6, bw_hz=bw)
    with pytest.raises(ValueError, match="Nyquist"):
        RX.input_noise(np.zeros(g.n), grid=g, rms=1e-3, bw_hz=0.6 * g.fs)


def test_receiver_noise_is_not_the_instrument_noise_floor():
    """The two noises are DIFFERENT MECHANISMS AT DIFFERENT POINTS, and the difference is
    measurable, not a naming preference. The receiver's noise is input-referred: it arrives
    BEFORE the CTLE, whose high-frequency peaking amplifies it. The instrument's floor is added
    at the digitizer, after everything. MEASURED with the same 5 mV rms through the same
    CTLE (fz 2 / fp1 8 / fp2 16 GHz, dc_gain 0.65) on an otherwise empty record: placed at the
    receiver input, a floor band-limited to the receiver's own 20 GHz lands +3.671 dB more noise
    at the slicer than the same number placed after the equaliser.

    AND THE BANDWIDTH IS PART OF THE MECHANISM, which is why `input_noise` takes it. The same
    5 mV spread white to the SAMPLER's 128 GHz Nyquist comes out -1.576 dB — the CTLE
    ATTENUATES it, because most of a white floor's energy sits above the peak and past the
    second pole. A receiver's noise quoted without its noise bandwidth gets the sign wrong."""
    g = Grid(fs=FS, baud=BAUD, n=1 << 16, v_full=0.8)
    rms = 5e-3
    st = Streams(11)
    ctle = dict(op="ctle", fz_ghz=2.0, fp1_ghz=8.0, fp2_ghz=16.0, dc_gain=0.65)
    n_rx = RX.input_noise(np.zeros(g.n), grid=g, rms=rms, bw_hz=20e9,
                          rng=np.random.default_rng(5))
    db = 20 * np.log10(np.std(_EXEC["ctle"](n_rx, ctle, st, g, 0)) / np.std(n_rx))
    assert db == pytest.approx(3.671, abs=0.02), f"{db:.3f} dB"
    n_white = RX.input_noise(np.zeros(g.n), grid=g, rms=rms, rng=np.random.default_rng(5))
    db_white = 20 * np.log10(np.std(_EXEC["ctle"](n_white, ctle, st, g, 0)) / np.std(n_white))
    assert db_white == pytest.approx(-1.576, abs=0.02), f"{db_white:.3f} dB"
    assert db - db_white > 5.0, "the noise bandwidth is part of the mechanism, not a detail"


# ==========================================================================================
# U-13  duty-cycle distortion as a direct knob
# ==========================================================================================

DCD_CASES = [0.125, 1.0, 4.0, -2.5]


@pytest.mark.parametrize("dcd_ps", DCD_CASES)
def test_a_stated_dcd_in_ps_is_the_high_minus_low_ui_width(dcd_ps):
    """THE DEFINITION, ASSERTED. DCD is ``mean(high pulse width) - mean(low pulse width)``, and
    that is what `dcd(ps=)` realises — MEASURED +0.125384 / +1.002991 / +4.007498 / -2.506490 ps
    for the four requests. Read off threshold crossings of the op's own output.

    The realisation carries a +0.3 % scale error, which is 3 fs on a 1 ps request and two orders
    below anything an instrument resolves. Its origin is the mechanism rather than the
    interpolator: the displacement is ``sign(gradient) * dcd/2``, which is zero on the flats and
    ±dcd/2 on the edges, so the shift field STEPS at each edge boundary and resampling across a
    discontinuous shift field is not a pure delay. It read +0.055 % while the edge shaper
    delivered an edge twice as wide as asked for, which made that step a smaller fraction of the
    transition. A mechanism that shifted each half-period as a whole, putting the step where the
    waveform is flat, would not have the error at all.

    `test_the_dcd_scale_error_is_linear_in_the_request` is what keeps this honest: a bounded
    scale error is tolerable, a mechanism that has stopped being proportional is not."""
    x, g = _clock(1024)
    hi0, lo0 = _ui_widths(x)
    assert abs(hi0 - lo0) < 2e-17, f"the reference pattern is not symmetric: {hi0 - lo0:.3e} s"
    y = _EXEC["dcd"](x, {"op": "dcd", "ps": dcd_ps}, Streams(0), g, 0)
    hi, lo = _ui_widths(y)
    got_ps = (hi - lo) * 1e12
    assert got_ps == pytest.approx(dcd_ps, rel=5e-3), f"asked {dcd_ps} ps, realised {got_ps:.8f}"


@pytest.mark.parametrize("scale", [2, 4, 8, 32])
def test_the_dcd_scale_error_is_linear_in_the_request(scale):
    """The realisation is 0.3 % high. That is tolerable only while it stays a SCALE error: a
    mechanism that has started to saturate, or to depend on where the edge sits between samples,
    shows up as a ratio that moves with the request. Pinned over a 32x range."""
    x, g = _clock(1024)
    base = 0.125
    out = {}
    for ps in (base, base * scale):
        y = _EXEC["dcd"](x, {"op": "dcd", "ps": ps}, Streams(0), g, 0)
        hi, lo = _ui_widths(y)
        out[ps] = (hi - lo) * 1e12 / ps          # realised / requested
    assert out[base * scale] == pytest.approx(out[base], rel=2e-3), (
        f"ratio moved from {out[base]:.6f} to {out[base*scale]:.6f} over {scale}x")
    assert 1.0 < out[base] < 1.006, f"scale error {100*(out[base]-1):.3f} % is outside its bound"


def test_dcd_is_stateable_as_a_fraction_of_a_ui_and_only_one_way_at_a_time():
    """`frac_ui=` is the other unit DCD is quoted in. 6.4 % UI at 16 GBd is 4 ps, and the two
    spellings must land on the same record."""
    x, g = _clock(1024)
    a = _EXEC["dcd"](x, {"op": "dcd", "ps": 4.0}, Streams(0), g, 0)
    b = _EXEC["dcd"](x, {"op": "dcd", "frac_ui": 4.0e-12 * BAUD}, Streams(0), g, 0)
    assert np.array_equal(a, b)
    with pytest.raises(ValueError, match="exactly one"):
        _EXEC["dcd"](x, {"op": "dcd", "ps": 4.0, "frac_ui": 0.064}, Streams(0), g, 0)
    with pytest.raises(ValueError, match="baud"):
        _EXEC["dcd"](x, {"op": "dcd", "ps": 4.0}, Streams(0), Grid(fs=FS, n=len(x)), 0)


def test_zero_dcd_is_the_interpolators_identity():
    """Asking for no impairment must not quietly apply the resampler's error budget as one.

    EXACTLY identity, not merely close: the op returns the input untouched when the displacement
    is all zero, so there is no error budget to apply."""
    x, g = _clock(1024)
    y = _EXEC["dcd"](x, {"op": "dcd", "ps": 0.0}, Streams(0), g, 0)
    assert float(np.max(np.abs(y - x))) == 0.0
    hi, lo = _ui_widths(y)
    # the pattern's own asymmetry, an order of magnitude smaller than it was because the edge
    # shaper now delivers the rise time it is asked for
    assert abs(hi - lo) * 1e12 == pytest.approx(8.67e-7, rel=0.05)


# ---------------------------------------------------------------- THE GATE OBSERVED FAILING
# What `carrier(..., jitter=dict(dcd=<samples>))` — the only route to DCD before this — realises
# for mean(high)-mean(low), MEASURED on a 1024 UI clock pattern at fs=256e9, baud=16e9.
INDIRECT_DCD_PS = {0.256: -7.835911, 1.024: -7.835911, 2.0: -15.673202, 4.0: -31.331451}


@pytest.mark.parametrize("dcd_samples", sorted(INDIRECT_DCD_PS))
def test_the_indirect_dcd_route_still_fails_this(dcd_samples):
    """THE GATE OBSERVED FAILING, on the mechanism that was already there. `physics._edge_disp`
    displaces rising edges by +dcd/2 and falling by -dcd/2, which makes the HIGH pulse SHORTER
    by 2*dcd — so the realised DCD is about -2x the request, wrong in sign and in scale. And
    `_place_symbols` places the displaced edges with `searchsorted` on the INTEGER sample grid,
    so 0.256 samples (1 ps) and 1.024 samples (4 ps) produce the same -7.835911 ps: every
    sub-sample DCD, which is all of them, is unreachable.

    That code is in `physics.py`, which this unit does not own — the numbers are logged in
    BACKLOG.md. It stays reachable and measured here rather than being described."""
    x0, g = _clock(1024)
    y, _ = _clock(1024, jitter=dict(dcd=dcd_samples))
    hi, lo = _ui_widths(y)
    got = (hi - lo) * 1e12
    assert got == pytest.approx(INDIRECT_DCD_PS[dcd_samples], abs=2e-3), f"{got:.6f} ps"
    stated_ps = dcd_samples / FS * 1e12
    assert got / stated_ps < -1.9, f"ratio to stated {got / stated_ps:+.4f}"
    with pytest.raises(AssertionError):                       # the definition test, on that path
        assert got == pytest.approx(stated_ps, rel=2e-3)


def test_the_indirect_route_quantises_sub_sample_dcd_to_a_bit_identical_record():
    """The quantisation, on the bytes rather than on a statistic: two DCD requests a factor of
    four apart come out of the source as the SAME waveform, sample for sample."""
    a, _ = _clock(1024, jitter=dict(dcd=0.256))
    b, _ = _clock(1024, jitter=dict(dcd=1.024))
    assert np.array_equal(a, b), "if this ever fails the sub-sample quantisation was fixed"
    d, g = _clock(1024)
    p = _EXEC["dcd"](d, {"op": "dcd", "ps": 1.0}, Streams(0), g, 0)
    q = _EXEC["dcd"](d, {"op": "dcd", "ps": 4.0}, Streams(0), g, 0)
    assert not np.array_equal(p, q) and float(np.max(np.abs(p - q))) > 0.04


# ==========================================================================================
# where the three stages sit in a chain that renders a LEAD-IN
# ==========================================================================================

def test_a_block_agc_ranges_to_the_delivered_window_and_not_to_the_guard():
    """THE SHARP CASE OF THE LEAD-IN RULE, and a gate observed failing on the wrong version.

    `Signal.lead_in` renders a guard before the record and throws it away, because the guard is
    where the turn-on lives and every stage with memory answers that step. `digitize` and
    `store` already range their VERTICAL to the delivered window for this reason. A block AGC is
    the same problem one step worse: it does not merely allocate codes, it MULTIPLIES the whole
    record, so a gain set from the guard puts the delivered record at the wrong absolute level
    and every stated full scale downstream — `dfe(scale=)`, `levels=` — is then wrong by that
    factor with nothing said.

    MEASURED on a 2048 UI record through `ac_couple(fc_hz=0.2e9)` with a 4096-sample guard: the
    extended render peaks at 1.7817 (the turn-on's baseline-wander transient, in the guard)
    while the delivered window peaks at 1.2256. So a guard-ranged AGC asked for a peak of 1.0
    delivers 0.6879 — 3.249 dB below its own stated full scale."""
    import warnings
    import wfmsynth.compose as C
    g = _grid(2048)

    def chain():
        return (Signal(seed=3, grid=g, lead_in=4096, lead_out=4096)
                .carrier("nrz", n_ui=2048, tr_frac=0.35, causal=True)
                .ac_couple(fc_hz=0.2e9).agc(target=1.0, metric="peak"))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        no_agc = (Signal(seed=3, grid=g, lead_in=4096, lead_out=4096)
                  .carrier("nrz", n_ui=2048, tr_frac=0.35, causal=True)
                  .ac_couple(fc_hz=0.2e9))
        ext, plan = no_agc.rendered_lead()
        ext = np.asarray(ext, float)
        assert float(np.max(np.abs(ext))) == pytest.approx(1.7784, abs=2e-3)
        assert float(np.max(np.abs(ext[plan.window]))) == pytest.approx(1.2332, abs=2e-3)

        assert "agc" in C._LEAD_WINDOW_RANGED
        got = float(np.max(np.abs(chain().waveform())))
        assert got == pytest.approx(1.0, rel=1e-12), f"window-ranged AGC delivered {got:.6f}"

        orig = C._op_agc                       # the same stage, ranged to the guard instead
        try:
            C._EXEC["agc"] = lambda x, p, st, gr, i, win=None: orig(x, p, st, gr, i, win=None)
            bad = float(np.max(np.abs(chain().waveform())))
        finally:
            C._EXEC["agc"] = orig
    assert bad == pytest.approx(0.6934, abs=2e-3), f"{bad:.6f}"
    assert 20 * np.log10(bad) == pytest.approx(-3.180, abs=0.02)
    with pytest.raises(AssertionError):
        assert bad == pytest.approx(1.0, rel=1e-12)


def test_an_independent_clock_cannot_be_rendered_inside_a_lead_in():
    """Refusing is the honest answer, and it is the same refusal `acquire` and
    `digitize(n_out=)` already get: a clock with a frequency offset covers a DIFFERENT SPAN of
    link time than the link's own grid, so the guard's sample count is not the record's and the
    accumulated slip moves the window's own boundaries. Sample the clock after the window is
    sliced out, or render the whole thing on the clock's grid."""
    import wfmsynth.compose as C
    g = _grid(512)
    assert "sample_clock" in C._LEAD_REJECT
    s = (Signal(seed=3, grid=g, lead_in=1024)
         .carrier("nrz", n_ui=512, tr_frac=0.35, causal=True).sample_clock(ppm=25.0))
    with pytest.raises(ValueError, match="cannot be rendered with a lead-in"):
        s.waveform()
    # the two stages that CAN: a sub-UI warp and an additive floor
    ok = (Signal(seed=3, grid=g, lead_in=1024)
          .carrier("nrz", n_ui=512, tr_frac=0.35, causal=True)
          .dcd(ps=2.0).rx_noise(rms=0.003))
    assert len(ok.waveform()) == g.n


def test_the_model_reproduces_the_offset_the_real_captures_carry():
    """GROUNDED IN REAL RECORDS, not in a plausible-looking number. Three real captures (40
    GSa/s, 8 M samples = 200 us each, three different symbol rates) carry a -1.987 / -1.822 /
    -1.983 ppm offset between the link's symbol clock and the record's timebase, fitted from
    163k-272k threshold crossings each; see the block comment in `instrument.py`. On capture A
    it is stable across ten segments (-1.751 to -2.435 ppm), so there is no drift to model over
    200 us — only an offset.

    Held here: the model's convention reproduces that magnitude, ~15.9 samples of slip over an
    8 M-sample record, and does it in the direction the measurement has. The captures live in a
    separate private tree, so what is asserted here is the arithmetic and the realised
    behaviour, not the file."""
    n_real = 8_000_000
    for ppm, slip in ((1.987, 15.896), (1.822, 14.576), (1.983, 15.864)):
        assert INST.clock_slip_samples(n_real, ppm=ppm) == pytest.approx(slip, abs=0.01)
    # and the realised behaviour at that magnitude, on a record short enough to render
    x, g = _clock(2048)
    ci, _ = _crossings(x)
    y = INST.sample_clock(x, g, ppm=1.987)
    co, _ = _crossings(y)
    m = min(len(ci), len(co))
    want = INST.clock_slip_samples(ci[:m], ppm=1.987)
    assert np.max(np.abs((co[:m] - ci[:m]) - want)) < 2e-3
    assert _realised_ppm(_tone(1 << 17, 0.05 * FS),
                         INST.sample_clock(_tone(1 << 17, 0.05 * FS), Grid(fs=FS, n=1 << 17),
                                           ppm=1.987)) == pytest.approx(1.987, abs=1e-4)
