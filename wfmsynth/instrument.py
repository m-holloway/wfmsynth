"""
wfmsynth.instrument — front-end / digitizer models. A real high-speed scope is a
TIME-INTERLEAVED ADC: M sub-ADC cores sample in round-robin, and the mismatch between
them is much of what makes a measured capture look measured. Perfectly white noise on a
perfectly uniform grid is exactly the "too clean" signature a model trained on synthetic
learns to rely on and then breaks on real data.

`interleave_adc` injects per-core **gain**, **offset** and **timing-skew** mismatch,
which produce the characteristic interleave spurs: offset mismatch -> tones at k*fs/M
(input-independent); gain/skew mismatch -> image tones mirrored about fs/(2M), i.e. at
k*fs/M +/- f_in. Set all mismatch to zero for an ideal ADC (no spurs). numpy/scipy only.
"""
from __future__ import annotations
import math

import numpy as np
from scipy import signal as _sig
from scipy.special import i0 as _i0

from . import physics as _P


def interleave_adc(x, m_cores=4, gain_mm=0.0, offset_mm=0.0, skew_mm=0.0,
                   rng=None, mismatch=None, offset_v=None):
    """Pass x through an M-way time-interleaved ADC with per-core mismatch.

      gain_mm    per-core gain error, std (fractional, e.g. 0.01 = 1%)
      offset_mm  per-core offset, std as a fraction of the signal span
      offset_v   per-core offset, std in ABSOLUTE amplitude units (volts). A real
                 converter's offset error is a property of the ADC, not the signal: it
                 stays put when the signal shrinks (which is when it matters most).
                 Takes precedence over offset_mm when given.
      skew_mm    per-core sampling-time skew, std in SAMPLES (sub-sample)

    Cores are assigned round-robin: sample i is taken by core i % m_cores. Draw the M
    per-core errors once (seed via rng), or pass `mismatch=(g, o, s)` explicitly for a
    reproducible fixed pattern. Returns the digitized array (same length)."""
    x = np.asarray(x, float)
    n = len(x)
    span = np.ptp(x) + 1e-9
    core = np.arange(n) % m_cores
    if mismatch is not None:
        g, o, s = (np.asarray(v, float) for v in mismatch)
    else:
        rng = rng or np.random.default_rng()
        g = rng.normal(0.0, gain_mm, m_cores)
        o_std = offset_v if offset_v is not None else offset_mm * span   # absolute vs fraction
        o = rng.normal(0.0, o_std, m_cores)
        s = rng.normal(0.0, skew_mm, m_cores)
    y = x.copy()
    if skew_mm > 0 or (mismatch is not None and np.any(s)):
        xi = np.arange(n)                          # sub-sample timing skew per core
        for c in range(m_cores):
            idx = np.where(core == c)[0]
            y[idx] = np.interp(idx + s[c], xi, x)
    y = y * (1.0 + g[core]) + o[core]              # per-core gain + offset
    return y


# ------------------------------------------------------------------ analog vs digital
# THE DISTINCTION THIS MODULE NOW MAKES EXPLICIT, because getting it wrong halves every
# stated bandwidth in the library. A band limit sits in one of two places in an instrument,
# and the two are not the same operator:
#
#   ANALOG, BEFORE the converter -- the front end, the probe's own roll-off, an RC load. It is
#   a physical network: causal, single-pass, and it DELAYS. Its response cannot be symmetric
#   about t=0 because nothing in it knows the future.
#
#   DIGITAL, AFTER the converter -- the instrument's selected-bandwidth filter, which is a DSP
#   stage operating on a stored record and can therefore look forwards. Zero phase is not a
#   defect there, it is what the instrument does: MEASURED on three real exports from a
#   high-bandwidth real-time sampling oscilloscope, the record's noise floor drops 38 dB across
#   2 GHz at the corner, with no group delay to be found.
#
# Every analog kind here defaults to `causal=True` and every digital kind is zero-phase, and
# asking for the wrong one raises rather than silently obliging.
ANALOG_KINDS = ("bessel", "gaussian")
DIGITAL_KINDS = ("brickwall",)


def _gaussian_mag(n, wn):
    """Gaussian |H| on the rfft grid of `n` samples, -3 dB EXACTLY at `wn` (of Nyquist).

    ``exp(-0.5*(f/sigma)**2) = 1/sqrt(2)`` at ``f = sigma*sqrt(ln 2)``, so a Gaussian whose
    -3 dB point is the requested corner has ``sigma = fc/sqrt(ln 2)`` -- i.e. the exponent is
    ``-0.5*ln(2)*(f/fc)**2``. The legacy branch used ``sigma = fc``, which puts its own -3 dB
    at ``0.8326*fc``: a 32 GHz request realised at 26.6 GHz before the double pass, 22.2 after.
    """
    f = np.fft.rfftfreq(n) / (wn / 2.0 + 1e-12)          # f in units of the requested corner
    return np.exp(-0.5 * np.log(2.0) * f ** 2)


def scope_bandwidth(x, grid, bw_hz, kind="bessel", order=4, causal=None):
    """A bandwidth limit. ``kind='bessel'`` (flat group delay) or ``'gaussian'`` model the
    ANALOG front end, which is band-limited by physics and rolls off gently. ``'brickwall'``
    models the instrument's DIGITAL selected-bandwidth filter, which sits after the converter
    and is not gentle at all: MEASURED on three real exports from a high-bandwidth real-time sampling oscilloscope,
    the record's noise
    floor drops **38 dB across 2 GHz** at the corner and is flat on both sides of it. Use the
    analog kinds before the converter and ``'brickwall'`` after it.

    THE STATED CORNER IS NOW THE REALISED CORNER, WHICH IT WAS NOT
    -------------------------------------------------------------
    ``causal=True`` (the DEFAULT for every analog kind) applies the band limit ONCE, forwards.
    ``causal=False`` is the legacy zero-phase form kept as an explicit opt-out, and it is wrong
    in two independent ways at the same time -- both MEASURED, at ``bw_hz=32e9``,
    ``kind='bessel'``, ``order=4``, ``grid.fs=256e9``:

      * it ran the filter forwards AND backwards (`sosfiltfilt`), so the magnitude was the
        design SQUARED: |H| = 0.7039 at 16 GHz and 0.1747 (-15.1 dB) at 32 GHz. A front end
        asked for 32 GHz behaved like a **16.0 GHz** one.
      * it designed the Bessel with scipy's default ``norm='phase'``, whose -3 dB point is at
        0.68 of ``Wn`` even in a single pass -- 21.76 GHz for a 32 GHz request.

    The causal path designs with ``norm='mag'``, which puts -3 dB exactly at ``bw_hz``
    (measured 32.000 GHz, |H| = 0.7071), and runs one forward pass, so the stage also has the
    group delay a physical front end has: **9.97 ps** at DC here, against the analog Bessel-4
    closed form ``2.1139/(2*pi*32 GHz) = 10.51 ps`` (the 0.54 ps is the bilinear warp at
    fc/Nyquist = 0.25) and against **+0.000 ps** for the zero-phase form.

    The forward filter is initialised to STEADY STATE at ``x[0]`` (`sosfilt_zi`), not to zero:
    with zero state the output starts at 0 whatever the signal is, planting a full-scale
    settling transient at the head of every record. ONE CONSEQUENCE IS WORTH NAMING, because it
    is measurable and it is not an improvement: fed a unit impulse at index 0, this stage is
    primed to the DC steady state of that sample, so what comes out is the impulse response
    PLUS a step-down settling, and `compose._lead_extent` -- which probes a chain's memory with
    exactly that impulse -- therefore over-measures the front end's extent. MEASURED on the
    shipped chain, the lead-in plan moves by at most one UI quantum (2048 samples of a 1 M
    render) and in the conservative direction, so it is left as it is.

    ``causal`` may not be set on a digital kind -- ``'brickwall'`` IS zero-phase, that is the
    point of the distinction, and quietly accepting ``causal=True`` there would make the flag
    a lie."""
    x = np.asarray(x, float)
    if kind in DIGITAL_KINDS:
        if causal:
            raise ValueError(
                f"kind={kind!r} is the instrument's DIGITAL selected-bandwidth filter, which "
                f"sits AFTER the converter and is legitimately zero-phase; there is no causal "
                f"form of it. causal= applies to the analog kinds {ANALOG_KINDS}.")
    elif kind not in ANALOG_KINDS:
        raise ValueError(f"unknown scope_bandwidth kind={kind!r}: expected one of "
                         f"{ANALOG_KINDS + DIGITAL_KINDS}")
    if causal is None:
        causal = kind in ANALOG_KINDS
    wn = min(bw_hz / (grid.fs / 2.0), 0.99)
    if kind == "brickwall":
        X = np.fft.rfft(x)
        X[np.fft.rfftfreq(len(x)) > wn / 2.0] = 0.0   # rfftfreq's Nyquist is 0.5
        return np.fft.irfft(X, len(x))
    if kind == "gaussian":
        if not causal:
            f = np.fft.rfftfreq(len(x))
            H = np.exp(-0.5 * (f / (wn / 2.0 + 1e-12)) ** 2)
            return np.fft.irfft(np.fft.rfft(x) * H, len(x))
        # A magnitude-only response is zero-phase and therefore non-causal (symmetric
        # pre-ringing). The MINIMUM-PHASE response with the same magnitude is the causal one
        # with the least delay, and it is the same construction the causal channel uses
        # (`physics._min_phase_H`), applied as a LINEAR convolution so the tail does not wrap.
        return _P.apply_transfer(
            x, lambda nfft: _P._min_phase_H(_gaussian_mag(nfft, wn), nfft, half=True))
    if not causal:
        return _sig.sosfiltfilt(_sig.bessel(order, wn, output="sos"), x)
    sos = _sig.bessel(order, wn, output="sos", norm="mag")   # -3 dB AT wn, not at 0.68*wn
    zi = _sig.sosfilt_zi(sos) * x[0]
    y, _ = _sig.sosfilt(sos, x, zi=zi)
    return y


# ------------------------------------------------------- an INDEPENDENT sampling clock (U-09)
# WHY A RATIONAL RATE CHANGE IS NOT A CLOCK.
#
# `compose._op_digitize`'s `n_out=` reaches the acquisition rate through `resample_poly`, which is
# a RATIONAL rate change: the output's sample times are exactly `k*(down/up)/fs`. That is a
# PHASE-LOCKED relationship -- the record's timebase is a divided-down copy of the link's, so no
# matter how long the record is, the two never drift apart. A real instrument's timebase is a
# DIFFERENT OSCILLATOR from the link's: it has a frequency offset (ppm), it has an arbitrary
# starting phase within a sample, and it drifts as it warms. Over a long capture that offset is
# most of why a recovered clock has to keep moving.
#
# MEASURED, on a 65536-sample record, asking for a 300 ppm offset:
#   * default path (no `n_out`)  -- `resample_poly` is not called; the output is BIT-IDENTICAL to
#     the input, so the realised offset is exactly 0.000000 ppm and the slip exactly 0 samples.
#   * `n_out=round(n*(1+300e-6))` -- `n_out` is an INTEGER, so the realisable offsets are
#     quantised to `1/n` = 7.6294 ppm here: 300 ppm becomes 297.5458 ppm, an error of 2.454 ppm
#     (0.82 % of the request) that depends on the record length.
#   * a DRIFTING clock is not expressible at all: one integer ratio holds for the whole record.
#   * and `n_out` changes the record LENGTH at a fixed time span, which is the wrong mechanism --
#     a clock offset keeps the length and changes the span.
#
# `resample_at` is the primitive that can: bandlimited (windowed-sinc) interpolation at ARBITRARY
# real sample positions, so the sample times can be any function of k at all.
#
# WHAT A REAL RECORD HAS THAT THE FIXTURE DOES NOT -- MEASURED, not argued. Three real captures
# from a real-time sampling oscilloscope, 40 GSa/s, 8 M samples (200 us) each, at three
# different symbol rates. The realised symbol rate is fitted from ~163k-272k threshold
# crossings: each crossing is assigned a UI index by rounding its interval to the nearest UI,
# and one least-squares line through (UI index, crossing position) gives samples-per-UI over the
# whole record.
#
#     capture   crossings   realised samples/UI   offset from nominal   slip over the record
#     A            272127   14.814844247          -1.987 ppm            -15.89 samples
#     B            163277   (24.691...)           -1.822 ppm            -14.58 samples
#     C            215999   ( 7.407...)           -1.983 ppm            -15.87 samples
#
# Fit residuals 0.343-0.561 samples (jitter and ISI); on capture A the offset is stable across
# ten equal segments at -1.751 to -2.435 ppm, i.e. no measurable drift over 200 us. So EVERY
# real record here carries a ~2 ppm offset between the link's symbol clock and the record's
# timebase, worth ~16 samples of slip in 200 us, and the modelled scope reproduces 0.000000 ppm
# and 0 samples.
#
# WHAT THIS DOES NOT ESTABLISH: which of the two oscillators is off. The measurement is a RATIO,
# and all three symbol rates plausibly derive from one reference on the source side, so a common
# offset is consistent with either the source's reference or the instrument's timebase being the
# fast one. It establishes that the ratio is offset and stable, which is the mechanism; it does
# not attribute it.

# Re-exported from `resample`, which holds the kernel so `physics` can use it too without
# importing this module. The names here came first and are kept.
from . import resample as _RS                                           # noqa: E402
from .resample import resample_at                                       # noqa: E402,F401

RESAMPLE_HALF_WIDTH = _RS.HALF_WIDTH
RESAMPLE_CUTOFF = _RS.CUTOFF
RESAMPLE_BETA = _RS.BETA
RESAMPLE_PASSBAND_FRAC = _RS.PASSBAND_FRAC


def out_of_band_fraction(x, passband_frac=RESAMPLE_PASSBAND_FRAC):
    """Fraction of ``x``'s energy above ``passband_frac`` of fs -- what `resample_at` cannot
    interpolate correctly. Reported rather than assumed."""
    X = np.abs(np.fft.rfft(np.asarray(x, float) - float(np.mean(x)))) ** 2
    f = np.fft.rfftfreq(len(x))
    tot = float(X.sum())
    return 0.0 if tot <= 0 else float(X[f > passband_frac].sum() / tot)


def clock_slip_samples(i, ppm=0.0, drift_ppm_per_s=0.0, fs=None):
    """THE CONVENTION, as a closed form. A feature at LINK sample index ``i`` is recorded by a
    sample clock running ``ppm`` fast at RECORD index ``i + slip``, with

        slip(i) = p*i + r*i**2/(2*fs),     p = ppm*1e-6,  r = drift_ppm_per_s*1e-6

    ``ppm`` is the offset of the sample clock's FREQUENCY: ``fs_actual = fs*(1 + p + r*t)``.
    Positive ppm = a fast clock = more record samples per link second, so features drift LATER
    in the record. ``drift_ppm_per_s`` needs ``fs`` (the second term is quadratic in time).

    This is the number `sample_clock` is asserted against, and it is exact -- not a first-order
    expansion. 300 ppm over 3 M UI at 16 samples/UI = 48e6 samples is 14,400 samples of slip."""
    i = np.asarray(i, float)
    p, r = ppm * 1e-6, drift_ppm_per_s * 1e-6
    if r != 0.0:
        if fs is None:
            raise ValueError("clock_slip_samples needs fs= when drift_ppm_per_s is nonzero")
        return p * i + r * i ** 2 / (2.0 * float(fs))
    return p * i


def sample_positions(n_out, fs, ppm=0.0, drift_ppm_per_s=0.0, phase0=0.0):
    """The LINK-grid position at which each of ``n_out`` record samples was taken -- the exact
    inverse of `clock_slip_samples`.

    The clock's phase accumulator advances at ``fs*(1 + p + r*t)``, so sample ``k`` is taken when
    it reaches ``k``: ``(r/2)t**2 + (1+p)t = k/fs``. Solved exactly,

        r == 0:  src[k] = k/(1+p)
        r != 0:  src[k] = fs*(-(1+p) + sqrt((1+p)**2 + 2*r*k/fs))/r

    ``phase0`` is the clock's starting phase in LINK samples (a real timebase has no reason to
    line up with the link's sample grid, and an integer output length cannot express it)."""
    p, r = ppm * 1e-6, drift_ppm_per_s * 1e-6
    k = np.arange(int(n_out), dtype=float)
    if r == 0.0:
        return float(phase0) + k / (1.0 + p)
    return float(phase0) + fs * (-(1.0 + p) + np.sqrt((1.0 + p) ** 2 + 2.0 * r * k / fs)) / r


def sample_clock(x, grid, ppm=0.0, drift_ppm_per_s=0.0, phase0_s=0.0, n_out=None,
                 half_width=RESAMPLE_HALF_WIDTH, band_tol=0.01, span="fit"):
    """Sample the record on an INDEPENDENT timebase: a free-running sample clock with a ``ppm``
    frequency offset from the link, an optional drift, and an arbitrary starting phase.

    This is the mechanism `resample_poly` structurally cannot provide (see the block comment
    above): the sample times are `sample_positions(...)`, an arbitrary real-valued function of
    the sample number, realised by `resample_at`.

      ppm               sample-clock frequency offset. Positive = fast clock; a feature at link
                        sample i is recorded at i*(1+ppm*1e-6). See `clock_slip_samples`.
      drift_ppm_per_s   the offset's rate of change (a warming oscillator). The slip is then
                        quadratic in time.
      phase0_s          the clock's starting phase, in SECONDS.
      n_out             record length. Default (``span="fit"``) is the longest record the input
                        supports without extrapolating past its last sample, capped at
                        ``len(x)``. ``span="strict"`` keeps ``len(x)`` and RAISES if that would
                        run past the input, naming the shortfall -- because holding the last
                        sample to fill a record is fabricating samples.
      band_tol          raise if more than this fraction of the input's energy is above the
                        interpolator's measured passband (0.386*fs), where it cannot be
                        interpolated. ``None`` to skip the check.

    Returns the record. It is generally NOT ``len(x)`` long -- that is the point: a clock with a
    frequency offset covers a different span of link time than the link's own grid does."""
    x = np.asarray(x, float)
    n = len(x)
    p = ppm * 1e-6
    if (ppm, drift_ppm_per_s, phase0_s) == (0.0, 0.0, 0.0) and n_out in (None, n):
        return x.copy()      # the same clock == the same samples, not a re-interpolation of them
    if p <= -1.0:
        raise ValueError(f"ppm={ppm} means a stopped or reversed sample clock")
    if band_tol is not None:
        frac = out_of_band_fraction(x)
        if frac > band_tol:
            raise ValueError(
                f"{frac:.3%} of the record's energy is above {RESAMPLE_PASSBAND_FRAC}*fs, the "
                f"windowed-sinc passband, where bandlimited interpolation is wrong by up to "
                f"4 % of full scale. Raise half_width=, resample from a finer grid, or pass "
                f"band_tol=None if you accept it.")
    phase0 = float(phase0_s) * grid.fs
    if n_out is None:
        # the largest k whose sample time is still inside the input
        want = n
        if span == "fit":
            src_full = sample_positions(want, grid.fs, ppm, drift_ppm_per_s, phase0)
            ok = np.nonzero(src_full <= n - 1)[0]
            if len(ok) == 0:
                raise ValueError("no requested sample time falls inside the input record")
            want = int(ok[-1]) + 1
        n_out = want
    src = sample_positions(int(n_out), grid.fs, ppm, drift_ppm_per_s, phase0)
    if src[-1] > n - 1 or src[0] < 0:
        short = max(float(src[-1]) - (n - 1), -float(src[0]))
        if span == "strict":
            raise ValueError(
                f"a {n_out}-sample record on this clock needs link samples out to "
                f"{src[-1]:.3f} but the input has {n} -- {short:.3f} samples short. Render a "
                f"longer input (see Signal.lead_in), pass n_out=, or span='fit'.")
    return resample_at(x, src, half_width=half_width)


def rc_pole_hz(r_ohm, c_f):
    """The RC pole a capacitive load puts on a source resistance: ``1/(2*pi*R*C)`` [Hz]."""
    return 1.0 / (2.0 * np.pi * float(r_ohm) * float(c_f))


def probe_loading(x, grid, c_load_f=0.5e-12, r_source=50.0, causal=True, r_term_ohm=None):
    """A passive probe's input capacitance LOADS the node it measures — an RC low-pass with a
    pole at ``1/(2*pi*R*C)`` that attenuates high frequency. Real probes perturb the DUT.

    ``causal=True`` (THE DEFAULT SINCE THE ZERO-PHASE FIX) applies the single pole the closed
    form names, magnitude AND phase: ``H(f) = 1/(1 + j*f/fc)``, so the loss is exactly
    ``10*log10(1 + (f/fc)**2)`` dB and the group delay is the pole's, not zero.

    ``causal=False`` IS THE OPT-OUT FOR THE OLD DEFAULT, and it is not this response: it routes
    through the legacy zero-phase Bessel, which runs the pole forwards and backwards, so its
    magnitude is the closed form SQUARED -- MEASURED, twice the dB at every frequency
    (6.027 dB where the closed form says 3.014, 14.014 where it says 6.989) with -0.007 deg of
    phase where an RC pole has -45. Pass it only to reproduce a record made before the fix.

    ``r_term_ohm`` is the probe's own input resistance (1 MOhm for a classic high-Z passive
    probe, 50 Ohm for a 50-Ohm-terminated input). The default (``None``) is the ORIGINAL
    behaviour: an implicitly infinite termination, so the pole comes from ``r_source`` alone
    and there is no resistive divider. A FINITE termination adds two things a real 50-Ohm-
    terminated probe actually has: the pole moves to ``1/(2*pi*(r_source||r_term)*C)``, and the
    signal is attenuated by the divider ``r_term/(r_source+r_term)`` even at DC.

    A note on what this path does NOT fix: the pole is applied by dividing the record's rfft,
    which is a CIRCULAR convolution, so the response to the record's tail lands on its head.
    The pole's time constant is 22.5 ps at R=50, C=0.45 pF, so on any record long against that
    the wrap is negligible -- but it is not zero, and making it linear would change the
    arithmetic that the exact-closed-form gate checks. Logged in BACKLOG.md."""
    if r_term_ohm is None:
        r_eff, gain = r_source, 1.0
    else:
        r_eff = (r_source * r_term_ohm) / (r_source + r_term_ohm)
        gain = r_term_ohm / (r_source + r_term_ohm)
    fc = rc_pole_hz(r_eff, c_load_f)
    if not causal:
        return gain * scope_bandwidth(x, grid, fc, kind="bessel", order=1, causal=False)
    x = np.asarray(x, float)
    f = np.fft.rfftfreq(len(x), d=1.0 / grid.fs)
    return gain * np.fft.irfft(np.fft.rfft(x) / (1.0 + 1j * (f / fc)), len(x))


def _compensation_and_ground_lead_H(f, fc, compensate, l_gnd_h, c_load_f, r_eff):
    """One combined frequency response for the probe's compensation trimmer and its ground-lead
    inductance -- both real, both usually taught with the SAME square-wave test signal, so one
    stage carries both rather than two independent circular-convolution passes.

    Compensation: a passive attenuator probe has a trimmer capacitor across its tip resistor
    that a technician adjusts so the divider is frequency-INDEPENDENT. Modelled as a zero at
    ``fc/compensate`` against the loading pole at ``fc``: at ``compensate=1`` the zero sits
    exactly on the pole and cancels it (``H=1``, correctly compensated = flat, the identity --
    which is why the default leaves `probe`'s existing output untouched). Below 1 the zero
    moves to a HIGHER frequency than the pole, so the pole dominates sooner -- extra roll-off,
    the textbook "rounded corners" of an under-compensated probe. Above 1 the zero moves BELOW
    the pole and boosts the band between them -- the "peaked corners" of over-compensation.

    Ground-lead inductance: a long ground lead in series with the tip capacitance forms a
    series R-L-C -- a real second-order low-pass, ``H(f) = wn^2 / (wn^2 - w^2 + j*2*zeta*wn*w)``
    with ``wn = 1/sqrt(L*C)`` and ``zeta = (R/2)*sqrt(C/L)`` -- the same construction as
    `events.second_order_step`, applied here to the whole record instead of one localized event.
    ``l_gnd_h=0`` (the default) leaves this factor at exactly 1.
    """
    H = np.ones_like(f, dtype=complex)
    if compensate != 1.0:
        fc_zero = fc / float(compensate)
        H = H * (1.0 + 1j * (f / fc_zero)) / (1.0 + 1j * (f / fc))
    if l_gnd_h:
        w = 2 * np.pi * f
        wn = 1.0 / math.sqrt(float(l_gnd_h) * float(c_load_f))
        zeta = 0.5 * float(r_eff) * math.sqrt(float(c_load_f) / float(l_gnd_h))
        H = H * (wn * wn) / (wn * wn - w * w + 2j * zeta * wn * w)
    return H


def probe(x, grid, c_load_f=0.5e-12, r_source=50.0, bw_hz=None, kind="bessel", order=4,
          noise_rms=0.0, atten=1.0, rng=None, causal=None, r_term_ohm=None, compensate=1.0,
          l_gnd_h=0.0, coupling="dc", ac_fc_hz=None, overload_range=None, overload_tau_s=1e-6):
    """The thing a measurement is made THROUGH. A chain that runs channel -> front end models
    an ideal tap, which does not exist: a probe is an instrument in its own right and
    contributes several separate mechanisms, all of them here.

      * **loading** — its input capacitance across the (possibly terminated) source resistance
        is an RC pole; see `probe_loading` for ``r_term_ohm``.
      * **compensation / ground lead** — a trimmer capacitor that can be under/over-adjusted
        (``compensate``, 1.0 = correctly compensated = no change) and a ground lead's series
        inductance with the tip capacitance (``l_gnd_h``, a real second-order ring). See
        `_compensation_and_ground_lead_H`. Both are LINEAR, so they run through
        `physics.apply_transfer` (a linear, not circular, convolution) rather than the older
        bare-pole division above.
      * **atten** — the divider ratio as a GAIN (``0.1`` for a 10:1 probe).
      * **bandwidth** — ``bw_hz`` is the probe's own analog roll-off (the "bandwidth limiter"
        switch on a real instrument is exactly this knob at a stated corner, e.g. 20 MHz — no
        separate knob for it).
      * **coupling** — ``'dc'`` (default) or ``'ac'``, the latter applying `physics.ac_couple`
        at ``ac_fc_hz`` (default: `ac_couple`'s own default corner) so the probe blocks the
        node's DC level the way an AC-coupled input does.
      * **overload recovery** — ``overload_range`` (default ``None`` = off) hard-clips to the
        probe's linear range and adds a decaying "excess" tail with time constant
        ``overload_tau_s``, the slow baseline recovery a real overdriven front end shows after
        the signal returns inside range. This stage is NONLINEAR and STATEFUL, unlike
        everything else here — see `compose._lead_extent`, which strips it before probing this
        op's impulse response, exactly as it already strips `noise_rms`.
      * **noise** — its own input-referred noise, ``noise_rms``, added at the probe tip.

    Defaults are a bare loading pole with no bandwidth limit and no noise, so ``probe(x, g)``
    is exactly ``probe_loading(x, g, causal=True)`` -- every new knob above is an identity at
    its default, so existing recipes render bit-identically."""
    r_eff = r_source if r_term_ohm is None else (r_source * r_term_ohm) / (r_source + r_term_ohm)
    y = probe_loading(x, grid, c_load_f=c_load_f, r_source=r_source, causal=True,
                      r_term_ohm=r_term_ohm)
    if compensate != 1.0 or l_gnd_h:
        fc = rc_pole_hz(r_eff, c_load_f)

        def make_H(nfft):
            f = np.fft.rfftfreq(nfft, d=1.0 / grid.fs)
            return _compensation_and_ground_lead_H(f, fc, compensate, l_gnd_h, c_load_f, r_eff)

        y = _P.apply_transfer(y, make_H, linear=True)
    if atten != 1.0:
        y = y * float(atten)
    if bw_hz is not None:
        y = scope_bandwidth(y, grid, float(bw_hz), kind=kind, order=order, causal=causal)
    if coupling == "ac":
        kw = {"fc_hz": ac_fc_hz} if ac_fc_hz is not None else {}
        y = _P.ac_couple(y, grid=grid, **kw)
    elif coupling != "dc":
        raise ValueError(f"probe: coupling must be 'dc' or 'ac', got {coupling!r}")
    if overload_range is not None:
        y = _overload_recovery(y, grid, float(overload_range), float(overload_tau_s))
    if noise_rms:
        rng = rng or np.random.default_rng()
        y = y + rng.normal(0.0, float(noise_rms), len(y))
    return y


def _overload_recovery(x, grid, overload_range, overload_tau_s):
    """Hard-clip to the probe's linear range -- a real front end cannot exceed its own rail no
    matter how hard it is driven, so the output stays AT ``clipped`` for as long as the input is
    outside range -- then, once the input returns inside range, add a DECAYING baseline residual
    left over from how far outside range it just was. This is a peak-hold-with-slow-release
    charge (like `open_drain_line`, a per-sample recurrence rather than a plain filter, because
    the charge and release phases follow different rules): it tracks the excess exactly while
    overloaded, and relaxes toward zero with time constant ``overload_tau_s`` once the excess
    ends -- so the recovery tail is a real POST-overload artefact, not a way to exceed the rail
    during the overload itself."""
    x = np.asarray(x, float)
    clipped = np.clip(x, -overload_range, overload_range)
    excess = x - clipped
    decay = math.exp(-(1.0 / grid.fs) / overload_tau_s)
    n = len(x)
    charge = np.empty(n)
    acc = 0.0
    for i in range(n):
        acc = excess[i] if excess[i] != 0.0 else acc * decay
        charge[i] = acc
    return clipped + (charge - excess)


def timebase_jitter(x, grid, rms_ps=0.5, rng=None):
    """Sample-clock / timebase jitter — each sample is taken at a slightly wrong time, which
    smears the eye HORIZONTALLY. Adds independent per-sample timing error (RMS in ps)."""
    rng = rng or np.random.default_rng()
    x = np.asarray(x, float)
    n = len(x)
    dev = (rms_ps * 1e-12 * grid.fs) * rng.standard_normal(n)
    idx = np.arange(n)
    return np.interp(idx - dev, idx, x, left=x[0], right=x[-1])


def sinad_noise_rms(enob, full_scale):
    """The in-band noise+distortion rms that an ENOB figure stands for, at `+/- full_scale`.

    ENOB IS A SINAD FIGURE, NOT A BIT DEPTH. It is defined from a full-scale sine:
    `SINAD_dB = 6.02*ENOB + 1.76`, so the noise it names is `(2*full_scale/2**enob)/sqrt(12)`
    -- the rms a uniform quantiser of that many bits WOULD have had. That equivalence is the
    whole reason ENOB is quoted in bits, and it is also the trap: the equivalent rms is a
    CONTINUOUS noise level, and rounding a record to `2**enob` codes reproduces the level while
    inventing a lattice the converter does not have. See `converter_noise_rms`."""
    return (2.0 * float(full_scale) / 2.0 ** float(enob)) / np.sqrt(12.0)


def converter_noise_rms(enob, full_scale, bandwidth_hz, nyquist_hz, bits=None):
    """The CONVERTER's own wideband noise rms behind a published `enob` measured at
    `bandwidth_hz`, given the converter runs to `nyquist_hz`.

    A converter's noise is (to first order) white across its own Nyquist band; the instrument's
    selected-bandwidth filter sits AFTER it and passes only `bandwidth_hz/nyquist_hz` of that
    power. So one fixed converter floor produces a whole ENOB-vs-bandwidth table:

        enob(B) = enob(B_ref) + 0.5*log2(B_ref/B)

    MEASURED against one real-time sampling oscilloscope's published table (13 bandwidth/ENOB pairs, 10 to
    110 GHz): anchored at 110 GHz -> 5.0, the other twelve settings come back at **-0.05 to
    +0.31 bits**, rms 0.16 -- one number reproducing twelve. Adding a second, frequency-rising
    noise term only improves the rms to 0.08, which is not enough structure to justify it. The
    filter's SHAPE cancels out entirely (any fixed shape has noise bandwidth proportional to
    its -3 dB point, and the proportionality is absorbed by the anchor), so this needs no
    filter model to be useful.

    `bits` (the converter's real depth) splits the answer honestly: the returned rms is the
    converter's ADDED noise with its own lattice's `q**2/12` taken out, so that
    `quantize_adc(x + noise, bits=bits)` lands on the published ENOB rather than overshooting
    it. It raises if the depth alone cannot reach the requested ENOB.

    For that instrument -- 10 bits, +/-423 mV, ENOB 5.0 at 110 GHz, Nyquist 128 GHz -- this is
    8.23 mV rms, **9.96 LSB of the converter's own lattice**: the lattice carries 0.084 % of
    the noise power that sets ENOB, and would on its own be worth 10.00 bits. That ratio is
    the entire argument for modelling the two mechanisms apart."""
    wide = sinad_noise_rms(enob, full_scale) * np.sqrt(float(nyquist_hz) / float(bandwidth_hz))
    if bits is None:
        return float(wide)
    q = 2.0 * float(full_scale) / 2.0 ** int(bits)
    added = wide ** 2 - q ** 2 / 12.0
    if added <= 0.0:
        raise ValueError(
            f"a {bits}-bit converter cannot reach ENOB {enob} at {bandwidth_hz / 1e9:g} GHz: its own "
            f"lattice already contributes {q / np.sqrt(12.0):.3e} V rms against the {wide:.3e} V rms "
            f"the figure allows")
    return float(np.sqrt(added))


def quantize_adc(x, enob=None, full_scale=None, bits=None):
    """Quantise to an ADC lattice — the one thing a converter does that nothing else does.

      bits        the converter's REAL depth -> exactly 2**bits codes across +/- full_scale.
                  This is the physical lattice: the converter has the depth it has, full stop.
      enob        legacy: treat an effective-bits figure as if it were a lattice depth.
                  KEPT FOR COMPATIBILITY AND IT IS THE CONFLATION THIS MODULE NOW SEPARATES --
                  ENOB is a SINAD figure (noise AND distortion, continuous); bit depth is a
                  lattice (discrete). Rounding to 2**enob codes gets the noise POWER roughly
                  right and the record's structure entirely wrong: a 10-bit lattice is
                  9.96 LSB below its own noise (so fully dithered, and invisible in a
                  histogram), while an ENOB-5.9 lattice is 17x coarser than the real one and
                  produces a comb no real capture has. Use `bits` + `converter_noise_rms`.
      full_scale  +/- range of the lattice; None takes it from the signal's peak

    Exactly one of `bits`/`enob`. Returns the quantised array; each sample moves by at most
    half an LSB. Pairs with clip_adc (clip first so out-of-range samples land on the top code,
    not beyond it)."""
    if (bits is None) == (enob is None):
        raise ValueError("quantize_adc needs exactly one of bits= (a converter depth) or "
                         "enob= (the legacy effective-bits lattice)")
    x = np.asarray(x, float)
    fs = float(np.max(np.abs(x))) + 1e-12 if full_scale is None else float(full_scale)
    lsb = 2.0 * fs / 2 ** (int(bits) if bits is not None else enob)
    return np.round(x / lsb) * lsb


def digitize(x, grid=None, interleave=None, clip_full_scale=None, enob=None,
             noise_floor=None, rng=None, bits=None):
    """Compose the ADC stages in the physically correct order and return ``(y, info)``.

    Order — all of it AFTER the channel and the additive impairment: additive noise floor
    -> interleave mismatch (at the sampling instant) -> hard clip (at the ADC input) ->
    quantise (last). Getting this order wrong is silent: quantising before the noise, or
    clipping after quantising, yields a plausible waveform with the wrong noise floor. It
    lives here once so a caller cannot get it wrong.

      noise_floor       kwargs for shaped_noise_floor, e.g. {"rms": 1e-3, "shape": "pink"}
      interleave        kwargs for interleave_adc, e.g. {"m_cores": 4, "offset_v": 1e-3}
      clip_full_scale   hard-clip level (absolute); None to skip. Also sets the quantiser range.
      bits              the converter's REAL depth for the final lattice; None to skip. Pair it
                        with a `noise_floor` sized by `converter_noise_rms` -- that is the honest
                        two-mechanism converter: a discrete lattice at the depth the part has,
                        and a continuous noise that sets the SINAD the data sheet publishes.
      enob              legacy: quantise to a 2**enob lattice instead, which models the two as
                        one thing. See `quantize_adc`.

    ``info`` records the applied settings and the clipped-sample mask fraction — feeding
    provenance (#5) and measured ground truth (#8). ``grid`` is accepted for API symmetry;
    amplitude stages need no rate conversion."""
    x = np.asarray(x, float)
    rng = rng or np.random.default_rng()
    info = {}
    if noise_floor:
        x = x + shaped_noise_floor(len(x), rng=rng, **noise_floor)
        info["noise_floor"] = dict(noise_floor)
    if interleave:
        x = interleave_adc(x, rng=rng, **interleave)
        info["interleave"] = dict(interleave)
    if clip_full_scale is not None:
        x, mask = clip_adc(x, clip_full_scale)
        info["clip_full_scale"] = float(clip_full_scale)
        info["clipped_fraction"] = float(mask.mean())
    if bits is not None and enob is not None:
        raise ValueError("digitize takes bits= (a converter depth) or enob= (the legacy "
                         "effective-bits lattice), not both")
    if bits is not None:
        x = quantize_adc(x, bits=bits, full_scale=clip_full_scale)
        info["bits"] = int(bits)
    elif enob is not None:
        x = quantize_adc(x, enob=enob, full_scale=clip_full_scale)
        info["enob"] = float(enob)
    return x, info


def store_record(x, bits=11, full_scale=None, headroom=1.05, clip=True,
                 dither_lsb=0.0, rng=None):
    """The LAST thing a real-time DSO does before it hands you a file: write integer codes.

    THIS IS AN EXPORT STEP, NOT A NOISE MODEL, AND IT IS WHY REAL RECORDS HAVE A FLOOR
    ---------------------------------------------------------------------------------
    `quantize_adc` models the CONVERTER, which sits in the middle of the chain: a real
    instrument filters, corrects and decimates AFTER its converter, and every one of those
    stages is a weighted sum of many codes, so by the time the record reaches memory the
    converter's lattice is gone. What comes back out of the instrument is on a lattice
    again -- because the record is STORED as int16 codes -- and that second, terminal
    lattice is the record's noise floor at every frequency the signal does not occupy.

    Measured on three real compliance-suite exports of 8 M samples each: 1,851 /
    1,880 / 2,035 distinct values, 1.0000 of them on the lattice, occupancy 0.997-1.000
    (a full lattice, not a sparse one), and a stop-band PSD floor equal to the lattice's
    own `q**2/12/(fs/2)` to under 0.1 dB in 3 of 3. A rendered record that stops at the
    converter and then filters holds ~7 M distinct values and a stop band at -245 dB/Hz,
    which is float64 numerical zero: 60-90 dB of missing floor, and an amplitude histogram
    5-11x smoother than any real capture's.

      bits        stored code width. `lsb = 2*full_scale/2**bits`.
      full_scale  +/- range the codes span, in the same amplitude units as `x`. None means the
                  vertical was RANGED TO THIS ACQUISITION: `headroom * max|x|`, chosen once for
                  the whole record. That is what an operator does before pressing Run, and it
                  is not the same thing as data-dependent rounding -- the pitch is constant
                  across the record either way, so the floor does not move with the signal
                  within it. Give a number instead to model a fixed vertical setting.
      headroom    how far above the record's peak the vertical sits when `full_scale` is None.
                  The three real captures hold 1,851 / 1,880 / 2,035 codes, so at 11 bits their
                  own ranging left 0.6-10.7 % of headroom; 1.05 is the middle of that.
      clip        clip to the representable code range (a real export cannot store a code
                  it has no room for). Set False to keep an out-of-range sample on-lattice
                  but out-of-range, which is a rendering, not an acquisition.
      dither_lsb  rms, in LSB, of independent noise added immediately BEFORE the rounding.
                  A real converter chain may carry dither -- a fixed-point DSP stage rounding at
                  the same word width just upstream contributes one more `q**2/12` and lands the
                  floor at `q**2/6`, i.e. +3.01 dB. `1/sqrt(12) = 0.2887` is that value.
                  **Default 0.0, a bare rounder, and that is what matches real captures.**

    THE +3 dB CLAIM THAT WAS HERE WAS WRONG, and how it went wrong is worth keeping. Two units
    reported the three real captures' stop bands sitting +2.68 / +3.00 / +3.08 dB above their own
    lattice's `q**2/12/(fs/2)`, and concluded a real store must be dithered. Re-measured with a
    Hann window on the same band of the same records, all three land on `q**2/12` to
    **0.00 / 0.00 / -0.03 dB**. The +3 dB was spectral leakage from an in-band signal ~80 dB above
    the floor, through the rectangular window's sidelobes.

    Both estimators that produced it were gated -- Parseval-exact, and recovering the closed form
    on constructed uniform error to 0.02 dB -- and both gates were blind, because neither
    constructed case had a signal in it to leak. The gate that catches it is a KNOWN floor
    measured underneath a LARGE in-band signal; a windowed estimator recovers it to 0.02 dB and an
    unwindowed one reads 3 dB high. An estimator is only as good as the hardest case it was gated
    on.

    A NOTE ON THE "COMB RATIO", because it is the metric most likely to be aimed at here.
    `mean|diff(hist)|/mean(hist)` at 2000 bins is NOT a physics measurement: it is the beat
    between the histogram's bin grid and the record's lattice, and it is a non-monotonic
    function of the distinct-value count almost alone. MEASURED on constructed Gaussian noise
    on a lattice, with no signal physics at all: 1,851 codes -> 0.184, 1,880 -> 0.153,
    2,035 -> 0.057, 2,099 -> 0.115, 2,399 -> 0.346. The three real captures' 0.197 / 0.144 /
    0.091 are recovered by their code counts and nothing else, the metric has a MINIMUM right
    where distinct ~ bins, and dither moves it by under 0.001. So a record reading just under
    the real band is telling you its code count is slightly high, not that its noise is wrong,
    and tuning noise to hit a comb number tunes against an aliasing artefact. Aim at the
    distinct count / occupancy / effective bits instead; the comb follows.

    Returns floats -- the voltage each stored code stands for -- so the record stays in the
    pipeline's units. Every value is an exact multiple of one LSB and, for `bits <= 16`,
    an exactly representable int16 code times that LSB."""
    x = np.asarray(x, float)
    bits = int(bits)
    if bits < 1:
        raise ValueError("store_record needs at least 1 bit of stored code width")
    if full_scale is None:
        full_scale = float(headroom) * float(np.max(np.abs(x)))
        if full_scale <= 0.0:
            return x * 0.0                        # an all-zero record has no range to set
    lsb = 2.0 * float(full_scale) / 2.0 ** bits
    if dither_lsb:
        rng = rng or np.random.default_rng()
        x = x + rng.normal(0.0, float(dither_lsb) * lsb, len(x))
    codes = np.round(x / lsb)
    if clip:
        codes = np.clip(codes, -(2.0 ** (bits - 1)), 2.0 ** (bits - 1) - 1.0)
    return codes * lsb


def quantisation_floor_db_per_hz(lsb, fs_hz):
    """The one-sided PSD of a quantiser's own error, `q**2/12/(fs/2)`, in dB/Hz.

    A uniform quantiser of step `q` contributes `q**2/12` of total noise power spread flat
    over the Nyquist band, so a stored record's stop band cannot sit below this. It is the
    prediction that matched three real captures to under 0.1 dB, and it is how a caller
    checks a rendered record has the floor it claims rather than asserting that it does."""
    return float(10.0 * np.log10((float(lsb) ** 2 / 12.0) / (float(fs_hz) / 2.0)))


def shaped_noise_floor(n, rms=0.01, shape="pink", rng=None):
    """A frequency-shaped noise floor (real front ends are not flat). shape in
    {'white','pink','blue'} — pink ~ 1/sqrt(f), blue ~ sqrt(f). Returns length-n noise
    with the requested RMS. Add to a signal to model a coloured floor."""
    rng = rng or np.random.default_rng()
    w = rng.standard_normal(n)
    if shape != "white":
        W = np.fft.rfft(w)
        f = np.arange(W.shape[0], dtype=float); f[0] = 1.0
        W = W / np.sqrt(f) if shape == "pink" else W * np.sqrt(f)
        w = np.fft.irfft(W, n)
    return (rms / (w.std() + 1e-12)) * w


def clip_adc(x, full_scale=1.0):
    """Hard-clip to +/- full_scale (a real ADC saturates at full scale). Returns
    (clipped_signal, clipped_mask) so downstream tools can exclude saturated samples."""
    x = np.asarray(x, float)
    mask = np.abs(x) >= full_scale
    return np.clip(x, -full_scale, full_scale), mask


# General-purpose alias (issue #51): the analog input-stage bandwidth limit.
input_bandwidth = scope_bandwidth
