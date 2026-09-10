"""
wfmsynth.rx — receiver-side equalization (companions to the transmit-side FFE, #11).

A real high-speed receiver equalizes the channel before slicing. Two standard blocks:

  * CTLE — a Continuous-Time Linear Equalizer: an analog high-frequency-peaking filter (a
    zero below its poles) that boosts the attenuated high frequencies to flatten the channel.
    Linear, memoryless of decisions, applied to the waveform.
  * DFE — a Decision-Feedback Equalizer: cancels post-cursor ISI by subtracting weighted PAST
    DECISIONS from the current sample before slicing. Nonlinear (it feeds back sliced
    symbols), so it can remove a sharp discrete post-cursor a linear EQ only smears.

CTLE is composable in the waveform chain (`Signal.ctle(...)`); DFE operates on the per-symbol
samples a receiver has already recovered (pair it with `measure.sample_at_phase`).
"""
from __future__ import annotations

import numpy as np
from scipy import signal as _sig

_PAM4 = np.array([-1.0, -1 / 3, 1 / 3, 1.0])


def ctle(x, grid, fz_ghz, fp1_ghz, fp2_ghz, dc_gain=1.0):
    """Continuous-time linear equalizer: ``H(s) = dc_gain·(1 + s/ωz) / ((1+s/ωp1)(1+s/ωp2))``
    with the zero below the poles, so it peaks at high frequency and flattens a lossy channel.
    ``dc_gain`` is the exact DC gain; the corner frequencies are in GHz. Applied to the
    waveform via a bilinear-transformed digital filter."""
    wz, wp1, wp2 = (2 * np.pi * f * 1e9 for f in (fz_ghz, fp1_ghz, fp2_ghz))
    num = dc_gain * np.array([1.0 / wz, 1.0])
    den = np.polymul([1.0 / wp1, 1.0], [1.0 / wp2, 1.0])
    b, a = _sig.bilinear(num, den, fs=grid.fs)
    return _sig.lfilter(b, a, np.asarray(x, float))


def ffe(x, taps, tap_spacing, pre=0):
    """Receiver feed-forward equalizer: a linear FIR with ``taps`` spaced ``tap_spacing`` samples apart,
    ``pre`` of them pre-cursor. Unlike CTLE (a fixed analog shape) this is an arbitrary-tap FIR the RX
    trains to the channel. ``tap_spacing = samples_per_ui`` is a T-spaced (baud-rate) FFE; ``= spb//2``
    is a fractionally (T/2) spaced FFE — the usual receiver form, insensitive to sampling phase.
    Applied to the WAVEFORM (post-channel), companion to the transmit-side ``tx_ffe``."""
    x = np.asarray(x, float); taps = np.asarray(taps, float); n = len(x); y = np.zeros(n)
    for k, c in enumerate(taps):
        d = int(round((k - pre) * tap_spacing))
        if d >= 0:
            y[d:] += c * x[:n - d]
        elif -d < n:
            y[:d] += c * x[-d:]
    return y


def dfe(samples, taps, levels=_PAM4):
    """Decision-feedback equalizer over per-symbol ``samples``. For each symbol it subtracts
    ``taps · [previous decisions]`` (the post-cursor estimate) before slicing to the nearest
    level. Returns ``(equalized, decisions)``. ``taps[j]`` is the post-cursor weight at lag
    ``j+1``; set them to the channel's post-cursor to cancel it exactly."""
    samples = np.asarray(samples, float)
    taps = np.asarray(taps, float)
    levels = np.asarray(levels, float)
    eq = np.empty_like(samples)
    dec = np.empty_like(samples)
    hist = np.zeros(len(taps))
    for k in range(len(samples)):
        c = samples[k] - np.dot(taps, hist)
        d = levels[int(np.argmin(np.abs(levels - c)))]
        eq[k] = c
        dec[k] = d
        if len(hist):
            hist = np.roll(hist, 1)
            hist[0] = d
    return eq, dec


# ------------------------------------------------------------------ AGC (U-12)
# WHY A CHAIN WITHOUT ONE CARRIES A LEVEL IT WOULD NOT HAVE ON HARDWARE.
#
# A receiver NORMALISES BEFORE IT EQUALISES. Its slicer thresholds, its DFE taps and its CTLE
# operating point are all defined against a full scale the AGC establishes, so the absolute
# amplitude arriving at the package is information the receiver deliberately throws away. A
# modelled chain with no AGC keeps it: change the driver swing or add 6 dB of channel loss and
# every downstream stage that has an absolute number in it -- `dfe(scale=...)`, `levels=`,
# a slicer threshold -- silently means something different. `compose._dfe_core` already takes
# `scale=`, "a known full scale, e.g. an AGC's"; this is the AGC whose output lands there.
#
# It is not a linear stage and is not listed as one: the gain is a functional of the input's own
# level, so `agc(3*x) == agc(x)`, not `3*agc(x)`. That is what an AGC is for.

AGC_METRICS = ("rms", "peak", "amplitude", "p99")


def agc_level(x, metric="rms", q=99.0):
    """The level an AGC measures. ``rms``; ``peak`` = max|x|; ``amplitude`` = half the
    peak-to-peak (the right one for a signal with a DC offset); ``p99`` = the ``q``-th
    percentile of |x| (robust to a few outliers, which is what a real detector's averaging
    does)."""
    x = np.asarray(x, float)
    if metric == "rms":
        return float(np.sqrt(np.mean(x ** 2)))
    if metric == "peak":
        return float(np.max(np.abs(x)))
    if metric == "amplitude":
        return 0.5 * float(np.ptp(x))
    if metric == "p99":
        return float(np.percentile(np.abs(x), q))
    raise ValueError(f"unknown AGC metric {metric!r}; use one of {AGC_METRICS}")


def agc_gain(x, target=1.0, metric="rms", q=99.0, tau_s=None, grid=None, gain_limits=None,
             on_limit="raise"):
    """The gain `agc` applies — a scalar for a block AGC, an array for a tracking one.

    A BLOCK AGC (``tau_s=None``) measures the whole record once: ``g = target/level(x)``, so the
    output's level is EXACTLY ``target``. That is the analytic property this is asserted on.

    A TRACKING AGC (``tau_s`` seconds, needs ``grid``) runs a one-pole loop on the instantaneous
    level, which is what real hardware does and which therefore does NOT hit the target until it
    has settled -- and which follows a level that moves, the reason to want it. The loop is
    causal (`lfilter`, initialised to the record's first-window level) so the gain at sample k
    depends only on samples up to k.

    ``gain_limits=(lo, hi)`` is a finite AGC range. ``on_limit='raise'`` (the default) refuses a
    target the range cannot reach and names the gain it would have needed -- a silent clamp here
    would put a record below its stated level with nothing said, which is the defect
    `physics.ac_couple` was carrying. ``on_limit='clip'`` saturates instead, which is the
    physical behaviour once you have deliberately stated a range."""
    x = np.asarray(x, float)
    target = float(target)
    if tau_s is None:
        lvl = agc_level(x, metric, q)
        g = target / (lvl + 1e-300)
    else:
        if grid is None:
            raise ValueError("a tracking AGC (tau_s=) needs grid= to convert seconds to samples")
        if metric not in ("rms", "peak", "amplitude"):
            raise ValueError(f"a tracking AGC supports rms/peak/amplitude, not {metric!r}")
        ns = max(1.0, float(tau_s) * grid.fs)
        a = float(np.exp(-1.0 / ns))                       # one-pole, tau = ns samples
        if metric == "rms":
            u = x ** 2
            zi = np.array([u[:int(np.ceil(ns))].mean() * a])
            env = _sig.lfilter([1.0 - a], [1.0, -a], u, zi=zi)[0]
            lvl = np.sqrt(np.maximum(env, 0.0))
        else:
            u = np.abs(x)
            zi = np.array([u[:int(np.ceil(ns))].max() * a])
            lvl = _sig.lfilter([1.0 - a], [1.0, -a], u, zi=zi)[0]
        g = target / (lvl + 1e-300)
    if gain_limits is not None:
        lo, hi = (float(v) for v in gain_limits)
        need = np.asarray(g, float)
        if np.any(need < lo) or np.any(need > hi):
            if on_limit == "raise":
                raise ValueError(
                    f"AGC range {lo:g}..{hi:g} cannot bring this record to target={target:g}: "
                    f"it needs gain {float(np.min(need)):.4g}..{float(np.max(need)):.4g}. Widen "
                    f"gain_limits, or pass on_limit='clip' to let it saturate.")
            if on_limit != "clip":
                raise ValueError(f"on_limit must be 'raise' or 'clip', not {on_limit!r}")
            g = np.clip(g, lo, hi)
    return g


def agc(x, target=1.0, metric="rms", q=99.0, tau_s=None, grid=None, gain_limits=None,
        on_limit="raise"):
    """Automatic gain control: bring the record to a stated ``target`` level so everything
    downstream can be written against a known full scale. See `agc_gain` for the parameters and
    for why a block AGC hits the target exactly and a tracking one does not."""
    return np.asarray(x, float) * agc_gain(x, target=target, metric=metric, q=q, tau_s=tau_s,
                                           grid=grid, gain_limits=gain_limits, on_limit=on_limit)


# --------------------------------------------------- the RECEIVER's own noise floor (U-12)
def input_noise(x, grid=None, rms=None, density=None, bw_hz=None, rng=None, exact_rms=True):
    """The RECEIVER's input-referred noise, added at its input.

    THIS IS NOT THE SCOPE'S NOISE. `compose._op_digitize`'s ``noise_rms``/``snr_db`` and
    `instrument.shaped_noise_floor` model the INSTRUMENT's noise floor -- what the measurement
    adds. A receiver has its own, and it enters somewhere else in the chain: at the receiver
    input, BEFORE the CTLE and the DFE, so the equaliser's high-frequency peaking amplifies it.
    Placing the same rms at the two ends of an equaliser is not the same experiment.

    State it as an ``rms`` in the record's units, or as a ``density`` in units/sqrt(Hz) with the
    receiver's noise bandwidth ``bw_hz`` (needs ``grid``), in which case
    ``rms = density*sqrt(bw_hz)`` and the noise is band-limited to ``bw_hz`` -- so the realised
    in-band one-sided PSD is exactly ``density**2`` and there is none above the band. A
    receiver's front end is not white to the sampler's Nyquist.

    ``exact_rms=True`` renormalises the draw so the realised rms is the stated one rather than
    the stated one plus a chi-square wobble (the same convention `physics.inject_jitter` uses
    for its Rj, and what makes the level assertable). It scales one Gaussian draw, so the
    samples stay jointly Gaussian and independent in shape."""
    x = np.asarray(x, float)
    n = len(x)
    rng = rng or np.random.default_rng()
    if (rms is None) == (density is None):
        raise ValueError("input_noise takes exactly one of rms= or density= (with bw_hz=)")
    if density is not None:
        if bw_hz is None or grid is None:
            raise ValueError("density= needs bw_hz= (the receiver's noise bandwidth) and grid=")
        rms = float(density) * np.sqrt(float(bw_hz))
    rms = float(rms)
    if rms == 0.0:
        return x.copy()
    w = rng.standard_normal(n)
    if bw_hz is not None:
        if grid is None:
            raise ValueError("bw_hz= needs grid= to know the sample rate")
        if float(bw_hz) >= grid.f_nyquist:
            raise ValueError(f"bw_hz={bw_hz:g} is at or above Nyquist {grid.f_nyquist:g}; omit "
                             f"it for a floor that is white to Nyquist")
        W = np.fft.rfft(w)
        W[np.fft.rfftfreq(n, d=1.0 / grid.fs) > float(bw_hz)] = 0.0
        w = np.fft.irfft(W, n)
    if exact_rms:
        w = w / (float(np.sqrt(np.mean(w ** 2))) + 1e-300) * rms
    else:
        w = w * rms
    return x + w
