"""
wfmsynth.physics — physics-informed waveform synthesis primitives.

Frequency-domain lossy channels (skin + dielectric, causal minimum-phase option),
transmission-line multi-reflection, crosstalk (NEXT/FEXT), AC-coupling, physically-
decomposed jitter (Rj/Pj/DCD), digital signaling (NRZ/PAM4 from PRBS), RF modulation
(AM/FM/PSK/QAM), chirp, PDN transients, and an ECG-like generator.

Every primitive is paired with an assertion in `wfmsynth.validate` that checks the
physical property actually holds — nothing is trusted without that check. Works on a
normalized, unitless time grid (`N` samples over [0,1)); rate-parameterizable to any
symbol/sample rate downstream. numpy/scipy only.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
import re

import numpy as np
import warnings

from scipy import signal

from . import quinary as Q
from . import resample as RS

N = 4096                       # default working grid
T = np.linspace(0.0, 1.0, N, endpoint=False)

# Every primitive below infers its grid length from the array it is given, and the
# generators take an optional `n`. N/T remain the DEFAULTS, so any call that worked
# against the fixed 4096-point grid behaves bit-identically. This is what makes the
# "rate-parameterizable" claim actually true: a deep-memory capture is tens of
# millions of points, and nothing here should care how long the record is.


# ------------------------------------------- transform length and LINEAR convolution
# Every frequency-domain stage below applies its response as ``irfft(rfft(x) * H)``. That
# product is a CIRCULAR convolution: the response to the record's last samples wraps around
# the end and lands on its first ones. A physical channel has no such periodicity -- before a
# record starts there is a quiescent line, not the record's own tail. The textbook treatment
# is to zero-pad to at least the linear-convolution length ``len(x) + len(h) - 1``, transform
# there, and truncate back to ``len(x)``. That is what `apply_transfer` does, and it is the
# default for every stage that calls it.
#
# The corruption the padding removes is CONFINED to the RECORD'S EDGES -- the wrapped tail lands
# on the first ``len(h) - 1`` samples, and for a response that is not causal the record's own
# tail loses the pre-cursor it should have had. It is a large error in a short prefix, not a
# small error everywhere. Measured on a 131,056-sample record, as the difference between the two
# paths against the record's peak-to-peak: a causal analytic channel moves by 7.1 % and only in
# its first 128 samples; a zero-phase one by 11.4 %, in its first 57 and last 238; a cascade read
# at the driver plane by 2.9 % in its first 850, and at the load by 29.3 % in its first 1047.
# Trim 2000 samples from each end of any of them and nothing moves by more than 0.02 % of span.
#
# The same padding also removes a performance cliff that has nothing to do with physics.
# Pinning the transform to exactly the record length hands the FFT whatever factorisation that
# length happens to have, and a record is very often a whole number of periods of some pattern:
# if the period carries a large prime factor, so does the record, and numpy falls back to
# Bluestein's algorithm (a chirp-z transform -- three transforms at a larger length, plus
# several arrays that size). Since the record is being padded anyway, it costs nothing to pad
# to the next 5-smooth length rather than to the exact minimum, and the cliff goes away.
#
# ``len(h)`` is not known in advance: a response is supplied as a function of frequency, not as
# a tap list. So it is MEASURED rather than assumed -- `response_extent` builds H on a short
# probe grid, transforms it to an impulse response, and takes the shortest arc of that circle
# holding everything above `rel` of the peak (an arc, not a prefix, because a response is not
# always causal). The probe doubles until the answer fits inside half of it, so the measurement
# is not itself corrupted by the wrap it exists to avoid.

# HOW FAR DOWN "OVER" IS, AND WHAT IT COSTS. A channel with a sqrt(f) term is not analytic at
# DC, so its impulse response does not end -- it decays algebraically, and the guard is a choice
# of where to stop paying for it, not a fact. Measured on a 131,056-sample record (16 samples/UI
# x an 8191-UI pattern period) against a reference padded to 4x the record, as
# max|y - y_ref| / peak-to-peak of the record:
#
#   threshold      lossy causal        lossy zero-phase     cascade, echoes     cascade, thru
#   (below peak)   guard  nfft/n err   guard  nfft/n err    guard nfft/n err    guard  nfft/n err
#   -80 dB  1e-4    2543   1.03  1.6e-5  3265  1.03  3.4e-5  1008  1.03 6.5e-6    9125  1.07 6.5e-5
#   -100 dB 1e-5   17564   1.14  1.2e-5 15389  1.13  2.0e-5  1548  1.03 6.5e-6   43987  1.35 3.3e-5
#   -120 dB 1e-6   66405   1.53  7.0e-6 73633  1.56  8.1e-6  4520  1.05 5.9e-6  217555  2.67 1.1e-5
#
# The CIRCULAR path those columns replace is wrong by 7.1 %, 11.4 %, 2.9 % and 29.3 % of the same
# record's span, so every row above is three to four orders of magnitude better than the thing it
# replaces. -100 dB is the default: it holds the residual below 3.3e-5 of span -- roughly a
# fifteenth of one LSB of an 11-bit record -- for at most a 35 % longer transform. Callers who
# want the algebraic tail chased further pass `guard=` (or a smaller `rel`) and pay for it.

SMOOTH_RADIX = (2, 3, 5)       # "5-smooth"/regular lengths: the FFT's fast path
RESPONSE_REL = 1e-5            # -100 dB below the peak: where an impulse response is "over"
PROBE_N0 = 4096                # first probe length for the response-extent measurement
PROBE_MAX = 1 << 21            # stop doubling the probe here, and warn


def next_smooth_length(n, radix=SMOOTH_RADIX):
    """The smallest integer >= `n` whose prime factors all lie in `radix`.

    With the default ``(2, 3, 5)`` these are the 5-smooth (regular) numbers, which is the set
    of lengths a mixed-radix FFT has a direct algorithm for. The overhead is small: no gap
    between consecutive 5-smooth numbers above 1000 exceeds ~6 %."""
    n = int(n)
    if n <= 1:
        return 1
    radix = sorted({int(r) for r in radix})
    if radix[0] < 2:
        raise ValueError("radix entries must be >= 2")
    best = None

    def rec(i, val):
        nonlocal best
        if best is not None and val >= best:
            return                                   # this branch can only grow
        if i == len(radix) - 1:
            r = radix[-1]
            while val < n:
                val *= r
            if best is None or val < best:
                best = val
            return
        while True:
            rec(i + 1, val)
            if val >= n:
                return
            val *= radix[i]

    rec(0, 1)
    return best


def response_extent(make_H, rel=RESPONSE_REL, n0=PROBE_N0, probe_max=PROBE_MAX, warn=True):
    """Measure the impulse-response length, in samples, of the response ``make_H(nfft)``.

    `make_H` takes a transform length and returns that response on the ``rfft`` grid of that
    length, at a FIXED sample interval -- so a longer probe is a finer frequency resolution
    over the same physical band, not a different channel. The extent is the width of the
    shortest arc of that impulse response holding every sample at or above `rel` times its
    peak -- an arc rather than a prefix, because the response may reach back before t=0.

    The measurement is itself a circular transform, so it doubles the probe until the answer
    fits in the probe's first half; only then is the tail known not to have wrapped onto the
    head. If `probe_max` is reached first the result is a LOWER BOUND and a warning says so --
    `warn=False` for a caller that is imposing `probe_max` as a deliberate cap and has said so
    in its own documentation (`apply_transfer` does exactly that)."""
    n = int(n0)
    while True:
        h = np.fft.irfft(np.asarray(make_H(n), complex), n)
        peak = float(np.abs(h).max())
        if peak == 0.0:
            return 1
        extent = _circular_support(np.abs(h) >= rel * peak)
        if extent <= n // 2:
            return extent
        if n >= probe_max:
            if not warn:
                return extent
            warnings.warn(
                f"response_extent: response still fills the probe at n={n} "
                f"(support {extent}); padding with a lower bound, so a little wrap may remain",
                RuntimeWarning, stacklevel=2)
            return extent
        n *= 2


def _circular_arc(loud):
    """``(start, width)`` of the shortest arc of the circle containing every True in `loud`.

    An impulse response on an FFT grid lives on a circle, and it is not always one-sided: a
    zero-phase response is symmetric about t=0 and a reflection can have a small pre-cursor,
    both of which put samples at NEGATIVE time -- stored at the END of the array. Measuring
    "the last index above threshold" would call those a late tail and report the whole record.
    The arc is instead the complement of the LONGEST quiet run, which is the same number for a
    causal response and the right one for a two-sided one.

    `start` is where that arc BEGINS, which `_response_taps` needs and the width alone cannot
    say: it is what decides how much of the response lies before t=0, and therefore how far a
    linear convolution has to reach to deliver the sample at t=0."""
    loud = np.asarray(loud, bool)
    n = len(loud)
    idx = np.nonzero(loud)[0]
    if idx.size == 0:
        return 0, 1
    if idx.size == n:
        return 0, n
    first = int(idx[0])
    rolled = np.nonzero(np.roll(loud, -first))[0]     # index 0 is now loud, so no run wraps it
    tail = n - 1 - int(rolled[-1])                    # the quiet run that ends at the wrap
    gaps = np.diff(rolled) - 1
    if gaps.size and int(gaps.max()) > tail:
        k = int(np.argmax(gaps))
        quiet, start_rolled = int(gaps[k]), int(rolled[k + 1])
    else:
        quiet, start_rolled = tail, 0
    return (first + start_rolled) % n, n - quiet


def _circular_support(loud):
    """Width of the shortest arc containing every True in `loud` -- see `_circular_arc`."""
    return _circular_arc(loud)[1]


def linear_fft_length(n_signal, n_response, radix=SMOOTH_RADIX):
    """The transform length for a LINEAR convolution of `n_signal` samples with an
    `n_response`-sample impulse response: the next `radix`-smooth length at or above
    ``n_signal + n_response - 1``."""
    return next_smooth_length(int(n_signal) + int(n_response) - 1, radix)


# A response whose memory is this small a FRACTION of the record is one an overlap-save
# convolution applies for a fraction of the cost, because the whole-record transform is paying
# log(n) for a filter that only reaches `guard` samples. Below this ratio `method="auto"` takes
# the block path; above it the transform is the better tool and nothing changes.
OVERLAP_MAX_FRAC = 0.125
OVERLAP_MIN_N = 1 << 16        # under this the whole-record transform is already cheap
OVERLAP_CHUNK = 1 << 14
# How far the extracted FIR may sit from the response it came from before it is judged an
# aliasing artefact rather than an honest truncation. The two are orders apart, not close:
# truncating at RESPONSE_REL costs ~1e-4 of the response's peak, while reading a delay off the
# wrong side of a circular probe costs O(1), so anything in between separates them.
TAPS_FAITHFUL_TOL = 0.05


def _taps_at(make_H, nh, rel):
    """The response's arc read off a probe of length `nh`, as ``(taps, lag_min)``."""
    h = np.fft.irfft(np.asarray(make_H(nh), complex), nh)
    peak = float(np.abs(h).max())
    if peak == 0.0:
        return np.zeros(1), 0
    start, width = _circular_arc(np.abs(h) >= rel * peak)
    width = min(width, nh)
    lag_min = start if start <= nh // 2 else start - nh
    return h[(np.arange(width) + start) % nh], lag_min


def _response_taps(make_H, guard, rel, radix=SMOOTH_RADIX, probe_max=PROBE_MAX):
    """The response as a LINEAR FIR: ``(taps, lag_min)``, where ``taps[j]`` is the response at
    lag ``lag_min + j``. ``None`` when it cannot be resolved without aliasing.

    The impulse response on an FFT grid is circular, and `lag_min` is how the two differ: a
    minimum-phase channel is causal and starts at lag 0, but a zero-phase response is symmetric
    about t=0 and a reflection has a pre-cursor, and for those the arc begins at a NEGATIVE lag
    stored at the end of the array. Reading the taps off in circular order and calling them
    causal is exactly the error that puts a channel's pre-cursor at the end of the record.

    THE EXTRACTED TAPS ARE VERIFIED AGAINST THE RESPONSE THEY CAME FROM, at a LONGER length
    than they were read off. A circular probe cannot tell a delay of `d` from one of
    ``d mod nh``, so a response that is mostly delay -- a cascade section, a long line -- can
    read as no delay at all, and comparing two probes does not reliably catch it: a
    power-of-two delay aliases IDENTICALLY at `nh` and ``2*nh`` (4096 lands on lag 0 at both
    1024 and 2048). Comparing ARC POSITIONS across probes does not work either, because a
    two-sided response's measured width legitimately grows with resolution and drags its start
    with it.

    So the check is direct instead of comparative: lay the taps at their stated lags in a
    longer buffer, transform, and require that to be the response `make_H` asks for. Truncating
    at `rel` costs a small relative error there; getting the lag wrong costs an O(1) one, so
    the two are never close. MEASURED: a 4096-sample pure delay reads as ZERO delay at the
    first probe and resolves at nh=8192, and this is what catches it."""
    nh = next_smooth_length(max(4 * int(guard), 1024), radix)
    while nh <= probe_max:
        taps, lag = _taps_at(make_H, nh, rel)
        if _taps_are_faithful(taps, lag, make_H, next_smooth_length(3 * nh, radix)):
            return taps, lag
        nh *= 2
    return None


def _taps_are_faithful(taps, lag_min, make_H, nh, tol=TAPS_FAITHFUL_TOL):
    """Do `taps`, laid at their stated lags, actually transform back into ``make_H(nh)``?

    `nh` is longer than the probe the taps were read off, so a tap set whose lags are an
    aliasing artefact of that probe cannot agree here, while one that is merely TRUNCATED at
    `rel` agrees to about the truncation level."""
    ref = np.asarray(make_H(nh), complex)
    scale = float(np.abs(ref).max())
    if scale == 0.0:
        return True
    buf = np.zeros(nh)
    buf[(np.arange(len(taps)) + lag_min) % nh] = taps
    return bool(np.abs(np.fft.rfft(buf) - ref).max() <= tol * scale)


def _apply_overlap_save(x, make_H, guard, rel, chunk=OVERLAP_CHUNK):
    """`apply_transfer`'s block path: the same linear convolution, applied `chunk` samples at a
    time, so the working set is the block rather than the record. ``None`` when the response
    cannot be resolved as a finite FIR, so the caller can fall back to the transform."""
    from .stream import stream_convolve
    n = len(x)
    got = _response_taps(make_H, guard, rel)
    if got is None:
        return None
    taps, lag_min = got
    # y[k] = z[k - lag_min] where z is the causal convolution of x with `taps`. A response that
    # reaches BEFORE t=0 therefore needs the convolution carried `-lag_min` samples past the
    # record's end; one that starts after it leaves that many zeros at the head.
    if lag_min < 0:
        pad = -lag_min
        xs = np.concatenate([x, np.zeros(pad)])
        z = stream_convolve(xs, taps, chunk)
        del xs
        return z[pad:pad + n].copy()
    z = stream_convolve(x, taps, chunk)
    if lag_min == 0:
        return z
    out = np.zeros(n)
    out[lag_min:] = z[:max(0, n - lag_min)]
    return out


def apply_transfer(x, make_H, linear=True, guard=None, radix=SMOOTH_RADIX,
                   rel=RESPONSE_REL, n0=PROBE_N0, probe_max=PROBE_MAX, method="fft"):
    """Apply a frequency response to `x` as a LINEAR convolution, and return ``len(x)`` samples.

    `make_H(nfft)` returns the response on the ``rfft`` grid of length `nfft` (same sample
    interval whatever `nfft` is). The record is zero-padded to a 5-smooth length at or above
    ``len(x) + len(h) - 1``, multiplied there, and truncated back -- so no part of the response
    to the record's tail lands on its head.

    `guard` fixes ``len(h)`` in samples instead of measuring it (`response_extent` does the
    measuring). ``linear=False`` restores the pinned-length CIRCULAR convolution, which is what
    this module did before and is kept so the two paths can be compared directly.

    `method` CHOOSES HOW THE SAME CONVOLUTION IS COMPUTED, and the choice is a behaviour change
    rather than a pure optimisation, which is why the default does not move:

        "fft"      (default) one whole-record transform. What this function has always done.
        "overlap"  overlap-save in blocks: the same LINEAR convolution, computed `chunk`
                   samples at a time, so the working set is the block rather than the record.
        "auto"     "overlap" when the response's measured memory is a small fraction of the
                   record (`OVERLAP_MAX_FRAC`) and the record is long enough to be worth
                   blocking (`OVERLAP_MIN_N`); "fft" otherwise.

    WHY IT IS NOT THE DEFAULT, stated plainly because the number matters. The two paths do not
    agree to round-off: MEASURED on an 8-inch causal channel over a 4 M-sample record, they
    differ by 2.5e-5 of the record's peak-to-peak, and that floor does NOT fall as the
    truncation threshold is tightened -- at `rel` from 1e-5 down to 1e-13 it stays at 2.5e-5
    while the tap count grows from 0.2 % to 50 % of the record. The reason is that the
    whole-record path's own answer is a function of the record LENGTH: it samples the response
    on the padded transform's frequency grid, and a block path samples it on a shorter one.
    Both are legitimate discretisations of the same continuous channel and they converge as the
    record grows against the channel's memory, but at any finite length they are 1e-5 apart --
    which `tests/test_byte_identity.py` classifies as a behaviour change (it starts catching at
    1e-5) rather than as the 3.8e-13 round-off it deliberately absorbs. So a recipe rendered on
    one path does not reproduce bit-for-bit on the other, and `method` is recorded in the recipe
    so that it replays on the path it was rendered with.

    WHAT IT BUYS, measured at 2 M samples on the same channel: 4.7x the speed, and peak memory
    falls from about 5x the record to about 1.2x -- which is the difference between a deep
    record rendering inside a GUI or a demo and not rendering there at all."""
    x = np.asarray(x, float)
    n = len(x)
    if method not in ("fft", "overlap", "auto"):
        raise ValueError(f"apply_transfer: unknown method {method!r} (use 'fft', 'overlap' or 'auto')")
    if not linear:
        if method == "overlap":
            raise ValueError("apply_transfer: method='overlap' is a LINEAR convolution; it has "
                             "no circular form. Pass linear=True, or method='fft'.")
        return np.fft.irfft(np.fft.rfft(x) * make_H(n), n)
    if guard is None:
        # The guard is capped at TWICE the record. A response that long is one whose level at
        # those lags is already at `rel`, so each further doubling of the transform buys less
        # than the last; without the cap a pathological channel (a steep fit against a deep
        # stop-band floor) pads an 8 k record to 1.1 M and spends 0.22 s doing it. Past the cap
        # the remaining wrap is bounded by the response's own level there; `guard=` overrides.
        cap = int(min(probe_max, max(4 * n, 4 * n0)))
        guard = min(response_extent(make_H, rel=rel, n0=min(n0, max(8, n)), probe_max=cap,
                                    warn=False), 2 * n)
    guard = max(1, int(guard))
    if method == "auto":
        method = ("overlap" if n >= OVERLAP_MIN_N and guard <= OVERLAP_MAX_FRAC * n else "fft")
    if method == "overlap":
        got = _apply_overlap_save(x, make_H, guard, rel)
        if got is not None:
            return got
        # Naming it rather than quietly producing the wrong answer: a response this function
        # cannot resolve as a finite FIR is one whose taps would be aliased, and the transform
        # applies it correctly at a cost in memory rather than in correctness.
        warnings.warn(
            "apply_transfer: method='overlap' could not resolve this response as a finite "
            "impulse response without aliasing (it is probably mostly delay, or longer than "
            f"the {PROBE_MAX}-sample probe cap); falling back to the whole-record transform.",
            RuntimeWarning, stacklevel=2)
    nfft = linear_fft_length(n, guard, radix)
    xp = np.zeros(nfft)
    xp[:n] = x
    X = np.fft.rfft(xp)
    del xp
    X *= make_H(nfft)                       # in place: one fewer record-sized temporary
    y = np.fft.irfft(X, nfft)
    del X
    return y[:n].copy()                     # a copy, not a view on the padded buffer


# ---------------------------------------------------------------- channel physics
def _min_phase_H(Hmag, n=None, half=False):
    """Causal minimum-phase complex response from a real magnitude |H| (rfft bins),
    via the cepstral / Hilbert relation so loss and phase are physically LINKED
    (Kramers-Kronig; the Djordjevic-Sarkar-style causal channel). A real magnitude-
    only response is zero-phase -> non-causal (symmetric pre/post-ringing); the
    minimum-phase version concentrates the response AFTER t=0 (asymmetric post-cursor
    ISI), which is what a real dispersive interconnect does.

    `n` is the full (two-sided) transform length; inferred from Hmag when omitted.

    `half=True` returns only the rfft half of the result (`n//2+1` bins for even `n`,
    `(n+1)//2` for odd), which is all any FILTERING caller wants -- it is about to multiply the
    record's own rfft by it. It is not a convenience: the full two-sided complex array is the
    largest single allocation in a deep render, and not building it is most of why this function
    is no longer the memory ceiling. The default stays two-sided so callers that want the whole
    response (and the pinned references that check it) are unaffected.

    THIS RUNS AT THE RECORD'S OWN LENGTH, so its temporaries are the ones that decide whether a
    deep record renders at all. Three record-length arrays that used to be built here are gone:

      * the symmetric MIRROR of the magnitude. `mag_full` is real and EVEN, so its inverse DFT
        is real and even -- which is exactly what `irfft` computes from the half already in
        hand. The mirror never needs to exist.
      * the COMPLEX cepstrum whose real part was then copied out. `irfft` returns the real
        sequence directly, so neither the complex array nor the copy is allocated.
      * the causal-folding WEIGHT VECTOR. Its entries are exactly 1.0, 2.0 and 0.0, all exact in
        binary floating point, so applying them by slice in place is the same arithmetic
        without a record-length temporary.

    The two real-input transforms also replace two complex ones of the same length, which is
    where the time goes as well."""
    if n is None:
        n = 2 * (len(Hmag) - 1)
    logmag = np.asarray(Hmag, float) + 1e-12
    np.log(logmag, out=logmag)
    c = np.fft.irfft(logmag, n)                           # the real, even cepstrum
    del logmag
    if n % 2 == 0:
        # even length: DC and Nyquist stand alone, the bins between them double, the
        # anti-causal half is discarded
        c[1:n // 2] *= 2.0
        c[n // 2 + 1:] = 0.0
    else:
        # odd length: no Nyquist bin
        c[1:(n + 1) // 2] *= 2.0
        c[(n + 1) // 2:] = 0.0
    spec = np.fft.rfft(c) if half else np.fft.fft(c)
    del c
    return np.exp(spec, out=spec)                         # complex min-phase


def insertion_loss_db(f_ghz, length_in=6.0, tand=0.02, eps_r=4.3, skin_k=0.0,
                      loss_db=None, loss_at_ghz=None, trend=None, trend_floor_db=80.0):
    """The insertion-loss MAGNITUDE, in dB, on an arbitrary frequency axis `f_ghz`.

    This is the loss law of `lossy_channel` lifted out of it, so that a lumped channel and one
    SECTION of a cascaded path (`wfmsynth.sparam.cascade`) compute their loss with the same
    arithmetic instead of two implementations that agree until someone edits one of them. The
    argument meanings, and why `trend` beats `loss_db`+`loss_at_ghz`, are documented on
    `lossy_channel`; nothing here is new physics.

    Returns a non-negative array (dB of loss, so `|H| = 10**(-il/20)`)."""
    f_ghz = np.asarray(f_ghz, float)
    if trend is not None:
        a, b, c = (float(t) for t in trend)
        il_db = -(a * np.sqrt(f_ghz) + b * f_ghz + c)
        # passive (a channel cannot amplify) and bounded (see `lossy_channel`'s `trend_floor_db`)
        return np.clip(il_db, 0.0, float(trend_floor_db))
    b_diel = 2.3 * np.sqrt(eps_r) * tand
    a_skin = skin_k if skin_k > 0 else 0.35               # ~dB/in/sqrt(GHz) typ
    il_db = (a_skin * np.sqrt(f_ghz) + b_diel * f_ghz) * length_in
    if loss_db is not None and loss_at_ghz is not None:
        # keep the skin+dielectric SHAPE; scale so IL(loss_at_ghz) == loss_db exactly
        shape_at = (a_skin * np.sqrt(loss_at_ghz) + b_diel * loss_at_ghz) * length_in
        il_db = il_db * (loss_db / (shape_at + 1e-12))
    return il_db


def lossy_channel(x, length_in=6.0, tand=0.02, eps_r=4.3, f_nyq_ghz=8.0,
                  skin_k=0.0, causal=False, grid=None, loss_db=None, loss_at_ghz=None,
                  trend=None, trend_floor_db=80.0, linear=True, guard=None, method="fft"):
    """Apply a frequency-dependent SI channel: insertion loss
        IL(f)[dB] = (a_skin*sqrt(f_GHz) + b_diel*f_GHz) * length_in
    with dielectric-loss coefficient b_diel = 2.3*sqrt(eps_r)*tand (dB/in/GHz)
    [KB: All-About-Circuits/Bogatin]. f axis is scaled so the Nyquist bin maps to
    f_nyq_ghz, whatever the record length.
    causal=False: zero-phase magnitude response (legacy). causal=True: physically
    correct minimum-phase response -> asymmetric post-cursor ISI (real dispersion).

    Absolute units (wfmsynth.grid.Grid): pass grid=Grid(...) to take the real
    frequency axis (f_nyq_ghz) from the grid's Nyquist. Pass loss_db + loss_at_ghz to
    request a channel with a stated insertion loss (dB) at a stated frequency (GHz):
    the skin+dielectric SHAPE is kept and scaled so IL(loss_at_ghz) == loss_db exactly.

    THE SHAPE IS THE PART THAT MATTERS, AND ONE ANCHOR DOES NOT FIX IT
    -----------------------------------------------------------------
    `loss_db`+`loss_at_ghz` pin the curve at ONE frequency and leave it free everywhere
    else, because the skin/dielectric mix above is HARD-WIRED: 0.35*sqrt(f) against
    2.3*sqrt(eps_r)*tand*f in a fixed ratio, whatever `loss_db` says. That mix implies
    IL(4 GHz)/IL(8 GHz) = 0.617 for every channel this function can make. Real measured
    backplanes run 0.468-0.678 (pure dielectric is 0.500, pure skin effect 0.707), and
    anchored at 8 GHz on a Gen4-budget board the fixed mix is 1.7-2.0 dB optimistic at
    1-2 GHz and 3.5 dB pessimistic at 12 GHz -- 2.94 dB rms, 11.18 dB pp across the band.
    Measured through a reference receiver that error is worth ~52 mV of eye height, more
    than band truncation (19 mV) and the boards' resonances (15 mV) combined, and it is
    why an analytic channel quoted at a real board's own insertion loss opens an eye where
    the board has none.

    So `trend=(a, b, c)` takes the whole curve instead of one point on it:

        |S21|(f)[dB] = a*sqrt(f_GHz) + b*f_GHz + c        (a fitted magnitude, <= 0)
        IL(f)[dB]    = -(a*sqrt(f_GHz) + b*f_GHz + c),  clamped at >= 0 (passivity)

    which is the three-parameter least-squares fit of a measured differential |SDD21|.
    It needs no measured file at run time -- three numbers per board are enough, and a
    caller's manifest of them is the whole channel. `length_in`, `tand`, `eps_r`,
    `skin_k`, `loss_db` and `loss_at_ghz` are all ignored when `trend` is given: a fitted
    trend is not a per-inch coefficient and must not be scaled like one. Passivity is
    enforced rather than assumed -- a fit whose `c` is positive would otherwise deliver
    GAIN at DC, which is how a resonance-dominated file (whose loss at one frequency is a
    point on a resonance skirt, not a rung on a loss ladder) silently becomes an amplifier.
    A fitted trend is smooth by construction and cannot make a resonance or a stub notch;
    use `sparam` or `resonant_reflection` for those.

    `trend_floor_db` caps how deep the stop band goes, and it is not cosmetic. A fitted trend
    keeps growing outside the band it was fitted in -- a Gen4-budget board's fit reaches 366 dB
    at a 256 GSa/s grid's Nyquist -- and the causal (minimum-phase) reconstruction takes the
    LOGARITHM of the magnitude, so an unbounded roll-off pushes the cepstrum through the guard
    term and comes back with a different group delay. Measured through the Gen4 reference
    receiver on B12: uncapped, the eye reads 116.69 mV; capped at 80 dB, 128.08 mV, which is
    the number the smooth-fit arm of the measurement that motivated this reports (127.99). The
    cap is also physically honest -- no real channel's stop band is bottomless, and 80 dB is
    already ~14 dB below the noise floor of an 11-bit stored record.

    LINEAR, NOT CIRCULAR
    --------------------
    The response is applied as a LINEAR convolution: the record is zero-padded past the
    channel's own impulse-response length, transformed there, and truncated back, so the
    channel's answer to the record's last samples does not wrap onto its first. See
    `apply_transfer` for the mechanics and `response_extent` for how the padding is sized.
    `guard=` states that length in samples instead of measuring it; ``linear=False`` restores
    the pinned-length circular convolution this function used to do, which is what a
    transfer-function measurement wants (an input tone is an eigenvector of the circular
    product and of nothing else) and is otherwise kept only for comparison.

    Note what the linear form implies at the record's HEAD: with nothing before sample 0 the
    line is quiescent, so a record that begins mid-pattern starts with a turn-on edge. That is
    the honest answer for a link that starts transmitting at t=0, and the wrong one for a
    record that is a WINDOW on a link that was already running -- for which the fix is a
    lead-in the caller renders and discards, not a wrap. `BACKLOG.md` carries that item."""
    x = np.asarray(x, float)
    if grid is not None:
        f_nyq_ghz = grid.f_nyquist / 1e9                  # real frequency axis from the grid

    def make_H(nfft):
        # rfftfreq is NORMALIZED here, so the physical band is the same at any `nfft` -- a
        # longer transform is finer resolution over that band, not a different channel.
        f_ghz = np.fft.rfftfreq(nfft) * 2.0 * f_nyq_ghz   # 0..f_nyq_ghz at Nyquist
        il_db = insertion_loss_db(f_ghz, length_in=length_in, tand=tand, eps_r=eps_r,
                                  skin_k=skin_k, loss_db=loss_db, loss_at_ghz=loss_at_ghz,
                                  trend=trend, trend_floor_db=trend_floor_db)
        Hmag = 10.0 ** (-il_db / 20.0)
        if not causal:
            return Hmag
        # the minimum-phase fold is itself a transform of length `nfft`, so padding makes the
        # cepstrum resolve the response instead of time-aliasing it.
        return _min_phase_H(Hmag, nfft, half=True)

    return apply_transfer(x, make_H, linear=linear, guard=guard, method=method)


def _coupled_kernel(aggressor, kind, d_samples, n):
    """The peak-normalized coupling kernel one aggressor contributes, for `kind` in
    {'fext', 'next'} on an `n`-sample record. The ONE place the shape of a crosstalk coupling
    is decided; `crosstalk` and `crosstalk_sum` are both callers, so a four-pair sum and a
    single aggressor cannot drift apart.

    WHY FEXT IS A DERIVATIVE AND NEXT IS NOT, given both couple through the same mutual L/C.
    Both are integrals of the aggressor's d/dt over the coupled length. FEXT's contributions all
    arrive together, so the integral collapses to Td·d/dt and its transfer function rises at
    20 dB/decade right across the band. NEXT's arrive spread over the round trip, so its
    transfer function rises at 20 dB/decade only below about 1/(4·Td) and is FLAT above it --
    and on a link long enough that 1/(4·Td) sits below the signal band, in-band NEXT is a
    filtered copy of the aggressor's LEVEL, not of its derivative. A 100 m twisted-pair channel
    puts that corner near 500 kHz against a 62.5 MHz Nyquist, which is why the near-end branch
    here is a delayed level copy.

    WHAT WOULD FALSIFY IT: a short coupled section -- centimetres of package or connector rather
    than metres of cable -- puts 1/(4·Td) ABOVE the band and NEXT becomes derivative-coupled too.
    This kernel does not model that case; use kind='fext' with the near-end delay for it."""
    a = np.asarray(aggressor, float)
    if kind == "fext":
        k = np.gradient(a)
    elif kind == "next":
        d = int(d_samples); k = np.zeros(n); k[d:] = a[:n - d]
    else:
        raise ValueError(f"unknown crosstalk kind {kind!r}; use 'fext' or 'next'")
    return k / (np.abs(k).max() + 1e-9)


def crosstalk(x, aggressor, coupling=0.12, kind="fext", td_frac=0.05):
    """Add coupled noise from a neighboring (aggressor) line. FEXT couples through
    the mutual C/L as the DERIVATIVE of the aggressor (∝ d/dt); NEXT is a delayed,
    broadband coupling. `coupling` scales the aggressor relative to x's span. Real
    high-speed links are frequently crosstalk-limited — a major missing effect."""
    x = np.asarray(x, float)
    n = len(x)
    k = _coupled_kernel(aggressor, kind, int(td_frac * n), n)
    return x + coupling * (np.ptp(x) + 1e-9) * k


def db_to_coupling(loss_db):
    """A coupling/isolation/return loss in dB -> the linear `coupling` these primitives take.

    AMPLITUDE convention: 10**(-dB/20), and the primitives scale the coupled term's PEAK by it
    relative to the victim's peak-to-peak. A NEXT loss, FEXT loss or return loss read out of a
    cabling or PHY document is an amplitude (|S| ratio) figure, so it converts directly. An
    rms-referred or power-summed figure would land a few dB away, and this is the pessimistic
    reading of the two. A non-finite dB value (the mechanism switched off) is exactly 0.0, so the
    degenerate case is an exact zero rather than an underflow."""
    d = float(loss_db)
    return 0.0 if not np.isfinite(d) else 10.0 ** (-d / 20.0)


def crosstalk_sum(x, aggressors, couplings, kinds="fext", td_samples=0, grid=None, td_ps=None):
    """Sum the coupling from EXPLICIT aggressor streams into one victim.

    `crosstalk_matrix` MANUFACTURES its aggressors (one `nrz` per coupling, at a seed and a baud
    offset) and has no parameter that takes a waveform. That is right when the neighbours are
    unknown traffic, and wrong when they are known: on a four-pair link the three near-end
    aggressors are the OTHER THREE LANES OF THE SAME TRANSMITTER, carrying the same line code and
    correlated with the victim's own transmit by construction. This op takes those streams.

    `aggressors` is a sequence of arrays as long as `x`; `couplings` one linear coupling each
    (see `db_to_coupling`); `kinds` one kind each, or one kind for all. The near-end delay is
    absolute — `td_samples`, or `td_ps` with a `grid` — NOT a fraction of the record, so a
    lead-in or a longer capture does not move it.

    Each aggressor's contribution is computed against the ORIGINAL `x`, so the terms superpose
    exactly and the order of the aggressors does not matter — which is what makes a coupling
    budget auditable term by term."""
    x = np.asarray(x, float)
    n = len(x)
    m = len(couplings)
    if len(aggressors) != m:
        raise ValueError(f"{len(aggressors)} aggressor streams for {m} couplings")
    if td_ps is not None:
        if grid is None:
            raise ValueError("td_ps requires grid=Grid(...)")
        td_samples = td_ps * 1e-12 * grid.fs
    kinds = list(kinds) if isinstance(kinds, (list, tuple)) else [kinds] * m
    span = np.ptp(x) + 1e-9
    d = None
    for a, c, kd in zip(aggressors, couplings, kinds):
        if c == 0.0:
            continue          # an absent mechanism adds nothing, so the reduction is exact
        t = c * span * _coupled_kernel(a, kd, td_samples, n)
        d = t if d is None else d + t
    # the couplings are accumulated and added ONCE, so the total does not depend on the order the
    # aggressors were listed in, and an all-zero coupling vector returns the victim untouched
    return x.copy() if d is None else x + d


# The default corner of the echo's high-pass shaping, as a fraction of Nyquist. WHY THERE IS A
# HIGH-PASS AT ALL: the echo is own transmit reflected off the near-end impedance discontinuity,
# and a return-loss limit is a LOW-frequency-good specification -- 15 dB or better at the bottom
# of the band, degrading upward -- so the reflection coefficient GROWS with frequency. A flat
# echo would put low-frequency energy in the residual that the real hybrid cancels well. The
# fraction is a judgement (a real corner is a property of the connector and the cable, not of the
# standard); what would falsify it is a measured return-loss curve, which belongs in `sparam`.
ECHO_HP_FRAC_NYQUIST = 0.02


def hybrid_echo(x, own_tx, isolation_db=20.0, grid=None, td_ps=None, td_samples=None,
                f_hp_hz=None, f_hp_frac=ECHO_HP_FRAC_NYQUIST, linear=True, guard=None):
    """The transceiver's OWN TRANSMIT leaking into its own receive: a scaled, delayed, filtered
    copy of a DIFFERENT stream, added to `x`.

    THIS IS NOT A REFLECTION OF THE RECEIVED SIGNAL, and that is why `reflect` cannot express it.
    `multi_reflection` is a functional of its input alone: hand it silence and it returns
    silence. On a bidirectional pair the hybrid subtracts own transmit from the line and what
    survives — the hybrid's finite balance plus own transmit reflected off the near-end
    mismatch — is present whether or not anything is being received. Feed this op a zero `x` and
    the output is not zero.

      isolation_db  how far down the echo sits, amplitude convention (`db_to_coupling`). This is
                    the hybrid's balance and the channel's return loss taken together, because a
                    probe cannot separate them. ``inf`` switches the mechanism off and returns
                    `x` bit-for-bit.
      td_ps/        the round trip to the reflecting discontinuity. Absolute, not a fraction of
      td_samples    the record.
      f_hp_hz/      the corner of the return-loss shaping (see `ECHO_HP_FRAC_NYQUIST`).
      f_hp_frac

    Applied as a LINEAR convolution on `own_tx` (`apply_transfer`), so the echo of the transmit
    record's tail does not wrap onto its head.
    """
    x = np.asarray(x, float)
    g = db_to_coupling(isolation_db)
    if g == 0.0:
        return x            # not a shortcut: zero coupling IS no mechanism, and must be exact
    own = np.asarray(own_tx, float)
    if len(own) != len(x):
        raise ValueError(f"own_tx is {len(own)} samples, x is {len(x)}")
    if grid is not None and f_hp_hz is not None:
        f_hp_frac = grid.hz_to_frac_nyquist(f_hp_hz)
    elif f_hp_hz is not None:
        raise ValueError("f_hp_hz requires grid=Grid(...)")
    if td_samples is None:
        if not td_ps:                       # None or exactly zero: no delay, no grid needed
            td_samples = 0.0
        elif grid is None:
            raise ValueError("a nonzero td_ps requires grid=Grid(...)")
        else:
            td_samples = td_ps * 1e-12 * grid.fs

    def make_H(nfft):
        # normalized frequency: cycles per sample, so the delay is in samples and the corner is
        # a fraction of Nyquist -- the same convention the rest of this module's fraction paths use
        f = np.fft.rfftfreq(nfft, d=1.0)
        s = 1j * (f / (0.5 * f_hp_frac + 1e-18))
        return (s / (1.0 + s)) * np.exp(-1j * 2 * np.pi * f * td_samples)

    echo = apply_transfer(own, make_H, linear=linear, guard=guard)
    # normalized to the ECHO's own peak so `isolation_db` means what it says about the term that
    # is added, rather than about the transmit amplitude before an unknown filter gain
    echo = echo / (np.abs(echo).max() + 1e-12)
    return x + g * (np.ptp(x) + 1e-9) * echo


def single_pair_observation(wanted, own_tx, near, far, grid=None,
                            echo_db=None, echo_td_ps=0.0, echo_f_hp_hz=None,
                            next_db=None, next_td_ps=0.0, fext_db=None):
    """What a probe on ONE pair of a live four-pair link actually sees, as ONE channel.

    wanted + echo + NEXT + FEXT:

      wanted   the FAR-END transmitter of this pair, through the channel — the only term a
               naive single-pair model has.
      echo     `own_tx`, the NEAR-END transmitter of this same pair, leaking back through the
               hybrid's finite isolation (`hybrid_echo`).
      NEXT     the three other NEAR-END transmitters (`near`). They have traversed no cable, so
               this is normally the largest crosstalk term.
      FEXT     the three other FAR-END transmitters (`far`), which have.

    Emulating this into one stored channel is not a convenience: it is what the measurement IS.
    A differential probe on one pair sums these four whether or not the dataset admits it. What
    a single stored channel LOSES is the ability to correlate an aggressor with the victim — and
    a model trained on a single-pair capture is in the probe's position anyway.

    Each `*_db` is a LOSS in dB (amplitude convention, `db_to_coupling`); ``None`` or ``inf``
    switches that mechanism off, and with all three off the return value is `wanted` ITSELF.
    `wfmsynth.quinary.CLAUSE_40_BUDGET` carries a worked set of values and says what they rest on.

    EVERY TERM IS REFERENCED TO THE WANTED SIGNAL'S SPAN, not to the running sum, and the terms
    are accumulated before being added once. Both matter: referencing to the running sum would
    make the second mechanism's realized coupling depend on the first one's setting, and adding
    them one at a time would leave the total dependent on the order. As written, a budget can be
    attributed term by term and a sweep of one term leaves the others where they were.
    """
    w = np.asarray(wanted, float)
    d = None

    def add(term):
        return term - w if d is None else d + (term - w)

    if echo_db is not None:
        d = add(hybrid_echo(w, own_tx, isolation_db=echo_db, grid=grid, td_ps=echo_td_ps,
                            **({"f_hp_hz": echo_f_hp_hz} if echo_f_hp_hz is not None else {})))
    if next_db is not None and len(near):
        c = db_to_coupling(next_db)
        d = add(crosstalk_sum(w, list(near), [c] * len(near), kinds="next",
                              grid=grid, td_ps=next_td_ps))
    if fext_db is not None and len(far):
        c = db_to_coupling(fext_db)
        d = add(crosstalk_sum(w, list(far), [c] * len(far), kinds="fext"))
    return w if d is None else w + d


def de_emphasis_taps(db):
    """2-tap Tx FFE weights ``[main, post]`` realizing a de-emphasis of ``db`` — the
    standardized preset real transmitters expose, quoted the way every spec quotes it
    (PCI Express Gen1/Gen2: "-3.5 dB"). The first bit after a transition is at full
    amplitude and steady-state bits are reduced, with 20·log10(V_steady/V_transition) = db
    -- so a NEGATIVE db is de-emphasis (steady state below transition) and a positive one is
    pre-emphasis. Use with `tx_ffe(..., pre=0)`.

    GitHub #53: this used to compute the ratio the other way around
    (20·log10(V_transition/V_steady) = db), so a negative db produced pre-emphasis and a
    positive one produced de-emphasis -- backwards from how every spec quotes it. Existing
    callers passing a POSITIVE db for de-emphasis now need its sign flipped."""
    r = 10 ** (-db / 20.0)
    a = (r - 1.0) / (r + 1.0)
    return [1.0, -a]


def tx_ffe(x, taps, spb, pre=1):
    """Transmitter feed-forward equalizer (FFE) — the T-spaced pre-emphasis real high-speed
    transmitters almost always run. Applies per-UI weights `taps` at unit-interval spacing
    `spb` (samples per UI, may be non-integer -> fractional delay via interpolation), with
    `pre` leading (pre-cursor) taps. This puts a deliberate PRE-CURSOR in the pulse response
    — the qualitative shape a real link has and a naive synthetic waveform lacks — and
    de-emphasizes post-cursor ISI. Apply at the transmitter, BEFORE the channel. Taps are
    used as given (normalize to sum 1 for unity DC gain, or |taps| for a peak-power budget)."""
    x = np.asarray(x, float)
    idx = np.arange(len(x))
    y = np.zeros_like(x)
    for k, c in enumerate(taps):
        d = (k - pre) * spb                        # pre-cursor taps (k<pre) pull from the future
        y += c * np.interp(idx - d, idx, x, left=0.0, right=0.0)
    return y


def supply_coupling(x, grid, f_ripple_hz=1e6, am_depth=0.0, psij_ps=0.0, supply=None):
    """Couple a power-supply / PDN rail onto the signal as BOTH amplitude modulation and
    power-supply-induced jitter (PSIJ), from the SAME supply waveform — so the two artifacts
    are correlated and a downstream tool can attribute them to the rail.

      f_ripple_hz  frequency of the supply ripple tone (e.g. a switching regulator)
      am_depth     fractional amplitude modulation by the supply
      psij_ps      peak timing deviation in ps induced by the supply (PM/jitter)
      supply       optional supply waveform (same length as x) to use instead of a pure tone
                   — e.g. switching-activity-correlated ripple for a realistic scenario

    Real signals carry structured supply coupling (we otherwise model only a PDN-transient
    fault); this is the composable primitive for it."""
    x = np.asarray(x, float)
    n = len(x)
    if supply is None:
        s = np.sin(2 * np.pi * f_ripple_hz * (np.arange(n) / grid.fs))
    else:
        s = np.asarray(supply, float)
    y = x * (1.0 + am_depth * s)                              # amplitude modulation
    if psij_ps:
        dev = (psij_ps * 1e-12 * grid.fs) * s                # timing deviation (samples) ∝ supply
        y = RS.displace(y, dev)
    return y


def differential_pair(x, grid=None, skew_ps=0.0, skew_samples=None, gain_imbalance=0.0, cm=0.0):
    """Split a single-ended data waveform into a DIFFERENTIAL pair ``(p, n)`` with real
    non-idealities. p carries +½·data, n carries −½·data; ``differential_mode(p,n) = p−n``
    recovers the data and ``common_mode(p,n) = (p+n)/2`` is ideally zero. Non-idealities:

      skew_ps / skew_samples  intra-pair (P/N) skew — n is delayed. Skew closes the
                              differential eye and converts differential energy to
                              common-mode at every transition.
      gain_imbalance          fractional P/N amplitude mismatch — direct differential→
                              common-mode conversion, proportional to the data.
      cm                      an added common-mode term (scalar or array), e.g. a supply
                              tone shared by both legs.

    Real links are differential, so P/N skew, common-mode and mode conversion are first-order
    effects a differential probe sees; this is the primitive for them."""
    x = np.asarray(x, float)
    n = len(x)
    idx = np.arange(n)
    if skew_samples is None:
        skew_samples = (skew_ps * 1e-12 * grid.fs) if grid is not None else 0.0
    xd = RS.shift(x, skew_samples)      # bandlimited: a stated skew in ps is sub-sample
    p = 0.5 * (1.0 + gain_imbalance) * x + cm
    nn = -0.5 * (1.0 - gain_imbalance) * xd + cm
    return p, nn


def differential_mode(p, n):
    """The differential signal ``p - n`` (what a differential receiver slices)."""
    return np.asarray(p, float) - np.asarray(n, float)


def common_mode(p, n):
    """The common-mode signal ``(p + n)/2`` (ideally zero; nonzero from skew/imbalance/CM)."""
    return 0.5 * (np.asarray(p, float) + np.asarray(n, float))


def crosstalk_matrix(x, grid, couplings, baud_offsets=None, seeds=None, kind="fext",
                     synchronous=False):
    """Add crosstalk from MULTIPLE aggressors weighted by a coupling vector (one entry per
    aggressor — a column of the coupling matrix for this victim). Aggressors are ASYNCHRONOUS
    by default: each runs at a slightly offset baud so its timing is not locked to the victim
    clock. That matters — a synchronous aggressor's interference is locked to the victim's
    clock, which a receiver's CDR partly tracks out and which then looks like ISI rather than
    crosstalk, so a synchronous default quietly makes crosstalk easier to detect than it is.
    Pass ``synchronous=True`` (or explicit ``baud_offsets``) for the locked case. Returns the
    victim with every aggressor's coupling summed in."""
    x = np.asarray(x, float)
    n = len(x)
    m = len(couplings)
    if grid is None or grid.baud is None:
        raise ValueError("crosstalk_matrix needs grid=Grid(fs, baud, ...)")
    if baud_offsets is None:
        baud_offsets = [0.0] * m if synchronous else [0.017 * (i + 1) for i in range(m)]
    if seeds is None:
        seeds = [7 + i for i in range(m)]
    kinds = kind if isinstance(kind, (list, tuple)) else [kind] * m
    y = x.copy()
    for c, off, sd, kd in zip(couplings, baud_offsets, seeds, kinds):
        n_ui = int(round(n * grid.baud * (1.0 + off) / grid.fs))
        aggr = nrz(n_ui=n_ui, seed=sd, n=n, causal=True)
        y = y + (crosstalk(x, aggr, coupling=c, kind=kd) - x)
    return y


# The lowest corner, as a fraction of Nyquist, at which `signal.butter(1, ...)` still puts the
# corner where it was asked for. MEASURED by bisecting the realised |H| = 0.5 point: 1e-4 lands
# at 1.00x, 1e-5 at 1.05x, and 1e-6 at 0.02x -- below that the sos coefficients saturate toward
# a pure DC block and the corner runs away to DC. The old floor was 1e-4, ten times more
# conservative than the design needs, and it clamped SILENTLY.
AC_COUPLE_MIN_FRAC = 1e-5


def ac_couple(x, fc_frac=0.004, fc_hz=None, grid=None):
    """AC-coupling (series cap) as a 1st-order high-pass -> baseline wander/droop
    that grows with run length. fc_frac is the corner as a fraction of Nyquist.
    Ubiquitous on real serial links. Absolute units: pass fc_hz + grid=Grid(...) to
    state the corner in Hz (converted to a fraction of the grid's Nyquist)."""
    if fc_hz is not None:
        if grid is None:
            raise ValueError("fc_hz requires grid=Grid(...)")
        fc_frac = grid.hz_to_frac_nyquist(fc_hz)
    fc = float(np.clip(fc_frac, AC_COUPLE_MIN_FRAC, 0.5))
    if fc != fc_frac:
        nyq = f" = {fc * 0.5 / grid.dt:.4g} Hz on this grid" if grid is not None else ""
        warnings.warn(
            f"AC-coupling corner clamped: asked fc_frac={fc_frac:g}, using {fc:g}{nyq} "
            f"({fc / fc_frac:.0f}x higher). A first-order corner this far below Nyquist is "
            f"not representable in normalized units -- measured, the sos design tracks the "
            f"request to 1.05x at 1e-5 and collapses to 0.02x by 1e-6. A real link's corner "
            f"is often well below this floor, and on a record shorter than its time constant "
            f"the physical answer is almost no droop, NOT the steeper high-pass this clamp "
            f"applies. Lower the sample rate, lengthen the record, or drop the op.",
            RuntimeWarning, stacklevel=3)
    sos = signal.butter(1, fc, btype="high", output="sos")
    return signal.sosfiltfilt(sos, np.asarray(x, float))


def resonant_reflection(x, grid=None, td_ps=None, td_frac=0.12, f0_ghz=None, f0_frac=0.25,
                        q=10.0, gamma0=0.4, linear=True, guard=None, method="fft"):
    """A single RESONANT discontinuity. `multi_reflection` uses a frequency-flat Γ; real
    discontinuities (a stub, an open) resonate — their reflection coefficient has
    frequency-dependent magnitude AND phase, peaking near a resonant frequency. Here Γ(f)
    is a 2nd-order band-pass shape peaking at f0 with quality `q`, delayed by td.

    Absolute units: pass grid=Grid(...) + td_ps + f0_ghz. Otherwise td_frac (of the record)
    and f0_frac (of Nyquist) are used. `gamma0` scales the peak reflection.

    Applied as a LINEAR convolution (`apply_transfer`): the echo of the record's tail does not
    wrap onto its head. ``linear=False`` restores the pinned-length circular form."""
    x = np.asarray(x, float)
    n = len(x)
    if grid is not None:
        if td_ps is None or f0_ghz is None:
            raise ValueError("grid path needs td_ps and f0_ghz")
        dt, f0 = grid.dt, f0_ghz * 1e9
        td = td_ps * 1e-12
    else:
        # the record-fraction convention: the delay is `td_frac` of THIS record, in samples.
        # It is pinned to the signal's own length here so a longer transform does not move it.
        dt, f0 = 1.0, f0_frac * 0.5
        td = td_frac * n

    def make_H(nfft):
        f = np.fft.rfftfreq(nfft, d=dt)
        sv = 1j * (f / f0)
        G = gamma0 * (sv / q) / (sv ** 2 + sv / q + 1.0)  # |Γ| peaks at f0, with phase
        return 1.0 + G * np.exp(-1j * (2 * np.pi * f * td))

    return apply_transfer(x, make_H, linear=linear, guard=guard, method=method)


def nominal_nonlinearity(x, compression=0.05, level_noise=0.0, rise_fall_ratio=1.0,
                         a_base=0.4, rng=None):
    """Nominal, ALWAYS-ON transmitter imperfections — so a "nominal" (unfaulted) waveform is
    not suspiciously perfect (a too-perfect class is itself a giveaway). Combines three real
    effects, all small by default:

      compression       soft odd compression (``x - c·sign(x)·x²``): outer levels squish, so
                        PAM4 level spacing is no longer exactly uniform (RLM < 1).
      level_noise       additive noise with std ∝ |signal|, so outer levels are noisier.
      rise_fall_ratio   ≠ 1 gives rising and falling edges different slew (rise/fall-time
                        asymmetry) via a nonlinear one-pole; ``a_base`` sets the nominal
                        edge rate that the ratio splits (√-balanced)."""
    x = np.asarray(x, float)
    y = x - compression * np.sign(x) * x ** 2
    if level_noise > 0:
        rng = rng or np.random.default_rng()
        y = y + rng.normal(0.0, 1.0, len(y)) * level_noise * np.abs(x)
    if rise_fall_ratio != 1.0:
        a_r = min(0.95, a_base * np.sqrt(rise_fall_ratio))
        a_f = min(0.95, a_base / np.sqrt(rise_fall_ratio))
        z = np.empty_like(y)
        z[0] = y[0]
        for k in range(1, len(y)):
            d = y[k] - z[k - 1]
            z[k] = z[k - 1] + (a_r if d > 0 else a_f) * d
        y = z
    return y


def multi_reflection(x, td_frac=0.12, gamma_s=0.3, gamma_l=0.4, n_bounce=6,
                     td_samples=None, td_ps=None, grid=None, node="load"):
    """Transmission-line bounce diagram — the lattice/bounce superposition for a mismatched line,
    observed at a chosen NODE. Each round trip is delayed by 2*td.

    node="load" (default): the far-end received signal = incident + reflected train,
        y = x + sum_k (gamma_s*gamma_l)^k * gamma_l * x(t - 2k*td). The direct path plus every
        wave that has bounced off BOTH ends and arrived forward again.
    node="source": the near-end (driver-plane) signal = the launch plus every echo that RETURNS to
        the source, y = x + sum_k (gamma_s*gamma_l)^(k-1) * gamma_l * x(t - 2k*td). This is the
        reverse wave — return loss / an upstream echo — visible one bounce earlier than at the load
        (a single load reflection returns to the source even when the source is matched).

    Both reduce to the incident `x` at a matched termination, so a matched line is bit-identical to
    the forward-only path. `td_frac` is a fraction of the record; `td_samples` overrides with an
    absolute sample delay; `td_ps` (with grid) states the one-way delay in picoseconds."""
    x = np.asarray(x, float)
    nx = len(x)
    if td_ps is not None:
        if grid is None:
            raise ValueError("td_ps requires grid=Grid(...)")
        td_samples = td_ps * 1e-12 * grid.fs
    # A FRACTIONAL delay. Rounding it to a whole sample made the echo position a staircase: on a
    # grid sized from the edge, 100.0, 100.5 and 101.0 ps all landed at 100.0 ps, so a fine sweep
    # did not move the echo and its phase within the eye was quantised to a sample.
    #
    # Taken with the sampler's own kernel, NOT as a phase ramp over the whole record. A phase ramp
    # is exact in frequency and wrong in time for an echo: its impulse response is an untruncated
    # sinc, so energy appears BEFORE the echo arrives. A whole-sample delay stays bit-exact.
    d = float(td_samples) if td_samples is not None else float(td_frac * nx)
    if node not in ("load", "source"):
        raise ValueError("node must be 'load' or 'source'")
    if d <= 0 or gamma_l == 0.0:
        return x.copy()                  # no discontinuity is no mechanism, and must be exact
    y = x.copy()
    g = gamma_s * gamma_l
    for k in range(1, n_bounce + 1):
        shift = 2.0 * d * k
        if shift >= nx:
            break
        weight = (g ** k) if node == "load" else (g ** (k - 1))   # source sees the echo one bounce sooner
        y = y + weight * gamma_l * RS.shift(x, shift, fill=0.0)
    return y


# ---------------------------------------------------------------- jitter physics
def inject_jitter(x, sigma_rj=0.0, a_pj=0.0, f_pj=5.0, dcd=0.0, rng=None,
                  sigma_rj_s=None, a_pj_s=None, f_pj_hz=None, dcd_s=None, grid=None):
    """Physically decomposed jitter via time-axis warp then resample.
      Rj: BANDLIMITED Gaussian phase noise (smooth, so edges shift COHERENTLY),
          renormalized to RMS = sigma_rj samples. (Per-sample white noise would
          scramble the waveform, not jitter its edges — validated.)
      Pj: sinusoidal A*sin(2*pi*f*t) (samples)
      DCD: polarity-dependent offset (rising +dcd/2, falling -dcd/2)
    sigma_rj/a_pj/dcd are in SAMPLES of timing displacement; f_pj in cycles-per-record.

    Absolute units (grid=Grid(...)): sigma_rj_s/a_pj_s/dcd_s state the displacement in
    SECONDS (converted to samples via fs), and f_pj_hz states the periodic-jitter tone
    in Hz (converted to cycles-per-record). Real jitter is quoted in fs/ps and Hz."""
    rng = rng or np.random.default_rng()
    x = np.asarray(x, float)
    n = len(x)
    if grid is not None:
        if sigma_rj_s is not None:
            sigma_rj = grid.to_samples(sigma_rj_s)
        if a_pj_s is not None:
            a_pj = grid.to_samples(a_pj_s)
        if dcd_s is not None:
            dcd = grid.to_samples(dcd_s)
        if f_pj_hz is not None:
            f_pj = grid.hz_to_cycles_per_record(f_pj_hz)
    t = np.linspace(0.0, 1.0, n, endpoint=False)
    disp = np.zeros(n)
    if sigma_rj > 0:
        w = rng.standard_normal(n)
        sos = signal.bessel(2, 0.03, output="sos")          # smooth to ~phase-noise BW
        w = signal.sosfiltfilt(sos, w)
        w = w / (w.std() + 1e-9) * sigma_rj                 # exact RMS = sigma_rj
        disp += w
    if a_pj > 0:
        disp += a_pj * np.sin(2 * np.pi * f_pj * t)
    if dcd != 0:
        slope = np.sign(np.gradient(x))
        disp += (dcd / 2.0) * slope
    if not np.any(disp):
        return np.asarray(x, float).copy()   # no displacement is no mechanism, and must be exact
    # Bandlimited, not linear. The displacement here IS the mechanism -- a stated DCD in
    # picoseconds is a sub-sample number -- so the interpolator's error is an error in the
    # impairment itself.
    src = np.clip(np.arange(n) + disp, 0, n - 1)
    return RS.resample_at(x, src)


# ---------------------------------------------------------------- signaling
# Standard PRBS generator polynomials, as (descending) tap exponents.
# Order 13 is the IEEE 802.3 Clause 120.5.11.2.1 polynomial used by PRBS13Q:
#     G(x) = 1 + x + x^2 + x^12 + x^13
# THE POLYNOMIAL IS NOT A FREE PARAMETER. Several other maximal-length degree-13
# polynomials exist (13,12,11,8 among them) and produce a perfectly valid
# pseudo-random sequence with the right level statistics -- that a protocol
# analyser will never pattern-lock to. If a capture is meant to be analysable by
# an instrument, it has to be the standard's polynomial.
#
# Orders 11 and 23 are the ITU-T O.150 test sequences (O.152 and O.151
# respectively), added because they appear in transport standards:
#     order 11: G(x) = 1 + x^9  + x^11   period 2047
#     order 23: G(x) = 1 + x^18 + x^23   period 8388607
# Every polynomial in this table is asserted PRIMITIVE by wfmsynth.validate (the
# order of x in GF(2)[x]/G is exactly 2^order - 1). A non-primitive tap set yields a
# sequence that looks random, has short sub-periods, and is not the standard's --
# the failure mode the paragraph above is about, so it is checked rather than
# trusted. Orders whose standard polynomial is not stated here are NOT offered:
# guessing a tap set is worse than declining the order.
PRBS_TAPS = {7: (7, 6), 9: (9, 5), 11: (11, 9), 13: (13, 12, 2, 1), 15: (15, 14),
             23: (23, 18), 31: (31, 28)}

# Gray mapping from a bit pair to a PAM4 level, per IEEE 802.3 120.5.11.2.1.
GRAY_PAM4 = {(0, 0): -1.0, (0, 1): -1.0 / 3.0, (1, 1): 1.0 / 3.0, (1, 0): 1.0}


def lfsr(taps, length, seed=1, phase=0):
    """Fibonacci LFSR bits for an ARBITRARY polynomial -- the one bit engine in this module.

    `taps` are the exponents of the feedback polynomial, x^a + x^b + ... + 1, as 1-based tap
    positions; the register width is `max(taps)`. `(7, 6)` is x^7 + x^6 + 1. `prbs` is this
    function with `taps` looked up by order, so a polynomial the table does not carry runs
    through exactly the same arithmetic as one it does -- which is the only way an
    arbitrary-polynomial claim is worth anything.

    A polynomial that is not PRIMITIVE still runs: it produces a shorter cycle, or several
    disjoint cycles, rather than a maximal-length sequence. Nothing here checks primitivity
    (there is no cheap test), so a caller who wants the 2**order-1 period must either take the
    polynomial from a document or measure the period of what comes back.

    `seed` is the initial state and therefore the PHASE; state 0 is the dead state and falls
    back to 1. See `prbs` for what that means for a record's starting position.
    """
    taps = tuple(int(t) for t in taps)
    if not taps or min(taps) < 1:
        raise ValueError(f"taps must be 1-based exponents of the feedback polynomial, not {taps!r}")
    mask = (1 << max(taps)) - 1
    st = int(seed) & mask or 1
    for _ in range(int(phase)):                 # advance without emitting
        b = 0
        for t in taps:
            b ^= (st >> (t - 1)) & 1
        st = ((st << 1) | b) & mask
    out = np.empty(int(length), np.int8)
    for i in range(int(length)):
        b = 0
        for t in taps:
            b ^= (st >> (t - 1)) & 1
        out[i] = st & 1
        st = ((st << 1) | b) & mask
    return out


def prbs(order, length, seed=1, phase=0):
    """Fibonacci LFSR PRBS of the given order. See PRBS_TAPS for polynomials.

    `seed` IS THE PHASE. For a maximal-length LFSR the non-zero state space and the set of
    starting positions are the same set, so a state drawn uniformly from 1..2**order-1 is a
    starting position drawn uniformly from the sequence -- at no cost, which is what
    `random_phase` returns. That matters for training data: every record starting at the same
    place in the pattern lets a model learn the position rather than the signal, and a real
    capture starts wherever the link happened to be.

    `phase` advances the generator that many symbols further before the first output, for when
    a specific offset is wanted rather than a random one. It costs one LFSR step per symbol,
    so a large phase on a high-order PRBS is slow; prefer a random `seed` for randomness.

    CAUTION: `seed=0` masks to state 0, which is the LFSR's dead state, so it falls back to 1
    -- `seed=0` and `seed=1` produce the IDENTICAL sequence. Two records seeded 0 and 1 are
    not two draws. This is kept because output seeded 0 is already pinned downstream.
    """
    # One engine: `lfsr` is this loop with the polynomial passed in rather than looked up, and
    # `max(PRBS_TAPS[order]) == order` for every row, so the register width and every masked
    # shift are identical to what this function did before it delegated. Bit-for-bit, asserted.
    return lfsr(PRBS_TAPS[order], length, seed, phase)


def random_phase(order, rng=None):
    """A starting position in a PRBS of this order, drawn uniformly, as a `seed`.

    The LFSR's non-zero state space is its phase space, so this is an exact uniform draw over
    all 2**order-1 starting positions and costs nothing. Pass the result as `seed`.

        seeds = [P.random_phase(13, rng) for _ in range(n_records)]

    State 0 is excluded because it is the dead state.
    """
    rng = np.random.default_rng() if rng is None else rng
    return int(rng.integers(1, (1 << int(order)) - 1, endpoint=True))


def prbs13q(n_symbols, seed=1):
    """PRBS13Q symbol sequence per IEEE 802.3 Clause 120.5.11.2.1.

    An 8191-symbol repeating sequence formed by Gray coding CONSECUTIVE BIT PAIRS
    taken from TWO repetitions of PRBS13. Two repetitions are required because 8191
    is odd, so bit pairs do not align to the PRBS period -- taking pairs from a
    single repetition silently produces a different (non-conformant) sequence.

    Returns PAM4 levels in {-1, -1/3, +1/3, +1}, tiled to n_symbols.

    Conformance (asserted in wfmsynth.validate against published IEEE figures):
    transition density 0.7501 and level probabilities 0.2499/0.2500/0.2500/0.2500.
    """
    period = 8191
    bits = prbs(13, period * 2, seed)                 # 16382 bits -> 8191 symbols
    pairs = bits.reshape(-1, 2)
    base = np.array([GRAY_PAM4[(int(a), int(b))] for a, b in pairs], dtype=float)
    reps = int(np.ceil(n_symbols / period))
    return np.tile(base, reps)[:n_symbols]


def prbs31q(n_symbols, seed=1):
    """PRBS31Q PAM4 symbols: Gray-coded consecutive bit-pairs from PRBS31 (period 2^31-1), the
    100G/400G-class long-pattern analogue of PRBS13Q. The period (2.1e9 symbols) is far too large to
    materialize, so this generates the first ``n_symbols`` from the LFSR state ``seed`` — which is
    exactly a capture-length SEGMENT starting at the position encoded by that 31-bit state. Returns
    PAM4 levels in {-1,-1/3,+1/3,+1}. (Detection recovers the state, i.e. the position in the pattern.)"""
    bits = prbs(31, int(n_symbols) * 2, seed)             # 2 bits per PAM4 symbol
    pairs = bits.reshape(-1, 2)
    return np.array([GRAY_PAM4[(int(a), int(b))] for a, b in pairs], dtype=float)


def clock_pattern(n_symbols):
    """The alternating 1010... clock pattern, as NRZ levels starting at +1.

    The other sequence compliance work actually uses, and the deliberate contrast case
    to a long PRBS: it has one transition per UI, no runs, and no low-frequency content
    at all, so a channel's response to it is pure Nyquist-tone attenuation with
    essentially NO pattern-dependent ISI. An eye that is open on a clock pattern and
    shut on PRBS31 through the same channel is the signature of ISI rather than loss.
    Independent of seed -- there is nothing random in it."""
    return np.where(np.arange(int(n_symbols)) % 2 == 0, 1.0, -1.0)


# Carrier pattern names. The BINARY (two-level) names spell the LFSR order out --
# `prbs7` .. `prbs31` -- while the QUATERNARY sequences keep IEEE's "Q" suffix
# (`prbs13q`, `prbs31q`). That is the whole naming rule, and it is chosen so that no
# name can be ambiguous about how many levels it carries: `prbs13` and `prbs13q` are
# different sequences on different carriers and now read as different names, rather
# than one being a prefix-shaped guess at the other. Derived from PRBS_TAPS so an
# order added to the table is exposed as a pattern by construction and cannot drift
# out of sync with it.
NRZ_PRBS_PATTERNS = {f"prbs{order}": order for order in sorted(PRBS_TAPS)}
NRZ_PATTERNS = ("legacy", *NRZ_PRBS_PATTERNS, "clock")
PAM4_PATTERNS = ("legacy", "prbs13q", "prbs31q")
# PAM5's real line codes, from `wfmsynth.quinary`. They are patterns of `pam5` rather than a
# carrier kind of their own because they ARE five-level PAM -- what makes them different from
# `pam(5, pattern='uniform')` is the bit-to-level mapping and its statistics, which is exactly
# what a pattern selects everywhere else in this module.
PAM5_PATTERNS = Q.PATTERNS


def _pattern_error(pattern, kind, accepted, other_kind, other_accepted):
    """A ValueError that names what IS accepted, and says so specifically when the
    pattern is real but belongs to the OTHER carrier. A quaternary pattern on an NRZ
    carrier is a user error about how many levels the link has; silently coercing it
    would hide exactly the mistake worth catching. ``kind``/``other_kind`` are the
    carrier kind strings, so the message can be pasted straight back into the call."""
    levels = {"nrz": "binary", "pam3": "ternary", "pam4": "quaternary",
              "pam5": "quinary", "pam8": "octal"}
    accepted_list = ", ".join(repr(p) for p in accepted)
    if pattern in other_accepted:
        return ValueError(
            f"pattern {pattern!r} is a {levels[other_kind]} ({other_kind}) pattern and "
            f"cannot drive a {levels[kind]} ({kind}) carrier; pass kind={other_kind!r}, "
            f"or one of: {accepted_list}")
    return ValueError(f"unknown {kind} pattern {pattern!r}; use one of: {accepted_list}")


# Rise time cannot be faster than the grid supports. BW * t_r ~= 0.35, and the
# highest cutoff _shape_edges can realise is 0.98 * Nyquist, so the fastest
# representable edge is 0.7 / 0.98 ~= 0.714 samples. That is a PHYSICAL limit,
# derived rather than chosen.
TR_NYQUIST_LIMIT_SAMPLES = 0.7 / 0.98

# The floor actually applied by default. 2.0 samples is 2.8x more conservative
# than the grid requires; it is kept as the default only so existing output stays
# bit-identical. Pass floor_samples=TR_NYQUIST_LIMIT_SAMPLES to get the fastest
# edge the grid can carry.
TR_DEFAULT_FLOOR_SAMPLES = 2.0


def resolve_rise_time(tr_frac, spb, floor_samples=TR_DEFAULT_FLOOR_SAMPLES,
                      warn=True):
    """Rise time in samples from a fraction of a symbol, with the clamp made VISIBLE.

    BACKLOG B2: call sites used ``max(tr_frac * spb, 2)``, which silently discards
    the requested rise time whenever ``tr_frac * spb < 2``. At realistic
    samples-per-UI that is the normal case, not an edge case -- at 4.82 samples/UI
    a requested ``tr_frac=0.02`` asks for 0.096 samples and receives 2.0, a
    **20x inflation**, with no indication that anything was overridden. Edge
    shaping then dominates the pulse response and is easily mistaken for channel
    ISI: measured downstream as a 1 UI post-cursor of 0.93 where the generating
    model implied 0.06.

    Returns ``(tr_samples, clamped)`` so callers can record whether the value they
    asked for is the value they got.
    """
    requested = tr_frac * spb
    floor = max(float(floor_samples), TR_NYQUIST_LIMIT_SAMPLES)
    if requested >= floor:
        return requested, False
    if warn:
        warnings.warn(
            f"rise time clamped: tr_frac={tr_frac:g} at {spb:g} samples/UI asks "
            f"for {requested:.3f} samples, floor is {floor:.3f} "
            f"({floor / requested:.1f}x). Edge shaping will dominate the pulse "
            f"response. Raise the sample rate, raise tr_frac, or pass "
            f"floor_samples=TR_NYQUIST_LIMIT_SAMPLES.",
            RuntimeWarning, stacklevel=3)
    return floor, True


# The corner the edge shaper designs to, as a multiple of 0.7/tr_samples, so that the DELIVERED
# 10-90 % rise time is the one asked for.
#
# scipy's `bessel` defaults to `norm='phase'`, whose realised -3 dB corner sits at about 0.66x the
# frequency requested. `instrument.scope_bandwidth` was already fixed for this; the edge shaper was
# not, and a requested rise time came back 1.55x too slow on the causal path and 2.10x on the
# zero-phase one, at every grid density.
#
# The zero-phase path runs the filter TWICE (`sosfiltfilt`) and needs a wider corner for the same
# rise time. Matching the pair's -3 dB point instead is 4.5 % wrong, because two passes are a
# different filter shape with a different bandwidth-rise-time product -- so the factor is solved
# against the rise time itself. `tests/test_edge_rise_time.py` re-solves both and fails on drift.
EDGE_CORNER_CAUSAL = 1.017046
EDGE_CORNER_ZEROPHASE = 1.390084


def _shape_edges(x, tr_samples, causal=False):
    """Band-limit a piecewise-constant symbol stream into finite-rise-time edges.

    The delivered 10-90 % rise time is `tr_samples`, to a fraction of a percent once the grid
    resolves the edge. See `EDGE_CORNER_CAUSAL`.

    causal=False keeps the original zero-phase (sosfiltfilt) shaping, which is
    symmetric and therefore adds pre-cursor as well as post-cursor content.
    causal=True uses a forward-only filter, so the edge shaping cannot move energy
    backwards in time. The library's headline principle is causality, and
    zero-phase edge shaping quietly violates it; the flag defaults to the legacy
    behaviour so existing output is unchanged.

    The forward filter is initialised to STEADY STATE at x[0] (sosfilt_zi). With
    zero initial conditions the output starts at 0 regardless of the signal, which
    plants a full-scale settling transient at the head of every record -- caught by
    the validation check for pre-edge disturbance.
    """
    scale = EDGE_CORNER_CAUSAL if causal else EDGE_CORNER_ZEROPHASE
    sos = signal.bessel(4, min(scale * 0.7 / tr_samples, 0.98), output="sos", norm="mag")
    if not causal:
        return signal.sosfiltfilt(sos, x)
    zi = signal.sosfilt_zi(sos) * x[0]
    y, _ = signal.sosfilt(sos, x, zi=zi)
    return y


@dataclass(frozen=True)
class Jitter:
    """Transmitter jitter, applied to symbol EDGE TIMES *before* pulse shaping — the
    physical source of jitter, so DDJ emerges from the channel for free and noise added
    after the channel is not itself jittered. rj/pj/dcd are in SAMPLES; f_pj in
    cycles-per-record. Use `Jitter.at(grid, rj_s=, pj_s=, f_pj_hz=, dcd_s=)` to specify
    in seconds/Hz.
      rj   random jitter, RMS (bandlimited Gaussian -> coherent edge motion)
      pj   periodic jitter amplitude at frequency f_pj
      dcd  duty-cycle distortion (rising edges +dcd/2, falling -dcd/2)"""
    rj: float = 0.0
    pj: float = 0.0
    f_pj: float = 5.0
    dcd: float = 0.0

    @classmethod
    def at(cls, grid, rj_s=0.0, pj_s=0.0, f_pj_hz=5e6, dcd_s=0.0):
        return cls(rj=grid.to_samples(rj_s), pj=grid.to_samples(pj_s),
                   f_pj=grid.hz_to_cycles_per_record(f_pj_hz), dcd=grid.to_samples(dcd_s))


def _edge_disp(levels_per_ui, jitter, rng):
    """Per-interior-edge time displacement (samples) for a symbol stream."""
    n_ui = len(levels_per_ui); ne = n_ui - 1
    disp = np.zeros(max(ne, 0))
    if ne <= 0:
        return disp
    if jitter.rj > 0:
        w = rng.standard_normal(ne)
        if ne > 8:
            w = signal.sosfiltfilt(signal.bessel(2, 0.1, output="sos"), w)   # coherent phase noise
        disp += w / (w.std() + 1e-9) * jitter.rj
    if jitter.pj > 0:
        k = np.arange(1, ne + 1)
        disp += jitter.pj * np.sin(2 * np.pi * jitter.f_pj * k / n_ui)        # Pj phase at each edge time
    if jitter.dcd != 0:
        disp += (jitter.dcd / 2.0) * np.sign(np.diff(levels_per_ui))          # rising +, falling -
    return disp


def _place_symbols(levels_per_ui, n, spb, jitter=None, rng=None):
    """Map a per-UI symbol-level array onto n samples. jitter=None -> uniform UI
    boundaries (bit-identical legacy). With a Jitter, the interior symbol EDGES are
    displaced (source jitter) before the piecewise-constant stream is returned."""
    n_ui = len(levels_per_ui)
    if jitter is None:
        idx = np.clip((np.arange(n) / spb).astype(int), 0, n_ui - 1)
        return levels_per_ui[idx]
    rng = rng or np.random.default_rng()
    edges = np.arange(1, n_ui) * spb + _edge_disp(levels_per_ui, jitter, rng)
    idx = np.clip(np.searchsorted(edges, np.arange(n), side="right"), 0, n_ui - 1)
    return levels_per_ui[idx]


def carrier_symbols(kind, n_ui, seed=1, pattern="legacy", phase=0, pair=0,
                    partial_response=True, role="master"):
    """The ideal transmitted symbol levels (one per UI) for a carrier — the reference
    stream for realized symbol alignment and any per-symbol ground-truth statistic.
    Deterministic given (kind, n_ui, seed, pattern); the single source of truth that
    `nrz`/`pam4` shape into a waveform.

    NRZ patterns: 'legacy' (== 'prbs7', the default), 'prbs7'/'prbs9'/'prbs11'/
    'prbs13'/'prbs15'/'prbs23'/'prbs31', and 'clock' (1010...).
    PAM4 patterns: 'legacy' (the default), 'prbs13q', 'prbs31q'.

    CHOOSE THE ORDER DELIBERATELY. Channel ISI is a function of pattern HISTORY: the
    long runs and low-frequency content of a long PRBS are what actually close an eye
    through a lossy or reflective channel. PRBS7 repeats every 127 bits -- 23622 times
    inside a 3 M UI record -- and carries almost none of that, so a lossy case rendered
    on it comes out systematically MORE OPEN than the same link would be in the lab.
    Compliance and SI work use PRBS31 (period 2147483647, i.e. never repeating inside
    any realistic record) for exactly this reason. 'prbs7' remains the default only
    because this kernel is pinned by SHA and its output diffed sample-for-sample
    downstream; it is a compatibility default, not a recommendation."""
    n_ui = int(n_ui)
    phase = int(phase)
    if kind == "nrz":
        if pattern in ("legacy", "prbs7"):
            # The historical NRZ stream, kept bit-for-bit: 'legacy' means PRBS7 here
            # and always will, because every NRZ waveform this kernel has ever
            # produced came out of this line.
            return np.where(prbs(7, n_ui, seed, phase) > 0, 1.0, -1.0)
        if pattern == "clock":
            return clock_pattern(n_ui)
        order = NRZ_PRBS_PATTERNS.get(pattern)
        if order is not None:
            return np.where(prbs(order, n_ui, seed, phase) > 0, 1.0, -1.0)
        raise _pattern_error(pattern, "nrz", NRZ_PATTERNS, "pam4", PAM4_PATTERNS)
    if kind == "pam4":
        levels = np.array([-1.0, -1 / 3, 1 / 3, 1.0])
        if pattern == "prbs13q":
            return prbs13q(n_ui, seed)
        if pattern == "prbs31q":
            return prbs31q(n_ui, seed)
        if pattern == "legacy":
            b0 = prbs(7, n_ui, seed, phase); b1 = prbs(9, n_ui, seed + 3, phase)
            return levels[np.clip(b0 * 2 + (b0 ^ b1), 0, 3)]
        raise _pattern_error(pattern, "pam4", PAM4_PATTERNS, "nrz", NRZ_PATTERNS)
    # pamN for any other N. 'pam4' is handled above and keeps its own map, so this branch
    # can never change a PAM4 stream.
    m = re.fullmatch(r"pam(\d+)", str(kind))
    if m:
        n_lv = int(m.group(1))
        if n_lv == 5 and pattern in PAM5_PATTERNS:
            # the four-pair quinary line code, not a uniform PAM5 draw. `pair` picks which of
            # the four coordinates of the same octet stream this lane is.
            return Q.pam5_symbols(pair=pair, n_ui=n_ui, seed=seed, pattern=pattern,
                                  partial_response=partial_response, role=role)
        if pattern in ("legacy", "uniform"):
            return pam_symbols(n_lv, n_ui, seed, phase=phase)
        if pattern == "clock":
            lv = pam_levels(n_lv)
            return np.where(np.arange(n_ui) % 2 == 0, lv[-1], lv[0])
        extra = f", or one of {PAM5_PATTERNS}" if n_lv == 5 else ""
        raise ValueError(f"unknown PAM{n_lv} pattern {pattern!r}; use 'uniform' or 'clock'{extra}")
    raise ValueError(f"unknown carrier kind {kind!r}; use 'nrz', 'pam4', or 'pam<N>'")


def from_symbols(symbols, n=None, tr_frac=0.15, causal=False, jitter=None, rng=None,
                 tr_floor_samples=TR_DEFAULT_FLOOR_SAMPLES):
    """Build a carrier from an ARBITRARY per-UI symbol sequence instead of a PRBS pattern, so a
    line-coded / scrambled / custom stream (see `wfmsynth.coding`) can drive synthesis
    end-to-end. Same edge-shaping and source-jitter pipeline as `nrz`/`pam4`. ``symbols`` is a
    1-D array of per-UI levels; the record length ``n`` sets samples/UI."""
    symbols = np.asarray(symbols, float)
    n = N if n is None else int(n)
    spb = n / len(symbols)
    tr, _ = resolve_rise_time(tr_frac, spb, floor_samples=tr_floor_samples)
    return _shape_edges(_place_symbols(symbols, n, spb, jitter, rng), tr, causal)


def nrz(n_ui=32, tr_frac=0.15, seed=1, n=None, causal=False, jitter=None, rng=None,
        tr_floor_samples=TR_DEFAULT_FLOOR_SAMPLES, pattern="legacy", phase=0):
    """NRZ carrier.

    pattern="legacy"  PRBS7 -- the historical default, kept so existing output stays
                      bit-identical. Its 127-bit period repeats ~23600 times in a 3 M UI
                      record and carries almost no low-frequency content, so a lossy or
                      reflective channel renders MORE OPEN on it than the real link is.
    pattern="prbs31"  the compliance/SI choice: period 2147483647, never repeats inside a
                      realistic record, and exercises the run lengths that actually close
                      an eye. Also 'prbs9'/'prbs11'/'prbs13'/'prbs15'/'prbs23'.
    pattern="clock"   alternating 1010... -- one transition per UI, no runs: the
                      deliberately ISI-free contrast case.
    jitter=Jitter(...) applies transmitter jitter at the symbol edge times (source
                      jitter) before shaping; rng seeds it.
    """
    n = N if n is None else int(n)
    spb = n / n_ui
    lv = carrier_symbols("nrz", n_ui, seed, pattern, phase)
    tr, _ = resolve_rise_time(tr_frac, spb, floor_samples=tr_floor_samples)
    return _shape_edges(_place_symbols(lv, n, spb, jitter, rng), tr, causal)


def pam4(n_ui=32, tr_frac=0.15, seed=1, n=None, causal=False, pattern="legacy",
         tr_floor_samples=TR_DEFAULT_FLOOR_SAMPLES,
         jitter=None, rng=None, phase=0):
    """PAM4 carrier.

    pattern="legacy"  the original PRBS7/PRBS9 Gray-ish map. Not a standard
                      sequence -- fine for shape coverage, will NOT pattern-lock.
    pattern="prbs13q" IEEE 802.3 PRBS13Q (8191 symbols). Use this when the capture
                      has to be analysable by an instrument.
    jitter=Jitter(...) applies transmitter jitter at the symbol edge times (source
                      jitter) before shaping; rng seeds it.
    """
    n = N if n is None else int(n)
    spb = n / n_ui
    syms = carrier_symbols("pam4", n_ui, seed, pattern, phase)
    tr, _ = resolve_rise_time(tr_frac, spb, floor_samples=tr_floor_samples)
    return _shape_edges(_place_symbols(syms, n, spb, jitter, rng), tr, causal)



# ---------------------------------------------------------------- PAM-N, any N
#
# Evenly spaced levels spanning [-1, +1]. PAM4's row is written out rather than computed
# because `np.linspace(-1, 1, 4)` and `[-1, -1/3, 1/3, 1]` differ in the last bit -- 5.6e-17
# -- and every PAM4 waveform this kernel has produced came from the explicit array.
PAM_LEVELS = {
    2: np.array([-1.0, 1.0]),
    3: np.array([-1.0, 0.0, 1.0]),
    4: np.array([-1.0, -1.0 / 3.0, 1.0 / 3.0, 1.0]),
    5: np.array([-1.0, -0.5, 0.0, 0.5, 1.0]),
    8: np.array([-1.0, -5.0 / 7.0, -3.0 / 7.0, -1.0 / 7.0,
                 1.0 / 7.0, 3.0 / 7.0, 5.0 / 7.0, 1.0]),
}


def pam_levels(n_levels):
    """The `n_levels` transmit levels, evenly spaced over [-1, +1].

    An M-level signal has M-1 eyes, and the spacing here is UNIFORM -- which is the ideal
    case. Real drivers compress the outer levels (RLM < 1); that is an impairment and lives
    in `impairments`, not in the level definition.
    """
    n = int(n_levels)
    if n < 2:
        raise ValueError(f"n_levels must be at least 2, not {n}")
    lv = PAM_LEVELS.get(n)
    return lv.copy() if lv is not None else np.linspace(-1.0, 1.0, n)


def pam_symbols(n_levels, n_symbols, seed=1, order=13, phase=0):
    """Uniform PAM-N symbols from a PRBS, by rejection.

    `ceil(log2(N))` bits are drawn per symbol and words landing outside `0..N-1` are
    discarded, which keeps the symbol distribution uniform for an N that is not a power of
    two. Deterministic given (n_levels, n_symbols, seed, order).

    THIS IS NOT A STANDARD'S LINE CODE. 1000BASE-T's PAM5 is 4D-PAM5 with 8B1Q4 coding
    across four pairs, and USB4 v2's PAM3 is 11 bits to 7 ternary symbols; both carry
    spectral shaping and DC balance this does not. What this gives is the right NUMBER of
    levels with the right spacing, uniformly exercised -- which is what an eye, a slicer and
    a level-separation measurement need. A faithful line code is per-standard work; feed one
    through `from_symbols` when it exists.
    """
    n_lv, n_out = int(n_levels), int(n_symbols)
    levels = pam_levels(n_lv)
    if n_lv == 1 << (n_lv.bit_length() - 1) and n_lv > 1:
        bits_per = n_lv.bit_length() - 1          # exact power of two: no rejection needed
    else:
        bits_per = n_lv.bit_length()
    span = 1 << bits_per
    keep = np.empty(0, dtype=np.int64)
    draw = max(int(n_out * span / max(n_lv, 1) * 1.4) + 64, 256)
    while keep.size < n_out:
        bits = (prbs(order, draw * bits_per, seed, phase) > 0).astype(np.int64)
        words = bits[: (bits.size // bits_per) * bits_per].reshape(-1, bits_per)
        idx = (words * (1 << np.arange(bits_per - 1, -1, -1))).sum(axis=1)
        keep = idx[idx < n_lv]
        if keep.size >= n_out:
            break
        draw *= 2                                  # the PRBS is periodic; ask for more of it
        if draw > (n_out + 1) * span * 64:
            raise RuntimeError(f"cannot draw {n_out} PAM{n_lv} symbols from a PRBS{order}")
    return levels[keep[:n_out]]


def pam(n_levels=4, n_ui=32, tr_frac=0.15, seed=1, n=None, causal=False,
        pattern="uniform", tr_floor_samples=TR_DEFAULT_FLOOR_SAMPLES,
        jitter=None, rng=None, phase=0, pair=0, partial_response=True, role="master"):
    """PAM-N carrier for any N >= 2.

    `pam(4, pattern='legacy')` is not the same stream as `pam4(pattern='legacy')` and does
    not try to be: `pam4` keeps its own Gray map and its own PRBS pair because its output is
    pinned. Use `pam4` for PAM4 and this for everything else.

    pattern='uniform'  PRBS-driven, uniform over the N levels (see `pam_symbols`)
    pattern='clock'    the two outer levels alternating: the ISI-free contrast case

    For N = 5 only, the real line code and the standard's transmitter test patterns:
    pattern='8b1q4'    4D-PAM5 / 8B1Q4 -- one pair of the four-pair gigabit link, with the
                       non-uniform symbol statistics the code actually has. `pair` selects
                       which of the four lanes of the same octet stream this is,
                       `role` ('master'|'slave') picks which END of the link this is -- the two
                       ends run DIFFERENT scrambler polynomials, so that and not a second seed
                       is how to get an independent stream -- and `partial_response` applies the
                       transmit shaping (default on, so the levels are the 17 at the MDI rather
                       than the 5 the coder emitted).
    pattern='test_mode_1'..'test_mode_4'
                       the transmitter test-mode symbol sequences of Clause 40.6.1.1.2.
    See `wfmsynth.quinary` for what in those is cited and what is PRELIMINARY.
    """
    n = N if n is None else int(n)
    spb = n / int(n_ui)
    if pattern == "clock":
        lv = pam_levels(n_levels)
        syms = np.where(np.arange(int(n_ui)) % 2 == 0, lv[-1], lv[0])
    elif pattern == "uniform":
        syms = pam_symbols(n_levels, n_ui, seed, phase=phase)
    elif int(n_levels) == 5 and pattern in PAM5_PATTERNS:
        syms = Q.pam5_symbols(pair=pair, n_ui=int(n_ui), seed=seed, pattern=pattern,
                              partial_response=partial_response, role=role)
    else:
        extra = f", or one of {PAM5_PATTERNS}" if int(n_levels) == 5 else ""
        raise ValueError(f"unknown PAM-N pattern {pattern!r}; use 'uniform' or 'clock'{extra}")
    tr, _ = resolve_rise_time(tr_frac, spb, floor_samples=tr_floor_samples)
    return _shape_edges(_place_symbols(syms, n, spb, jitter, rng), tr, causal)



# ---------------------------------------------------------------- analog and arbitrary
#
# Not every waveform is data. A corpus of high-speed serial links has no sine in it, and a
# model trained only on eye diagrams has never seen a clock, a ramp, a supply rail or
# whatever a user's own function produces. These carriers put those on the same grid, through
# the same impairment and acquisition path, so a record of a sine is a record like any other.
#
# THE EDGES ARE BAND-LIMITED. An ideal square has infinite bandwidth and sampling one aliases
# every harmonic above Nyquist back into the record as a spur that is not in the signal. The
# non-sinusoidal shapes here are therefore shaped by the same `_shape_edges` the digital
# carriers use, with `tr_frac` a fraction of the PERIOD.

def timebase(n=None, fs=None, grid=None):
    """Sample instants. SECONDS where a rate is known, else the legacy [0, 1) ramp.

    `am`/`fm`/`chirp` predate absolute units and take a normalised ramp; anything given `fs`
    or a `grid` gets real seconds, which is what a frequency in Hz needs to mean anything.
    """
    if grid is not None:
        n = int(getattr(grid, "n", n) or N)
        fs = float(getattr(grid, "fs", fs) or 0.0) or None
    n = N if n is None else int(n)
    if fs:
        return np.arange(n, dtype=float) / float(fs)
    return np.linspace(0.0, 1.0, n, endpoint=False)


def _cycles_and_t(f_hz, cycles, n, fs, grid):
    t = timebase(n, fs, grid)
    if f_hz is not None:
        span = t[-1] - t[0] + (t[1] - t[0] if t.size > 1 else 0.0)
        return t, float(f_hz), span
    c = 10.0 if cycles is None else float(cycles)
    span = t[-1] - t[0] + (t[1] - t[0] if t.size > 1 else 0.0)
    return t, c / span, span


def sine(f_hz=None, cycles=None, amp=1.0, phase_rad=0.0, offset=0.0,
         n=None, fs=None, grid=None):
    """A sinusoid. Give `f_hz` with `fs`/`grid`, or `cycles` over the record."""
    t, f, _ = _cycles_and_t(f_hz, cycles, n, fs, grid)
    return offset + amp * np.sin(2 * np.pi * f * t + phase_rad)


def square(f_hz=None, cycles=None, amp=1.0, duty=0.5, phase_rad=0.0, offset=0.0,
           tr_frac=0.05, n=None, fs=None, grid=None, causal=False):
    """A square/pulse train, band-limited. `duty` is the high fraction, `tr_frac` the
    transition time as a fraction of the PERIOD."""
    t, f, _ = _cycles_and_t(f_hz, cycles, n, fs, grid)
    ph = (f * t + phase_rad / (2 * np.pi)) % 1.0
    ideal = np.where(ph < float(duty), 1.0, -1.0)
    spp = (1.0 / f) / (t[1] - t[0]) if t.size > 1 else 2.0     # samples per period
    tr = max(float(tr_frac) * spp, 2.0)
    return offset + amp * _shape_edges(ideal, tr, causal)


def triangle(f_hz=None, cycles=None, amp=1.0, symmetry=0.5, phase_rad=0.0, offset=0.0,
             n=None, fs=None, grid=None):
    """A triangle. `symmetry` 0.5 is symmetric; 1.0 is a rising ramp, 0.0 a falling one."""
    t, f, _ = _cycles_and_t(f_hz, cycles, n, fs, grid)
    ph = (f * t + phase_rad / (2 * np.pi)) % 1.0
    r = float(np.clip(symmetry, 1e-6, 1 - 1e-6))
    up = ph / r
    down = (1.0 - ph) / (1.0 - r)
    return offset + amp * (2.0 * np.where(ph < r, up, down) - 1.0)


def sawtooth(f_hz=None, cycles=None, amp=1.0, phase_rad=0.0, offset=0.0,
             tr_frac=0.02, n=None, fs=None, grid=None, causal=False):
    """A ramp with a band-limited flyback, so the retrace does not alias."""
    t, f, _ = _cycles_and_t(f_hz, cycles, n, fs, grid)
    ph = (f * t + phase_rad / (2 * np.pi)) % 1.0
    ideal = 2.0 * ph - 1.0
    spp = (1.0 / f) / (t[1] - t[0]) if t.size > 1 else 2.0
    tr = max(float(tr_frac) * spp, 2.0)
    return offset + amp * _shape_edges(ideal, tr, causal)


def dc(level=0.0, n=None, fs=None, grid=None):
    """A constant. Useful as a rail, and as the degenerate case a pipeline should survive."""
    return np.full(timebase(n, fs, grid).size, float(level))


def arbitrary(fn, n=None, fs=None, grid=None, band_limit_tr=None, causal=False):
    """A carrier from a USER FUNCTION of time.

    `fn(t)` takes the sample instants -- seconds where a rate is known -- and returns the
    waveform. It is called once with the whole array, so it should be vectorised; a scalar
    function works through `np.vectorize` at the usual cost.

        sig = arbitrary(lambda t: np.sign(np.sin(2*np.pi*1e9*t)) * np.exp(-t/1e-6),
                        grid=g, band_limit_tr=4)

    NOTHING IS CHECKED about what comes back except its length. A function with content above
    Nyquist will alias, and that is the caller's business -- pass `band_limit_tr` (a rise time
    in SAMPLES) to shape it first if the function has steps in it.

    IT CANNOT ROUND-TRIP THROUGH A STORED RECIPE, and that is a wart, not a design. `fn` is a
    CALLABLE: it cannot be serialised to JSON, content-addressed, or replayed by anyone who has
    the recipe and not the caller's code, so a record built on this carrier is not reproducible
    from its own provenance -- which every other carrier here is. `compose` refuses the op rather
    than storing something unreplayable, so the failure is loud, but the limitation is real.

    The reproducible route is `wfmsynth.patterns`: register a NAMED generator (the recipe then
    carries the name, its resolved parameters and a hash of the generator's source, so a consumer
    is told what it needs instead of getting different samples), or embed the resolved samples as
    data. Use this function for exploration and for one-off analysis, not for anything that has to
    be rebuilt later from what was stored.
    """
    t = timebase(n, fs, grid)
    y = np.asarray(fn(t), dtype=float).ravel()
    if y.size != t.size:
        raise ValueError(f"fn returned {y.size} samples, expected {t.size}")
    if band_limit_tr:
        y = _shape_edges(y, float(band_limit_tr), causal)
    return y


# ---------------------------------------------------------------- the open-drain line
# `bus.open_drain` combines drivers at LOGIC level -- a wired-AND over ones and zeros. That is the
# protocol. This is the waveform, and on an open-drain bus the two edges are produced by different
# mechanisms and are not each other's mirror:
#
#     fall    a sink pulls the line down through its on-resistance, in parallel with the pull-up.
#             Fast, and its floor is a DIVIDER between the two resistances, not ground.
#     rise    nothing drives it. The pull-up charges the bus capacitance. RC, and slow.
#
# Shaping that as a symmetric slow edge deletes the asymmetry, which is the most characteristic
# thing about the class, and it also deletes the bus's real failure mode: raise the capacitance far
# enough and the line never reaches the input-high threshold inside a bit. A symmetric edge always
# arrives, just later; an RC charge that runs out of time does not arrive at all.
#
# The rise time an open-drain bus specification quotes is measured between 30 % and 70 % of the
# supply, NOT the 10-90 % that driven logic uses. On an exponential charge those are 0.3567 tau and
# 1.2040 tau, so tr(30-70) = ln(0.7/0.3) tau = 0.8473 tau, against 2.1972 tau for 10-90 %. Reading
# one convention's number as the other's is worth a factor of 2.59.
def rc_tau_for_rise_time(tr_s, lo_frac=0.3, hi_frac=0.7):
    """The RC time constant whose exponential charge has a `lo_frac`-to-`hi_frac` rise of `tr_s`.

    Defaults are the 30 %/70 % convention open-drain bus specifications use. Inverting:

        v(t)/V = 1 - exp(-t/tau)   =>   t(f) = -tau ln(1 - f)
        tr     = tau * ln((1 - lo) / (1 - hi))
    """
    lo, hi = float(lo_frac), float(hi_frac)
    if not 0.0 <= lo < hi < 1.0:
        raise ValueError(f"need 0 <= lo_frac < hi_frac < 1, got {lo_frac}, {hi_frac}")
    return float(tr_s) / math.log((1.0 - lo) / (1.0 - hi))


def open_drain_line(sink_on, fs, r_pullup_ohm, c_bus_f, v_dd=3.3, r_sink_ohm=20.0, v0=None,
                    sink_b_on=None, r_sink_b_ohm=None):
    """Line voltage on an open-drain bus, integrated as a first-order RC.

      `sink_on`        boolean per sample: is device A pulling the line down.
      `r_pullup_ohm`   the pull-up to `v_dd`
      `c_bus_f`        total bus capacitance (wiring + every device's input)
      `r_sink_ohm`     device A's on-resistance while it is pulling down
      `v0`             initial line voltage; default is the settled value for the first sample, so
                       a record does not open with an edge nobody asked for
      `sink_b_on`      a SECOND device's boolean per sample (same shape as `sink_on`), or None
                       (the default -- one sink, unchanged from before this parameter existed)
      `r_sink_b_ohm`   device B's on-resistance; defaults to `r_sink_ohm` when B is given

    With one sink, two resistances give two time constants and two targets:

        released       target v_dd,                          tau = Rp * Cb
        A sinking       target v_dd * Rs/(Rp+Rs)  (a divider), tau = (Rp||Rs) * Cb

    With a second sink the bus is WIRED-AND resolved at the ANALOG level, not the logic level
    `bus.open_drain`'s OR of booleans: several devices sharing one line is a resistor network,
    so when A and B sink TOGETHER the target is the divider against their PARALLEL on-resistance
    (Rs*Rs_b/(Rs+Rs_b)) -- lower than either alone, and dominated by whichever sink is stronger
    (lower Ron), which is what "a low anywhere wins" means once the line is a real circuit rather
    than a boolean. Each of the four (A, B) states has its own closed-form target and time
    constant; which one applies is chosen per sample.

    Integrated by exponential stepping, v <- target + (v - target) exp(-dt/tau), which is EXACT for
    a piecewise-constant target rather than an Euler approximation of it -- so the rise time the
    waveform shows is the rise time the arithmetic asked for, at any oversampling ratio above the
    resolvability check below.
    """
    sink = np.asarray(sink_on)
    if sink.dtype != bool:
        sink = sink.astype(bool)
    if sink.ndim != 1:
        raise ValueError(f"sink_on must be 1-D, got shape {sink.shape}")
    rp, rs, cb = float(r_pullup_ohm), float(r_sink_ohm), float(c_bus_f)
    if rp <= 0 or rs <= 0 or cb <= 0:
        raise ValueError("r_pullup_ohm, r_sink_ohm and c_bus_f must all be positive")
    fs = float(fs)
    has_b = sink_b_on is not None
    if has_b:
        sink_b = np.asarray(sink_b_on)
        if sink_b.dtype != bool:
            sink_b = sink_b.astype(bool)
        if sink_b.shape != sink.shape:
            raise ValueError(f"open_drain_line: sink_b_on must be the same shape as sink_on "
                             f"({sink.shape}), got {sink_b.shape}")
        rs_b = float(r_sink_b_ohm) if r_sink_b_ohm is not None else rs
        if rs_b <= 0:
            raise ValueError("r_sink_b_ohm must be positive")
        r_par = (rs * rs_b) / (rs + rs_b)

    def _divider(r):
        target = float(v_dd) * r / (rp + r)
        tau = (rp * r / (rp + r)) * cb
        return target, tau

    tau_hi = rp * cb                                     # released: the slow one
    v_release = float(v_dd), tau_hi
    v_a, tau_a = _divider(rs)
    if has_b:
        v_b, tau_b = _divider(rs_b)
        v_both, tau_both = _divider(r_par)
        # the smallest tau among every sinking case is the hardest edge to resolve -- always the
        # BOTH-on case, since a parallel resistance is never larger than either resistor alone
        tau_lo = tau_both
    else:
        tau_lo = tau_a
    dt = 1.0 / fs
    # Which edge must the grid resolve? Exponential stepping is EXACT for a piecewise-constant
    # target at any dt, so at dt >> tau the line settling inside one sample is the correct answer
    # for that grid rather than an artefact of the integration. What is not acceptable is a grid
    # that cannot show the RISE: that is the edge an open-drain bus specification constrains, it is
    # the parameter this model exists to represent, and a record whose specified edge is invisible
    # does not represent the thing it claims to. So the rise is an error and the fall is a warning.
    #
    # On this bus that ordering is also the permissive one: a pull-up is hundreds of ohms to
    # kilohms and a sink is tens, so tau_hi is always the larger by one to two orders of magnitude,
    # and a grid that resolves the rise is the only requirement in practice. Real captures of this
    # bus routinely under-resolve the fall, and refusing them would be refusing reality.
    if dt > 0.5 * tau_hi:
        raise ValueError(
            f"sample rate {fs:.4g} Sa/s cannot resolve the rise: dt = {dt:.4g} s against "
            f"tau = {tau_hi:.4g} s (Rp * Cb). The rise is the edge this bus specifies, so a grid "
            f"that cannot show it makes the record meaningless. Raise fs above "
            f"{2.0 / tau_hi:.4g} Sa/s, or lower r_pullup_ohm / c_bus_f.")
    if dt > 0.5 * tau_lo:
        warnings.warn(
            f"sample rate {fs:.4g} Sa/s does not resolve the fall: dt = {dt:.4g} s against "
            f"tau = {tau_lo:.4g} s (the fastest sinking case's (Rp||Rs) * Cb). The level is "
            f"correct and the rise is resolved; the falling edge completes inside one sample, "
            f"as it would on a real capture at this rate.", RuntimeWarning, stacklevel=2)
    n = sink.size
    v = np.empty(n, dtype=float)
    if has_b:
        cases = {(False, False): (v_release[0], math.exp(-dt / v_release[1])),
                 (True, False): (v_a, math.exp(-dt / tau_a)),
                 (False, True): (v_b, math.exp(-dt / tau_b)),
                 (True, True): (v_both, math.exp(-dt / tau_both))}
        first = cases[(bool(sink[0]), bool(sink_b[0]))][0] if n else float(v_dd)
        cur = first if v0 is None else float(v0)
        for i in range(n):
            target, k = cases[(bool(sink[i]), bool(sink_b[i]))]
            cur = target + (cur - target) * k
            v[i] = cur
        return v
    k_hi = math.exp(-dt / tau_hi)
    k_lo = math.exp(-dt / tau_a)
    cur = (v_a if (n and sink[0]) else float(v_dd)) if v0 is None else float(v0)
    for i in range(n):
        if sink[i]:
            cur = v_a + (cur - v_a) * k_lo
        else:
            cur = v_dd + (cur - v_dd) * k_hi
        v[i] = cur
    return v


def _resolve_abs_edge(requested_samples, floor_samples=TR_DEFAULT_FLOOR_SAMPLES, warn=True):
    """Like `resolve_rise_time`, for an edge stated directly in samples rather than as a
    fraction of a symbol (`step`/`pulse`/`cmos`, whose edges are `tr_s`/`tr_frac` of the
    RECORD, not of a UI). Same floor, same visible clamp -- a silently inflated edge is the
    same defect `resolve_rise_time`'s docstring describes, on a different set of callers."""
    floor = max(float(floor_samples), TR_NYQUIST_LIMIT_SAMPLES)
    if requested_samples >= floor:
        return requested_samples, False
    if warn:
        warnings.warn(
            f"edge time clamped: {requested_samples:.3f} samples requested, floor is "
            f"{floor:.3f} ({floor / max(requested_samples, 1e-12):.1f}x). Edge shaping will "
            f"dominate the pulse response. Raise the sample rate, or raise the requested "
            f"edge time.", RuntimeWarning, stacklevel=3)
    return floor, True


def _fs_of(fs, grid):
    """The sample rate an absolute-time knob needs, from whichever of `fs`/`grid` was given."""
    if fs:
        return float(fs)
    if grid is not None and getattr(grid, "fs", None):
        return float(grid.fs)
    return None


def step(amp=1.0, offset=0.0, t_step_s=None, t_step_frac=0.5, tr_s=None, tr_frac=0.02,
         n=None, fs=None, grid=None, causal=False):
    """A single band-limited step from ``offset - amp`` to ``offset + amp``.

    ``t_step_s`` (seconds, needs ``fs``/``grid``) or ``t_step_frac`` (fraction of the record,
    default the midpoint) place it; ``tr_s`` (seconds) is the edge's absolute 10-90 % duration,
    else ``tr_frac`` of the RECORD -- there is no period here to be a fraction of, unlike
    `square`'s ``tr_frac``."""
    t = timebase(n, fs, grid)
    nn = t.size
    if t_step_s is not None:
        fs_eff = _fs_of(fs, grid)
        if fs_eff is None:
            raise ValueError("step: t_step_s needs fs= or grid=")
        i_step = int(round(float(t_step_s) * fs_eff))
    else:
        i_step = int(round(float(t_step_frac) * nn))
    ideal = np.where(np.arange(nn) < i_step, -1.0, 1.0)
    if tr_s is not None:
        fs_eff = _fs_of(fs, grid)
        if fs_eff is None:
            raise ValueError("step: tr_s needs fs= or grid=")
        tr, _ = _resolve_abs_edge(float(tr_s) * fs_eff)
    else:
        tr, _ = _resolve_abs_edge(float(tr_frac) * nn)
    return offset + amp * _shape_edges(ideal, tr, causal)


def pulse(amp=1.0, offset=0.0, t_start_s=None, t_start_frac=0.4, width_s=None,
          width_frac=0.1, tr_s=None, tr_frac=0.02, n=None, fs=None, grid=None, causal=False):
    """A single band-limited rectangular pulse -- `square`'s one-shot cousin. ``t_start``/
    ``width`` take the same seconds-or-fraction-of-record pair `step` does."""
    t = timebase(n, fs, grid)
    nn = t.size
    fs_eff = _fs_of(fs, grid)
    if t_start_s is not None:
        if fs_eff is None:
            raise ValueError("pulse: t_start_s needs fs= or grid=")
        i0 = int(round(float(t_start_s) * fs_eff))
    else:
        i0 = int(round(float(t_start_frac) * nn))
    if width_s is not None:
        if fs_eff is None:
            raise ValueError("pulse: width_s needs fs= or grid=")
        w = int(round(float(width_s) * fs_eff))
    else:
        w = int(round(float(width_frac) * nn))
    w = max(w, 1)
    idx = np.arange(nn)
    ideal = np.where((idx >= i0) & (idx < i0 + w), 1.0, -1.0)
    if tr_s is not None:
        if fs_eff is None:
            raise ValueError("pulse: tr_s needs fs= or grid=")
        tr, _ = _resolve_abs_edge(float(tr_s) * fs_eff)
    else:
        tr, _ = _resolve_abs_edge(float(tr_frac) * nn)
    return offset + amp * _shape_edges(ideal, tr, causal)


def exp(amp=1.0, offset=0.0, t_start_s=None, t_start_frac=0.0, tau_s=None, tau_frac=0.1,
        decay=False, n=None, fs=None, grid=None):
    """A single-pole exponential step response: from ``offset`` toward ``offset + amp``
    (``decay=False``, the default) or from ``offset + amp`` back toward ``offset``
    (``decay=True``), starting at ``t_start``, with time constant ``tau``.

    Exact integration of ``v <- target + (v - target) * exp(-dt/tau)`` (see
    `open_drain_line`/`rc_tau_for_rise_time`, the same construction), so a caller can size a
    channel or a probe against a KNOWN RC answer rather than a step and a guessed filter."""
    t = timebase(n, fs, grid)
    nn = t.size
    if t_start_s is not None:
        fs_eff = _fs_of(fs, grid)
        if fs_eff is None:
            raise ValueError("exp: t_start_s needs fs= or grid=")
        i0 = int(round(float(t_start_s) * fs_eff))
    else:
        i0 = int(round(float(t_start_frac) * nn))
    if tau_s is not None:
        fs_eff = _fs_of(fs, grid)
        if fs_eff is None:
            raise ValueError("exp: tau_s needs fs= or grid=")
        tau = max(float(tau_s) * fs_eff, 1e-9)
    else:
        tau = max(float(tau_frac) * nn, 1e-9)
    idx = np.arange(nn, dtype=float)
    dtt = np.clip(idx - i0, 0.0, None)
    progress = 1.0 - np.exp(-dtt / tau)                # 0 at t_start, -> 1
    if decay:
        progress = 1.0 - progress                      # 1 at t_start, -> 0
    y = np.where(idx < i0, (1.0 if decay else 0.0), progress)
    return offset + amp * y


def chirp_sweep(f0_hz=None, f1_hz=None, amp=1.0, offset=0.0, phase_rad=0.0, method="linear",
                n=None, fs=None, grid=None):
    """A frequency sweep in REAL Hz between ``f0_hz`` and ``f1_hz`` over the record's duration.

    This is deliberately a different function from `chirp` below: that one predates absolute
    units, sweeps in CYCLES over a normalised ``[0, 1)`` record, and stays bit-identical (see
    the SKILL file). Undefaulted frequencies fall back to the same 3-to-40-cycles-per-record
    shape `chirp` uses, translated into Hz by the record's own duration, so a caller who omits
    them still gets a sweep rather than a silent single tone."""
    t = timebase(n, fs, grid)
    span = t[-1] - t[0] + (t[1] - t[0] if t.size > 1 else 1.0)
    f0 = 3.0 / span if f0_hz is None else float(f0_hz)
    f1 = 40.0 / span if f1_hz is None else float(f1_hz)
    y = signal.chirp(t, f0=f0, f1=f1, t1=span, method=method, phi=math.degrees(phase_rad))
    return offset + amp * y


def two_tone(f1_hz=None, f2_hz=None, amp1=1.0, amp2=1.0, offset=0.0, phase1_rad=0.0,
             phase2_rad=0.0, n=None, fs=None, grid=None):
    """Two sinusoids summed at real frequencies ``f1_hz``/``f2_hz`` -- the standard two-tone /
    intermodulation stimulus. Undefaulted frequencies fall back to 5 and 11 cycles per record
    (Hz via the record's duration), two frequencies with no small common factor."""
    t = timebase(n, fs, grid)
    span = t[-1] - t[0] + (t[1] - t[0] if t.size > 1 else 1.0)
    f1 = 5.0 / span if f1_hz is None else float(f1_hz)
    f2 = 11.0 / span if f2_hz is None else float(f2_hz)
    return (offset + amp1 * np.sin(2 * np.pi * f1 * t + phase1_rad)
            + amp2 * np.sin(2 * np.pi * f2 * t + phase2_rad))


def analog_noise(rms=0.2, offset=0.0, df=None, pink_frac=0.0, band_lo_hz=None,
                 band_hi_hz=None, n=None, fs=None, grid=None, rng=None):
    """Broadband, optionally band-limited and/or partly 1/f, noise as a carrier in its own
    right -- for a record that is a noise floor rather than a link with a noise floor on it.

    Reuses `impairments.realistic_noise` (heavy tails via `df`, 1/f via `pink_frac`) rather
    than a second noise generator; ``band_lo_hz``/``band_hi_hz`` (needs ``fs``/``grid``) apply
    an additional brick-wall band limit and renormalise back to the requested rms, the same
    two-step `instrument.scope_bandwidth`'s ``'brickwall'`` kind already uses elsewhere."""
    from .impairments import realistic_noise
    t = timebase(n, fs, grid)
    nn = t.size
    y = realistic_noise(nn, rms=1.0, df=(1e6 if df is None else df), pink_frac=pink_frac, rng=rng)
    if band_lo_hz is not None or band_hi_hz is not None:
        fs_eff = _fs_of(fs, grid)
        if fs_eff is None:
            raise ValueError("analog_noise: band_lo_hz/band_hi_hz need fs= or grid=")
        freqs = np.fft.rfftfreq(nn, d=1.0 / fs_eff)
        mask = np.ones_like(freqs, dtype=bool)
        if band_lo_hz is not None:
            mask &= freqs >= float(band_lo_hz)
        if band_hi_hz is not None:
            mask &= freqs <= float(band_hi_hz)
        y = np.fft.irfft(np.fft.rfft(y) * mask, n=nn)
        cur = float(np.sqrt((y * y).mean()))
        if cur > 0:
            y = y / cur
    return offset + rms * y


def cmos(v_lo=0.0, v_hi=3.3, duty=0.5, f_hz=None, cycles=None, tr_s=1e-9, phase_rad=0.0,
         n=None, fs=None, grid=None, causal=False):
    """A rail-to-rail CMOS/PWM waveform IN REAL VOLTS, ``v_lo`` to ``v_hi`` -- unipolar, not
    the normalised +/-1 the other analog carriers use. A clock is ``duty=0.5``; PWM is any
    other duty. See the module docstring above `open_drain_line` for why a unipolar source is
    kept in real volts rather than stretched onto +/-1 (0 V has to stay 0, not become -1).

    ``tr_s`` is the edge's 10-90 % duration in ABSOLUTE SECONDS, unlike `square`'s ``tr_frac``:
    a real gate's output-stage edge time does not shrink as the gate is clocked faster. Compose
    with `probe(r_source=..., c_load_f=...)` for the loaded-output RC pole (an output
    impedance only does something where it drives a load) rather than a knob here."""
    t, f, _ = _cycles_and_t(f_hz, cycles, n, fs, grid)
    fs_eff = _fs_of(fs, grid)
    if fs_eff is None:
        raise ValueError("cmos: needs fs= or grid= (tr_s is an absolute time)")
    ph = (f * t + phase_rad / (2 * np.pi)) % 1.0
    ideal = np.where(ph < float(duty), 1.0, -1.0)
    tr, _ = _resolve_abs_edge(float(tr_s) * fs_eff)
    y01 = 0.5 * (1.0 + _shape_edges(ideal, tr, causal))          # 0 .. 1
    return float(v_lo) + (float(v_hi) - float(v_lo)) * y01


def pass_fet(x, gate_on, rds_on_ohm=5.0, r_load_ohm=1e6, v_rail_hi=None, v_rail_lo=None,
             diode_drop=0.6, diode_on_ohm=1.0):
    """A gate-controlled series element -- a pass-FET or analog switch -- between `x` and a
    load `r_load_ohm`.

      ON  (`gate_on` True):  the channel conducts through `rds_on_ohm`, a plain resistive
          divider against the load: ``x * r_load/(r_load + rds_on)``.
      OFF (`gate_on` False): the channel is OPEN -- HIGH-Z, reads as 0 -- UNLESS the input has
          pushed past a stated rail. A real switch's body diode conducts once the node exceeds
          ``v_rail_hi + diode_drop`` or falls below ``v_rail_lo - diode_drop``, clamping it
          there through the diode's own (small) on-resistance against the same load. Either
          rail defaults to None (no diode on that side, e.g. an unbounded rail).

    `gate_on` is a boolean per sample -- the composed knob is `Vth`: a gate DRIVE carrier
    compared against a threshold upstream of this function (see `compose._op_pass_fet`, which
    renders the gate the same way `open_drain`'s second sink or `crosstalk`'s aggressor do)."""
    x = np.asarray(x, float)
    gate = np.asarray(gate_on, bool)
    on_out = x * (r_load_ohm / (r_load_ohm + rds_on_ohm))
    off_out = np.zeros_like(x)
    diode_gain = r_load_ohm / (r_load_ohm + diode_on_ohm)
    if v_rail_hi is not None:
        hi_clip = float(v_rail_hi) + float(diode_drop)
        off_out = np.where(x > hi_clip, hi_clip * diode_gain, off_out)
    if v_rail_lo is not None:
        lo_clip = float(v_rail_lo) - float(diode_drop)
        off_out = np.where(x < lo_clip, lo_clip * diode_gain, off_out)
    return np.where(gate, on_out, off_out)


def am_modulate(carrier, message, depth=1.0, suppressed=False):
    """AM/ASK/OOK, on real Hz/seconds via whatever generated `carrier`/`message`: a plain
    per-sample combination, so ASK/OOK fall out of choosing a two-level `message` (a `cmos`
    carrier or a digital one) rather than needing their own function.

    ``suppressed=False`` (AM, ASK, OOK): ``(1 + depth*message) * carrier``.
    ``suppressed=True`` (double-sideband suppressed carrier): ``depth * message * carrier`` --
    no carrier term, so the spectrum has no energy left at `fc` itself.
    """
    carrier = np.asarray(carrier, float)
    message = np.asarray(message, float)
    if suppressed:
        return float(depth) * message * carrier
    return (1.0 + float(depth) * message) * carrier


def fm_modulate(message, fc_hz, dev_hz, amp=1.0, phase_rad=0.0, n=None, fs=None, grid=None):
    """FM/FSK in REAL Hz: instantaneous frequency ``fc_hz + dev_hz * message``, generated
    fresh (frequency modulation is not expressible as a per-sample transform of an
    already-rendered fixed-frequency carrier, unlike `am_modulate`) -- FSK falls out of a
    two-level `message` the same way OOK falls out of `am_modulate`.

    This is a NEW function on the Grid, not `physics.fm` (which predates absolute units, takes
    a unitless `beta`, and stays bit-identical -- see the SKILL file)."""
    message = np.asarray(message, float)
    t = timebase(len(message) if n is None else n, fs, grid)
    dt = t[1] - t[0] if t.size > 1 else (1.0 / fs if fs else 1.0)
    phase = 2 * np.pi * fc_hz * t + 2 * np.pi * float(dev_hz) * np.cumsum(message) * dt
    return float(amp) * np.cos(phase + phase_rad)


def pm_modulate(message, fc_hz, dev_rad=1.0, amp=1.0, phase_rad=0.0, n=None, fs=None, grid=None):
    """PM in REAL Hz/radians: phase ``2*pi*fc_hz*t + dev_rad*message``, generated fresh for
    the same reason `fm_modulate` is. A NEW function, not `physics.psk`, which predates
    absolute units and takes a symbol count over a normalised record."""
    message = np.asarray(message, float)
    t = timebase(len(message) if n is None else n, fs, grid)
    return float(amp) * np.cos(2 * np.pi * fc_hz * t + float(dev_rad) * message + phase_rad)


ANALOG_KINDS = ("sine", "square", "triangle", "sawtooth", "dc",
                "step", "pulse", "exp", "chirp", "two_tone", "noise")
# Unipolar, volts-native sources: NOT on the +/-1 amp/offset convention the rest of
# ANALOG_KINDS shares, so `compose._carrier` dispatches them separately.
VOLTS_ANALOG_KINDS = ("cmos",)

# ---------------------------------------------------------------- RF / analog
def am(fc=40.0, fm=3.0, depth=0.6, n=None):
    t = T if n is None else np.linspace(0.0, 1.0, int(n), endpoint=False)
    msg = np.sin(2 * np.pi * fm * t)
    return (1 + depth * msg) * np.cos(2 * np.pi * fc * t)


def fm(fc=40.0, fm_rate=3.0, beta=5.0, n=None):
    t = T if n is None else np.linspace(0.0, 1.0, int(n), endpoint=False)
    msg = np.sin(2 * np.pi * fm_rate * t)
    phase = 2 * np.pi * fc * t + beta * np.cumsum(msg) / len(t) * (2 * np.pi)
    return np.cos(phase)


def psk(fc=40.0, n_sym=16, m=4, seed=1, n=None):
    n = N if n is None else int(n)
    t = np.linspace(0.0, 1.0, n, endpoint=False)
    rng = np.random.default_rng(seed)
    syms = rng.integers(0, m, n_sym)
    spb = n / n_sym
    idx = np.clip((np.arange(n) / spb).astype(int), 0, n_sym - 1)
    ph = syms[idx] * (2 * np.pi / m)
    return np.cos(2 * np.pi * fc * t + ph)


def qam(fc=40.0, n_sym=16, seed=1, n=None):
    n = N if n is None else int(n)
    t = np.linspace(0.0, 1.0, n, endpoint=False)
    rng = np.random.default_rng(seed)
    I = rng.choice([-1, -1 / 3, 1 / 3, 1], n_sym); Q = rng.choice([-1, -1 / 3, 1 / 3, 1], n_sym)
    spb = n / n_sym
    idx = np.clip((np.arange(n) / spb).astype(int), 0, n_sym - 1)
    return I[idx] * np.cos(2 * np.pi * fc * t) - Q[idx] * np.sin(2 * np.pi * fc * t)


def chirp(f0=3.0, f1=40.0, n=None):
    t = T if n is None else np.linspace(0.0, 1.0, int(n), endpoint=False)
    return signal.chirp(t, f0=f0, f1=f1, t1=1.0, method="linear")


def pdn_transient(droop=0.08, tau1=0.03, tau2=0.25, t0=0.3, n=None):
    t = T if n is None else np.linspace(0.0, 1.0, int(n), endpoint=False)
    tt = t - t0
    sag = np.where(tt > 0, droop * (0.5 * np.exp(-tt / tau1) + 0.5 * np.exp(-tt / tau2)), 0.0)
    return 1.0 - sag


def ecg_like(hr=8.0, n=None):
    """Synthetic ECG-ish: Gaussian-bump QRS + P/T waves per beat."""
    n = N if n is None else int(n)
    T = np.linspace(0.0, 1.0, n, endpoint=False)
    x = np.zeros(n)
    for k in range(int(hr)):
        c = (k + 0.5) / hr
        x += 1.0 * np.exp(-((T - c) ** 2) / (2 * 0.004 ** 2))          # R
        x -= 0.15 * np.exp(-((T - c + 0.012) ** 2) / (2 * 0.004 ** 2))  # Q
        x -= 0.15 * np.exp(-((T - c - 0.012) ** 2) / (2 * 0.004 ** 2))  # S
        x += 0.2 * np.exp(-((T - c - 0.05) ** 2) / (2 * 0.012 ** 2))    # T
        x += 0.1 * np.exp(-((T - c + 0.06) ** 2) / (2 * 0.010 ** 2))    # P
    return x


# family registry: name -> (generator thunk taking rng) ; broad + physical
def family_bank():
    return {
        "nrz_si": lambda r: lossy_channel(nrz(seed=r.integers(1, 127)),
                                          length_in=r.uniform(2, 14), tand=r.uniform(0.005, 0.025)),
        "pam4": lambda r: lossy_channel(pam4(seed=r.integers(1, 127)), length_in=r.uniform(2, 10)),
        "nrz_reflect": lambda r: multi_reflection(nrz(seed=r.integers(1, 127)),
                                                  td_frac=r.uniform(0.06, 0.2),
                                                  gamma_s=r.uniform(0.1, 0.4), gamma_l=r.uniform(0.1, 0.5)),
        "am": lambda r: am(fc=r.uniform(25, 55), fm=r.uniform(2, 5), depth=r.uniform(0.3, 0.9)),
        "fm": lambda r: fm(fc=r.uniform(25, 55), fm_rate=r.uniform(2, 5), beta=r.uniform(2, 8)),
        "psk": lambda r: psk(fc=r.uniform(25, 55), m=int(r.choice([2, 4, 8])), seed=r.integers(1, 999)),
        "qam": lambda r: qam(fc=r.uniform(25, 55), seed=r.integers(1, 999)),
        "chirp": lambda r: chirp(f0=r.uniform(1, 5), f1=r.uniform(20, 50)),
        "pdn": lambda r: pdn_transient(droop=r.uniform(0.03, 0.15), t0=r.uniform(0.2, 0.5)),
        "ecg": lambda r: ecg_like(hr=r.uniform(5, 12)),
    }
