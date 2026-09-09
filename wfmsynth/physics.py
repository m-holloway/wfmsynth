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
import numpy as np
import warnings

from scipy import signal

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


def _circular_support(loud):
    """Width of the shortest arc of the circle that contains every True in `loud`.

    An impulse response on an FFT grid lives on a circle, and it is not always one-sided: a
    zero-phase response is symmetric about t=0 and a reflection can have a small pre-cursor,
    both of which put samples at NEGATIVE time -- stored at the END of the array. Measuring
    "the last index above threshold" would call those a late tail and report the whole record.
    The support is instead the complement of the LONGEST quiet run, which is the same number
    for a causal response and the right one for a two-sided one."""
    loud = np.asarray(loud, bool)
    n = len(loud)
    idx = np.nonzero(loud)[0]
    if idx.size == 0:
        return 1
    if idx.size == n:
        return n
    first = int(idx[0])
    rolled = np.nonzero(np.roll(loud, -first))[0]     # index 0 is now loud, so no run wraps it
    gaps = np.diff(rolled) - 1
    quiet = max(int(gaps.max()) if gaps.size else 0, n - 1 - int(rolled[-1]))
    return n - quiet


def linear_fft_length(n_signal, n_response, radix=SMOOTH_RADIX):
    """The transform length for a LINEAR convolution of `n_signal` samples with an
    `n_response`-sample impulse response: the next `radix`-smooth length at or above
    ``n_signal + n_response - 1``."""
    return next_smooth_length(int(n_signal) + int(n_response) - 1, radix)


def apply_transfer(x, make_H, linear=True, guard=None, radix=SMOOTH_RADIX,
                   rel=RESPONSE_REL, n0=PROBE_N0, probe_max=PROBE_MAX):
    """Apply a frequency response to `x` as a LINEAR convolution, and return ``len(x)`` samples.

    `make_H(nfft)` returns the response on the ``rfft`` grid of length `nfft` (same sample
    interval whatever `nfft` is). The record is zero-padded to a 5-smooth length at or above
    ``len(x) + len(h) - 1``, multiplied there, and truncated back -- so no part of the response
    to the record's tail lands on its head.

    `guard` fixes ``len(h)`` in samples instead of measuring it (`response_extent` does the
    measuring). ``linear=False`` restores the pinned-length CIRCULAR convolution, which is what
    this module did before and is kept so the two paths can be compared directly."""
    x = np.asarray(x, float)
    n = len(x)
    if not linear:
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
    nfft = linear_fft_length(n, max(1, int(guard)), radix)
    xp = np.zeros(nfft)
    xp[:n] = x
    X = np.fft.rfft(xp)
    del xp
    X *= make_H(nfft)                       # in place: one fewer record-sized temporary
    y = np.fft.irfft(X, nfft)
    del X
    return y[:n].copy()                     # a copy, not a view on the padded buffer


# ---------------------------------------------------------------- channel physics
def _min_phase_H(Hmag, n=None):
    """Causal minimum-phase complex response from a real magnitude |H| (rfft bins),
    via the cepstral / Hilbert relation so loss and phase are physically LINKED
    (Kramers-Kronig; the Djordjevic-Sarkar-style causal channel). A real magnitude-
    only response is zero-phase -> non-causal (symmetric pre/post-ringing); the
    minimum-phase version concentrates the response AFTER t=0 (asymmetric post-cursor
    ISI), which is what a real dispersive interconnect does.

    `n` is the full (two-sided) transform length; inferred from Hmag when omitted."""
    if n is None:
        n = 2 * (len(Hmag) - 1)
    # Every step below is in place or explicitly freed. This runs at the FULL two-sided length
    # of the record, so a deep record's cepstrum is the single largest allocation in a render;
    # the arithmetic is untouched (elementwise ops in the same order), only the temporaries go.
    if n % 2 == 0:
        # even length: rfft has a distinct Nyquist bin; drop DC+Nyquist from the mirror.
        mag_full = np.concatenate([Hmag, Hmag[-2:0:-1]])  # symmetric, length n
    else:
        # odd length: no Nyquist bin; mirror all bins except DC.
        mag_full = np.concatenate([Hmag, Hmag[-1:0:-1]])  # symmetric, length n
    mag_full += 1e-12
    np.log(mag_full, out=mag_full)                        # logmag
    spec = np.fft.ifft(mag_full)
    del mag_full
    c = spec.real.copy()                                  # a copy, not a view: frees the complex half
    del spec
    w = np.zeros(n)
    if n % 2 == 0:
        w[0] = 1.0; w[1:n // 2] = 2.0; w[n // 2] = 1.0    # causal folding
    else:
        w[0] = 1.0; w[1:(n + 1) // 2] = 2.0               # causal folding
    c *= w                                                # exactly `c * w`, without its output array
    del w
    spec = np.fft.fft(c)
    del c
    return np.exp(spec, out=spec)                         # complex min-phase, length n


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
                  trend=None, trend_floor_db=80.0, linear=True, guard=None):
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
        return _min_phase_H(Hmag, nfft)[:nfft // 2 + 1]

    return apply_transfer(x, make_H, linear=linear, guard=guard)


def crosstalk(x, aggressor, coupling=0.12, kind="fext", td_frac=0.05):
    """Add coupled noise from a neighboring (aggressor) line. FEXT couples through
    the mutual C/L as the DERIVATIVE of the aggressor (∝ d/dt); NEXT is a delayed,
    broadband coupling. `coupling` scales the aggressor relative to x's span. Real
    high-speed links are frequently crosstalk-limited — a major missing effect."""
    x = np.asarray(x, float)
    n = len(x)
    a = np.asarray(aggressor, float)
    if kind == "fext":
        k = np.gradient(a)
    else:                                                 # next: delayed coupling
        d = int(td_frac * n); k = np.zeros(n); k[d:] = a[:n - d]
    k = k / (np.abs(k).max() + 1e-9)
    return x + coupling * (np.ptp(x) + 1e-9) * k


def de_emphasis_taps(db):
    """2-tap Tx FFE weights ``[main, post]`` realizing a de-emphasis of ``db`` — the
    standardized preset real transmitters expose. The first bit after a transition is at
    full amplitude and steady-state bits are reduced, with 20·log10(V_transition/V_steady)
    = db. Use with `tx_ffe(..., pre=0)`."""
    r = 10 ** (db / 20.0)
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
        idx = np.arange(n)
        y = np.interp(idx - dev, idx, y, left=y[0], right=y[-1])
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
    xd = np.interp(idx - skew_samples, idx, x, left=x[0], right=x[-1])
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


def ac_couple(x, fc_frac=0.004, fc_hz=None, grid=None):
    """AC-coupling (series cap) as a 1st-order high-pass -> baseline wander/droop
    that grows with run length. fc_frac is the corner as a fraction of Nyquist.
    Ubiquitous on real serial links. Absolute units: pass fc_hz + grid=Grid(...) to
    state the corner in Hz (converted to a fraction of the grid's Nyquist)."""
    if fc_hz is not None:
        if grid is None:
            raise ValueError("fc_hz requires grid=Grid(...)")
        fc_frac = grid.hz_to_frac_nyquist(fc_hz)
    fc = float(np.clip(fc_frac, 1e-4, 0.5))
    sos = signal.butter(1, fc, btype="high", output="sos")
    return signal.sosfiltfilt(sos, np.asarray(x, float))


def resonant_reflection(x, grid=None, td_ps=None, td_frac=0.12, f0_ghz=None, f0_frac=0.25,
                        q=10.0, gamma0=0.4, linear=True, guard=None):
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

    return apply_transfer(x, make_H, linear=linear, guard=guard)


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
        td_samples = round(td_ps * 1e-12 * grid.fs)
    d = int(td_samples) if td_samples is not None else int(td_frac * nx)
    if node not in ("load", "source"):
        raise ValueError("node must be 'load' or 'source'")
    y = x.copy()
    g = gamma_s * gamma_l
    for k in range(1, n_bounce + 1):
        shift = 2 * d * k
        if shift >= nx:
            break
        refl = np.zeros_like(x)
        refl[shift:] = x[:nx - shift]
        weight = (g ** k) if node == "load" else (g ** (k - 1))   # source sees the echo one bounce sooner
        y = y + weight * gamma_l * refl
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
    src = np.clip(np.arange(n) + disp, 0, n - 1)
    return np.interp(src, np.arange(n), x)


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


def prbs(order, length, seed=1):
    """Fibonacci LFSR PRBS of the given order. See PRBS_TAPS for polynomials."""
    taps = PRBS_TAPS[order]
    st = seed & ((1 << order) - 1) or 1
    out = np.empty(length, np.int8)
    for i in range(length):
        b = 0
        for t in taps:
            b ^= (st >> (t - 1)) & 1
        out[i] = st & 1
        st = ((st << 1) | b) & ((1 << order) - 1)
    return out


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


def _pattern_error(pattern, kind, accepted, other_kind, other_accepted):
    """A ValueError that names what IS accepted, and says so specifically when the
    pattern is real but belongs to the OTHER carrier. A quaternary pattern on an NRZ
    carrier is a user error about how many levels the link has; silently coercing it
    would hide exactly the mistake worth catching. ``kind``/``other_kind`` are the
    carrier kind strings, so the message can be pasted straight back into the call."""
    levels = {"nrz": "binary", "pam4": "quaternary"}
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


def _shape_edges(x, tr_samples, causal=False):
    """Band-limit a piecewise-constant symbol stream into finite-rise-time edges.

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
    sos = signal.bessel(4, min(0.7 / tr_samples, 0.98), output="sos")
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


def carrier_symbols(kind, n_ui, seed=1, pattern="legacy"):
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
    if kind == "nrz":
        if pattern in ("legacy", "prbs7"):
            # The historical NRZ stream, kept bit-for-bit: 'legacy' means PRBS7 here
            # and always will, because every NRZ waveform this kernel has ever
            # produced came out of this line.
            return np.where(prbs(7, n_ui, seed) > 0, 1.0, -1.0)
        if pattern == "clock":
            return clock_pattern(n_ui)
        order = NRZ_PRBS_PATTERNS.get(pattern)
        if order is not None:
            return np.where(prbs(order, n_ui, seed) > 0, 1.0, -1.0)
        raise _pattern_error(pattern, "nrz", NRZ_PATTERNS, "pam4", PAM4_PATTERNS)
    if kind == "pam4":
        levels = np.array([-1.0, -1 / 3, 1 / 3, 1.0])
        if pattern == "prbs13q":
            return prbs13q(n_ui, seed)
        if pattern == "prbs31q":
            return prbs31q(n_ui, seed)
        if pattern == "legacy":
            b0 = prbs(7, n_ui, seed); b1 = prbs(9, n_ui, seed + 3)
            return levels[np.clip(b0 * 2 + (b0 ^ b1), 0, 3)]
        raise _pattern_error(pattern, "pam4", PAM4_PATTERNS, "nrz", NRZ_PATTERNS)
    raise ValueError(f"unknown carrier kind {kind!r}; use 'nrz' or 'pam4'")


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
        tr_floor_samples=TR_DEFAULT_FLOOR_SAMPLES, pattern="legacy"):
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
    lv = carrier_symbols("nrz", n_ui, seed, pattern)
    tr, _ = resolve_rise_time(tr_frac, spb, floor_samples=tr_floor_samples)
    return _shape_edges(_place_symbols(lv, n, spb, jitter, rng), tr, causal)


def pam4(n_ui=32, tr_frac=0.15, seed=1, n=None, causal=False, pattern="legacy",
         tr_floor_samples=TR_DEFAULT_FLOOR_SAMPLES,
         jitter=None, rng=None):
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
    syms = carrier_symbols("pam4", n_ui, seed, pattern)
    tr, _ = resolve_rise_time(tr_frac, spb, floor_samples=tr_floor_samples)
    return _shape_edges(_place_symbols(syms, n, spb, jitter, rng), tr, causal)


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
