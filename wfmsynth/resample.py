"""Bandlimited interpolation at arbitrary real sample positions.

Sub-sample displacement is the mechanism behind jitter, duty-cycle distortion, intra-pair skew,
a free-running sample clock and a fractional reflection delay, so how it is interpolated is
physics rather than convenience. Linear interpolation is cheap and wrong in a way that scales
with how sharp the content is. MEASURED against the kernel here, on band-limited content, for a
half-sample shift:

    samples per symbol      4        10       32
    linear error        1.52 %    0.30 %   0.024 %      of peak-to-peak

On a link graded on a 6 mV eye out of 800 mV -- 0.75 % -- the coarse case is twice the quantity
being measured.

This module holds the kernel so `physics` and `instrument` can both use it without one importing
the other. `instrument` re-exports the names it published first.
"""
from __future__ import annotations

import math

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.special import i0 as _i0

HALF_WIDTH = 32          # taps each side; the kernel is 2*half_width long
CUTOFF = 0.45            # kernel cutoff as a fraction of fs
BETA = 13.0              # Kaiser beta
PASSBAND_FRAC = 0.386    # MEASURED: exact to about -130 dB below this, useless above it


# The kernel is smooth, so it is evaluated once onto a dense table and interpolated from there
# rather than calling sinc and a Bessel i0 for every output sample x every tap. At this spacing the
# table's own interpolation error is below 1e-9 of full scale -- four orders under the kernel's own
# -130 dB floor -- and it turns the hot path into a gather and two multiplies.
_TABLE_PER_SAMPLE = 2048


def _kernel_table(half_width, cutoff, beta):
    n = int(half_width) * _TABLE_PER_SAMPLE
    dt = np.arange(-n, n + 1, dtype=float) / _TABLE_PER_SAMPLE
    return np.sinc(2.0 * cutoff * dt) * _kaiser(dt, half_width, beta)


_TABLES: dict[tuple, np.ndarray] = {}


def _table_for(half_width, cutoff, beta):
    key = (int(half_width), float(cutoff), float(beta))
    t = _TABLES.get(key)
    if t is None:
        t = _TABLES[key] = _kernel_table(*key)
    return t


def _weights(dt, half_width, cutoff, beta):
    """Kernel values at offsets `dt`, read off the table with linear interpolation.

    The general form, for arbitrary offsets. `resample_at`'s hot path does not use it -- see
    `_polyphase_for` for the factored version and why the two agree."""
    tab = _table_for(half_width, cutoff, beta)
    mid = (tab.size - 1) // 2
    pos = np.clip(dt * _TABLE_PER_SAMPLE + mid, 0.0, tab.size - 1.000001)
    i = pos.astype(np.int64)
    f = pos - i
    return tab[i] * (1.0 - f) + tab[i + 1] * f


# The kernel weights for a whole output sample are a ROW of a polyphase table, not 2*half_width
# independent lookups. The offset of tap j is ``dt_j = frac - tap_j``, so its table position is
# ``(frac - tap_j)*TPS + mid``; every ``tap_j`` is an integer and ``TPS`` is an integer, so
#
#     the FRACTIONAL part of that position is the same for every tap (it is frac*TPS's), and
#     the INTEGER parts differ by exactly tap_j*TPS.
#
# So one phase index per output sample selects a contiguous row of precomputed weights, and the
# whole (chunk x 2*half_width) ladder of offset/clip/floor/gather temporaries the general form
# builds -- about ten record-shaped arrays per chunk, which is what made this the slowest stage
# in a long render -- collapses to one row gather. Same table, same linear interpolation, same
# arithmetic; only factored. MEASURED: 3.7-4.0x faster, agreeing to 1.6e-15 relative, which is
# three orders under the byte-identity gate's own 3.8e-13 round-off band.
_POLY: dict[tuple, tuple] = {}


def _polyphase_for(half_width, cutoff, beta):
    """``(lo, hi, taps, lo_sum, hi_sum)``. `lo` and `hi` are each
    ``(_TABLE_PER_SAMPLE, 2*half_width)``: the two table rows that the per-sample linear
    interpolation blends, for every phase. `lo_sum`/`hi_sum` are their row sums, which is all
    the per-sample normaliser needs. Cached per kernel shape."""
    key = (int(half_width), float(cutoff), float(beta))
    got = _POLY.get(key)
    if got is None:
        tab = _table_for(half_width, cutoff, beta)
        mid = (tab.size - 1) // 2
        taps = np.arange(-int(half_width) + 1, int(half_width) + 1)
        phase = np.arange(_TABLE_PER_SAMPLE, dtype=float)
        # exactly `_weights`' clipped position, evaluated on the phase grid instead of per sample
        pos = np.clip(phase[:, None] - taps[None, :] * _TABLE_PER_SAMPLE + mid,
                      0.0, tab.size - 1.000001)
        i = pos.astype(np.int64)
        lo, hi = tab[i], tab[i + 1]
        # The per-sample NORMALISER is a function of the phase alone: the weights are
        # ``lo[ph]*(1-fr) + hi[ph]*fr``, so their sum is ``sum(lo[ph])*(1-fr) + sum(hi[ph])*fr``
        # and both sums can be taken once, here, over the 2048 phases instead of once per
        # output sample over 64 taps. MEASURED: 50.2 -> 5.4 us per 2048-sample chunk, 11.5 % of
        # the whole resample, agreeing to 6e-16 (the two differ only in the order the same 64
        # numbers are added).
        got = _POLY[key] = (lo, hi, taps, lo.sum(axis=1), hi.sum(axis=1))
    return got


def resample_at(x, src, half_width=HALF_WIDTH, cutoff=CUTOFF, beta=BETA, chunk=2048):
    """``out[k] = x(src[k])`` for arbitrary real positions `src`, by Kaiser-windowed sinc.

    Normalised per output sample so the DC gain is exactly 1, and linear in `x` (`src` does not
    depend on it), so it obeys superposition.

    Exact to about -130 dB out to ``PASSBAND_FRAC = 0.386*fs``, which is the Kaiser passband edge
    for this length, and USELESS above it. Content above the passband is the one way to misuse
    this: pass a bigger `half_width` (the transition narrows as 1/half_width) or resample from a
    finer grid.

    ENDS: an output sample within `half_width` of either end has no kernel support there, and the
    indices are CLAMPED (a hold at ``x[0]`` / ``x[-1]``). That is exact for a record whose ends are
    a settled quiescent line -- which is what `Signal.lead_in` renders and discards -- and an edge
    artefact over `half_width` samples for a record that starts mid-pattern.

    `chunk` bounds the working set and is INVISIBLE in the output (pinned by
    `tests/test_resample_polyphase.py`). It is the one knob here that trades memory against time,
    and the default is set for memory because this stage is reached by `acquire`, `sample_clock`,
    `dcd` and `shift`, so a deep record pays it on the way to a GUI or a demo. MEASURED at 2 M
    samples, peak as a multiple of the record: 1.08x at chunk=512 (369 ms), **1.32x at 2048
    (317 ms)**, 1.64x at 4096 (311 ms), 2.28x at 8192 (306 ms), 21.5x at 131072 (306 ms). Time is
    flat above a few thousand and memory is not, so the default sits at the knee: 3 % slower than
    the fastest setting for 42 % less peak.
    """
    x = np.asarray(x, dtype=float)
    src = np.asarray(src, dtype=float)
    n = x.size
    if n == 0:
        raise ValueError("nothing to resample")
    hw = int(half_width)
    lo, hi, taps, lo_sum, hi_sum = _polyphase_for(half_width, cutoff, beta)
    # The tap window of an INTERIOR output sample lies wholly inside the record, so it is a
    # stride view on `x` rather than a (chunk x 2*half_width) gathered copy. Only samples whose
    # window would run off an end need the clamped index path, and there are normally a
    # kernel's worth of those. Taking the view instead of the copy is what keeps this stage's
    # peak memory flat in the record length rather than a multiple of it.
    window = sliding_window_view(x, 2 * hw) if n >= 2 * hw else None
    out = np.empty(src.size, dtype=float)
    for start in range(0, src.size, chunk):
        s = src[start:start + chunk]
        base = np.floor(s).astype(np.int64)
        p = (s - base) * _TABLE_PER_SAMPLE
        ph = p.astype(np.int64)
        np.clip(ph, 0, _TABLE_PER_SAMPLE - 1, out=ph)
        fr = (p - ph)[:, None]
        w = lo[ph] * (1.0 - fr) + hi[ph] * fr

        if window is None:
            seg = x[np.clip(base[:, None] + taps[None, :], 0, n - 1)]
        else:
            inner = (base >= hw - 1) & (base <= n - 1 - hw)
            if inner.all():
                seg = window[base - hw + 1]
            else:
                seg = np.empty(w.shape)
                bi = base[inner]
                if bi.size:
                    seg[inner] = window[bi - hw + 1]
                edge = ~inner
                seg[edge] = x[np.clip(base[edge][:, None] + taps[None, :], 0, n - 1)]

        num = np.einsum("ij,ij->i", w, seg)
        f1 = fr[:, 0]
        den = lo_sum[ph] * (1.0 - f1) + hi_sum[ph] * f1      # phase-only; see `_polyphase_for`
        out[start:start + chunk] = num / np.where(den == 0.0, 1.0, den)
    return out


def _kaiser(dt, half_width, beta):
    """The Kaiser window evaluated at each tap's offset, zero outside the kernel."""
    r = dt / half_width
    inside = np.abs(r) <= 1.0
    arg = np.sqrt(np.clip(1.0 - r * r, 0.0, None))
    return np.where(inside, _i0(beta * arg) / _i0(beta), 0.0)


def shift(x, samples, half_width=HALF_WIDTH, fill="hold"):
    """Delay `x` by `samples` (a single real number).

    A whole-sample delay is taken exactly rather than through the kernel, so an integer request
    stays bit-exact and cannot pick up interpolation error it does not need.

    `fill` says what lies outside the record, and it is a physical choice rather than a detail.
    "hold" repeats the end sample, which is exact for a record whose ends are a settled quiescent
    line -- what `Signal.lead_in` renders and discards. `fill=0.0` says nothing is there, which is
    what an ECHO needs: the reflection of a record that had not started yet has not arrived, and
    holding the first sample would invent a precursor that the line never carried.
    """
    x = np.asarray(x, dtype=float)
    d = float(samples)
    if d == 0.0:
        return x.copy()
    zero = not isinstance(fill, str)
    if float(d).is_integer():
        k = int(d)
        y = np.zeros_like(x) if zero else np.empty_like(x)
        if k > 0:
            if not zero:
                y[:min(k, x.size)] = x[0]
            if k < x.size:
                y[k:] = x[:x.size - k]
        else:
            k = -k
            if k < x.size:
                y[:x.size - k] = x[k:]
            if not zero:
                y[max(x.size - k, 0):] = x[-1]
        return y
    return _shift_const(x, d, half_width, fill, zero)


def _shift_const(x, d, half_width, fill, zero):
    """One fractional delay applied to a whole record.

    The kernel does not depend on the output sample here -- every one wants the same fractional
    offset -- so the 64 weights are computed ONCE and applied as a fixed-tap filter, instead of
    building an (n x 64) matrix of kernel values. That is the difference between 0.16 s and a few
    milliseconds on a 131k-sample record.
    """
    n = x.size
    taps = np.arange(-half_width + 1, half_width + 1)
    base = int(math.floor(-d))
    frac = -d - base
    w = _weights(frac - taps.astype(float), half_width, CUTOFF, BETA)
    w = w / w.sum()
    lo = base - half_width + 1
    hi = base + half_width
    pad_l = max(0, -lo)
    pad_r = max(0, hi)
    left = float(fill) if (zero and d > 0) else x[0]
    right = float(fill) if (zero and d < 0) else x[-1]
    xp = np.concatenate([np.full(pad_l, left), x, np.full(pad_r, right)])
    y = np.zeros(n)
    for j, t in enumerate(taps):
        start = pad_l + base + int(t)
        y += w[j] * xp[start:start + n]
    return y


def _shift_via_kernel(x, d, half_width, fill, zero):
    if zero:
        # Pad ASYMMETRICALLY. Only the side the signal is coming FROM is unknown -- for a delay
        # that is the time before the record started, and `fill` is what was there. The far end
        # is padded by holding, because the record's own last sample is the best statement about
        # what the line was doing then; filling it would roll the tail off and invent an ending.
        pad = half_width + int(math.ceil(abs(d))) + 1
        if d > 0:
            xp = np.concatenate([np.full(pad, float(fill)), x, np.full(pad, x[-1])])
        else:
            xp = np.concatenate([np.full(pad, x[0]), x, np.full(pad, float(fill))])
        src = np.arange(pad, pad + x.size, dtype=float) - d
        return resample_at(xp, src, half_width=half_width)
    return resample_at(x, np.arange(x.size, dtype=float) - d, half_width=half_width)


def displace(x, dev, half_width=HALF_WIDTH):
    """Displace each sample of `x` by its own `dev[k]` samples -- the jitter/DCD mechanism.

    `dev` is positive for a LATER edge, matching the `np.interp(idx - dev, ...)` convention this
    replaced.
    """
    x = np.asarray(x, dtype=float)
    dev = np.asarray(dev, dtype=float)
    if dev.size == 1:
        return shift(x, float(dev), half_width=half_width)
    if dev.shape != x.shape:
        raise ValueError(f"dev is {dev.shape}, x is {x.shape}")
    if not np.any(dev):
        return x.copy()
    return resample_at(x, np.arange(x.size, dtype=float) - dev, half_width=half_width)
