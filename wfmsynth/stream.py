"""
wfmsynth.stream — chunked / streaming generation for deep-memory records.

A multi-megapoint record needs several float64 arrays live at once, and a frequency-domain
channel needs a full-length FFT — the memory bottleneck, and deep-memory captures are the
headline use case. Overlap-save FFT convolution applies a channel's FIR while holding only
one chunk (plus the filter tail) live at a time, so the working set is ``O(chunk)`` rather
than ``O(record)``. The result equals the linear convolution truncated to the input length.

    h = channel_fir(apply_fn, n_taps=512)        # a channel's impulse response
    y = stream_convolve(x, h, chunk=1 << 16)     # bounded-memory application
    for blk in stream_blocks(x, h, chunk=1 << 16):  # or consume output chunk by chunk
        write(blk)
"""
from __future__ import annotations

import numpy as np


def _plan(nh, chunk):
    nfft = 1 << int(np.ceil(np.log2(chunk + nh - 1)))
    return nfft, nfft - nh + 1                    # (fft size, valid samples per block)


def stream_blocks(x, h, chunk=1 << 16):
    """Overlap-save FFT convolution as a generator of output chunks. Applies FIR ``h`` to
    ``x`` holding only ~``chunk`` + ``len(h)`` samples live at once — no full-length FFT and
    no whole-record output array. Yields consecutive output blocks whose concatenation is the
    linear convolution truncated to ``len(x)``."""
    x = np.asarray(x, float)
    h = np.asarray(h, float)
    nh, n = len(h), len(x)
    nfft, step = _plan(nh, chunk)
    H = np.fft.rfft(h, nfft)
    prev = np.zeros(nh - 1)                       # last nh-1 input samples (the overlap)
    pos = 0
    while pos < n:
        seg = x[pos:pos + step]
        m = len(seg)
        block = np.zeros(nfft)
        block[:nh - 1] = prev
        block[nh - 1:nh - 1 + m] = seg
        y = np.fft.irfft(np.fft.rfft(block) * H, nfft)
        yield y[nh - 1:nh - 1 + m]
        tail = block[nh - 1:nh - 1 + m]           # this block's input samples
        prev = tail[-(nh - 1):] if m >= nh - 1 else np.concatenate([prev, tail])[-(nh - 1):]
        pos += m


def stream_convolve(x, h, chunk=1 << 16):
    """Overlap-save FFT convolution into a single array (bounded working set). Equals the
    linear convolution of ``x`` with ``h`` truncated to ``len(x)`` — but never builds a
    full-length FFT. See `stream_blocks` for a generator that never holds the whole output."""
    x = np.asarray(x, float)
    out = np.empty(len(x))
    pos = 0
    for blk in stream_blocks(x, h, chunk):
        out[pos:pos + len(blk)] = blk
        pos += len(blk)
    return out


def channel_fir(apply_fn, n_taps=512):
    """Approximate a (linear, time-invariant) channel's FIR by probing it with a unit
    impulse: ``apply_fn(impulse) -> response``, truncated/centred to ``n_taps``. Use the
    result with `stream_convolve` to apply that channel to a deep-memory record in chunks.
    ``apply_fn`` must be linear (e.g. a lossy/S-parameter channel with no added noise)."""
    L = 4 * n_taps
    imp = np.zeros(L)
    imp[0] = 1.0
    h = np.asarray(apply_fn(imp), float)
    return h[:n_taps]
