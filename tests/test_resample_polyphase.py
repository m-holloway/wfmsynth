"""`resample_at` is the acquisition/jitter/DCD hot path, and it was the single slowest stage in
a long render: `acquire`, `sample_clock`, `dcd`, `shift` and a fractional reflection delay all
reach it, so most realistic instrument chains pay it.

The weight computation is factored as a POLYPHASE table gather. The identity that makes that
exact: the table position of tap `j` is ``(frac - tap_j)*TPS + mid``, every ``tap_j`` is an
INTEGER and the stride ``TPS`` is an integer, so the fractional part of that position is the
SAME for every tap and the integer parts differ by exactly ``tap_j*TPS``. The per-output-sample
work therefore collapses from a (chunk x 2*half_width) offset/clip/floor/gather to one phase
index plus a contiguous row gather -- the same table, the same linear interpolation, the same
arithmetic, only factored.

These tests pin the SPEC (the documented kernel, evaluated directly) rather than the incumbent
implementation, so they would still be meaningful if the internals changed again.
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy.special import i0

from wfmsynth import resample as RS


def _reference(x, src, half_width=RS.HALF_WIDTH, cutoff=RS.CUTOFF, beta=RS.BETA):
    """The documented kernel, evaluated directly one output sample at a time.

    Deliberately slow and obvious: a windowed sinc, normalised per output sample, with the ends
    clamped. No table at all, so agreement with it is bounded by the TABLE's interpolation error
    rather than by float round-off -- see `SPEC_TOL` for what that measures."""
    x = np.asarray(x, float)
    n = x.size
    taps = np.arange(-half_width + 1, half_width + 1)
    out = np.empty(len(src))
    for k, s in enumerate(src):
        base = int(np.floor(s))
        dt = (s - base) - taps
        r = dt / half_width
        win = np.where(np.abs(r) <= 1.0,
                       i0(beta * np.sqrt(np.clip(1.0 - r * r, 0.0, None))) / i0(beta), 0.0)
        w = np.sinc(2.0 * cutoff * dt) * win
        idx = np.clip(base + taps, 0, n - 1)
        out[k] = (w * x[idx]).sum() / w.sum()
    return out


# MEASURED, not chosen. The table-vs-exact-kernel gap on this pipeline, as a fraction of the
# record's own peak: 3.3e-8 in the interior at the default half_width, 3.8e-8 at half_width
# 8 and 16, and 1e-9..5e-9 on records shorter than the kernel. The kernel's OWN stated accuracy
# is -130 dB = 3.2e-7, an order ABOVE all of those -- so a threshold here has to sit above the
# table's error and below the kernel's, and 1e-6 is the round number in that gap.
SPEC_TOL = 1e-6


def _band_limited(n, seed=0):
    """Content strictly inside the kernel's stated passband, where it is exact to ~-130 dB."""
    rng = np.random.default_rng(seed)
    k = np.arange(n)
    y = np.zeros(n)
    for f in rng.uniform(0.01, 0.30, 12):          # well under PASSBAND_FRAC = 0.386
        y += rng.uniform(0.3, 1.0) * np.sin(2 * np.pi * f * k + rng.uniform(0, 2 * np.pi))
    return y


def test_matches_the_documented_kernel_evaluated_directly():
    """The spec test: the fast path equals a direct evaluation of the kernel to the table's
    own documented interpolation bound."""
    x = _band_limited(4096)
    src = np.clip(np.arange(300, dtype=float) * 1.37 + 1000.4, 0, x.size - 1)
    got = RS.resample_at(x, src)
    want = _reference(x, src)
    assert np.abs(got - want).max() < SPEC_TOL * np.abs(x).max()


def test_dc_gain_is_exactly_one():
    """Normalised per output sample, so a constant record resamples to the same constant --
    at arbitrary fractional positions, including inside the clamped end regions."""
    x = np.full(2048, 0.75)
    src = np.array([0.0, 0.5, 1.25, 7.1, 1023.5, 2046.9, 2047.0])
    got = RS.resample_at(x, src)
    assert np.abs(got - 0.75).max() < 1e-12


def test_is_linear_in_the_record():
    """`src` does not depend on `x`, so the operator obeys superposition. Jitter, DCD and a
    sample clock are all this operator, and a nonlinearity here would be a physical error."""
    a, b = _band_limited(2048, seed=1), _band_limited(2048, seed=2)
    src = np.clip(np.arange(2048, dtype=float) * 1.0003 + 0.31, 0, 2047)
    ya, yb = RS.resample_at(a, src), RS.resample_at(b, src)
    yab = RS.resample_at(2.5 * a - 1.5 * b, src)
    assert np.abs(yab - (2.5 * ya - 1.5 * yb)).max() < 1e-10


def test_integer_positions_reproduce_the_record_in_the_interior():
    """Content well inside the passband, sampled at whole-sample positions, comes back
    unchanged -- the kernel is a 0.45*fs low-pass and this content stops at 0.30*fs.

    THE INTERIOR ONLY, deliberately. Within `half_width` of either end the kernel has no
    support and the indices are clamped, so those samples legitimately differ -- measured
    6.8e-3 of peak there against 1.8e-7 in the interior. `test_ends_are_clamped_not_wrapped`
    is what pins the end behaviour; conflating the two would hide both."""
    x = _band_limited(1024, seed=3)
    src = np.arange(1024, dtype=float)
    got = RS.resample_at(x, src)
    hw = RS.HALF_WIDTH
    assert np.abs(got[hw:-hw] - x[hw:-hw]).max() < SPEC_TOL * np.abs(x).max()


def test_ends_are_clamped_not_wrapped():
    """An output sample within half_width of either end has no kernel support there and the
    indices are CLAMPED -- a hold at x[0]/x[-1]. It must not wrap onto the other end, which
    would import the record's tail into its head."""
    x = np.concatenate([np.zeros(512), np.ones(512)])
    got = RS.resample_at(x, np.array([0.5, 1.5, 2.5]))
    assert np.abs(got).max() < 1e-6, "the record's tail of ones leaked into its head"


def test_agrees_with_the_unfactored_table_path_to_round_off():
    """The refactor itself: the polyphase factoring must reproduce the straightforward
    per-tap table lookup to float round-off, which is what makes it a factoring rather than a
    different filter. 1e-12 relative is far under the byte-identity gate's own 3.8e-13
    round-off band expressed as a fraction of peak-to-peak."""
    x = _band_limited(8192, seed=4)
    src = np.clip(np.arange(4000, dtype=float) * 1.618 + 0.27, 0, x.size - 1)

    taps = np.arange(-RS.HALF_WIDTH + 1, RS.HALF_WIDTH + 1)
    base = np.floor(src).astype(np.int64)
    dt = (src - base)[:, None] - taps[None, :]
    w = RS._weights(dt, RS.HALF_WIDTH, RS.CUTOFF, RS.BETA)
    idx = np.clip(base[:, None] + taps[None, :], 0, x.size - 1)
    want = (w * x[idx]).sum(axis=1) / w.sum(axis=1)

    got = RS.resample_at(x, src)
    assert np.abs(got - want).max() < 1e-12 * np.abs(x).max()


@pytest.mark.parametrize("n", [8, 33, 64, 65])
def test_a_record_shorter_than_the_kernel_still_resamples(n):
    """Below 2*half_width there is no interior at all -- every output sample is in the clamped
    region. The short-record path must still produce the clamped answer rather than failing on
    a window that cannot be formed."""
    x = np.linspace(-1.0, 1.0, n)
    src = np.clip(np.linspace(0.0, n - 1.0, 2 * n), 0, n - 1)
    got = RS.resample_at(x, src)
    assert got.shape == src.shape
    assert np.isfinite(got).all()
    assert np.abs(got - _reference(x, src)).max() < SPEC_TOL * np.abs(x).max()


def test_chunking_does_not_change_the_answer():
    """`chunk` bounds the working set; it must not be visible in the output."""
    x = _band_limited(20000, seed=5)
    src = np.clip(np.arange(20000, dtype=float) * 1.0001 + 0.17, 0, x.size - 1)
    a = RS.resample_at(x, src, chunk=1 << 10)
    b = RS.resample_at(x, src, chunk=1 << 16)
    assert np.abs(a - b).max() < 1e-12 * np.abs(x).max()


def test_non_default_half_width_is_honoured():
    """`half_width` is the documented knob for content near the passband edge, so the fast
    path has to build its table for whatever width it is handed, not just the default."""
    x = _band_limited(4096, seed=6)
    src = np.clip(np.arange(500, dtype=float) * 2.5 + 900.3, 0, x.size - 1)
    for hw in (8, 16):
        got = RS.resample_at(x, src, half_width=hw)
        want = _reference(x, src, half_width=hw)
        assert np.abs(got - want).max() < SPEC_TOL * np.abs(x).max(), f"half_width={hw}"
