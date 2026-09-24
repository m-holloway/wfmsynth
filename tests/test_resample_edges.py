"""Edge cases of the polyphase `resample_at` that its own test file does not reach.

`test_resample_polyphase.py` pins the claims the refactor makes -- DC gain, linearity, end
clamping, chunk invisibility, agreement with the unfactored table path. These are the cases
found by probing the factored form adversarially afterwards: the ones where the phase
arithmetic (`frac * TABLE_PER_SAMPLE`, split into an integer phase and a remainder) could
plausibly fall off an end, and where the resampler is reached by a real op with degenerate
arguments.

They matter because this function is the hot path for jitter, DCD, intra-pair skew, a
free-running sample clock and acquisition: it is reached by more ops than any other primitive
here, so an edge case it gets wrong is one that surfaces somewhere far from this file.
"""
from __future__ import annotations

import numpy as np
import pytest

from wfmsynth import resample as RS


def _unfactored(x, src, half_width=RS.HALF_WIDTH):
    """The general form, via the `_weights` path the module still exposes. Deliberately not the
    shipped hot path -- this is what the polyphase factoring has to agree with."""
    x = np.asarray(x, float)
    src = np.asarray(src, float)
    n = x.size
    taps = np.arange(-half_width + 1, half_width + 1)
    out = np.empty(src.size)
    for s0 in range(0, src.size, 4096):
        s = src[s0:s0 + 4096]
        base = np.floor(s).astype(np.int64)
        frac = s - base
        w = RS._weights(frac[:, None] - taps[None, :], half_width, RS.CUTOFF, RS.BETA)
        idx = np.clip(base[:, None] + taps[None, :], 0, n - 1)
        num = (w * x[idx]).sum(axis=1)
        den = w.sum(axis=1)
        out[s0:s0 + 4096] = num / np.where(den == 0.0, 1.0, den)
    return out


def _agree(x, src, half_width=RS.HALF_WIDTH, tol=1e-12):
    a = _unfactored(x, src, half_width)
    b = RS.resample_at(x, src, half_width=half_width)
    scale = max(float(np.abs(a).max()), 1e-30)
    assert np.abs(a - b).max() / scale < tol


@pytest.fixture
def x():
    return np.random.default_rng(0).standard_normal(4096)


def test_positions_far_before_the_record_clamp_to_the_first_sample(x):
    """Every tap is off the front, so the answer is a hold at x[0] -- and the phase index must
    not go negative getting there."""
    src = np.arange(200, dtype=float) - 5000.0
    _agree(x, src)
    assert np.allclose(RS.resample_at(x, src), x[0])


def test_positions_far_after_the_record_clamp_to_the_last_sample(x):
    src = np.arange(200, dtype=float) + 5000.0
    _agree(x, src)
    assert np.allclose(RS.resample_at(x, src), x[-1])


def test_a_position_large_enough_that_its_fraction_rounds_to_one(x):
    """`frac` is `s - floor(s)` and is meant to be in [0, 1). At a large magnitude the
    subtraction loses the low bits and it can round to exactly 1.0, which drives the phase
    index one past the table's last row -- the clip that guards that is what this pins."""
    src = np.full(50, 1e15) + np.linspace(0.0, 1.0, 50)
    _agree(x, src)
    assert np.all(np.isfinite(RS.resample_at(x, src)))


def test_negative_fractional_positions(x):
    """floor() of a negative number rounds AWAY from zero, so `base` and `frac` are not the
    truncation a reader might assume. The two paths have to make the same assumption."""
    _agree(x, -np.linspace(0.01, 3.99, 100))


@pytest.mark.parametrize("half_width", [1, 2, 64])
def test_degenerate_and_wide_kernels(x, half_width):
    """half_width=1 is two taps -- the smallest kernel the indexing supports, and the one where
    an off-by-one in the tap range would be invisible in the default case."""
    _agree(x, np.linspace(0.0, x.size - 1.0, 500), half_width=half_width)


def test_an_empty_request_returns_an_empty_record(x):
    assert RS.resample_at(x, np.array([], float)).shape == (0,)


def test_an_empty_record_is_still_refused():
    """There is nothing to interpolate FROM, and returning zeros would be a silent answer."""
    with pytest.raises(ValueError, match="nothing to resample"):
        RS.resample_at(np.array([], float), np.array([1.0]))


def test_a_constant_record_survives_every_position_exactly(x):
    """The normalisation claim, stated where it is easiest to check: DC gain is exactly 1, so a
    constant comes back constant whatever the fractional offsets are."""
    y = RS.resample_at(np.ones(4096), np.linspace(0.5, 4000.5, 777))
    assert np.abs(y - 1.0).max() < 1e-12


def test_an_all_zero_record_does_not_divide_by_zero(x):
    y = RS.resample_at(np.zeros(4096), np.linspace(0.5, 4000.5, 333))
    assert np.array_equal(y, np.zeros_like(y))
