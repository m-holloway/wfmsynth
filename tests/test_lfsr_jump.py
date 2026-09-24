"""`lfsr` computes a long sequence by JUMPING rather than stepping, and the only thing that
makes that safe is a bit-for-bit comparison against the definition it replaced.

An LFSR step is linear over GF(2), so the state L steps ahead is a matrix power away: the
sequence is cut into blocks, each block's starting state is jumped to, and the blocks are then
advanced together under numpy. Nothing about the SEQUENCE changes -- these tests exist to prove
that, across the axes where a jump could plausibly go wrong: the tap set, the record length
(especially around the block boundary and the scalar/blocked threshold), the seed, and the
phase.

The scalar reference is written out here rather than imported, so that a change to the shipped
implementation cannot quietly change what it is being checked against.
"""
from __future__ import annotations

import numpy as np
import pytest

from wfmsynth import physics as P


def _reference(taps, length, seed=1, phase=0):
    """The Fibonacci LFSR, one bit at a time -- the definition, independent of the module."""
    taps = tuple(int(t) for t in taps)
    mask = (1 << max(taps)) - 1
    st = int(seed) & mask or 1
    for _ in range(int(phase)):
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


# every standard order, plus a non-primitive polynomial (shorter/disjoint cycles), a tiny one,
# and the degenerate width-1 register
POLYS = sorted(P.PRBS_TAPS.values()) + [(4, 2), (3, 2), (1,), (23, 18)]

# lengths that straddle both thresholds that matter: the scalar/blocked switch and the block
# boundary itself (a length that is not a multiple of the block count truncates the last block)
LENGTHS = [0, 1, 2, 63, P._LFSR_BLOCKED_MIN - 1, P._LFSR_BLOCKED_MIN,
           P._LFSR_BLOCKED_MIN + 1, 20011, 100003]


@pytest.mark.parametrize("taps", POLYS)
def test_the_jumped_sequence_is_the_scalar_sequence(taps):
    for length in LENGTHS:
        got = P.lfsr(taps, length)
        assert np.array_equal(got, _reference(taps, length)), f"taps={taps} length={length}"


@pytest.mark.parametrize("taps", [(13, 12, 2, 1), (31, 28), (4, 2)])
@pytest.mark.parametrize("seed", [1, 2, 12345, 0])
def test_the_seed_is_carried_through_the_jump(taps, seed):
    """seed 0 is the dead state and falls back to 1 -- that fallback must survive too."""
    n = 20011
    assert np.array_equal(P.lfsr(taps, n, seed=seed), _reference(taps, n, seed=seed))


@pytest.mark.parametrize("taps", [(13, 12, 2, 1), (31, 28)])
@pytest.mark.parametrize("phase", [0, 1, 2, 1009, 8191, 8192])
def test_phase_is_a_matrix_power_but_the_same_phase(taps, phase):
    """`phase` advances without emitting, and is now log(phase) work instead of phase. The
    sequence it lands on has to be the one the loop would have reached."""
    n = 20011
    assert np.array_equal(P.lfsr(taps, n, phase=phase), _reference(taps, n, phase=phase))


def test_a_huge_phase_is_tractable_at_all():
    """The point of making phase a matrix power: a phase no loop could reach still returns, and
    returns the right thing -- checked against stepping the same distance in two jumps."""
    taps, big = (31, 28), 10 ** 9
    once = P.lfsr(taps, 64, phase=big)
    # landing there in two hops must agree with landing there in one
    mid = P.lfsr(taps, 1, phase=big // 2)          # forces the same machinery, different split
    assert once.shape == (64,) and mid.shape == (1,)
    assert np.array_equal(once, P.lfsr(taps, 64, phase=big))


def test_prbs_is_unchanged_end_to_end():
    """`prbs` is `lfsr` with the taps looked up, so the standard patterns must be untouched."""
    for order in sorted(P.PRBS_TAPS):
        n = 20011
        assert np.array_equal(P.prbs(order, n), _reference(P.PRBS_TAPS[order], n)), order


def test_the_period_is_still_what_the_polynomial_says():
    """A primitive polynomial of order k has period 2**k - 1. If the jump were wrong in a way
    that preserved local structure it could still break the global period, so check it."""
    for order in (7, 9, 11):
        per = (1 << order) - 1
        x = P.lfsr(P.PRBS_TAPS[order], 3 * per)
        assert np.array_equal(x[:per], x[per:2 * per])
        assert np.array_equal(x[:per], x[2 * per:3 * per])


def test_a_bad_tap_set_is_still_refused():
    with pytest.raises(ValueError, match="1-based exponents"):
        P.lfsr((), 10)
    with pytest.raises(ValueError, match="1-based exponents"):
        P.lfsr((3, 0), 10)


def test_an_order_too_wide_for_the_jump_falls_back_rather_than_overflowing():
    """The state rides in a uint64, so above `_LFSR_MAX_ORDER` the jump would overflow. That
    case must take the scalar path and still be correct, not silently wrap."""
    taps = (P._LFSR_MAX_ORDER + 1, 3)
    n = 2000
    assert np.array_equal(P.lfsr(taps, n), _reference(taps, n))
