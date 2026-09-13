"""Level coding: the bit-to-level layer that sits between a bit stream and a PAM carrier.

Three things are under test, and none of them is "the eye looks right":

  * THE LEVEL MAP IS EXACT. Four PAM4 levels must be equally spaced to the last bit the
    format has. `np.linspace(-1, 1, 4)` is not equally spaced -- its inner levels sit
    5.6e-17 off the thirds -- so `physics.PAM_LEVELS` states the array explicitly and this
    file pins that, in both directions: the explicit array's spacings agree to 1 ulp, and
    linspace's disagreement is measured rather than assumed.
  * GRAY CODING IS A STATEMENT ABOUT ERRORS, not a lookup table. The property that earns it
    a place in every PAM4 standard is that the dominant error -- a slice into the ADJACENT
    level -- costs exactly ONE bit. So the tests count bit errors through a level error,
    against a natural-binary map as the control.
  * PRECODING CHANGES THE ERROR STRUCTURE AND NOTHING ELSE. Level histogram and transition
    density are preserved to <1e-3; the symbol sequence is different in ~75% of positions;
    and a single received-symbol error decodes to exactly 2 bit errors instead of
    propagating without bound. The unbounded case is constructed here as the control,
    because "bounded" is only a claim if the alternative is shown to be unbounded.

Run:  PYTHONPATH=. python -m pytest tests/test_level_coding.py -q
"""
from __future__ import annotations

import math

import numpy as np
import pytest

import wfmsynth as ws
from wfmsynth import coding as C
from wfmsynth import physics as P

ULP = float(np.spacing(2.0 / 3.0))          # 1.11e-16: the quantum the thirds are held to


# ============================================================ the level map is exact
def test_the_four_pam4_levels_are_equally_spaced_to_one_ulp():
    """Equal spacing is what makes the three eyes the same height and what a level-separation
    (RLM) measurement is measured AGAINST. 2/3 is not representable, so the tolerance is the
    format's own resolution -- one ulp of 2/3 -- not an engineering fudge."""
    lv = P.pam_levels(4)
    d = np.diff(lv)
    assert lv[0] == -1.0 and lv[-1] == 1.0                  # the span is exact
    assert np.ptp(d) <= ULP, f"spacings differ by {np.ptp(d):.3e}, more than 1 ulp"
    assert max(abs(float(x) - 2.0 / 3.0) for x in d) <= ULP
    # RLM is 1 by construction on this map: the mid-level separation equals the outer one.
    assert abs(float(d[1] / d[0]) - 1.0) <= 2 * ULP


def test_the_pam4_level_map_is_not_linspace_and_the_difference_is_measured():
    """The regression this pins: `linspace(-1, 1, 4)` differs from the explicit thirds by
    5.55e-17 at the inner levels. Small, and not zero -- every PAM4 waveform this kernel has
    emitted came from the explicit array, so swapping in linspace would move every one of
    them. Fails the day someone "simplifies" PAM_LEVELS."""
    lv = P.pam_levels(4)
    ls = np.linspace(-1.0, 1.0, 4)
    delta = float(np.max(np.abs(ls - lv)))
    assert 5.0e-17 < delta < 6.0e-17, f"linspace now differs by {delta:.3e}, not ~5.6e-17"
    assert not np.array_equal(lv, ls)
    assert np.array_equal(lv, np.array([-1.0, -1.0 / 3.0, 1.0 / 3.0, 1.0]))


def test_the_millivolt_level_map_is_the_normalised_one_scaled_and_the_spacing_is_800_over_3():
    """The volts a standard states and the normalised levels the kernel carries must be the
    SAME map: -400/-133.3/+133.3/+400 mV is 400 mV times [-1, -1/3, +1/3, +1], spacing
    800/3 = 266.67 mV. Two independently written level arrays is how a 0.3 mV disagreement
    gets into a dataset's labels."""
    assert np.array_equal(C.PAM4_LEVELS_MV, C.PAM4_OUTER_MV * P.pam_levels(4))
    d = np.diff(C.PAM4_LEVELS_MV)
    assert np.ptp(d) <= np.spacing(800.0 / 3.0)
    assert abs(float(d[0]) - 800.0 / 3.0) <= np.spacing(800.0 / 3.0)
    assert C.PAM4_LEVELS_MV[0] == -400.0 and C.PAM4_LEVELS_MV[-1] == 400.0


def test_the_three_pam3_levels_are_exact_integers():
    """PAM3's map has no rounding to argue about -- [-1, 0, +1] is exact in binary floating
    point, so the assertion is equality, not a tolerance."""
    lv = P.pam_levels(3)
    assert np.array_equal(lv, np.array([-1.0, 0.0, 1.0]))
    assert np.array_equal(np.diff(lv), np.array([1.0, 1.0]))


# ============================================================ PAM4 Gray coding
def test_gray_coding_makes_adjacent_levels_differ_in_exactly_one_bit():
    """The defining property, asserted over the whole map: walking up the levels flips one
    bit per step. The two SKIP-ONE pairs (0-2, 1-3) are at distance 2, which is the other
    half of the statement -- a Gray map that also collapsed those would be carrying less
    than 2 bits per symbol."""
    bits = C.GRAY_PAM4_BITS
    assert bits.shape == (4, 2)
    def hamming(i, j):
        return int(np.sum(bits[i] != bits[j]))
    assert [hamming(i, i + 1) for i in range(3)] == [1, 1, 1]
    assert hamming(0, 2) == 2 and hamming(1, 3) == 2
    # The map is IEEE 802.3 120.5.11.2.1's: 00b, 01b, 11b, 10b from the bottom level up.
    assert [tuple(r) for r in bits] == [(0, 0), (0, 1), (1, 1), (1, 0)]
    # ... and it is a CYCLE, so the outer pair is also at distance 1. Stated because the
    # tempting assertion "hamming == min(|di|, 2)" would pass on a non-Gray map too.
    assert hamming(0, 3) == 1


def test_gray_encode_and_decode_round_trip_over_every_bit_pair():
    """Exhaustive over the alphabet, then over a long PRBS: the map is a bijection, so no
    bit pattern is unrepresentable and none aliases onto another."""
    allpairs = np.array([0, 0, 0, 1, 1, 1, 1, 0])
    assert np.array_equal(C.pam4_gray_encode(allpairs), np.array([0, 1, 2, 3]))
    assert np.array_equal(C.pam4_gray_decode(np.array([0, 1, 2, 3])), allpairs)
    bits = P.prbs(13, 2 * 4096, seed=7).astype(int)
    sym = C.pam4_gray_encode(bits)
    assert sym.shape == (4096,) and set(np.unique(sym)) <= {0, 1, 2, 3}
    assert np.array_equal(C.pam4_gray_decode(sym), bits)


def test_gray_levels_come_from_the_kernels_own_level_array():
    """`pam4_gray_levels` must not re-derive the levels: a second copy of [-1,-1/3,1/3,1] in
    the codebase is a second place for the 5.6e-17 to differ."""
    bits = P.prbs(9, 2 * 512, seed=3).astype(int)
    lv = C.pam4_gray_levels(bits)
    assert np.array_equal(np.unique(lv), P.pam_levels(4))
    assert np.array_equal(lv, P.pam_levels(4)[C.pam4_gray_encode(bits)])


def test_an_adjacent_level_error_costs_exactly_one_bit_under_gray_and_more_under_binary():
    """The statistic Gray coding exists for, measured against a control rather than asserted:
    the natural binary map 00/01/10/11 is the SAME four levels with the same eye, and it is
    strictly worse out of the slicer.

    Two error models, because the penalty depends on which one is meant and 'Gray saves 33%'
    is quoted without saying:

      per THRESHOLD  each of the three eyes is equally likely to be the one crossed.
                     Gray 1.0 bits/error, binary 4/3 -- the 1.33x usually quoted.
      per SYMBOL     every symbol errs by one level, outer levels necessarily inward.
                     Gray 1.0, binary 1.25 -- the outer levels are cheap under binary too,
                     so the advantage is smaller than the 1.33x headline.

    Gray is 1.0 under BOTH, which is the point: it is one bit per level error whatever the
    error model, because the property is a property of the map."""
    gray, binary = C.GRAY_PAM4_BITS, np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

    def cost(m, i, j):
        return int(np.sum(m[i] != m[j]))

    # per-threshold: the three adjacent pairs, equally weighted. Closed form, no sampling.
    per_thresh = [(0, 1), (1, 2), (2, 3)]
    assert [cost(gray, i, j) for i, j in per_thresh] == [1, 1, 1]
    assert [cost(binary, i, j) for i, j in per_thresh] == [1, 2, 1]
    assert abs(np.mean([cost(binary, i, j) for i, j in per_thresh]) - 4 / 3) < 1e-12

    # per-symbol: sampled, as a slicer fed uniform data would actually err.
    rng = np.random.default_rng(11)
    sym = rng.integers(0, 4, 20_000)
    step = np.where(sym == 0, 1, np.where(sym == 3, -1, rng.choice([-1, 1], sym.size)))
    bad = sym + step
    gray_err = np.sum(gray[sym] != gray[bad], axis=1)
    bin_err = np.sum(binary[sym] != binary[bad], axis=1)
    assert np.all(gray_err == 1), "a one-level error cost more than one bit under Gray"
    assert abs(bin_err.mean() - 1.25) < 0.01, f"binary cost {bin_err.mean():.3f}, expected 1.25"


# ============================================================ precoding
@pytest.mark.parametrize("lag", [1, 2])
@pytest.mark.parametrize("state", [0, 1])
def test_precoding_is_invertible(lag, state):
    """1/(1+D) over GF(2) is invertible by (1+D) exactly, for any initial state -- there is
    no tolerance here, it is XOR arithmetic. The `state` sweep matters because a receiver
    that starts with the wrong assumption about the precoder's history is the realistic
    case, and it must cost a bounded number of bits, not the record."""
    bits = P.prbs(13, 4096, seed=5).astype(int)
    y = C.precode(bits, lag=lag, state=state)
    assert np.array_equal(C.precode_inverse(y, lag=lag, state=state), bits)
    # wrong assumed state: only the first `lag` bits are wrong, and nothing after them.
    wrong = C.precode_inverse(y, lag=lag, state=1 - state)
    assert np.sum(wrong != bits) == lag


def test_precoding_actually_changes_the_stream():
    """A knob that does nothing is the defect this repo has been bitten by. Precoding must
    move ~half the bits of a pseudo-random stream (it is x[n] XOR y[n-1], and y is balanced),
    and the ops must not be each other's no-op."""
    bits = P.prbs(13, 8191, seed=2).astype(int)
    y = C.precode(bits)
    assert 0.4 < np.mean(y != bits) < 0.6
    assert not np.array_equal(C.precode(bits, lag=1), C.precode(bits, lag=2))


def test_precoding_preserves_the_eye_statistics_and_changes_the_sequence():
    """WHY a record without precoding is wrong for learning from transitions: the first-order
    statistics are IDENTICAL -- same four level probabilities, same 0.75 transition density,
    so the eye diagram is the same picture -- while ~75% of the symbols differ. A model fed
    un-precoded symbols sees a plausible eye and the wrong sequence."""
    bits = P.prbs(13, 2 * 8191, seed=1).astype(int)
    raw = C.pam4_gray_encode(bits)
    pre = C.pam4_gray_encode(C.precode(bits))
    p_raw = np.bincount(raw, minlength=4) / raw.size
    p_pre = np.bincount(pre, minlength=4) / pre.size
    assert np.max(np.abs(p_pre - p_raw)) < 1e-3, "precoding moved the level histogram"
    assert np.max(np.abs(p_pre - 0.25)) < 1e-3
    t_raw = float(np.mean(np.diff(raw) != 0))
    t_pre = float(np.mean(np.diff(pre) != 0))
    assert abs(t_pre - t_raw) < 1e-3 and abs(t_raw - 0.75) < 1e-3
    assert np.mean(raw != pre) > 0.5, "the precoded symbol sequence is barely different"


def test_a_single_symbol_error_decodes_to_a_bounded_burst_of_exactly_two_bits():
    """The claim precoding is made for. One adjacent-level error anywhere in the received
    symbol stream is exactly ONE wrong bit (Gray), and the receiver's inverse is FEEDFORWARD
    -- y[n] XOR y[n-1] over RECEIVED symbols -- so it multiplies that into exactly 2 wrong
    bits and stops. Swept over every error position in a short record, so the bound is a
    bound and not an average."""
    bits = P.prbs(9, 2 * 256, seed=4).astype(int)
    y = C.precode(bits)
    sym = C.pam4_gray_encode(y)
    for k in range(1, len(sym) - 1):
        bad = sym.copy()
        bad[k] = sym[k] + (1 if sym[k] < 3 else -1)             # slice one level off
        got = C.precode_inverse(C.pam4_gray_decode(bad))
        n_err = int(np.sum(got != bits))
        assert n_err == 2, f"symbol error at {k} decoded to {n_err} bit errors, not 2"


def test_the_unprecoded_alternative_is_unbounded_which_is_why_precoding_is_mandatory():
    """The control, because "bounded" says nothing without it. The recursion is the SAME
    arithmetic in both cases -- y[n] = x[n] XOR y[n-1] -- and the only difference is WHERE it
    runs:

      precoded      TX integrates (its input is the data, which is never wrong), RX
                    differentiates from RECEIVED symbols. Feedforward: 2 bit errors, always.
      not precoded  the (1+D) post-cursor is left for the receiver to undo, so the RECEIVER
                    integrates -- and it is driven by symbols that can be wrong. One error
                    flips every bit after it.

    Measured below: 2 errors against 412, running to the end of the record."""
    bits = P.prbs(9, 512, seed=4).astype(int)
    k = 100

    # precoded: the line carries the integrated stream, the receiver differentiates it.
    y = C.precode(bits)
    bad_y = y.copy(); bad_y[k] ^= 1
    assert int(np.sum(C.precode_inverse(bad_y) != bits)) == 2

    # not precoded: the line carries the data through a (1+D) post-cursor (the mod-2
    # caricature of the response a DFE is there to cancel), so the receiver must integrate.
    r = C.precode_inverse(bits)
    bad_r = r.copy(); bad_r[k] ^= 1
    d = C.precode(bad_r)                      # the identical recursion, at the wrong end
    assert np.array_equal(C.precode(r), bits), "the control's error-free case must decode clean"
    burst = int(np.sum(d != bits))
    assert burst == len(bits) - k == 412, f"the recursive burst was {burst} bits, not 412"
    assert np.all(d[k:] != bits[k:]), "the recursive error did not propagate to the end"


# ============================================================ PAM3
def test_the_pam3_block_size_is_the_smallest_one_that_holds_eleven_bits():
    """The arithmetic that FIXES the block code rather than guessing it. 1.57 bits/symbol at
    25.6 GBd is 40.2 Gb/s; log2(3) = 1.58496 is the ceiling, so the code is a block code, and
    7 is the shortest ternary block that can carry 11 bits (3^6 = 729 < 2048 <= 3^7 = 2187).
    11/7 = 1.5714 reaches 99.15% of the ternary capacity."""
    b, s = C.PAM3_BLOCK_BITS, C.PAM3_BLOCK_SYMBOLS
    assert (b, s) == (11, 7)
    assert 3 ** s >= 2 ** b > 3 ** (s - 1)                       # 7 is minimal for 11 bits
    assert abs(C.PAM3_BITS_PER_SYMBOL - b / s) < 1e-15
    assert abs(C.PAM3_BITS_PER_SYMBOL - 1.57) < 0.005
    assert C.PAM3_BITS_PER_SYMBOL < math.log2(3)
    assert abs(25.6e9 * C.PAM3_BITS_PER_SYMBOL - 40.23e9) < 0.02e9
    assert 3 ** s - 2 ** b == 139                                # spare codewords


def test_pam3_round_trips_over_every_codeword():
    """Exhaustive over all 2048 11-bit words: the mapping is injective and the inverse is
    exact. A block code that aliases two words is silently lossy, and nothing downstream of
    the waveform would show it."""
    words = np.arange(1 << C.PAM3_BLOCK_BITS)
    bits = ((words[:, None] >> np.arange(C.PAM3_BLOCK_BITS - 1, -1, -1)) & 1).ravel()
    sym = C.pam3_encode(bits)
    assert sym.size == words.size * C.PAM3_BLOCK_SYMBOLS
    assert np.array_equal(np.unique(sym), P.pam_levels(3))       # ternary, exact levels
    assert np.array_equal(C.pam3_decode(sym), bits)
    blocks = sym.reshape(-1, C.PAM3_BLOCK_SYMBOLS)
    assert len({tuple(r) for r in blocks}) == words.size, "two words share a codeword"


def test_pam3_rejects_a_partial_block_rather_than_padding_it():
    """Padding a short block would put invented bits into a record. The alphabet is also
    checked on decode: a level that is not one of the three is a slicer output, not a
    codeword, and guessing at it is how a decode silently succeeds on garbage."""
    with pytest.raises(ValueError, match="whole 11-bit blocks"):
        C.pam3_encode(np.zeros(12, dtype=int))
    with pytest.raises(ValueError, match="whole 7-symbol blocks"):
        C.pam3_decode(np.zeros(8))
    with pytest.raises(ValueError, match="ternary"):
        C.pam3_decode(np.full(7, 0.5))


def test_pam3_symbol_statistics_are_near_balanced_and_carry_the_stated_rate():
    """Statistics, not just the mapping. Over the whole codeword space the three levels come
    out within 3% of uniform and the mean symbol within 0.03 of zero (so the code is roughly
    DC-neutral without claiming to be DC-balanced), and the measured per-symbol entropy is
    1.584 bits -- the ternary capacity, i.e. the mapping wastes essentially nothing. The
    residual skew is the 139 spare codewords showing up in the LEADING digit, which is a
    property of this base-3 mapping and would move under the standard's own table."""
    words = np.arange(1 << C.PAM3_BLOCK_BITS)
    bits = ((words[:, None] >> np.arange(C.PAM3_BLOCK_BITS - 1, -1, -1)) & 1).ravel()
    sym = C.pam3_encode(bits)
    frac = np.bincount((sym + 1).astype(int), minlength=3) / sym.size
    assert np.max(np.abs(frac - 1 / 3)) < 0.03, f"level fractions {frac.round(4)}"
    assert abs(float(sym.mean())) < 0.03
    ent = float(-(frac * np.log2(frac)).sum())
    assert abs(ent - 1.584) < 0.002 and ent <= math.log2(3)
    # the skew lives in the leading digit, and is gone by the trailing one
    blocks = sym.reshape(-1, C.PAM3_BLOCK_SYMBOLS)
    lead = np.bincount((blocks[:, 0] + 1).astype(int), minlength=3) / len(blocks)
    tail = np.bincount((blocks[:, -1] + 1).astype(int), minlength=3) / len(blocks)
    assert np.max(np.abs(lead - 1 / 3)) > 0.02
    assert np.max(np.abs(tail - 1 / 3)) < 0.002


# ============================================================ they compose
def _slice_to_levels(w, n_sym, levels):
    """Mid-UI sample of each symbol, snapped to the nearest of `levels`. The whole receiver
    this test needs: no CDR, no equaliser, because the claim under test is that the coder's
    symbols survive rendering, not that a slicer works."""
    spb = len(w) / n_sym
    mid = w[(np.arange(n_sym) * spb + spb / 2).astype(int)]
    return np.argmin(np.abs(mid[:, None] - np.asarray(levels)[None, :]), axis=1)


def test_the_coded_symbols_drive_a_waveform_and_the_bits_come_back():
    """The point of a level coder in this library: its output is a symbol stream, so it feeds
    the existing `symbols` source and every impairment and instrument stage downstream. Both
    codes are rendered on a real grid and decoded back to the bits that went in -- which is
    also an end-to-end check that the coder emits the kernel's own level values, since the
    slicer's reference set is `physics.pam_levels`."""
    pam4_bits = P.prbs(9, 2 * 256, seed=6).astype(int)
    pam3_bits = P.prbs(9, 11 * 32, seed=6).astype(int)
    cases = [
        ("pam4+gray+precode", C.pam4_gray_levels(C.precode(pam4_bits)), P.pam_levels(4),
         lambda i: C.precode_inverse(C.pam4_gray_decode(i)), pam4_bits),
        ("pam3", C.pam3_encode(pam3_bits), P.pam_levels(3),
         lambda i: C.pam3_decode(P.pam_levels(3)[i]), pam3_bits),
    ]
    for name, lv, levels, decode, bits in cases:
        lv = np.asarray(lv, float)
        n = len(lv) * 16
        g = ws.Grid(fs=200e9, baud=12.5e9, n=n)
        w = ws.Signal(seed=1, grid=g).symbols(symbols=lv.tolist(), causal=True).waveform()
        idx = _slice_to_levels(w, len(lv), levels)
        assert np.array_equal(levels[idx], lv), f"{name}: the rendered levels did not slice back"
        assert np.array_equal(decode(idx), bits), f"{name}: the bits did not survive the round trip"
