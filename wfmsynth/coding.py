"""
wfmsynth.coding — line coding & scrambling (DC balance, run-length control).

Raw PRBS is a stand-in for real transmitted bits: its running disparity random-walks (so the
DC content wanders) and it can have long runs. Real links run a line code or scrambler that
bounds both, which changes how the signal interacts with AC-coupling (#baseline wander) and
what an analyser locks to. Two primitives:

  * ``dc_balanced`` — the principle behind 8b/10b: each block is sent as-is or inverted
    (whichever pulls the running disparity toward zero), with a flag bit marking inversion.
    The accumulated disparity stays bounded → DC-balanced (a random-walking PRBS is not). It
    bounds disparity, not run length as tightly as a full 8b/10b table (logged as #45).
  * ``scramble_64b66b`` — the 64b/66b self-synchronous scrambler (x^58 + x^39 + 1) that real
    high-rate links use to break up patterns (2 sync bits per 64-bit block).
"""
from __future__ import annotations

import numpy as np


def dc_balanced(bits, block=8):
    """DC-balanced line code (8b/10b-style running-disparity block inversion). ``bits`` is a
    0/1 array; each ``block`` bits are emitted with a leading flag bit, inverted when that pulls
    the running disparity toward zero. Returns the coded 0/1 stream — bounded disparity, hence
    DC-balanced with bounded run length."""
    bits = np.asarray(bits).astype(int)
    out = []
    disp = 0
    for i in range(0, len(bits) - block + 1, block):
        blk = bits[i:i + block]
        a = np.concatenate([[0], blk])                    # as-is (flag 0)
        b = np.concatenate([[1], 1 - blk])                # inverted (flag 1)
        da = disp + int((2 * a - 1).sum())
        db = disp + int((2 * b - 1).sum())
        if abs(db) < abs(da):
            out.extend(b.tolist()); disp = db
        else:
            out.extend(a.tolist()); disp = da
    return np.array(out, dtype=int)


def scramble_64b66b(bits):
    """64b/66b self-synchronous scrambler (polynomial x^58 + x^39 + 1). Prepends a 2-bit sync
    header to each 64-bit block and scrambles the payload; whitens the spectrum without the
    lookup tables of 8b/10b. ``bits`` is a 0/1 array (truncated to whole 64-bit blocks)."""
    bits = np.asarray(bits).astype(int)
    state = np.ones(58, dtype=int)
    out = []
    nblk = len(bits) // 64
    for j in range(nblk):
        out.extend([0, 1])                                # sync header
        for b in bits[j * 64:(j + 1) * 64]:
            fb = state[57] ^ state[38]                    # x^58 + x^39
            s = b ^ fb
            out.append(int(s))
            state = np.roll(state, 1); state[0] = s
    return np.array(out, dtype=int)


def running_disparity(bits):
    """Cumulative running disparity of a 0/1 stream (sum of ±1). A bounded envelope means the
    code is DC-balanced; a random walk (raw PRBS) means it is not."""
    return np.cumsum(2 * np.asarray(bits).astype(int) - 1)


def max_run(bits):
    """Longest run of identical bits."""
    b = np.asarray(bits).astype(int)
    if len(b) == 0:
        return 0
    return int(np.max(np.diff(np.flatnonzero(np.concatenate(([1], np.diff(b) != 0, [1]))))))


# ==================================================================== LEVEL CODING
# The layer BETWEEN a bit stream and a PAM carrier, and the reason it is a separate layer:
# everything above (`dc_balanced`, `scramble_64b66b`) shapes the bit stream's spectrum, and
# everything below (`physics.from_symbols`, `Signal.symbols`) turns symbols into volts. What
# happens in between -- which bit pair becomes which level, and whether the bits were
# precoded first -- changes the SYMBOL STATISTICS and the ERROR STATISTICS while leaving the
# eye diagram alone. A record that gets this layer wrong has a plausible eye and the wrong
# sequence, which is the worst of the available failure modes: nothing looks broken.
#
# THE LEVEL VALUES ARE NOT RESTATED HERE. `physics.PAM_LEVELS` is the one definition, because
# the explicit thirds and `np.linspace(-1, 1, 4)` differ by 5.6e-17 and a second copy of the
# array in this module is a second chance to pick the wrong one. Everything below indexes
# `physics.pam_levels()`.
from . import physics as _P

# The Gray map, derived from `physics.GRAY_PAM4` (IEEE 802.3 Clause 120.5.11.2.1) by sorting
# its entries into level order: 00b, 01b, 11b, 10b from the bottom level to the top. Derived
# rather than re-typed so the two cannot drift apart.
#
# WHY GRAY AND NOT NATURAL BINARY. A PAM4 slicer's dominant error is a decision into the
# ADJACENT level -- the three eyes are 1/3 the height of the binary one and the noise has to
# cross only one of them. Under this map adjacent levels differ in exactly one bit, so that
# error costs one bit; under 00/01/10/11 the 1->2 step costs two. Measured penalty for the
# IDENTICAL waveform (tests/test_level_coding.py): 4/3 when each of the three eyes is equally
# likely to be the one crossed, 1.25 when every symbol errs by one level (the outer levels can
# only err inward, which is cheap under either map). Gray is 1.0 under both.
GRAY_PAM4_BITS = np.array([k for k, _ in sorted(_P.GRAY_PAM4.items(), key=lambda kv: kv[1])],
                          dtype=np.int8)
# bit pair packed as 2*MSB+LSB -> level index. The inverse of GRAY_PAM4_BITS, built once.
_GRAY_CODE_TO_LEVEL = np.empty(4, dtype=np.int8)
_GRAY_CODE_TO_LEVEL[2 * GRAY_PAM4_BITS[:, 0] + GRAY_PAM4_BITS[:, 1]] = np.arange(4, dtype=np.int8)

# The same four levels in millivolts, for a transmitter whose nominal differential swing is
# 800 mV peak-to-peak: -400 / -133.33 / +133.33 / +400 mV, equally spaced at 800/3 = 266.67 mV.
# The SPACING is arithmetic (the level map times the amplitude), not a separate measurement --
# which is why this is `PAM4_OUTER_MV * pam_levels(4)` and not a typed-out array. JUDGEMENT:
# 800 mV pk-pk is a nominal swing, chosen here as the scale the mV figures are quoted at; a
# transmitter specification stating a different nominal amplitude falsifies the numbers and
# nothing else -- the ratios, the spacing-to-amplitude relation and every test above survive.
PAM4_OUTER_MV = 400.0
PAM4_LEVELS_MV = PAM4_OUTER_MV * _P.pam_levels(4)


def _bits01(bits, what="bits"):
    """A 0/1 integer array, or a ValueError naming what was wrong. A float array of levels
    passed where bits belong would otherwise encode silently."""
    b = np.asarray(bits)
    if b.ndim != 1:
        raise ValueError(f"{what} must be a 1-D array, not shape {b.shape}")
    b = b.astype(np.int64)
    if b.size and (b.min() < 0 or b.max() > 1):
        raise ValueError(f"{what} must be 0/1; got values in [{b.min()}, {b.max()}]")
    return b


def pam4_gray_encode(bits):
    """Gray-map consecutive bit pairs to PAM4 level INDICES 0..3 (IEEE 802.3 120.5.11.2.1).

    The first bit of each pair is the MSB. Returns one index per two bits; the input must be
    an even number of bits (a half symbol is not a symbol, and padding it would put a bit into
    the record that the caller never sent). Use `pam4_gray_levels` for the level values.
    """
    b = _bits01(bits)
    if b.size % 2:
        raise ValueError(f"PAM4 takes whole bit PAIRS; got {b.size} bits")
    p = b.reshape(-1, 2)
    return _GRAY_CODE_TO_LEVEL[2 * p[:, 0] + p[:, 1]].astype(np.int64)


def pam4_gray_decode(symbols):
    """PAM4 level indices 0..3 back to bits — the exact inverse of `pam4_gray_encode`.

    Takes INDICES, not volts: slicing a waveform to indices is the receiver's job and depends
    on thresholds this module knows nothing about.
    """
    s = np.asarray(symbols).astype(np.int64)
    if s.size and (s.min() < 0 or s.max() > 3):
        raise ValueError(f"PAM4 level indices must be 0..3; got [{s.min()}, {s.max()}]")
    return GRAY_PAM4_BITS[s].reshape(-1).astype(np.int64)


def pam4_gray_levels(bits):
    """Gray-map bit pairs straight to PAM4 LEVELS in {-1, -1/3, +1/3, +1}.

    The array is `physics.pam_levels(4)` indexed, so the levels are bit-identical to every
    other PAM4 waveform this kernel emits. Feed it to `physics.from_symbols` or
    `Signal.symbols` to render. Multiply by `PAM4_OUTER_MV` for millivolts.
    """
    return _P.pam_levels(4)[pam4_gray_encode(bits)]


# ---------------------------------------------------------------- precoding
# WHAT IT IS. y[n] = x[n] XOR y[n-1]: the GF(2) 1/(1+D) recursion, run over the bit stream at
# the TRANSMITTER, inverted at the receiver by the feedforward x[n] = y[n] XOR y[n-1]. PCI
# Express Base Specification Rev 6.0 makes it MANDATORY at 64 GT/s (PAM4). It is applied
# before the Gray map, so it changes which levels are sent, not where the levels are.
#
# WHY A LINK IS REQUIRED TO DO IT -- and this is the whole point of the op, because precoding
# improves no eye measurement whatsoever. A receiver that equalises a post-cursor with
# decision feedback decides from its OWN PAST DECISIONS: d[n] = r[n] XOR d[n-1] in the mod-2
# picture. One wrong decision poisons the next, and the next, without bound -- measured in
# tests/test_level_coding.py as a burst running to the end of the record. Precoding moves that
# recursion to the transmitter, where the data is known exactly and cannot be wrong, and
# leaves the receiver a FEEDFORWARD XOR of two RECEIVED symbols. A single received-symbol
# error then multiplies into exactly 2 bit errors and stops. Same eye, same level histogram,
# same 0.75 transition density; bounded error bursts instead of unbounded ones.
#
# SO: a synthetic record generated without precoding has the right first-order statistics and
# the wrong error structure. Anything learning from symbol transitions, or scored on burst
# length, is learning from a link that does not exist.
#
# PRELIMINARY -- the DELAY'S UNIT. The recursion is stated on the bit stream, so `lag=1` (one
# BIT of delay) is the default and is what the specification's equation reads as. Whether the
# 64 GT/s precoder's D is one bit or one SYMBOL (i.e. the recursion run independently over the
# MSB and LSB lanes, `lag=2`) is not settled from a citation here, and the distinction is
# real: at lag=1 a symbol error's two decoded bit errors can land in two adjacent symbols,
# at lag=2 they land in the same bit lane one symbol apart. Both are bounded at 2 bits, which
# is why the property this library tests does not depend on the answer. What would settle it:
# the precoding clause of the Base Specification. Cited by designation only, no section
# number, because the number is not verified here.
def precode(bits, lag=1, state=0):
    """GF(2) 1/(1+D) precoder: ``y[n] = x[n] XOR y[n-lag]``, with ``y[n<0] = state``.

    `lag=1` is the bit-serial recursion (the default, and what the 64 GT/s requirement reads
    as); `lag=2` runs it independently over the two bit lanes of a PAM4 symbol, i.e. one
    SYMBOL of delay. Apply BEFORE the Gray map.
    """
    b = _bits01(bits)
    lag, st = int(lag), int(state) & 1
    if lag < 1:
        raise ValueError(f"lag must be at least 1 bit, not {lag}")
    out = np.empty_like(b)
    # Per residue class the recursion is a running XOR, so the closed form is a prefix XOR of
    # that class -- O(n) and exact, rather than a Python loop over the record.
    for j in range(lag):
        out[j::lag] = np.bitwise_xor.accumulate(b[j::lag]) ^ st
    return out


def precode_inverse(bits, lag=1, state=0):
    """The receiver's inverse of `precode`: ``x[n] = y[n] XOR y[n-lag]``.

    FEEDFORWARD — it reads received bits, never its own output, which is exactly why an error
    cannot propagate: one wrong received bit produces wrong output at n and n+lag and nowhere
    else. A wrong assumed `state` costs the first `lag` bits and nothing after them.
    """
    y = _bits01(bits)
    lag, st = int(lag), int(state) & 1
    if lag < 1:
        raise ValueError(f"lag must be at least 1 bit, not {lag}")
    prev = np.empty_like(y)
    prev[:lag] = st
    prev[lag:] = y[:-lag]
    return y ^ prev


# ---------------------------------------------------------------- PAM3, 11b/7t
# THE BLOCK SIZE IS THE THING TO ESTABLISH, and it is fixed by arithmetic plus one number
# from the standard. USB4 Version 2.0's Gen 4 lane runs 25.6 GBd PAM3 and carries 40 Gb/s,
# i.e. ~1.57 bits per symbol. A ternary symbol holds at most log2(3) = 1.58496 bits, so 1.57
# is not a whole-symbol mapping and the code must be a BLOCK code; 11 bits to 7 symbols is
# the block that lands there:
#
#     11/7 = 1.571429 bits/symbol           99.15 % of the ternary capacity
#     3^6 = 729 < 2^11 = 2048 <= 3^7 = 2187  so 7 is the SHORTEST ternary block holding 11 bits
#     25.6 GBd * 11/7 = 40.23 Gb/s          against the 40 Gb/s per-lane figure: 0.6 % of margin
#
# The 0.6 % is the framing overhead's room, and the fact that 7 is minimal is what makes the
# choice non-arbitrary: a 12b/8t or 22b/14t code would carry the same rate at the same
# efficiency but needs a longer block for no gain.
#
# PRELIMINARY -- THE CODEWORD TABLE. What is established above is the block SIZE and the rate.
# The mapping implemented here is the canonical one: the 11-bit word read as an integer and
# written as 7 base-3 digits, most significant digit FIRST, digits {0,1,2} to levels
# {-1,0,+1}. It is injective, exactly invertible, and exercises all three levels at the right
# rate, which is what an eye, a slicer and a level-separation measurement need. It is NOT
# asserted to be the standard's table: 2^11 of 3^7 codewords are used, leaving 139 spare that
# a real standard would spend on control/sync symbols and would likely also choose to balance
# disparity, and the digit ORDER is a free choice here. Consequence, measured: the leading
# symbol of a block is skewed (0.356/0.356/0.288 across the three levels) because the top of
# the base-3 range is unused, while the trailing symbol is uniform to 0.002. What would settle
# it: the PAM3 encoding table in the USB4 Version 2.0 specification. Cited by designation
# only -- no section number, because none is verified here.
PAM3_BLOCK_BITS = 11
PAM3_BLOCK_SYMBOLS = 7
PAM3_BITS_PER_SYMBOL = PAM3_BLOCK_BITS / PAM3_BLOCK_SYMBOLS      # 1.5714, vs log2(3) = 1.58496

# MSD-first place values, and the bit weights of an 11-bit word. Module-level because they are
# constants of the code, not of a call.
_PAM3_POW3 = 3 ** np.arange(PAM3_BLOCK_SYMBOLS - 1, -1, -1, dtype=np.int64)
_PAM3_POW2 = 1 << np.arange(PAM3_BLOCK_BITS - 1, -1, -1, dtype=np.int64)


def pam3_encode(bits):
    """PAM3 block code: 11 bits to 7 ternary LEVELS in {-1, 0, +1} (USB4 Version 2.0 Gen 4).

    Whole blocks only — a partial block is refused rather than padded, because padding invents
    bits that the caller did not send and nothing downstream of the waveform could reveal it.
    Returns `physics.pam_levels(3)` values, so the output feeds `physics.from_symbols` or
    `Signal.symbols` directly. See the PRELIMINARY note above on the codeword table.
    """
    b = _bits01(bits)
    if b.size % PAM3_BLOCK_BITS:
        raise ValueError(f"PAM3 takes whole {PAM3_BLOCK_BITS}-bit blocks; got {b.size} bits "
                         f"({b.size % PAM3_BLOCK_BITS} over)")
    words = b.reshape(-1, PAM3_BLOCK_BITS) @ _PAM3_POW2
    digits = (words[:, None] // _PAM3_POW3[None, :]) % 3          # MSD first
    return _P.pam_levels(3)[digits.reshape(-1)]


def pam3_decode(symbols):
    """Ternary levels back to bits — the exact inverse of `pam3_encode`.

    The alphabet is CHECKED: a value that is not one of the three levels is a slicer output or
    a waveform sample, not a codeword, and rounding it here is how a decode silently succeeds
    on garbage. Slice to the levels first, then decode.
    """
    s = np.asarray(symbols, dtype=float)
    if s.ndim != 1:
        raise ValueError(f"symbols must be a 1-D array, not shape {s.shape}")
    if s.size % PAM3_BLOCK_SYMBOLS:
        raise ValueError(f"PAM3 takes whole {PAM3_BLOCK_SYMBOLS}-symbol blocks; got {s.size} "
                         f"symbols ({s.size % PAM3_BLOCK_SYMBOLS} over)")
    d = np.rint(s)
    if s.size and (np.max(np.abs(s - d)) > 0 or np.max(np.abs(d)) > 1):
        raise ValueError("symbols must be exactly ternary (-1, 0, +1); slice the waveform to "
                         "physics.pam_levels(3) before decoding")
    digits = (d.reshape(-1, PAM3_BLOCK_SYMBOLS) + 1.0).astype(np.int64)
    words = digits @ _PAM3_POW3
    if words.size and words.max() >= 1 << PAM3_BLOCK_BITS:
        raise ValueError(f"symbol block {int(words.max())} is outside the {1 << PAM3_BLOCK_BITS} "
                         f"codewords this mapping uses (3^{PAM3_BLOCK_SYMBOLS} = "
                         f"{3 ** PAM3_BLOCK_SYMBOLS} exist, 139 are spare)")
    return ((words[:, None] // _PAM3_POW2[None, :]) % 2).reshape(-1)


# ==================================================================== BLOCK CODES AND SCRAMBLERS
# The layer that decides what the transmitted BIT STREAM looks like, above the level coding and
# below the waveform. Two families, and the difference between them is the reason a dataset
# carries one rather than the other:
#
#   * A BLOCK CODE bounds running disparity and run length BY CONSTRUCTION. 8b/10b keeps
#     |RD| <= 3 at every bit and the run <= 5, on every payload including an all-zeros idle.
#     (`dc_balanced` above is the same principle stripped to its core.)
#   * A SCRAMBLER RANDOMISES them. The mean goes to 0.5 and a long run becomes improbable
#     rather than impossible, but the running disparity still random-walks -- so an AC-coupled
#     record of a scrambled link HAS baseline wander that a block-coded record does not, and a
#     record that carries the wrong one of the two is wrong in exactly that.
#
# Each scrambler IS a polynomial and a seed and nothing else, so both are arguments, with the
# standard's value as the documented default (the constants below carry the clause). Two
# mechanisms, which differ in what they cost:
#
#   * SELF-SYNCHRONOUS (64b/66b): the register is fed the TRANSMITTED bit, so a receiver rebuilds
#     the state from what it receives -- no seed agreement, at the price of multiplying one bit
#     error by the number of taps.
#   * ADDITIVE / synchronous (128b/130b, 128b/132b): a free-running LFSR keystream XORed with the
#     payload. One bit error stays one bit error, and the seed must be agreed -- which is why
#     those standards state per-lane seed values.
#
# WHY A SOURCE OP AND NOT A CHAIN STAGE. The composer's ops transform SAMPLES, and a line code
# transforms BITS; there is no sample-domain stage that can express one. So the composable path
# is `bits -> [code] -> symbols` inside one source op (`Signal.coded`), which is what puts the
# scheme, the polynomial and the seed into the recipe and therefore into the content digest.
#
# PRELIMINARY, stated here rather than implied by silence:
#   * The 8b/10b tables are transcribed from IEEE 802.3 Clause 36 Tables 36-1/36-2 (the same
#     tables as ANSI INCITS 230 / FC-PH) WITHOUT the document in hand. They are checked against
#     every structural property the standard forces -- per-codeword disparity, injectivity of the
#     RD- column, all 20 disparity-neutral 6-bit words used, the deliberately omitted
#     111100/000011 pair, |RD| <= 3, run <= 5, the comma in K.28.1/5/7 and nowhere else, and a
#     256-byte round trip through an independent decoder -- which a mis-transcribed row fails.
#     That is strong evidence, not a citation; the rows still want checking against the tables.
#   * 128b/132b: the framing (4 header bits + 128 payload) is arithmetic, but the polynomial and
#     seed defaults are the 16-bit scrambler of the same specification family's earlier
#     signalling rather than a value read off the 128b/132b clause, and the header VALUE is a
#     placeholder, not the standard's block-header encoding. Both are arguments.
#   * The bit-level conventions an additive scrambler's own pseudocode fixes -- which register bit
#     is the output tap, what phase a block starts at -- are this module's Fibonacci convention,
#     not one read off a specification. They do not move any property asserted here (the
#     keystream is the polynomial's m-sequence either way); they do decide whether the bits match
#     a real link's capture.

# Taps are the EXPONENTS of G(x) = 1 + sum(x^t); the constant term is implied and never listed.
# The register is max(taps) bits wide and tap t reads the bit inserted t shifts ago -- the same
# convention as `physics.PRBS_TAPS` / `physics.prbs`, so a polynomial can be read across.
POLY_64B66B = (58, 39)                  # IEEE 802.3 Clause 49.2.6: G(x) = 1 + x^39 + x^58
SEED_64B66B = (1 << 58) - 1             # all ones -- the state the existing scrambler starts from
POLY_128B130B = (23, 21, 16, 8, 5, 2)   # PCI Express Base Specification Rev 4.0 sec 4.2.2.2:
SEED_128B130B = 0x1DBFBC                # G(x) = x^23+x^21+x^16+x^8+x^5+x^2+1, and lane 0's seed.
# The per-lane seed TABLE of that clause is not reproduced: lane 0's value is the one established
# here, and a second lane wants its own seed passed in rather than this one silently reused.
POLY_128B132B = (16, 5, 4, 3)           # PRELIMINARY (see above): G(x) = x^16+x^5+x^4+x^3+1 and
SEED_128B132B = 0xFFFF                  # its all-ones seed, from the 128b/132b family's Gen 1.


def _lfsr_mask(width):
    return (1 << int(width)) - 1


def _lfsr_feedback(state, taps):
    fb = 0
    for t in taps:
        fb ^= (state >> (int(t) - 1)) & 1
    return fb


def _lfsr_poly(poly):
    """(taps, width) from a polynomial, refusing what cannot be an LFSR. A tap of 0 would be the
    implied constant term restated, and a repeated tap cancels itself in GF(2)."""
    taps = tuple(int(t) for t in poly)
    if not taps or min(taps) < 1 or len(set(taps)) != len(taps):
        raise ValueError(f"poly is the distinct, non-zero EXPONENTS of G(x) = 1 + sum(x^t); "
                         f"got {poly!r}")
    return taps, max(taps)


def lfsr_keystream(n, poly, seed):
    """``n`` bits of an additive scrambler's keystream: the m-sequence of ``poly`` from ``seed``.

    XOR it into the payload to scramble and again to descramble. The register never sees the
    data, which is what makes a bit error stay one bit error and the seed something the two ends
    must agree on."""
    taps, width = _lfsr_poly(poly)
    st = int(seed) & _lfsr_mask(width) or 1     # state 0 is the dead state, as in physics.prbs
    out = np.empty(int(n), dtype=np.int64)
    for i in range(int(n)):
        out[i] = st & 1
        st = ((st << 1) | _lfsr_feedback(st, taps)) & _lfsr_mask(width)
    return out


def scramble_additive(bits, poly, seed):
    """Additive (synchronous) scrambling: payload XOR keystream. Its own inverse, so the same
    function descrambles -- with the same seed, which it cannot discover."""
    b = _bits01(bits)
    return b ^ lfsr_keystream(len(b), poly, seed)


def scramble_self_sync(bits, poly, seed):
    """Self-synchronous scrambling: the TRANSMITTED bit is fed back into the register, so the
    receiver rebuilds the state from what it receives. Costs an error multiplied by the number of
    taps; buys a descrambler that needs no seed."""
    b = _bits01(bits)
    taps, width = _lfsr_poly(poly)
    st = int(seed) & _lfsr_mask(width)
    out = np.empty(len(b), dtype=np.int64)
    for i in range(len(b)):
        s = int(b[i]) ^ _lfsr_feedback(st, taps)
        out[i] = s
        st = ((st << 1) | s) & _lfsr_mask(width)
    return out


def descramble_self_sync(bits, poly, seed):
    """Inverse of `scramble_self_sync`: identical feedback, but the register is fed the RECEIVED
    bit. That is the whole of self-synchronisation -- a wrong initial state is wrong for
    max(poly) bits and exact after that, and one corrupted bit corrupts len(taps)+1 and no more."""
    b = _bits01(bits)
    taps, width = _lfsr_poly(poly)
    st = int(seed) & _lfsr_mask(width)
    out = np.empty(len(b), dtype=np.int64)
    for i in range(len(b)):
        r = int(b[i])
        out[i] = r ^ _lfsr_feedback(st, taps)
        st = ((st << 1) | r) & _lfsr_mask(width)
    return out


# --------------------------------------------------------------------- framing
def _whole_blocks(bits, payload):
    """Whole payload blocks only, and the remainder is DROPPED rather than padded: padding
    invents bits the caller never sent, and nothing downstream of the waveform could reveal it."""
    b = _bits01(bits)
    nblk = len(b) // int(payload)
    return b[:nblk * int(payload)]


def _frame(payload_bits, header, payload):
    """Interleave a per-block header with the already-coded payload. The header is NOT scrambled
    (IEEE 802.3 49.2.4, PCI Express Rev 4.0 sec 4.2.2.1), which is also what bounds the run
    length a scrambler only makes improbable: a block boundary always carries a transition."""
    blk = payload_bits.reshape(-1, int(payload))
    hdr = np.tile(np.asarray(header, dtype=np.int64).reshape(1, -1), (len(blk), 1))
    return np.concatenate([hdr, blk], axis=1).reshape(-1)


def _unframe(bits, header_len, payload):
    total = int(header_len) + int(payload)
    b = _bits01(bits)
    if len(b) % total:
        raise ValueError(f"a {total}-bit block stream is a whole number of blocks; got {len(b)} "
                         f"bits ({len(b) % total} over)")
    return b.reshape(-1, total)[:, int(header_len):].reshape(-1)


def scramble_64b66b_framed(bits, poly=POLY_64B66B, seed=SEED_64B66B):
    """64b/66b (IEEE 802.3 Clause 49): a 2-bit sync header per 64-bit block, payload scrambled by
    the self-synchronous G(x) = 1 + x^39 + x^58. The scrambler runs CONTINUOUSLY across blocks --
    its register is not reset per block and the header bits do not advance it.

    This is `scramble_64b66b` with the polynomial and the seed exposed; with the defaults it is
    bit-identical to it (asserted), so the audited function is the implementation and this is not
    a second copy of it."""
    return _frame(scramble_self_sync(_whole_blocks(bits, 64), poly, seed), (0, 1), 64)


def descramble_64b66b(bits, poly=POLY_64B66B, seed=SEED_64B66B):
    """Strip the sync headers, descramble, and hand back the payload exactly."""
    return descramble_self_sync(_unframe(bits, 2, 64), poly, seed)


def scramble_128b130b(bits, poly=POLY_128B130B, seed=SEED_128B130B, sync=(0, 1)):
    """128b/130b (PCI Express Base Specification Rev 4.0 sec 4.2.2.2 -- the Gen 3 / Gen 4
    encoding): a 2-bit sync header per 128-bit block, payload scrambled ADDITIVELY by a
    free-running 23-bit LFSR. ``sync=(0, 1)`` is that clause's Data Block header; an Ordered Set
    block header is (1, 0)."""
    return _frame(scramble_additive(_whole_blocks(bits, 128), poly, seed), sync, 128)


def descramble_128b130b(bits, poly=POLY_128B130B, seed=SEED_128B130B):
    """Strip the sync headers and XOR the same keystream back out. Needs the seed: an additive
    scrambler does not self-synchronise."""
    return scramble_additive(_unframe(bits, 2, 128), poly, seed)


def scramble_128b132b(bits, poly=POLY_128B132B, seed=SEED_128B132B, header=(0, 0, 0, 0)):
    """128b/132b: four header bits per 128-bit payload block, payload additively scrambled.

    PRELIMINARY, and specifically: the 4 + 128 framing is arithmetic and the scrambling is a
    real additive scrambler, but ``header`` is a PLACEHOLDER rather than the standard's
    data/control block header encoding, and the ``poly``/``seed`` defaults come from the same
    specification family's earlier signalling. What is missing to finish it: the block-header
    values and the scrambler clause of the 128b/132b specification itself. Pass all three
    arguments to match a real link."""
    return _frame(scramble_additive(_whole_blocks(bits, 128), poly, seed), header, 128)


def descramble_128b132b(bits, poly=POLY_128B132B, seed=SEED_128B132B):
    return scramble_additive(_unframe(bits, 4, 128), poly, seed)


# --------------------------------------------------------------------- 8b/10b
# IEEE 802.3 Clause 36 Tables 36-1 (5b/6b) and 36-2 (3b/4b). See the PRELIMINARY note above for
# what is established about these rows and what is not.
#
# Each entry is (code used when the running disparity is NEGATIVE, code used when it is
# POSITIVE). Where the standard lists one code for an input its disparity is neutral and it is
# legal from either, so both halves are that code; where it lists two, the RD- column has
# disparity +2 or 0 and the RD+ column -2 or 0. Selecting by the current RD is the whole
# mechanism: it is what holds the accumulated disparity inside +-1 at every sub-block boundary.
# D.07 and y=3 are the rows that show the rule is not just about disparity -- BOTH of their codes
# are neutral, and the standard still selects by RD, to stop a run crossing the boundary.
def _cw(s):
    return tuple(int(c) for c in s)


def _cw_pair(minus, plus=None):
    return (_cw(minus), _cw(plus if plus is not None else minus))


# x = EDCBA, the low 5 bits of the byte -> abcdei
CODE_5B6B = {
    0:  _cw_pair("100111", "011000"),  1:  _cw_pair("011101", "100010"),
    2:  _cw_pair("101101", "010010"),  3:  _cw_pair("110001"),
    4:  _cw_pair("110101", "001010"),  5:  _cw_pair("101001"),
    6:  _cw_pair("011001"),            7:  _cw_pair("111000", "000111"),
    8:  _cw_pair("111001", "000110"),  9:  _cw_pair("100101"),
    10: _cw_pair("010101"),            11: _cw_pair("110100"),
    12: _cw_pair("001101"),            13: _cw_pair("101100"),
    14: _cw_pair("011100"),            15: _cw_pair("010111", "101000"),
    16: _cw_pair("011011", "100100"),  17: _cw_pair("100011"),
    18: _cw_pair("010011"),            19: _cw_pair("110010"),
    20: _cw_pair("001011"),            21: _cw_pair("101010"),
    22: _cw_pair("011010"),            23: _cw_pair("111010", "000101"),
    24: _cw_pair("110011", "001100"),  25: _cw_pair("100110"),
    26: _cw_pair("010110"),            27: _cw_pair("110110", "001001"),
    28: _cw_pair("001110"),            29: _cw_pair("101110", "010001"),
    30: _cw_pair("011110", "100001"),  31: _cw_pair("101011", "010100"),
}
# K.28 is the only control character with a 5b/6b code of its own; K.23/27/29/30 reuse the data
# code for the same x, which is why only those four x values can carry a K.x.7.
CODE_K28 = _cw_pair("001111", "110000")

# y = HGF, the high 3 bits -> fghj. The K column differs from the D column at y = 1, 2, 5, 6:
# the same neutral words, assigned to the opposite running disparity. That difference is what
# puts the comma into K.28.1/5/7 -- with the D assignment, K.28.5 would not contain one.
CODE_3B4B = {
    0: _cw_pair("1011", "0100"), 1: _cw_pair("1001"),          2: _cw_pair("0101"),
    3: _cw_pair("1100", "0011"), 4: _cw_pair("1101", "0010"),  5: _cw_pair("1010"),
    6: _cw_pair("0110"),         7: _cw_pair("1110", "0001"),
}
CODE_3B4B_K = {
    0: _cw_pair("1011", "0100"), 1: _cw_pair("0110", "1001"),  2: _cw_pair("1010", "0101"),
    3: _cw_pair("1100", "0011"), 4: _cw_pair("1101", "0010"),  5: _cw_pair("0101", "1010"),
    6: _cw_pair("1001", "0110"), 7: _cw_pair("0111", "1000"),
}
# D.x.A7, the ALTERNATE y=7 code, used instead of the primary exactly when the primary would put
# five like bits against the sub-block boundary and so risk a run of 6 with the neighbouring code
# group: x in {17,18,20} when RD is negative entering the 3b/4b sub-block, x in {11,13,14} when it
# is positive. Dropping this rule shows up immediately as a run of 6.
CODE_3B4B_ALT7 = _cw_pair("0111", "1000")
ALT7_WHEN_RD_MINUS = (17, 18, 20)
ALT7_WHEN_RD_PLUS = (11, 13, 14)

# Every control character the code defines: the eight K.28.y, and the four K.x.7 whose 5b/6b code
# carries non-zero disparity. Anything else is not a code group, and encoding one is refused
# rather than approximated -- an invented K character would decode to a plausible byte.
VALID_K = frozenset([(28, y) for y in range(8)] + [(23, 7), (27, 7), (29, 7), (30, 7)])


def _cw_disparity(word):
    return 2 * sum(word) - len(word)


def _cw_pick(pair, rd):
    """The RD- column when the running disparity is negative, the RD+ column when positive, plus
    the RD the code leaves. RD is carried as -1/+1 because every code's disparity is -2, 0 or +2:
    a non-neutral code always FLIPS it and a neutral one leaves it alone."""
    w = pair[0] if rd < 0 else pair[1]
    d = _cw_disparity(w)
    return w, (rd if d == 0 else (1 if d > 0 else -1))


def encode_8b10b_words(words, rd=-1, control=None):
    """Encode bytes (and control characters) to a 0/1 stream in transmission order
    a b c d e i f g h j. Returns ``(bits, rd)``: the ending running disparity, so a long stream
    can be encoded in pieces without a seam.

    ``words`` are ints 0..255 read as HGFEDCBA (x is the low 5 bits, y the high 3), ``rd`` is the
    starting running disparity as -1 or +1, and ``control[i]`` marks word i as a K character.

    The 3b/4b sub-block is selected by the RD that the 5b/6b sub-block LEAVES, not by the RD the
    code group started from. That ordering is the procedure, not a detail: get it backwards and
    K.28.5 stops containing the comma, which is the one thing byte alignment depends on."""
    words = [int(v) for v in words]
    if control is not None:
        control = [bool(c) for c in control]
        if len(control) != len(words):
            raise ValueError(f"control has {len(control)} flags for {len(words)} words")
    if rd not in (-1, 1):
        raise ValueError(f"rd is the running disparity as -1 or +1; got {rd!r}")
    out = []
    for i, v in enumerate(words):
        if not 0 <= v <= 255:
            raise ValueError(f"8b/10b encodes bytes; got {v}")
        k = bool(control[i]) if control is not None else False
        x, y = v & 31, (v >> 5) & 7
        if k and (x, y) not in VALID_K:
            raise ValueError(f"K.{x}.{y} is not a code group of this code; the control characters "
                             f"are {sorted(VALID_K)}")
        w6, rd = _cw_pick(CODE_K28 if (k and x == 28) else CODE_5B6B[x], rd)
        if k:
            four = CODE_3B4B_K[y]
        elif y == 7 and ((rd < 0 and x in ALT7_WHEN_RD_MINUS)
                         or (rd > 0 and x in ALT7_WHEN_RD_PLUS)):
            four = CODE_3B4B_ALT7
        else:
            four = CODE_3B4B[y]
        w4, rd = _cw_pick(four, rd)
        out.extend(w6)
        out.extend(w4)
    return np.array(out, dtype=np.int64), rd


def _reverse_code_tables():
    """Decode maps built FROM the encode tables, so the module holds ONE copy of the table: a
    hand-written reverse table is a second chance to mis-transcribe a row, and the round-trip
    test would then pass on two matching errors."""
    six, four = {}, {}
    for x, pair in CODE_5B6B.items():
        for w in pair:
            six.setdefault(w, x)
    for w in CODE_K28:
        six[w] = 28                                  # K.28's own 6-bit code, shared with nothing
    for y, pair in CODE_3B4B.items():
        for w in pair:
            four.setdefault(w, set()).add(("D", y))
    for y, pair in CODE_3B4B_K.items():
        for w in pair:
            four.setdefault(w, set()).add(("K", y))
    for w in CODE_3B4B_ALT7:
        four.setdefault(w, set()).add(("D", 7))
    return six, four, {w for w in CODE_K28}


_DEC_5B6B, _DEC_3B4B, _K28_WORDS = _reverse_code_tables()


def decode_8b10b(bits, rd=-1):
    """Decode a code-group stream to ``(values, control, rd)`` -- bytes, the K flag per byte, and
    the running disparity the stream ends in.

    THE 3b/4b SUB-BLOCK CANNOT BE DECODED FROM ITS FOUR BITS ALONE. The control rows y=1 and y=6,
    and y=2 and y=5, are the same two neutral words assigned to opposite running disparities, so
    K.28.2 and K.28.5 differ only in the RD they were emitted from -- which is why the decoder
    tracks RD rather than reading a table. The RD entering the sub-block is read off the 5b/6b
    word's OWN disparity whenever that is non-zero (so every K character decodes correctly from
    either starting disparity, K.28's code being +-2 always) and from history only after a neutral
    one; ``rd`` is where the stream is assumed to start.

    An unknown 6-bit or 4-bit word RAISES. A code group no encoder can emit means a corrupted
    capture, and handing back a plausible byte for it is how the corruption disappears.

    D or K is decidable without the encoder's state: K.28 owns its own 5b/6b code, and the four
    K.x.7 characters use precisely the x values that never take the alternate D.x.A7 -- which is
    why those four were chosen."""
    b = _bits01(bits)
    if len(b) % 10:
        raise ValueError(f"a code-group stream is a multiple of 10 bits; got {len(b)}")
    if rd not in (-1, 1):
        raise ValueError(f"rd is the running disparity as -1 or +1; got {rd!r}")
    vals, ctrl = [], []
    for g in b.reshape(-1, 10):
        w6, w4 = tuple(int(v) for v in g[:6]), tuple(int(v) for v in g[6:])
        if w6 not in _DEC_5B6B or w4 not in _DEC_3B4B:
            raise ValueError(f"not a code group: {''.join(str(int(v)) for v in g)}")
        x = _DEC_5B6B[w6]
        d6 = _cw_disparity(w6)
        rd = rd if d6 == 0 else (1 if d6 > 0 else -1)         # RD entering the 3b/4b sub-block
        col = 0 if rd < 0 else 1
        is_k = (w6 in _K28_WORDS) or ((x, 7) in VALID_K and w4 == CODE_3B4B_ALT7[col])
        table = CODE_3B4B_K if is_k else CODE_3B4B
        y = next((yy for yy, pair in table.items() if pair[col] == w4), None)
        if y is None and not is_k and w4 == CODE_3B4B_ALT7[col]:
            y = 7                                            # D.x.A7, the alternate y=7 code
        if y is None:
            raise ValueError(f"{''.join(str(int(v)) for v in g)} carries a 3b/4b sub-block that "
                             f"is not legal from running disparity {rd:+d}")
        d4 = _cw_disparity(w4)
        rd = rd if d4 == 0 else (1 if d4 > 0 else -1)
        vals.append(int(x) | (int(y) << 5))
        ctrl.append(bool(is_k))
    return np.array(vals, dtype=np.int64), np.array(ctrl, dtype=bool), rd


def code_8b10b(bits, rd=-1, lsb_first=True):
    """8b/10b over a BIT stream: pack ``bits`` into bytes, encode, return the coded bits.

    ``lsb_first`` makes the first bit of each group of eight the byte's bit A, the order the
    standard's own figures transmit an octet in. It decides WHICH code group each group of eight
    becomes; it moves no statistic asserted of the output. What would falsify the choice is a
    capture that has to decode back to a particular octet stream.

    Data characters only -- a control character is not data, so it is not reachable from a bit
    stream. `encode_8b10b_words(..., control=...)` is the door to those."""
    b = _bits01(bits)
    g = b[:len(b) // 8 * 8].reshape(-1, 8)
    w = 1 << (np.arange(8) if lsb_first else np.arange(7, -1, -1))
    return encode_8b10b_words(g @ w, rd=rd)[0]


# --------------------------------------------------------------------- the composable surface
# One name per code so a caller -- and the op that records it -- never has to know which family a
# scheme belongs to. Each entry: (encoder, bit-stream decoder or None, payload bits per block,
# coded bits per block). The two block sizes are what let a caller size a record before coding it.
SCHEMES = {
    "64b66b":      (scramble_64b66b_framed, descramble_64b66b,   64,  66),
    "128b130b":    (scramble_128b130b,      descramble_128b130b, 128, 130),
    "128b132b":    (scramble_128b132b,      descramble_128b132b, 128, 132),
    "8b10b":       (code_8b10b,             None,                8,   10),
    "dc_balanced": (dc_balanced,            None,                8,   9),
}


def line_code(scheme, bits, **kw):
    """``bits -> coded bits`` for a named scheme. ``kw`` goes to that scheme's own function (a
    scrambler's ``poly``/``seed``, 8b/10b's ``rd``), so the standard's defaults live in exactly
    one place: pass nothing and a recipe records nothing, and the day a default is corrected
    every record moves with it instead of carrying a stale copy."""
    if scheme not in SCHEMES:
        raise ValueError(f"unknown line code {scheme!r}; offered: {sorted(SCHEMES)}")
    return SCHEMES[scheme][0](bits, **kw)


def line_decode(scheme, bits, **kw):
    """Inverse of `line_code` where the scheme has one on the bit stream."""
    if scheme not in SCHEMES:
        raise ValueError(f"unknown line code {scheme!r}; offered: {sorted(SCHEMES)}")
    dec = SCHEMES[scheme][1]
    if dec is None:
        raise ValueError(f"{scheme!r} has no bit-stream inverse here: 8b/10b decodes to BYTES "
                         f"(the packing order is the caller's) via decode_8b10b, and "
                         f"dc_balanced's per-block flag bit is the caller's to strip")
    return dec(bits, **kw)


def coded_symbols(scheme, bits, levels=(-1.0, 1.0), **kw):
    """``bits -> [code] -> symbols``: the coded bit stream as per-UI NRZ levels, ready for
    `physics.from_symbols` / `Signal.symbols`. ``levels`` maps (0, 1); the default is the
    unit-amplitude pair every other carrier in the library emits."""
    out = line_code(scheme, bits, **kw)
    lo, hi = (float(v) for v in levels)
    return np.where(out > 0, hi, lo).astype(float)
