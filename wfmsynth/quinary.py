"""
wfmsynth.quinary — 4D-PAM5 / 8B1Q4: the quinary line code of the four-pair gigabit link.

WHY THIS IS NOT `pam(5)`. `physics.pam_symbols(5, ...)` draws five evenly spaced levels
UNIFORMLY, and says so in its own docstring. The real code is not uniform and is not
one-dimensional:

  * 8B1Q4 maps one octet to one QUARTET of quinary symbols -- four symbols sent at the same
    instant, one on each wire pair. The rate identity is 125 MBd * 2 bit/symbol * 4 pairs =
    1000 Mb/s (IEEE 802.3 1.4.183 gives the 125 MBd).
  * The quinary alphabet is PARTITIONED into two one-dimensional subsets, X = {-1, +1} and
    Y = {-2, 0, +2}. A quartet's TYPE is which subset each of its four coordinates came from;
    the sixteen types pair off with their complements into eight 4D subsets, and a rate-8/9
    convolutional code picks the subset while six data bits pick the point inside it. So
    consecutive quartets are not independent, and the marginal level distribution is NOT 1/5
    each -- see `quartet_constellation`.
  * The transmitted symbol stream is then filtered by a PARTIAL-RESPONSE shaping filter,
    which turns five transmit levels into seventeen observed ones. A naive model that applies
    (1 + D)/2 gets nine, and is therefore wrong in a way that is trivially visible.

WHAT IS CITED AND WHAT IS NOT. Everything here that carries a clause number was read from the
standard's own numbering. The pieces marked PRELIMINARY are corroborated from secondary
technical descriptions of Clause 40 and need the primary text to be promoted:

  PRELIMINARY  the partial-response transfer function F(z) = 3/4 + 1/4 z^-1, and the 17
               observed levels in 1/8 V steps that follow from it.
  PRELIMINARY  the side-stream scrambler polynomials 1 + x^13 + x^33 (MASTER) and
               1 + x^20 + x^33 (SLAVE) of Clause 40.3.1.4.
  PRELIMINARY  test mode 4's three derived bit sequences and their 3-bit-to-quinary map. The
               11-bit generator gs1 = 1 + x^9 + x^11 and the update Scr[0] <- Scr[10] ^ Scr[8]
               ARE stated; the map from the three derived bits to a symbol is Clause
               40.6.1.1.2's table, which is not reproduced here, so `test_mode_symbols(4)`
               uses a stated stand-in map and says so.
  JUDGEMENT    which 64 of each 4D subset's points the code uses. That is Table 40-1's
               business and the table is not reproduced here; `quartet_constellation` states
               a rule (lowest energy, in antipodal pairs) and `_SELECTION_RULE` says what
               would falsify it.

numpy only -- no import from `physics`, so `physics` can import this.
"""
from __future__ import annotations

import itertools
from collections import defaultdict

import numpy as np

# ------------------------------------------------------------------ the alphabet and the rate
QUINARY_SYMBOLS = (-2, -1, 0, 1, 2)
PAIRS = 4                        # four wire pairs, one quinary symbol on each per symbol period
BITS_PER_SYMBOL = 2              # 8 bits per quartet / 4 pairs
SYMBOL_RATE_BD = 125e6           # IEEE 802.3 1.4.183 and 40.6.1.2.6: 125.00 MBd, all four pairs

# IEEE 802.3 40.6.1.2.1 (peak differential output voltage and level accuracy), measured on the
# test-mode-1 pattern: the +/-2 symbol (points A, B) is 670..820 mV peak differential, and the
# +/-1 symbol (points C, D) sits within 2 % of HALF the average of |A| and |B|. That 0.5 is why
# `quinary_volts` is symbol/2 and nothing more: the clause requires the five levels to be
# evenly spaced, so a compression model belongs in `physics.nominal_nonlinearity`, not here.
PEAK_DIFFERENTIAL_V = (0.670, 0.820)
LEVEL_ACCURACY = 0.02
# 40.6.1.2.4: peak transmitter distortion, measured against the ideal reference AFTER partial
# response filtering, shall be under 10 mV -- about 1 % of the 1 V test-mode-4 signal. Useful as
# the floor a synthetic transmitter should sit under if it claims to be compliant.
PEAK_DISTORTION_V = 0.010

# PRELIMINARY. The transmit partial-response filter, F(z) = 3/4 + 1/4 z^-1: one symbol period of
# DELIBERATE intersymbol interference, introduced to roll the spectrum off inside the emissions
# budget the cabling was qualified to. Unit DC gain by construction (3/4 + 1/4 = 1), so it moves
# no mean level. Its observable signature is the LEVEL COUNT: 17 distinct levels on a 1/8 V
# lattice, against 9 on a 1/4 V lattice for the (1 + D)/2 a naive model reaches for. Corroborated
# from secondary descriptions of Clause 40 and from the "pseudo-random 17 level waveform" the
# transmitter-distortion test observes; needs the primary clause text to drop PRELIMINARY.
PARTIAL_RESPONSE_TAPS = (0.75, 0.25)


def quinary_volts(symbols, v_peak=1.0):
    """Quinary symbols in {-2..+2} -> peak differential volts. `v_peak` is the amplitude of the
    +/-2 symbol; the +/-1 symbol comes out at exactly half of it, which is what 40.6.1.2.1
    requires of a conforming transmitter (to within 2 %; this map spends none of that)."""
    return np.asarray(symbols, float) * (0.5 * float(v_peak))


# ------------------------------------------------------- the 1D subsets and the 4D constellation
#
# The partition every part of the code is built on: X carries the two odd levels and Y the three
# even ones. The minimum distance inside X is 2 and inside Y is 2, while the distance between a
# point of X and its nearest point of Y is 1 -- which is the whole reason the code exists. A
# receiver that knows which subset a coordinate came from has twice the noise margin of one
# slicing all five levels.
SUBSET_X = (-1, 1)
SUBSET_Y = (-2, 0, 2)


def quartet_subsets():
    """The eight 4D subsets, as lists of 4-tuples. Together they PARTITION all 5^4 = 625
    quartets.

    Construction: label each coordinate X or Y (sixteen types), then pair each type with its
    complement (X<->Y on every coordinate). Sixteen types, eight complementary pairs, hence
    eight subsets. Their sizes are 97 (XXXX + YYYY), 78 (three X and one Y, four ways) and 72
    (two and two, three ways) -- 97 + 4*78 + 3*72 = 625. The smallest is 72, which is why 64
    points per subset is reachable in all eight.

    Complement pairing rather than any other pairing: a type and its complement differ on every
    coordinate, so the two halves of a subset are as far apart as the geometry allows, which is
    what makes the subset a usable decision region."""
    seen, out = set(), []
    for t in itertools.product((0, 1), repeat=PAIRS):          # 0 = X, 1 = Y
        if t in seen:
            continue
        c = tuple(1 - b for b in t)
        seen.add(t)
        seen.add(c)
        pts = set()
        for tt in (t, c):
            pts.update(itertools.product(*[(SUBSET_X if b == 0 else SUBSET_Y) for b in tt]))
        out.append(sorted(pts))
    return out


# JUDGEMENT, and this is the one thing in this module that Table 40-1 would overrule. The code
# places 9 bits (8 data + 1 convolutional) as 3 bits of subset and 6 bits of point, so it uses
# exactly 64 of each subset's 72..97 points. WHICH 64 is the table's business. The rule used
# here: lowest energy first (a set-partitioned code minimises average transmit power for a given
# minimum distance), taking points in ANTIPODAL PAIRS so the 512-point constellation is exactly
# closed under negation.
#
# The pairing constraint is not cosmetic. An unbalanced constellation has a nonzero mean level,
# the transmitter injects DC that the real one does not, and AC-coupling turns that into baseline
# wander that is an artifact of the model. Requiring pairs makes the mean EXACTLY zero. It also
# forces one exclusion: (0,0,0,0) is its own negation, so a 64-point antipodally closed set
# cannot contain it, and the lowest-energy quartet is therefore dropped from its subset.
#
# WHAT WOULD FALSIFY IT: Table 40-1 itself. If the standard's 512 points include the all-zero
# quartet, this selection differs from it and the marginal below moves by at most 4/2048.
_SELECTION_RULE = "lowest energy, in antipodal pairs, excluding the self-negating all-zero point"
_ZERO = (0,) * PAIRS


def quartet_constellation():
    """The 512 quartets the code transmits, as an int array of shape (8, 64, 4): eight 4D
    subsets (the 3 subset bits) of 64 points each (the 6 point bits).

    Selected by `_SELECTION_RULE` -- see the note above for what that is a judgement about.
    The consequence a caller cares about is the marginal level distribution, which is
    {-2, -1, 0, 1, 2} -> {21, 28, 30, 28, 21}/128: the outer levels 18 % rarer than uniform,
    the zero level 17 % more common, mean exactly 0 and mean square exactly 1.75 (0.58 dB less
    transmit power than uniform PAM5 at the same peak)."""
    out = np.empty((8, 64, PAIRS), dtype=np.int64)
    for k, sub in enumerate(quartet_subsets()):
        groups = defaultdict(list)
        for p in sub:
            if p == _ZERO:
                continue
            groups[(sum(v * v for v in p), tuple(abs(v) for v in p))].append(p)
        chosen = []
        for key in sorted(groups):
            done = set()
            for p in sorted(groups[key], reverse=True):
                if p in done:
                    continue
                q = tuple(-v for v in p)
                done.update((p, q))
                if len(chosen) + 2 <= 64:
                    chosen += [p, q]
            if len(chosen) >= 64:
                break
        out[k] = np.array(sorted(chosen), dtype=np.int64)
    return out


# ------------------------------------------------------------------ the side-stream scrambler
#
# PRELIMINARY: Clause 40.3.1.4 gives the transmit side-stream scrambler as a 33-bit LFSR,
# 1 + x^13 + x^33 in the MASTER PHY and 1 + x^20 + x^33 in the SLAVE. The polynomials are
# corroborated from secondary descriptions; the 33-bit width and the master/slave split are what
# matter here and are not in doubt.
#
# WHY IT IS IN THIS MODULE AT ALL: the marginal level distribution of the code is the
# constellation's marginal ONLY if the 9-bit index is uniform. The scrambler is what makes it
# uniform, and without it the "correct symbol statistics" claim rests on the data happening to be
# random. It also gives master and slave DIFFERENT streams from the same seed, which is exactly
# what the two ends of a real link have -- and what makes the far-end aggressors of a four-pair
# link uncorrelated with the near-end ones.
SCRAMBLER_TAPS = {"master": (33, 13), "slave": (33, 20)}

# Steps to run the register before the first bit is emitted. A real link's scrambler has been
# running since the link came up, and a record is a WINDOW on that -- the same reasoning the
# composer's lead-in rests on. Without a warm-up, a register loaded with a small state emits its
# transient: the single 1 bit takes 13 steps (master) or 20 (slave) to reach the second tap, so
# both roles open with a long run of zeros and the two ENDS of a link come out correlated.
# MEASURED on 512-symbol shaped lanes: master vs slave correlate at 0.166 with no warm-up and
# 0.047 with this one, against the 1/sqrt(512) = 0.044 expected of two independent streams -- so
# the warm-up is what makes the two ends of the link independent rather than nearly the same
# signal. 1024 is 31 register lengths, chosen with margin rather than tuned to the measurement.
SCRAMBLER_WARMUP = 1024


def side_stream_bits(length, seed=1, role="master", warmup=SCRAMBLER_WARMUP):
    """`length` scrambler bits from the Clause 40.3.1.4 side-stream LFSR.

    A local Fibonacci LFSR rather than `physics.prbs`: that one is table-driven and its table
    carries no order-33 entry, and the point of this generator is the specific 33-bit
    polynomial. `seed` is the initial register state, hence the PHASE; state 0 is the dead state
    and falls back to 1.

    SEEDS ARE PHASES, NOT DRAWS, and on a 33-bit maximal-length sequence that bites: state 2 is
    state 1 advanced one step, so `seed=1` and `seed=2` are the SAME sequence one bit apart, and
    two lanes seeded that way are almost perfectly correlated. Measured: the transmit waveforms
    of two pairs seeded 1 and 2 correlate at 0.58 where two independent streams correlate at
    ~0.02. The two ends of a real link are not two seeds of one scrambler -- they run the two
    DIFFERENT polynomials, which is what `role` is for. Use `role` for the two ends and keep
    `seed` for choosing a phase within one end."""
    taps = SCRAMBLER_TAPS[role]
    mask = (1 << max(taps)) - 1
    st = int(seed) & mask or 1
    for _ in range(int(warmup)):                # a window on a register already running
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


# --------------------------------------------------------------------------- 8B1Q4 proper
#
# The convolutional encoder. Clause 40.3.1.3's rate-8/9 code runs an 8-state (three delay
# element) convolutional encoder whose output bit joins two data bits to make the 3-bit 4D
# subset index; the other six data bits pick the point inside the subset.
#
# JUDGEMENT, stated: the GENERATOR below is not the clause's. Figure 40-8's encoder diagram is
# not in hand, so the recurrence used is a three-stage shift register with the output taken as
# the parity of all three stages. What the choice of generator does NOT change -- and what
# everything downstream depends on -- is the structural consequence: one of the three subset bits
# is a function of the encoder's HISTORY, so the subset sequence is not independent from symbol
# to symbol, and a decoder has a trellis to run. A different generator moves which subsets follow
# which; it cannot make the sequence memoryless.
#
# WHAT WOULD FALSIFY IT: Figure 40-8. The fix is the two lines marked below and nothing else --
# no caller, and no property in the tests except the exact transition pattern, sees the generator.
_ENC_STAGES = 3
_ENC_MASK = (1 << _ENC_STAGES) - 1


def _subset_stream(octets):
    """(subset index, point index, encoder state) per octet, as three int arrays.

    Six bits of the octet index the point; two bits plus the convolutional bit index the subset.
    The state trajectory comes back too, because the trellis structure is invisible once the
    symbols are on the wire and a caller conditioning on it should not have to re-derive it."""
    o = np.asarray(octets, dtype=np.int64)
    n = len(o)
    hi = ((o >> 6) & 0b11).astype(np.int64)     # the two data bits that go to the subset index
    pt = (o & 0b111111).astype(np.int64)        # the six that pick the point
    sub = np.empty(n, np.int64)
    st = np.empty(n, np.int64)
    s = 0
    for i in range(n):
        st[i] = s
        h = int(hi[i])
        p = (s ^ (s >> 1) ^ (s >> 2)) & 1       # <- generator: parity of the three stages
        sub[i] = (h << 1) | p
        s = ((s << 1) | (h & 1)) & _ENC_MASK    # <- generator: shift in the low data bit
    return sub, pt, st


def _octets(bits):
    """0/1 bits -> octets, MSB first, whole octets only."""
    b = np.asarray(bits, dtype=np.int64).ravel()
    b = b[: (len(b) // 8) * 8].reshape(-1, 8)
    return (b * (1 << np.arange(7, -1, -1))).sum(axis=1)


def subset_indices(bits):
    """The 4D subset index (0..7) chosen for each octet. Exposed because the trellis structure
    is a property a caller may want to check or condition on, and it is invisible in the
    symbols once they are on the wire."""
    return _subset_stream(_octets(bits))[0]


def encode_8b1q4(bits):
    """8B1Q4: one octet -> one quartet of quinary symbols, one symbol per wire pair.

    Returns an int array of shape (n_octets, 4) with values in {-2..+2}. Column `k` is wire
    pair `k`'s symbol stream at 125 MBd; all four columns are transmitted simultaneously, which
    is what makes this a 4D code rather than four independent PAM5 lanes.

    `bits` is a 0/1 array -- normally `side_stream_bits`, which is what makes the marginal level
    distribution the constellation's own (see `quartet_constellation`)."""
    o = _octets(bits)
    sub, pt, _ = _subset_stream(o)
    return quartet_constellation()[sub, pt]


def partial_response(symbols, taps=PARTIAL_RESPONSE_TAPS):
    """The transmit partial-response filter, applied at SYMBOL rate: y[n] = a*s[n] + b*s[n-1].

    Deliberately at symbol rate, before any edge shaping -- the filter is a digital one inside
    the transmitter operating on the symbol stream, not an analogue response of the line, so
    applying it to a shaped waveform would smear edges that the real transmitter does not
    smear. Its unit DC gain (the taps sum to 1) means it moves no mean level.

    The first symbol has no predecessor. It is given ITSELF as history, which is the choice that
    keeps the record's head on the level lattice; the alternative (zero history) puts the first
    sample at 3/4 of its level, a transient that is an artifact of where the record starts and
    not of the link. Use `physics`' lead-in if the head has to be right for a different reason."""
    s = np.asarray(symbols, float)
    a, b = float(taps[0]), float(taps[1])
    prev = np.concatenate([s[:1], s[:-1]])
    return a * s + b * prev


# --------------------------------------------------------------- the standard's test patterns
#
# Clause 40.6.1.1.2 defines four transmitter test modes, and they are the patterns every
# transmitter electrical test in 40.6.1.2 is measured on. Modes 1 and 2 are written out
# symbol-for-symbol in the clause, so they are exact here. Mode 4's LFSR is stated; its
# bits-to-symbol map is not reproduced here -- see the PRELIMINARY note at the top.
_TEST_MODE_4_TAPS = (11, 9)     # gs1 = 1 + x^9 + x^11, updated Scr[0] <- Scr[10] ^ Scr[8]


def test_mode_symbols(mode, n_symbols=None):
    """The quinary symbol sequence of transmitter test mode `mode` (1, 2, 3 or 4).

    mode 1  2048 symbols, exactly as the clause writes them: {+2 then 127 zeros}, {-2 then 127
            zeros}, {+1 then 127 zeros}, {-1 then 127 zeros}, {128 x +2, 128 x -2, 128 x +2,
            128 x -2}, {1024 zeros}. The four ISOLATED pulses are the measurement points A, B,
            C and D of 40.6.1.2.1, which is what makes this a known-answer pattern: the peak
            differential output and the level accuracy are read straight off them, and the
            1024-zero tail is where 40.6.1.2.2 measures droop.
    mode 2  {+2, -2} repeating -- the maximum-slew pattern, MASTER timing.
    mode 3  the same symbol sequence as mode 2; the modes differ in the TIMING source (SLAVE
            rather than MASTER), which is a clock property and not a symbol property, so the
            sequence returned is identical.
    mode 4  the 11-bit LFSR gs1 = 1 + x^9 + x^11, period 2047 -- which is why the distortion
            test captures 2047 consecutive symbols. PRELIMINARY: the map from the three derived
            bit sequences to a symbol is the clause's table and is not reproduced here, so the
            map used is a stated stand-in (three bits -> one of the five levels, the eighth
            code folded onto 0). The LFSR, its period and the level ALPHABET are right; the
            particular permutation of levels is not claimed to be.

    `n_symbols` repeats/truncates the sequence to that length; modes 2, 3 and 4 need it (they
    are defined as continuous transmission), mode 1 defaults to its own 2048.
    """
    mode = int(mode)
    if mode == 1:
        out = []
        for lead in (2, -2, 1, -1):
            out.append(np.concatenate([[lead], np.zeros(127, np.int64)]))
        for lvl in (2, -2, 2, -2):
            out.append(np.full(128, lvl, np.int64))
        out.append(np.zeros(1024, np.int64))
        seq = np.concatenate(out).astype(np.int64)
    elif mode in (2, 3):
        seq = np.array([2, -2], np.int64)
    elif mode == 4:
        want = 2047 if n_symbols is None else int(n_symbols)
        mask = (1 << 11) - 1
        st = 1
        bits = np.empty(max(want, 2047) + 2, np.int8)
        for i in range(len(bits)):
            bits[i] = st & 1
            st = ((st << 1) | (((st >> 10) & 1) ^ ((st >> 8) & 1))) & mask
        # PRELIMINARY map: three consecutive scrambler bits -> one quinary level. The clause
        # derives x0n/x1n/x2n from taps of the same register and maps them through a table;
        # this stands in for that table and preserves the alphabet and the 2047 period.
        tri = np.stack([np.roll(bits, -k) for k in range(3)], axis=1)[:2047]
        idx = (tri * np.array([4, 2, 1])).sum(axis=1)
        lut = np.array([0, 1, -1, 2, -2, 1, -1, 0], np.int64)     # eighth code folded onto 0
        seq = lut[idx]
    else:
        raise ValueError(f"test mode must be 1, 2, 3 or 4, not {mode!r}")
    if n_symbols is None:
        return seq
    n = int(n_symbols)
    return np.resize(seq, n) if n != len(seq) else seq


# ------------------------------------------------- one pair's waveform, and the coupling budget

def pam5_symbols(pair=0, n_ui=512, seed=1, role="master", partial_response=True,
                 pattern="8b1q4"):
    """One wire pair's transmitted SYMBOL LEVELS, one per 125 MBd symbol period, normalized so
    the +/-2 symbol sits at +/-1.

    `pair` selects which of the four coordinates of the SAME octet stream this is. That is the
    point: pairs 0..3 of one transmitter are four views of one 4D code, so they are correlated
    exactly as a real four-pair transmitter's lanes are, and a near-end aggressor is NOT an
    independent random stream. Give the far end a different `seed` (or the other `role`) and its
    four lanes are a different code word, which is what the two ends of a real link have.

    `partial_response` applies the transmit shaping (`partial_response`) at symbol rate, so the
    levels are the 17 a probe at the MDI sees rather than the 5 the coder emitted. Turn it off to
    look at the coder.

    Levels, not a waveform: edge shaping, rise time and source jitter are `physics.from_symbols`'
    job and this module holds no filter of the line. `physics.pam(5, pattern='8b1q4')` is the
    one-call path that does both.
    """
    n_ui = int(n_ui)
    if pattern == "8b1q4":
        # one octet per symbol period, plus one so the last quartet is complete
        bits = side_stream_bits(8 * (n_ui + 1), seed=seed, role=role)
        sym = encode_8b1q4(bits)[:n_ui, int(pair) % PAIRS]
    elif pattern.startswith("test_mode_"):
        # a test mode is the SAME sequence on all four pairs (Clause 40.6.1.1.2 says "on all four
        # transmitters"), so `pair` does not select anything here and is accepted and ignored
        sym = test_mode_symbols(int(pattern.rsplit("_", 1)[1]), n_symbols=n_ui)
    else:
        raise ValueError(f"unknown quinary pattern {pattern!r}; use one of {PATTERNS}")
    v = quinary_volts(sym, v_peak=1.0)
    return _partial_response(v) if partial_response else v


# `partial_response` is also the name of `pam5_symbols`' flag, so that function needs another way
# to reach the filter. One implementation, two names -- the alias is not a second code path.
_partial_response = partial_response

# Every quinary pattern this module can generate, for a caller (and a composer) to validate against.
PATTERNS = ("8b1q4", "test_mode_1", "test_mode_2", "test_mode_3", "test_mode_4")


# The coupling budget, in dB of loss, for the four terms a single-pair probe sums.
#
# PRELIMINARY, ALL FOUR NUMBERS. Clause 40.7's link-segment limits (return loss, pair-to-pair
# NEXT loss, equal-level FEXT loss) are the right source and the primary text is not in hand.
# The values below are the Category 5 channel figures those limits are built on, extrapolated
# from their 100 MHz anchors to the 62.5 MHz Nyquist of a 125 MBd link by the conventional
# slopes (-15*log10(f/100) for NEXT, -20*log10(f/100) for FEXT):
#
#   echo_db 16   the near-end reflection of own transmit. Return loss >= 15 dB from 1 to 40 MHz
#                degrading above it; 16 dB is that limit with nothing spare, i.e. a worst-case
#                conforming channel rather than a typical one.
#   next_db 30   NEXT loss >= 27.1 dB at 100 MHz -> 30.2 dB at 62.5 MHz. Near-end: the
#                aggressor has traversed no cable, so this is the largest crosstalk term.
#   fext_db 21   ELFEXT >= 17 dB at 100 MHz -> 21.1 dB at 62.5 MHz.
#
# These are LOSS figures for ONE disturber. A four-pair link has three of each, and
# `single_pair_observation` applies the budget per aggressor, so the total is about 4.8 dB worse
# than the per-pair number -- which is the multiple-disturber case the cabling specs also state
# separately. WHAT WOULD FALSIFY IT: Clause 40.7's tables, which would replace these three
# scalars with frequency-dependent limits and turn the flat couplings into shaped ones.
# The dB -> linear conversion is `physics.db_to_coupling` (amplitude convention).
CLAUSE_40_BUDGET = {"echo_db": 16.0, "next_db": 30.0, "fext_db": 21.0}
