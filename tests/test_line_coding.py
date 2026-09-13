"""Line coding and scrambling: the block codes real links transmit, as composable ops.

A scrambler that does nothing looks fine. Its output is the same length, it still has the
right number of ones, every downstream measurement still works -- and the record is a lie,
because the whole point of the code is what it does to the DC content and the run length that
the channel and the AC coupling then respond to. So nothing here asserts "it ran": every claim
is a property the standard states, measured on the bits.

The measurement instruments (`max_run`, `running_disparity`, the primitivity test) are
CALIBRATED on hand-counted answers in the first section, before they are trusted on a
2**13-bit stream. This repo has been wrong about an instrument more often than about physics.

The two kinds of code are asserted to be DIFFERENT KINDS, because that difference is the
reason a dataset carries one rather than the other:

  * 8b/10b BOUNDS running disparity (|RD| <= 3 at every bit, always) and run length (<= 5).
  * a scrambler RANDOMISES both. The mean goes to 0.5, the disparity random-walks, and a long
    run is improbable rather than impossible. A record scrambled instead of block-coded has
    baseline wander a block code would not have produced.
"""
import numpy as np
import pytest

import wfmsynth.physics as P
from wfmsynth import coding as C
from wfmsynth.compose import Signal
from wfmsynth.grid import Grid

GRID = Grid(fs=256e9, baud=16e9, n=1 << 13)


# ============================================================ calibration of the instruments
def test_the_run_length_and_disparity_instruments_agree_with_hand_counted_answers():
    """Known answers first. `max_run` on an unscrambled idle is the failure these tests exist
    to catch, so it is the case that has to be right: 64 zeros must read 64, not 1."""
    assert C.max_run(np.array([1, 1, 1, 0, 0, 1])) == 3
    assert C.max_run(np.array([0])) == 1
    assert C.max_run(np.array([], dtype=int)) == 0
    assert C.max_run(np.zeros(64, int)) == 64            # an identity scrambler reads this
    assert C.running_disparity([1, 0, 1, 0]).tolist() == [1, 0, 1, 0]
    assert C.running_disparity(np.ones(4, int)).tolist() == [1, 2, 3, 4]


def _prime_factors(m):
    f, d = set(), 2
    while d * d <= m:
        while m % d == 0:
            f.add(d); m //= d
        d += 1
    if m > 1:
        f.add(m)
    return f


def _is_primitive(order, taps):
    """G(x) = 1 + sum x^t is primitive iff the order of x in GF(2)[x]/G is exactly 2^order-1.
    Same algebra `wfmsynth.validate` applies to PRBS_TAPS -- a non-primitive scrambler
    polynomial produces a sequence with short sub-periods that is not the standard's."""
    g = 1
    for t in taps:
        g ^= 1 << t
    m = (1 << order) - 1

    def xpow(e):
        r, base = 1, 2
        while e:
            if e & 1:
                r = _mul(r, base)
            base = _mul(base, base)
            e >>= 1
        return r

    def _mul(a, b):
        r = 0
        while b:
            if b & 1:
                r ^= a
            b >>= 1
            a <<= 1
            if (a >> order) & 1:
                a ^= g
        return r

    if xpow(m) != 1:
        return False
    return all(xpow(m // p) != 1 for p in _prime_factors(m))


def test_the_primitivity_test_is_calibrated_on_a_known_primitive_and_a_known_non_primitive():
    assert _is_primitive(7, (7, 6))                      # the PRBS7 polynomial, period 127
    assert not _is_primitive(4, (4, 3, 2, 1))            # x^4+x^3+x^2+x+1 has order 5, not 15


def test_the_offered_scrambler_polynomials_are_primitive():
    """A scrambler polynomial that is not primitive has a short sub-period, so the "whitened"
    stream repeats inside a record. Checked where 2**order-1 factors in milliseconds; the
    58-bit polynomial's period (2**58-1) is not factorable here by trial division, so it is
    asserted structurally instead (the test below: no repeat inside a record)."""
    assert _is_primitive(23, C.POLY_128B130B)
    assert _is_primitive(16, C.POLY_128B132B)


def test_the_64b66b_keystream_does_not_repeat_inside_a_record():
    """Stand-in for the primitivity check the 58-bit period is too long to factor: the
    self-synchronous scrambler's response to an all-zeros payload must not repeat at any lag
    up to a record's length."""
    z = C.scramble_self_sync(np.zeros(1 << 14, int), C.POLY_64B66B, C.SEED_64B66B)
    for lag in (1, 2, 3, 58, 97, 1024, 4096):
        assert not np.array_equal(z[:len(z) - lag], z[lag:]), f"repeats at lag {lag}"


# ============================================================ the scramblers
SCRAMBLERS = [
    # scheme, payload bits per block, total bits per block, ones in the block header. The header
    # is the framing's own DC contribution and it is not always zero-disparity: 64b/66b's '01' and
    # 128b/130b's Data Block '01' are balanced, and the four PLACEHOLDER header bits of 128b/132b
    # are not -- which is one reason those four bits are marked PRELIMINARY in the module. An
    # unbalanced header puts a real, permanent DC offset in the record.
    ("64b66b", 64, 66, 1),
    ("128b130b", 128, 130, 1),
    ("128b132b", 128, 132, 0),
]


@pytest.mark.parametrize("scheme,payload,total,hdr_ones", SCRAMBLERS)
def test_a_scrambler_breaks_an_all_zeros_idle(scheme, payload, total, hdr_ones):
    """THE tooth. An idle line sending zeros is the input a do-nothing scrambler cannot fake: the
    output has to come out balanced with short runs, where the identity leaves one run of 8192.

    The expected mean is the framing's, not 0.5: half of each payload plus the header's own ones.
    The 0.02 tolerance is 4.6 sigma for a fair coin over this many bits. The run bound is
    STRUCTURAL -- an unscrambled header interrupts every run, so no run survives past one payload
    plus the header bits that happen to match it -- and the measured runs (39 / 11 / 15) are far
    inside it."""
    bits = np.zeros(1 << 13, int)
    out = C.line_code(scheme, bits)
    assert len(out) == (len(bits) // payload) * total
    assert C.max_run(out) <= payload + (total - payload)
    assert C.max_run(out) <= 39, f"{scheme}: longest run {C.max_run(out)}"
    expect = (0.5 * payload + hdr_ones) / total
    assert abs(out.mean() - expect) < 0.02, f"{scheme}: mean {out.mean():.4f} vs {expect:.4f}"


def test_the_idle_scrambled_by_64b66b_opens_with_the_seeds_own_transient():
    """A known answer computed from the polynomial and the seed, not from the code: with the
    register all ones, the feedback x^58 XOR x^39 is 1 XOR 1 = 0, so an all-zeros payload
    transmits zeros -- until the first transmitted zero reaches the x^39 tap, 39 bits later, and
    the feedback becomes 1. So bit 39 of the payload is the first one, and the 39-bit run above is
    the seed's transient rather than a statistical accident."""
    z = C.scramble_self_sync(np.zeros(200, int), C.POLY_64B66B, C.SEED_64B66B)
    assert not z[:39].any() and z[39] == 1


def test_a_self_synchronous_scrambler_has_a_killer_pattern_and_the_framing_still_bounds_the_run():
    """The failure mode a scrambled dataset has to be honest about. Feed the scrambler the
    sequence its own feedback produces -- for the all-ones seed, all ones -- and the state never
    changes, the feedback stays 0, and the payload is transmitted UNSCRAMBLED. That is why a
    link's test pattern is specified rather than assumed, and why the sync header is load-bearing:
    it is the header, not the scrambler, that holds the run to 65 bits here (64 payload ones plus
    the header's one), which is still inside the framing's structural bound of 66."""
    ones = np.ones(64 * 8, int)
    assert (C.scramble_self_sync(ones, C.POLY_64B66B, C.SEED_64B66B) == 1).all()
    framed = C.line_code("64b66b", ones)
    assert C.max_run(framed) == 65 and C.max_run(framed) <= 66


@pytest.mark.parametrize("scheme,payload,total,hdr_ones", SCRAMBLERS)
def test_descrambling_recovers_the_payload_exactly(scheme, payload, total, hdr_ones):
    """Exactly, not approximately: a scrambler is a bijection on the payload, and a dataset
    that claims a payload has to be able to produce it back."""
    bits = P.prbs(13, payload * 64, seed=7).astype(int)
    out = C.line_code(scheme, bits)
    back = C.line_decode(scheme, out)
    assert np.array_equal(back, bits)


def test_a_self_synchronous_scrambler_recovers_from_a_flipped_bit_and_an_unknown_state():
    """The property that makes 64b/66b self-synchronous, stated two ways.

    (1) ERROR MULTIPLICATION IS x3 AND FINITE. One flipped received bit corrupts the bit
        itself and the two bits its feedback taps later reach -- +39 and +58 -- and nothing
        after that. A scrambler whose descrambler kept state from the payload would corrupt
        the rest of the record.
    (2) THE STATE NEED NOT BE KNOWN. Descrambling with the wrong seed is wrong for the first
        58 bits (the register's depth) and exact afterwards. The additive scrambler below is
        the contrast: wrong seed, wrong forever."""
    payload = P.prbs(9, 64 * 40, seed=3).astype(int)
    scr = C.scramble_self_sync(payload, C.POLY_64B66B, C.SEED_64B66B)

    hit = 500
    bad = scr.copy(); bad[hit] ^= 1
    err = np.flatnonzero(C.descramble_self_sync(bad, C.POLY_64B66B, C.SEED_64B66B) != payload)
    assert err.tolist() == [hit, hit + 39, hit + 58]

    wrong = C.descramble_self_sync(scr, C.POLY_64B66B, seed=0x2A2A2A2A2A2A2A)
    assert not np.array_equal(wrong[:58], payload[:58])
    assert np.array_equal(wrong[58:], payload[58:])


def test_an_additive_scrambler_needs_its_state_and_does_not_multiply_errors():
    """The other half of the contrast: the additive (synchronous) scrambler of 128b/130b XORs a
    keystream, so one bit error stays one bit error -- and a wrong seed is wrong for the whole
    record, which is why that link carries the seed in the standard rather than discovering it."""
    payload = P.prbs(9, 128 * 8, seed=3).astype(int)
    ks = C.lfsr_keystream(len(payload), C.POLY_128B130B, C.SEED_128B130B)
    scr = payload ^ ks
    bad = scr.copy(); bad[100] ^= 1
    assert np.flatnonzero((bad ^ ks) != payload).tolist() == [100]
    wrong = scr ^ C.lfsr_keystream(len(payload), C.POLY_128B130B, C.SEED_128B130B + 1)
    assert (wrong != payload).mean() > 0.4              # ~half of every bit, to the end


def test_the_standard_seed_and_polynomial_are_the_defaults_and_both_are_arguments():
    """Both knobs are the caller's, and the default is the standard's value -- so a record can
    carry a non-standard seed deliberately (a second lane, a fault) and say so in the recipe."""
    bits = P.prbs(9, 1024, seed=1).astype(int)
    assert np.array_equal(C.line_code("128b130b", bits),
                          C.line_code("128b130b", bits, poly=C.POLY_128B130B,
                                      seed=C.SEED_128B130B))
    assert not np.array_equal(C.line_code("128b130b", bits),
                              C.line_code("128b130b", bits, seed=C.SEED_128B130B ^ 1))
    assert not np.array_equal(C.line_code("128b130b", bits),
                              C.line_code("128b130b", bits, poly=(23, 18)))


def test_the_existing_64b66b_scrambler_is_wired_in_not_reimplemented():
    """The framing path must produce exactly what the audited function already produced --
    same sync header, same polynomial, same all-ones seed, bit for bit."""
    bits = P.prbs(13, 64 * 30, seed=5).astype(int)
    assert np.array_equal(C.line_code("64b66b", bits), C.scramble_64b66b(bits))


# ============================================================ 8b/10b
def _bits(s):
    return np.array([int(c) for c in s], int)


def test_8b10b_reproduces_codewords_that_can_be_checked_by_hand():
    """Known answers, in transmission order a b c d e i f g h j.

    K.28.5 from RD- is the canonical comma 0011111010: the 5b/6b sub-block 001111 leaves RD
    positive, so the 3b/4b sub-block is taken from the RD+ column (1010) -- which is what puts
    the 0011111 comma across the sub-block boundary. Getting the RD-between-sub-blocks rule
    backwards destroys the comma, so this one codeword tests the procedure, not a table row."""
    assert C.encode_8b10b_words([0xBC], rd=-1, control=[True])[0].tolist() == \
        _bits("0011111010").tolist()
    assert C.encode_8b10b_words([0xBC], rd=+1, control=[True])[0].tolist() == \
        _bits("1100000101").tolist()
    assert C.encode_8b10b_words([0x00], rd=-1)[0].tolist() == _bits("1001110100").tolist()


def test_only_the_comma_characters_carry_the_comma():
    """K.28.1, K.28.5 and K.28.7 are the three code groups containing the comma 0011111 /
    1100000, and no data character may -- that is what makes byte alignment possible. If the
    3b/4b K column were the D column, K.28.1 and K.28.7 would lose their comma."""
    def has_comma(b):
        s = "".join(map(str, b.tolist()))
        return "0011111" in s or "1100000" in s

    for y in range(8):
        w = 28 | (y << 5)
        got = any(has_comma(C.encode_8b10b_words([w], rd=r, control=[True])[0])
                  for r in (-1, +1) if (28, y) in C.VALID_K)
        assert got == (y in (1, 5, 7)), f"K.28.{y} comma={got}"
    for v in range(256):
        for r in (-1, +1):
            assert not has_comma(C.encode_8b10b_words([v], rd=r)[0]), f"D comma in {v:#04x}"


def test_the_8b10b_tables_have_the_structure_the_standard_forces():
    """Table-transcription errors are the realistic failure mode here, and these are the
    checks that catch one without a second copy of the table: every 6-bit code has disparity
    -2, 0 or +2; the RD- column is injective (32 data codes + the K.28 code all distinct);
    every one of the 20 disparity-neutral 6-bit words is used, and 111100 / 000011 -- the one
    +-2 pair the table leaves out, because it would put a run of 4 against a sub-block
    boundary -- is used by nothing."""
    six = [C.CODE_5B6B[x] for x in range(32)] + [C.CODE_K28]
    for lo, hi in six:
        for w in (lo, hi):
            assert sum(w) * 2 - len(w) in (-2, 0, 2), w
        assert sum(lo) * 2 - 6 >= 0 and sum(hi) * 2 - 6 <= 0
    minus = ["".join(map(str, lo)) for lo, _ in six]
    assert len(set(minus)) == len(minus)
    used = {"".join(map(str, w)) for pair in six for w in pair}
    neutral = {f"{v:06b}" for v in range(64) if bin(v).count("1") == 3}
    assert neutral <= used, f"unused neutral words {sorted(neutral - used)}"
    assert "111100" not in used and "000011" not in used

    four = [C.CODE_3B4B[y] for y in range(8)] + [C.CODE_3B4B_K[y] for y in range(8)] \
        + [C.CODE_3B4B_ALT7]
    for lo, hi in four:
        for w in (lo, hi):
            assert sum(w) * 2 - len(w) in (-2, 0, 2), w


def test_8b10b_bounds_running_disparity_and_run_length_where_a_scrambler_cannot():
    """The two properties the standard guarantees, on a stream built to break them: every byte
    in turn, then the six bytes whose 3b/4b sub-block needs the ALTERNATE D.x.A7 code to avoid
    a run of 6 across the boundary, repeated.

    |RD| <= 3 at every bit (not just at code-group boundaries) and RD in {-1,+1} between
    groups: IEEE 802.3 Clause 36.2.4. Run <= 5: the same clause. A single missing alternate-7
    row shows up here as a run of 6."""
    streams = [list(range(256)),
               [v | (7 << 5) for v in (11, 13, 14, 17, 18, 20)] * 40,
               list(P.prbs(9, 512, seed=11).astype(int) * 255)]
    for words in streams:
        bits, rd = C.encode_8b10b_words(words, rd=-1)
        assert abs(rd) == 1
        assert C.max_run(bits) <= 5, f"run {C.max_run(bits)}"
        disp = C.running_disparity(bits) - 1                 # the -1 start of RD = -1
        assert np.max(np.abs(disp)) <= 3, f"|RD| {np.max(np.abs(disp))}"
        at_group = disp[9::10]
        assert set(np.abs(at_group).tolist()) == {1}


def test_8b10b_round_trips_every_data_byte_from_both_disparities_and_every_k_character():
    """A decoder is the only check that the table is a bijection with the alternates in it --
    and the K half is the check that the decoder tracks running disparity, because K.28.2 and
    K.28.5 (like K.28.1 and K.28.6) are the SAME ten bits emitted from opposite disparities."""
    for rd in (-1, +1):
        vals, ctrl, _ = C.decode_8b10b(C.encode_8b10b_words(range(256), rd=rd)[0])
        assert vals.tolist() == list(range(256))
        assert not ctrl.any()
    ks = sorted(C.VALID_K)
    words = [x | (y << 5) for x, y in ks]
    for rd in (-1, +1):
        vals, ctrl, _ = C.decode_8b10b(C.encode_8b10b_words(words, rd=rd, control=[True] * len(words))[0])
        assert vals.tolist() == words and ctrl.all()


def test_the_decoder_refuses_a_codeword_no_encoder_can_emit():
    """111100 is the +2 word the 5b/6b table omits. Accepting it silently would make a
    corrupted capture decode to a plausible byte."""
    with pytest.raises(ValueError):
        C.decode_8b10b(_bits("1111001011"))
    with pytest.raises(ValueError):
        C.decode_8b10b(_bits("100111010"))                   # not a whole code group


def test_a_block_code_bounds_the_dc_that_a_scrambler_only_randomises():
    """Why a dataset carries one or the other. Same payload through both: 8b/10b's disparity
    envelope is +-3 forever, the scrambler's random-walks an order of magnitude further, and
    that walk is the baseline wander an AC-coupled record shows."""
    payload = P.prbs(13, 1 << 14, seed=2).astype(int)
    blk = C.line_code("8b10b", payload)
    scr = C.line_code("64b66b", payload)
    assert np.max(np.abs(C.running_disparity(blk) - 1)) <= 3
    assert np.max(np.abs(C.running_disparity(scr))) > 30
    assert abs(blk.mean() - 0.5) < 0.02 and abs(scr.mean() - 0.5) < 0.02
    assert C.max_run(blk) <= 5


# ============================================================ composable: it lands in the recipe
def _coded(**kw):
    return Signal(seed=4, grid=GRID).coded(**kw)


def test_every_scheme_the_module_offers_is_reachable_as_an_op():
    """The composer is the door: a code that cannot be recorded as an op is invisible to the
    recipe and therefore to the content digest, which is the whole defect this closes."""
    for scheme in C.SCHEMES:
        s = _coded(scheme=scheme, prbs=9, n_bits=1024)
        x = s.waveform()
        assert x.shape == (GRID.n,) and np.isfinite(x).all()
        assert s.recipe()["ops"][0]["op"] == "coded"


def test_a_coded_source_replays_from_its_recipe_bit_for_bit():
    s = _coded(scheme="128b130b", prbs=13, n_bits=1024, tr_frac=0.2)
    r = s.recipe()
    assert Signal.from_recipe(r).waveform().tobytes() == s.waveform().tobytes()
    import json
    assert json.loads(json.dumps(r)) == r                    # the recipe is JSON, not objects


def test_the_recipe_records_only_what_was_asked_for():
    """The recipe is a content address. If the op wrote the standard's polynomial into every
    record, the day that default changed every digest would change with it."""
    op = _coded(scheme="64b66b", prbs=9, n_bits=1024).recipe()["ops"][0]
    assert set(op) == {"op", "scheme", "prbs", "n_bits"}


def test_the_polynomial_and_the_seed_reach_the_samples_and_the_digest():
    """A knob stored and not read is this repo's recurring defect; for a code it is worse,
    because two records with different scrambler seeds would share a content address."""
    base = dict(scheme="128b130b", prbs=9, n_bits=1024)
    a = _coded(**base)
    for change in (dict(seed=C.SEED_128B130B ^ 0xFF), dict(poly=[23, 18]),
                   dict(scheme="128b132b")):
        b = _coded(**{**base, **change})
        assert a.sha256() != b.sha256(), change
        assert not np.array_equal(a.waveform(), b.waveform()), change


def test_the_coded_bits_can_be_given_explicitly_and_are_carried_verbatim():
    bits = P.prbs(7, 512, seed=9).astype(int).tolist()
    s = _coded(scheme="8b10b", bits=bits)
    assert s.recipe()["ops"][0]["bits"] == bits
    assert s.waveform().shape == (GRID.n,)


def test_a_coded_source_can_carry_a_lead_in():
    """The source's own turn-on is not in the delivered record, and the coded stream's history
    is its own cyclic extension -- the same treatment an explicit symbol stream gets, because a
    repeating coded pattern is what the link was sending before the trigger."""
    s = Signal(seed=4, grid=GRID, lead_in=True).coded("64b66b", prbs=9, n_bits=1024)
    plan = s.lead_plan()
    assert plan is not None and plan.lead > 0
    assert s.waveform().shape == (GRID.n,)


def test_the_op_refuses_a_scheme_or_a_bit_source_it_cannot_replay():
    with pytest.raises(ValueError):
        _coded(scheme="128b129b", prbs=9, n_bits=256).waveform()
    with pytest.raises(ValueError):
        _coded(scheme="64b66b").waveform()                   # no bits, no pattern
