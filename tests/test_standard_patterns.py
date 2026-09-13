"""The standard test patterns: the declared period is the real period, and a maximal claim is maximal.

WHY THIS FILE IS ONE PROPERTY REPEATED. A caller sizes a record from `Pattern.period` -- an
integer number of repetitions is what makes a capture foldable and what an analyser pattern-locks
to -- so a wrong period raises nothing and silently produces a record that cannot be folded. The
sweep below therefore asserts, for EVERY registered name:

  * the sequence repeats at the declared period, and
  * it repeats at NO PROPER DIVISOR of it (a period that is a multiple of the true one folds, so
    only the divisor direction catches a sequence that is secretly shorter -- a broken LFSR, a
    block with a repeat inside it), and
  * a period too long to materialise does NOT repeat inside a record-sized window.

THE INSTRUMENT IS CALIBRATED FIRST. `_minimal_period` is checked against closed-form answers -- a
hand-built block, the kernel's clock, a maximal-length PRBS -- and against a deliberately
NON-primitive polynomial, whose short cycle it has to report rather than the register width's
2^k-1. An instrument that cannot fail that case cannot be trusted on SSPRQ.

SSPRQ gets its own section because it is the pattern with a normative artefact behind it: IEEE
publishes the 65535 symbols as a machine-readable extract of IEEE Std 802.3, and the digest pinned
here was computed from that file. Every constant in the construction is checked against it at once.
"""
import hashlib
import json

import numpy as np
import pytest

import wfmsynth.patterns as PAT
import wfmsynth.physics as P
import wfmsynth.standard_patterns as SP
from wfmsynth.compose import Signal
from wfmsynth.grid import Grid
from wfmsynth.measure import pattern_period

# Names this file's module registers. Kept explicit rather than derived: the sweep's job is to
# fail when an entry is added without a period that survives it, and deriving the list from the
# registry would let a name slip in without also appearing here.
STANDARD_NAMES = ["ssprq", "prbs13q", "prbs31q", "kr_training", "kr_square16", "pcie_compliance",
                  "dp_prbs7", "dp_prbs31", "usb_cp1_nyquist", "usb_cp3_com", "usb_cp7_lowfreq50",
                  "usb_cp12_lfsr15"]

# The longest period this file will materialise three times over. Above it a pattern is checked for
# the opposite property: that it does NOT close inside a record.
MATERIALISE_CAP = 1 << 17


@pytest.fixture(autouse=True)
def _registry_sandbox():
    """Every test leaves the registry as it found it -- the unregister cases above would otherwise
    make the rest of the sweep pass for the wrong reason."""
    saved = dict(PAT.REGISTRY)
    yield
    PAT.REGISTRY.clear()
    PAT.REGISTRY.update(saved)


def _is_period(x, p):
    """Is `x` exactly p-periodic over its whole length? Exact equality on symbols -- these are
    transmitted levels, not measured ones."""
    a = np.asarray(x, float)
    return bool(p and p < len(a) and np.array_equal(a[p:], a[:-p]))


def _divisors(n):
    """The PROPER divisors of n. These are the only candidates worth testing for a sequence that is
    secretly shorter than it claims: if a sequence of declared period n has any smaller period q,
    then gcd-stepping makes some divisor of n a period too, so a divisor sweep is not a sample of
    the possibilities -- it is all of them."""
    return sorted({d for k in range(1, int(n ** 0.5) + 1) if n % k == 0
                   for d in (k, n // k)} - {n})


def _minimal_period(x, cap):
    """The smallest p in 1..cap such that `x` is p-periodic, or None. O(cap * len(x)), so `cap` is
    required rather than defaulted -- an accidental sweep to 65535 on a 131070-symbol pattern is
    minutes of comparisons for an answer the divisor sweep gets in fifteen."""
    for p in range(1, int(cap) + 1):
        if _is_period(x, p):
            return p
    return None


# --------------------------------------------------------------- calibration: known answers
def test_the_period_instrument_reports_a_hand_built_block():
    """A 5-symbol block repeated has period 5, and a block that is itself two repetitions has the
    shorter one. If the instrument misses that, every minimality assertion below is decoration."""
    assert _minimal_period(np.tile([1.0, -1, -1, 1, -1], 20), cap=50) == 5
    assert _minimal_period(np.tile([1.0, -1, 1, -1], 20), cap=50) == 2
    assert _minimal_period(P.clock_pattern(64), cap=32) == 2


def test_the_period_instrument_reports_a_maximal_prbs_and_a_broken_one():
    """CALIBRATION ON THE FAILURE CASE. x^7+x^6+1 is primitive, so its period is 127. The tap set
    (4,3,2,1) is NOT primitive -- x^4+x^3+x^2+x+1 has order 5 -- and a sequence generator that
    believed register width alone would claim 15. The instrument has to report 5, because that is
    exactly the defect (a wrong polynomial, a short cycle) the sweep is meant to catch."""
    assert _minimal_period(PAT.lfsr(127 * 3, taps=(7, 6), seed=1), cap=190) == 127
    assert _minimal_period(PAT.lfsr(15 * 4, taps=(4, 3, 2, 1), seed=1), cap=30) == 5


def test_the_period_instrument_agrees_with_the_autocorrelation_one_up_to_a_multiple():
    """Two instruments on one known sequence, and the DIFFERENCE between them is the reason this
    file folds rather than correlates.

    Every multiple of a period is also a period, so the autocorrelation of `measure.pattern_period`
    peaks equally at 8191, 16382, ... on PRBS13Q and its argmax picks whichever the normalization
    nudges highest -- here 16382, twice the answer. That is not a defect in it (it reports a lag a
    capture locks at, and 2p locks) but it disqualifies it from establishing a REGISTERED period,
    where a factor of two is a record sized twice as long as it needs to be and a divisor sweep that
    never runs. Hence: correlation to confirm a lock exists, exact folding to pin which lag is the
    smallest one."""
    q = SP.prbs13q(8191 * 3)
    assert _is_period(q, 8191) and not any(_is_period(q, d) for d in _divisors(8191))
    lag, peak, _ = pattern_period(q, max_lag=8191 * 2)
    assert peak > 0.99 and lag % 8191 == 0                       # a multiple, not necessarily p
    assert _minimal_period(q, cap=2) is None                      # and nothing shorter folds
    seg = np.where(P.prbs(31, 1 << 14) > 0, 1.0, -1.0)          # a segment of a 2^31-1 pattern
    assert pattern_period(seg)[1] < 0.5                          # nothing to lock to


# --------------------------------------------------------------- the sweep that matters
@pytest.mark.parametrize("name", STANDARD_NAMES)
def test_every_registered_pattern_repeats_with_the_period_it_declares(name):
    """The load-bearing assertion. Generate three declared periods; the second and third must be
    the first, exactly. A period longer than the cap is checked the other way round: it must NOT
    close inside a record-sized window, because a caller told "2147483647" will size a record that
    never folds and would be silently wrong if the sequence closed early."""
    entry = PAT.get(name)
    assert entry.period is not None, f"{name}: no declared period; a caller cannot size a record"
    assert entry.levels in (2, 4), f"{name}: levels must say which carrier the symbols belong on"
    if entry.period <= MATERIALISE_CAP:
        x = PAT.resolve(name, length=3 * entry.period)
        p = entry.period
        assert np.array_equal(x[:p], x[p:2 * p]), f"{name}: does not repeat at {p}"
        assert np.array_equal(x[:p], x[2 * p:3 * p]), f"{name}: second repetition differs"
    else:
        x = PAT.resolve(name, length=MATERIALISE_CAP)
        lag, peak, _ = pattern_period(x)
        assert peak < 0.5, (
            f"{name}: declares period {entry.period} but locks at lag {lag} (peak {peak:.3f}) "
            f"inside {MATERIALISE_CAP} symbols")


@pytest.mark.parametrize("name", STANDARD_NAMES)
def test_no_registered_pattern_is_secretly_shorter_than_it_declares(name):
    """The other direction, and the one that catches a broken generator: no PROPER DIVISOR of the
    declared period may also be a period. A sequence that folds at half its declared period still
    folds -- the record is valid and every per-position label in it is wrong."""
    entry = PAT.get(name)
    if entry.period is None or entry.period > MATERIALISE_CAP:
        return
    p = entry.period
    x = PAT.resolve(name, length=2 * p)
    short = [d for d in _divisors(p) if _is_period(x, d)]
    assert not short, f"{name}: declares {p}, also repeats every {short[0]}"


# --------------------------------------------------------------- maximal means maximal
@pytest.mark.parametrize("name,order", [("dp_prbs7", 7), ("usb_cp12_lfsr15", 15)])
def test_a_maximal_length_claim_is_maximal(name, order):
    """A period of 2^k-1 is only a maximal-length sequence if every non-zero k-bit window occurs
    exactly once per period. That is the definition, and it is a much sharper check than the
    period: a sequence can have period 2^k-1 without being an m-sequence at all (see SSPRQ below).
    """
    entry = PAT.get(name)
    assert entry.period == (1 << order) - 1
    bits = (PAT.resolve(name, length=entry.period) > 0).astype(int)
    wrapped = np.concatenate([bits, bits[:order - 1]])              # the cycle, so windows wrap
    win = np.lib.stride_tricks.sliding_window_view(wrapped, order)
    vals = win @ (1 << np.arange(order - 1, -1, -1))
    assert len(np.unique(vals)) == entry.period                     # every state once
    assert 0 not in vals                                            # never the dead state


def test_ssprq_has_a_maximal_sequences_period_and_is_not_one():
    """THE TRAP THIS TEST EXISTS FOR. SSPRQ's period is 2^16-1, which reads exactly like a PRBS16's
    and is not one: it is four repetitions of three PRBS31 sections, Gray coded, with two of the
    repetitions inverted. A consumer that treated the period as evidence of an m-sequence would try
    to recover a 16-bit LFSR state from a capture and never lock. So: period 65535, and the 16-bit
    window property that an m-sequence would satisfy must FAIL."""
    entry = PAT.get("ssprq")
    assert entry.period == (1 << 16) - 1
    sym = SP.ssprq(entry.period)
    lv = np.array([-1.0, -1 / 3, 1 / 3, 1.0])
    idx = np.argmin(np.abs(sym[:, None] - lv[None, :]), axis=1)      # levels -> 0..3
    bits = np.unpackbits(idx.astype(np.uint8).reshape(-1, 1), axis=1)[:, -2:].ravel()
    win = np.lib.stride_tricks.sliding_window_view(bits[:entry.period + 15], 16)
    vals = win @ (1 << np.arange(15, -1, -1))
    assert len(np.unique(vals)) < entry.period                      # not an m-sequence


# --------------------------------------------------------------- SSPRQ against the published file
#
# sha256 of one period's 65535 symbols as level indices 0..3, one uint8 each, in transmission
# order. Computed from the SSPRQ symbol sequence IEEE publishes as a machine-readable extract of
# IEEE Std 802.3 (`SSPRQ_sequence.csv` on the IEEE 802.3 machine-readable extracts page), NOT from
# this library's output -- which is the whole point: it is an external answer key, so a drift in the
# seeds, the section lengths, the Gray mapping, the trim or the inverted ranges fails here.
SSPRQ_SHA256 = "f17f5effb8e68863e5355258186456e1c3b3c48b5519c1a0c46dac855fae3582"


def _ssprq_indices(n=None):
    sym = SP.ssprq(SP.SSPRQ_PERIOD if n is None else n)
    lv = np.array([-1.0, -1 / 3, 1 / 3, 1.0])
    d = np.abs(sym[:, None] - lv[None, :])
    assert d.min(axis=1).max() < 1e-12, "SSPRQ emitted a level that is not one of the four"
    return np.argmin(d, axis=1).astype(np.uint8)


def test_ssprq_is_byte_identical_to_the_published_symbol_sequence():
    """The one assertion that makes the construction above more than a story."""
    got = hashlib.sha256(_ssprq_indices().tobytes()).hexdigest()
    assert got == SSPRQ_SHA256, (
        f"SSPRQ does not match the published sequence (sha256 {got[:16]}... != "
        f"{SSPRQ_SHA256[:16]}...). One of the seeds, the section lengths, the Gray mapping, the "
        f"trim of the second pair of repetitions, or the inverted symbol ranges is wrong")


def test_each_ssprq_section_is_a_prbs31_run_with_the_output_inverter():
    """The sections, checked as sequences rather than as constants: each satisfies the PRBS31
    recursion of G(x) = 1 + x^28 + x^31 with Figure 49-9's output inverter,

        out[i] = NOT (out[i-31] XOR out[i-28])      (the inverter cancels on the two taps)

    and the three of them are 32768 bits, which is the building block the pattern repeats."""
    secs = [SP._prbs31_section(s, n) for s, n in zip(SP.SSPRQ_SEEDS, SP.SSPRQ_SECTION_BITS)]
    assert sum(len(s) for s in secs) == 32768
    for seed, sec in zip(SP.SSPRQ_SEEDS, secs):
        b = sec.astype(int)
        pred = 1 ^ b[:-31] ^ b[3:-28]                 # out[i-31] xor out[i-28], complemented
        assert np.array_equal(pred, b[31:]), f"section seeded 0x{seed:08X} is not a PRBS31 run"


def test_the_ssprq_seed_is_not_the_first_bits_emitted():
    """The distinction the standard was corrected for, kept as a test because it is invisible in
    the output: the Table 120-2 hex presets the register, so the first 31 bits emitted are the
    sequence 31 steps later, not the seed. Reading the table the other way gives a sequence with
    the right statistics that no analyser locks to -- so the two must differ."""
    sec = SP._prbs31_section(SP.SSPRQ_SEEDS[0], 64)
    as_first_bits = np.array([(SP.SSPRQ_SEEDS[0] >> (30 - j)) & 1 for j in range(31)])
    assert not np.array_equal(sec[:31].astype(int), as_first_bits)
    assert not np.array_equal(sec[:31].astype(int), 1 - as_first_bits)


def test_ssprq_statistics_match_the_published_sequence():
    """Level probabilities and transition density, pinned from the published sequence. These are
    what a document quotes and what a caller compares a capture against; they also catch a wrong
    inverted range that happens to preserve the period."""
    st = PAT.symbol_stats(SP.ssprq(SP.SSPRQ_PERIOD))
    assert st["levels"] == 4
    got = sorted(st["level_probability"].values())
    assert got == pytest.approx([0.232166, 0.232166, 0.267826, 0.267842], abs=1e-5)
    assert st["transition_density"] == pytest.approx(0.699011, abs=1e-5)


def test_ssprq_phase_is_a_starting_position_in_the_pattern():
    """A capture starts wherever the link happened to be. `phase` must be a rotation of the same
    cycle -- not a different sequence -- and must be cyclic in the period."""
    base = SP.ssprq(SP.SSPRQ_PERIOD)
    rolled = SP.ssprq(SP.SSPRQ_PERIOD, phase=1234)
    assert np.array_equal(rolled, np.roll(base, -1234))
    assert np.array_equal(SP.ssprq(64, phase=SP.SSPRQ_PERIOD + 7), SP.ssprq(64, phase=7))


def test_ssprq_is_quaternary_and_says_so():
    """It is the pattern for a PAM4 link and is not applicable to a binary one. The registry
    entry's `levels` is how a consumer knows which carrier the symbols belong on, and crossing
    them is the error `physics.carrier_symbols` already refuses to coerce."""
    assert PAT.get("ssprq").levels == 4
    assert len(np.unique(SP.ssprq(4096))) == 4
    with pytest.raises(ValueError, match="quaternary"):
        P.carrier_symbols("nrz", 64, pattern="prbs13q")          # the refusal SSPRQ inherits


# --------------------------------------------------------------- the NRZ answer
def test_the_kr_training_pattern_is_prbs11_and_two_zeros():
    """IEEE Std 802.3 Clause 72.6.10.2.6, term by term: 4094 bits of PRBS11 on the polynomial of
    Equation (72-1), then two zeros, 512 octets in all. The two zeros are the part worth asserting
    -- they break the recursion once per frame, so this is not simply PRBS11 tiled."""
    x = SP.kr_training(SP.KR_TRAINING_BITS, seed=7)
    assert len(x) == 4096
    assert np.array_equal(x[:4094], np.where(P.prbs(11, 4094, 7) > 0, 1.0, -1.0))
    assert np.array_equal(x[4094:], [-1.0, -1.0])
    assert P.PRBS_TAPS[11] == SP.KR_PRBS11_TAPS                 # 1 + x^9 + x^11
    b = (x > 0).astype(int)
    assert not np.array_equal(1 ^ b[:-11] ^ b[2:-9], b[11:])     # the zeros break the recursion


def test_the_kr_training_pattern_folds_only_because_the_seed_is_held():
    """The docstring's warning, as a test. The clause requires a random seed per training frame, so
    a real capture does not fold at 4096 -- holding the seed is a choice a dataset makes, and a
    caller who re-seeds must not be told the record is foldable."""
    n = SP.KR_TRAINING_BITS
    held = SP.kr_training(2 * n, seed=7)
    assert _is_period(held, n) and not any(_is_period(held, d) for d in _divisors(n))
    reseeded = np.concatenate([SP.kr_training(n, seed=7), SP.kr_training(n, seed=11)])
    assert not _is_period(reseeded, n)


def test_the_kr_square_wave_has_a_period_of_sixteen_unit_intervals():
    """IEEE Std 802.3 Clause 72.6.4: a square wave with a period of 16 UI, and Clause 72.7.1.11's
    run of at least eight consecutive ones. One transition per eight UI is the contrast case --
    essentially no pattern-dependent ISI -- which is what makes it useless as a stress pattern and
    useful as a reference."""
    x = PAT.resolve("kr_square16", length=64)
    assert np.array_equal(x[:16], [1.0] * 8 + [-1.0] * 8)
    assert np.count_nonzero(np.diff(x) != 0) == 2 * (len(x) // 16) - 1
    assert x.sum() == 0                                          # DC balanced by construction


# --------------------------------------------------------------- the other standards
def test_the_pcie_compliance_pattern_is_the_four_code_groups_with_the_documented_disparity():
    """PCI Express Base Specification Revision 4.0 Section 4.2.8 prints both the symbol sequence
    (K28.5, D21.5, K28.5, D10.2) and the ten-bit code groups with their running disparity, so this
    is a verbatim check. 40 bits and not 80: the disparity returns to negative at the end of the
    fourth group, so the sequence closes there."""
    x = PAT.resolve("pcie_compliance", length=80)
    bits = "".join("1" if v > 0 else "0" for v in x[:40])
    assert bits == "0011111010" "1010101010" "1100000101" "0101010101"
    assert PAT.get("pcie_compliance").period == 40
    assert x.sum() == 0                                          # disparity closes over a period


def test_the_displayport_prbs7_is_the_phase_the_document_prints():
    """The DisplayPort standard prints the 127 bits of its PRBS7 link-quality pattern, upper left
    transmitted first, so the PHASE is specified and not only the polynomial x^7 + x^6 + 1. A
    generator on the right polynomial and the wrong phase is a different sequence to an instrument
    that expects to see this one."""
    printed = ("0010000011000010100"
               "011110010001011001110101001"
               "111101000011100010010011011"
               "010110111101100011010010111"
               "011100110010101011111110000")
    x = PAT.resolve("dp_prbs7", length=127)
    assert "".join("1" if v > 0 else "0" for v in x) == printed
    assert len(printed) == 127 and printed.count("1") == 64       # maximal-length ones/zeros split


def test_the_displayport_and_ethernet_prbs31_are_the_same_polynomial():
    """Two documents, one polynomial (x^31 + x^28 + 1). The names are separate because the
    document that designates the sequence is the part worth citing; the samples must be identical,
    or one of the two entries has the polynomial wrong."""
    a = PAT.resolve("dp_prbs31", length=4096, seed=3)
    b = np.where(P.prbs(31, 4096, 3) > 0, 1.0, -1.0)
    assert np.array_equal(a, b)


def test_the_usb_nyquist_pattern_is_the_clock_and_the_com_pattern_alternates_disparity():
    """CP1 is D10.2, a disparity-neutral code group of alternating bits -- so it IS the clock
    pattern, at a period of 2 UI rather than the 10 UI of the code group, and a caller folding on
    10 would still fold. CP3 is K28.5, whose disparity is +-2, so consecutive COMs alternate
    between the two code groups and the period is 20 UI. Getting that one wrong is a fold on half a
    pattern."""
    assert np.array_equal(PAT.resolve("usb_cp1_nyquist", length=64), P.clock_pattern(64) * -1)
    assert PAT.get("usb_cp3_com").period == 20
    com = "".join("1" if v > 0 else "0" for v in PAT.resolve("usb_cp3_com", length=20))
    assert com == "0011111010" "1100000101"
    assert com[:10].count("1") - com[:10].count("0") == 2         # +2 disparity, then -2 back


def test_the_usb_lfsr15_is_the_polynomial_the_table_names():
    """CP12 is uncoded (not 128b/132b encoded), so the LFSR bits are the line sequence and the
    period is the register's 2^15-1 = 32767 UI with no encoding overhead on it."""
    entry = PAT.get("usb_cp12_lfsr15")
    assert entry.period == 32767 and entry.params["taps"] == [15, 14]
    assert np.array_equal(PAT.resolve("usb_cp12_lfsr15", length=999, seed=5),
                          PAT.lfsr(999, taps=(15, 14), seed=5))


# --------------------------------------------------------------- provenance is not optional
def test_a_record_built_from_a_standard_name_carries_the_period_and_the_citation():
    """What the caller actually gets: a recipe whose block says WHICH pattern (the name), HOW LONG
    (the resolved length), what to fold on (the period) and where the sequence comes from (the
    clause). A record that carries the samples and not the period is a record nobody can fold
    later, which is the whole reason the field is on the entry rather than in a caller's head."""
    grid = Grid(fs=256e9, baud=16e9, n=1 << 13)
    sig = Signal(seed=3, grid=grid).pattern("ssprq", length=grid.n // 16)
    recipe = json.loads(json.dumps(sig.recipe()))                 # through JSON, as an archive does
    block = recipe["ops"][0]["pattern"]
    assert block["name"] == "ssprq" and block["period"] == 65535
    assert block["levels"] == 4 and "120.5.11.2.3" in block["source"]
    assert block["params"] == {"length": 512}                     # no default restated
    assert np.array_equal(Signal.from_recipe(recipe).waveform(), sig.waveform())


@pytest.mark.parametrize("name", STANDARD_NAMES)
def test_every_standard_entry_cites_the_document_it_came_from(name):
    """The field that makes a name auditable. An entry in this layer without a designation and a
    clause is a sequence nobody can check against anything -- which is the state the whole
    mechanism/knowledge split exists to keep out of the kernel."""
    src = PAT.get(name).source
    assert src, f"{name}: no source"
    assert any(k in src for k in ("Clause", "Section", "Table", "Appendix")), \
        f"{name}: source {src!r} names no clause, section, table or appendix"


def test_a_standard_entry_round_trips_through_a_recipe_without_this_module():
    """A consumer who has the library and not this module still renders the record: the block names
    a generic mechanism and its resolved parameters. The bespoke ones (SSPRQ, the quaternary
    sequences, the training pattern) cannot -- they are code, not a polynomial -- so their blocks
    must at least NAME what is missing rather than render something else."""
    block = PAT.describe("pcie_compliance", length=80)
    assert block["generator"] == "block_repeat" and block["period"] == 40
    want = PAT.resolve("pcie_compliance", length=80)
    PAT.unregister("pcie_compliance")
    assert np.array_equal(PAT.replay(block), want)               # renders from the mechanism alone
    ssprq_block = PAT.describe("ssprq", length=128)
    PAT.unregister("ssprq")
    with pytest.raises(PAT.PatternUnavailable, match="ssprq"):
        PAT.replay(ssprq_block)
