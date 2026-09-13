"""4D-PAM5 / 8B1Q4, and four pairs' aggression summed into ONE observed channel.

WHY THIS FILE EXISTS. A probe on one pair of a live four-pair link does not see the wanted
signal. It sees a SUM: the far-end transmitter of its own pair, its own transmitter leaking
back through the hybrid's finite isolation, near-end crosstalk from the three local
transmitters, and far-end crosstalk from the three remote ones. Storing that as a single
channel is not an approximation of the measurement -- it IS the measurement. What a single
stored channel loses is the ability to CORRELATE an aggressor with the victim, and a model
trained on a single-pair capture is in the probe's position anyway.

Six claims, and the last three are the ones that catch a wrong implementation:

  * THE SYMBOL STATISTICS ARE NOT UNIFORM. A naive PAM5 model draws five levels with
    probability 1/5 each. 8B1Q4's 512-point 4D constellation gives {21, 28, 30, 28, 21}/128
    -- the outer levels are 18 % RARER than uniform and the zero level 17 % more common.
    Pinned exactly, because it is a counting fact about the constellation, not a measurement.
  * THE LEVEL SPACING IS UNIFORM AND THE SPACING IS THE STANDARD'S. Clause 40.6.1.2.1 fixes
    the +/-1 symbol at half the +/-2 symbol to within 2 %, so the five levels really are
    evenly spaced -- that half of the naive model is right, and the test says which half.
  * PARTIAL RESPONSE IS 3/4 + 1/4 z^-1, NOT (1+D)/2. Those are distinguishable by counting
    observed levels: 17 in 1/8 V steps for the first, 9 in 1/4 V steps for the second. The
    count is the calibration -- it is a known answer, so it is measured before anything
    harder is trusted.
  * AN ECHO IS NOT A REFLECTION. `multi_reflection` is a functional of its input alone, so a
    zero input gives exactly zero out. A hybrid echo with zero wanted signal is NOT zero: it
    is a scaled copy of a DIFFERENT stream. That is why `reflect` cannot express it.
  * CROSSTALK COUPLES THE AGGRESSOR'S DERIVATIVE, NOT ITS LEVEL. Correlation ~1 against
    d/dt(aggressor) and ~0 against the aggressor itself. Calibrated on a tone first, where
    the answer is known in closed form (d/dt sin = cos), before being believed on data.
  * THE SUM REDUCES TO THE WANTED SIGNAL EXACTLY. Every coupling zero must give back the
    wanted samples BIT-FOR-BIT, and each term must superpose, so a coupling budget is
    auditable term by term rather than only in total.

Run:  PYTHONPATH=. python -m pytest tests/test_multipair_4d_pam5.py -q
"""
from __future__ import annotations

import itertools
import math

import numpy as np
import pytest

from wfmsynth import physics as P
from wfmsynth import quinary as Q
from wfmsynth.compose import Signal, _EXEC
from wfmsynth.grid import Grid
from wfmsynth.streams import Streams

# 125 MBd is the symbol rate Clause 40 states (IEEE 802.3 1.4.183, 40.6.1.2.6). 16 samples/UI
# is the test grid, not a claim about any instrument: it resolves the 1/8 V partial-response
# staircase and keeps the records small.
BAUD = Q.SYMBOL_RATE_BD
SPUI = 16
FS = BAUD * SPUI
N_UI = 512
GRID = Grid(fs=FS, baud=BAUD, n=N_UI * SPUI)


def _lane(pair, role="master", pattern="8b1q4"):
    """One pair's transmitted WAVEFORM: the quinary line code's symbol levels through the same
    edge-shaping pipeline every other carrier in this library uses. Shaped edges matter here --
    crosstalk couples a derivative, and an unshaped hold has an impulsive one.

    `role` is which END of the link this transmitter is, and it is how the far end is made
    independent of the near end: the two ends run different scrambler polynomials. Two SEEDS
    would not do it -- see `quinary.side_stream_bits`."""
    sym = Q.pam5_symbols(pair=pair, n_ui=N_UI, role=role, pattern=pattern)
    return P.from_symbols(sym, n=GRID.n, tr_frac=0.15, causal=True)


def _corr(a, b):
    a = np.asarray(a, float) - np.mean(a)
    b = np.asarray(b, float) - np.mean(b)
    d = np.std(a) * np.std(b) * len(a)
    return 0.0 if d == 0 else float(a @ b / d)


# ==================================================================================
# 1. THE AUDIT: what `crosstalk_matrix` actually does
# ==================================================================================
def test_crosstalk_matrix_generates_its_own_aggressors_and_cannot_be_given_streams():
    """The contract the caller needs stated plainly, as an executable fact.

    `crosstalk_matrix(x, grid, couplings, baud_offsets, seeds, kind, synchronous)` has NO
    parameter that takes a waveform. It MANUFACTURES one aggressor per coupling from
    `physics.nrz` at a seed and a baud offset. Two consequences:

      * the aggressor content is the op's choice, not the caller's -- which is why moving
        `seeds` alone moves the output;
      * every aggressor is NRZ, whatever the victim carries. On a 4D-PAM5 link that is the
        wrong alphabet, the wrong baud and the wrong spectrum.

    It DOES take four (it takes len(couplings)), so the count was never the problem.
    """
    import inspect
    params = set(inspect.signature(P.crosstalk_matrix).parameters)
    assert "aggressor" not in params and "aggressors" not in params, (
        f"crosstalk_matrix's signature is {sorted(params)}; if it grew a stream parameter "
        f"this test is the thing to update")

    x = P.nrz(n_ui=N_UI, n=GRID.n, seed=1, causal=True)
    four = [0.05, 0.04, 0.03, 0.02]
    a = P.crosstalk_matrix(x, GRID, four)
    b = P.crosstalk_matrix(x, GRID, four, seeds=[101, 102, 103, 104])
    assert a.shape == x.shape and len(four) == 4          # four aggressors is fine
    assert np.std(a - b) > 1e-6 * np.ptp(x), (
        "the aggressor content is generated from `seeds`; if the seed did not move the "
        "output the aggressors would not be the op's own")


def test_crosstalk_sum_is_the_op_that_takes_explicit_aggressor_streams():
    """The gap the audit found, closed: `crosstalk_sum` takes the streams themselves, so the
    aggressors can be the OTHER THREE PAIRS of the same link, carrying the same line code."""
    x = P.nrz(n_ui=N_UI, n=GRID.n, seed=1, causal=True)
    aggr = [P.nrz(n_ui=N_UI, n=GRID.n, seed=s, causal=True) for s in (11, 12, 13)]
    y = P.crosstalk_sum(x, aggr, [0.05, 0.05, 0.05])
    assert y.shape == x.shape
    # given streams, the output is a function of THOSE streams: change one and it moves
    aggr2 = list(aggr)
    aggr2[0] = P.nrz(n_ui=N_UI, n=GRID.n, seed=99, causal=True)
    assert np.std(y - P.crosstalk_sum(x, aggr2, [0.05, 0.05, 0.05])) > 1e-9


# ==================================================================================
# 2. 8B1Q4 / 4D-PAM5
# ==================================================================================
def test_eight_bits_become_four_quinary_symbols_one_per_pair():
    """8B1Q4 is its own contract: one octet in, one QUARTET out -- four quinary symbols, one
    per wire pair, per symbol period. The rate identity is then arithmetic:
    125 MBd * 2 bit/symbol * 4 pairs = 1000 Mb/s (IEEE 802.3 1.4.183 for the baud)."""
    bits = Q.side_stream_bits(8 * 300, seed=1)
    q = Q.encode_8b1q4(bits)
    assert q.shape == (300, Q.PAIRS) and q.dtype.kind == "i"
    assert set(np.unique(q)) <= set(Q.QUINARY_SYMBOLS)
    assert Q.SYMBOL_RATE_BD * Q.BITS_PER_SYMBOL * Q.PAIRS == 1_000_000_000


def test_the_five_levels_are_evenly_spaced_and_plus_one_is_half_of_plus_two():
    """Clause 40.6.1.2.1 measures the +/-2 symbol (points A, B) and the +/-1 symbol (points C,
    D) and requires |C| to sit within 2 % of 0.5*mean(|A|,|B|). That is a statement that the
    five levels are UNIFORMLY spaced -- so the naive model's SPACING is right even though its
    distribution is not, and the volt map is symbol/2 * V_peak with nothing else in it.

    Held to one ulp, not to 2 %: the 2 % is a transmitter's allowance, and the ideal map this
    library generates should not spend any of it."""
    v = Q.quinary_volts(np.array(Q.QUINARY_SYMBOLS), v_peak=1.0)
    assert np.allclose(v, [-1.0, -0.5, 0.0, 0.5, 1.0], atol=0.0, rtol=0.0)
    d = np.diff(v)
    assert np.ptp(d) <= np.spacing(0.5), f"level spacings differ by {np.ptp(d):.3e}"
    assert abs(abs(v[3]) - 0.5 * abs(v[4])) <= np.spacing(0.5)   # C vs 0.5*A
    lo, hi = Q.PEAK_DIFFERENTIAL_V
    assert lo < 1.0 and hi < 1.0 or True                          # the window, for the record
    assert (lo, hi) == (0.670, 0.820)


def test_every_quartet_lives_in_one_4d_subset_built_from_the_1d_subsets_X_and_Y():
    """The structure the trellis code is built on: the quinary alphabet is partitioned into
    X = {-1, +1} and Y = {-2, 0, +2}; a 4D point's TYPE is which of the two each coordinate
    came from; the sixteen types pair off with their complements into eight 4D subsets.

    Counting check, which is the whole claim: 2^k * 3^(4-k) summed over the sixteen types is
    5^4 = 625 quartets, the eight subsets partition them, and the code uses 8 * 64 = 512 --
    the 9 bits (8 data + 1 convolutional) that 8B1Q4 has to place."""
    assert Q.SUBSET_X == (-1, 1) and Q.SUBSET_Y == (-2, 0, 2)
    subsets = Q.quartet_subsets()
    assert len(subsets) == 8
    all_pts = [p for s in subsets for p in s]
    assert len(all_pts) == 625 and len(set(all_pts)) == 625     # a partition of the 4D space
    assert set(all_pts) == set(itertools.product(Q.QUINARY_SYMBOLS, repeat=4))

    used = Q.quartet_constellation()
    assert used.shape == (8, 64, 4)                             # 3 subset bits + 6 point bits
    for k, sub in enumerate(subsets):
        assert set(map(tuple, used[k])) <= set(sub), f"subset {k} leaked a point"
    flat = {tuple(p) for blk in used for p in blk}
    assert len(flat) == 512

    # each used point's coordinates really do come from X or Y according to its type
    for p in flat:
        assert all(v in Q.SUBSET_X or v in Q.SUBSET_Y for v in p)


def test_the_constellation_marginal_is_not_uniform_and_is_pinned_exactly():
    """THE DISCRIMINATING FACT. A naive PAM5 model gives each level probability 1/5. The
    512-point constellation gives {21, 28, 30, 28, 21}/128: the outer levels 18 % rarer, the
    zero level 17 % more common. Exact rationals, because this is a count over 8*64*4 = 2048
    coordinates, not an estimate.

    Two consequences that matter downstream, both checked: the mean level is EXACTLY zero (so
    the code injects no DC of its own -- an asymmetric constellation would, and AC-coupling
    would turn that into baseline wander that is not in the real link), and the mean square is
    1.75 rather than the uniform 2.0, i.e. 0.58 dB less transmit power for the same peak."""
    used = Q.quartet_constellation()
    counts = {s: int((used == s).sum()) for s in Q.QUINARY_SYMBOLS}
    assert counts == {-2: 336, -1: 448, 0: 480, 1: 448, 2: 336}
    total = 8 * 64 * 4
    assert sum(counts.values()) == total == 2048

    p = {s: counts[s] / total for s in counts}
    assert p == pytest.approx({-2: 21 / 128, -1: 28 / 128, 0: 30 / 128,
                              1: 28 / 128, 2: 21 / 128}, abs=0.0)
    assert max(abs(p[s] - 0.2) for s in p) > 0.03, "uniform would be indistinguishable"

    mean = sum(s * p[s] for s in p)
    ms = sum(s * s * p[s] for s in p)
    assert mean == 0.0, f"the constellation must be antipodally balanced, got mean {mean}"
    assert ms == 1.75
    assert 20 * math.log10(math.sqrt(2.0 / ms)) == pytest.approx(0.58, abs=0.01)


def test_the_coded_stream_reproduces_the_constellation_marginal_and_beats_uniform_on_a_test():
    """The constellation's marginal is only the STREAM's marginal if the encoder draws from it
    uniformly -- which is what the side-stream scrambler is for. Measured over all four pairs of
    a long record and put through a chi-square against both hypotheses: the coded one must
    survive and the naive uniform one must be crushed, or "correct symbol statistics" is a claim
    no test could fail.

    (The four lanes of one quartet are not independent draws, so the 4-dof threshold is being
    used as a sanity bound rather than as an exact test size. It does not need to be exact: the
    two hypotheses are three orders of magnitude apart.)"""
    q = Q.encode_8b1q4(Q.side_stream_bits(8 * 40000, seed=3))
    n = q.size
    obs = np.array([int((q == s).sum()) for s in Q.QUINARY_SYMBOLS], float)

    exp_code = np.array([21, 28, 30, 28, 21], float) / 128.0 * n
    chi_code = float(((obs - exp_code) ** 2 / exp_code).sum())
    exp_uni = np.full(5, n / 5.0)
    chi_uni = float(((obs - exp_uni) ** 2 / exp_uni).sum())

    assert chi_code < 13.28, f"stream does not match the constellation marginal (chi2 {chi_code:.1f})"
    assert chi_uni > 1000.0, f"uniform PAM5 is not being rejected (chi2 {chi_uni:.1f})"


def test_each_pair_alone_is_balanced_and_non_uniform_and_nearly_the_whole_code_s_marginal():
    """What a SINGLE-PAIR capture carries, which is the thing this library stores. Three claims,
    and the third one is a measured limitation rather than a success:

      * every pair's marginal is EXACTLY symmetric about zero -- the antipodal-pair selection
        rule guarantees it per subset, so no pair injects DC of its own;
      * every pair is far from uniform, so a per-pair capture still carries the code's signature;
      * the pairs are not quite interchangeable. The 64-of-N point selection is documented as a
        judgement and it is lexicographic, hence not symmetric under permuting the four
        coordinates, so a lane's marginal sits NEAR the whole code's rather than on it. Measured:
        the lanes stray by at most 0.016 from the code's marginal, while every lane is at least
        0.028 away from uniform -- so the residual is well inside the signal it would have to
        swamp to matter. Table 40-1 would remove it; it is pinned here so it cannot grow
        unnoticed."""
    con = Q.quartet_constellation()
    whole = np.array([int((con == s).sum()) for s in Q.QUINARY_SYMBOLS], float) / con.size
    for k in range(Q.PAIRS):
        lane = con[:, :, k]
        p = np.array([int((lane == s).sum()) for s in Q.QUINARY_SYMBOLS], float) / lane.size
        assert p[0] == p[4] and p[1] == p[3], f"pair {k} is not balanced: {p}"
        assert max(abs(p - 0.2)) > 0.025, f"pair {k} is indistinguishable from uniform: {p}"
        assert max(abs(p - whole)) < 0.02, f"pair {k} strays from the code's marginal: {p}"


def test_the_subset_index_depends_on_history_not_only_on_the_octet():
    """8B1Q4's ninth bit comes from a convolutional encoder, so the 4D subset a quartet is drawn
    from is a function of the encoder STATE as well as of the data. The consequence, stated
    without a statistical test because it is a structural fact: the SAME octet maps to more than
    one subset depending on where in the stream it lands. A memoryless coder -- which is what a
    "PAM5 with the right histogram" model is -- cannot do that, and that is the control built
    here.

    The other half is that the memory must not bias the code: the eight subsets have to come up
    equally often, or the constellation's marginal is not the stream's."""
    bits = Q.side_stream_bits(8 * 20000, seed=5)
    oct_ = Q._octets(bits)
    sub, _, st = Q._subset_stream(oct_)

    per_octet = {}
    for o, k in zip(oct_.tolist(), sub.tolist()):
        per_octet.setdefault(o, set()).add(k)
    ambiguous = sum(1 for v in per_octet.values() if len(v) > 1)
    assert ambiguous > 0.9 * len(per_octet), (
        f"only {ambiguous}/{len(per_octet)} octet values reach more than one subset; without "
        f"history in the subset index there is no trellis to decode")

    # the control: with no encoder state, the octet alone would fix the subset
    memoryless = {}
    for o in oct_.tolist():
        memoryless.setdefault(o, set()).add((o >> 5) & 0b111)
    assert all(len(v) == 1 for v in memoryless.values())

    # ... and the state must be a function of the history, i.e. it actually moves
    assert len(set(st.tolist())) == 1 << Q._ENC_STAGES

    hist = np.bincount(sub, minlength=8).astype(float)
    chi = float((((hist - hist.mean()) ** 2) / hist.mean()).sum())
    assert chi < 24.32, f"the subsets are not equally used (chi2 {chi:.1f} on 7 dof)"


def test_the_two_ends_of_the_link_are_independent_and_two_seeds_would_not_have_been():
    """THE TRAP, and it is the kind of thing that makes a whole dataset wrong quietly. The far-end
    transmitter has to be independent of the near-end one, or "wanted signal" and "echo" are the
    same signal and every correlation measured on the record is meaningless.

    Two seeds of one 33-bit LFSR do NOT give that: a seed is the register STATE, so state 2 is
    state 1 advanced one step and the two streams are one sequence a bit apart. The two ends of a
    real link run the two DIFFERENT scrambler polynomials instead, which is what `role` selects.
    Both cases are measured here, so the wrong one cannot come back unnoticed.

    The register is also warmed up before the first bit: loaded with a small state it emits its
    own transient (the single 1 takes 13 steps in one polynomial and 20 in the other to reach the
    second tap, so both roles open with a long run of zeros) and the two ends come out correlated
    through that shared head, not through anything physical."""
    ends = _corr(_lane(0, "master"), _lane(0, "slave"))
    indep = N_UI ** -0.5                      # what two independent streams give at this length
    assert abs(ends) < 1.5 * indep, f"master vs slave corr {ends:+.4f} against {indep:.4f}"

    seeded = _corr(*[P.from_symbols(Q.pam5_symbols(pair=0, n_ui=N_UI, seed=sd),
                                    n=GRID.n, tr_frac=0.15, causal=True) for sd in (1, 2)])
    assert abs(seeded) > 0.4, (
        f"two seeds correlate at {seeded:+.4f}; if this ever drops, `side_stream_bits` has "
        f"stopped treating the seed as a phase and the warning in its docstring is stale")

    raw = _corr(*[P.from_symbols(
        Q._partial_response(Q.quinary_volts(
            Q.encode_8b1q4(Q.side_stream_bits(8 * (N_UI + 1), role=r, warmup=0))[:N_UI, 0])),
        n=GRID.n, tr_frac=0.15, causal=True) for r in ("master", "slave")])
    assert abs(raw) > 3 * indep, (
        f"without the warm-up the two ends correlate at {raw:+.4f}; if that is no longer true "
        f"`SCRAMBLER_WARMUP` is no longer earning its place")


# ==================================================================================
# 3. PARTIAL RESPONSE -- calibrate the level count, then trust it
# ==================================================================================
def test_the_partial_response_filter_is_three_quarters_plus_one_quarter_not_one_plus_D_over_two():
    """CALIBRATION ON A KNOWN ANSWER. Clause 40's transmit shaping introduces one symbol of
    deliberate ISI, and the observable consequence is the number of distinct levels at the
    MDI. Enumerated in closed form over the 25 (current, previous) symbol pairs:

      F(z) = 3/4 + 1/4 z^-1   ->  17 levels, spaced 1/8 V
      F(z) = (1 + z^-1)/2     ->   9 levels, spaced 1/4 V

    So a wrong filter is not a subtle amplitude error, it is the wrong staircase, and this
    test can tell them apart before any of it is believed on a waveform. Both have unit DC
    gain, which is the other half of the check: the shaping must not move the mean level."""
    alpha = Q.quinary_volts(np.array(Q.QUINARY_SYMBOLS), v_peak=1.0)
    for taps, expect, step in ((Q.PARTIAL_RESPONSE_TAPS, 17, 0.125), ((0.5, 0.5), 9, 0.25)):
        lv = sorted({round(taps[0] * a + taps[1] * b, 12) for a in alpha for b in alpha})
        assert len(lv) == expect, f"taps {taps} give {len(lv)} levels, expected {expect}"
        assert np.allclose(np.diff(lv), step)
        assert sum(taps) == 1.0                                # unit DC gain

    assert Q.PARTIAL_RESPONSE_TAPS == (0.75, 0.25)


def test_partial_response_applied_to_a_symbol_stream_realizes_those_seventeen_levels():
    """The filter on the real stream, not on the enumeration: a long 8B1Q4 lane through
    `partial_response` must land on the 1/8 V lattice and use every one of its 17 rungs. The
    op is causal by construction (the previous symbol, never the next), so the first output
    symbol has no predecessor and the stream's own first sample is its own history."""
    lane = Q.encode_8b1q4(Q.side_stream_bits(8 * 4000, seed=7))[:, 0]
    v = Q.quinary_volts(lane, v_peak=1.0)
    y = Q.partial_response(v)
    assert y.shape == v.shape
    on_lattice = np.abs(y / 0.125 - np.round(y / 0.125)).max()
    assert on_lattice < 1e-12, f"off the 1/8 V lattice by {on_lattice:.3e}"
    assert len(np.unique(np.round(y * 8).astype(int))) == 17
    assert y[0] == pytest.approx(0.75 * v[0] + 0.25 * v[0])      # its own history at the head
    assert np.mean(y) == pytest.approx(np.mean(v), abs=2e-3)     # unit DC gain, realized


def test_the_standard_test_patterns_are_the_symbol_sequences_the_clause_states():
    """Real test patterns, not a stand-in. Clause 40.6.1.1.2 test mode 1 is a fully written-out
    2048-symbol sequence -- four isolated pulses (+2, -2, +1, -1) each followed by 127 zeros,
    then 128-symbol +2/-2 blocks, then 1024 zeros -- and it is the pattern the peak-voltage,
    droop and template tests (40.6.1.2.1 through 40.6.1.2.3) are measured on. Test mode 2 is
    {+2, -2} repeating. The isolated pulses are what make mode 1 a KNOWN-ANSWER case: the four
    measurement points A, B, C, D are literally the four pulse amplitudes."""
    tm1 = Q.test_mode_symbols(1)
    assert len(tm1) == 2048
    assert [tm1[0], tm1[128], tm1[256], tm1[384]] == [2, -2, 1, -1]      # points A, B, C, D
    assert (tm1[1:128] == 0).all() and (tm1[129:256] == 0).all()
    assert (tm1[512:640] == 2).all() and (tm1[640:768] == -2).all()
    assert (tm1[1024:] == 0).all() and len(tm1) - 1024 == 1024
    assert tm1.sum() == 0                                                # DC-free by design

    tm2 = Q.test_mode_symbols(2, n_symbols=64)
    assert list(np.unique(tm2)) == [-2, 2] and (tm2[::2] == 2).all()

    # mode 4 is the 11-bit LFSR gs1 = 1 + x^9 + x^11 (Clause 40.6.1.1.2): period 2^11 - 1,
    # which is why the distortion test captures 2047 consecutive symbols.
    tm4 = Q.test_mode_symbols(4, n_symbols=2047 * 2)
    assert set(np.unique(tm4)) <= set(Q.QUINARY_SYMBOLS)
    assert (tm4[:2047] == tm4[2047:]).all(), "test mode 4 must repeat with the LFSR's period"


# ==================================================================================
# 4. THE HYBRID ECHO -- a scaled copy of a DIFFERENT stream
# ==================================================================================
def test_a_hybrid_echo_is_not_a_reflection_of_the_received_signal():
    """The reason this needed a new op. `multi_reflection` is a functional of its input alone,
    so feeding it silence returns silence. The hybrid echo is the LOCAL transmitter leaking
    into the LOCAL receiver: with no wanted signal at all it is still there. No arrangement of
    `reflect`'s parameters can produce output from an input that is zero."""
    zero = np.zeros(GRID.n)
    own = P.nrz(n_ui=N_UI, n=GRID.n, seed=4, causal=True)

    assert np.abs(P.multi_reflection(zero, grid=GRID, td_ps=200.0,
                                     gamma_s=0.4, gamma_l=0.4)).max() == 0.0
    e = P.hybrid_echo(zero, own, grid=GRID, isolation_db=20.0, td_ps=200.0)
    assert np.std(e) > 0.0, "an echo of the OWN transmit cannot vanish with the wanted signal"


def test_the_echo_tracks_the_own_transmit_and_not_the_wanted_signal():
    """Which stream the echo is a copy of, measured rather than asserted: the added term
    correlates with the local transmit and not with what is being received. This is the
    property that makes the echo a DIFFERENT mechanism from ISI or a reflection, and the one
    an echo canceller exploits (it has the local transmit; it does not have the wanted)."""
    wanted = P.nrz(n_ui=N_UI, n=GRID.n, seed=1, causal=True)
    own = P.nrz(n_ui=N_UI, n=GRID.n, seed=4, causal=True)
    term = P.hybrid_echo(wanted, own, grid=GRID, isolation_db=14.0, td_ps=0.0,
                         f_hp_hz=1e3) - wanted
    assert abs(_corr(term, own)) > 0.9, f"echo vs own transmit: {_corr(term, own):.3f}"
    assert abs(_corr(term, wanted)) < 0.1, f"echo vs wanted: {_corr(term, wanted):.3f}"


def test_perfect_isolation_returns_the_wanted_samples_bit_for_bit():
    """The degenerate case has to be exact, not close. A dataset that sweeps isolation must
    have a genuine zero at one end, or the "no echo" record carries an echo nobody labelled."""
    wanted = P.nrz(n_ui=N_UI, n=GRID.n, seed=1, causal=True)
    own = P.nrz(n_ui=N_UI, n=GRID.n, seed=4, causal=True)
    y = P.hybrid_echo(wanted, own, grid=GRID, isolation_db=np.inf, td_ps=200.0)
    assert np.array_equal(y, wanted)


@pytest.mark.parametrize("db,other", [(20.0, 26.0), (14.0, 20.0), (26.0, 32.0)])
def test_the_isolation_knob_delivers_the_decibels_it_was_asked_for(db, other):
    """Calibrated, not nominal: 6 dB more isolation must halve the echo's amplitude. Measured
    on the added term, against the 20*log10 amplitude convention the parameter is documented
    with, so a caller reading a NEXT/return-loss figure out of a document gets that figure."""
    wanted = P.nrz(n_ui=N_UI, n=GRID.n, seed=1, causal=True)
    own = P.nrz(n_ui=N_UI, n=GRID.n, seed=4, causal=True)
    kw = dict(grid=GRID, td_ps=200.0)
    a = P.hybrid_echo(wanted, own, isolation_db=db, **kw) - wanted
    b = P.hybrid_echo(wanted, own, isolation_db=other, **kw) - wanted
    realized = 20 * np.log10(np.std(a) / np.std(b))
    assert realized == pytest.approx(other - db, abs=0.05), (
        f"asked for {other - db:g} dB of change, realized {realized:.3f} dB")


# ==================================================================================
# 5. CROSSTALK COUPLES THE DERIVATIVE -- calibrated on a tone first
# ==================================================================================
def test_fext_couples_the_derivative_calibrated_on_a_tone_then_measured_on_data():
    """The discriminating property of far-end crosstalk, and the measurement is calibrated
    before it is used.

    KNOWN ANSWER: for an aggressor sin(wt), d/dt is w*cos(wt), which is ORTHOGONAL to sin over
    a whole number of periods. So a correct FEXT term correlates ~1 with the cosine and ~0
    with the aggressor itself -- and if the instrument reported anything else on this case,
    nothing it said about a data aggressor would be worth reading.

    THEN THE REAL CASE: a 4D-PAM5 aggressor from a neighbouring pair. Correlation ~1 against
    its numerical derivative, ~0 against its level. A model that couples the LEVEL produces
    crosstalk that looks like a second data signal added in, which is the wrong artifact and
    the wrong spectrum (crosstalk rises with frequency; a level copy does not)."""
    n = GRID.n
    cyc = 8
    t = np.arange(n) / n
    tone = np.sin(2 * np.pi * cyc * t)
    cos = np.cos(2 * np.pi * cyc * t)
    victim = P.nrz(n_ui=N_UI, n=n, seed=1, causal=True)

    term = P.crosstalk(victim, tone, coupling=0.1, kind="fext") - victim
    assert abs(_corr(term, cos)) > 0.999, f"tone calibration failed: {_corr(term, cos):.4f}"
    assert abs(_corr(term, tone)) < 0.01, f"tone calibration failed: {_corr(term, tone):.4f}"

    lane = _lane(1, 'slave')
    term = P.crosstalk(victim, lane, coupling=0.1, kind="fext") - victim
    d = np.gradient(lane)
    assert abs(_corr(term, d)) > 0.99, f"FEXT vs d/dt(aggressor): {_corr(term, d):.4f}"
    assert abs(_corr(term, lane)) < 0.1, f"FEXT vs aggressor level: {_corr(term, lane):.4f}"


def test_summing_three_aggressors_couples_each_ones_derivative_and_nothing_else():
    """The matrix case: with three aggressors the summed term must be the sum of the three
    derivative couplings, so the budget is auditable per-aggressor. Checked by reconstructing
    the sum from its parts -- exactly, because superposition of a linear coupling is exact."""
    victim = P.nrz(n_ui=N_UI, n=GRID.n, seed=1, causal=True)
    aggr = [_lane(k, 'slave') for k in (1, 2, 3)]
    c = [0.05, 0.04, 0.03]
    y = P.crosstalk_sum(victim, aggr, c) - victim
    parts = sum(P.crosstalk_sum(victim, [a], [ci]) - victim for a, ci in zip(aggr, c))
    # the reconstruction differs only by the rounding of adding and subtracting the victim, which
    # is ~1e-16 of the VICTIM's scale; measured against the coupled term's own scale that is
    # still 1e-13 or better, and any real failure of superposition would be a fraction of one
    assert np.abs(y - parts).max() < 1e-12 * np.abs(y).max()


# ==================================================================================
# 6. ONE OBSERVED CHANNEL: wanted + echo + NEXT + FEXT
# ==================================================================================
def _link(**kw):
    """The four-pair link the probe sits on: the local transmitter's four lanes (own pair plus
    three near-end aggressors) and the remote transmitter's four (the wanted signal plus three
    far-end aggressors). The near-end aggressors are the SAME octet stream as the own transmit
    on different coordinates -- which is what a real four-pair PHY does, and it is the reason a
    stored single-pair channel loses correlation information rather than inventing it."""
    loc = [_lane(k, "master") for k in range(4)]
    rem = [_lane(k, "slave") for k in range(4)]
    return dict(wanted=rem[0], own_tx=loc[0], near=loc[1:], far=rem[1:], grid=GRID, **kw)


def test_every_coupling_at_zero_gives_back_the_wanted_signal_bit_for_bit():
    """THE REDUCTION. Four mechanisms are summed in; with all four budgets off, the output
    must be the wanted array itself, sample for sample. Not "within 1e-12" -- identical. This
    is what makes an ablation over the budget honest: the baseline record has nothing in it."""
    y = P.single_pair_observation(**_link(echo_db=None, next_db=None, fext_db=None))
    assert np.array_equal(y, _link()["wanted"])

    # and an explicitly infinite budget is the same zero, so a sweep can reach it by number
    z = P.single_pair_observation(**_link(echo_db=np.inf, next_db=np.inf, fext_db=np.inf))
    assert np.array_equal(z, _link()["wanted"])


def test_the_four_mechanisms_superpose_so_the_budget_is_auditable_term_by_term():
    """Each term contributes independently of the others, so "how much of this record is
    echo?" has an answer. Exact, because every one of the four is a linear functional of a
    stream that none of the others touches -- and if a future op made one of them depend on
    the running sum, this is the test that would notice."""
    full = P.single_pair_observation(**_link(echo_db=16.0, next_db=30.0, fext_db=21.0))
    wanted = _link()["wanted"]
    parts = (P.single_pair_observation(**_link(echo_db=16.0)) - wanted
             + P.single_pair_observation(**_link(next_db=30.0)) - wanted
             + P.single_pair_observation(**_link(fext_db=21.0)) - wanted)
    # exact superposition up to the rounding of adding and subtracting the wanted signal: the
    # residual is ~1e-16 of the WANTED scale, i.e. ~1e-13 of the aggression it is a statement
    # about. A term that depended on the running sum would miss by percent, not by 1e-13.
    assert np.abs((full - wanted) - parts).max() < 1e-12 * np.abs(full - wanted).max()


def test_the_observation_is_dominated_by_the_wanted_signal_but_is_not_the_wanted_signal():
    """A budget taken from Clause 40's loss figures must land in the regime the measurement is
    actually in: the wanted signal dominates (or the link would not close), and yet the
    observation differs from it by far more than a numerical tolerance (or the multi-pair
    emulation would be decoration). Both bounds, so neither direction can drift."""
    budget = Q.CLAUSE_40_BUDGET
    y = P.single_pair_observation(**_link(**budget))
    wanted = _link()["wanted"]
    rel = np.std(y - wanted) / np.std(wanted)
    assert 0.02 < rel < 0.5, f"aggression is {rel:.4f} of the wanted signal"
    assert _corr(y, wanted) > 0.85


def test_the_composer_renders_the_summed_observation_and_the_recipe_replays_it():
    """The op has to be reachable from a recipe, because that is how a dataset is built: the
    victim's own carrier comes from the chain and the seven other streams from the op's own
    parameters, so the whole four-pair scene round-trips as JSON."""
    s = (Signal(seed=5, grid=GRID)
         .carrier("pam5", n_ui=N_UI, pattern="8b1q4", causal=True)
         .multipair(echo_db=16.0, next_db=30.0, fext_db=21.0, echo_td_ps=200.0,
                    remote_seed=2, pattern="8b1q4"))
    y = s.waveform()
    base = Signal(seed=5, grid=GRID).carrier("pam5", n_ui=N_UI, pattern="8b1q4",
                                             causal=True).waveform()
    assert y.shape == base.shape and np.std(y - base) > 1e-6

    import json
    replayed = Signal.from_recipe(json.loads(s.to_json())).waveform()
    assert np.array_equal(y, replayed)


def test_the_composer_op_reduces_to_the_carrier_with_no_budget():
    """The same reduction, through the composer: a `multipair` op with nothing switched on is
    the identity, so adding the op to a recipe never silently changes a record."""
    base = Signal(seed=5, grid=GRID).carrier("pam5", n_ui=N_UI, pattern="8b1q4", causal=True)
    assert np.array_equal(base.multipair().waveform(), base.waveform())


def test_the_new_ops_are_classified_for_the_lead_in_and_swept_for_dead_knobs():
    """The two gates this repo keeps, checked here as well as in their own files, because an op
    that defaults into the wrong pile is silent: the lead-in classifier must know both new ops,
    and the knob sweep must carry a row for each."""
    from wfmsynth import compose as C
    for op in ("hybrid_echo", "multipair"):
        assert op in C._EXEC and op in C.OP_KIND
        assert (op in C._LEAD_LTI) + (op in C._LEAD_SKIP) + (op in C._LEAD_REJECT) \
            + (op in C._LEAD_ANALYTIC) == 1, f"{op} must be in exactly one lead-in class"
    from tests.test_composition import KNOBS
    swept = {o for o, _, _, _ in KNOBS}
    assert {"hybrid_echo", "multipair"} <= swept


def test_the_op_executes_standalone_through_the_registry():
    """The op-level path, not just the builder: `_EXEC` is what every audit in this repo drives,
    so the op has to work when handed a bare params dict."""
    x = _lane(0, 'slave')
    y = _EXEC["multipair"](x, {"op": "multipair", "echo_db": 16.0, "next_db": 30.0,
                               "fext_db": 21.0}, Streams(0), GRID, 0)
    assert y.shape == x.shape and np.std(y - x) > 1e-9
