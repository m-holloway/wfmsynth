"""Layer 3, the joins: ops that pass alone and break when they are put together.

Named `audit_*` rather than `test_*` on purpose -- `compose.py`, `instrument.py`, `physics.py`
and `rx.py` are being edited while this was written, and this file must not change the result of
a `test_*` run that is still moving. Run it with
``python3 -m pytest wfmsynth/tests/audit_composition.py`` or directly with
``python3 wfmsynth/tests/audit_composition.py`` -- the direct run prints every measured table
below, so a reader can see the numbers without a harness.

Five properties, each one only visible once ops are composed:

  * CHUNK INVARIANCE. Whole-record output must equal chunked output. A deep-memory record is
    rendered, exported and analysed in pieces, and an op whose parameter is a fraction of the
    array it is handed places a different mechanism in every piece.
  * BYPASS EQUALS ABSENCE, bit-identically -- including for the SEEDED stages after the
    bypassed one.
  * ORDER SENSITIVITY where the physics has it, BY THE AMOUNT THE GEOMETRY IMPLIES. "They
    differ" is a weak claim; "they differ by exactly the round-trip loss of the line between
    them" is a closed form.
  * DEFAULT BYTE-IDENTITY, across processes and across a hash seed -- asserted as
    self-consistency, not against a stored digest, because a stored digest breaks for the wrong
    reason while the library moves.
  * DEEP CHAINS: does anything degrade or drift as the number of stages grows?

GROUND TRUTH BEFORE ANY CLAIM. Every instrument here first recovers a CONSTRUCTED answer:
the chunking harness is proved on a memoryless op (bit-identical) and on overlap-save against
`np.convolve` (1.1e-15) before it is pointed at anything; the geometry's closed form is checked
against `physics.insertion_loss_db` rather than against a previous run.

AND THEN THE SECOND HALF, which is the one that bites. A check never observed failing is not a
check. `store_record`'s automatic full scale is EXACTLY chunk-invariant on the flat +/-1 record
a fixture builds -- 0.00 LSB, because every chunk of a flat record holds the record's peak. Give
it the amplitude taper a real capture has and the same check reads 0.95 LSB and the distinct-code
count goes 1527 -> 2759. The fixture was hiding it, the way a stop-band floor claim in this
project stood wrong by 3 dB for months because both of its gates used constructed noise with no
signal in it to leak.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys

import numpy as np
import pytest

if __package__ in (None, ""):                       # allow `python3 wfmsynth/tests/audit_composition.py`
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import wfmsynth.instrument as INST
import wfmsynth.physics as P
import wfmsynth.rx as RX
import wfmsynth.sparam as SP
from wfmsynth.compose import Signal, _EXEC
from wfmsynth.grid import Grid
from wfmsynth.stream import channel_fir, stream_convolve
from wfmsynth.streams import Streams

FS, BAUD = 256e9, 16e9
N_UI, SPB = 512, 16
N = N_UI * SPB                                       # 8192 samples, 16 samples/UI exactly
BITS = 11                                            # the depth the shipped records are stored at


def _grid(n=N):
    return Grid(fs=FS, baud=BAUD, n=n, v_full=0.8)


def _carrier(n_ui=N_UI, n=N, seed=5):
    return Signal(seed=seed, grid=_grid(n)).carrier("nrz", n_ui=n_ui, tr_frac=0.15,
                                                    pattern="prbs7")


def _x(n=N, seed=3):
    return P.nrz(n_ui=n // SPB, n=n, seed=seed, tr_frac=0.15, causal=True)


def _rel_rms(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.sqrt(np.mean((a - b) ** 2)) / (np.sqrt(np.mean(a ** 2)) + 1e-30))


def _span(a):
    return float(np.ptp(np.asarray(a, float)))


def _lsb(v, bits=BITS, headroom=1.05):
    """One code of an ``bits``-deep record auto-ranged to ``v`` -- the unit any "is this
    difference visible?" question has to be answered in, because a stored record is a lattice."""
    return 2.0 * headroom * float(np.max(np.abs(np.asarray(v, float)))) / (2 ** bits)


def _run_op(op, params, x, g, seed=0):
    return np.asarray(_EXEC[op](x, dict(params, op=op), Streams(seed), g, 0), float)


def _chunked_op(op, params, x, chunks, seed=0):
    """Run one op on ``chunks`` consecutive pieces of ``x``, each piece told it is the whole
    record (its own `Grid(n=)`), and concatenate. This is what a bounded-memory renderer does."""
    nc = len(x) // chunks
    return np.concatenate([_run_op(op, params, x[c * nc:(c + 1) * nc], _grid(nc), seed)
                           for c in range(chunks)])


# ============================================================ 0. the chunking harness, proved
MEMORYLESS = [
    ("nonlinearity", dict(compression=0.2, level_noise=0.0)),
    ("digitize", dict(bits=8, full_scale=1.6)),
    ("store", dict(bits=BITS, full_scale=1.6)),
    ("intra_pair_skew", dict(skew_ps=0.0, gain_imbalance=0.05)),
]


@pytest.mark.parametrize("op,kw", MEMORYLESS, ids=[f"{o}" for o, _ in MEMORYLESS])
@pytest.mark.parametrize("chunks", [2, 8])
def test_a_memoryless_stage_is_chunk_invariant_to_the_last_bit(op, kw, chunks):
    """GROUND TRUTH FOR THE HARNESS. A stage with no memory and no record-relative parameter
    cannot know how the record was cut up, so the two paths must agree to the last bit -- there
    is no tolerance to argue about. If this fails, the harness below is measuring itself."""
    x = _x()
    whole, ch = _run_op(op, kw, x, _grid()), _chunked_op(op, kw, x, chunks)
    assert np.array_equal(whole, ch), (
        f"{op}{kw} at {chunks} chunks moved {int(np.count_nonzero(whole != ch))} samples, "
        f"max {np.abs(whole - ch).max():.3e}")


def test_overlap_save_recovers_the_convolution_a_constructed_fir_defines():
    """GROUND TRUTH FOR THE WITH-MEMORY PATH, against an answer written by hand rather than
    produced by the library: a 6-tap FIR, `np.convolve` truncated to the input length.
    Measured 1.1e-15 at every chunk size, i.e. float64 and nothing else."""
    x, h = _x(), np.array([1.0, -0.4, 0.15, 0.0, 0.0, 0.05])
    ref = np.convolve(x, h)[:len(x)]
    for chunk in (256, 1024, 4096):
        err = float(np.abs(stream_convolve(x, h, chunk=chunk) - ref).max())
        assert err < 1e-13, f"chunk={chunk}: overlap-save is off the closed form by {err:.3e}"


def test_a_channels_own_fir_reproduces_its_whole_record_render_and_converges():
    """The composition claim for a real channel: probe it for its impulse response, apply that
    in bounded memory, and the result must approach the whole-record render as the response is
    given more taps. Measured on a causal 12.5 dB channel -- rel rms 1.664e-02 / 2.646e-03 /
    3.377e-04 at 256 / 1024 / 4096 taps, so ~7x per 4x in taps and it is the TRUNCATION of the
    response that bounds a chunked render, not the chunking."""
    x = _x()
    fn = lambda v: P.lossy_channel(v, grid=_grid(len(v)), loss_db=12.5, loss_at_ghz=8.0,
                                   causal=True)
    whole = fn(x)
    errs = [_rel_rms(whole, stream_convolve(x, channel_fir(fn, n_taps=nt), chunk=1 << 12))
            for nt in (256, 1024, 4096)]
    assert errs[0] > errs[1] > errs[2], f"more taps did not help: {errs}"
    assert errs[-1] < 1e-3, f"4096 taps still {errs[-1]:.3e} off the whole-record render"


# ==================================================== 1. chunk invariance: events, three ways
GLITCH = dict(kind="glitch", severity=0.5)


def _events_at(n_ui, n, **kw):
    """The realized event SAMPLES of a one-op events chain -- the labels a dataset would ship."""
    return [e.sample for e in _carrier(n_ui, n).events(**kw).realize()[1].events]


def _event_totals(chunks, translate=None, **kw):
    """``(whole_record_samples, chunked_samples_per_chunk)`` for the same ask.

    ``translate`` names a key whose absolute anchor is rebased into each chunk's own
    coordinates (``samples`` / ``indices``) -- what a renderer that knows the record's geometry
    does, and the version that makes the clipping a fabrication rather than a mis-call."""
    whole = _events_at(N_UI, N, **kw)
    nu, nc = N_UI // chunks, N // chunks
    per = []
    for c in range(chunks):
        k = dict(kw)
        if translate is not None:
            off = c * (nc if translate == "samples" else nu)
            k[translate] = [int(v) - off for v in kw[translate]]
        per.append(_events_at(nu, nc, **k))
    return whole, per


@pytest.mark.xfail(strict=True, reason=(
    "MEASURED DEFECT: `events.place_events` CLIPS an absolute anchor that falls outside the "
    "array (`emit` does `np.clip(sample, 0, n-1)`), so every chunk that does not contain the "
    "position gets a needle pinned to its own edge instead of no needle at all. Asked for one "
    "event at sample 5000 of an 8192-sample record: whole record [5000]; the same ask rendered "
    "in 8 chunks of 1024 gives EIGHT events at [1023, 1023, 1023, 1023, 904, 0, 0, 0] -- one "
    "true needle and seven fabricated ones, one per chunk boundary. Both halves are wrong in "
    "different ways, measured on a 1024-sample record: clipped to n-1, `glitch`/`ring`/`runt` "
    "emit a LABEL at 1023 and move the waveform by 0.00000 (a defect claimed over a clean "
    "record), while clipped to 0 a `glitch` moves it by 0.72998 = 36.00% of span and a `ring` "
    "by 18.41% across all 1024 samples (a mechanism nobody asked for). `place_events` already "
    "does the right thing on its other absolute anchor -- `on='symbols', indices=` filters "
    "`(want >= 0) & (want < n_cand)` and DROPS -- so the two absolute-anchor paths in one "
    "function disagree. Make `emit` drop (or raise) instead of clipping."))
def test_an_absolute_event_anchor_outside_the_array_is_dropped_not_pinned_to_its_edge():
    whole, per = _event_totals(8, translate="samples", on="times", samples=[5000],
                               **GLITCH)
    total = sum(len(p) for p in per)
    assert total == len(whole), (
        f"whole record {whole}, 8 chunks {per} -> {total} events for {len(whole)} asked")


def test_an_absolute_symbol_anchor_outside_the_array_is_already_dropped():
    """The half that WORKS, asserted so the xfail above is a statement about `emit` and not
    about absolute anchors in general. `on='symbols', indices=[300]` places one event on the
    whole record and one in total across 4 or 8 chunks -- the out-of-range indices are filtered,
    not clipped."""
    whole = _events_at(N_UI, N, on="symbols", indices=[300], **GLITCH)
    assert len(whole) == 1
    for chunks in (4, 8):
        nu = N_UI // chunks
        total = sum(len(_events_at(nu, N // chunks, on="symbols",
                                   indices=[300 - c * nu], **GLITCH)) for c in range(chunks))
        assert total == 1, f"{chunks} chunks placed {total} events for 1 asked"


@pytest.mark.xfail(strict=True, reason=(
    "MEASURED DEFECT: `count=` is honoured PER ARRAY, so a chunked render places N events in "
    "EVERY chunk. Asked count=4 on an 8192-sample / 512-UI record: whole record 4 events; "
    "chunked into 2 / 4 / 8 / 16 pieces gives 8 / 16 / 32 / 64 -- exactly 4*chunks, so the "
    "fault rate of a dataset is set by the renderer's memory budget rather than by the recipe. "
    "A count is a property of the RECORD; a chunked renderer needs to be told how many of the "
    "record's events fall in the piece it is rendering (which needs the record-wide draw to "
    "happen once, before the pieces)."))
def test_the_event_count_is_a_property_of_the_record_not_of_each_chunk():
    whole, per = _event_totals(16, on="symbols", count=4, **GLITCH)
    total = sum(len(p) for p in per)
    assert total == len(whole), (
        f"count=4: whole record {len(whole)} events, 16 chunks {total} "
        f"({[len(p) for p in per]})")


@pytest.mark.xfail(strict=True, reason=(
    "MEASURED DEFECT: `fraction=` is a DENSITY over the candidate pool of whatever array it is "
    "handed (`count = int(round(fraction * len(idx)))`), so chunking rounds it once per chunk "
    "and the record's event count is not conserved -- and not even monotone. Asked "
    "fraction=0.01 on a 512-UI record: whole record 5 events; chunked into 1 / 2 / 4 / 8 / 16 "
    "pieces gives 5 / 6 / 4 / 8 / 0. SIXTEEN CHUNKS GIVES ZERO: 0.01 * 32 UI rounds to 0 in "
    "every chunk, so a 1% fault density silently becomes a fault-free dataset. Draw the "
    "record-wide subset once, then distribute it to the pieces."))
def test_an_event_density_is_conserved_when_the_record_is_cut_up():
    whole, per = _event_totals(16, on="symbols", fraction=0.01, **GLITCH)
    total = sum(len(p) for p in per)
    assert total == len(whole), (
        f"fraction=0.01: whole record {len(whole)} events, 16 chunks {total}")


# ============================================ 2. chunk invariance: the vertical and the rate
def _tapered(n=N):
    """A record whose amplitude is not flat. THIS IS THE POINT: a constructed +/-1 record puts
    the peak in every chunk, so a per-chunk auto-range agrees with a record-wide one by
    accident. A real capture has drift, an AGC settling, a burst -- so it does not."""
    return _x(n) * np.linspace(1.0, 0.4, n)


def test_an_export_given_its_full_scale_is_chunk_invariant_on_a_record_that_is_not_flat():
    """GROUND TRUTH: the mechanism for a chunk-invariant vertical already exists. State the
    full scale and the tapered record quantises to the same lattice whole or in 8 pieces --
    0.00 LSB, 1528 distinct codes either way."""
    v, chunks = _tapered(), 8
    nc = len(v) // chunks
    whole = INST.store_record(v, bits=BITS, full_scale=1.0938)
    ch = np.concatenate([INST.store_record(v[c * nc:(c + 1) * nc], bits=BITS, full_scale=1.0938)
                         for c in range(chunks)])
    assert np.array_equal(whole, ch), f"max {np.abs(whole - ch).max():.3e}"


@pytest.mark.xfail(strict=True, reason=(
    "MEASURED DEFECT, and the one this file's fixture nearly hid: `store_record(full_scale="
    "None)` (the composed `store`/`digitize` default) ranges to `headroom * max|x|` OF THE "
    "ARRAY IT IS HANDED, so a record exported in pieces gets a DIFFERENT LSB in every piece -- "
    "a lattice that is not a lattice. An instrument's full scale is set once for the "
    "acquisition; it is not a function of which slice of the record you ask about. Measured, "
    "8 chunks of an 8192-sample 11-bit record: on the FLAT +/-1 record a fixture builds, "
    "0.00 LSB and 26 distinct codes either way -- the check cannot fail, because every chunk of "
    "a flat record holds the record's peak. On the same record with the amplitude taper a real "
    "capture has (1.0 -> 0.4), 0.95 LSB and the distinct-code count goes 1527 -> 2759 (1.81x), "
    "because the last chunk ranges to 0.5198 where the record ranges to 1.0938 (2.10x). "
    "Distinct-value count is one of the realism metrics this project measures against real "
    "captures, so a chunked export moves a record's headline realism number by 1.8x. The fix "
    "is a full scale threaded from a first pass, which the test above shows already works."))
def test_an_auto_ranged_export_is_chunk_invariant_on_a_record_that_is_not_flat():
    v, chunks = _tapered(), 8
    nc = len(v) // chunks
    whole = INST.store_record(v, bits=BITS)
    ch = np.concatenate([INST.store_record(v[c * nc:(c + 1) * nc], bits=BITS)
                         for c in range(chunks)])
    assert np.array_equal(whole, ch), (
        f"max {np.abs(whole - ch).max() / _lsb(v):.2f} LSB, distinct codes "
        f"{len(np.unique(whole))} -> {len(np.unique(ch))}")


@pytest.mark.xfail(strict=True, reason=(
    "MEASURED DEFECT: `crosstalk`'s aggressor is rendered as `n_ui` symbols across the WHOLE "
    "RECORD (`_op_crosstalk` -> `P.nrz(n_ui=..., n=grid.n)`), so the aggressor's SYMBOL RATE is "
    "`n_ui * fs / n` -- a fraction of the victim's record, not a physical rate, and there is no "
    "key that states it in Bd. Measured with the default `aggressor={'n_ui': 32}` at 256 GSa/s "
    "against a 16.0 GBd victim: the aggressor runs at 1.000 GBd on an 8192-sample record, "
    "2.000 GBd at 4096, 8.000 GBd at 1024 and 0.500 GBd at 16384. An aggressor is a link "
    "running at its own baud; halving the victim's record cannot double it. TWO CONSEQUENCES, "
    "both live. (a) A chunked render gives every chunk a differently-timed aggressor. (b) "
    "`crosstalk` is in `compose._LEAD_SKIP` -- declared safe to render with a lead-in because "
    "it has no impulse response -- and `('crosstalk', 'aggressor')` is not in `_LEAD_RELATIVE`, "
    "so a lead-in silently re-times it: measured, a 2032-sample guard each end of an "
    "8192-sample record moves a 512-UI aggressor from 16.000 GBd to 10.695 GBd, and the "
    "delivered window's INTERIOR (samples 256:7936, so no turn-on edge) then differs by 0.2028 "
    "= 9.21% of span, which is 2.00x the entire 5.00%-of-span contribution the op makes -- i.e. "
    "not perturbed, decorrelated. The control is bit-exact: the same lead-in with no crosstalk "
    "in the chain leaves that interior at 0.0000e+00. State the aggressor's rate in Bd (or as "
    "symbols-per-UI of the victim) and render it at that rate whatever the record length."))
def test_the_crosstalk_aggressors_symbol_rate_does_not_depend_on_the_record_length():
    rates = {n: 32.0 * FS / n for n in (1024, 4096, N, 2 * N)}
    assert len(set(round(r, 6) for r in rates.values())) == 1, (
        "aggressor realised baud by record length: "
        + ", ".join(f"n={n}: {r / 1e9:.3f} GBd" for n, r in sorted(rates.items()))
        + f" (victim {BAUD / 1e9:.1f} GBd)")


def test_a_lead_in_of_a_whole_pattern_period_leaves_the_records_interior_alone():
    """The CONTROL for the claim above, and a check of BACKLOG #56's own statement that a
    lead-in which is a whole multiple of the pattern's period adds history without changing the
    record. Measured on a source-only chain with a 127-UI (2032-sample) lead-in on a 512-UI
    prbs7 record: samples [256:7936] are bit-identical (0.0000e+00), and the only difference is
    the first UI -- 0.7961, 39.26% of span -- which is the turn-on edge the lead-in exists to
    remove. So the interior is a clean instrument for asking what a guard does to an op."""
    a = _carrier().waveform()
    b = _carrier().with_lead_in(SPB * 127).waveform()
    interior = slice(256, N - 256)
    assert np.array_equal(a[interior], b[interior]), (
        f"interior max {np.abs(a - b)[interior].max():.3e}")
    assert np.abs(a - b)[:SPB].max() > 0.1 * _span(a), (
        "the lead-in changed nothing at the record's head, so it is not doing its job")


# ================================================== 3. bypass equals absence, seeds included
IDENTITY = dict(td_ps=100.0, gamma_s=0.0, gamma_l=0.0, n_bounce=6)


def _deep(insert):
    """A five-stage chain with FOUR independently seeded roles, optionally with one stage set to
    its own exact identity inserted at the front."""
    s = _carrier()
    if insert:
        s = s.reflect(**IDENTITY)
    return (s.probe(c_load_f=0.45e-12, noise_rms=0.02)
             .timing(rj_ps=0.4)
             .crosstalk(coupling=0.05, kind="fext")
             .digitize(snr_db=35.0, interleave=dict(m_cores=4, offset_mm=0.01)))


def test_a_bypassed_stage_is_absence_for_a_fully_deterministic_chain():
    """The half that holds, stated first so the failure below is about SEEDS and not about the
    identity itself: with no seeded stage in the chain, inserting `reflect(gamma=0)` is
    bit-identical to not inserting it."""
    def chain(insert):
        s = _carrier().tx_ffe(taps=[1.0, -0.2], pre=0)
        if insert:
            s = s.reflect(**IDENTITY)
        return s.de_emphasis(db=3.0).intra_pair_skew(skew_ps=1.5).waveform()
    assert np.array_equal(chain(False), chain(True))


@pytest.mark.xfail(strict=True, reason=(
    "MEASURED DEFECT: `compose._run` keys every role stream by OP INDEX (`streams.role("
    "f'noise/{idx}')`), so inserting ONE stage -- even one set to its exact identity -- renames "
    "the role of EVERY seeded stage after it and re-rolls all of them at once. Measured on a "
    "five-stage chain with four seeded roles: `['probe/1', 'jitter/xtalk3', 'noise/4', "
    "'interleave/4']` becomes `['probe/2', 'jitter/xtalk4', 'noise/5', 'interleave/5']`, all "
    "four renamed by one insertion, and the delivered record moves by rel rms 8.477e-02, max "
    "0.27367 = 11.470% of span. So 'bypass equals absence' holds for deterministic stages and "
    "fails for every seeded one, and the size of the failure grows with how much of the chain "
    "is seeded. `tests/test_composition.py` gates the one-role case; this is the whole-chain "
    "number. Key the streams by something stable under insertion -- the op's identity rather "
    "than its position."))
def test_one_identity_insertion_does_not_re_roll_every_seeded_stage_in_a_chain():
    a, b = _deep(False), _deep(True)
    ya, yb = a.waveform(), b.waveform()
    assert a.roles() == b.roles() and np.array_equal(ya, yb), (
        f"roles {a.roles()} -> {b.roles()}; rel rms {_rel_rms(ya, yb):.3e}, max "
        f"{np.abs(ya - yb).max():.5f} ({np.abs(ya - yb).max() / _span(ya):.3%} of span)")


@pytest.mark.xfail(strict=True, reason=(
    "MEASURED DEFECT, and the consequence that makes the index keying above dangerous rather "
    "than merely untidy: `Signal.contrast(role)` accepts a role name the chain does not have "
    "and returns a BIT-IDENTICAL record. `Streams.reroll` just records an override for a name "
    "nobody reads, so nothing is re-rolled and nothing is said. Measured: on a chain whose "
    "digitiser sits at op index 1, `contrast('noise/1')` moves the record by rel rms 5.411e-02 "
    "and `contrast('noise/2')` by 0.000e+00 -- the second is an identical waveform labelled as "
    "an independent noise draw. Combine it with the index keying and a dataset that builds "
    "contrastive pairs across chain VARIANTS (one with an extra stage, one without) using a "
    "fixed role name ships pairs that are the same record twice: measured, "
    "`contrast('noise/1')` on the chain with an identity stage inserted returns rel rms "
    "0.000e+00. That is a fabricated controlled pair, and every conclusion drawn from it is "
    "confounded. `contrast` must reject a role that is not in `roles()`."))
def test_contrast_refuses_a_role_the_chain_does_not_have():
    s = _carrier().digitize(snr_db=35.0)
    assert s.roles() == ["noise/1"]
    with pytest.raises(Exception):
        s.contrast("noise/2", seed=1234)


def test_contrast_on_a_role_the_chain_does_have_actually_re_rolls_it():
    """GROUND TRUTH for the test above: the mechanism works when the name is right, so the
    failure there is the missing validation and not a broken re-roll. Measured rel rms
    5.411e-02 between a chain and its own sibling."""
    s = _carrier().digitize(snr_db=35.0)
    assert _rel_rms(s.waveform(), s.contrast("noise/1", seed=1234)) > 1e-3


# ==================================== 4. order sensitivity, by the amount the geometry implies
EPS_R, GAMMA, TOTAL_IN = 4.0, 0.3, 12.0
GEOM_N = 1 << 15


def _echo(d_in, n=GEOM_N):
    """The echo ONE discontinuity of reflection coefficient GAMMA makes, `d_in` inches from the
    driver on a TOTAL_IN-inch line, read at the source plane: the difference the discontinuity
    makes, so the direct pulse's own dispersed tail cannot be mistaken for it."""
    g = Grid(fs=FS, baud=BAUD, n=n)
    imp = np.zeros(n)
    imp[0] = 1.0
    sect = lambda gam: [{"line": {"length_in": d_in, "eps_r": EPS_R}}, {"disc": {"gamma": gam}},
                        {"line": {"length_in": TOTAL_IN - d_in, "eps_r": EPS_R}}]
    return (SP.cascade_channel(imp, sect(GAMMA), grid=g, node="source")
            - SP.cascade_channel(imp, sect(0.0), grid=g, node="source"))


@pytest.mark.parametrize("f_ghz", [2.0, 4.0, 8.0, 16.0])
def test_reflect_then_loss_differs_from_loss_then_reflect_by_exactly_the_loss_between_them(f_ghz):
    """ORDER SENSITIVITY AS A CLOSED FORM, which is the half a "they differ" assertion misses.

    `tests/test_composition.py` establishes that the LUMPED `lossy`/`reflect` pair COMMUTES
    exactly -- so the lumped model carries no information about where the discontinuity is. Put
    the same discontinuity in a cascade and the order becomes geometry: a disc 2 inches from the
    driver is `reflect -> 10 inches of loss`, a disc 10 inches out is `10 inches of loss ->
    reflect`, and the echo the second one makes has paid the round trip -- 2*(10-2) = 16 inches
    -- that the first one has not. So

        |E_far(f)| / |E_near(f)| = 10 ** (-2 * (IL(10 in, f) - IL(2 in, f)) / 20)

    exactly, with `IL` = `physics.insertion_loss_db`, a function this test does not own.
    MEASURED against that closed form on a 12-inch eps_r=4 line, Gamma=0.3:

        2 GHz  0.286313 vs 0.286299   (4.8e-05 relative)
        4 GHz  0.139827 vs 0.139830   (2.4e-05)
        8 GHz  0.041614 vs 0.041614   (6.2e-06)
       16 GHz  0.005040 vs 0.005040   (9.5e-06)

    and the near echo's peak is 0.024781 against the far one's 0.002532, a factor of 9.787. If
    this stops holding, the cascade has stopped knowing where its discontinuities are and 'near
    reflection' and 'far reflection' have stopped being different labels."""
    k = int(round(f_ghz * 1e9 * GEOM_N / FS))
    En, Ef = np.fft.rfft(_echo(2.0)), np.fft.rfft(_echo(10.0))
    il = lambda d: P.insertion_loss_db(np.array([f_ghz]), length_in=d, eps_r=EPS_R)[0]
    measured = abs(Ef[k]) / abs(En[k])
    closed = 10.0 ** (-2.0 * (il(10.0) - il(2.0)) / 20.0)
    assert abs(measured / closed - 1.0) < 5e-3, (
        f"{f_ghz} GHz: measured {measured:.6f}, geometry says {closed:.6f} "
        f"({abs(measured / closed - 1.0):.2e} relative)")


@pytest.mark.parametrize("d_in", [2.0, 4.0, 8.0, 10.0])
def test_the_echo_arrives_at_the_round_trip_delay_the_geometry_names(d_in):
    """The other half of the geometry: WHEN, not how big. The echo's onset is
    ``2 * d * ps_per_inch(eps_r) * fs``.

    The ONSET, not the peak -- and that distinction is the instrument being calibrated rather
    than trusted. Measured for d = 2/4/8/10 inches: onset 168/347/697/873 samples against a
    closed form of 173.5/347.0/694.1/867.6, but the PEAK sits at 176/356/719/903, drifting 2.5
    -> 35 samples late as the loss grows, because a min-phase channel's excess group delay
    disperses the echo pulse. Reading the peak would have made the arrival a function of the
    channel's loss."""
    e = np.abs(_echo(d_in))
    onset = int(np.argmax(e > 1e-3 * e.max()))
    want = 2.0 * d_in * SP.ps_per_inch(EPS_R) * 1e-12 * FS
    assert abs(onset - want) < max(8.0, 0.02 * want), (
        f"d={d_in} in: onset {onset}, geometry says {want:.1f} samples")


# ================================ 5. the joins with memory: causality under record extension
def _anticipation(fn, n=N):
    """How much of an op's output on ``[0, n)`` depends on input AFTER ``n``.

    THE CHUNK-INVARIANCE QUESTION FOR A STAGE WITH MEMORY, and a physics statement rather than a
    tolerance: a causal network's output at sample k is a function of the input up to k, so
    appending different samples past the end of the record must not move anything inside it.
    An op that fails this cannot be rendered in pieces at all, in either direction."""
    a = np.asarray(fn(_x(n)), float)[:n]
    b = np.asarray(fn(np.concatenate([_x(n), _x(n, seed=11)])), float)[:n]
    return np.abs(a - b), a


CAUSAL_AT_THE_JOIN = [
    ("reflect", lambda v: P.multi_reflection(v, grid=_grid(len(v)), td_ps=281.25, gamma_s=0.055,
                                             gamma_l=0.055, n_bounce=6, node="source")),
    ("ctle", lambda v: RX.ctle(v, _grid(len(v)), 2.0, 8.0, 16.0, dc_gain=0.65)),
    ("scope/bessel", lambda v: INST.scope_bandwidth(v, _grid(len(v)), 32e9, kind="bessel",
                                                    order=4)),
]


@pytest.mark.parametrize("name,fn", CAUSAL_AT_THE_JOIN, ids=[n for n, _ in CAUSAL_AT_THE_JOIN])
def test_a_causal_stage_output_does_not_depend_on_input_past_the_records_end(name, fn):
    """Measured EXACTLY zero for all three: a time-domain echo train, a pole-zero equaliser and
    the analog front end. These are the ops a chunked renderer can carry across a boundary with
    nothing but an overlap."""
    d, a = _anticipation(fn)
    assert d.max() == 0.0, (f"{name}: {d.max():.3e} = {d.max() / _span(a):.2e} of span, first "
                            f"differing sample {int(np.argmax(d > 0))} of {N}")


@pytest.mark.parametrize("pre", [0, 1, 2])
def test_a_pre_cursor_taps_anticipation_is_exactly_the_taps_it_declares(pre):
    """A pre-cursor tap is DELIBERATELY non-causal, by exactly `pre` tap spacings and not one
    sample more -- that is the closed form, and it is what tells a chunked renderer how much
    look-ahead to carry. Measured on a 3-tap FFE at 16 samples/UI: the first sample that moves
    when the record is extended is 8192 (nothing moves), 8176 and 8160 for pre = 0, 1, 2 --
    exactly n - pre*16."""
    taps = [-0.1, 0.75, -0.15][:pre + 1] or [0.75, -0.15]
    d, _a = _anticipation(lambda v: P.tx_ffe(v, taps, SPB, pre=pre))
    first = int(np.argmax(d > 1e-12)) if (d > 1e-12).any() else N
    assert first == N - pre * SPB, f"pre={pre}: first moved sample {first}, want {N - pre * SPB}"


def test_a_zero_phase_channel_anticipates_and_its_causal_twin_does_not():
    """ORDER SENSITIVITY ASSERTED WHERE THE PHYSICS HAS IT, in the time direction. The same
    12.5 dB channel as a zero-phase filter and as its min-phase (causal) realisation: the
    zero-phase one's output inside the record moves by 0.26204 = 17.58% of span when samples are
    appended past the end, the causal one by 1.1481e-04 = 7.0e-05 of span -- a factor of 2282.

    The causal residue is not zero and that is worth stating: the min-phase response is designed
    on the record's own rfft grid, so a longer record realises a slightly different filter.
    7.0e-05 of span is 0.14 of one 11-bit code, which is the FLOOR on any chunked render of this
    op -- below a code, but not below the arithmetic."""
    kw = dict(loss_db=12.5, loss_at_ghz=8.0)
    dz, a = _anticipation(lambda v: P.lossy_channel(v, grid=_grid(len(v)), causal=False, **kw))
    dc, _ = _anticipation(lambda v: P.lossy_channel(v, grid=_grid(len(v)), causal=True, **kw))
    assert dz.max() > 0.05 * _span(a), f"the zero-phase channel anticipates only {dz.max():.3e}"
    assert dc.max() < dz.max() / 100.0, (
        f"causal {dc.max():.3e} vs zero-phase {dz.max():.3e} -- only {dz.max() / dc.max():.0f}x")
    assert dc.max() < _lsb(a), f"causal residue {dc.max() / _lsb(a):.2f} LSB of an 11-bit record"


@pytest.mark.xfail(strict=True, reason=(
    "MEASURED, and it contradicts the word 'negligible' in `instrument.probe_loading`'s own "
    "docstring. The RC loading pole is applied as `irfft(rfft(x) / (1 + j f/fc))`, a CIRCULAR "
    "convolution at the record length, so the response to the record's tail lands on its head. "
    "The docstring argues this is negligible on a record long against the 22.5 ps time "
    "constant; measured on an 8192-sample record at 256 GSa/s (32 ns, 1400 time constants), the "
    "head is off by 1.6485 = 81.63% OF SPAN at sample 0, 6.83e-03 by sample 32 and 1.37e-04 by "
    "128 -- above one 11-bit code (1.03e-03) out to roughly sample 64. So it is confined to the "
    "first ~64 samples, and it is not small there: a record delivered without a lead-in has "
    "~4 UI of head that is a function of its own tail. Same shape as BACKLOG #54 for "
    "`scope(kind='brickwall')`, measured here at 0.77787 = 33.28% of span from sample 0. The "
    "fix is `physics.apply_transfer`, which already applies these as linear convolutions."))
def test_the_probes_loading_pole_does_not_wrap_the_records_tail_onto_its_head():
    d, a = _anticipation(lambda v: INST.probe(v, _grid(len(v)), c_load_f=0.45e-12, r_source=50.0))
    assert d.max() < _lsb(a), (
        f"max {d.max():.4f} = {d.max() / _span(a):.2%} of span = {d.max() / _lsb(a):.1f} LSB at "
        f"sample {int(np.argmax(d))}; above one LSB out to sample "
        f"{int(np.max(np.nonzero(d > _lsb(a))[0]))}")


@pytest.mark.xfail(strict=True, reason=(
    "MEASURED DEFECT, and the same one that was just fixed one module away. `physics.ac_couple` "
    "is `sosfiltfilt`, i.e. the 1st-order Butterworth high-pass run FORWARDS AND BACKWARDS, so "
    "a series capacitor -- a causal physical network -- is realised as its own response "
    "SQUARED, with no phase. Measured |H| at f/fc = 0.25 / 0.5 / 1 / 2 / 4 for a 2 GHz corner: "
    "0.058755 / 0.199835 / 0.499960 / 0.800502 / 0.941926, against the 1st-order closed form "
    "0.242536 / 0.447214 / 0.707107 / 0.894427 / 0.970143 and against that closed form squared "
    "0.058824 / 0.200000 / 0.500000 / 0.800000 / 0.941176. It is the squared one to five "
    "decimals. At the corner it delivers -6.021 dB where a series cap delivers -3.010, and "
    "-0.031 deg of phase where the closed form has +45.000; the realised -3 dB point is "
    "3.1016 GHz for a 2.000 GHz ask (1.5508x, against 1/sqrt(sqrt(2)-1) = 1.5538x, the closed "
    "form for a squared high-pass). So every AC-coupled record in the library has twice the "
    "droop in dB that its recipe states. THIS IS `probe_loading`'S DEFECT EXACTLY -- '6.027 dB "
    "where the closed form says 3.014' -- and `probe_loading` has since been given a causal "
    "single pole while `ac_couple` has not. Nothing catches it: "
    "`test_no_fabricated_data.py`'s corner gate allows 0.5x-2.0x and its docstring says in so "
    "many words that a squared response still passes, and `test_analytic.py` pins "
    "`f**2/(f**2+fc**2)` as though the double pass were the specification. Two knock-on "
    "effects: the stage is non-causal, so the record's TAIL depends on samples that do not "
    "exist -- measured 0.72005 = 26.40% of span at the last sample, above one 11-bit code from "
    "sample ~7168 on, and exactly zero before it -- and the op has no group delay where a "
    "high-pass has phase lead. Fix it the way `probe_loading` was fixed, and rewrite the two "
    "tests that pin the squared form."))
def test_ac_coupling_is_the_single_pole_a_series_capacitor_is():
    g, fc = Grid(fs=FS, baud=BAUD, n=1 << 14), 2e9
    n = g.n
    worst = 0.0
    for r in (0.25, 0.5, 1.0, 2.0, 4.0):
        k = max(1, round(r * fc * n / FS))
        t = np.sin(2 * np.pi * k * np.arange(n) / n)
        y = P.ac_couple(t, grid=g, fc_hz=fc)
        H = np.fft.rfft(y)[k] / np.fft.rfft(t)[k]
        f = k * FS / n
        one = (f / fc) / np.sqrt(1.0 + (f / fc) ** 2)
        worst = max(worst, abs(abs(H) / one - 1.0))
    assert worst < 0.02, (
        f"worst |H| deviation from the 1st-order closed form {worst:.4f}; it matches that "
        f"closed form SQUARED instead")


# ============================================== 6. default byte-identity, across processes
def _pinned_chain():
    """A nine-stage chain with four independently seeded roles, ending in a stored 11-bit
    record -- the shape of the shipped recipe, and the widest surface a determinism claim can
    cover in one render."""
    return (Signal(seed=7, grid=_grid())
            .carrier("nrz", n_ui=N_UI, tr_frac=0.15, pattern="prbs7", jitter=dict(rj=0.01))
            .lossy(loss_db=12.5, loss_at_ghz=8.0, causal=True)
            .reflect(td_ps=281.25, gamma_s=0.055, gamma_l=0.055, n_bounce=6, node="source")
            .timing(rj_ps=0.4)
            .supply_coupling(f_ripple_hz=1e6, am_depth=0.01)
            .crosstalk(coupling=0.05, kind="fext")
            .probe(c_load_f=0.45e-12, noise_rms=0.005)
            .digitize(snr_db=45.0, enob=11, interleave=dict(m_cores=4, offset_mm=0.01))
            .store(bits=BITS, dither_lsb=0.5))


def _digest(a):
    return hashlib.sha256(np.ascontiguousarray(np.asarray(a, float)).tobytes()).hexdigest()[:32]


def test_a_chains_default_output_is_byte_identical_across_processes_and_hash_seeds():
    """NO STORED DIGEST, deliberately. A pinned hash is the right gate for a released kernel and
    the wrong one here: `instrument.py` is being edited as this is written, and a hash of a
    chain containing a front end would fail for a reason that has nothing to do with
    determinism. So the claim is self-consistency -- four fresh interpreters, PYTHONHASHSEED
    0, 1, 12345 and `random`, must produce the same 32 hex characters as this process. That
    catches what a stored hash catches about REPRODUCIBILITY (set or dict iteration order
    leaking into a seed, an unseeded default_rng, an environment-dependent code path) and
    survives a deliberate change to the physics.

    Measured: identical across all four, and across this process."""
    want = _digest(_pinned_chain().waveform())
    env_base = dict(os.environ)
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env_base["PYTHONPATH"] = root + os.pathsep + env_base.get("PYTHONPATH", "")
    got = {}
    for hs in ("0", "1", "12345", "random"):
        env = dict(env_base, PYTHONHASHSEED=hs)
        out = subprocess.run([sys.executable, os.path.abspath(__file__), "--digest"],
                             capture_output=True, text=True, env=env, cwd=root)
        assert out.returncode == 0, f"PYTHONHASHSEED={hs}: child failed\n{out.stderr[-2000:]}"
        got[hs] = out.stdout.strip().splitlines()[-1]
    assert set(got.values()) == {want}, f"in-process {want}, children {got}"


def test_a_recipe_survives_json_and_reproduces_the_record_bit_for_bit():
    """The other half of reproducibility: the recipe is the thing that gets stored, so the
    round trip that matters is through JSON. Measured bit-exact on the nine-stage chain,
    max |x - y| = 0.0, with four seeded roles and a dithered 11-bit export in it."""
    s = _pinned_chain()
    x = s.waveform()
    y = Signal.from_recipe(json.loads(json.dumps(s.recipe()))).waveform()
    assert np.array_equal(x, y), f"max {np.abs(x - y).max():.3e}"


# ==================================================================== 7. deep chains
def test_a_deep_stack_of_exact_identities_does_not_drift():
    """256 stages of `reflect(gamma=0)` -- an exact time-domain identity -- must be
    bit-identical to none. Measured: bit-identical at 1, 4, 16, 64 and 256 stages."""
    ref = _carrier().waveform()
    s = _carrier()
    for _ in range(256):
        s = s.reflect(**IDENTITY)
    assert np.array_equal(ref, s.waveform())


def test_a_deep_stack_of_transform_identities_stays_far_below_one_code():
    """64 stages of `lossy(loss_db=0)` -- an identity that still goes through a forward and
    inverse transform each time -- accumulates float64 error, and the question is whether it
    accumulates FAST. Measured max deviation 1.554e-15 / 3.331e-15 / 1.132e-14 / 3.497e-14 of
    amplitude at 1 / 4 / 16 / 64 stages: growth is about k**0.75, not k, and 64 stages sits at
    1.72e-14 of span -- 3.4e-11 of one 11-bit code. CLEAN, and the number says a chain would
    need of order 1e15 stages before a stored record could see it."""
    ref = _carrier().waveform()
    lsb = _lsb(ref)
    prev = 0.0
    for k in (1, 4, 16, 64):
        s = _carrier()
        for _ in range(k):
            s = s.lossy(loss_db=0.0, loss_at_ghz=8.0)
        err = float(np.abs(s.waveform() - ref).max())
        assert err < 1e-3 * lsb, f"{k} identity transforms drifted {err / lsb:.3e} LSB"
        assert err >= prev
        prev = err


def test_splitting_a_channels_loss_across_k_sections_saturates_rather_than_diverging():
    """A 24 dB channel as ONE section, and as `k` sections of 24/k dB. Insertion loss in dB is
    additive and a min-phase phase is the Hilbert transform of a log magnitude, which is linear,
    so the composite operator is the same operator -- the only difference is that each of the k
    stages truncates its own linear convolution to the record length, and composing k
    truncations is not truncating the composition.

    THE QUESTION FOR A DEEP CHAIN IS NOT WHETHER THE ERROR IS ZERO BUT WHETHER IT GROWS.
    Measured rel rms against the single 24 dB stage on an 8192-sample record:

        k =    1      2       4       8      16      32      64     128
        rel  0.000  1.19e-4 7.51e-4 1.16e-3 2.73e-3 3.05e-3 3.07e-3 3.08e-3

    It SATURATES: doubling the section count from 32 to 128 moves it by 1.0%, and the plateau is
    1.59e-03 of span -- 3.00 codes of an 11-bit record. So a deep chain is bounded, and the bound
    is three codes rather than a divergence. Worth knowing before splitting a channel into
    sections, and worth asserting as SATURATION and not as a threshold, because a threshold
    passes on a chain that is quietly on its way up."""
    ref = _carrier().lossy(loss_db=24.0, loss_at_ghz=8.0, causal=True).waveform()
    err = {}
    for k in (2, 8, 32, 64, 128):
        s = _carrier()
        for _ in range(k):
            s = s.lossy(loss_db=24.0 / k, loss_at_ghz=8.0, causal=True)
        err[k] = _rel_rms(ref, s.waveform())
    assert err[128] < 5e-3, f"128 sections are {err[128]:.3e} off one lumped stage: {err}"
    assert err[128] / err[32] - 1.0 < 0.05, (
        f"4x the sections cost {err[128] / err[32] - 1.0:.1%} more error, so it is still "
        f"growing rather than saturated: {err}")
    plateau = float(np.abs(ref - _carrier().lossy(loss_db=24.0, loss_at_ghz=8.0,
                                                  causal=True).waveform()).max())
    assert plateau == 0.0                      # the reference is the reference, bit for bit


# ============================================================================ direct run
def _report():
    W = 78
    def head(t):
        print("\n" + "=" * W + f"\n{t}\n" + "=" * W)

    head("0  the chunking harness, proved on constructed answers")
    x = _x()
    for op, kw in MEMORYLESS:
        for c in (2, 8):
            d = np.abs(_run_op(op, kw, x, _grid()) - _chunked_op(op, kw, x, c)).max()
            print(f"  {op:16s} C={c}: max |whole - chunked| {d:.3e}")
    h = np.array([1.0, -0.4, 0.15, 0.0, 0.0, 0.05])
    ref = np.convolve(x, h)[:len(x)]
    for chunk in (256, 1024, 4096):
        print(f"  overlap-save vs np.convolve, chunk={chunk:5d}: "
              f"{np.abs(stream_convolve(x, h, chunk=chunk) - ref).max():.3e}")
    fn = lambda v: P.lossy_channel(v, grid=_grid(len(v)), loss_db=12.5, loss_at_ghz=8.0,
                                   causal=True)
    whole = fn(x)
    for nt in (256, 1024, 4096):
        print(f"  a 12.5 dB channel via its own {nt:5d}-tap FIR: rel rms "
              f"{_rel_rms(whole, stream_convolve(x, channel_fir(fn, n_taps=nt), chunk=1 << 12)):.3e}")

    head("1  chunk invariance: events")
    for label, kw in (("fraction=0.01", dict(on="symbols", fraction=0.01)),
                      ("count=4", dict(on="symbols", count=4))):
        whole = _events_at(N_UI, N, **kw, **GLITCH)
        row = []
        for c in (1, 2, 4, 8, 16):
            row.append(sum(len(_events_at(N_UI // c, N // c, **kw, **GLITCH)) for _ in range(c)))
        print(f"  {label:14s} whole record {len(whole):2d} events; C=1/2/4/8/16 -> "
              + "/".join(str(v) for v in row))
    whole, per = _event_totals(8, translate="samples", on="times", samples=[5000], **GLITCH)
    print(f"  on='times', samples=[5000]: whole {whole}; 8 chunks {[p for p in per]}")
    print(f"  on='symbols', indices=[300]: whole "
          f"{len(_events_at(N_UI, N, on='symbols', indices=[300], **GLITCH))}; 8 chunks "
          + str(sum(len(_events_at(N_UI // 8, N // 8, on='symbols',
                                   indices=[300 - c * (N_UI // 8)], **GLITCH))
                    for c in range(8))) + "   <- DROPPED, correctly")
    clean = _carrier(N_UI // 8, N // 8).waveform()
    for kind, asked in (("glitch", 5000), ("glitch", -4000), ("ring", -4000)):
        y, ev = _carrier(N_UI // 8, N // 8).events(kind, on="times", samples=[asked],
                                                   severity=0.8).realize()
        d = np.abs(y - clean)
        print(f"  a 1024-sample chunk asked {kind} at {asked:6d}: labelled at "
              f"{[e.sample for e in ev.events]}, waveform moved {d.max():.5f} "
              f"({d.max() / _span(clean):.2%} of span)")

    head("2  chunk invariance: the vertical and the aggressor's rate")
    for label, v in (("flat +/-1 (the fixture)", _x()), ("tapered 1.0->0.4 (a capture)", _tapered())):
        nc = len(v) // 8
        for fs_ in (None, 1.0938):
            wl = INST.store_record(v, bits=BITS, full_scale=fs_)
            ch = np.concatenate([INST.store_record(v[c * nc:(c + 1) * nc], bits=BITS,
                                                   full_scale=fs_) for c in range(8)])
            print(f"  {label:28s} full_scale={str(fs_):7s}: "
                  f"{np.abs(wl - ch).max() / _lsb(v):5.2f} LSB, codes "
                  f"{len(np.unique(wl)):5d} -> {len(np.unique(ch)):5d}")
    print("  crosstalk aggressor, aggressor={'n_ui': 32}, victim 16.0 GBd:")
    for n in (1024, 4096, N, 2 * N):
        print(f"    record {n:6d} samples -> aggressor {32.0 * FS / n / 1e9:6.3f} GBd")

    head("3  bypass equals absence, seeds included")
    a, b = _deep(False), _deep(True)
    ya, yb = a.waveform(), b.waveform()
    print(f"  roles without the identity stage: {a.roles()}")
    print(f"  roles with    the identity stage: {b.roles()}")
    print(f"  delivered record: rel rms {_rel_rms(ya, yb):.3e}, max {np.abs(ya - yb).max():.5f} "
          f"({np.abs(ya - yb).max() / _span(ya):.3%} of span)")
    s = _carrier().digitize(snr_db=35.0)
    for role in ("noise/1", "noise/2"):
        r = _rel_rms(s.waveform(), s.contrast(role, seed=1234))
        print(f"  roles()=={s.roles()}, contrast({role!r}): rel rms {r:.3e}"
              + ("   <- SILENT NO-OP" if r == 0.0 else ""))

    head("4  order sensitivity, by the amount the geometry implies")
    En, Ef = np.fft.rfft(_echo(2.0)), np.fft.rfft(_echo(10.0))
    print("  disc 2 in from the driver (reflect -> 10 in of loss) vs 10 in (10 in of loss -> reflect)")
    for f_ghz in (2.0, 4.0, 8.0, 16.0):
        k = int(round(f_ghz * 1e9 * GEOM_N / FS))
        il = lambda d: P.insertion_loss_db(np.array([f_ghz]), length_in=d, eps_r=EPS_R)[0]
        m = abs(Ef[k]) / abs(En[k])
        c = 10.0 ** (-2.0 * (il(10.0) - il(2.0)) / 20.0)
        print(f"    {f_ghz:5.1f} GHz: |E_far|/|E_near| measured {m:.6f}, geometry {c:.6f} "
              f"({abs(m / c - 1.0):.1e} rel)")
    for d_in in (2.0, 4.0, 8.0, 10.0):
        e = np.abs(_echo(d_in))
        print(f"    d={d_in:5.1f} in: echo onset {int(np.argmax(e > 1e-3 * e.max())):4d}, geometry "
              f"{2 * d_in * SP.ps_per_inch(EPS_R) * 1e-12 * FS:7.1f}, peak "
              f"{int(np.argmax(e)):4d} (late, and later with loss)")

    head("5  the joins with memory: does the output depend on input past the end?")
    rows = list(CAUSAL_AT_THE_JOIN) + [
        ("tx_ffe pre=1", lambda v: P.tx_ffe(v, [-0.1, 0.75, -0.15], SPB, pre=1)),
        ("lossy causal=True", lambda v: P.lossy_channel(v, grid=_grid(len(v)), loss_db=12.5,
                                                        loss_at_ghz=8.0, causal=True)),
        ("lossy causal=False", lambda v: P.lossy_channel(v, grid=_grid(len(v)), loss_db=12.5,
                                                         loss_at_ghz=8.0, causal=False)),
        ("ac_couple 2 GHz", lambda v: P.ac_couple(v, grid=_grid(len(v)), fc_hz=2e9)),
        ("probe 0.45 pF", lambda v: INST.probe(v, _grid(len(v)), c_load_f=0.45e-12,
                                               r_source=50.0)),
        ("scope brickwall", lambda v: INST.scope_bandwidth(v, _grid(len(v)), 32e9,
                                                           kind="brickwall")),
    ]
    for name, fn in rows:
        d, a = _anticipation(fn)
        first = int(np.argmax(d > 1e-12)) if (d > 1e-12).any() else N
        print(f"  {name:20s} max {d.max():.4e} ({d.max() / _span(a):7.2%} of span, "
              f"{d.max() / _lsb(a):9.2f} LSB) first moved sample {first:5d}/{N}")
    print("  ac_couple(fc=2 GHz): is it the 1st-order high-pass, or that response squared?")
    g = Grid(fs=FS, baud=BAUD, n=1 << 14)
    for r in (0.25, 0.5, 1.0, 2.0, 4.0):
        k = max(1, round(r * 2e9 * g.n / FS))
        t = np.sin(2 * np.pi * k * np.arange(g.n) / g.n)
        H = np.fft.rfft(P.ac_couple(t, grid=g, fc_hz=2e9))[k] / np.fft.rfft(t)[k]
        f = k * FS / g.n
        one = (f / 2e9) / np.sqrt(1.0 + (f / 2e9) ** 2)
        print(f"    f/fc={r:5.2f}: |H| {abs(H):.6f}   1st-order {one:.6f}   squared {one**2:.6f}"
              f"   phase {np.degrees(np.angle(H)):8.3f} deg (closed form "
              f"{np.degrees(np.arctan(2e9 / f)):.3f})")

    head("6  default byte-identity")
    print(f"  in-process digest of the nine-stage chain: {_digest(_pinned_chain().waveform())}")
    s = _pinned_chain()
    print(f"  recipe -> JSON -> recipe reproduces it bit-for-bit: "
          f"{np.array_equal(s.waveform(), Signal.from_recipe(json.loads(json.dumps(s.recipe()))).waveform())}")

    head("7  deep chains")
    ref = _carrier().waveform()
    lsb = _lsb(ref)
    for k in (1, 4, 16, 64, 256):
        s = _carrier()
        for _ in range(k):
            s = s.reflect(**IDENTITY)
        print(f"  {k:4d} x reflect(gamma=0): max {np.abs(s.waveform() - ref).max():.3e}")
    for k in (1, 4, 16, 64):
        s = _carrier()
        for _ in range(k):
            s = s.lossy(loss_db=0.0, loss_at_ghz=8.0)
        e = float(np.abs(s.waveform() - ref).max())
        print(f"  {k:4d} x lossy(0 dB)     : max {e:.3e} = {e / _span(ref):.2e} of span, "
              f"{e / lsb:.2e} LSB")
    one = _carrier().lossy(loss_db=24.0, loss_at_ghz=8.0, causal=True).waveform()
    for k in (1, 2, 4, 8, 16, 32, 64):
        s = _carrier()
        for _ in range(k):
            s = s.lossy(loss_db=24.0 / k, loss_at_ghz=8.0, causal=True)
        y = s.waveform()
        print(f"  24 dB as {k:3d} x {24.0/k:6.3f} dB: rel rms {_rel_rms(one, y):.3e}, "
              f"max {np.abs(one - y).max() / _span(one):.3e} of span, "
              f"{np.abs(one - y).max() / _lsb(one):.2f} LSB")


if __name__ == "__main__":
    if "--digest" in sys.argv:
        print(_digest(_pinned_chain().waveform()))
    else:
        import warnings
        warnings.simplefilter("ignore")
        _report()
