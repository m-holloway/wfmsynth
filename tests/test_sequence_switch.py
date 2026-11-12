"""GitHub #77: `sequence` and `switch` -- chains joined in time, or selected by a signal.

What the two ops have to mean is checked against things that do not go through them:

  * the SPLICE -- a sequence equals its chains rendered ALONE, as their own records, cut and
    concatenated by hand: `array_equal`, not a tolerance, everywhere outside the join window;
  * the JOIN -- inside the window the record is the stated raised cosine, compared with the
    closed form and with its analytic 10-90 % time;
  * the STATE -- "continue" of one tone across a boundary is that tone, uninterrupted;
    "restart" is the tone from its own t = 0;
  * the GROUND TRUTH -- boundary and switch events sit where the recipe and the selector's
    analytic crossings say;
  * the SELECTION -- a switch with a constant selector IS that input, and with a moving one
    is each input, exactly, wherever no change is in progress;
  * the REFUSALS -- a duration off the grid, segments that do not fill the record, a join
    that says nothing: refused by name, never rounded.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

import wfmsynth as ws
import wfmsynth.compose as C
from wfmsynth import sequence as SEQ
from wfmsynth.compose import Signal
from wfmsynth.grid import Grid

G = Grid(fs=1e9, baud=50e6, n=8000)                  # 20 samples/UI, 8 us
N_A = 2000                                            # the first segment: 2 us
T_A = N_A / G.fs


def tone(f_hz=5e6, **kw):
    return Signal(grid=G).carrier("sine", f_hz=f_hz, **kw)


def lane(**kw):
    """A lane with randomness (jitter, noise) and memory (the channel)."""
    return (Signal(grid=G, **kw).carrier("nrz", n_ui=400, jitter=dict(rj=0.03))
            .lossy(loss_db=3.0, loss_at_ghz=0.2).rx_noise(rms=0.01))


def dc(level):
    return Signal(grid=G).carrier("dc", level=level)


def two(join="cut", t_join_s=None, a=None, b=None, state_b="restart", seed=3):
    a = {"from": tone().recipe()} if a is None else a
    b = {"from": lane(seed=7).recipe()} if b is None else b
    return Signal(seed=seed, grid=G).sequence(
        [{"chain": a, "t_s": T_A}, {"chain": b, "state": state_b}], join=join,
        t_join_s=t_join_s)


A = tone().waveform()
B = lane(seed=7).waveform()


# ------------------------------------------------------------------------------ the splice
def test_a_sequence_is_its_chains_rendered_alone_and_spliced():
    y = two().waveform()
    assert np.array_equal(y, np.concatenate([A[:N_A], B[:G.n - N_A]]))
    assert np.any(B)                                   # and the second half is not trivially 0


def test_a_signal_segment_is_the_chain_as_this_record_renders_it():
    """A Signal joins the record: its randomness is the record's, under the segment's
    namespace. `subchain()` returns that chain, which is not `lane().waveform()`."""
    s = Signal(seed=3, grid=G).sequence([{"chain": tone(), "t_s": T_A}, {"chain": lane()}])
    y = s.waveform()
    b = s.subchain("sequence/0/segments/1/chain")
    assert np.array_equal(y, np.concatenate([A[:N_A], b[:G.n - N_A]]))
    assert not np.array_equal(b, lane().waveform())
    assert list(s.subchains()) == ["sequence/0/segments/0/chain", "sequence/0/segments/1/chain"]
    assert "[sequence/0/segments/1/chain]" in s.summary()


def test_three_segments_and_the_layout_that_says_where_they_are():
    segs = [{"chain": {"from": tone().recipe()}, "t_s": 1e-6},
            {"chain": {"from": lane(seed=7).recipe()}, "t_s": 4e-6},
            {"chain": {"from": tone(2e6).recipe()}, "t_s": 3e-6}]
    s = Signal(grid=G).sequence(segs)
    lay = SEQ.layout(s.ops[0]["segments"], G)
    assert [(d["start"], d["stop"]) for d in lay] == [(0, 1000), (1000, 5000), (5000, 8000)]
    assert [d["t_start_s"] for d in lay] == [0.0, 1e-6, 5e-6]
    c = tone(2e6).waveform()
    assert np.array_equal(s.waveform(), np.concatenate([A[:1000], B[:4000], c[:3000]]))


def test_ops_after_a_sequence_act_on_the_joined_record():
    from wfmsynth import stages as ST
    y = two().filter("butter", f_hz=20e6, order=2).waveform()
    by_hand = ST.apply_filter(np.concatenate([A[:N_A], B[:G.n - N_A]]), G.fs, "butter",
                              f_hz=20e6, order=2)
    assert np.array_equal(y, by_hand)


# ------------------------------------------------------------------------------- the state
def test_continue_carries_phase_and_restart_does_not():
    """5.3 MHz is not a whole number of cycles in 2 us, so the two differ at the boundary."""
    t = tone(5.3e6)
    w = t.waveform()

    def seq(state):
        return Signal(grid=G).sequence([{"chain": t, "t_s": T_A},
                                        {"chain": t, "state": state}]).waveform()

    assert np.array_equal(seq("continue"), w)                      # one uninterrupted tone
    restart = seq("restart")
    assert np.array_equal(restart[N_A:], w[:G.n - N_A])            # from its own t = 0
    step = abs(restart[N_A] - restart[N_A - 1])
    assert step > 0.3 and abs(w[N_A] - w[N_A - 1]) < 0.04          # a phase jump vs none


def test_continue_carries_filter_memory_and_pattern_position():
    y = two(state_b="continue").waveform()
    assert np.array_equal(y[N_A:], B[N_A:])
    assert not np.array_equal(y[N_A:], B[:G.n - N_A])


# -------------------------------------------------------------------------------- the join
M = 200                                                # a 200 ns join
T_J = M / G.fs


def test_a_crossfade_is_the_stated_shape_inside_the_window_and_exact_outside():
    y, cut = two("crossfade", T_J).waveform(), two().waveform()
    win = slice(N_A, N_A + M)
    assert np.array_equal(y[:N_A], cut[:N_A]) and np.array_equal(y[N_A + M:], cut[N_A + M:])
    r = 0.5 * (1 - np.cos(np.pi * (np.arange(M) + 0.5) / M))        # written out, not ramp()
    want = (1 - r) * A[N_A:N_A + M] + r * B[:M]                     # the outgoing chain runs on
    assert np.max(np.abs(y[win] - want)) < 1e-15
    assert np.max(np.abs(y[win] - cut[win])) > 0.1                  # and it is not the cut


def test_a_crossfade_between_two_levels_has_the_analytic_rise_time():
    y = two("crossfade", T_J, a=dc(0.0), b=dc(1.0)).waveform()
    assert np.all(np.diff(y) >= 0) and y[N_A - 1] == 0.0 and y[N_A + M] == 1.0

    def cross(level):
        i = int(np.argmax(y > level))
        return i - 1 + (level - y[i - 1]) / (y[i] - y[i - 1])

    t1090 = (cross(0.9) - cross(0.1)) / G.fs
    analytic = (np.arccos(-0.8) - np.arccos(0.8)) / np.pi * T_J      # 0.5903 * t_join_s
    assert abs(t1090 - analytic) < 0.01 / G.fs, (t1090, analytic)
    mid = cross(0.5) / G.fs
    assert abs(mid - (T_A + T_J / 2 - 0.5 / G.fs)) < 1e-3 / G.fs     # centred in the window


def test_a_hold_keeps_the_last_value_then_the_next_segment_is_already_running():
    y = two("hold", T_J).waveform()
    assert np.all(y[N_A:N_A + M] == A[N_A - 1])
    assert np.array_equal(y[N_A + M:], B[M:G.n - N_A]) and np.array_equal(y[:N_A], A[:N_A])


# ------------------------------------------------------------------------------ the events
def test_boundary_events_are_at_the_requested_times_with_their_segment_index():
    segs = [{"chain": tone(), "t_s": 1e-6}, {"chain": tone(2e6), "t_s": 4e-6, "state": "continue"},
            {"chain": tone(3e6)}]
    _, ev = Signal(grid=G).sequence(segs, join="crossfade", t_join_s=T_J).realize()
    got = [(e.kind, e.sample, e.t_s, e.width, e.placement, e.params["segment"],
            e.params["state"], e.params["join"], e.params["t_join_s"]) for e in ev]
    assert got == [("segment", 1000, 1e-6, M, "sequence/0", 1, "continue", "crossfade", T_J),
                   ("segment", 5000, 5e-6, M, "sequence/0", 2, "restart", "crossfade", T_J)]
    assert all(e.at == "" for e in ev)                 # they are on the record itself
    assert ev.mask().sum() == 2 * M                    # the mask is the two join windows


def comparator(f_hz=1e6):
    return tone(f_hz).law("comparator", threshold=0.0, v_lo=0.0, v_hi=1.0)


def test_a_restarted_segments_own_events_are_on_the_records_clock():
    """A 1 MHz tone's comparator fires every 500 ns of the CHAIN's time. Restarted at 2 us,
    the same transitions are 2 us later in the record; the ones past the segment are gone."""
    alone = comparator().event_list()
    s = Signal(grid=G).sequence([{"chain": tone(), "t_s": T_A}, {"chain": comparator()}])
    ev = [e for e in s.event_list() if e.kind == "transition"]
    keep = [e for e in alone if e.sample < G.n - N_A]
    assert len(keep) == len(ev) == 12 and len(alone) == 16
    assert [e.sample for e in ev] == [e.sample + N_A for e in keep]
    assert np.allclose([e.t_s for e in ev], [e.t_s + T_A for e in keep], rtol=0, atol=1e-18)
    assert {e.at for e in ev} == {"sequence/0/segments/1/chain"}


def test_a_continued_segments_events_are_the_chains_inside_its_window():
    alone = comparator().event_list()
    s = Signal(grid=G).sequence([{"chain": tone(), "t_s": T_A},
                                 {"chain": comparator(), "state": "continue"}])
    ev = [e for e in s.event_list() if e.kind == "transition"]
    assert [(e.sample, e.t_s) for e in ev] == [(e.sample, e.t_s) for e in alone
                                               if e.sample >= N_A]
    assert len(ev) == 12


def test_a_sequence_inside_a_sequence_composes():
    inner = Signal(grid=G).sequence([{"chain": tone(), "t_s": 1e-6}, {"chain": comparator()}])
    s = Signal(grid=G).sequence([{"chain": dc(0.5), "t_s": T_A}, {"chain": inner}])
    y, ev = s.realize()
    assert np.array_equal(y[N_A:N_A + 1000], A[:1000])
    seg = [(e.sample, e.at) for e in ev if e.kind == "segment"]
    assert seg == [(3000, "sequence/0/segments/1/chain"), (2000, "")]
    first = min(e.sample for e in ev if e.kind == "transition")
    assert first == 3000 + min(e.sample for e in comparator().event_list())


# ------------------------------------------------------------ recipe, digest, roles, units
def test_the_recipe_is_data_and_replays_bit_identically():
    for s in (two("crossfade", T_J, a=tone(), b=lane()),
              Signal(seed=5, grid=G).switch(select=tone(1e6), inputs=[tone(), lane()],
                                            threshold=0.2, t_switch_s=40e-9)):
        r = json.loads(json.dumps(s.recipe()))
        again = Signal.from_recipe(r)
        assert again.sha256() == s.sha256()
        assert np.array_equal(again.waveform(), s.waveform())


@pytest.mark.parametrize("change", [
    dict(join="hold", t_join_s=T_J), dict(join="crossfade", t_join_s=T_J),
    dict(join="crossfade", t_join_s=2 * T_J), dict(state_b="continue")])
def test_every_sequence_parameter_moves_the_samples_and_the_digest(change):
    base, other = two(a=tone(), b=lane()), two(a=tone(), b=lane(), **change)
    assert base.sha256() != other.sha256()
    assert not np.array_equal(base.waveform(), other.waveform())


def test_a_duration_moves_the_boundary():
    def at(t_s):
        s = Signal(grid=G).sequence([{"chain": dc(0.0), "t_s": t_s}, {"chain": dc(1.0)}])
        return int(np.argmax(s.waveform() > 0.5)), s.sha256()

    (i1, d1), (i2, d2) = at(2e-6), at(3e-6)
    assert (i1, i2) == (2000, 3000) and d1 != d2


def test_roles_reach_inside_and_contrast_rerolls_one_segment_only():
    s = Signal(seed=3, grid=G).sequence([{"chain": lane(), "t_s": T_A}, {"chain": lane()}])
    roles = s.roles()
    assert roles == ["sequence/0/segments/0/chain/jitter/0",
                     "sequence/0/segments/1/chain/jitter/0"]
    y, c = s.waveform(), s.contrast("sequence/0/segments/1/chain/jitter/0")
    assert np.array_equal(c[:N_A], y[:N_A])
    assert not np.array_equal(c[N_A:], y[N_A:])
    # only the jitter moved: the 0.01 rms noise is the same draw in both renders, so the
    # typical sample differs by far less than a fresh noise draw would make it
    assert np.median(np.abs(c[N_A:] - y[N_A:])) < 0.002
    # the same chain in two segments does not share its draws
    assert not np.array_equal(y[:N_A], y[N_A:2 * N_A])


def test_a_tolerance_inside_a_segment_is_this_records_unit():
    part = Signal(grid=G).carrier("dc", level=ws.tol(1.0, 10))
    s = Signal(seed=4, grid=G).sequence([{"chain": dc(0.0), "t_s": T_A}, {"chain": part}])
    assert s.roles() == ["unit/0/segments.1.chain.ops.0.level"]
    level = s.ops[0]["segments"][1]["chain"]["ops"][0]["level"]
    assert 0.9 <= level <= 1.1 and level != 1.0
    assert np.all(s.waveform()[N_A:] == level)
    other = s.unit(9)
    assert other.ops[0]["segments"][1]["chain"]["ops"][0]["level"] != level


def test_a_from_segment_is_pinned_and_a_signal_segment_follows_the_seed():
    def y(seed, chain):
        return two(b=chain, seed=seed).waveform()[N_A:]

    pinned = {"from": lane(seed=7).recipe()}
    assert np.array_equal(y(1, pinned), y(2, pinned))
    assert not np.array_equal(y(1, lane()), y(2, lane()))


# --------------------------------------------------------------------- sequence: refusals
def refused(fn, *words):
    with pytest.raises(ValueError) as e:
        fn()
    msg = str(e.value)
    for w in words:
        assert w in msg, (w, msg)


def seq(segs, **kw):
    return Signal(grid=G).sequence(segs, **kw)


def test_durations_are_never_rounded_into_a_different_record():
    a, b = tone(), tone(2e6)
    refused(lambda: seq([{"chain": a, "t_s": 2.0005e-6}, {"chain": b}]),
            "segments[0]", "2000.5 samples", "not on the grid", "never rounded", "2001")
    refused(lambda: seq([{"chain": a, "t_s": 2e-6}, {"chain": b, "t_s": 5e-6}]),
            "7000 samples", "8000 samples", "fill it exactly", "1e-06 s more")
    refused(lambda: seq([{"chain": a, "t_s": 6e-6}, {"chain": b, "t_s": 5e-6}]),
            "11000 samples", "shorten")
    refused(lambda: seq([{"chain": a, "t_s": 8e-6}, {"chain": b}]), "already", "fill")
    refused(lambda: seq([{"chain": a}, {"chain": b, "t_s": 2e-6}]), "Only the LAST")
    refused(lambda: seq([{"chain": a, "t_s": 0.0}, {"chain": b}]), "at least 1")
    refused(lambda: seq([{"chain": a, "t_s": "2us"}, {"chain": b}]), "duration in seconds")
    refused(lambda: seq([{"chain": a, "t_s": ws.tol(2e-6, 5)}, {"chain": b}]),
            "toleranced", "sample grid")
    # a duration that is on the grid to within float error is accepted, not refused
    assert seq([{"chain": a, "t_s": 0.1 * 3 * 1e-5}, {"chain": b}]).waveform().shape == (G.n,)


def test_a_join_that_says_nothing_is_refused():
    a, b = tone(), tone(2e6)
    segs = [{"chain": a, "t_s": 2e-6}, {"chain": b}]
    refused(lambda: seq(segs, t_join_s=1e-8), "join='cut'", "ignored", "crossfade")
    refused(lambda: seq(segs, join="crossfade"), "t_join_s=<seconds>")
    refused(lambda: seq(segs, join="fade", t_join_s=1e-8), "['cut', 'crossfade', 'hold']")
    refused(lambda: seq(segs, join="hold", t_join_s=1.5e-9), "t_join_s", "not on the grid")
    refused(lambda: seq([{"chain": a, "t_s": 7.9e-6}, {"chain": b}], join="crossfade",
                        t_join_s=2e-7), "longer than segment 1", "100 samples")


def test_a_malformed_segment_is_refused_by_name():
    a = tone()
    refused(lambda: seq([]), "empty list")
    refused(lambda: seq({"chain": a}), "list of", "dict")
    refused(lambda: seq([{"t_s": 1e-6}, {"chain": a}]), "segments[0]", "'chain'")
    refused(lambda: seq([{"chain": a, "t_s": 2e-6, "phase": 0}, {"chain": a}]),
            "['phase']", "nothing reads")
    refused(lambda: seq([{"chain": a, "t_s": 2e-6, "state": "resume"}, {"chain": a}]),
            "'restart'", "'continue'", "'resume'")
    refused(lambda: seq([{"chain": np.zeros(8000), "t_s": 2e-6}, {"chain": a}]).waveform(),
            "segments[0]", "`chain`", "ndarray")
    refused(lambda: Signal().sequence([{"chain": a}]), "grid", "Grid(fs=")


def test_a_segment_chain_is_never_resampled_or_fitted():
    other = Signal(grid=Grid(fs=2e9, baud=50e6, n=8000)).carrier("sine", f_hz=5e6)
    short = Signal(grid=Grid(fs=1e9, baud=50e6, n=2000)).carrier("sine", f_hz=5e6)
    refused(lambda: seq([{"chain": other, "t_s": 2e-6}, {"chain": tone()}]),
            "`segments`[0]", "`chain`", "nothing is resampled")
    refused(lambda: seq([{"chain": {"from": short.recipe()}, "t_s": 2e-6},
                         {"chain": tone()}]).waveform(), "segments[0]", "n=2000", "never")
    refused(lambda: seq([{"chain": lane(seed=5), "t_s": 2e-6}, {"chain": tone()}]),
            "`segments`[0]", "seed=5", "{'from':")


@pytest.mark.parametrize("op", ["sequence", "switch"])
def test_both_are_sources_and_say_so(op):
    def build(sig):
        part = Signal().carrier("dc", level=1.0)            # no grid of its own: the record's
        if op == "sequence":
            return sig.sequence([{"chain": part}])
        return sig.switch(select=part, inputs=[part, part])

    refused(lambda: build(tone()), f"{op} is a SOURCE", "discarded")
    recipe = build(Signal(grid=G)).recipe()                 # ... and in a hand-written recipe
    recipe["ops"] = [{"op": "carrier", "kind": "dc"}] + recipe["ops"]
    refused(lambda: Signal.from_recipe(recipe).waveform(), f"{op} is a SOURCE")
    assert C.OP_KIND[op] == "source"
    assert ((op in C._LEAD_LTI) + (op in C._LEAD_SKIP) + (op in C._LEAD_REJECT)
            + (op in C._LEAD_ANALYTIC)) == 1
    refused(lambda: build(Signal(grid=G, lead_in=True)).waveform(), "lead_in")
    iq = Grid(fs=1e9, baud=50e6, n=8000, domain="iq", fc_hz=1e8)
    refused(lambda: build(Signal(grid=iq)), op, "domain 'real'")


# ---------------------------------------------------------------------------------- switch
def sw(select, inputs=None, **kw):
    inputs = [{"from": tone().recipe()}, {"from": lane(seed=7).recipe()}] if inputs is None \
        else inputs
    return Signal(seed=3, grid=G).switch(select=select, inputs=inputs, **kw)


@pytest.mark.parametrize("t_switch_s", [0.0, 50e-9])
def test_a_constant_selector_is_that_input(t_switch_s):
    lo, ev_lo = sw(dc(0.0), threshold=0.5, t_switch_s=t_switch_s).realize()
    hi, ev_hi = sw(dc(1.0), threshold=0.5, t_switch_s=t_switch_s).realize()
    assert np.array_equal(lo, A) and np.array_equal(hi, B)
    assert len(ev_lo) == len(ev_hi) == 0
    # exactly ON the threshold has not crossed it
    assert np.array_equal(sw(dc(0.5), threshold=0.5).waveform(), A)


def test_three_inputs_take_two_ascending_thresholds():
    c = tone(2e6).waveform()
    ins = [{"from": tone().recipe()}, {"from": lane(seed=7).recipe()}, tone(2e6)]
    for level, want in ((-1.0, A), (0.0, B), (1.0, c)):
        assert np.array_equal(sw(dc(level), ins, threshold=[-0.5, 0.5]).waveform(), want)
    # a selector that walks through all three: each input, exactly, while it is selected
    stair = tone(250e3)                                 # one cycle in 4 us
    y = sw(stair, ins, threshold=[-0.5, 0.5]).waveform()
    s = stair.waveform()
    pick = np.where(s > 0.5, 2, np.where(s > -0.5, 1, 0))
    assert set(pick) == {0, 1, 2}
    assert np.array_equal(y, np.stack([A, B, c])[pick, np.arange(G.n)])


def test_a_switch_is_each_input_exactly_wherever_no_change_is_in_progress():
    m = 40
    gate = Signal(grid=G).carrier("square", f_hz=500e3, tr_frac=0.01)
    s = sw(gate, threshold=0.0, t_switch_s=m / G.fs)
    y, ev = s.realize()
    sel = gate.waveform()
    ideal = np.where(sel > 0.0, B, A)
    busy = np.zeros(G.n, bool)
    for e in ev:
        assert e.kind == "switch" and e.width == m and e.placement == "switch/0"
        busy[e.sample:e.sample + m] = True
    n_changes = int(np.count_nonzero(np.diff(sel > 0.0)))
    assert len(ev) == n_changes == 7 and busy.sum() == n_changes * m
    assert np.array_equal(y[~busy], ideal[~busy])
    assert np.max(np.abs(y[busy] - ideal[busy])) > 0.1          # the switching time is real
    assert np.array_equal(sw(gate, threshold=0.0).waveform(), ideal)    # 0 is an ideal switch


def test_the_switching_transient_is_the_stated_raised_cosine():
    m = 100
    gate = Signal(grid=G).carrier("step", t_step_s=4e-6, tr_s=2e-9)
    y, ev = sw(gate, [dc(0.0), dc(1.0)], threshold=0.0, t_switch_s=m / G.fs).realize()
    (e,) = ev.events
    r = 0.5 * (1 - np.cos(np.pi * (np.arange(m) + 0.5) / m))
    assert np.max(np.abs(y[e.sample:e.sample + m] - r)) < 1e-12
    assert np.all(y[:e.sample] == 0.0) and np.all(y[e.sample + m:] == 1.0)
    assert abs(e.t_s - 4e-6) < 1 / G.fs and e.params == {
        "input": 1, "was": 0, "threshold": 0.0, "t_switch_s": m / G.fs}


def test_switch_events_are_the_selectors_crossings_between_samples():
    """A 1.3 MHz sine crosses 0.25 at t = (asin(0.25) + 2 pi k) / w rising and
    (pi - asin(0.25) + 2 pi k) / w falling: none of them on a sample."""
    f = 1.3e6
    _, ev = sw(tone(f), threshold=0.25).realize()
    w, a = 2 * np.pi * f, np.arcsin(0.25)
    k = np.arange(12)
    want = np.sort(np.concatenate([(a + 2 * np.pi * k) / w, (np.pi - a + 2 * np.pi * k) / w]))
    want = want[want < G.n / G.fs]
    got = np.array([e.t_s for e in ev])
    assert len(got) == len(want) == 21
    # a chord under a sine sags by at most (w/fs)^2 / 8, which the slope w*cos(a)/fs turns
    # into time: 1.1e-3 of a sample here. Measured 2.6e-4.
    bound = (w / G.fs) ** 2 / 8 / (w / G.fs * np.cos(a))
    assert 1e-4 < np.max(np.abs(got - want)) * G.fs < bound < 1.1e-3
    assert [e.sample for e in ev] == list(np.ceil(want * G.fs).astype(int))
    assert [e.direction for e in ev] == [1.0, -1.0] * 10 + [1.0]
    assert [(e.params["was"], e.params["input"]) for e in ev.events[:2]] == [(0, 1), (1, 0)]


def test_overlapping_changes_still_share_the_line_completely():
    """A selector that toggles faster than the switch settles: the inputs' shares must sum to
    one throughout, so two equal inputs give exactly that level and nothing else."""
    fast = tone(20e6)                                   # a change every 25 samples
    y, ev = sw(fast, [dc(1.0), dc(1.0)], threshold=0.0, t_switch_s=60e-9).realize()
    assert len(ev) > 300 and np.max(np.abs(y - 1.0)) < 1e-12
    y = sw(fast, [dc(0.0), dc(1.0)], threshold=0.0, t_switch_s=60e-9).waveform()
    assert 0.0 <= y.min() and y.max() <= 1.0 and 0.2 < y[4000:].min() and y[4000:].max() < 0.8


def test_switch_parameters_move_the_samples_and_roles_reach_inside():
    gate = tone(1e6)
    base = sw(gate, threshold=0.0)
    for other in (sw(gate, threshold=0.3), sw(gate, threshold=0.0, t_switch_s=30e-9),
                  sw(tone(2e6), threshold=0.0)):
        assert other.sha256() != base.sha256()
        assert not np.array_equal(other.waveform(), base.waveform())
    s = Signal(seed=3, grid=G).switch(select=gate, inputs=[tone(), lane()])
    assert s.roles() == ["switch/0/inputs/1/jitter/0"]
    y, c = s.waveform(), s.contrast("switch/0/inputs/1/jitter/0")
    on = gate.waveform() > 0.0
    assert np.array_equal(c[~on], y[~on]) and not np.array_equal(c[on], y[on])
    assert list(s.subchains()) == ["switch/0/select", "switch/0/inputs/0", "switch/0/inputs/1"]


def test_a_comparator_in_the_select_chain_stops_the_chatter():
    noisy = Signal(grid=G).carrier("sine", f_hz=500e3).rx_noise(rms=0.05)
    clean = Signal(grid=G).carrier("sine", f_hz=500e3).rx_noise(rms=0.05).law(
        "comparator", threshold=0.0, hysteresis=0.4, v_lo=-1.0, v_hi=1.0)
    n_noisy = sum(e.kind == "switch" for e in sw(noisy, threshold=0.0).event_list())
    n_clean = sum(e.kind == "switch" for e in sw(clean, threshold=0.0).event_list())
    assert n_clean == 7 and n_noisy > 3 * n_clean        # 500 kHz crosses at 1, 2, ... 7 us


def test_switch_refusals():
    a, b = tone(), tone(2e6)
    refused(lambda: sw(a, [a, b, a], threshold=0.0), "3 inputs", "2 thresholds", "ascending")
    refused(lambda: sw(a, [a, b, a], threshold=[0.5, -0.5]), "strictly ascending")
    refused(lambda: sw(a, [a, b], threshold=[0.1, 0.2]), "2 inputs", "1 threshold")
    refused(lambda: sw(a, [a], threshold=0.0), "at least two")
    refused(lambda: sw(a, a), "`inputs` is a list")
    refused(lambda: sw(a, [a, b], t_switch_s=1.5e-9), "t_switch_s", "not on the grid")
    refused(lambda: sw(a, [a, b], t_switch_s=-1e-9), "t_switch_s", "at least 0")
    refused(lambda: Signal().switch(select=a, inputs=[a, b]), "grid", "Grid(fs=")
    recipe = sw(a, [a, b]).recipe()
    del recipe["ops"][0]["select"]
    refused(lambda: Signal.from_recipe(recipe).waveform(), "switch needs select=")
    refused(lambda: sw(np.zeros(8000), [a, b]).waveform(), "`select`", "ndarray")


def test_the_array_level_functions_agree_with_the_ops():
    y, changes = SEQ.mux(tone(1e6).waveform(), [A, B], threshold=0.0, m=0)
    assert np.array_equal(y, sw(tone(1e6), threshold=0.0).waveform()) and len(changes) == 16
    plan = SEQ.layout([{"chain": {}, "t_s": T_A}, {"chain": {}}], G)
    assert np.array_equal(SEQ.splice([A, B], plan), two().waveform())
    r = SEQ.ramp(4)
    assert np.allclose(r + r[::-1], 1.0) and np.all(np.diff(r) > 0) and 0 < r[0] < r[-1] < 1
