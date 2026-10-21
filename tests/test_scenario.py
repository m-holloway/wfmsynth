"""The scenario layer: holds + agents -> a net schedule (GitHub #73).

This layer adds no physics. What it can get wrong is therefore entirely about BOOKKEEPING --
an event one sample late, a release reaction that fires into a hand-over, a merge that splits
a byte into eight transactions -- and every one of those renders perfectly well and is not the
scenario that was asked for. So most of this file compares a compiled event list against one
written out by hand, which is the only way to see a bookkeeping error.

The rest pins the three things the net op had to be able to do for this layer to work at all,
found while prototyping it: simultaneous events applied as ONE step, a released driver's level
staying out of the solve, and several Thevenin drivers on one node.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

import wfmsynth as ws
from wfmsynth import net as NET
from wfmsynth.scenario import Hold, Agent, merge_holds, schedule, hold_windows


# ------------------------------------------------------------------ the compiled event list

def test_a_hand_checked_case_compiles_to_exactly_the_expected_events():
    """Lead, lag, merge and `unless` at once, against a list written out by hand.

    `a` holds twice back to back, so the two merge into one [1, 3] us hold, widened by lead
    and lag to [0.9, 3.2]. Its release reaction would fire at 3.7, but `b` is on the net from
    2.9 to 4.0 -- the net changed hands rather than being released -- so none of it is
    emitted. Every number below is derived in this docstring, not read off a run.
    """
    holds = [Hold("a", 1e-6, 2e-6), Hold("a", 2e-6, 3e-6), Hold("b", 2.9e-6, 4e-6)]
    agents = {"a": Agent(during={"da": "driving"}, after={"da": "hiz"},
                         lead=1e-7, lag=2e-7, then=[(5e-7, "da", {"g": 0.0})], unless=("b",)),
              "b": Agent(during={"db": "driving"}, after={"db": "hiz"})}
    assert schedule(holds, agents) == [
        {"t_s": 0.0, "id": "da", "state": "hiz"},        # rest: `after` ...
        {"t_s": 0.0, "id": "da", "g": 0.0},              # ... then the whole release reaction
        {"t_s": 0.0, "id": "db", "state": "hiz"},
        {"t_s": 0.9e-6, "id": "da", "state": "driving"},
        {"t_s": 2.9e-6, "id": "db", "state": "driving"},
        {"t_s": 3.2e-6, "id": "da", "state": "hiz"},
        {"t_s": 4.0e-6, "id": "db", "state": "hiz"},
    ]


def test_the_release_reaction_fires_when_the_net_really_was_released():
    """Same agents, but `b` has gone before the reaction window opens."""
    holds = [Hold("a", 1e-6, 3e-6), Hold("b", 1.5e-6, 2.0e-6)]
    agents = {"a": Agent(during={"da": "driving"}, after={"da": "hiz"},
                         lead=1e-7, lag=2e-7, then=[(5e-7, "da", {"g": 0.0})], unless=("b",)),
              "b": Agent(during={"db": "driving"}, after={"db": "hiz"})}
    fired = [ev for ev in schedule(holds, agents) if ev.get("g") == 0.0 and ev["t_s"] > 0]
    assert len(fired) == 1 and fired[0]["t_s"] == pytest.approx(3.7e-6)


def test_a_reaction_is_all_or_nothing_never_half_applied():
    """A two-step reaction whose second step were dropped would park the thing it was
    settling in its intermediate state, permanently. The agent retakes the net inside the
    reaction window, so neither step is emitted and the `after` row stands."""
    ag = {"a": Agent(during={"d": "driving"}, after={"d": "hiz"},
                     then=[(0.0, "d", {"g": 1e-4}), (1e-6, "d", "hiz")])}
    lone = schedule([Hold("a", 1e-6, 2e-6)], ag)
    assert [ev["t_s"] for ev in lone if ev["t_s"] > 0] == [1e-6, 2e-6, 2e-6, 3e-6]

    # The first release at 2 us would react over [2, 3] us, but the agent is back on the net
    # at 2.5. Neither step of that reaction appears; the second release at 4 us reacts fully.
    retaken = schedule([Hold("a", 1e-6, 2e-6), Hold("a", 2.5e-6, 4e-6)], ag)
    assert [ev["t_s"] for ev in retaken if ev["t_s"] > 0] == pytest.approx(
        [1e-6, 2e-6, 2.5e-6, 4e-6, 4e-6, 5e-6])
    assert [ev["t_s"] for ev in retaken if ev.get("g") == 1e-4 and ev["t_s"] > 0] == [4e-6]


def test_the_rest_row_is_the_agent_settled_not_merely_after():
    """`after` is the instant of release; the rest row is long afterwards. For a switch that
    keeps conducting for a while and then opens, those are opposite states -- and seeding the
    record with `after` would start an idle bus with the switch shut."""
    ag = {"x": Agent(during={"q": "closed"}, after={"q": "closed"},
                     then=[(3e-7, "q", "open")])}
    at_zero = [ev for ev in schedule([Hold("x", 1e-6, 2e-6)], ag) if ev["t_s"] == 0.0]
    assert at_zero[-1] == {"t_s": 0.0, "id": "q", "state": "open"}
    assert schedule([Hold("x", 1e-6, 2e-6)], ag, rest=False)[0]["t_s"] == 1e-6


def test_schedule_is_pure_deterministic_and_json():
    holds = [Hold("a", 1e-6, 2e-6, bit=1), Hold("b", 1.5e-6, 3e-6)]
    agents = {"a": Agent(during={"d": "driving"}, after={"d": "hiz"}, lead=1e-8),
              "b": Agent(during={"e": {"g": 0.02}}, after={"e": {"g": 0.0}})}
    first = schedule(holds, agents)
    assert first == schedule(holds, agents)
    assert json.loads(json.dumps(first)) == first


# ------------------------------------------------------------------------------- merge_holds

def test_consecutive_bit_times_merge_despite_float_noise():
    """`t0 + i*dt + dt` and `t0 + (i+1)*dt` are the same instant and differ in the last bit.
    An exact comparison leaves eight consecutive 0 bits as eight holds, and the bus then
    releases and re-acquires seven times inside one byte -- seven edges nothing asked for."""
    dt = 2.5e-6
    holds = [Hold("a", i * dt, i * dt + dt) for i in range(8)]
    assert any(h.t1 != holds[i + 1].t0 for i, h in enumerate(holds[:-1]))   # the noise is real
    merged = merge_holds(holds)
    assert len(merged) == 1
    assert merged[0].t0 == 0.0 and merged[0].t1 == pytest.approx(8 * dt)


def test_merge_keeps_only_the_tags_its_parts_agreed_on():
    holds = [Hold("a", 0.0, 1e-6, field="addr", bit=0), Hold("a", 1e-6, 2e-6, field="addr", bit=1)]
    assert merge_holds(holds)[0].tag == {"field": "addr"}


def test_merge_is_per_agent_and_a_gap_is_respected():
    holds = [Hold("a", 0.0, 1e-6), Hold("b", 0.0, 1e-6), Hold("a", 1.5e-6, 2e-6)]
    assert [(h.who, h.t0, h.t1) for h in merge_holds(holds)] == [
        ("a", 0.0, 1e-6), ("b", 0.0, 1e-6), ("a", 1.5e-6, 2e-6)]
    assert len(merge_holds(holds, gap=1e-6)) == 2


# ----------------------------------------------------------------- the labels are the holds

def test_a_hold_window_lands_on_the_sample_the_solver_switched_at():
    """The label and the switch are the same sample, not neighbours: both go through
    `net.sample_at`, which snaps a time that is within a billionth of a sample of a boundary
    onto it. `2.5e-6 * 1e8` is 250.00000000000003, and a bare ceil puts that at 251."""
    dt = 2.5e-6
    holds = [Hold("a", i * dt, i * dt + dt) for i in range(3)]
    assert [(w["start"], w["stop"]) for w in hold_windows(holds, 1e8, 1000)] == [
        (0, 250), (250, 500), (500, 750)]


def test_hold_windows_carry_who_and_the_tags_and_feed_label_windows():
    holds = [Hold("host/sda", 1e-6, 2e-6, field="addr", bit=3)]
    w = hold_windows(holds, 1e9, 4000)[0]
    assert (w["who"], w["field"], w["bit"], w["start"], w["stop"]) == ("host/sda", "addr", 3,
                                                                      1000, 2000)
    rows = ws.events.label_windows(hold_windows(holds, 1e9, 4000), [], x=np.zeros(4000))
    assert rows[0]["i"] == w["i"]                 # join the hold's own fields back on `i`


# ------------------------------------------------- what the net op had to be able to do

def _one_node(drivers, elements=()):
    return {"nodes": ["bus"], "rails": {"vdd": 3.3},
            "elements": [{"id": "rp", "kind": "r", "a": "bus", "b": "vdd", "ohm": 1000.0},
                         {"id": "cb", "kind": "c", "a": "bus", "b": "gnd", "farad": 100e-12},
                         *elements],
            "drivers": list(drivers)}


def test_events_on_one_sample_are_one_step_not_a_sequence():
    """A hand-over written as "the old driver lets go, the new one takes over" is two events
    at one instant. Applied in sequence with a solve between them the net would pass through
    a moment with nobody driving, and the record would carry a transient worth a good
    fraction of the swing that no event asked for -- and it would depend on which order the
    two events happened to be listed in."""
    spec = _one_node([{"id": "d0", "node": "bus", "volts": 0.0, "ohm": 50.0, "state": "driving"},
                      {"id": "d1", "node": "bus", "volts": 0.0, "ohm": 50.0, "state": "hiz"}])
    off = {"t_s": 1e-6, "id": "d0", "state": "hiz"}
    on = {"t_s": 1e-6, "id": "d1", "state": "driving"}
    a = NET.solve(spec, fs=1e9, n=3000, schedule=[off, on], probes="bus")
    b = NET.solve(spec, fs=1e9, n=3000, schedule=[on, off], probes="bus")
    assert np.array_equal(a, b)                                  # order cannot matter
    assert float(np.ptp(a)) < 1e-12                              # and nothing moved at all


def test_a_released_drivers_level_cannot_leak_into_the_solve():
    """A hi-Z driver is REMOVED from the stamp, not given a big resistance, so parking a
    released driver at some non-rest voltage is invisible rather than a slow pull."""
    base = NET.solve(_one_node([{"id": "d", "node": "bus", "volts": 0.0, "ohm": 50.0,
                                 "state": "hiz"}]), fs=1e9, n=2000, probes="bus")
    odd = NET.solve(_one_node([{"id": "d", "node": "bus", "volts": -7.5, "ohm": 50.0,
                                "state": "hiz"}]), fs=1e9, n=2000, probes="bus")
    assert np.array_equal(base, odd)


def test_several_thevenin_drivers_on_one_node_combine_as_parallel_sources():
    """A weaker driver holding a level against a stronger one is two drivers, and the answer
    is the parallel Thevenin combination -- which falls out of the nodal stamp rather than
    needing a rule. Checked against that combination computed by hand."""
    spec = _one_node([{"id": "s", "node": "bus", "volts": 0.0, "ohm": 50.0, "state": "driving"},
                      {"id": "w", "node": "bus", "volts": 3.3, "ohm": 500.0, "state": "driving"}])
    v = NET.solve(spec, fs=1e9, n=40000, probes="bus")
    gs, gw, gp = 1 / 50.0, 1 / 500.0, 1 / 1000.0
    expect = (gs * 0.0 + gw * 3.3 + gp * 3.3) / (gs + gw + gp)
    assert float(v[-1]) == pytest.approx(expect, rel=0, abs=1e-12)


# ---------------------------------------------------------- the extended event vocabulary

def test_g_sets_the_strength_and_the_state_together():
    """A conductance of zero IS an open, so carrying `g` and `state` separately would let
    them disagree. Setting `g` on a driver declared hi-Z turns it on at that strength."""
    spec = _one_node([{"id": "d", "node": "bus", "volts": 0.0, "ohm": 50.0, "state": "hiz"}])
    v = NET.solve(spec, fs=1e9, n=40000, probes="bus",
                  schedule=[{"t_s": 1e-6, "id": "d", "g": 1 / 250.0}])
    assert float(v[-1]) == pytest.approx(3.3 * (1 / 1000.0) / (1 / 1000.0 + 1 / 250.0), abs=1e-12)
    back = NET.solve(spec, fs=1e9, n=40000, probes="bus",
                     schedule=[{"t_s": 1e-6, "id": "d", "g": 1 / 250.0},
                               {"t_s": 2e-6, "id": "d", "g": 0.0}])
    assert float(back[-1]) == pytest.approx(3.3, abs=1e-9)


def test_src_sets_a_drivers_level_without_touching_its_strength():
    spec = _one_node([{"id": "d", "node": "bus", "volts": 0.0, "ohm": 50.0, "state": "driving"}])
    v = NET.solve(spec, fs=1e9, n=40000, probes="bus",
                  schedule=[{"t_s": 1e-6, "id": "d", "src": 1.8}])
    gd, gp = 1 / 50.0, 1 / 1000.0
    assert float(v[-1]) == pytest.approx((gd * 1.8 + gp * 3.3) / (gd + gp), abs=1e-12)


@pytest.mark.parametrize("key", ["gate", "relax"])
def test_a_time_varying_element_is_refused_by_name_with_what_to_do_instead(key):
    spec = _one_node([{"id": "d", "node": "bus", "volts": 0.0, "ohm": 50.0, "state": "hiz"}])
    with pytest.raises(ValueError, match="staircase"):
        NET.solve(spec, fs=1e9, n=100, schedule=[{"t_s": 1e-7, "id": "d", key: 1e-8}])


def test_an_event_that_says_nothing_and_one_that_contradicts_itself_are_both_refused():
    spec = _one_node([{"id": "d", "node": "bus", "volts": 0.0, "ohm": 50.0, "state": "hiz"}])
    with pytest.raises(ValueError, match="says nothing"):
        NET.solve(spec, fs=1e9, n=100, schedule=[{"t_s": 1e-7, "id": "d"}])
    with pytest.raises(ValueError, match="disagree"):
        NET.solve(spec, fs=1e9, n=100,
                  schedule=[{"t_s": 1e-7, "id": "d", "state": "hiz", "g": 0.02}])
    with pytest.raises(ValueError, match="unknown key"):
        NET.solve(spec, fs=1e9, n=100, schedule=[{"t_s": 1e-7, "id": "d", "stat": "hiz"}])


# ------------------------------------------------------------------- refusals in the layer

def test_the_layer_refuses_what_it_cannot_mean():
    with pytest.raises(ValueError, match="t1 must be after t0"):
        Hold("a", 2e-6, 1e-6)
    with pytest.raises(ValueError, match="not configured"):
        schedule([Hold("ghost", 0.0, 1e-6)], {"a": Agent(during={"d": "driving"})})
    with pytest.raises(ValueError, match="bare number is ambiguous"):
        Agent(during={"d": 0.02})
    with pytest.raises(ValueError, match="vocabulary"):
        Agent(during={"d": {"conductance": 0.02}})
    with pytest.raises(ValueError, match="cannot be negative"):
        Agent(during={"d": "driving"}, lead=-1e-9)
    with pytest.raises(ValueError, match="AFTER the release"):
        Agent(during={"d": "driving"}, then=[(-1e-9, "d", "hiz")])


# -------------------------------------------------------------------------- the op and the recipe

def _chain(**extra):
    spec = {"nodes": ["bus"], "rails": {"vdd": 3.3},
            "elements": [{"id": "rp", "kind": "r", "a": "bus", "b": "vdd", "ohm": 1000.0},
                         {"id": "cb", "kind": "c", "a": "bus", "b": "gnd", "farad": 100e-12}],
            "drivers": [{"id": "d", "node": "bus", "volts": 0.0, "ohm": 50.0, "state": "hiz"}]}
    return (ws.Signal(seed=3, grid=ws.Grid(fs=1e9, n=8192, v_full=3.3))
            .carrier("dc", level=0.0)
            .net(**spec, probe="bus",
                 holds=[Hold("a", 1e-6, 3e-6, field="addr")],
                 agents={"a": Agent(during={"d": "driving"}, after={"d": "hiz"}, lag=2e-8)},
                 **extra))


def test_holds_and_agents_round_trip_through_the_recipe_bit_for_bit():
    sig = _chain()
    x = sig.waveform()
    assert np.array_equal(x, ws.Signal.from_recipe(sig.recipe()).waveform())


def test_the_recipe_stores_the_compact_form_and_is_plain_json():
    r = _chain().recipe()
    assert json.loads(json.dumps(r)) == r
    op = next(o for o in r["ops"] if o["op"] == "net")
    assert op["holds"] == [{"who": "a", "t0": 1e-6, "t1": 3e-6, "field": "addr"}]
    assert op["agents"] == {"a": {"during": {"d": {"state": "driving"}},
                                  "after": {"d": {"state": "hiz"}}, "lag": 2e-8}}
    assert "schedule" not in op              # the authored form, not the events it expands to


def test_the_op_is_identical_to_handing_the_same_events_in_directly():
    """The criterion from the issue: a scenario reproduces a hand-written schedule model
    event for event, and therefore sample for sample."""
    via_layer = _chain().waveform()
    spec = {"nodes": ["bus"], "rails": {"vdd": 3.3},
            "elements": [{"id": "rp", "kind": "r", "a": "bus", "b": "vdd", "ohm": 1000.0},
                         {"id": "cb", "kind": "c", "a": "bus", "b": "gnd", "farad": 100e-12}],
            "drivers": [{"id": "d", "node": "bus", "volts": 0.0, "ohm": 50.0, "state": "hiz"}]}
    by_hand = NET.solve(spec, fs=1e9, n=8192, probes="bus", schedule=[
        {"t_s": 0.0, "id": "d", "state": "hiz"},
        {"t_s": 1e-6, "id": "d", "state": "driving"},
        {"t_s": 3.02e-6, "id": "d", "state": "hiz"}])
    assert np.array_equal(via_layer, by_hand)


def test_a_signal_chain_refuses_a_hold_naming_no_agent():
    """A hold with no agent says when somebody was on the net but not what that did, so it
    would render as nothing at all -- and the record would be missing a transaction while
    still carrying its label."""
    sig = _chain()
    sig.ops[-1]["holds"] = [{"who": "nobody", "t0": 1e-6, "t1": 2e-6}]
    with pytest.raises(ValueError, match="not configured"):
        sig.waveform()
