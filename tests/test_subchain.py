"""GitHub #75: a sub-chain is a value -- one nested-chain form for every op that takes a signal.

`crosstalk(aggressor=)`, `modulate(message=)`, `open_drain(second=)` and `net` drivers each took
a second signal in their own shape, `crosstalk_matrix` took none, and "the neighbouring lane
after its own channel" could not be said at all. Now every such parameter takes
`{"ops": [...]}`, `{"from": <recipe>}` or a `Signal`, through one resolver
(`wfmsynth.subchain`).

What that has to mean is checked here against things that do not go through the resolver:

  * the RESULT -- an op given a sub-chain equals the same chain rendered separately and
    combined by hand with the physics primitive, sample for sample;
  * the RANDOMNESS -- re-rolling one role inside a sub-chain changes that factor and nothing
    else, shown by rendering the pieces independently;
  * the PAST -- every form that existed before still reads the role streams it always read and
    stores the recipe it always stored, compared against the primitive called directly;
  * the REFUSALS -- a sub-chain on another grid, at another rate, complex, or carrying a
    parameter nothing reads is refused by name, never fitted.
"""
from __future__ import annotations

import json
import warnings

import numpy as np
import pytest

import wfmsynth.physics as P
from wfmsynth import subchain as SUB
from wfmsynth.compose import Signal, _carrier
from wfmsynth.grid import Grid
from wfmsynth.streams import Streams

G = Grid(fs=64e9, baud=4e9, n=4096)                  # 16 samples/UI
GS = Grid(fs=1e8, baud=1e6, n=4096)                  # a slow bus, for open_drain
J = dict(rj=0.4)
N_UI = 256


@pytest.fixture(autouse=True)
def _quiet():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


def victim(seed=1):
    return Signal(seed=seed, grid=G).carrier("nrz", n_ui=N_UI, jitter=J)


def lane():
    """A neighbouring lane AFTER ITS OWN CHANNEL, with two random factors of its own."""
    return (Signal(grid=G).carrier("nrz", n_ui=N_UI, seed=9, jitter=J)
            .lossy(loss_db=6.0, loss_at_ghz=2.0).digitize(noise_rms=0.02))


def namespaced(sig_ops, seed, at):
    """A sub-chain rendered WITHOUT the resolver: its ops under the prefixed role names."""
    s = Signal(grid=G)
    s.ops = [dict(o) for o in sig_ops]
    return s._run(SUB.Namespaced(Streams(seed), at + "/"))


# ------------------------------------------------------------------ the result
def test_an_aggressor_sub_chain_equals_the_chain_rendered_separately_and_coupled_by_hand():
    sig = victim().crosstalk(aggressor=lane(), coupling=0.1)
    by_hand = P.crosstalk(victim().waveform(),
                          namespaced(lane().ops, 1, "crosstalk/1/aggressor"), coupling=0.1)
    assert np.array_equal(sig.waveform(), by_hand)
    # and the aggressor is not a bystander: this is a real, measurable coupling
    assert np.abs(sig.waveform() - victim().waveform()).max() > 0.05


def test_subchain_returns_the_signal_exactly_as_it_was_used():
    sig = victim().crosstalk(aggressor=lane(), coupling=0.1)
    assert np.array_equal(sig.subchain("crosstalk/1/aggressor"),
                          namespaced(lane().ops, 1, "crosstalk/1/aggressor"))
    assert list(sig.subchains()) == ["crosstalk/1/aggressor"]
    assert sig.subchains()["crosstalk/1/aggressor"].ops == lane().ops
    with pytest.raises(ValueError, match="no sub-chain at 'crosstalk/7/aggressor'.*holds"):
        sig.subchain("crosstalk/7/aggressor")


def test_a_recipe_under_from_is_that_record_bit_for_bit_with_its_own_seed_and_baud():
    other = (Signal(seed=4, grid=Grid(fs=G.fs, baud=2.5e9, n=G.n))
             .carrier("pam4", n_ui=160, jitter=J).lossy(loss_db=3.0, loss_at_ghz=2.0))
    for seed in (1, 2):                         # the parent's seed does not reach inside it
        sig = victim(seed).crosstalk(aggressor={"from": other.recipe()}, coupling=0.2)
        assert np.array_equal(sig.subchain("crosstalk/1/aggressor"), other.waveform())
        assert np.array_equal(sig.waveform(),
                              P.crosstalk(victim(seed).waveform(), other.waveform(), coupling=0.2))


def test_a_message_sub_chain_modulates_like_the_same_message_by_hand():
    msg = Signal(grid=G).carrier("noise", rms=0.3).scope(bw_hz=1e9)
    sig = Signal(seed=3, grid=G).carrier("sine", f_hz=8e9).modulate(message=msg, depth=0.5)
    m = sig.subchain("modulate/1/message")
    carrier = Signal(seed=3, grid=G).carrier("sine", f_hz=8e9).waveform()
    assert np.array_equal(sig.waveform(), P.am_modulate(carrier, m, depth=0.5))
    assert m.std() > 0.05                                  # the message is really there
    fm = Signal(seed=3, grid=G).modulate(kind="fm", fc_hz=8e9, dev_hz=1e9, message=msg)
    assert np.array_equal(fm.waveform(), P.fm_modulate(fm.subchain("modulate/0/message"),
                                                       fc_hz=8e9, dev_hz=1e9, n=G.n, fs=G.fs))


def test_a_second_open_drain_driver_given_as_a_sub_chain_sinks_when_its_chain_is_low():
    first = Signal(seed=2, grid=GS).carrier("nrz", n_ui=40)
    second = Signal(grid=GS).carrier("nrz", n_ui=32, seed=9).lossy(loss_db=1.0, loss_at_ghz=0.001)
    kw = dict(r_pullup_ohm=1e3, c_bus_f=50e-12, r_sink_b_ohm=40.0)
    sig = Signal(seed=2, grid=GS).carrier("nrz", n_ui=40).open_drain(second=second, **kw)
    xa, xb = first.waveform(), sig.subchain("open_drain/1/second")
    mid = lambda v: 0.5 * (np.percentile(v, 1) + np.percentile(v, 99))      # noqa: E731
    by_hand = P.open_drain_line(xa < mid(xa), GS.fs, r_pullup_ohm=1e3, c_bus_f=50e-12, v_dd=3.3,
                                r_sink_ohm=20.0, sink_b_on=xb < mid(xb), r_sink_b_ohm=40.0)
    assert np.array_equal(sig.waveform(), by_hand)


def test_crosstalk_matrix_takes_real_aggressors_and_superposes_them():
    a = lane()
    b = Signal(grid=G).carrier("nrz", n_ui=200, seed=3)            # another symbol count
    sig = victim().crosstalk_matrix(couplings=[0.08, 0.03], kind=["fext", "next"],
                                    aggressors=[a, b])
    v = victim().waveform()
    wa, wb = (sig.subchain(f"crosstalk_matrix/1/aggressors/{k}") for k in (0, 1))
    by_hand = v + (P.crosstalk(v, wa, coupling=0.08, kind="fext") - v) \
                + (P.crosstalk(v, wb, coupling=0.03, kind="next") - v)
    assert np.array_equal(sig.waveform(), by_hand)
    # each term is separately audited: dropping one aggressor removes exactly its contribution
    only_a = victim().crosstalk_matrix(couplings=[0.08], aggressors=[a]).waveform()
    assert np.array_equal(only_a, P.crosstalk(v, wa, coupling=0.08))


def test_the_other_ops_that_take_a_signal_take_a_sub_chain_too():
    """`hybrid_echo(own=)`, `pass_fet(gate=)` and `events(aggressor=)` are the same parameter."""
    v = victim().waveform()
    own = Signal(grid=G).carrier("nrz", n_ui=N_UI, seed=4).lossy(loss_db=3.0, loss_at_ghz=2.0)
    s = victim().hybrid_echo(own=own, isolation_db=20.0, td_samples=5)
    assert np.array_equal(s.waveform(), P.hybrid_echo(v, s.subchain("hybrid_echo/1/own"), grid=G,
                                                      isolation_db=20.0, td_samples=5))
    gate = Signal(grid=G).carrier("square", f_hz=5e8).scope(bw_hz=4e9)
    s = victim().pass_fet(gate=gate)
    on = s.subchain("pass_fet/1/gate") > 0.0
    assert 0.3 < on.mean() < 0.7                                  # the gate really switches
    assert np.array_equal(s.waveform(), P.pass_fet(v, on, rds_on_ohm=5.0, r_load_ohm=1e6,
                                                   v_rail_hi=None, v_rail_lo=None,
                                                   diode_drop=0.6, diode_on_ohm=1.0))
    s = victim().events("glitch", on="aggressor", aggressor=own, count=4)
    as_array = victim().events("glitch", on="aggressor", count=4,
                               aggressor=s.subchain("events/1/aggressor"))
    assert np.array_equal(s.waveform(), as_array.waveform())
    assert [e.sample for e in s.event_list()] == [e.sample for e in as_array.event_list()]


def test_a_sub_chain_can_hold_a_sub_chain():
    inner = Signal(grid=G).carrier("nrz", n_ui=N_UI, seed=5, jitter=J)
    mid = Signal(grid=G).carrier("nrz", n_ui=N_UI, seed=9).crosstalk(aggressor=inner, coupling=0.3)
    sig = victim().crosstalk(aggressor=mid, coupling=0.1)
    deep = "crosstalk/1/aggressor/crosstalk/1/aggressor"
    assert list(sig.subchains()) == ["crosstalk/1/aggressor", deep]
    assert sig.roles() == ["jitter/0", f"{deep}/jitter/0"]
    assert np.array_equal(sig.subchain(deep), namespaced(inner.ops, 1, deep))
    base = sig.waveform()
    assert not np.array_equal(sig.contrast(f"{deep}/jitter/0"), base)
    assert np.array_equal(Signal.from_recipe(json.loads(sig.to_json())).waveform(), base)


# ------------------------------------------------------------------ the randomness
def test_roles_reach_inside_and_each_re_rolls_exactly_one_factor():
    sig = victim().crosstalk(aggressor=lane(), coupling=0.1).digitize(noise_rms=0.01)
    at = "crosstalk/1/aggressor"
    assert sig.roles() == ["jitter/0", "noise/2", f"{at}/jitter/0", f"{at}/noise/2"]
    base, a0 = sig.waveform(), sig.subchain(at)
    for role in sig.roles():
        assert not np.array_equal(sig.contrast(role, seed=11), base), role

    # Re-roll ONLY the aggressor's noise. Rendered independently of the record: the aggressor
    # before its digitizer is untouched (its jitter held), and what moved is noise of the stated
    # size -- 20 mV rms, times sqrt(2) because it is the difference of two independent draws.
    # (4096 samples estimate an rms to 1.1 %, so 5 % is 4.5 sigma.)
    st = Streams(1).reroll(f"{at}/noise/2", seed=11)
    a1 = sig.subchain(at, streams=st)
    clean = namespaced(lane().ops[:2], 1, at)
    for a in (a0, a1):
        assert abs(np.std(a - clean) / 0.02 - 1.0) < 0.05, np.std(a - clean)
    d = a1 - a0
    assert abs(np.std(d) / (np.sqrt(2.0) * 0.02) - 1.0) < 0.05, np.std(d)
    assert abs(np.corrcoef(a0 - clean, a1 - clean)[0, 1]) < 0.07       # 4.5 sigma of 1/sqrt(n)
    # ... and in the record that is the ONLY thing that moved: victim, its jitter and the
    # record's own digitizer noise are bit-identical, so the difference is the coupling of `d`.
    v = victim().waveform()
    noise = base - P.crosstalk(v, a0, coupling=0.1)             # the record's own digitizer
    assert np.allclose(sig.contrast(f"{at}/noise/2", seed=11),
                       P.crosstalk(v, a1, coupling=0.1) + noise, rtol=0, atol=1e-12)


def test_one_seed_moves_a_signal_sub_chain_and_two_holders_never_share_a_draw():
    def build(seed):
        return victim(seed).crosstalk_matrix(couplings=[0.05, 0.05], aggressors=[lane(), lane()])
    s = build(1)
    a, b = (s.subchain(f"crosstalk_matrix/1/aggressors/{k}") for k in (0, 1))
    assert not np.array_equal(a, b), "the same chain held twice drew the same jitter"
    assert np.array_equal(a, build(1).subchain("crosstalk_matrix/1/aggressors/0"))
    assert not np.array_equal(a, build(2).subchain("crosstalk_matrix/1/aggressors/0"))


def test_contrast_reaches_a_record_embedded_under_from():
    other = Signal(seed=4, grid=G).carrier("nrz", n_ui=N_UI, jitter=J).digitize(snr_db=30.0)
    sig = victim().crosstalk(aggressor={"from": other}, coupling=0.2)
    at = "crosstalk/1/aggressor"
    assert sig.roles() == ["jitter/0", f"{at}/jitter/0", f"{at}/noise/1"]
    st = Streams(1).reroll(f"{at}/noise/1", seed=5)
    # independently: the embedded record with ITS noise role re-rolled the same way
    want = other.waveform(Streams(4, {"noise/1": 5}))
    assert np.array_equal(sig.subchain(at, streams=st), want)
    assert not np.array_equal(want, other.waveform())


# ------------------------------------------------------------------ the recipe
def test_every_way_of_writing_a_sub_chain_stores_the_same_json_and_replays(tmp_path):
    ln = lane()
    as_signal = victim().crosstalk(aggressor=ln)
    as_ops = victim().crosstalk(aggressor={"ops": ln.ops})
    as_ops_signal = victim().crosstalk(aggressor={"ops": ln})
    assert as_signal.ops[1] == {"op": "crosstalk", "aggressor": {"ops": ln.ops}}
    assert as_signal.sha256() == as_ops.sha256() == as_ops_signal.sha256()

    other = Signal(seed=4, grid=G).carrier("nrz", n_ui=N_UI, jitter=J)
    path = tmp_path / "other.json"
    path.write_text(other.to_json())
    forms = [victim().crosstalk(aggressor={"from": f}) for f in (other, other.recipe(), str(path))]
    assert forms[0].ops[1]["aggressor"] == {"from": other.recipe()}
    assert len({f.sha256() for f in forms}) == 1
    path.unlink()                               # embedded: the record needs no extra file
    with pytest.raises(ValueError, match="not the path of a saved recipe"):
        victim().crosstalk(aggressor={"from": str(path)})
    for s in (as_signal, forms[2]):
        again = Signal.from_recipe(json.loads(s.to_json()))
        assert np.array_equal(again.waveform(), s.waveform())
    # the digest covers the sub-chain: one knob inside it is a different record
    ln2 = lane()
    ln2.ops[1]["loss_db"] = 6.5
    assert victim().crosstalk(aggressor=ln2).sha256() != as_signal.sha256()


def test_the_forms_that_predate_sub_chains_store_and_draw_exactly_what_they_did():
    """Bit-identity of the legacy shorthand, against the primitive and the role stream it has
    always used -- neither of which goes through the resolver."""
    spec = dict(n_ui=N_UI, seed=9, jitter=J)
    v = victim().waveform()

    def legacy(kind, role):
        return np.asarray(_carrier({"kind": kind, **spec}, Streams(1), G, role), float)

    s = victim().crosstalk(aggressor=spec, coupling=0.2)
    assert s.ops[1] == {"op": "crosstalk", "aggressor": spec, "coupling": 0.2}
    assert np.array_equal(s.waveform(), P.crosstalk(v, legacy("nrz", "xtalk1"), coupling=0.2))
    assert s.roles() == ["jitter/0", "jitter/xtalk1"]
    s = victim().crosstalk()
    assert np.array_equal(s.waveform(), P.crosstalk(v, np.asarray(_carrier(
        {"kind": "nrz", "n_ui": 32, "seed": 7}, Streams(1), G, "xtalk1"), float), coupling=0.1))
    s = victim().modulate(message=spec, depth=0.5)
    assert np.array_equal(s.waveform(), P.am_modulate(v, legacy("sine", "modulate_msg1"),
                                                      depth=0.5))
    s = victim().hybrid_echo(own=spec, isolation_db=20.0, td_samples=5)
    assert np.array_equal(s.waveform(), P.hybrid_echo(v, legacy("nrz", "echo1"), grid=G,
                                                      isolation_db=20.0, td_samples=5))
    s = victim().crosstalk_matrix(couplings=[0.05, 0.03])
    assert s.ops[1] == {"op": "crosstalk_matrix", "couplings": [0.05, 0.03]}
    assert np.array_equal(s.waveform(), P.crosstalk_matrix(v, G, [0.05, 0.03]))


def test_net_drivers_are_the_same_form_and_roles_now_list_them():
    host = Signal(grid=GS).carrier("nrz", n_ui=40, jitter=J)
    sig = Signal(seed=5, grid=GS).net(
        nodes=["bus"], rails={"vdd": 3.3}, probe="bus",
        elements=[{"id": "rp", "kind": "r", "a": "bus", "b": "vdd", "ohm": 885.0},
                  {"id": "cb", "kind": "c", "a": "bus", "b": "gnd", "farad": 40e-12}],
        drivers=[{"id": "host", "node": "bus", "ohm": 60.0, "state": "driving",
                  "source": {"ops": host, "volts": [0.0, 3.3]}}])
    assert sig.roles() == ["net/0/host/jitter/0"]
    assert np.array_equal(sig.subchain("net/0/host"), namespaced(host.ops, 5, "net/0/host"))
    assert not np.array_equal(sig.contrast("net/0/host/jitter/0"), sig.waveform())


# ------------------------------------------------------------------ events
def test_an_event_inside_a_sub_chain_surfaces_under_its_namespace_and_labels_nothing_here():
    aggr = Signal(grid=G).carrier("nrz", n_ui=N_UI, seed=9).events("glitch", count=3)
    sig = victim().crosstalk(aggressor=aggr, coupling=0.1).events("runt", count=2)
    x, ev = sig.realize()
    assert np.array_equal(x, sig.waveform())            # collecting events changes nothing
    inner = [e for e in ev if e.at]
    assert [e.kind for e in ev] == ["glitch"] * 3 + ["runt"] * 2
    assert {e.at for e in inner} == {"crosstalk/1/aggressor"} and len(inner) == 3
    assert all(d.get("at") == "crosstalk/1/aggressor" for d in ev.to_dicts()[:3])
    assert "at" not in ev.to_dicts()[3]
    # where the aggressor was glitched is ground truth about the AGGRESSOR, found independently
    # by rendering it with and without its events op
    clean = namespaced(aggr.ops[:1], 1, "crosstalk/1/aggressor")
    moved = np.flatnonzero(sig.subchain("crosstalk/1/aggressor") != clean)
    for e in inner:
        lo, hi = e.span(G.n)
        assert moved[(moved >= lo) & (moved < hi)].size, e
    assert all(any(e.span(G.n)[0] <= k < e.span(G.n)[1] for e in inner) for k in moved)
    # but the record's own mask and window labels are about the record's own events only
    on = ev.on_record()
    assert [e.kind for e in on] == ["runt", "runt"]
    assert ev.mask().sum() == sum(e.span(G.n)[1] - e.span(G.n)[0] for e in on)
    lo, hi = inner[0].span(G.n)
    assert all(not e.at for e in ev.overlapping(lo, hi))


# ------------------------------------------------------------------ the tree
def test_summary_prints_the_chain_as_a_tree_with_each_namespace():
    other = Signal(seed=4, grid=Grid(fs=G.fs, baud=2.5e9, n=G.n)).carrier("pam4", n_ui=160)
    sig = (victim().crosstalk(aggressor=lane(), coupling=0.1)
           .crosstalk_matrix(couplings=[0.05], aggressors=[{"from": other}]))
    assert sig.summary() == """\
Signal  seed=1  fs=6.4e+10 Hz, baud=4e+09, n=4096
  0  carrier           kind='nrz', n_ui=256, jitter={'rj': 0.4}
  1  crosstalk         coupling=0.1
       aggressor  [crosstalk/1/aggressor]
         0  carrier   kind='nrz', n_ui=256, seed=9, jitter={'rj': 0.4}
         1  lossy     loss_db=6.0, loss_at_ghz=2.0
         2  digitize  noise_rms=0.02
  2  crosstalk_matrix  couplings=[0.05]
       aggressors[0]  [crosstalk_matrix/2/aggressors/0]  its own record: seed=4, \
fs=6.4e+10 Hz, baud=2.5e+09, n=4096
         0  carrier  kind='pam4', n_ui=160"""
    # every namespace it prints is one `subchain()` accepts
    for at in sig.subchains():
        assert f"[{at}]" in sig.summary() and sig.subchain(at).shape == (G.n,)


def test_summary_of_a_flat_chain_is_one_line_per_op_and_a_long_value_prints_as_its_size():
    sig = (Signal(seed=7, grid=G).symbols([1, -1] * 128).lossy(loss_db=6.0, loss_at_ghz=2.0)
           .annotate(stage="channel"))
    assert sig.summary() == """\
Signal  seed=7  fs=6.4e+10 Hz, baud=4e+09, n=4096
  0  symbols  symbols=[256 values]
  1  lossy    loss_db=6.0, loss_at_ghz=2.0"""
    assert Signal().carrier("nrz").summary() == "Signal  seed=0  no grid\n  0  carrier  kind='nrz'"


# ------------------------------------------------------------------ refusals
def _xt(aggressor):
    return victim().crosstalk(aggressor=aggressor)


@pytest.mark.parametrize("build,match", [
    # another grid, as a Signal: it would be rendered on THIS grid, so say so
    (lambda: _xt(Signal(grid=Grid(fs=32e9, baud=4e9, n=4096)).carrier("nrz", n_ui=N_UI)),
     r"Signal on fs=3.2e\+10 Hz.*nothing is resampled"),
    (lambda: _xt(Signal(grid=Grid(fs=G.fs, baud=2e9, n=G.n)).carrier("nrz", n_ui=128)),
     r"baud=2e\+09.*'from'"),
    # a seed that would be ignored
    (lambda: _xt(Signal(seed=7, grid=G).carrier("nrz", n_ui=N_UI)),
     r"seed=7.*THIS record's seed \(1\).*'from'"),
    (lambda: _xt(Signal(grid=G, lead_in=64).carrier("nrz", n_ui=N_UI)), "lead-in.*'from'"),
])
def test_a_signal_that_cannot_become_part_of_this_record_is_refused_when_it_is_added(build, match):
    with pytest.raises(ValueError, match=match):
        build()


@pytest.mark.parametrize("aggressor,match", [
    # another RATE and another LENGTH, as an embedded record: never resampled, never cut
    ({"from": Signal(grid=Grid(fs=32e9, baud=4e9, n=4096)).carrier("nrz", n_ui=512).recipe()},
     r"fs=3.2e\+10 Hz, n=4096.*fs=6.4e\+10 Hz, n=4096.*never resampled"),
    ({"from": Signal(grid=Grid(fs=64e9, baud=4e9, n=8192)).carrier("nrz", n_ui=512).recipe()},
     "n=8192.*never resampled"),
    # the SAME sample count at another rate: an acquisition that lands on n by coincidence
    ({"ops": [{"op": "carrier", "kind": "nrz", "n_ui": N_UI},
              {"op": "acquire", "profile": {"sample_rate_hz": 32e9, "record_length": 4096}}]},
     r"3.2e\+10 Sa/s.*6.4e\+10 Sa/s.*different timebase"),
    ({"ops": [{"op": "carrier", "kind": "nrz", "n_ui": N_UI}, {"op": "digitize", "n_out": 2048}]},
     "rendered 2048 samples but this record has 4096.*SAME grid"),
    ({"ops": [{"op": "carrier", "kind": "nrz", "n_ui": N_UI, "n": 1000}]}, "rendered 1000"),
    # an optical field: its imaginary part is not dropped
    ({"ops": [{"op": "carrier", "kind": "nrz", "n_ui": N_UI}, {"op": "eo"}]},
     "COMPLEX.*`photodetect`"),
    # malformed
    ({"ops": []}, "empty `ops`"),
    ({"ops": [{"op": "carier", "kind": "nrz"}]}, "op 'carier'.*does not exist"),
    ({"ops": [{"op": "carrier", "kind": "nrz", "n_uu": 4}]}, "n_uu"),
    ({"ops": [{"op": "carrier", "kind": "nrz"}], "seed": 3}, r"takes only 'ops'.*\['seed'\]"),
    ({"ops": [{"op": "carrier", "kind": "nrz"}], "from": {"ops": [], "seed": 0}},
     "exactly one of"),
    ({"ops": [{"op": "carrier", "kind": "nrz"}], "kind": "nrz"}, "exactly one of"),
    ({"from": {"ops": [{"op": "carrier", "kind": "nrz"}]}}, "RECIPE"),
])
def test_a_sub_chain_that_does_not_fit_is_refused_by_name_never_fitted(aggressor, match):
    sig = Signal(seed=1, grid=G)
    sig.ops = [{"op": "carrier", "kind": "nrz", "n_ui": N_UI},
               {"op": "crosstalk", "aggressor": aggressor}]
    with pytest.raises((ValueError, TypeError), match=match):
        sig.waveform()


def test_a_path_in_a_hand_written_recipe_is_refused_because_the_digest_would_not_cover_it():
    sig = Signal(seed=1, grid=G)
    sig.ops = [{"op": "carrier", "kind": "nrz", "n_ui": N_UI},
               {"op": "crosstalk", "aggressor": {"from": "lane.json"}}]
    with pytest.raises(ValueError, match="from='lane.json'.*RECIPE.*json.load"):
        sig.waveform()


def test_crosstalk_matrix_refuses_what_only_describes_manufactured_aggressors():
    ln = lane()
    with pytest.raises(ValueError, match=r"\['seeds', 'synchronous'\].*MANUFACTURE"):
        victim().crosstalk_matrix(couplings=[0.1], aggressors=[ln], seeds=[3],
                                  synchronous=True).waveform()
    with pytest.raises(ValueError, match="2 aggressors for 1 couplings"):
        victim().crosstalk_matrix(couplings=[0.1], aggressors=[ln, ln]).waveform()
    with pytest.raises(ValueError, match=r"`aggressors`\[0\] needs exactly one of 'ops'"):
        victim().crosstalk_matrix(couplings=[0.1], aggressors=[dict(kind="nrz")]).waveform()
    with pytest.raises(ValueError, match="list of sub-chains"):
        victim().crosstalk_matrix(couplings=[0.1], aggressors={"ops": ln.ops}).waveform()


def test_a_lead_in_refuses_a_record_that_holds_a_sub_chain():
    sig = (Signal(seed=1, grid=G, lead_in=True).carrier("nrz", n_ui=N_UI)
           .crosstalk(aggressor=lane()))
    with pytest.raises(ValueError, match="lead_in: the sub-chain at 'crosstalk/1/aggressor'"):
        sig.waveform()
    # the shorthand a lead-in always accepted is untouched
    Signal(seed=1, grid=G, lead_in=64).carrier("nrz", n_ui=N_UI).crosstalk().waveform()


# ------------------------------------------------------------------ the seam for a new op
def test_an_executor_takes_a_sub_chain_in_one_line():
    """What a later op writes. No registration: the namespace is <op>/<idx>/<parameter>."""
    p = {"op": "mixer", "lo": {"ops": lane().ops}}
    lo = SUB.signal(p, "lo", Streams(1), G, 3, n=G.n)
    assert np.array_equal(lo, namespaced(lane().ops, 1, "mixer/3/lo"))
    assert [at for _i, at, _l, _s in SUB.walk([{}, {}, {}, p])] == ["mixer/3/lo"]
    with pytest.raises(ValueError, match="mixer needs lo=<a sub-chain>"):
        SUB.signal({"op": "mixer"}, "lo", Streams(1), G, 3)
    with pytest.raises(ValueError, match="mixer: `lo` needs exactly one of 'ops'"):
        SUB.signal({"op": "mixer", "lo": {"kind": "sine"}}, "lo", Streams(1), G, 3)
    with pytest.raises(ValueError, match="mixer: `lo` is ndarray"):
        SUB.signal({"op": "mixer", "lo": np.zeros(4)}, "lo", Streams(1), G, 3)
    with pytest.raises(ValueError, match="there is none"):
        SUB.signal(p, "lo", Streams(1), None, 3)
