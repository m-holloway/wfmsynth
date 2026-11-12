"""GitHub #78: several correlated channels from one recipe.

`ws.Channels({name: chain}, shared={name: chain})` renders a dict of channels from one recipe,
and a chain refers to another by name with the one op `use`. What that has to mean is checked
against things that do not go through it:

  * CORRELATION -- two different channels through one shared timing source are displaced by the
    same samples, recovered from each channel independently; and the same two channels with a
    timing source each are not;
  * ONE FACTOR -- re-rolling the shared role moves every channel that uses it and no other,
    re-rolling a private role moves that channel alone;
  * THE SAME SIGNAL -- a chain used from several places is one waveform: a channel used as an
    aggressor equals the #75 sub-chain build of the same lane, sample for sample;
  * REPLAY -- the record replays bit-identically from its JSON, under one digest;
  * REFUSALS -- a cycle, an unknown name, a stage with nothing to act on, a signal that would
    discard what arrived, a chain with its own seed/grid/lead-in: each by name.
"""
from __future__ import annotations

import json
import warnings

import numpy as np
import pytest

import wfmsynth as ws
from wfmsynth import Grid, Signal

SPB, N_UI = 32, 512
G = Grid(fs=SPB * 10e9, baud=10e9, n=SPB * N_UI)
RAMP = np.arange(G.n, dtype=float)


@pytest.fixture(autouse=True)
def _quiet():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


def _pll():
    return Signal().timing(phase_noise=dict(rms_ps=2.0))


def _clk():
    return Signal().carrier("square", f_hz=5e9, tr_frac=0.3)


def _data():
    return Signal().carrier("nrz", n_ui=N_UI, pattern="prbs7", tr_frac=0.3)


def _crossings(y):
    s = np.signbit(y)
    i = np.nonzero(s[1:] != s[:-1])[0]
    return i + y[i] / (y[i] - y[i + 1])


def _tie(ideal, jittered):
    a, b = _crossings(ideal), _crossings(jittered)
    assert len(a) == len(b)
    return a, b - a


def _common(tc, jc, td, jd):
    """TIE of the clock and of the data at the edges they share (within one sample)."""
    k = np.abs(tc[None, :] - td[:, None]).argmin(axis=1)
    keep = np.abs(tc[k] - td) < 1.0
    return jc[k][keep], jd[keep]


def _shared_record():
    return ws.Channels({"clk": _clk().use("pll"), "data": _data().use("pll"),
                        "ramp": Signal().capture(values=RAMP).use("pll"),
                        "ramp3": Signal().capture(values=3.0 * RAMP + 7.0).use("pll")},
                       shared={"pll": _pll()}, seed=1, grid=G)


# ------------------------------------------------------------------ correlation by construction
def test_two_channels_through_one_timing_source_are_displaced_identically():
    """A linear ramp through a timing warp reads back the displacement itself, so two ramps of
    different slope and offset recover the shared source twice, independently."""
    x = _shared_record().waveforms()
    d1 = RAMP - x["ramp"]
    d3 = RAMP - (x["ramp3"] - 7.0) / 3.0
    core = slice(8, -8)                                    # the warp clamps at the record ends
    rms_ps = d1[core].std() / G.fs * 1e12
    print(f"displacement {rms_ps:.4f} ps rms (asked 2.0); the two channels differ by "
          f"{np.abs(d1 - d3)[core].max():.2e} samples")
    assert abs(rms_ps - 2.0) < 0.02
    assert np.abs(d1 - d3)[core].max() < 1e-10             # of a 0.64-sample displacement


def test_clock_and_data_sharing_a_timing_source_show_the_same_jitter():
    ideal = ws.Channels({"clk": _clk(), "data": _data()}, seed=1, grid=G).waveforms()
    x = _shared_record().waveforms()
    tc, jc = _tie(ideal["clk"], x["clk"])
    td, jd = _tie(ideal["data"], x["data"])
    a, b = _common(tc, jc, td, jd)
    disp = RAMP - x["ramp"]                                # the displacement that was applied
    err_c = np.abs(jc - np.interp(tc + jc, RAMP, disp)).max()   # y(t) = x(t - d(t)): the
    err_d = np.abs(jd - np.interp(td + jd, RAMP, disp)).max()   # edge lands where d is read
    print(f"{len(a)} shared edges: TIE rms {a.std():.3f} samples, max |clk - data| "
          f"{np.abs(a - b).max():.3f}, correlation {np.corrcoef(a, b)[0, 1]:.6f}; against the "
          f"applied displacement clk {err_c:.3f}, data {err_d:.3f}")
    assert len(a) > 200
    assert np.corrcoef(a, b)[0, 1] > 0.9995
    # each channel's edges sit on the applied displacement; clock and data differ only
    # because their edges cross a fraction of a sample apart, where the source has moved on
    assert max(err_c, err_d) < 0.01 * a.std()
    assert np.abs(a - b).max() < 0.1 * a.std()


def test_a_timing_source_each_is_not_correlated_by_accident():
    """The control: the same two chains, each with its own `timing`, do not agree."""
    ideal = ws.Channels({"clk": _clk(), "data": _data()}, seed=1, grid=G).waveforms()
    each = ws.Channels({"clk": _clk().timing(phase_noise=dict(rms_ps=2.0)),
                        "data": _data().timing(phase_noise=dict(rms_ps=2.0))}, seed=1, grid=G)
    assert each.roles() == ["clk/timing/1", "data/timing/1"]
    x = each.waveforms()
    a, b = _common(*_tie(ideal["clk"], x["clk"]), *_tie(ideal["data"], x["data"]))
    print(f"own sources: max |clk - data| {np.abs(a - b).max():.2f} samples of {a.std():.2f} rms")
    assert np.abs(a - b).max() > a.std()


def test_a_chain_draws_the_same_whichever_channels_use_it_and_in_any_order():
    full = _shared_record().waveforms()
    alone = ws.Channels({"data": _data().use("pll")}, shared={"pll": _pll()}, seed=1,
                        grid=G).waveforms()
    swapped = ws.Channels({"data": _data().use("pll"), "clk": _clk().use("pll")},
                          shared={"pll": _pll()}, seed=1, grid=G).waveforms()
    assert np.array_equal(full["data"], alone["data"])
    assert np.array_equal(full["data"], swapped["data"])
    assert np.array_equal(full["clk"], swapped["clk"])
    assert list(swapped) == ["data", "clk"]                # returned in the order given


# ------------------------------------------------------------------ one factor at a time
def test_rerolling_the_shared_role_moves_every_channel_that_uses_it_and_no_other():
    rec = ws.Channels({"clk": _clk().use("pll"), "data": _data().use("pll"),
                       "quiet": _data()}, shared={"pll": _pll()}, seed=1, grid=G)
    assert rec.roles() == ["pll/timing/0"]
    x, y = rec.waveforms(), rec.contrast("pll/timing/0")
    moved = {k: float(np.abs(x[k] - y[k]).max()) for k in x}
    print("re-rolled pll/timing/0:", moved)
    assert moved["clk"] > 0.1 and moved["data"] > 0.1 and moved["quiet"] == 0.0
    # and the re-rolled pair is still one timing source: both follow the SAME new displacement
    again = rec.contrast("pll/timing/0")
    assert all(np.array_equal(y[k], again[k]) for k in y)
    other = rec.contrast("pll/timing/0", seed=77)
    assert not np.array_equal(other["clk"], y["clk"])


def test_rerolling_a_private_role_moves_that_channel_alone():
    rec = ws.Channels({"a": _data().use("pll").digitize(noise_rms=0.01),
                       "b": _data().use("pll").digitize(noise_rms=0.01)},
                      shared={"pll": _pll()}, seed=3, grid=G)
    assert rec.roles() == ["a/noise/2", "b/noise/2", "pll/timing/0"]
    x, y = rec.waveforms(), rec.contrast("a/noise/2")
    assert not np.array_equal(x["a"], y["a"]) and np.array_equal(x["b"], y["b"])
    # same chain, same seed, different NAME: the two channels' noise is independent
    assert abs(np.corrcoef(x["a"] - x["b"], x["a"])[0, 1]) < 0.9
    assert not np.array_equal(x["a"], x["b"])


def test_contrast_refuses_a_role_that_names_no_chain():
    rec = _shared_record()
    with pytest.raises(ValueError, match="names no chain.*pll/timing/0"):
        rec.contrast("timing/0")


# ------------------------------------------------------------------ one signal, used twice
def test_the_two_legs_of_a_pair_are_one_bit_stream():
    rec = ws.Channels({"p": Signal().carrier("nrz", n_ui=N_UI, jitter=dict(rj=0.3)),
                       "n": Signal().use("p").law("poly", coeffs=[0.0, -1.0])}, seed=5, grid=G)
    x = rec.waveforms()
    assert rec.roles() == ["p/jitter/0"]
    assert np.array_equal(x["n"], -x["p"])                 # the common mode is exactly zero
    y = rec.contrast("p/jitter/0")
    assert not np.array_equal(y["p"], x["p"]) and np.array_equal(y["n"], -y["p"])


def test_a_channel_used_as_an_aggressor_is_that_channel():
    """Against the #75 route: the same lane written as a sub-chain of one Signal."""
    def lane():
        return Signal().carrier("nrz", n_ui=N_UI, pattern="prbs9", seed=9).lossy(
            loss_db=6.0, loss_at_ghz=5.0)

    rec = ws.Channels({"victim": _data().crosstalk(aggressor=Signal().use("lane"),
                                                   coupling=0.1),
                       "lane": lane()}, seed=1, grid=G)
    x = rec.waveforms()
    ref = Signal(seed=1, grid=G)
    ref.ops = _data().crosstalk(aggressor=lane(), coupling=0.1).ops
    assert np.array_equal(x["victim"], ref.waveform())
    assert np.array_equal(x["lane"], ref.subchain("crosstalk/1/aggressor"))
    assert not np.array_equal(x["victim"], ws.Channels({"v": _data()}, seed=1,
                                                       grid=G).waveforms()["v"])


def test_two_lanes_that_are_each_others_crosstalk():
    shared = {"tx0": Signal().carrier("nrz", n_ui=N_UI, pattern="prbs7", seed=1),
              "tx1": Signal().carrier("nrz", n_ui=N_UI, pattern="prbs9", seed=2)}
    rec = ws.Channels(
        {"lane0": Signal().use("tx0").crosstalk(aggressor=Signal().use("tx1"), coupling=0.1),
         "lane1": Signal().use("tx1").crosstalk(aggressor=Signal().use("tx0"), coupling=0.1)},
        shared=shared, seed=1, grid=G)
    x = rec.waveforms()
    tx0, tx1 = rec.waveform("tx0"), rec.waveform("tx1")
    # what reached lane0 beyond its own launch is linear in what lane1 LAUNCHED, and vice versa
    doubled = ws.Channels(
        {"lane0": Signal().use("tx0").crosstalk(aggressor=Signal().use("tx1"), coupling=0.2)},
        shared=shared, seed=1, grid=G).waveforms()["lane0"]
    c0, c1 = x["lane0"] - tx0, x["lane1"] - tx1
    assert np.abs(c0).max() > 1e-3 and np.abs(c1).max() > 1e-3
    assert np.allclose(doubled - tx0, 2.0 * c0, rtol=0, atol=1e-12)
    assert abs(np.corrcoef(c0, c1)[0, 1]) < 0.5            # each carries the OTHER's pattern


def test_a_shared_signal_is_ground_truth_and_a_stage_has_no_waveform():
    rec = _shared_record()
    with pytest.raises(ValueError, match="'pll' is a stage.*`timing`"):
        rec.waveform("pll")
    with pytest.raises(ValueError, match="no chain 'nope'.*clk"):
        rec.waveform("nope")
    assert np.array_equal(rec.waveform("data"), rec.waveforms()["data"])


def test_use_inside_a_sub_chain_reaches_the_record_and_not_a_pinned_one():
    pinned = Signal(seed=1, grid=G)
    pinned.ops = Signal().use("pll").ops
    bad = ws.Channels({"v": _data().crosstalk(aggressor={"from": {"seed": 1, "ops": [
        {"op": "carrier", "kind": "nrz", "n_ui": N_UI}, {"op": "use", "name": "pll"}],
        "grid": dict(fs=G.fs, baud=G.baud, n=G.n)}}, coupling=0.1)},
        shared={"pll": _pll()}, seed=1, grid=G)
    with pytest.raises(ValueError, match="rendered alone.*its own record"):
        bad.waveforms()
    with pytest.raises(ValueError, match="rendered alone"):
        pinned.waveform()


# ------------------------------------------------------------------ replay, digest, summary
def test_a_two_channel_record_replays_bit_identically_from_its_recipe():
    rec = ws.Channels({"clk": _clk().use("pll"),
                       "data": _data().use("pll").lossy(loss_db=6.0, loss_at_ghz=5.0)
                       .digitize(noise_rms=0.01)},
                      shared={"pll": _pll()}, seed=11, grid=G)
    x = rec.waveforms()
    back = ws.Channels.from_recipe(json.loads(json.dumps(rec.recipe())))
    y = back.waveforms()
    assert list(y) == ["clk", "data"]
    assert all(np.array_equal(x[k], y[k]) for k in x)
    assert back.sha256() == rec.sha256() and back.recipe() == rec.recipe()
    r = rec.recipe()
    assert set(r) == {"wfmsynth_version", "seed", "grid", "channels", "shared"}
    assert r["channels"]["clk"] == {"ops": [{"op": "carrier", "kind": "square", "f_hz": 5e9,
                                             "tr_frac": 0.3}, {"op": "use", "name": "pll"}]}


def test_one_digest_covers_every_chain_and_the_seed():
    def rec(seed=1, rms=2.0, loss=6.0):
        return ws.Channels({"clk": _clk().use("pll"),
                            "data": _data().use("pll").lossy(loss_db=loss, loss_at_ghz=5.0)},
                           shared={"pll": Signal().timing(phase_noise=dict(rms_ps=rms))},
                           seed=seed, grid=G).sha256()
    assert rec() == rec()
    assert len({rec(), rec(seed=2), rec(rms=2.5), rec(loss=7.0)}) == 4


def test_a_recipe_goes_to_the_class_that_renders_it():
    rec = _shared_record()
    with pytest.raises(ValueError, match="several channels.*Channels.from_recipe"):
        Signal.from_recipe(rec.recipe())
    with pytest.raises(ValueError, match="no `channels`.*Signal.from_recipe"):
        ws.Channels.from_recipe(Signal(seed=1, grid=G).carrier("nrz", n_ui=N_UI).recipe())


def test_summary_names_every_chain():
    text = ws.Channels({"clk": _clk().use("pll"), "data": _data().use("pll")},
                       shared={"pll": _pll()}, seed=1, grid=G).summary()
    print(text)
    assert text.splitlines()[0].startswith("Channels  seed=1  fs=3.2e+11 Hz")
    for line in ("  channel [clk]", "  channel [data]", "  shared [pll]",
                 "    1  use      name='pll'", "    0  timing  phase_noise={'rms_ps': 2.0}"):
        assert line in text.splitlines()


# ------------------------------------------------------------------ events and acquisition
def test_events_come_back_per_channel_and_a_shared_chains_events_say_where():
    burst = Signal().carrier("nrz", n_ui=N_UI).events("glitch", count=3)
    rec = ws.Channels({"a": Signal().use("burst"), "b": _data().events("glitch", count=2)},
                      shared={"burst": burst}, seed=2, grid=G)
    x, ev = rec.realize()
    assert set(ev) == {"a", "b"} and all(np.array_equal(x[k], rec.waveforms()[k]) for k in x)
    assert len(ev["a"]) > 0 and {e.at for e in ev["a"]} == {"burst"}
    assert len(ev["b"]) > 0 and {e.at for e in ev["b"]} == {""}
    assert rec.roles() == ["b/events/1", "burst/events/1"]


def test_each_channel_states_the_grid_it_was_stored_on():
    prof = {"sample_rate_hz": G.fs / 4, "record_length": G.n // 4}
    rec = ws.Channels({"fast": _data(), "slow": Signal().use("tx").acquire(prof)},
                      shared={"tx": _data()}, seed=1, grid=G)
    x = rec.waveforms()
    assert rec.stored_grid("fast", x["fast"]).fs == G.fs
    slow = rec.stored_grid("slow", x["slow"])
    assert (slow.fs, slow.n) == (G.fs / 4, G.n // 4)


def test_hdf5_holds_every_channel_its_name_and_the_one_recipe(tmp_path):
    pytest.importorskip("h5py")
    from wfmsynth import hdf5 as H5
    rec = ws.Channels({"clk": _clk().use("pll"), "data": _data().use("pll")},
                      shared={"pll": _pll()}, seed=1, grid=G)
    p = rec.write_hdf5(tmp_path / "pair.h5", date="01-Jan-2026 00:00:00")
    assert H5.channel_names(p) == {1: "clk", 2: "data"}
    recipe = H5.read_recipe(p)
    assert recipe == rec.recipe()
    x = ws.Channels.from_recipe(recipe).waveforms()
    stored = H5.read_hdf5(p)
    for n, name in H5.channel_names(p).items():
        volts, t = stored[n]
        step = np.ptp(x[name]) / 65534
        err = np.abs(volts - x[name]).max()
        print(f"{name}: replayed from the file's recipe to {err:.2e} V (half a code "
              f"{step / 2:.2e} V), dt {t[1] - t[0]:.4e} s")
        assert err <= 0.51 * step and abs((t[1] - t[0]) * G.fs - 1) < 1e-12
    prof = {"sample_rate_hz": G.fs / 4, "record_length": G.n // 4}
    two = ws.Channels({"a": _data(), "b": _data().acquire(prof)}, seed=1, grid=G)
    with pytest.raises(ValueError, match="different rates.*one time axis"):
        two.write_hdf5(tmp_path / "no.h5")
    with pytest.raises(ValueError, match="carries no recipe"):
        H5.read_recipe(H5.write_hdf5(tmp_path / "bare.h5", x["clk"], fs=G.fs))


# ------------------------------------------------------------------ refusals
def test_a_cycle_is_refused_naming_it():
    with pytest.raises(ValueError, match="lane0 -> lane1 -> lane0 is a cycle.*LAUNCHED"):
        ws.Channels(
            {"lane0": _data().crosstalk(aggressor=Signal().use("lane1"), coupling=0.1),
             "lane1": _data().crosstalk(aggressor=Signal().use("lane0"), coupling=0.1)},
            seed=1, grid=G)
    with pytest.raises(ValueError, match="a -> a is a cycle"):
        ws.Channels({"a": Signal().use("a")}, seed=1, grid=G)


@pytest.mark.parametrize("build, message", [
    (lambda: ws.Channels({"a": _data().use("pl")}, shared={"pll": _pll()}, grid=G),
     "chain 'a' uses 'pl', which is not a chain of this record. It has: .'a', 'pll'."),
    (lambda: ws.Channels({"a": _pll()}, grid=G),
     "channel 'a' starts with `timing`.*belongs under `shared`"),
    (lambda: ws.Channels({"a": Signal().use("pll")}, shared={"pll": _pll()}, grid=G),
     "channel 'a' starts with `use`"),
    (lambda: ws.Channels({"a": _data()}, shared={"a": _pll()}, grid=G),
     "both `channels` and `shared`"),
    (lambda: ws.Channels({}, grid=G), "non-empty"),
    (lambda: ws.Channels({"a/b": _data()}, grid=G), "without '/'"),
    (lambda: ws.Channels({"a": _data()}, shared={"": _pll()}, grid=G), "shared chain name"),
    (lambda: ws.Channels({"a": Signal(seed=4).carrier("nrz", n_ui=8)}, seed=1, grid=G),
     "seed=4.*RECORD's seed .1."),
    (lambda: ws.Channels({"a": Signal(grid=Grid(fs=1e9, n=64)).carrier("sine", f_hz=1e6)},
                         grid=G), "its own grid"),
    (lambda: ws.Channels({"a": _data().with_lead_in(64)}, grid=G), "lead-in"),
    (lambda: ws.Channels({"a": {"ops": []}}, grid=G), "no ops"),
    (lambda: ws.Channels({"a": {"ops": [{"op": "carier"}]}}, grid=G), "'carier'.*does not exist"),
    (lambda: ws.Channels({"a": np.zeros(4)}, grid=G), "a chain is a Signal"),
])
def test_what_a_record_cannot_carry_is_refused_at_construction(build, message):
    with pytest.raises(ValueError, match=message):
        build()


def test_use_says_which_of_its_two_meanings_does_not_fit():
    whole = ws.Channels({"a": _data().use("b"), "b": _data()}, seed=1, grid=G)
    with pytest.raises(ValueError, match="'b' is a whole signal .*`carrier`.*discard"):
        whole.waveforms()
    stage = ws.Channels({"a": _data().crosstalk(aggressor=Signal().use("pll"), coupling=0.1)},
                        shared={"pll": _pll()}, seed=1, grid=G)
    with pytest.raises(ValueError, match="'pll' is a stage .*`timing`.*Put a source"):
        stage.waveforms()
    with pytest.raises(ValueError, match="rendered alone"):
        Signal(grid=G).carrier("nrz", n_ui=N_UI).use("pll").waveform()


def test_use_is_classified_everywhere_an_op_must_be():
    with pytest.raises(ValueError, match="lead_in: the 'use' op cannot"):
        Signal(grid=G, lead_in=64).carrier("nrz", n_ui=N_UI).use("pll").waveform()
    with pytest.raises(ValueError, match="'use' is a general stage"):
        Signal(grid=G).carrier("nrz", n_ui=N_UI).use("pll").canonical()
    with pytest.raises(TypeError, match="did you mean 'name'"):
        ws.Channels({"a": {"ops": [{"op": "carrier", "kind": "nrz"},
                                   {"op": "use", "nam": "pll"}]}}, grid=G)


# ------------------------------------------------------------------ nothing else moved
def test_a_random_timing_source_is_a_listed_role():
    sig = Signal(seed=3, grid=G).carrier("nrz", n_ui=N_UI).timing(rj_ps=0.5)
    assert sig.roles() == ["timing/1"]
    assert Signal(grid=G).carrier("nrz", n_ui=N_UI).timing(
        pj=dict(amp_ps=1.0, f_hz=1e8)).roles() == []       # deterministic: nothing to re-roll
    x = sig.waveform()
    assert not np.array_equal(sig.contrast("timing/1"), x)


# ------------------------------------------------- a named chain inside the other composition ops
def test_a_named_chain_is_a_segment_a_mux_input_and_the_signal_a_join_takes():
    """`use` is a sub-chain like any other, so every op that takes one takes a named chain:
    `sequence` and `switch` (written, like every chain of the record, with no grid of their
    own), `combine`, and the `by=` of a signal-driven parameter. Each is checked against the
    named chains' own waveforms, combined here by hand."""
    n, k = G.n, G.n // 4
    f = G.fs / 64                                           # a whole number of cycles in the record
    rec = ws.Channels(
        {"line": Signal().sequence([{"chain": Signal().use("pre"), "t_s": k / G.fs},
                                    {"chain": Signal().use("data")}]),
         "mux": Signal().switch(select=Signal().use("en"),
                                inputs=[Signal().use("pre"), Signal().use("data")]),
         "sum": Signal().use("data").combine(Signal().use("pre")).gate(by=Signal().use("en"))},
        shared={"pre": Signal().carrier("sine", f_hz=f),
                "data": Signal().carrier("nrz", n_ui=N_UI, jitter=dict(rj=0.3)),
                "en": Signal().carrier("square", f_hz=f / 8, amp=0.5, offset=0.5)},
        seed=4, grid=G)
    x, events = rec.realize()
    pre, data, en = (rec.waveform(name) for name in ("pre", "data", "en"))
    assert np.array_equal(x["line"], np.concatenate([pre[:k], data[:n - k]]))
    assert np.array_equal(x["mux"], np.where(en > 0.5, data, pre))
    assert np.array_equal(x["sum"], np.where(en > 0.5, data + pre, 0.0))
    assert [e.kind for e in events["line"].on_record()] == ["segment"]
    n_changes = int(np.count_nonzero(np.diff(en > 0.5)))
    assert n_changes > 4
    assert [e.kind for e in events["mux"].on_record()] == ["switch"] * n_changes
    assert [e.kind for e in events["sum"].on_record()] == ["transition"] * n_changes
    # one realisation of the data, wherever it is used: re-rolling it moves all three
    y = rec.contrast("data/jitter/0")
    assert all(not np.array_equal(y[name], x[name]) for name in x)
    back = ws.Channels.from_recipe(json.loads(rec.to_json()))
    assert back.sha256() == rec.sha256()
    assert all(np.array_equal(back.waveforms()[name], x[name]) for name in x)


def test_one_trigger_shared_by_two_channels_fires_with_one_jitter():
    """A `trigger` in a shared stage is one scope's trigger circuit: each channel is cut where
    its own waveform crosses, and the trigger's jitter -- one role, `scope/trigger/0` -- is the
    same draw on both."""
    f = G.baud / 8
    rec = ws.Channels(
        {"a": Signal().carrier("sine", f_hz=f).use("scope"),
         "b": Signal().carrier("sine", f_hz=f, amp=0.5).use("scope")},
        shared={"scope": Signal().trigger(pre_s=4 / f, post_s=4 / f, level=0.0,
                                          jitter_s=2.0 / G.fs)},
        seed=2, grid=G)
    x, events = rec.realize()
    assert rec.roles() == ["scope/trigger/0"]
    drawn = {name: [e.params["jitter_s"] for e in events[name] if e.kind == "trigger"]
             for name in x}
    assert len(drawn["a"]) == 1 and drawn["a"] == drawn["b"] and drawn["a"][0] != 0.0
    assert len(x["a"]) == len(x["b"]) == rec.stored_grid("a", x["a"]).n == round(8 / f * G.fs)
    assert np.allclose(x["b"], 0.5 * x["a"], atol=1e-12)    # the same window of the same tone
    y = rec.contrast("scope/trigger/0")
    assert all(not np.array_equal(y[name], x[name]) for name in x)


# ---------------------------------------------------------------- the seams with other pieces
def test_a_trigger_taken_from_another_channel_cuts_both_at_the_same_instants():
    """`trigger(on=<clock>)` is the scope's trigger input being another channel. Checked
    against slices of the untriggered record taken at the CLOCK's own crossings, found
    independently by `trigger.find`; and against triggering each channel on itself, which
    lands the data somewhere else."""
    lanes = {"clk_line": Signal().carrier("square", f_hz=G.baud / 2).lossy(loss_db=2.0,
                                                                         loss_at_ghz=5.0),
             "tx": Signal().carrier("nrz", n_ui=N_UI, seed=5).lossy(loss_db=4.0, loss_at_ghz=5.0)}
    pre, post = 0.5 / G.baud, 3.5 / G.baud

    def record(**trig):
        return ws.Channels(
            {"clk": Signal().use("clk_line").use("scope"),
             "data": Signal().use("tx").events("glitch", count=40).use("scope")},
            shared=dict(lanes, scope=Signal().trigger(pre_s=pre, post_s=post, level=0.0,
                                                      n_segments=6, holdoff_s=40 / G.baud,
                                                      **trig)), seed=3, grid=G)

    rec = record(on=Signal().use("clk_line"))
    x, events = rec.realize()
    stamps = {name: [e.params["timestamp_s"] for e in events[name] if e.kind == "trigger"]
              for name in x}
    assert len(stamps["clk"]) == 6 and stamps["clk"] == stamps["data"]

    plain = ws.Channels({"clk": Signal().use("clk_line"),
                         "data": Signal().use("tx").events("glitch", count=40)},
                        shared=lanes, seed=3, grid=G)
    long, placed = plain.realize()
    from wfmsynth import trigger as TRIG
    fires = TRIG.find(long["clk"], G.fs, level=0.0)["t_s"]
    assert set(np.round(np.array(stamps["clk"]) * G.fs, 6)) <= set(np.round(fires * G.fs, 6))
    n_win = round((pre + post) * G.fs)
    starts = [e.params["start"] for e in events["data"] if e.kind == "trigger"]
    for name in x:                                # every window is the long render, unresampled
        for k, a in enumerate(starts):
            assert np.array_equal(x[name][k * n_win:(k + 1) * n_win], long[name][a:a + n_win])
    # the clock's crossing sits at pre_s in BOTH channels' windows, to half a sample
    for e in events["data"]:
        if e.kind == "trigger":
            assert abs(e.t_s - e.params["segment"] * n_win / G.fs - pre) * G.fs <= 0.5

    # the glitches placed before the scope are re-timed into the windows or dropped with them
    glitches = [e for e in placed["data"] if e.kind == "glitch"]
    kept = [e for e in events["data"] if e.kind == "glitch"]
    inside = [e for e in glitches if any(a <= e.sample < a + n_win for a in starts)]
    assert len(glitches) > len(kept) == len(inside) > 0
    assert all(0 <= e.sample < len(x["data"]) for e in events["data"])
    dropped = {e.params["dropped"] for e in events["data"] if e.kind == "trigger"}
    assert dropped == {len(glitches) - len(kept)}

    own = record().waveforms()                    # each channel triggering on itself
    assert np.array_equal(own["clk"], x["clk"]) and not np.array_equal(own["data"], x["data"])
    again = ws.Channels.from_recipe(json.loads(rec.to_json()))
    assert again.sha256() == rec.sha256()
    assert all(np.array_equal(again.waveforms()[k], x[k]) for k in x)


def test_a_single_signal_triggers_on_a_sub_chain_too():
    clock = Signal(grid=G).carrier("square", f_hz=G.baud / 4)
    sig = (Signal(seed=1, grid=G).carrier("nrz", n_ui=N_UI, seed=2)
           .trigger(on=clock, pre_s=1 / G.baud, post_s=3 / G.baud, level=0.0))
    x, ev = sig.realize()
    (e,) = [e for e in ev if e.kind == "trigger"]
    long = Signal(seed=1, grid=G).carrier("nrz", n_ui=N_UI, seed=2).waveform()
    assert np.array_equal(x, long[e.params["start"]:e.params["start"] + len(x)])
    edge = e.params["timestamp_s"] * G.fs         # the clock's first rising edge after pre_s
    assert abs(edge - 4 * SPB) < 1.0, edge        # one clock period = 4 UI
    assert "on  [trigger/1/on]" in sig.summary()
    with pytest.raises(ValueError, match="never resampled"):
        short = Signal(grid=Grid(fs=G.fs, baud=G.baud, n=G.n // 2)).carrier("sine", f_hz=1e9)
        (Signal(seed=1, grid=G).carrier("nrz", n_ui=N_UI)
         .trigger(on={"from": short.recipe()}, pre_s=1e-10, post_s=1e-10).waveform())


def test_crosstalk_matrix_couples_the_real_neighbour_channels():
    """`crosstalk_matrix(aggressors=[...])` with each aggressor a NAMED lane of the record:
    the victim holds exactly the coupling of the waveforms the record returns for those lanes
    (`physics.crosstalk_matrix` on the arrays), and re-rolling one neighbour's jitter moves
    that neighbour and the victim and nothing else."""
    from wfmsynth import physics as P

    def tx(seed):
        return Signal().carrier("nrz", n_ui=N_UI, seed=seed, jitter=dict(rj=0.02))

    couplings = [0.08, 0.03]
    rec = ws.Channels(
        {"lane0": Signal().use("tx0").crosstalk_matrix(
            couplings=couplings, aggressors=[Signal().use("tx1"), Signal().use("tx2")]),
         "lane1": Signal().use("tx1"), "lane2": Signal().use("tx2")},
        shared={"tx0": tx(1), "tx1": tx(2), "tx2": tx(3)}, seed=4, grid=G)
    x = rec.waveforms()
    want = P.crosstalk_matrix(rec.waveform("tx0"), G, couplings,
                              aggressors=[x["lane1"], x["lane2"]])
    assert np.array_equal(x["lane0"], want)
    coupled = x["lane0"] - rec.waveform("tx0")
    assert np.abs(coupled).max() > 0.01           # and it is there
    y = rec.contrast("tx2/jitter/0")
    assert np.array_equal(y["lane1"], x["lane1"])
    assert not np.array_equal(y["lane2"], x["lane2"])
    assert not np.array_equal(y["lane0"], x["lane0"])
    assert np.array_equal(y["lane0"], P.crosstalk_matrix(
        rec.waveform("tx0"), G, couplings, aggressors=[x["lane1"], y["lane2"]]))


def test_a_toleranced_part_is_drawn_per_chain_and_a_group_is_one_part_across_the_record():
    """`ws.tol` in a chain of a record: the drawn value is in the recipe and in the waveform
    (the RC rise it sets, measured), an ungrouped part is its chain's own, a `group` is one
    draw for every chain that names it, and `unit(seed)` is another unit under the same
    noise."""
    g = Grid(fs=20e9, n=1 << 13)

    def lane(group=None):
        return (Signal().carrier("cmos", v_lo=0.0, v_hi=3.3, f_hz=5e6, tr_s=0.2e-9)
                .probe(r_source=ws.tol(1e3, 20, group=group), c_load_f=2e-12))

    def rise(x):                                   # 10-90 % of the first rising edge, seconds
        lo, hi = 0.33, 2.97
        a = np.flatnonzero((x[:-1] < lo) & (x[1:] >= lo))[0]
        b = a + np.flatnonzero(x[a:] >= hi)[0]
        ta = a + (lo - x[a]) / (x[a + 1] - x[a])
        tb = b - 1 + (hi - x[b - 1]) / (x[b] - x[b - 1])
        return (tb - ta) / g.fs

    rec = ws.Channels({"a": lane(), "b": lane(), "c": lane("pack"), "d": lane("pack")},
                      seed=7, grid=g)
    assert rec.roles() == ["a/unit/1/r_source", "b/unit/1/r_source", "unit/pack"]
    r = {k: rec.chains[k][1]["r_source"] for k in rec.names}
    assert all(800.0 <= v <= 1200.0 for v in r.values())
    assert r["a"] != r["b"] and r["c"] == r["d"] and r["c"] not in (r["a"], r["b"])
    x = rec.waveforms()
    for k in rec.names:                            # the drawn value is the one in the waveform
        want = np.log(9.0) * r[k] * 2e-12
        assert abs(rise(x[k]) / want - 1) < 5e-3, (k, rise(x[k]), want)

    again = ws.Channels.from_recipe(json.loads(rec.to_json()))
    assert again.sha256() == rec.sha256() and again.roles() == rec.roles()
    assert all(np.array_equal(again.waveforms()[k], x[k]) for k in x)

    other = rec.unit(8)
    r2 = {k: other.chains[k][1]["r_source"] for k in rec.names}
    assert all(r2[k] != r[k] for k in r) and r2["c"] == r2["d"] and other.seed == rec.seed
    y = other.waveforms()
    assert all(abs(rise(y[k]) / (np.log(9.0) * r2[k] * 2e-12) - 1) < 5e-3 for k in y)
    assert np.array_equal(rec.contrast("unit/*", seed=8)["a"], y["a"])
    with pytest.raises(ValueError, match="another unit is another record"):
        rec.contrast("a/unit/1/r_source")
    with pytest.raises(ValueError, match="no toleranced parameter"):
        ws.Channels({"a": _data()}, grid=G).unit(3)

    # the labels, from the recipe alone
    rows = ws.tolerance.resolved(rec.recipe())
    assert [(q["chain"], q["role"], q["value"]) for q in rows] == [
        ("a", "a/unit/1/r_source", r["a"]), ("b", "b/unit/1/r_source", r["b"]),
        ("c", "unit/pack", r["c"]), ("d", "unit/pack", r["d"])]
    assert ws.unit_of(rec.recipe()) == 7 and ws.unit_of(other.recipe()) == 8

    # a tolerance written in a recipe by hand is drawn the same way
    by_hand = ws.Channels({"a": {"ops": [dict(o) for o in lane().ops[:1]] + [
        {"op": "probe", "r_source": {"nom": 1e3, "tol_pct": 20}, "c_load_f": 2e-12}]}},
        seed=7, grid=g)
    assert by_hand.chains["a"][1]["r_source"] == r["a"]


def test_a_units_noise_is_the_same_noise():
    """`unit()` moves the part and nothing else: the receiver noise of the other unit is the
    same record of noise, to rounding, although the edge under it moved."""
    def build():
        return ws.Channels(
            {"a": Signal().carrier("nrz", n_ui=N_UI).lossy(loss_db=ws.tol(6.0, 30),
                                                           loss_at_ghz=5.0).rx_noise(rms=0.02)},
            seed=5, grid=G)
    rec = build()
    assert rec.roles() == ["a/rx_noise/2", "a/unit/1/loss_db"]
    quiet = {s: ws.Channels({"a": {"ops": r.chains["a"][:2]}}, seed=5, grid=G).waveforms()["a"]
             for s, r in (("this", rec), ("other", rec.unit(6)))}
    n_this = rec.waveforms()["a"] - quiet["this"]
    n_other = rec.unit(6).waveforms()["a"] - quiet["other"]
    assert np.abs(quiet["this"] - quiet["other"]).max() > 0.01       # the unit moved
    assert abs(n_this.std() / 0.02 - 1) < 0.05
    assert np.abs(n_this - n_other).max() < 1e-12                    # its noise did not


def test_summary_reads_as_a_tree_with_segments_parts_and_short_numbers():
    tx = Signal().sequence([
        {"chain": Signal().carrier("square", f_hz=5e9), "t_s": 64 / G.baud},
        {"chain": Signal().carrier("nrz", n_ui=N_UI), "state": "continue"}])
    rec = ws.Channels(
        {"data": Signal().use("tx").lossy(loss_db=ws.tol(6.0, 15, group="board"),
                                          loss_at_ghz=5.0).use("scope")},
        shared={"tx": tx, "scope": Signal().trigger(pre_s=0.1e-9 + 0.2e-9, post_s=1e-9)},
        seed=1, grid=G)
    loss = rec.chains["data"][1]["loss_db"]
    assert rec.summary() == f"""\
Channels  seed=1  fs=3.2e+11 Hz, baud=1e+10, n=16384
  channel [data]
    0  use    name='tx'
    1  lossy  loss_db={loss:.6g} (unit/board), loss_at_ghz=5.0
    2  use    name='scope'
  shared [tx]
    0  sequence  join='cut'
         segments[0].chain  t_s=6.4e-09  [tx/sequence/0/segments/0/chain]
           0  carrier  kind='square', f_hz=5e+09
         segments[1].chain  state='continue'  [tx/sequence/0/segments/1/chain]
           0  carrier  kind='nrz', n_ui=512
  shared [scope]
    0  trigger  pre_s=3e-10, post_s=1e-09"""
    one = Signal(seed=2, grid=G).carrier("nrz", n_ui=N_UI).lossy(loss_db=ws.tol(6.0, 15),
                                                                loss_at_ghz=5.0)
    assert "(unit/1/loss_db)" in one.summary()


def test_summary_names_a_chain_used_as_a_parameter_on_one_line():
    rec = ws.Channels({"a": _data().gain(by=Signal().use("vdd"))},
                      shared={"vdd": Signal().carrier("sine", f_hz=1e6, amp=0.1, offset=1.0)},
                      seed=1, grid=G)
    lines = rec.summary().splitlines()
    at = next(i for i, ln in enumerate(lines) if ln.split()[:2] == ["1", "gain"])
    assert lines[at + 1].strip() == "by  = [vdd]" and lines[at + 2].strip() == "shared [vdd]"
