"""`roles()` lists exactly the randomness a record draws from.

Two directions, both checked against the generators themselves rather than against a list
written by hand. After a render, a role's generator has moved from its seeded state if and only
if something drew from it; so for each chain below

  * every role `roles()` lists was drawn from, `contrast()` accepts it, and re-rolling it
    changes samples -- a listed role that draws nothing would make `contrast()` hand back the
    same record, which reads as "this factor does not matter";
  * nothing was drawn under a role `roles()` does not list -- randomness that cannot be named
    cannot be held fixed or re-rolled.

Until 0.46.0 the roles that draw only under a condition were missing (`acquire`, `store`,
`nonlinearity`, the optical ops, a `noise` carrier, `symbols`/`coded` jitter, a bare carrier
spec's jitter in six ops), and two were listed that drew nothing (a `crosstalk` aggressor
without jitter, a carrier whose `jitter=` has no `rj`).
"""
import warnings

import numpy as np
import pytest

import wfmsynth as ws
from wfmsynth import AcquisitionProfile, Grid, Signal
from wfmsynth.streams import Streams

G = Grid(fs=64e9, baud=16e9, n=4096)
GS = Grid(fs=1e8, baud=1e6, n=4096)
GIQ = Grid(fs=64e6, baud=4e6, n=4096, domain="iq", fc_hz=1e9)
J = dict(rj=0.4)
NET = dict(nodes=["bus"], rails={"vdd": 3.3}, probe="bus",
           elements=[{"id": "rp", "kind": "r", "a": "bus", "b": "vdd", "ohm": 885.0},
                     {"id": "cb", "kind": "c", "a": "bus", "b": "gnd", "farad": 40e-12}])


def base():
    return Signal(seed=3, grid=G).carrier("nrz", n_ui=1024, pattern="prbs7", jitter=J)


def quiet():
    return Signal(seed=3, grid=G).carrier("nrz", n_ui=1024, pattern="prbs7")


def slow():
    return Signal(seed=2, grid=GS).carrier("nrz", n_ui=40)


def profile(**kw):
    return AcquisitionProfile(sample_rate_hz=32e9, record_length=2048,
                              input_bandwidth_hz=13e9, **kw)


# name -> (builder, the roles it must list). An empty list is a chain that draws nothing.
CHAINS = {
    "no randomness": (lambda: quiet().lossy(loss_db=6, loss_at_ghz=8).scope(bw_hz=20e9), []),
    "carrier jitter": (base, ["jitter/0"]),
    "carrier dcd only, which is stated and not drawn":
        (lambda: Signal(seed=3, grid=G).carrier("nrz", n_ui=1024, jitter=dict(dcd=1.0)), []),
    "carrier noise": (lambda: Signal(seed=1, grid=G).carrier("noise", rms=0.3),
                      ["carrier_noise/0"]),
    "symbols jitter": (lambda: Signal(seed=7, grid=G).symbols([1.0, -1.0] * 512, jitter=J),
                       ["jitter/0"]),
    "coded jitter": (lambda: Signal(seed=7, grid=G).coded("64b66b", prbs=13, n_bits=1024,
                                                          jitter=J), ["jitter/0"]),
    "crosstalk default aggressor": (lambda: quiet().crosstalk(coupling=0.1), []),
    "crosstalk aggressor spec": (
        lambda: quiet().crosstalk(coupling=0.2, aggressor=dict(n_ui=512, seed=9, jitter=J)),
        ["jitter/xtalk1"]),
    "crosstalk sub-chain": (
        lambda: quiet().crosstalk(coupling=0.2, aggressor=Signal().carrier(
            "nrz", n_ui=1024, jitter=J).nonlinearity(level_noise=0.05)),
        ["crosstalk/1/aggressor/jitter/0", "crosstalk/1/aggressor/nl_noise/1"]),
    "modulate noise message": (
        lambda: Signal(seed=1, grid=G).carrier("sine", f_hz=4e9).modulate(
            message=dict(kind="noise", rms=0.2), depth=0.5), ["carrier_noise/modulate_msg1"]),
    "modulate nrz message": (
        lambda: Signal(seed=1, grid=G).carrier("sine", f_hz=4e9).modulate(
            kind="fsk", fc_hz=4e9, dev_hz=5e8, message=dict(kind="nrz", n_ui=64, jitter=J)),
        ["jitter/modulate_msg1"]),
    "open_drain second": (
        lambda: slow().open_drain(r_pullup_ohm=1e3, c_bus_f=2e-9,
                                  second=dict(n_ui=32, seed=9, jitter=J)),
        ["jitter/open_drain_second1"]),
    "hybrid_echo own": (
        lambda: quiet().hybrid_echo(isolation_db=20.0, td_samples=5,
                                    own=dict(n_ui=1024, seed=4, jitter=J)), ["jitter/echo1"]),
    "pass_fet gate": (
        lambda: Signal(seed=2, grid=GS).carrier("sine", f_hz=1e6).pass_fet(
            gate=dict(kind="nrz", n_ui=64, jitter=J)), ["jitter/pass_fet_gate1"]),
    "events on an aggressor": (
        lambda: quiet().events("glitch", on="aggressor", count=4,
                               aggressor=dict(n_ui=64, seed=2, jitter=J)),
        ["events/1", "jitter/events_aggr1"]),
    "events drawn": (lambda: quiet().events("runt", on="symbols", count=3), ["events/1"]),
    "events at stated symbols": (
        lambda: quiet().events("glitch", on="symbols", indices=[100, 200]), []),
    "events at stated times": (
        lambda: quiet().events("glitch", on="times", samples=[1000, 2000]), []),
    "net bare driver": (
        lambda: Signal(seed=5, grid=GS).net(**NET, drivers=[
            {"id": "tgt", "node": "bus", "ohm": 30.0, "state": "driving",
             "source": {"kind": "nrz", "n_ui": 64, "seed": 5, "jitter": J,
                        "volts": [0.0, 1.8]}}]), ["jitter/net/0/tgt/carrier"]),
    "net driver chain": (
        lambda: Signal(seed=5, grid=GS).net(**NET, drivers=[
            {"id": "host", "node": "bus", "ohm": 60.0, "state": "driving",
             "source": {"ops": Signal().carrier("nrz", n_ui=40, jitter=J),
                        "volts": [0.0, 3.3]}}]), ["net/0/host/jitter/0"]),
    "nonlinearity level noise": (lambda: quiet().nonlinearity(compression=0.1, level_noise=0.05),
                                 ["nl_noise/1"]),
    "nonlinearity without it": (lambda: quiet().nonlinearity(compression=0.1), []),
    "timing rj": (lambda: quiet().timing(rj_ps=0.4), ["timing/1"]),
    "optical rin and shot": (
        lambda: Signal(seed=9, grid=G).carrier("pam4", n_ui=512).optical(
            rin_db_per_hz=-130.0, shot=True), ["rin/1", "shot/1"]),
    "optical without noise": (lambda: Signal(seed=9, grid=G).carrier("pam4", n_ui=512).optical(),
                              []),
    "the field path": (
        lambda: Signal(seed=9, grid=G).carrier("pam4", n_ui=512).eo(kind="mzm", linewidth_hz=1e6)
        .fiber(length_km=2.0).edfa().photodetect().tia(bw_hz=8e9, thermal_rms=0.01),
        ["linewidth/1", "ase/3", "shot/4", "tia/5"]),
    "the field path, noiseless": (
        lambda: Signal(seed=9, grid=G).carrier("pam4", n_ui=512).eo(kind="mzm")
        .fiber(length_km=2.0).photodetect(shot=False).tia(bw_hz=8e9), []),
    "probe noise": (lambda: quiet().probe(bw_hz=20e9, noise_rms=0.01), ["probe/1"]),
    "store dither": (lambda: quiet().store(bits=8, dither_lsb=0.5), ["store/1"]),
    "store without it": (lambda: quiet().store(bits=8), []),
    "timebase and rx_noise": (lambda: quiet().timebase(rms_ps=0.5).rx_noise(rms=0.01),
                              ["timebase/1", "rx_noise/2"]),
    "digitize noise and interleave": (
        lambda: quiet().digitize(snr_db=32, interleave=dict(m_cores=4, offset_mm=0.01)),
        ["noise/1", "interleave/1"]),
    "digitize bits only": (lambda: quiet().digitize(bits=8), []),
    "acquire, ideal": (lambda: quiet().acquire(profile(bits=8)), []),
    "acquire clock jitter": (
        lambda: quiet().acquire(profile(bits=8, sample_clock_jitter_rms_s=1e-13)), ["acquire/1"]),
    "acquire noise floor": (
        lambda: quiet().acquire(profile(noise_floor=dict(rms=0.01))), ["acquire/1"]),
    "acquire converter noise": (
        lambda: quiet().acquire(profile(bits=8, enob=6.0, clip_full_scale=1.2)), ["acquire/1"]),
    "acquire interleave": (
        lambda: quiet().acquire(profile(interleave=dict(m_cores=4, offset_mm=0.01))),
        ["acquire/1"]),
    "trigger jitter": (lambda: quiet().trigger(pre_s=1e-9, post_s=2e-9, jitter_s=2e-12),
                       ["trigger/1"]),
    "trigger without it": (lambda: quiet().trigger(pre_s=1e-9, post_s=2e-9), []),
    "iq noise": (
        lambda: Signal(seed=1, grid=GIQ).iq("qam16", prbs=15)
        .phase_noise(profile=[(1e5, -80), (1e7, -110)]).awgn(snr_db=20, bw_hz=4e6),
        ["phase_noise/1", "awgn/2"]),
}


def _drawn(render, seed):
    """The roles a render drew from: those whose generator is no longer in its seeded state."""
    st = Streams(seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        render(st)
    fresh = Streams(seed)
    moved = {name for name in st.asked
             if st.role(name).bit_generator.state != fresh.role(name).bit_generator.state}
    assert moved == st.drawn()             # the library's own account, which contrast() goes by
    return moved


@pytest.mark.parametrize("name", sorted(CHAINS))
def test_the_roles_listed_are_the_roles_drawn(name):
    build, want = CHAINS[name]
    sig = build()
    assert sig.roles() == want
    assert _drawn(lambda st: sig.waveform(streams=st), sig.seed) == set(want)


@pytest.mark.parametrize("name", sorted(k for k, v in CHAINS.items() if v[1]))
def test_every_listed_role_is_accepted_and_moves_samples(name):
    sig = CHAINS[name][0]()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        x = sig.waveform()
        for role in sig.roles():
            y = sig.contrast(role, seed=11)
            assert y.shape == x.shape and not np.array_equal(x, y), role


@pytest.mark.parametrize("name,role", [
    ("crosstalk default aggressor", "jitter/xtalk1"),
    ("store without it", "store/1"),
    ("acquire, ideal", "acquire/1"),
    ("nonlinearity without it", "nl_noise/1"),
    ("carrier dcd only, which is stated and not drawn", "jitter/0"),
])
def test_a_role_the_parameters_do_not_draw_is_not_listed_and_says_so(name, role):
    """Such a name was always accepted and still is (existing loops pass `jitter/0` to every
    record), but the sibling is the record again and a warning says that. A name that is no
    op's role at all is refused, as before."""
    sig = CHAINS[name][0]()
    assert role not in sig.roles()
    with pytest.warns(RuntimeWarning, match="draws\\s+nothing from it"):
        y = sig.contrast(role, seed=11)
    assert np.array_equal(y, sig.waveform())
    with pytest.raises(ValueError, match="names no randomness"):
        sig.contrast("no_such_role/7", seed=11)


def test_a_several_channel_record_lists_and_draws_the_same_way():
    rec = ws.Channels(
        {"a": Signal().carrier("nrz", n_ui=1024).use("clk").store(bits=8, dither_lsb=0.5),
         "b": Signal().carrier("noise", rms=0.2).tia(thermal_rms=0.01),
         "c": Signal().carrier("nrz", n_ui=1024).store(bits=8)},
        shared={"clk": Signal().timing(rj_ps=0.4)}, seed=4, grid=G)
    want = ["a/store/2", "b/carrier_noise/0", "b/tia/1", "clk/timing/0"]
    assert rec.roles() == want
    st = Streams(rec.seed)                 # each chain renders under its own generators,
    rec.waveforms(streams=st)              # so here the record's own account is the instrument
    assert st.drawn() == set(want)
    with pytest.warns(RuntimeWarning, match="draws\\s+nothing from it"):
        rec.contrast("c/store/1", seed=9)
    x = rec.waveforms()
    for role in want:
        y = rec.contrast(role, seed=9)
        moved = {k for k in x if not np.array_equal(x[k], y[k])}
        assert moved == {role.split("/")[0]} or (role.startswith("clk/") and moved == {"a"})


def test_a_pinned_sub_record_draws_under_its_own_seed_and_is_still_named():
    """`{"from": recipe}` renders under its own seed with its own generators; what it draws
    is noted on the record under the sub-chain's namespace, so the role is listed, accepted,
    and moves samples like any other."""
    lane = Signal(seed=9, grid=G).carrier("nrz", n_ui=1024, jitter=J).store(bits=8)
    sig = quiet().crosstalk(aggressor={"from": lane.recipe()}, coupling=0.2)
    assert sig.roles() == ["crosstalk/1/aggressor/jitter/0"]
    st = Streams(sig.seed)
    x = sig.waveform(streams=st)
    assert st.drawn() == {"crosstalk/1/aggressor/jitter/0"}
    assert not np.array_equal(x, sig.contrast("crosstalk/1/aggressor/jitter/0", seed=5))
    with pytest.warns(RuntimeWarning, match="draws\\s+nothing from it"):
        sig.contrast("crosstalk/1/aggressor/store/1", seed=5)
