"""The general stages on an I/Q record, each way (the table in `wfmsynth.domain`).

A stage is opted into domain 'iq' only where it means on an envelope what it means on a
voltage, and that is checked here against the PASSBAND: the same stage applied to the real
record the envelope stands for, compared after `upconvert`. Where a stage has no such meaning
the refusal is checked instead, with what it says to do.

Also here, because they follow from it: a real signal inside an I/Q record (a supply, an
enable), one oscillator shared by name across I/Q channels, and the marker channel an I/Q
export gets from a sequence's boundaries.
"""
from __future__ import annotations

import json
import warnings

import numpy as np
import pytest

import wfmsynth as ws
from wfmsynth import Grid, Signal
from wfmsynth import domain as DOMAIN
from wfmsynth import iq as IQ

FS, BAUD, FC, N = 400e6, 5e6, 60e6, 1 << 15
G = Grid(fs=FS, baud=BAUD, n=N, domain="iq", fc_hz=FC)
T = np.arange(N) / FS
RIPPLE = dict(f_hz=40e3, amp=0.2, offset=1.0)           # a supply: 1 +/- 20 %
ENABLE = dict(f_hz=25e3, duty=0.5)                      # a square enable, +/-1


@pytest.fixture(autouse=True)
def _quiet():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


def env(seed=3, **kw):
    return Signal(seed=1, grid=G).iq("qpsk", prbs=9, prbs_seed=seed, **kw)


def passband(x):
    return IQ.upconvert(x, FC, FS)


# ------------------------------------------------------------------ the ones that are opted in
def test_gain_by_a_real_supply_scales_the_envelope_as_it_scales_the_passband():
    ripple = Signal().carrier("sine", **RIPPLE)          # a voltage, inside an I/Q record
    sig = env().gain(by=ripple)
    x = sig.waveform()
    assert np.iscomplexobj(x)
    r = sig.subchain("gain/1/by")
    assert r.dtype == np.float64 and abs(r.max() - 1.2) < 1e-3 and abs(r.min() - 0.8) < 1e-3
    assert np.array_equal(x, env().waveform() * r)
    # the same supply on the real passband record, through the real-domain op
    real = env().upconvert().gain(by=Signal(grid=Grid(fs=FS, baud=BAUD, n=N)).carrier(
        "sine", **RIPPLE)).waveform()
    assert np.abs(passband(x) - real).max() < 1e-12
    assert np.array_equal(env().gain(by=0.5).waveform(), 0.5 * env().waveform())
    again = Signal.from_recipe(json.loads(sig.to_json()))
    assert again.sha256() == sig.sha256() and np.array_equal(again.waveform(), x)


def test_gate_by_a_real_enable_switches_both_rails_and_emits_the_instants():
    sig = env().gate(by=Signal().carrier("square", **ENABLE), threshold=0.0)
    x, ev = sig.realize()
    en = sig.subchain("gate/1/by")
    full = env().waveform()
    assert np.array_equal(x, np.where(en > 0.0, full, 0.0))
    on = np.mean(np.abs(x) > 0)
    assert 0.45 < on < 0.55 and np.any(x.imag != 0)
    real = np.where(en > 0.0, passband(full), 0.0)
    assert np.abs(passband(x) - real).max() < 1e-12
    edges = [e for e in ev if e.kind == "transition"]
    want = np.flatnonzero(np.diff((en > 0.0).astype(int))) + 1
    assert [e.sample for e in edges] == list(want) and len(want) >= 3
    held = env().gate(by=Signal().carrier("square", **ENABLE), threshold=0.0, off="hold")
    h = held.waveform()
    k = int(want[0])                              # the gate closes or opens here
    closed = k if en[k] <= 0.0 else int(want[1])
    assert h[closed] == full[closed - 1] and np.iscomplexobj(h)


def test_combine_adds_an_interferer_about_the_same_centre():
    """An adjacent channel 20 MHz up, 20 dB down: the sum of envelopes is the sum of the
    passband records, and the neighbour's power lands where and as large as stated."""
    other = env(seed=11).freq_offset(f_hz=20e6)
    sig = env().combine(other, gains=[0.1])
    x = sig.waveform()
    alone, nb = env().waveform(), sig.subchain("combine/1/others/0")
    assert np.iscomplexobj(nb) and np.array_equal(x, alone + 0.1 * nb)
    assert np.abs(passband(x) - (passband(alone) + 0.1 * passband(nb))).max() < 1e-12
    spec = ws.rf.spectrum(x, G, rbw_hz=200e3)
    wanted = ws.rf.band_power(spec, -4e6, 4e6)
    neighbour = ws.rf.band_power(spec, 16e6, 24e6)
    assert abs(10 * np.log10(neighbour / wanted) + 20.0) < 0.5, (wanted, neighbour)
    assert "combine/1/others/0" in sig.subchains()


def test_sequence_and_switch_join_iq_chains_and_keep_a_real_selector():
    pre = Signal(grid=G).iq(symbols=[1, -1] * 20)        # 40 symbols of BPSK alternation
    t_pre = 40 / BAUD
    sig = Signal(seed=1, grid=G).sequence([{"chain": pre, "duration_s": t_pre}, {"chain": env()}])
    x, ev = sig.realize()
    n_pre = round(t_pre * FS)
    assert np.iscomplexobj(x)
    assert np.array_equal(x[:n_pre], pre.waveform()[:n_pre])
    assert np.array_equal(x[n_pre:], env().waveform()[:N - n_pre])
    (seg,) = [e for e in ev if e.kind == "segment"]
    assert seg.sample == n_pre and seg.t_s == t_pre
    assert sig.stored_grid(x).domain == "iq" and sig.upconvert().waveform().dtype == np.float64

    sel = Signal().carrier("square", **ENABLE)
    sw = Signal(seed=1, grid=G).switch(select=sel, inputs=[pre, env()], threshold=0.0)
    y = sw.waveform()
    s = sw.subchain("switch/0/select")
    assert s.dtype == np.float64
    assert np.array_equal(y, np.where(s > 0.0, sw.subchain("switch/0/inputs/1"),
                                      sw.subchain("switch/0/inputs/0")))
    soft = Signal(seed=1, grid=G).switch(select=sel, inputs=[pre, env()], threshold=0.0,
                                         t_switch_s=1e-6).waveform()
    assert np.iscomplexobj(soft) and not np.array_equal(soft, y)
    assert np.mean(soft == y) > 0.8                       # identical outside the windows


# ------------------------------------------------------------------ the ones that are refused
@pytest.mark.parametrize("build,says", [
    (lambda s: s.offset(by=0.1), "iq_offset(i=, q=)"),
    (lambda s: s.law("tanh", gain=2.0), "dac(bits=)"),
    (lambda s: s.trigger(pre_s=1e-6, post_s=1e-6), "triggers on a voltage"),
])
def test_a_stage_with_no_meaning_on_an_envelope_is_refused_where_written(build, says):
    s = env()
    with pytest.raises(ValueError) as e:
        build(s)
    msg = str(e.value)
    assert "complex baseband" in msg and says in msg and "upconvert(" in msg
    assert len(s.ops) == 1
    # ... and each is accepted on the real passband record
    build(env().upconvert())


@pytest.mark.parametrize("how", ["product", "min", "max"])
def test_combine_on_iq_is_a_sum_only(how):
    with pytest.raises(ValueError, match="how='sum'"):
        env().combine(env(seed=5), how=how)
    with pytest.raises(ValueError, match="how='sum'"):
        ws.joins.combine(np.ones(4, complex), [np.ones(4, complex)], how=how)


def test_the_other_signal_is_refused_in_the_wrong_domain():
    with pytest.raises(ValueError, match="REAL waveform.*I/Q chain"):
        env().combine(Signal().carrier("sine", f_hz=1e6)).waveform()
    with pytest.raises(ValueError, match="COMPLEX"):     # an enable is real; a gain may be
        env().gate(by=env(seed=5)).waveform()             # complex (tests/test_envelope_joins.py)
    with pytest.raises(ValueError, match="COMPLEX"):
        Signal(seed=1, grid=G).switch(select=env(), inputs=[env(), env(seed=5)]).waveform()
    # a Signal that STATES an I/Q grid is still held to it
    with pytest.raises(ValueError, match="carrier.*I/Q grid.*starts with .iq"):
        Signal(grid=G).carrier("sine", f_hz=1e6)


def test_a_chain_with_no_grid_starts_in_the_domain_its_ops_need():
    assert DOMAIN.needs(Signal().iq("qpsk", prbs=7).ops) == "iq"
    assert DOMAIN.needs(Signal().carrier("nrz").ops) == "real"
    assert DOMAIN.needs(Signal().phase_noise(profile=[(1e5, -90), (1e6, -110)]).ops) == "iq"
    assert DOMAIN.needs([{"op": "filter", "kind": "fir", "taps": [1.0]}]) is None
    s = Signal().iq("qpsk", prbs=7).upconvert(fc_hz=FC).lossy(loss_db=1.0, loss_at_ghz=1.0)
    assert [d for _k, d in s.stage_kinds(domains=True)] == ["iq", "real", "real"]
    with pytest.raises(ValueError, match="awgn.*real voltage"):
        Signal().carrier("nrz").awgn(es_n0_db=20)


# ------------------------------------------------------------------ one LO, several channels
PROFILE = [(2e4, -70.0), (2e5, -90.0), (2e6, -110.0)]


def test_an_oscillator_shared_by_name_turns_every_channel_by_the_same_phase():
    """Two different envelopes through ONE shared `phase_noise` stage: the phase each was
    turned by, recovered from each channel against its own clean envelope, is the same record
    of phase; with an oscillator each it is not. Re-rolling the shared role moves both
    channels and leaves a third alone."""
    def chan(seed):
        return Signal().iq("qpsk", prbs=9, prbs_seed=seed)

    rec = ws.Channels({"a": chan(3).use("lo"), "b": chan(5).use("lo"), "c": chan(7)},
                      shared={"lo": Signal().phase_noise(profile=PROFILE)}, seed=2, grid=G)
    assert rec.roles() == ["lo/phase_noise/0"]
    x = rec.waveforms()
    clean = {k: Signal(seed=2, grid=G).iq("qpsk", prbs=9, prbs_seed=s).waveform()
             for k, s in (("a", 3), ("b", 5), ("c", 7))}
    assert np.array_equal(x["c"], clean["c"])
    busy = (np.abs(clean["a"]) > 0.2) & (np.abs(clean["b"]) > 0.2)
    turn = {k: np.angle(x[k][busy] / clean[k][busy]) for k in "ab"}
    rms = turn["a"].std()
    want = ws.rf.phase_rms_rad(PROFILE)
    assert 0.5 * want < rms < 2.0 * want, (rms, want)     # one record estimates it loosely
    assert np.abs(turn["a"] - turn["b"]).max() < 1e-9 * rms

    each = ws.Channels({"a": chan(3).phase_noise(profile=PROFILE),
                        "b": chan(5).phase_noise(profile=PROFILE)}, seed=2, grid=G).waveforms()
    own = {k: np.angle(each[k][busy] / clean[k][busy]) for k in "ab"}
    assert (own["a"] - own["b"]).std() > rms              # independent: about sqrt(2) rms

    y = rec.contrast("lo/phase_noise/0")
    assert np.array_equal(y["c"], x["c"])
    assert not np.array_equal(y["a"], x["a"]) and not np.array_equal(y["b"], x["b"])
    again = ws.Channels.from_recipe(json.loads(rec.to_json()))
    assert again.sha256() == rec.sha256()
    assert all(np.array_equal(again.waveforms()[k], x[k]) for k in x)


def test_a_record_holds_an_iq_channel_and_the_real_supply_that_scales_it():
    rec = ws.Channels(
        {"rf": Signal().iq("qpsk", prbs=9).gain(by=Signal().use("vdd")).upconvert(),
         "rail": Signal().use("vdd")},
        shared={"vdd": Signal().carrier("sine", **RIPPLE)}, seed=1, grid=G)
    x = rec.waveforms()
    assert x["rail"].dtype == np.float64 and x["rf"].dtype == np.float64
    want = passband(Signal(seed=1, grid=G).iq("qpsk", prbs=9).waveform() * x["rail"])
    assert np.array_equal(x["rf"], want)
    assert rec.stored_grid("rail", x["rail"]).domain == "real"
    assert rec.stored_grid("rf", x["rf"]).domain == "real"
    iq_only = ws.Channels({"rf": Signal().iq("qpsk", prbs=9)}, seed=1, grid=G)
    assert rec.stored_grid("rf", x["rf"]).fs == FS
    assert iq_only.stored_grid("rf", iq_only.waveforms()["rf"]).domain == "iq"
    with pytest.raises(ValueError, match="reflect.*complex baseband"):
        ws.Channels({"rf": Signal().iq("qpsk", prbs=9).use("line")},
                    shared={"line": Signal().reflect(td_ps=100.0, gamma_s=0.2, gamma_l=0.2)},
                    seed=1, grid=G)


# ------------------------------------------------------------------ markers on an I/Q export
def test_a_sequence_boundary_is_a_marker_on_an_iq_export(tmp_path):
    pre = Signal(grid=G).iq(symbols=[1, -1] * 20)
    t_pre = 40 / BAUD
    sig = Signal(seed=1, grid=G).sequence(
        [{"chain": pre, "duration_s": t_pre}, {"chain": env(), "duration_s": 200 / BAUD}, {"chain": pre}],
        join="crossfade", t_join_s=2 / BAUD)
    x, ev = sig.realize()
    out = ws.to_interleaved(str(tmp_path / "burst.iq"), x, dtype="float64", events=ev)
    marks = np.fromfile(out["markers"], dtype=np.uint8)
    b1, b2 = round(t_pre * FS), round((t_pre + 200 / BAUD) * FS)
    m = round(2 / BAUD * FS)                                # the join window, in samples
    want = np.zeros(N, np.uint8)
    want[b1:b1 + m] = 1
    want[b2:b2 + m] = 1
    assert len(marks) == N and np.array_equal(marks, want)
    back = ws.read_iq(out["path"], format="iq_interleaved", dtype="float64", fs_hz=FS,
                      fc_hz=FC).waveform()
    assert np.array_equal(back, x)
    meta = ws.to_sigmf(str(tmp_path / "burst"), x, sig.stored_grid(x), events=ev,
                       dtype="float64")
    notes = json.load(open(meta["meta"]))["annotations"]
    assert [(a["core:sample_start"], a["core:sample_count"], a["core:label"])
            for a in notes] == [(b1, m, "segment"), (b2, m, "segment")]
