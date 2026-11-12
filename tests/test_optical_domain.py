"""The optical field is a stated domain (`wfmsynth.domain`, GitHub #92).

Three things are checked here.

NOTHING THAT EXISTED MOVED. A chain reaches the field through `eo` from a real drive, so its
grid, its recipe's JSON and its samples are what they were: the composed link is compared, to
the bit, with the same functions of `wfmsynth.optical` called by hand, and a recipe written
before the domain existed replays. (`tools/digest_compare.py` is the same proof against
another checkout.)

A MISTAKE IS A REFUSAL. An electrical op on the field used to cast it to float under a numpy
warning; it is refused by name and told `photodetect` is the bridge. A field op on a voltage,
or on an I/Q envelope, likewise.

THE GENERAL STAGES, EACH WAY (the table in `wfmsynth.domain`). Each one that is accepted is
checked against an analytic result AND against its I/Q counterpart on the same samples, which
it must equal to the bit; each one that is refused, for what it says to do instead.
"""
from __future__ import annotations

import json
import warnings

import numpy as np
import pytest

import wfmsynth as ws
from wfmsynth import Grid, Signal
from wfmsynth import compose as C
from wfmsynth import domain as DOMAIN
from wfmsynth import optical as OPT
from wfmsynth.streams import Streams

FS, BAUD, N, N_UI = 80e9, 10e9, 4096, 512
G = Grid(fs=FS, baud=BAUD, n=N)                       # states no domain: it follows the source
GO = Grid(fs=FS, baud=BAUD, n=N, domain="optical")    # a chain that STARTS in the field
GI = Grid(fs=FS, baud=BAUD, n=N, domain="iq")
T = np.arange(N) / FS
ENABLE = dict(f_hz=40e6, duty=0.5)                    # a square enable, +/-1, 2000 samples


@pytest.fixture(autouse=True)
def _quiet():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


def drive(seed=1, **kw):
    return Signal(seed=seed, grid=G).carrier("nrz", n_ui=N_UI, **kw)


def field(**kw):
    return drive().eo(**kw)


def cw(phase=0.0, f_hz=0.0, grid=GO):
    """A captured field: constant power 1, at `phase`, `f_hz` off the optical carrier."""
    return Signal(grid=grid).capture(values=np.exp(1j * (phase + 2 * np.pi * f_hz * T)))


def detected(sig):
    return sig.photodetect(shot=False).waveform()


# ------------------------------------------------------------------ nothing that existed moved
LINK = [{"op": "carrier", "kind": "nrz", "n_ui": 512},
        {"op": "eo", "kind": "mzm", "er_db": 10, "alpha": 0.3, "linewidth_hz": 1000000.0},
        {"op": "fiber", "length_km": 10.0, "gamma_per_w_km": 1.3},
        {"op": "optical_mpi", "delay_samples": 37, "reflectivity": 0.05},
        {"op": "edfa", "gain_db": 12, "nf_db": 5},
        {"op": "photodetect", "responsivity": 0.8, "photons_per_unit": 5000.0},
        {"op": "tia", "gain": 2.0, "bw_hz": 8000000000.0, "thermal_rms": 0.01}]
# a recipe as every release before the field was a domain wrote it: no domain anywhere
WRITTEN_BEFORE = {"seed": 7, "ops": LINK,
                  "grid": {"fs": 80000000000.0, "baud": 10000000000.0, "n": 4096, "v_full": 1.0}}


def _link(seed=7):
    return (drive(seed).eo(kind="mzm", er_db=10, alpha=0.3, linewidth_hz=1e6)
            .fiber(length_km=10.0, gamma_per_w_km=1.3)
            .optical_mpi(delay_samples=37, reflectivity=0.05).edfa(gain_db=12, nf_db=5)
            .photodetect(responsivity=0.8, photons_per_unit=5e3)
            .tia(gain=2.0, bw_hz=8e9, thermal_rms=0.01))


def _by_hand(seed=7, upto=7):
    """The same link from `wfmsynth.optical`'s functions, with no composer and no domain."""
    st = Streams(seed)
    x = drive(seed).waveform()
    x = OPT.modulate_field(x, kind="mzm", er_db=10, alpha=0.3, linewidth_hz=1e6, grid=G,
                           rng=st.role("linewidth/1"))
    x = OPT.fiber(x, length_km=10.0, gamma_per_w_km=1.3, grid=G)
    x = OPT.field_mpi(x, delay_samples=37, reflectivity=0.05)
    x = OPT.edfa(x, gain_db=12, nf_db=5, rng=st.role("ase/4"))
    if upto == 5:
        return x
    x = OPT.photodetect(x, responsivity=0.8, photons_per_unit=5e3, rng=st.role("shot/5"))
    return OPT.tia(x, gain=2.0, bw_hz=8e9, thermal_rms=0.01, grid=G, rng=st.role("tia/6"))


def test_an_existing_link_renders_writes_and_replays_exactly_as_before_the_domain():
    sig = _link()
    x = sig.waveform()
    assert x.dtype == np.float64 and np.array_equal(x, _by_hand())
    # the recipe: byte for byte what it was, so its sha256 is too
    r = sig.recipe()
    assert r.pop("wfmsynth_version") == ws.__version__
    assert json.dumps(r, sort_keys=True) == json.dumps(WRITTEN_BEFORE, sort_keys=True)
    assert (sig.grid.domain, sig.grid.domain_stated, sig.grid.fc_hz) == ("real", False, None)
    old = Signal.from_recipe(json.loads(json.dumps(WRITTEN_BEFORE)))
    assert np.array_equal(old.waveform(), x) and old.sha256() == sig.sha256()
    assert sig.roles() == ["linewidth/1", "ase/4", "shot/5", "tia/6"]
    assert sig.stage_kinds() == ["source", "shape", "channel", "channel", "channel",
                                 "instrument", "instrument"]
    assert sig.stored_grid(x) == G


def test_a_record_that_ends_in_the_field_is_still_complex_on_a_grid_that_says_nothing():
    sig = Signal(seed=7, grid=G)
    sig.ops = [dict(o) for o in LINK[:5]]
    e = sig.waveform()
    assert e.dtype == np.complex128 and np.array_equal(e, _by_hand(upto=5))
    assert "domain" not in sig.recipe()["grid"]
    # what is new is that the record can SAY what it is
    assert sig.stored_grid(e).domain == "optical" and sig.stored_grid(e).fc_hz is None
    assert np.array_equal(Signal.from_recipe(json.loads(sig.to_json())).waveform(), e)


def test_stage_kinds_and_summary_report_the_domain():
    sig = field(kind="mzm").fiber(length_km=2.0).photodetect(shot=False).tia(bw_hz=8e9)
    assert sig.stage_kinds(domains=True) == [
        ("source", "real"), ("shape", "optical"), ("channel", "optical"),
        ("instrument", "real"), ("instrument", "real")]
    assert sig.summary() == """\
Signal  seed=1  fs=8e+10 Hz, baud=1e+10, n=4096
  0  carrier      kind='nrz', n_ui=512
  1  eo           kind='mzm'  -> optical
  2  fiber        length_km=2.0
  3  photodetect  shot=False  -> real
  4  tia          bw_hz=8e+09"""
    # a chain that does not start as a voltage says so, and a real chain's text is untouched
    assert cw().fiber().summary().splitlines()[0].endswith("n=4096  domain=optical")
    assert "->" not in drive().lossy(loss_db=3.0, loss_at_ghz=5.0).summary()
    assert "domain" not in drive().summary()
    iq = Signal(grid=G).iq("qpsk", prbs=9).upconvert(fc_hz=20e9).summary().splitlines()
    assert iq[0].endswith("domain=iq") and iq[2].endswith("-> real")


# ------------------------------------------------------------------ a mistake is a refusal
@pytest.mark.parametrize("op,kw,says", [
    ("lossy", dict(loss_db=3.0, loss_at_ghz=5.0), "fiber(length_km=, atten_db_km=)"),
    ("tia", {}, "takes the photocurrent"),
    ("rx_noise", dict(rms=0.01), "edfa(gain_db=, nf_db=)"),
    ("digitize", dict(bits=8), "photodetect("),
    ("scope", dict(bw_hz=8e9), "photodetect("),
    ("probe", dict(bw_hz=8e9), "photodetect("),
    ("ctle", dict(fz_ghz=2, fp1_ghz=6, fp2_ghz=12), "photodetect("),
    ("events", dict(kind="glitch", on="times", times=[1e-9]), "photodetect("),
    ("dispersion", {}, "fiber(length_km=, D_ps_nm_km=)"),
    ("optical", {}, "already a field"),
    ("eo", {}, "already a field"),
])
def test_an_electrical_op_on_the_field_refuses_by_name_and_names_the_bridge(op, kw, says):
    s = field()
    with pytest.raises(ValueError) as e:
        getattr(s, op)(**kw)
    msg = str(e.value)
    assert msg.startswith(f"{op} (op 2)") and "optical field (domain 'optical')" in msg
    assert "photodetect(responsivity=...) is the bridge" in msg and says in msg
    assert len(s.ops) == 2, "a refused op must not be left in the chain"
    getattr(field().photodetect(shot=False), op)(**kw) if op not in ("optical", "eo") else None


def test_a_hand_written_recipe_with_an_electrical_op_on_the_field_is_refused_not_cast():
    """It used to render: numpy cast the field to float and warned, which nobody reads."""
    r = dict(WRITTEN_BEFORE, ops=LINK[:2] + [{"op": "lossy", "loss_db": 3.0, "loss_at_ghz": 5.0}])
    with pytest.raises(ValueError, match=r"lossy \(op 2\).*photodetect\(.*is the bridge"):
        Signal.from_recipe(r).waveform()


@pytest.mark.parametrize("op", ["fiber", "edfa", "photodetect", "optical_mpi"])
def test_a_field_op_refuses_a_voltage_and_an_envelope_and_names_eo(op):
    kw = dict(delay_samples=5) if op == "optical_mpi" else {}
    with pytest.raises(ValueError) as e:
        getattr(drive(), op)(**kw)
    msg = str(e.value)
    assert msg.startswith(f"{op} (op 1)") and "real voltage record" in msg
    assert "'optical' only" in msg and "eo(kind='mzm' | 'dml') is the bridge" in msg
    with pytest.raises(ValueError) as e:
        getattr(Signal(grid=G).iq("qpsk", prbs=9), op)(**kw)
    assert "complex baseband" in str(e.value) and "starts at eo(kind=...)" in str(e.value)
    # an I/Q impairment on a field: both are complex, and they are not the same record
    with pytest.raises(ValueError, match="awgn.*edfa.*not the same record.*photodetect"):
        field().awgn(snr_db=20.0)
    with pytest.raises(ValueError, match="phase_noise.*eo\\(linewidth_hz=\\)"):
        field().phase_noise(profile=[[1e6, -90.0]])


def test_the_ops_that_accept_a_field_are_an_inventory():
    """Real-only unless listed: this fails when an op is opted in, so that opting in is a
    decision written down next to a test of what the op means on a field."""
    takes = {op for op in C._EXEC if "optical" in DOMAIN.accepts(op)}
    assert takes == {"fiber", "optical_mpi", "edfa", "photodetect",          # defined on E(t)
                     "capture",                                              # a stored field
                     "filter", "gain", "gate", "combine", "sequence", "switch", "use"}
    assert DOMAIN.EMITS["eo"] == "optical" and DOMAIN.EMITS["photodetect"] == "real"
    assert DOMAIN.accepts("eo") == ("real",)
    # ... and every other op is refused on a field, each naming the bridge
    for op in sorted(set(C._EXEC) - takes):
        with pytest.raises(ValueError, match="photodetect"):
            DOMAIN.trace(LINK[:2] + [{"op": op}], G)


# ------------------------------------------------------------------ the general stages: yes
STAGES = {
    "filter": lambda s: s.filter("butter", f_hz=4e9, order=3),
    "gain number": lambda s: s.gain(by=0.5),
    "gain signal": lambda s: s.gain(by=Signal().carrier("sine", f_hz=50e6, amp=0.2, offset=1.0)),
    "gate": lambda s: s.gate(by=Signal().carrier("square", **ENABLE), threshold=0.0),
    "gate hold": lambda s: s.gate(by=Signal().carrier("square", **ENABLE), threshold=0.0,
                                  off="hold"),
}


@pytest.mark.parametrize("stage", sorted(STAGES))
def test_a_general_stage_on_a_field_is_its_iq_counterpart_to_the_bit(stage):
    e = field(kind="dml", alpha=3.0, er_db=8).waveform()         # a chirped field: Im != 0
    assert np.abs(e.imag).max() > 0.1
    on_field = STAGES[stage](Signal(seed=1, grid=GO).capture(values=e))
    on_iq = STAGES[stage](Signal(seed=1, grid=GI).capture(values=e))
    y = on_field.waveform()
    assert y.dtype == np.complex128 and np.array_equal(y, on_iq.waveform())
    assert not np.array_equal(y, e)
    # ... and mid-chain after `eo`, where the grid says nothing, it is the same again
    assert np.array_equal(STAGES[stage](field(kind="dml", alpha=3.0, er_db=8)).waveform(), y)
    assert [d for _, d in on_field.stage_kinds(domains=True)] == ["optical", "optical"]
    again = Signal.from_recipe(json.loads(on_field.to_json()))
    assert again.grid.domain == "optical" and np.array_equal(again.waveform(), y)


def test_combine_on_a_field_is_its_iq_counterpart_to_the_bit():
    e = field(kind="dml", alpha=3.0, er_db=8).waveform()
    other = field(bias=0.3).waveform()
    y = (Signal(grid=GO).capture(values=e)
         .combine(Signal(grid=GO).capture(values=other), gains=[0.25]).waveform())
    yi = (Signal(grid=GI).capture(values=e)
          .combine(Signal(grid=GI).capture(values=other), gains=[0.25]).waveform())
    assert np.array_equal(y, yi) and np.array_equal(y, e + 0.25 * other)


def test_an_optical_filter_passes_half_the_power_at_its_corner():
    """A CW field `f` off the carrier through a 3rd-order Butterworth at 4 GHz, read from the
    detected power once the filter has settled, against the closed form of the digital
    (bilinear) design: |H|^2 = 1 / (1 + (tan(pi f/fs) / tan(pi fc/fs))^6). Half the power
    at the corner, and the same either side of the carrier."""
    got = {}
    for f in (-8e9, -4e9, 1e9, 4e9, 8e9):
        p = detected(cw(f_hz=f).filter("butter", f_hz=4e9, order=3))[N // 2:].mean()
        got[f] = p
        want = 1.0 / (1.0 + (np.tan(np.pi * abs(f) / FS) / np.tan(np.pi * 4e9 / FS)) ** 6)
        assert abs(p - want) < 1e-9, (f, p, want)
    assert abs(10 * np.log10(got[4e9]) + 3.0103) < 1e-4, got
    assert abs(got[8e9] - got[-8e9]) < 1e-12


def test_gain_on_a_field_is_an_attenuator_whose_power_goes_as_the_square():
    p0 = detected(field())
    assert np.allclose(detected(field().gain(by=0.5)), 0.25 * p0, rtol=1e-14, atol=0)
    ripple = Signal().carrier("sine", f_hz=50e6, amp=0.2, offset=1.0)
    sig = field().gain(by=ripple)
    r = sig.subchain("gain/2/by")
    assert r.dtype == np.float64 and np.array_equal(sig.waveform(), field().waveform() * r)
    assert np.allclose(detected(sig), p0 * r ** 2, rtol=1e-13, atol=0)


def test_a_complex_gain_on_a_field_is_an_arms_phase_and_a_moving_one_stays_real():
    """An interferometer arm: one laser split, one arm turned by `phase`, and rejoined detects
    2 (1 + cos phase) -- the same number a phase written into the captured field gives."""
    for phase in (0.0, np.pi / 3, np.pi / 2, np.pi):
        arm = Signal(grid=GO).capture(values=np.ones(N, complex)).gain(by=np.exp(1j * phase))
        got = detected(cw().combine(arm)).mean()
        assert abs(got - 2.0 * (1.0 + np.cos(phase))) < 1e-12, (phase, got)
    sig = field().gain(by=0.5j)
    assert sig.ops[-1] == {"op": "gain", "by": [0.0, 0.5]}          # data, as on an I/Q record
    assert np.array_equal(Signal.from_recipe(json.loads(json.dumps(sig.recipe()))).waveform(),
                          0.5j * field().waveform())
    # a factor that MOVES is real on a field: an I/Q chain is an RF envelope, not light
    for by in (ws.recipes.fading(doppler_hz=2e9), Signal().carrier("nrz", n_ui=N_UI).eo()):
        with pytest.raises(ValueError, match="COMPLEX waveform"):
            field().gain(by=by).waveform()
    with pytest.raises(ValueError, match="real voltage here"):
        drive().gain(by=0.5j)


def test_the_envelope_ops_say_what_a_field_is_when_they_refuse_one():
    for build, said in ((lambda s: s.law("rapp"), "eo(kind=)"),
                        (lambda s: s.law("tanh"), "eo(kind=)"),
                        (lambda s: s.delay(by=1e-10), "carrier's phase"),
                        (lambda s: s.lossy(loss_db=3.0, loss_at_ghz=5.0), "fiber(")):
        with pytest.raises(ValueError) as e:
            build(field())
        assert "optical field (domain 'optical')" in str(e.value) and said in str(e.value)


def test_gate_on_a_field_is_a_shutter_and_emits_its_instants():
    sig = cw().gate(by=Signal().carrier("square", **ENABLE), threshold=0.0)
    x, ev = sig.photodetect(shot=False).realize()
    en = sig.subchain("gate/1/by")
    assert np.array_equal(x, np.where(en > 0.0, 1.0, 0.0))        # dark while it is closed
    assert x.mean() == np.mean(en > 0.0) and 0.45 < x.mean() < 0.55
    want = np.flatnonzero(np.diff((en > 0.0).astype(int))) + 1
    assert [e.sample for e in ev if e.kind == "transition"] == list(want) and len(want) >= 3


@pytest.mark.parametrize("phase", [0.0, np.pi / 3, np.pi / 2, 2 * np.pi / 3, np.pi])
def test_combine_on_a_field_is_a_coupler_and_two_fields_interfere(phase):
    """Two unit fields `phase` apart: P = |1 + e^(j*phase)|^2 = 2 (1 + cos phase), from four
    times one field's power down to none. A sum of INTENSITIES would say 2 throughout."""
    p = detected(cw().combine(cw(phase)))
    assert np.abs(p - 2.0 * (1.0 + np.cos(phase))).max() < 1e-12
    # the same through `eo`: a field and its own copy
    both, one = detected(field().combine(field())), detected(field())
    assert np.allclose(both, 4.0 * one, rtol=1e-14, atol=0)
    assert np.abs(detected(field().combine(field(), gains=[-1.0]))).max() == 0.0


def test_two_lasers_are_incoherent_and_each_has_its_own_role():
    """A second source with its own linewidth: its phase walk is drawn under the sub-chain's
    namespace, so it is listed, and re-rolling it alone moves the beat and not the source.

    Two lasers of linewidth dv each drift apart as a random walk of variance 4 pi dv t, so
    E[cos] = exp(-2 pi dv t) and the mean power over a record T is
    2 P (1 + (1 - exp(-a)) / a) with a = 2 pi dv T: the incoherent 2 P once T is many
    coherence times, against the coherent 4 P of one laser split and rejoined."""
    dv = 200e6

    def beat(seed):
        return (drive(seed).eo(linewidth_hz=dv).combine(
            Signal().carrier("nrz", n_ui=N_UI).eo(linewidth_hz=dv)).photodetect(shot=False))
    sig = beat(1)
    assert sig.roles() == ["linewidth/1", "combine/2/others/0/linewidth/1"]
    p = sig.waveform()
    a = 2 * np.pi * dv * N / FS
    want = 2.0 * (1.0 + (1.0 - np.exp(-a)) / a)                       # 2.031
    ratio = np.mean([beat(s).waveform().mean() / detected(drive(s).eo()).mean()
                     for s in range(1, 25)])
    assert abs(ratio - want) < 0.15, (ratio, want)
    sib = sig.contrast("combine/2/others/0/linewidth/1", seed=3)
    assert not np.array_equal(sib, p)
    assert np.array_equal(sig.contrast("combine/2/others/0/linewidth/1", seed=3), sib)
    other = sig.subchain("combine/2/others/0")                        # the second laser's field
    assert other.dtype == np.complex128
    assert np.array_equal(Signal.from_recipe(sig.recipe()).subchain("combine/2/others/0"), other)


def test_sequence_and_switch_join_field_chains_and_the_grid_follows_them():
    idle = Signal().carrier("nrz", n_ui=N_UI).eo().gain(by=0.1)      # 1 % of the power
    data = Signal().carrier("nrz", n_ui=N_UI, pattern="prbs9").eo()
    n_idle, t_idle = 800, 800 / FS
    sig = Signal(seed=1, grid=G).sequence([{"chain": idle, "duration_s": t_idle},
                                           {"chain": data}])
    x, ev = sig.realize()
    assert x.dtype == np.complex128 and sig.grid.domain == "optical" and not G.domain_stated
    assert np.array_equal(x[:n_idle], sig.subchain("sequence/0/segments/0/chain")[:n_idle])
    assert np.array_equal(x[n_idle:], sig.subchain("sequence/0/segments/1/chain")[:N - n_idle])
    (seg,) = [e for e in ev if e.kind == "segment"]
    assert seg.sample == n_idle
    assert sig.stage_kinds(domains=True) == [("source", "optical")]
    r = sig.recipe()
    assert (r["grid"]["domain"], r["grid"]["fc_hz"]) == ("optical", None)
    again = Signal.from_recipe(json.loads(sig.to_json()))
    assert again.sha256() == sig.sha256() and np.array_equal(again.waveform(), x)
    # a burst as a receiver sees it: 20 dB down, then light
    p = detected(sig)
    assert p.dtype == np.float64
    assert abs(p[:n_idle].mean() / detected(field())[:n_idle].mean() - 0.01) < 1e-12
    assert p[n_idle:].mean() > 50 * p[:n_idle].mean()
    # the same parts ended on photodetect are voltages, and the record is a voltage
    volts = Signal(seed=1, grid=G).sequence([
        {"chain": idle.copy().photodetect(shot=False), "duration_s": t_idle},
        {"chain": data.copy().photodetect(shot=False)}])
    assert volts.grid.domain == "real" and np.array_equal(volts.waveform(), p)

    sel = Signal().carrier("square", **ENABLE)
    sw = Signal(seed=1, grid=G).switch(select=sel, inputs=[idle, data], threshold=0.0)
    y = sw.waveform()
    s = sw.subchain("switch/0/select")
    assert s.dtype == np.float64 and sw.grid.domain == "optical"
    assert np.array_equal(y, np.where(s > 0.0, sw.subchain("switch/0/inputs/1"),
                                      sw.subchain("switch/0/inputs/0")))


def test_polarisation_is_two_named_chains_through_one_fibre():
    """Not part of the domain: a `Channels` pair. Both through the shared span, and a
    detector that does not resolve polarisation sees the sum of the two POWERS, because
    orthogonal fields do not interfere -- which is why they are two chains and not a sum."""
    rec = ws.Channels(
        {"x": Signal().carrier("nrz", n_ui=N_UI).eo().use("span"),
         "y": Signal().carrier("nrz", n_ui=N_UI, pattern="prbs9").eo().use("span"),
         "pd": Signal().use("x").photodetect(shot=False).combine(
             Signal().use("y").photodetect(shot=False))},
        shared={"span": Signal().fiber(length_km=10.0)}, seed=1, grid=G)
    w = rec.waveforms()
    assert w["x"].dtype == w["y"].dtype == np.complex128 and w["pd"].dtype == np.float64
    assert not np.array_equal(w["x"], w["y"])
    assert np.abs(w["pd"] - (np.abs(w["x"]) ** 2 + np.abs(w["y"]) ** 2)).max() < 1e-12
    # the span is the one fibre: each channel is its own field through the same function
    ex = Signal(seed=1, grid=G).carrier("nrz", n_ui=N_UI).eo().waveform()
    assert np.array_equal(w["x"], OPT.fiber(ex, length_km=10.0, grid=G))
    assert rec.stored_grid("x", w["x"]).domain == "optical"
    assert rec.stored_grid("pd", w["pd"]).domain == "real"
    text = rec.summary()
    assert "shared [span]  domain=optical" in text and "name='x'  -> optical" in text
    assert "name='y'  -> optical" in text             # inside the detector's sub-chain too
    again = ws.Channels.from_recipe(json.loads(json.dumps(rec.recipe()))).waveforms()
    assert all(np.array_equal(again[k], w[k]) for k in w)


def test_a_shared_stage_of_general_ops_acts_on_a_field_as_a_field():
    """`use` hands the stage the domain the chain has reached: a stage that is only a `gain`
    cannot tell a field from an envelope by its samples, and must not have to."""
    rec = ws.Channels(
        {"a": Signal().carrier("nrz", n_ui=N_UI).eo().use("voa").photodetect(shot=False),
         "b": Signal().carrier("nrz", n_ui=N_UI).eo().photodetect(shot=False)},
        shared={"voa": Signal().gain(by=0.5).filter("butter", f_hz=30e9)}, seed=1, grid=G)
    w = rec.waveforms()
    lone = field().gain(by=0.5).filter("butter", f_hz=30e9).photodetect(shot=False).waveform()
    assert np.array_equal(w["a"], lone) and abs(w["a"].mean() / w["b"].mean() - 0.25) < 0.01
    # a stage that needs a voltage is refused on the field where the record is built
    with pytest.raises(ValueError, match="lossy.*optical field.*photodetect"):
        ws.Channels({"a": Signal().carrier("nrz", n_ui=N_UI).eo().use("ch")},
                    shared={"ch": Signal().lossy(loss_db=3.0, loss_at_ghz=5.0)}, seed=1, grid=G)


# ------------------------------------------------------------------ the general stages: no
@pytest.mark.parametrize("build,says", [
    (lambda s: s.offset(by=0.1), "combine(<a field chain>)"),
    (lambda s: s.delay(by=1e-11), "turned by its carrier's phase"),
    (lambda s: s.law("tanh", gain=2.0), "eo(kind=)"),
    (lambda s: s.trigger(pre_s=1e-9, post_s=1e-9), "triggers on a voltage"),
])
def test_a_stage_with_no_meaning_on_a_field_is_refused_where_written(build, says):
    s = field()
    with pytest.raises(ValueError) as e:
        build(s)
    msg = str(e.value)
    assert "optical field" in msg and says in msg and "photodetect(" in msg
    assert len(s.ops) == 2
    build(field().photodetect(shot=False))            # ... and accepted on the photocurrent


@pytest.mark.parametrize("how", ["product", "min", "max"])
def test_combine_on_a_field_is_a_sum_only(how):
    with pytest.raises(ValueError, match="optical field.*how='sum'.*photodetect"):
        field().combine(field(), how=how)
    r = field().combine(field()).recipe()
    r["ops"][-1]["how"] = how                         # ... and in a hand-written recipe
    with pytest.raises(ValueError, match="optical field.*how='sum'"):
        Signal.from_recipe(r).waveform()


def test_the_other_signal_is_refused_in_the_wrong_domain():
    # a voltage is not a peer of a field
    with pytest.raises(ValueError, match="REAL waveform.*joins an optical field.*`eo`"):
        field().combine(Signal().carrier("sine", f_hz=1e9)).waveform()
    # neither is an I/Q envelope, though it is complex too -- nor a field of an envelope
    with pytest.raises(ValueError, match="ends as an I/Q envelope.*is an optical field"):
        field().combine(Signal().iq("qpsk", prbs=9)).waveform()
    with pytest.raises(ValueError, match="ends as an optical field.*is an I/Q envelope"):
        (Signal(seed=1, grid=G).iq("qpsk", prbs=9)
         .combine(Signal().carrier("nrz", n_ui=N_UI).eo()).waveform())
    for stage in ("gain", "gate"):                    # `by` is a real factor or enable
        with pytest.raises(ValueError, match="COMPLEX.*`photodetect`"):
            getattr(field(), stage)(by=Signal().carrier("nrz", n_ui=N_UI).eo()).waveform()
    # a field where a voltage parameter is: the aggressor of a crosstalk
    with pytest.raises(ValueError, match="COMPLEX.*`photodetect`"):
        drive().crosstalk(aggressor=Signal().carrier("nrz", n_ui=N_UI).eo()).waveform()
    # a sequence of fields on a grid that STATES a voltage
    with pytest.raises(ValueError, match="COMPLEX.*`photodetect`"):
        Signal(seed=1, grid=Grid(fs=FS, baud=BAUD, n=N, domain="real")).sequence(
            [{"chain": Signal().carrier("nrz", n_ui=N_UI).eo()}]).waveform()


# ------------------------------------------------------------------ a chain that starts in the field
def test_a_grid_that_states_the_field_takes_a_captured_field_and_refuses_a_drive(tmp_path):
    e = field(kind="dml", alpha=3.0, er_db=8).fiber(length_km=2.0).waveform()
    sig = Signal(grid=GO).capture(values=e).fiber(length_km=2.0).photodetect(shot=False)
    assert sig.stage_kinds(domains=True) == [("source", "optical"), ("channel", "optical"),
                                             ("instrument", "real")]
    # the fibre after a capture is the fibre after the chain that made it
    whole = field(kind="dml", alpha=3.0, er_db=8).fiber(length_km=2.0).fiber(length_km=2.0)
    assert np.array_equal(sig.waveform(), detected(whole))
    path = tmp_path / "field.npy"
    np.save(path, e)
    assert np.array_equal(Signal(grid=GO).capture(path=str(path)).waveform(), e)
    # complex samples on a grid that says nothing are an I/Q record, as they always were
    assert Signal(grid=G).capture(values=e).grid.domain == "iq"
    with pytest.raises(ValueError, match="carrier.*STATES domain='optical'.*\\.eo\\(kind="):
        Signal(grid=GO).carrier("nrz", n_ui=N_UI)
    with pytest.raises(ValueError, match="iq.*STATES domain='optical'"):
        Signal(grid=GO).iq("qpsk", prbs=9)
    with pytest.raises(ValueError, match="fc_hz.*fiber\\(wavelength_nm=\\)"):
        Grid(fs=FS, n=N, domain="optical", fc_hz=193.4e12)
    with pytest.raises(ValueError, match="'real', 'iq' or 'optical'"):
        Grid(fs=FS, n=N, domain="field")


def test_a_voltage_measure_and_an_exporter_refuse_a_field_naming_photodetect(tmp_path):
    sig = field()
    e = sig.waveform()
    with pytest.raises(ValueError, match="optical field's.*after photodetect"):
        ws.to_sigmf(str(tmp_path / "f"), e, sig.stored_grid(e))
    with pytest.raises(ValueError, match="OPTICAL FIELD.*photodetect\\(\\)"):
        ws.eye_height(e, G, levels=2)
    with pytest.raises(ValueError, match="optical field is exported after photodetect"):
        ws.hdf5.counts_for(e)
    ws.eye_height(detected(field()), G, levels=2)
