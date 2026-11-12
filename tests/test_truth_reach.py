"""Ground truth is reachable wherever the record is: a frame made of segments, one channel of
several, a dataset of several-channel units, a part inside a sub-chain. None of these is
re-derived by the user; each is checked here against the record it describes."""
import numpy as np
import pytest

import wfmsynth as ws
from wfmsynth import Signal, rf

GI = ws.Grid(fs=400e6, baud=25e6, n=1 << 15, domain="iq", fc_hz=1e9)
G = ws.Grid(fs=64e9, baud=1e9, n=1 << 13)
LO = [(1e5, -75), (1e6, -95), (1e7, -115)]
N_PRE = 256


def frame():
    return Signal(seed=1, grid=GI).sequence([
        {"chain": Signal().iq("qpsk", prbs=9), "duration_s": N_PRE / GI.baud},     # preamble
        {"chain": Signal().iq("qam16", prbs=15)}])                                # payload


# ------------------------------------------------------------------ a frame of segments
def test_transmitted_follows_a_sequence_and_scores_the_frame():
    sent, t_s = frame().transmitted()
    pre = Signal(seed=1, grid=GI).iq("qpsk", prbs=9).transmitted()[0]
    pay = Signal(seed=1, grid=GI).iq("qam16", prbs=15).transmitted()[0]
    n = len(sent)
    assert n == GI.n * GI.baud / GI.fs and np.array_equal(sent[:N_PRE], pre[:N_PRE])
    assert np.array_equal(sent[N_PRE:], pay[:n - N_PRE])              # the payload RESTARTS
    assert np.allclose(t_s, (np.arange(n) + 0.5) / GI.baud, rtol=0, atol=1e-15)
    lay = ws.sequence.layout(frame().ops[0]["segments"], GI)
    assert np.sum(t_s < lay[1]["t_start_s"]) == N_PRE                 # which segment, from t_s

    r = rf.evm(frame().waveform(), GI, sent)
    err = np.abs(r["received"] - r["reference_symbols"])
    k = np.arange(n)
    far = (np.abs(k - N_PRE) > 8) & (k > 8) & (k < n - 8)             # half a pulse from a cut
    evm = np.sqrt(np.mean(err[far] ** 2) / np.mean(np.abs(sent[far]) ** 2))
    print(f"frame EVM away from the cut {100 * evm:.3f} % (pulse truncation alone: 0.32 %)")
    assert evm < 0.004


def test_a_continuing_segment_reports_the_symbols_inside_it():
    s = Signal(seed=1, grid=GI).sequence([
        {"chain": Signal().iq("qpsk", prbs=9), "duration_s": N_PRE / GI.baud},
        {"chain": Signal().iq("qam16", prbs=15), "state": "continue"}])
    sent, t_s = s.transmitted()
    pay = Signal(seed=1, grid=GI).iq("qam16", prbs=15).transmitted()[0]
    assert np.array_equal(sent[N_PRE:], pay[N_PRE:len(sent)])         # as if running since t=0
    assert np.all(np.diff(t_s) > 0)


def test_transmitted_says_what_it_cannot_report():
    sel = Signal().carrier("square", f_hz=1e6)
    sw = Signal(seed=1, grid=GI).switch(select=sel, threshold=0.0, inputs=[
        Signal().iq("qpsk", prbs=9), Signal().iq("qam16", prbs=15)])
    with pytest.raises(ValueError, match="`switch`.*subchains"):
        sw.transmitted()
    one = sw.subchains()["switch/0/inputs/1"].transmitted()[0]        # the route it names
    assert np.array_equal(one, Signal(seed=1, grid=GI).iq("qam16", prbs=15).transmitted()[0])


def test_ebn0_reads_the_constellation_through_a_sequence_or_asks():
    same = Signal(seed=1, grid=GI).sequence([
        {"chain": Signal().iq("qpsk", prbs=9), "duration_s": N_PRE / GI.baud},
        {"chain": Signal().iq("qpsk", prbs=15)}]).awgn(ebn0_db=10)
    assert same.ops[-1]["bits_per_symbol"] == 2
    with pytest.raises(ValueError, match="needs bits_per_symbol="):   # QPSK then 16-QAM: no one Eb
        frame().awgn(ebn0_db=10).waveform()


# ------------------------------------------------------------------ one channel of several
def two():
    return ws.Channels(
        {"a": Signal().iq("qpsk", prbs=9).use("lo").awgn(es_n0_db=30),
         "b": Signal().iq("qam16", prbs=15).use("lo")},
        shared={"lo": Signal().phase_noise(profile=LO)}, seed=2, grid=GI)


def test_a_channels_signal_renders_exactly_as_the_channel_does():
    rec = two()
    x, ev = rec.realize()
    for name in rec.names:
        s = rec.signal(name)
        assert np.array_equal(s.waveform(), x[name])
        y, e = s.realize()
        assert np.array_equal(y, x[name]) and e.to_dicts() == ev[name].to_dicts()
        assert s.stored_grid(y).fc_hz == 1e9


def test_a_channels_ground_truth_is_reachable_through_its_signal():
    rec = two()
    a, b = rec.signal("a"), rec.signal("b")
    assert np.array_equal(a.transmitted()[0],
                          Signal(seed=2, grid=GI).iq("qpsk", prbs=9).transmitted()[0])
    # the shared oscillator's phase is ONE label for both channels, and it is the phase the
    # record carries: the un-noised channel is its clean envelope turned by exactly this
    phase = b.lo_phase()
    assert np.array_equal(phase, a.lo_phase())
    clean = Signal(seed=2, grid=GI).iq("qam16", prbs=15).waveform()
    off = np.abs(rec.waveform("b") - clean * np.exp(1j * phase)).max()
    print(f"channel b against its clean envelope turned by lo_phase(): worst {off:.1e}")
    assert off == 0.0
    rep = a.rf_report()
    assert [(r.get("chain"), r["op"], r["name"]) for r in rep] == [
        ("lo", 0, "phase_noise"), (None, 2, "awgn")]
    assert rep[0]["realised_rms_rad"] == pytest.approx(float(np.sqrt(np.mean(phase ** 2))))
    assert rep[1]["es_n0_db"] == pytest.approx(30.0)
    evm = rf.evm(rec.waveform("b"), GI, b.transmitted()[0], skip=8)["evm"]
    print(f"channel b against its own symbols: EVM {100 * evm:.2f} %, "
          f"LO {rep[0]['realised_rms_rad']:.4f} rad rms")
    assert abs(evm / rep[0]["realised_rms_rad"] - 1) < 0.1      # small-angle: EVM = rms phase


def test_a_channels_signal_reaches_sub_chains_and_shared_signals():
    rec = ws.Channels(
        {"d": Signal().use("tx").crosstalk(aggressor=Signal().carrier("nrz", n_ui=128)
                                           .rx_noise(rms=0.05), coupling=0.2)},
        shared={"tx": Signal().carrier("nrz", n_ui=128, pattern="prbs7")}, seed=4, grid=G)
    d = rec.signal("d")
    agg = d.subchain("crosstalk/1/aggressor")
    # the aggressor it returns is the one that was coupled: coupling it by hand onto the
    # shared signal gives the channel
    from wfmsynth import physics as P
    assert np.abs(agg).max() > 0.5
    assert np.array_equal(P.crosstalk(rec.waveform("tx"), agg, coupling=0.2), rec.waveform("d"))
    assert np.array_equal(rec.signal("tx").waveform(), rec.waveform("tx"))


def test_a_channels_signal_is_a_view_and_says_so():
    rec = two()
    a = rec.signal("a")
    for call, word in ((a.recipe, "rec.recipe()"), (a.sha256, "rec.recipe()"),
                       (lambda: a.contrast("awgn/2"), "rec.contrast"),
                       (lambda: a.unit(3), "rec.unit(seed)")):
        with pytest.raises(ValueError, match="channel 'a' of a several-channel record") as e:
            call()
        assert word in str(e.value)
    with pytest.raises(ValueError, match="no chain 'c'"):
        rec.signal("c")
    with pytest.raises(ValueError, match="'lo' is a stage"):
        rec.signal("lo")
    assert a.ops is not rec.chains["a"]                       # not the record's own list


# ------------------------------------------------------------------ datasets and labels
def board(seed):
    return ws.Channels(
        {"clk": Signal().carrier("nrz", n_ui=64, pattern="clock")
            .lossy(loss_db=ws.tol(6.0, 10), loss_at_ghz=0.5).rx_noise(rms=0.01),
         "data": Signal().carrier("nrz", n_ui=64).lossy(loss_db=3.0, loss_at_ghz=0.5)},
        seed=seed, grid=G)


def test_unit_dataset_takes_a_several_channel_builder():
    X, recipes = ws.unit_dataset(lambda rng: board(int(rng.integers(1 << 30))),
                                 n_units=3, n_captures=2, seed=1)
    assert set(X) == {"clk", "data"} and X["clk"].shape == (6, G.n) == X["data"].shape
    unit = [ws.unit_of(r) for r in recipes]
    loss = [ws.op_params(r, "lossy", chain="clk")["loss_db"] for r in recipes]
    assert unit[0] == unit[1] != unit[2] and loss[0] == loss[1] != loss[2]   # unit-major
    assert len(set(unit)) == 3 and all(5.4 <= v <= 6.6 for v in loss)
    for i, r in enumerate(recipes):                    # row i IS recipe i, in every channel
        w = ws.Channels.from_recipe(r).waveforms()
        assert all(np.array_equal(X[k][i], w[k].astype(np.float32)) for k in X)
    assert not np.array_equal(X["clk"][0], X["clk"][1])                      # two captures


def test_dataset_keeps_an_envelopes_imaginary_part():
    X, _ = ws.dataset(lambda rng: Signal(seed=1, grid=GI).iq(
        "qpsk", prbs=9, prbs_seed=int(rng.integers(1, 500))), 3)
    assert X.dtype == np.complex64 and np.abs(X.imag).max() > 0.1


def test_op_params_reads_a_several_channel_recipe():
    r = board(1).recipe()
    assert ws.op_params(r, "rx_noise")["rms"] == 0.01                # in one chain: no need
    assert ws.op_params(r, "lossy", chain="data")["loss_db"] == 3.0
    with pytest.raises(ValueError, match=r"\['clk', 'data'\] each hold a 'lossy' op.*chain="):
        ws.op_params(r, "lossy")
    with pytest.raises(KeyError, match="no chain 'dat'.*clk"):
        ws.op_params(r, "lossy", chain="dat")
    with pytest.raises(KeyError, match="no 'probe' op in any chain.*clk"):
        ws.op_params(r, "probe")
    with pytest.raises(ValueError, match="one Signal's"):
        ws.op_params(Signal(seed=1, grid=G).carrier("nrz").recipe(), "carrier", chain="clk")


# ------------------------------------------------------------------ a part inside a sub-chain
def test_a_nested_parts_role_is_written_like_every_other_role_and_shown_in_the_tree():
    s = (Signal(seed=1, grid=G).carrier("nrz", n_ui=128)
         .crosstalk(aggressor=Signal().carrier("nrz", n_ui=128)
                    .lossy(loss_db=ws.tol(6.0, 10), loss_at_ghz=0.5).rx_noise(rms=0.01),
                    coupling=ws.tol(0.1, 10)))
    assert s.roles() == ["crosstalk/1/aggressor/rx_noise/2",
                         "crosstalk/1/aggressor/unit/1/loss_db", "unit/1/coupling"]
    loss = s.ops[1]["aggressor"]["ops"][1]["loss_db"]
    assert f"loss_db={loss:.6g} (crosstalk/1/aggressor/unit/1/loss_db)" in s.summary()
    x = s.waveform()
    for r in s.roles():
        assert not np.array_equal(s.contrast(r), x), r
    rows = {r["role"]: r for r in ws.tolerance.resolved(s.recipe())}
    assert rows["crosstalk/1/aggressor/unit/1/loss_db"]["value"] == loss
    rec = ws.Channels({"d": Signal().carrier("nrz", n_ui=128).crosstalk(
        aggressor=Signal().carrier("nrz", n_ui=128).lossy(loss_db=ws.tol(6.0, 10),
                                                          loss_at_ghz=0.5), coupling=0.1)},
        seed=1, grid=G)
    assert rec.roles() == ["d/crosstalk/1/aggressor/unit/1/loss_db"]
    assert "(d/crosstalk/1/aggressor/unit/1/loss_db)" in rec.summary()
