"""The OFDM source (`wfmsynth.ofdm`, GitHub #88), each claim against an analytic result or a
second method, with the number it came to in the assertion's message.

    the record            a brute-force sum of complex exponentials, sample by sample
    the receiver          returns the resource grid to floating-point rounding
    the occupied band     the Dirichlet-kernel spectrum of independent rectangular symbols
    the envelope          E|x|^4 = 2 - (2 - E|X|^4) / n_active, and the CCDF's distance to
                          the complex-Gaussian law falling as subcarriers are added
    EVM in noise          sqrt(n_active * scs_hz * N0) over the reference's RMS
    each correction       a known delay, rotation, level and two-path channel, recovered

Frames, gaps, a preamble and per-subcarrier constellations are not parameters of the op: they
are `sequence`, `gate` and `combine` around it, and each is shown here.
"""
from __future__ import annotations

import json
import warnings

import numpy as np
import pytest

import wfmsynth as ws
from wfmsynth import Grid, Signal
from wfmsynth import iq as IQ
from wfmsynth import ofdm as OFDM
from wfmsynth import physics as P
from wfmsynth import rf as RF

N_FFT, SCS, CP = 64, 312.5e3, 16
RATE = N_FFT * SCS                                     # 20 MSa/s
NUM = dict(n_fft=N_FFT, scs_hz=SCS, cp=CP, active=52)
SEED = 0x2F6E15            # a PRBS state away from the all-but-one-zero start (see the gotcha test)
EXACT = 1e-12              # the bound stated for a noise-free round trip; measured ~1e-15


@pytest.fixture(autouse=True)
def _quiet():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


def grid(n_symbols=50, over=1, **kw):
    return Grid(fs=RATE * over, n=(N_FFT + CP) * over * n_symbols, **kw)


def source(g=None, constellation="qam16", **kw):
    p = dict(NUM, constellation=constellation, prbs=15, prbs_seed=SEED)
    p.update(kw)
    return Signal(seed=1, grid=grid() if g is None else g).ofdm(**p)


# ------------------------------------------------------------------ the record itself
@pytest.mark.parametrize("over", [1, 3])
def test_the_record_is_the_sum_of_its_subcarriers(over):
    """An independent method: no FFT, each sample of two symbols as the stated finite sum."""
    g = grid(4, over)
    sig = source(g, pilots=[-21, -7, 7, 21], amp=0.7)
    x = sig.waveform()
    rg = OFDM.resource_grid(sig.ops[0], g)
    k = rg["subcarriers"]
    worst = 0.0
    for sym in (0, 3):
        a, n = int(rg["start"][sym]), int(rg["samples"][sym])
        t = (np.arange(a, a + n) - rg["useful"][sym]) / g.fs      # negative in the prefix
        want = 0.7 / np.sqrt(k.size) * (rg["grid"][sym][None, :]
                                        * np.exp(2j * np.pi * SCS * np.outer(t, k))).sum(axis=1)
        worst = max(worst, float(np.abs(x[a:a + n] - want).max()))
    assert worst < EXACT, worst
    cp = CP * over                                    # the prefix IS the symbol's own tail
    assert np.array_equal(x[:cp], x[N_FFT * over:N_FFT * over + cp])
    # unit-power values on every active subcarrier: the useful part's mean power is amp**2
    qpsk = source(g, constellation="qpsk", amp=0.7).waveform()
    u = int(rg["useful"][1])
    assert abs(np.mean(np.abs(qpsk[u:u + N_FFT * over]) ** 2) - 0.49) < EXACT


@pytest.mark.parametrize("kw", [
    {}, {"window": 8}, {"cp": [20, 16, 16, 12]}, {"pilots": [-21, -7, 7, 21]},
    {"active": [-30, -3, 0, 1, 2, 17]}, {"n_fft": 128, "scs_hz": SCS / 2, "active": 100},
    {"constellation": "bpsk"}, {"constellation": "qam64", "amp": 3.0}, {"n_symbols": 7}])
@pytest.mark.parametrize("over", [1, 4])
def test_noise_free_demodulation_returns_the_grid_exactly(kw, over):
    g = grid(40, over)
    sig = source(g, **kw)
    x = sig.waveform()
    sent, t_s = sig.transmitted()
    got = OFDM.demodulate(x, g, sig.ops[0])
    err = float(np.abs(got - kw.get("amp", 1.0) * sent).max())
    assert err < EXACT, err
    pl = OFDM.plan(sig.ops[0], g)
    assert sent.shape == (pl["n_symbols"], pl["subcarriers"].size) == got.shape
    assert np.array_equal(t_s, pl["useful"] / g.fs) and pl["oversample"] == over
    if "n_symbols" in kw:                             # a burst: the rest of the record is zero
        assert pl["n_symbols"] == 7 and not x[7 * 80 * over:].any()
    if kw.get("amp", 1.0) == 1.0:
        assert RF.evm(x, g, sent, ofdm=sig.ops[0])["evm"] < EXACT


def test_the_ground_truth_is_the_mapped_bits_in_the_stated_order():
    """The grid's data elements, read symbol by symbol and in ascending subcarrier, are
    `iq.map_bits` of the PRBS -- resolved here by hand, not by the op."""
    g = grid(6)
    pil = {"subcarriers": [-21, 7], "values": [1, 1j], "every": 2, "first": 1}
    sig = source(g, pilots=pil)
    rg = OFDM.resource_grid(sig.ops[0], g)
    assert list(rg["subcarriers"]) == list(range(-26, 0)) + list(range(1, 27))
    cols = [list(rg["subcarriers"]).index(k) for k in (-21, 7)]
    want = np.zeros((6, 52), bool)
    want[1::2, cols] = True
    assert np.array_equal(rg["pilot"], want)
    assert np.array_equal(rg["grid"][3, cols], [1, 1j])
    n_data = 6 * 52 - 6
    bits = P.prbs(15, 4 * n_data, seed=SEED)
    assert np.array_equal(rg["bits"], bits)
    assert np.array_equal(rg["grid"][~rg["pilot"]], IQ.map_bits("qam16", bits))
    assert np.array_equal(sig.transmitted()[0], rg["grid"])
    assert np.array_equal(rg["f_hz"], rg["subcarriers"] * SCS) and rg["noise_bw_hz"] == 52 * SCS


def test_a_staggered_pilot_pattern_is_several_sets_and_an_overlap_is_refused():
    g = grid(8)
    sets = [{"subcarriers": [-20, 4], "every": 4}, {"subcarriers": [-8, 16], "every": 4, "first": 2}]
    rg = OFDM.resource_grid(source(g, pilots=sets).ops[0], g)
    assert [int(r.sum()) for r in rg["pilot"]] == [2, 0, 2, 0, 2, 0, 2, 0]
    assert list(rg["subcarriers"][rg["pilot"][2]]) == [-8, 16]
    with pytest.raises(ValueError, match="two pilot sets claim the same element"):
        source(g, pilots=[{"subcarriers": [4]}, {"subcarriers": [4], "every": 3}]).waveform()


def test_it_draws_no_randomness_and_the_seeding_words_are_the_prbs_ones():
    a, b = source(), source()
    b.seed = 99
    assert a.roles() == [] and np.array_equal(a.waveform(), b.waveform())
    assert source().awgn(snr_db=20, bw_hz=52 * SCS).roles() == ["awgn/1"]
    assert not np.array_equal(a.waveform(), source(prbs_seed=SEED + 1).waveform())


# ------------------------------------------------------------------ the grid's rate
def test_a_grid_that_is_not_a_whole_multiple_of_the_fft_rate_is_refused_with_the_rate():
    with pytest.raises(ValueError) as e:
        Signal(grid=Grid(fs=30.72e6, n=4096)).ofdm(constellation="qpsk", prbs=9, **NUM).waveform()
    msg = str(e.value)
    assert "Grid(fs=20000000)" in msg and "Grid(fs=40000000) for 2x" in msg, msg
    assert "Nothing is resampled" in msg
    with pytest.raises(ValueError, match="shorter than one symbol"):
        Signal(grid=Grid(fs=RATE, n=79)).ofdm(constellation="qpsk", prbs=9, **NUM).waveform()
    # a whole multiple is the same signal on a finer grid: every 4th sample is the 1x record
    one = source(grid(10)).waveform()
    four = source(grid(10, 4)).waveform()
    assert np.abs(four[::4] - one).max() < EXACT


# ------------------------------------------------------------------ recipe, replay, knobs
@pytest.mark.parametrize("change", [
    {"cp": 8}, {"active": 48}, {"active": list(range(-26, 26))}, {"pilots": [-21, 7]},
    {"pilots": {"subcarriers": [-21, 7], "values": [1, -1]}}, {"constellation": "qpsk"},
    {"n_symbols": 30}, {"window": 4}, {"amp": 0.5}, {"prbs": 23}, {"prbs_seed": 77},
    {"prbs_phase": 5}, {"n_fft": 128, "scs_hz": SCS / 2}])
def test_every_parameter_moves_the_samples_and_the_content_address(change):
    base, moved = source(), source(**change)
    assert not np.array_equal(base.waveform(), moved.waveform())
    assert base.sha256() != moved.sha256()


def test_literal_bits_and_a_json_replay():
    g = grid(3)
    bits = np.random.default_rng(5).integers(0, 2, 3 * 52 * 2)
    sig = Signal(seed=1, grid=g).ofdm(constellation="qpsk", bits=bits, **NUM)
    sent, _ = sig.transmitted()
    assert np.array_equal(sent.ravel(), IQ.map_bits("qpsk", bits))
    again = Signal.from_recipe(json.loads(sig.to_json()))
    assert again.sha256() == sig.sha256() and np.array_equal(again.waveform(), sig.waveform())
    with pytest.raises(ValueError, match="pass exactly 312"):
        Signal(seed=1, grid=g).ofdm(constellation="qpsk", bits=bits[:-2], **NUM).waveform()


def test_a_named_set_records_its_name_and_its_numbers_and_replays_without_the_registry():
    OFDM.register("test_set", n_fft=N_FFT, scs_hz=SCS, cp=CP, active=52,
                  pilots=[-21, -7, 7, 21], source="tests/test_ofdm.py, not a standard")
    try:
        sig = Signal(seed=1, grid=grid()).ofdm("test_set", constellation="qpsk", prbs=15, prbs_seed=SEED)
        op = sig.recipe()["ops"][0]
        assert op["name"] == "test_set" and "not a standard" in op["source"]
        assert (op["n_fft"], op["scs_hz"], op["cp"], op["active"]) == (N_FFT, SCS, CP, 52)
        assert op["pilots"] == [-21, -7, 7, 21]
        plain = source(constellation="qpsk", pilots=[-21, -7, 7, 21])
        assert np.array_equal(sig.waveform(), plain.waveform())      # the name is a label
        assert sig.sha256() != plain.sha256()                        # ...that the digest covers
        with pytest.raises(ValueError, match="already states \\['cp'\\]"):
            Signal(grid=grid()).ofdm("test_set", cp=8, constellation="qpsk", prbs=15)
        with pytest.raises(ValueError, match="already registered"):
            OFDM.register("test_set", n_fft=64)
        text = sig.to_json()
    finally:
        OFDM.unregister("test_set")
    again = Signal.from_recipe(json.loads(text))
    assert np.array_equal(again.waveform(), sig.waveform()) and again.sha256() == sig.sha256()
    with pytest.raises(ValueError, match="no parameter set named 'test_set'"):
        Signal(grid=grid()).ofdm("test_set", constellation="qpsk", prbs=15)
    with pytest.raises(ValueError, match="does not take \\['constellation'\\]"):
        OFDM.register("bad", n_fft=64, constellation="qpsk")


# ------------------------------------------------------------------ refusals
@pytest.mark.parametrize("kw, says", [
    ({"active": 51}, "as a COUNT"),
    ({"active": [1, 2, 40]}, "distinct subcarriers in -32 .. 31"),
    ({"pilots": [0]}, "not in active="),
    ({"pilots": {"subcarriers": [7], "value": 1}}, "does not take \\['value'\\]"),
    ({"pilots": {"subcarriers": [7, 21], "values": [1]}}, "one per subcarrier"),
    ({"window": 17}, "longer than the shortest cyclic prefix"),
    ({"cp": 65}, "longer than the useful part"),
    ({"constellation": "qam256"}, "unknown constellation 'qam256'"),
    ({"constellation": None}, "needs constellation="),
    ({"prbs": None}, "exactly one source"),
    ({"bits": [0, 1]}, "exactly one source"),
    ({"n_symbols": 51}, "do not fit in a record"),
    ({"n_fft": 1}, "n_fft is a whole number >= 2"),
    ({"scs_hz": -1.0}, "must be positive"),
])
def test_what_cannot_be_rendered_is_refused_by_name(kw, says):
    with pytest.raises(ValueError, match=says):
        source(**kw).waveform()


def test_refusals_that_point_somewhere_else():
    with pytest.raises(ValueError, match="needs n_fft=, scs_hz=, cp=, active="):
        Signal(grid=grid()).ofdm(constellation="qpsk", prbs=9)
    with pytest.raises(TypeError, match="prbs_seed="):
        Signal(grid=grid()).ofdm(constellation="qpsk", prbs=9, seed=3, **NUM)
    with pytest.raises(ValueError, match="ofdm: this chain's grid STATES domain='real'"):
        Signal(grid=Grid(fs=RATE, n=800, domain="real")).ofdm(constellation="qpsk", prbs=9, **NUM)
    with pytest.raises(ValueError, match="every active element is a pilot"):
        source(pilots=list(range(-26, 0)) + list(range(1, 27))).waveform()
    with pytest.raises(ValueError, match="lead_in"):
        Signal(grid=grid(), lead_in=64).ofdm(constellation="qpsk", prbs=9, **NUM).waveform()
    # a real-only op after it is refused as after `iq`; the mixers and I/Q ops are not
    with pytest.raises(ValueError, match="takes domain 'real' only"):
        source().reflect(td_samples=8, gamma_s=0.2, gamma_l=0.2)
    with pytest.raises(ValueError, match="needs the symbol rate"):
        source().awgn(es_n0_db=20).waveform()              # an OFDM grid states no baud
    assert source().stage_kinds(domains=True) == [("source", "iq")]


# ------------------------------------------------------------------ events
def test_every_symbol_boundary_is_an_event_and_none_is_a_defect():
    g = grid(12, 2)
    sig = source(g, cp=[20, 12])
    x, ev = sig.realize()
    marks = [e for e in ev if e.kind == "ofdm_symbol"]
    pl = OFDM.plan(sig.ops[0], g)
    assert len(marks) == pl["n_symbols"] == 12
    assert [e.sample for e in marks] == list(pl["start"])
    assert [e.width for e in marks] == list(pl["samples"]) == [168, 152] * 6
    assert [e.params["symbol"] for e in marks] == list(range(12))
    assert marks[1].t_s == 168 / g.fs and marks[1].params["cp_s"] == 12 / RATE
    assert marks[1].params["t_useful_s"] == sig.transmitted()[1][1]
    assert marks[0].placement == "ofdm/0"
    assert not ev.mask().any() and ev.on_record() == []      # structure, not a label


# ------------------------------------------------------------------ frames come from the joins
def test_a_preamble_and_a_payload_are_a_sequence():
    g = grid(50)
    k = list(range(-26, 0)) + list(range(1, 27))
    known = [1.0 if i % 3 else -1.0 for i in range(52)]
    pre = Signal().ofdm(pilots={"subcarriers": k, "values": known}, n_symbols=2, **NUM)
    pay = Signal().ofdm(constellation="qam64", prbs=15, prbs_seed=SEED, **NUM)
    t_pre = 2 * 80 / RATE
    sig = Signal(seed=1, grid=g).sequence([{"chain": pre, "duration_s": t_pre}, {"chain": pay}])
    x, ev = sig.realize()
    sent, t_s = sig.transmitted()
    assert sent.shape == (50, 52) and np.array_equal(sent[:2], [known, known])
    assert np.allclose(t_s, (np.arange(50) * 80 + 16) / RATE, rtol=0, atol=1e-15)
    got = np.concatenate([OFDM.demodulate(x[:160], g, pre.ops[0]),
                          OFDM.demodulate(x[160:], g, pay.ops[0])])
    assert np.abs(got - sent).max() < EXACT
    late = OFDM.demodulate(x, g, dict(pay.ops[0], n_symbols=48), delay_s=t_pre)
    assert np.abs(late - sent[2:]).max() < EXACT
    marks = [e for e in ev if e.kind == "ofdm_symbol"]
    assert [e.sample for e in marks] == [80 * i for i in range(50)]      # on the record's clock
    assert [e.at for e in marks[:3]] == ["sequence/0/segments/0/chain"] * 2 + \
        ["sequence/0/segments/1/chain"]
    # a segment that ends inside a symbol did not send that symbol
    cut = Signal(seed=1, grid=g).sequence([{"chain": pay, "duration_s": 200 / RATE},
                                           {"chain": pre}])
    assert cut.transmitted()[0].shape[0] == 2 + 2
    # single-carrier symbols and rows of a resource grid are not one array: asked per segment
    gb = Grid(fs=RATE, baud=RATE / 8, n=4000)
    mixed = Signal(seed=1, grid=gb).sequence([
        {"chain": Signal().iq("bpsk", prbs=7), "duration_s": 160 / RATE}, {"chain": pay}])
    with pytest.raises(ValueError, match="do not send one kind of symbol"):
        mixed.transmitted()
    part = mixed.subchains()["sequence/0/segments/1/chain"]
    part.grid = gb
    assert part.transmitted()[0].shape == (50, 52)


def test_a_tdd_gap_is_a_gate():
    """Five symbols on, five off: the enable is a real square wave, and the symbols inside
    the on-time still demodulate exactly."""
    g = grid(40)
    period = 10 * 80 / RATE
    sig = source(g).gate(by=Signal().carrier("square", f_hz=1.0 / period, duty=0.5), threshold=0.0)
    x = sig.waveform()
    en = sig.subchain("gate/1/by") > 0.0
    full = source(g)
    assert np.array_equal(x, np.where(en, full.waveform(), 0.0)) and 0.45 < en.mean() < 0.55
    sent, _ = full.transmitted()
    got = OFDM.demodulate(x, g, full.ops[0])
    pl = OFDM.plan(full.ops[0], g)
    on = np.array([en[a:a + n].all() for a, n in zip(pl["start"], pl["samples"])])
    off = np.array([not en[a:a + n].any() for a, n in zip(pl["start"], pl["samples"])])
    assert on.sum() >= 15 and off.sum() >= 15
    assert np.abs(got[on] - sent[on]).max() < EXACT and not got[off].any()


def test_two_constellations_on_disjoint_subcarriers_are_a_combine():
    g = grid(20)
    lo = dict(NUM, active=list(range(-26, 0)), constellation="qpsk", prbs=15, prbs_seed=SEED)
    hi = dict(NUM, active=list(range(1, 27)), constellation="qam64", prbs=15, prbs_seed=SEED + 9)
    sig = Signal(seed=1, grid=g).ofdm(**lo).combine(Signal().ofdm(**hi))
    x = sig.waveform()
    other = sig.subchains()["combine/1/others/0"]
    for chain in (Signal(seed=1, grid=g).ofdm(**lo), other):
        chain.grid = g
        err = np.abs(OFDM.demodulate(x, g, chain.ops[0]) - chain.transmitted()[0]).max()
        assert err < EXACT, err


# ------------------------------------------------------------------ the occupied band
def _dirichlet_share(sc, lo_hz, hi_hz, over=1, m=1 << 16):
    """The share of an unwindowed record's power between two frequencies, analytically:
    independent symbols, each a rectangle of ``(n_fft + cp) * over`` samples, so the mean
    spectrum is a Dirichlet kernel squared on every active subcarrier."""
    fs, ns = RATE * over, (N_FFT + CP) * over
    f = (np.arange(m) - m // 2) * fs / m
    tot = np.zeros(m)
    for k in sc:
        u = np.pi * (f - k * SCS) / fs
        near = np.abs(np.sin(u)) < 1e-12
        tot += np.where(near, float(ns), np.sin(ns * u) / np.where(near, 1.0, np.sin(u))) ** 2
    return float(tot[(f >= lo_hz) & (f <= hi_hz)].sum() / tot.sum())


@pytest.mark.parametrize("active, over", [(52, 1), (52, 4), (list(range(5, 21)), 1)])
def test_the_occupied_band_is_the_active_subcarriers(active, over):
    g = grid(4000, over)
    sig = source(g, constellation="qpsk", prbs=23, active=active)
    sc = OFDM.plan(sig.ops[0], g)["subcarriers"]
    lo, hi = (sc.min() - 0.5) * SCS, (sc.max() + 0.5) * SCS
    spec = RF.spectrum(sig.waveform(), g, rbw_hz=SCS / 8)
    total = RF.band_power(spec, spec["f_hz"][0], spec["f_hz"][-1])
    share = RF.band_power(spec, lo, hi) / total
    want = _dirichlet_share(sc, lo, hi, over)
    assert abs(share - want) < 1e-3 and share > 0.97, (share, want)
    if not isinstance(active, int):                   # one-sided: the mirror band is empty
        assert RF.band_power(spec, -hi, -lo) / total < 0.005


def test_the_edge_window_lowers_the_skirts_and_leaves_the_grid_alone():
    g = grid(4000, 4)
    beyond = {}
    for w in (0, 8):
        sig = source(g, constellation="qpsk", prbs=23, window=w)
        x = sig.waveform()
        spec = RF.spectrum(x, g, rbw_hz=SCS / 8)
        total = RF.band_power(spec, spec["f_hz"][0], spec["f_hz"][-1])
        beyond[w] = 10 * np.log10(RF.band_power(spec, 30.5 * SCS, 34.5 * SCS) / total)
        assert np.abs(OFDM.demodulate(x, g, sig.ops[0]) - sig.transmitted()[0]).max() < EXACT
    assert beyond[0] - beyond[8] > 3.0, beyond        # measured: -32.6 -> -36.7 dB
    # the two ramps sum to one: a symbol repeated has no dip at its boundary
    k = list(range(-26, 0)) + list(range(1, 27))
    rep = Signal(grid=grid(4, 4)).ofdm(pilots={"subcarriers": k, "values": [1.0] * 52},
                                       n_fft=N_FFT, scs_hz=SCS, cp=0, active=52, n_symbols=3)
    assert np.array_equal(rep.waveform()[:256], rep.waveform()[256:512])
    ramp = OFDM._edge(32)
    assert np.abs(ramp + ramp[::-1] - 1.0).max() < 1e-15


# ------------------------------------------------------------------ the envelope's statistics
def test_the_envelope_approaches_the_complex_gaussian_law_as_subcarriers_are_added():
    """Two statements. The fourth moment is exact for independent data on n subcarriers,
    ``E|x|^4 = 2 - (2 - E|X|^4) / n`` (2 is the Gaussian's), and the CCDF's largest distance
    from ``exp(-level)`` over 0 .. 9 dB falls as n grows."""
    lv = np.arange(0.0, 9.01, 0.25)
    law = np.exp(-10 ** (lv / 10))
    dist, seen = {}, {}
    for n_active, n_fft in ((2, 64), (4, 64), (12, 64), (52, 64), (300, 512)):
        g = Grid(fs=n_fft * 15e3, n=(n_fft + n_fft // 8) * (460800 // n_fft))
        sig = Signal(seed=1, grid=g).ofdm(n_fft=n_fft, scs_hz=15e3, cp=n_fft // 8,
                                          active=n_active, constellation="qpsk", prbs=23, prbs_seed=SEED)
        x = sig.waveform()
        dist[n_active] = float(np.abs(RF.ccdf(x, levels_db=lv)["probability"] - law).max())
        p = np.abs(x) ** 2
        seen[n_active] = float(np.mean(p ** 2) / np.mean(p) ** 2)
        # measured 1.5006, 1.7531, 1.9168 | 1.9644, 1.9904: tight where the departure from 2
        # is large; at 52 and 300 it is within what one PRBS's 460800 samples scatter by
        tol = 0.005 if n_active <= 12 else 0.02
        assert abs(seen[n_active] - (2.0 - 1.0 / n_active)) < tol, (n_active, seen)
    # measured: 0.164, 0.042, 0.0085, 0.0031, 0.0019
    assert dist[2] > dist[4] > dist[12] > dist[52] > dist[300], dist
    assert dist[2] > 0.1 and dist[12] < 0.012 and dist[300] < 0.004, dist


def test_gotcha_the_default_prbs_start_is_one_constellation_point_for_many_symbols():
    """Not a defect of the op -- the bits are the bits -- but the first thing to know: an LFSR
    started in state 1 emits almost nothing but zeros at first, every subcarrier then carries
    the same point, and the inverse FFT of a constant is an impulse."""
    g = grid(40)
    cold = RF.ccdf(source(g, constellation="qpsk", prbs=31, prbs_seed=1).waveform())["papr_db"]
    warm = RF.ccdf(source(g, constellation="qpsk", prbs=31).waveform())["papr_db"]
    assert cold > 15.0 > 11.0 > warm, (cold, warm)       # measured: 15.6 against 9.7


# ------------------------------------------------------------------ EVM against the truth
@pytest.mark.parametrize("constellation", ["qpsk", "qam64"])
@pytest.mark.parametrize("snr_db", [10.0, 25.0])
def test_evm_in_white_noise_is_the_analytic_value(constellation, snr_db):
    """Noise of density N0 reaches each demodulated element with variance
    ``n_active * scs_hz * N0`` (in the grid's units), so EVM is its root over the RMS of the
    reference -- and 10**(-snr/20) when the SNR is stated in `noise_bw_hz`."""
    g = grid(2000)
    sig = source(g, constellation=constellation)
    bw = OFDM.plan(sig.ops[0], g)["noise_bw_hz"]
    noisy = sig.copy().awgn(snr_db=snr_db, bw_hz=bw)
    x = noisy.waveform()
    sent, _ = noisy.transmitted()
    got = RF.evm(x, g, sent, ofdm=noisy.ops[0])
    want = np.sqrt(bw * noisy.rf_report()[0]["n0"] / np.mean(np.abs(sent) ** 2))
    assert abs(got["evm"] / want - 1.0) < 0.01, (got["evm"], want)          # measured 0.9988
    assert abs(got["evm_db"] + snr_db) < 0.2, got["evm_db"]
    assert got["n_elements"] == 2000 * 52 and got["removed"] == {}
    assert got["evm_per_subcarrier"].shape == (52,) and got["evm_per_symbol"].shape == (2000,)
    assert abs(np.sqrt(np.mean(got["evm_per_subcarrier"] ** 2)) - got["evm"]) < 1e-12
    # what a fit absorbs: one coefficient per subcarrier over N symbols takes 1/N of the noise
    few = RF.evm(x[:80 * 8], g, sent[:8], ofdm=noisy.ops[0])["evm"]
    eq = RF.evm(x[:80 * 8], g, sent[:8], ofdm=noisy.ops[0], correct=("equalizer",))["evm"]
    assert abs(eq / few - np.sqrt(1 - 1 / 8)) < 0.03, (eq / few, np.sqrt(1 - 1 / 8))


def test_gain_is_counted_until_it_is_named():
    sig = source(amp=0.5)
    x, sent = sig.waveform(), sig.transmitted()[0]
    raw = RF.evm(x, sig.grid, sent, ofdm=sig.ops[0])
    assert abs(raw["evm"] - 0.5) < EXACT
    fit = RF.evm(x, sig.grid, sent, ofdm=sig.ops[0], correct="gain")
    assert fit["evm"] < EXACT and abs(fit["removed"]["gain_db"] - 20 * np.log10(0.5)) < 1e-9


@pytest.mark.parametrize("late", [3, 2.25])
def test_a_delay_is_found_as_the_slope_it_puts_across_the_subcarriers(late):
    """The record `late` FFT-rate samples late, built on a 4x grid so a quarter sample is a
    whole one. Inside the prefix that is a pure phase slope: found, and removed exactly."""
    g = grid(20, 4)
    sig = source(g)
    x, sent = sig.waveform(), sig.transmitted()[0]
    d = int(round(late * 4))
    y = np.concatenate([np.zeros(d, complex), x[:-d]])
    raw = RF.evm(y, g, sent, ofdm=sig.ops[0])
    fit = RF.evm(y, g, sent, ofdm=sig.ops[0], correct=("timing",))
    assert raw["evm"] > 0.5 and fit["evm"] < 1e-9, (raw["evm"], fit["evm"])
    assert abs(fit["removed"]["timing_s"] - late / RATE) < 1e-6 / RATE, fit["removed"]
    # told the delay instead, the receiver moves its window: the last symbol is then cut off
    # by the record's end, so 19 are asked for
    told = RF.evm(y, g, sent[:19], ofdm=dict(sig.ops[0], n_symbols=19), delay_s=late / RATE)
    assert told["evm"] < EXACT and told["removed"] == {}


def test_a_two_path_channel_is_counted_then_equalised_to_its_frequency_response():
    """`filter("fir", taps=[1, 0, 0, 0.4])`: an echo 3 samples late, inside the prefix. Every
    subcarrier is scaled by H(k) = 1 + 0.4*exp(-j*2*pi*3*k/64), so the raw EVM is 0.4 and the
    equaliser returns H."""
    sig = source()
    sent = sig.transmitted()[0]
    x = sig.copy().filter("fir", taps=[1.0, 0.0, 0.0, 0.4]).waveform()
    raw = RF.evm(x, sig.grid, sent, ofdm=sig.ops[0])
    k = raw["subcarriers"]
    h = 1 + 0.4 * np.exp(-2j * np.pi * 3 * k / N_FFT)
    assert abs(raw["evm"] - 0.4) < 1e-9, raw["evm"]
    rms = np.sqrt(np.mean(np.abs(sent) ** 2, axis=0) / np.mean(np.abs(sent) ** 2))
    assert np.abs(raw["evm_per_subcarrier"] - np.abs(h - 1) * rms).max() < 1e-9
    fit = RF.evm(x, sig.grid, sent, ofdm=sig.ops[0], correct=("equalizer",))
    assert fit["evm"] < EXACT and np.abs(fit["removed"]["equalizer"] - h).max() < EXACT


def test_a_frequency_error_shows_as_each_symbols_rotation():
    """500 Hz against 312.5 kHz subcarriers: each symbol turns by 2*pi*f*t, which
    `common_phase` returns per symbol; what it cannot remove is the leakage between
    subcarriers, about pi*(f/scs)/sqrt(3)."""
    f = 500.0
    sig = source()
    sent, t_s = sig.transmitted()
    x = sig.copy().freq_offset(f_hz=f).waveform()
    raw = RF.evm(x, sig.grid, sent, ofdm=sig.ops[0])
    fit = RF.evm(x, sig.grid, sent, ofdm=sig.ops[0], correct=("common_phase",))
    turn = np.unwrap(fit["removed"]["common_phase_rad"])
    slope = np.polyfit(t_s, turn, 1)[0] / (2 * np.pi)
    assert abs(slope / f - 1.0) < 2e-3, slope                       # measured 499.85 Hz
    leak = np.pi * (f / SCS) / np.sqrt(3)
    assert raw["evm"] > 0.3 and 0.8 * leak < fit["evm"] < 1.1 * leak, (raw["evm"], fit["evm"], leak)


@pytest.mark.parametrize("kw, says", [
    ({"correct": ("phase",)}, "'common_phase' is the rotation of each OFDM symbol"),
    ({"correct": ("frequency",)}, "leakage between subcarriers"),
    ({"correct": ("offset",)}, "subcarrier 0 alone"),
    ({"correct": ("gain", "gain")}, "at most once"),
    ({"reference": "decided", "constellation": "qam16"}, "'reference', 'constellation'"),
    ({"equalizer_taps": 5}, "one coefficient per subcarrier"),
    ({"skip": 25}, "nothing to measure"),
])
def test_evm_refuses_what_does_not_apply_to_an_ofdm_record(kw, says):
    sig = source()
    with pytest.raises(ValueError, match=says):
        RF.evm(sig.waveform(), sig.grid, sig.transmitted()[0], ofdm=sig.ops[0], **kw)


def test_evm_refuses_a_mismatched_truth_and_a_one_symbol_equaliser():
    sig = source()
    x, sent = sig.waveform(), sig.transmitted()[0]
    with pytest.raises(ValueError, match="sent has shape \\(49, 52\\)"):
        RF.evm(x, sig.grid, sent[:49], ofdm=sig.ops[0])
    with pytest.raises(ValueError, match="resource grid"):
        RF.evm(x, sig.grid, ofdm=sig.ops[0])
    with pytest.raises(ValueError, match="Count 2 symbols or more"):
        RF.evm(x[:80], sig.grid, sent[:1], ofdm=sig.ops[0], correct=("equalizer",))
    with pytest.raises(ValueError, match="one complex baseband record"):
        OFDM.demodulate(x.real, sig.grid, sig.ops[0])
    with pytest.raises(ValueError, match="outside the record"):
        OFDM.demodulate(x, sig.grid, sig.ops[0], delay_s=1e-6)


# ------------------------------------------------------------------ through the passband
def test_a_passband_round_trip_is_held_to_the_stated_low_pass():
    """16x the FFT rate, a carrier at 80 MHz, a real passband record and back: the error per
    element is within the passband error `iq.lowpass_error` measures for the stated filter.
    An unwindowed OFDM envelope's skirts fall only as 1/f**2, and `upconvert` refuses the
    part that would fold -- which is what `window=` or a following filter is for."""
    g = grid(40, 16, fc_hz=80e6)
    lpf = dict(pass_hz=9e6, stop_hz=30e6)
    with pytest.raises(ValueError, match="of the envelope's power lies outside"):
        source(g).upconvert().waveform()
    sig = source(g, window=8)
    sent = sig.transmitted()[0]
    rt = sig.copy().upconvert().downconvert(lpf=lpf)
    x = rt.waveform()
    assert x.dtype == complex and rt.transmitted()[0].shape == sent.shape
    got = RF.evm(x, rt.stored_grid(x), sent, ofdm=sig.ops[0], skip=1)
    bound = IQ.lowpass_error(lpf, g.fs)[0]
    assert got["evm"] < bound < 3e-4, (got["evm"], bound)           # measured 1.2e-4 < 2.1e-4


def test_the_module_is_public_and_the_op_is_registered():
    assert ws.ofdm is OFDM and "ofdm" in ws.__all__
    assert ws.compose.OP_KIND["ofdm"] == "source"
    assert set(OFDM.KEYS) | {"n"} == set(ws.opkeys.OP_KEYS["ofdm"])
