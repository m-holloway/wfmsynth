"""A call that cannot mean what was written is refused BY NAME, where it is written, and the
message says what to write instead. Each case here used to render something else silently,
give two digests for one record, or fail with a bare Python error."""
import numpy as np
import pytest

import wfmsynth as ws
from wfmsynth import Signal, opkeys

G = ws.Grid(fs=64e9, baud=1e9, n=1 << 12)
GI = ws.Grid(fs=400e6, baud=25e6, n=1 << 12, domain="iq", fc_hz=1e9)
SYM = [1, 1j, -1, -1j] * 60


def data():
    return Signal(seed=1, grid=G).carrier("nrz", n_ui=64)


def env():
    return Signal(seed=1, grid=GI).iq("qpsk", prbs=9)


def says(fn, *words, kind=(ValueError, TypeError)):
    with pytest.raises(kind) as e:
        fn()
    for w in words:
        assert w in str(e.value), f"{w!r} not in: {e.value}"


# ------------------------------------------------------------------ the iq source
@pytest.mark.parametrize("con, kw, words", [
    (None, dict(symbols=SYM, prbs=7), ("exactly one source", "symbols= and prbs=")),
    (None, dict(symbols=SYM, bits=[0, 1, 1, 0]), ("exactly one source", "symbols= and bits=")),
    ("qpsk", dict(bits=[0, 1] * 60, prbs=7), ("exactly one source", "bits= and prbs=")),
    (None, dict(symbols=SYM, prbs_phase=5), ("prbs_phase= goes with prbs=", "symbols=")),
    (None, dict(symbols=SYM, prbs_seed=5), ("prbs_seed= goes with prbs=",)),
    (None, dict(symbols=SYM, n_bits=3), ("n_bits= goes with prbs=",)),
    ("qpsk", dict(bits=[0, 1] * 60, prbs_phase=5), ("prbs_phase= goes with prbs=", "bits=")),
    ("qpsk", dict(bits=[0, 1] * 60, n_bits=3), ("n_bits= goes with prbs=",)),
    ("qam16", dict(symbols=SYM), ("constellation", "symbols=")),
    ("qpsk", {}, ("exactly one source", "none of them")),
])
def test_iq_takes_one_symbol_source_and_only_its_parameters(con, kw, words):
    says(lambda: Signal(seed=1, grid=GI).iq(con, **kw), *words)
    # the same op written by hand in a recipe is refused when it renders
    op = {"op": "iq", **({} if con is None else {"constellation": con}), **kw}
    says(lambda: Signal.from_recipe({"seed": 1, "grid": env().recipe()["grid"],
                                     "ops": [op]}).waveform(), words[0])


def test_what_iq_reads_still_moves_the_record():
    base = env().transmitted()[0]
    for kw in (dict(prbs_seed=5), dict(prbs_phase=5)):
        assert not np.array_equal(Signal(seed=1, grid=GI).iq("qpsk", prbs=9, **kw)
                                  .transmitted()[0], base)
    # Signal(seed=) does not move the data: the phase is the source's own
    assert np.array_equal(Signal(seed=99, grid=GI).iq("qpsk", prbs=9).transmitted()[0], base)


# ------------------------------------------------------------------ the word another op uses
@pytest.mark.parametrize("fn, words", [
    (lambda: Signal(seed=1, grid=GI).iq("qam16", pattern="prbs15"), ("prbs=<order>", "carrier")),
    (lambda: Signal(seed=1, grid=G).carrier("nrz", prbs=7), ("pattern='prbs7'", "iq")),
    (lambda: data().trigger(pre_s=1e-9, post_s=3e-9, threshold=0.0), ("level=",)),
    (lambda: data().gate(by=Signal().carrier("sine", f_hz=1e8), level=0.2), ("threshold=",)),
    (lambda: Signal(seed=1, grid=G).switch(
        select=Signal().carrier("sine", f_hz=1e8), level=0.0,
        inputs=[Signal().carrier("dc"), Signal().carrier("dc")]),
     ("threshold=",)),
    (lambda: data().law("comparator", level=0.2), ("threshold=",)),
    (lambda: data().combine(Signal().carrier("sine", f_hz=1e9), kind="product"), ("how=",)),
    (lambda: data().delay(delay_s=1e-12), ("by=",)),
    (lambda: env().upconvert().downconvert(lpf=20e6, fs_out_hz=100e6), ("fs_out=",)),
    (lambda: env().upconvert().downconvert(lpf_hz=20e6), ("lpf=",)),
    (lambda: Signal(seed=1, grid=G).sequence([{"chain": Signal().carrier("dc"), "t_s": 1e-9}]),
     ("duration_s", "instant")),
])
def test_the_refusal_names_this_ops_word_for_it(fn, words):
    says(fn, *words)


def test_every_builder_refuses_an_unread_keyword_the_same_way():
    for fn in (lambda: data().gain(by=2.0, factor=3), lambda: data().offset(by=0.1, dc=1),
               lambda: data().gate(by=1.0, hold=True), lambda: data().delay(by=1e-12, taps=8),
               lambda: data().combine(Signal().carrier("dc"), weights=[1]),
               lambda: data().trigger(pre_s=1e-9, post_s=3e-9, op="x")):
        says(fn, "does not take", kind=TypeError)
    assert not any("op" in keys for keys in opkeys.OP_KEYS.values())   # never listed as a parameter
    assert all(k not in opkeys.OP_KEYS[op] for op, k in opkeys.SAID_AS)   # no dead hint


# ------------------------------------------------------------------ the other domain's op
@pytest.mark.parametrize("op, kw, instead", [
    ("rx_noise", dict(rms=0.01), "awgn("), ("digitize", dict(bits=8), "dac(bits=)"),
    ("store", dict(bits=8), "dac(bits=)"), ("agc", {}, "gain(by="),
    ("timing", dict(rj_ps=1.0), "phase_noise(profile=)"), ("offset", dict(by=0.1), "iq_offset("),
    ("law", dict(kind="tanh"), "dac(bits=)"),
])
def test_a_real_only_op_on_an_envelope_names_its_counterpart(op, kw, instead):
    says(lambda: getattr(env(), op)(**kw), "complex baseband", instead)


@pytest.mark.parametrize("op, kw, instead", [
    ("awgn", dict(snr_db=20, bw_hz=1e6), "rx_noise(rms=)"), ("iq_offset", dict(i=0.1), "offset(by=)"),
    ("dac", dict(bits=8), "digitize(bits=)"),
    ("phase_noise", dict(profile=[(1e4, -80), (1e6, -120)]), "timing("),
])
def test_an_envelope_only_op_on_a_voltage_names_its_counterpart(op, kw, instead):
    says(lambda: getattr(data(), op)(**kw), "real voltage", instead)


def test_a_voltage_source_on_an_iq_grid_is_told_what_starts_an_iq_record():
    for start in (lambda s: s.carrier("nrz", n_ui=64), lambda s: s.symbols([1, -1, 1, 1])):
        says(lambda: start(Signal(seed=1, grid=GI)), "I/Q grid", ".iq(", "without domain='iq'")


# ------------------------------------------------------------------ envelopes about two centres
def test_combining_envelopes_about_different_centres_is_refused():
    at = lambda fc: ws.Grid(fs=400e6, baud=25e6, n=1 << 12, domain="iq", fc_hz=fc)  # noqa: E731
    other = Signal(seed=2, grid=at(2.4e9)).iq("qpsk", prbs=7)
    says(lambda: env().combine({"from": other.recipe()}).waveform(),
         "2.4e+09", "1e+09", "freq_offset(f_hz=1.4e+09)")
    says(lambda: Signal(seed=1, grid=GI).sequence(
        [{"chain": {"from": other.recipe()}, "duration_s": 64 / 25e6},
         {"chain": Signal().iq("qpsk", prbs=9)}]).waveform(), "about 2.4e+09 Hz")
    # about the SAME centre, moved by its own freq_offset: the stated way, and it lands there
    blocker = Signal(seed=2, grid=GI).iq("qpsk", prbs=7).freq_offset(f_hz=40e6)
    x = env().combine({"from": blocker.recipe()}).waveform()
    spec = ws.rf.spectrum(x, GI, rbw_hz=1e6)
    there = ws.rf.band_power(spec, 23e6, 57e6) / ws.rf.band_power(spec, -17e6, 17e6)
    print(f"blocker 40 MHz above the centre: {10 * np.log10(there):.2f} dB against the signal")
    assert abs(10 * np.log10(there)) < 0.5                       # two unit-power signals


# ------------------------------------------------------------------ tolerances
def test_a_toleranced_design_parameter_is_drawn_then_checked():
    """`filter` and `law` check a design when the op is written, which used to happen before
    the tolerance was drawn: a bare TypeError. The drawn corner is the one in the response."""
    s = data().filter("butter", f_hz=ws.tol(1e9, 5), order=1, zero_phase=True)
    fc = s.ops[-1]["f_hz"]
    assert 0.95e9 <= fc <= 1.05e9 and fc != 1e9 and s.roles()[-1] == "unit/1/f_hz"
    tone = lambda f: Signal(seed=1, grid=G).carrier("sine", f_hz=f)        # noqa: E731
    probe = tone(fc)
    probe.ops.append(dict(s.ops[-1]))
    gain = 20 * np.log10(probe.waveform()[500:-500].std() / tone(fc).waveform()[500:-500].std())
    print(f"this unit's corner {fc / 1e9:.4f} GHz; gain there {gain:.3f} dB (-3.01)")
    assert abs(gain - -3.01) < 0.05
    other = s.unit(7)
    assert other.ops[-1]["f_hz"] != fc and not np.array_equal(other.waveform(), s.waveform())
    assert Signal.from_recipe(s.recipe()).sha256() == s.sha256()

    lim = data().law("tanh", sat=ws.tol(1.0, 5))
    assert 0.95 <= lim.ops[-1]["sat"] <= 1.05 and np.abs(lim.waveform()).max() < lim.ops[-1]["sat"]
    q = data().law("quantize", bits=ws.corners({"lo": 3, "hi": 5}), lo=-1.0, hi=1.0)
    bits = q.ops[-1]["bits"]
    assert bits in (3, 5) and len(np.unique(q.waveform())) <= 2 ** bits
    order = data().filter("butter", f_hz=1e9, order=ws.corners({"a": 2, "b": 4})).ops[-1]["order"]
    assert order in (2, 4)
    says(lambda: data().filter("butter", f_hz=ws.tol(60e9, 1)), "filter")   # still a named refusal


@pytest.mark.parametrize("fn", [
    lambda: Signal(seed=1, grid=G).carrier("nrz", n_ui=ws.tol(256, 5)),
    lambda: data().digitize(bits=ws.tol(8, 20)),
    lambda: data().filter("butter", f_hz=1e9, order=ws.tol(4, 30)),
    lambda: data().trigger(pre_s=1e-9, post_s=3e-9, n_segments=ws.tol(3, 10)),
    lambda: Signal.from_recipe({"seed": 1, "ops": [
        {"op": "carrier", "kind": "nrz", "n_ui": {"nom": 64, "tol_pct": 5}}]}),
])
def test_a_tolerance_on_a_whole_number_is_refused_naming_corners(fn):
    says(fn, "whole number", "ws.corners(")


def test_a_tolerance_on_a_continuous_value_written_as_an_int_still_draws():
    r = data().probe(r_source=ws.tol(50, 5)).ops[-1]["r_source"]         # 50 ohms is not a count
    assert 47.5 <= r <= 52.5 and r != 50


def test_corners_wants_names():
    says(lambda: ws.corners([3.0, 6.0]), "{name: value}", "list")


# ------------------------------------------------------------------ files
def test_the_iq_writers_refuse_a_signal_and_a_grid_with_no_centre(tmp_path):
    says(lambda: ws.to_sigmf(tmp_path / "a", env(), GI), "got a Signal", "sig.realize()",
         "sig.stored_grid(x)")
    says(lambda: ws.to_interleaved(tmp_path / "b", env()), "got a Signal", "sig.realize()")
    x = env().waveform()
    says(lambda: ws.to_sigmf(tmp_path / "c", x, ws.Grid(fs=400e6, n=len(x))),
         "no centre", "sig.stored_grid(x)")
    ws.to_sigmf(tmp_path / "d", x, env().stored_grid(x))
    assert ws.read_iq(tmp_path / "d", format="sigmf").grid.fc_hz == 1e9
