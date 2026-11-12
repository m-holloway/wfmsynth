"""GitHub #80: two general stages, a named transfer law and a stated linear filter.

`law` and `filter` exist for a shape that has no named op -- a comparator, a rectifier, a dead
zone, a Chebyshev filter, an integrator -- so the claims worth testing are the ones a general
stage is most likely to get quietly wrong:

  * each kind IS the curve or the response it is named for, against a closed form or an
    independent method (`wfmsynth.validate` holds the analytic set; this file adds the
    independent solvers and the cases a single sine does not reach);
  * the recipe is DATA: every parameter of every kind moves both the samples and the content
    address, a recipe round-trips through JSON to the same samples, and a parameter a kind does
    not read is refused rather than recorded;
  * ground truth is EMITTED: a comparator's transitions arrive as events, and a filter's group
    delay is reported, so neither has to be re-derived from the samples;
  * what cannot be done honestly is REFUSED by name -- an unstable or unresolved design, a law
    on a complex field, a zero-phase request with no zero-phase form;
  * and nothing that existed before moved: the general Bessel is the front end's Bessel to the
    bit, and the named ops are pointed to rather than respelled.
"""
from __future__ import annotations

import json

import numpy as np
import pytest
from scipy import signal as sig
from scipy.optimize import brentq

import wfmsynth as ws
from wfmsynth import compose as C
from wfmsynth import stages as ST
from wfmsynth.compose import Signal
from wfmsynth.grid import Grid

FS, N = 1e9, 1 << 13
G = Grid(fs=FS, baud=50e6, n=N)


def _sine(amp=1.0, cycles=19):
    return Signal(seed=1, grid=G).carrier("sine", cycles=cycles, amp=amp)


def _noise():
    return np.random.default_rng(7).standard_normal(N)


# One working spelling of every kind, and for each of its parameters a second value. The sweep
# below is only a guarantee if it is complete, so completeness is asserted, not assumed.
LAWS = {
    "tanh": (dict(gain=2.0, sat=0.7), dict(gain=2.5, sat=0.5)),
    "poly": (dict(coeffs=[0.0, 1.0, 0.3]), dict(coeffs=[0.0, 1.0, 0.4])),
    "limiter": (dict(lo=-0.4, hi=0.5), dict(lo=-0.3, hi=0.6)),
    "deadzone": (dict(width=0.4, center=0.1), dict(width=0.5, center=0.2)),
    "rectify": (dict(mode="half"), dict(mode="full")),
    "quantize": (dict(bits=4, lo=-1.0, hi=1.0), dict(bits=5, lo=-0.9, hi=0.9)),
    "comparator": (dict(threshold=0.1, hysteresis=0.2, v_lo=0.0, v_hi=3.3, initial="lo"),
                   dict(threshold=0.2, hysteresis=0.4, v_lo=-1.0, v_hi=1.0, initial="hi")),
    "diode": (dict(r_load_ohm=1e3, i_sat=1e-12, ideality=1.0, v_thermal=0.025852),
              dict(r_load_ohm=2e3, i_sat=1e-9, ideality=1.5, v_thermal=0.03)),
}
_SOS = sig.butter(3, 0.2, output="sos").tolist()
FILTERS = {
    "butter": (dict(f_hz=40e6, order=3, btype="low", zero_phase=False),
               dict(f_hz=60e6, order=5, btype="high", zero_phase=True)),
    "bessel": (dict(f_hz=40e6, order=3, btype="low", zero_phase=False),
               dict(f_hz=60e6, order=5, btype="high", zero_phase=True)),
    "cheby1": (dict(f_hz=40e6, order=3, btype="low", zero_phase=False, ripple_db=1.0),
               dict(f_hz=60e6, order=5, btype="high", zero_phase=True, ripple_db=2.0)),
    "elliptic": (dict(f_hz=40e6, order=3, btype="low", zero_phase=False, ripple_db=1.0,
                      stop_db=40.0),
                 dict(f_hz=60e6, order=5, btype="high", zero_phase=True, ripple_db=2.0,
                      stop_db=50.0)),
    "rrc": (dict(beta=0.3, baud=50e6, span_ui=8, zero_phase=False),
            dict(beta=0.5, baud=40e6, span_ui=6, zero_phase=True)),
    "fir": (dict(taps=[0.25, 0.5, 0.25], zero_phase=False),
            dict(taps=[0.2, 0.6, 0.2], zero_phase=True)),
    "sos": (dict(sos=_SOS, zero_phase=False),
            dict(sos=sig.butter(3, 0.3, output="sos").tolist(), zero_phase=True)),
    "zpk": (dict(poles=[-2e8], zeros=[-9e8], gain=2e8 / 9e8, zero_phase=False),
            dict(poles=[-3e8], zeros=[-8e8], gain=0.5, zero_phase=True)),
    "integrate": (dict(tau_s=1e-8), dict(tau_s=2e-8)),
    "differentiate": (dict(tau_s=1e-8), dict(tau_s=2e-8)),
    "doppler": (dict(f_hz=40e6, shape="jakes"), dict(f_hz=60e6, shape="flat")),
}


def _knobs(table):
    return [(kind, key) for kind, (base, _) in table.items() for key in base]


def test_the_sweep_names_every_kind_and_every_parameter_of_each():
    # the envelope laws act on an I/Q record and are swept in tests/test_envelope_law.py
    assert set(LAWS) | set(ST.ENVELOPE_LAWS) == set(ws.LAW_KINDS)
    assert set(FILTERS) == set(ws.FILTER_KINDS)
    for table, spec in ((LAWS, ST._LAW), (FILTERS, ST._FILTER)):
        for kind, (base, other) in table.items():
            assert set(base) == set(other) == set(spec[kind]), kind


@pytest.mark.parametrize("kind,key", _knobs(LAWS))
def test_every_law_parameter_moves_the_samples_and_the_digest(kind, key):
    base, other = LAWS[kind]
    a = _sine().law(kind, **base)
    b = _sine().law(kind, **{**base, key: other[key]})
    assert a.sha256() != b.sha256()
    if (kind, key) == ("comparator", "initial"):
        # `initial` only decides when the first sample is INSIDE the hysteresis band -- which a
        # sine starting at zero, against a band of 0.0 to 0.2, is.
        assert a.waveform()[0] == 0.0 and b.waveform()[0] == 3.3
    assert not np.array_equal(a.waveform(), b.waveform())


@pytest.mark.parametrize("kind,key", _knobs(FILTERS))
def test_every_filter_parameter_moves_the_samples_and_the_digest(kind, key):
    base, other = FILTERS[kind]
    a = Signal(seed=1, grid=G).carrier("nrz", n_ui=32, pattern="prbs7").filter(kind, **base)
    b = (Signal(seed=1, grid=G).carrier("nrz", n_ui=32, pattern="prbs7")
         .filter(kind, **{**base, key: other[key]}))
    assert a.sha256() != b.sha256()
    assert not np.array_equal(a.waveform(), b.waveform())


@pytest.mark.parametrize("op,kind", [("law", k) for k in LAWS] + [("filter", k) for k in FILTERS])
def test_a_recipe_is_data_and_replays_to_the_same_samples(op, kind):
    base = (LAWS if op == "law" else FILTERS)[kind][0]
    s = getattr(Signal(seed=4, grid=G).carrier("nrz", n_ui=32, pattern="prbs7"), op)(kind, **base)
    back = Signal.from_recipe(json.loads(json.dumps(s.recipe())))
    assert back.sha256() == s.sha256()
    assert np.array_equal(back.waveform(), s.waveform())
    assert s.roles() == [] and s.ops[-1]["kind"] == kind      # no randomness, nothing to re-roll


def test_arrays_and_complex_poles_are_stored_as_plain_json():
    pole = complex(-1e8, 3e8)
    s = (Signal(seed=1, grid=G).carrier("nrz", n_ui=32)
         .filter("zpk", poles=np.array([pole, pole.conjugate()]), gain=1e17)
         .filter("fir", taps=np.array([0.5, 0.5]))
         .law("poly", coeffs=np.array([0.0, 1.0])))
    assert s.ops[1]["poles"] == [[-1e8, 3e8], [-1e8, -3e8]]
    assert s.ops[2]["taps"] == [0.5, 0.5] and s.ops[3]["coeffs"] == [0.0, 1.0]
    assert np.array_equal(Signal.from_recipe(json.loads(s.to_json())).waveform(), s.waveform())


# ------------------------------------------------------------------ the laws, independently
def test_diode_matches_an_independent_root_finder_on_the_circuit_equation():
    r, i_s, a = 470.0, 2e-9, 1.3 * 0.025852
    v_in = np.array([-0.2, 0.0, 0.1, 0.3, 0.5, 0.7, 1.0, 5.0, 12.0, 30.0, 400.0])
    got = ST.apply_law(v_in, "diode", r_load_ohm=r, i_sat=i_s, ideality=1.3)
    want = np.array([brentq(lambda vo: vo + a * np.log1p(vo / (r * i_s)) - v,
                            -r * i_s * (1 - 1e-12), abs(v) + 1.0, xtol=1e-15, rtol=1e-15)
                     for v in v_in])
    err = np.abs(got - want).max()
    assert err < 1e-9, err
    # deep in reverse the load carries the saturation current and nothing else
    assert ST.apply_law(np.array([-50.0]), "diode", r_load_ohm=r, i_sat=i_s)[0] == -r * i_s
    # and the two solvers meet without a step: exact W below L = 30 (1.36 V here), Newton above
    v = np.linspace(1.2, 1.5, 3001)
    y = ST.apply_law(v, "diode", r_load_ohm=r, i_sat=i_s, ideality=1.3)
    assert np.diff(y).min() > 0 and np.abs(np.diff(y, 3)).max() < 1e-11


def test_comparator_holds_inside_the_band_and_initial_decides_only_there():
    x = np.array([0.0, 0.05, -0.05, 0.3, 0.05, -0.05, -0.3, 0.0])
    kw = dict(threshold=0.0, hysteresis=0.4)
    assert ST.apply_law(x, "comparator", **kw).tolist() == [0, 0, 0, 1, 1, 1, 0, 0]
    assert ST.apply_law(x, "comparator", initial="hi", **kw).tolist() == [1, 1, 1, 1, 1, 1, 0, 0]
    # outside the band the first sample decides, whatever `initial` says
    assert ST.apply_law(x + 1.0, "comparator", initial="lo", **kw)[0] == 1.0


def test_comparator_events_index_the_delivered_edges():
    s = _sine().law("comparator", threshold=0.2, hysteresis=0.1, v_lo=0.0, v_hi=3.3)
    y, ev = s.realize()
    assert np.array_equal(y, s.waveform())                 # collecting events changes nothing
    assert len(ev) == 2 * 19 and set(np.unique(y)) == {0.0, 3.3}
    for e in ev:
        assert e.kind == "transition" and e.placement == "law/1"
        assert y[e.sample] == e.params["level"] != y[e.sample - 1]
        assert (e.sample - 1) / FS <= e.t_s <= e.sample / FS
        assert e.params["trip"] == pytest.approx(0.2 + 0.05 * e.direction)
    assert json.dumps(ev.to_dicts())                       # and they are data too


def test_comparator_events_under_a_lead_in_are_on_the_delivered_window():
    g = Grid(fs=256e9, baud=32e9, n=1 << 12)

    def chain(lead):
        return (Signal(seed=2, grid=g, lead_in=lead).carrier("nrz", pattern="prbs7", n_ui=512)
                .lossy(loss_db=6.0, loss_at_ghz=16.0, causal=True)
                .law("comparator", hysteresis=0.2, v_lo=0.0, v_hi=1.0))
    y, ev = chain(True).realize()
    assert len(y) == g.n and len(ev) > 100
    for e in ev:
        assert 0 <= e.sample < g.n and y[e.sample] == e.params["level"]
        assert abs(e.t_s - e.sample / g.fs) <= 1.0 / g.fs
    edges = np.flatnonzero(np.diff(y)) + 1
    assert [e.sample for e in ev] == edges.tolist()        # every edge, and only the edges


# --------------------------------------------------------------- the filters, independently
def test_bessel_order_4_is_the_front_end_to_the_bit():
    g = Grid(fs=256e9, baud=32e9, n=1 << 12)
    src = lambda: Signal(seed=3, grid=g).carrier("nrz", n_ui=32, pattern="prbs7")     # noqa: E731
    for bw in (20e9, 32e9, 60e9):
        assert np.array_equal(src().scope(bw_hz=bw).waveform(),
                              src().filter("bessel", f_hz=bw).waveform())


def test_fir_and_sos_are_applied_as_given_from_a_settled_start():
    x = _noise() + 2.0
    taps = [0.1, 0.4, 0.3, 0.2]
    y = ST.apply_filter(x, FS, "fir", taps=taps)
    full = np.convolve(np.r_[np.full(64, x[0]), x], taps)[64:64 + N]
    assert np.abs(y - full).max() < 1e-12
    y = ST.apply_filter(x, FS, "sos", sos=_SOS)
    ref = sig.sosfilt(_SOS, np.r_[np.full(4000, x[0]), x])     # from rest, long before
    assert np.abs(y - ref[4000:]).max() < 1e-12


@pytest.mark.parametrize("kind", [k for k in FILTERS if k != "integrate"])
def test_a_constant_goes_in_and_a_constant_comes_out(kind):
    # No turn-on transient at the head: the line is taken to have been at x[0] for ever.
    for zp in ((False, True) if "zero_phase" in FILTERS[kind][0] else (False,)):
        kw = dict(FILTERS[kind][0])
        if "zero_phase" in kw:
            kw["zero_phase"] = zp
        y = ST.apply_filter(np.full(2048, 0.7), FS, kind, **kw)
        assert np.ptp(y) < 1e-12, (kind, zp, np.ptp(y))


def test_zpk_single_pole_is_the_rc_step_response():
    tau = 20e-9
    x = np.r_[np.zeros(100), np.ones(N - 100)]
    y = ST.apply_filter(x, FS, "zpk", poles=[-1 / tau], gain=1 / tau)
    t = (np.arange(N - 100) + 0.5) / FS          # the bilinear rule is the trapezoid: half a sample
    err = np.abs(y[100:] - (1 - np.exp(-t / tau))).max()
    assert err < (1 / (FS * tau)) ** 2 / 8, err   # measured 3.0e-4 at T/tau = 0.05


def test_reported_group_delay_is_the_slope_of_the_phase():
    for kind, kw in (("butter", dict(f_hz=80e6, order=5)),
                     ("elliptic", dict(f_hz=80e6, order=4, ripple_db=1.0, stop_db=50.0)),
                     ("rrc", dict(beta=0.25, baud=50e6))):
        d = ST.design_filter(FS, kind, **kw)
        for w0 in (0.05, 0.2, 0.4):
            dw = 1e-5
            ph = np.unwrap(np.angle(ST._response(d, np.array([w0 - dw, w0 + dw]))))
            numeric = -(ph[1] - ph[0]) / (2 * dw)
            got = ST.describe_filter(FS, kind, at_hz=w0 * FS / (2 * np.pi), **kw)
            assert got["group_delay_s"] * FS == pytest.approx(numeric, abs=1e-5), (kind, w0)


def test_a_symmetric_fir_delays_by_exactly_half_its_length_and_zero_phase_takes_it_back():
    taps = ST.design_filter(FS, "rrc", beta=0.3, baud=50e6)["taps"]
    rep = ST.describe_filter(FS, "rrc", beta=0.3, baud=50e6)
    assert rep["group_delay_s"] * FS == pytest.approx((len(taps) - 1) / 2, abs=1e-9)
    x = _noise()
    k = (len(taps) - 1) // 2
    causal = ST.apply_filter(x, FS, "rrc", beta=0.3, baud=50e6)
    centred = ST.apply_filter(x, FS, "rrc", beta=0.3, baud=50e6, zero_phase=True)
    assert np.abs(causal[2 * k:] - centred[k:-k]).max() < 1e-12


def test_filter_delays_lists_every_filter_op_and_only_those():
    s = (Signal(seed=1, grid=G).carrier("nrz", n_ui=32, pattern="prbs7").filter("butter", f_hz=40e6)
         .scope(bw_hz=200e6).filter("fir", taps=[0.5, 0.0, 0.5]).filter("integrate"))
    rep = s.filter_delays()
    assert [r["op"] for r in rep] == [1, 3, 4] and [r["kind"] for r in rep] == [
        "butter", "fir", "integrate"]
    assert rep[1]["group_delay_s"] == pytest.approx(1.0 / FS) and rep[2]["group_delay_s"] == 0.0
    assert rep[0]["at_hz"] == 0.0 and rep[0]["group_delay_s"] > 0
    band = ST.describe_filter(FS, "butter", f_hz=[40e6, 90e6], btype="bandpass")
    assert 40e6 < band["at_hz"] < 90e6                      # where it passes, since DC is blocked


def test_zero_phase_iir_does_not_wrap_and_does_not_depend_on_its_padding():
    x = np.cumsum(_noise()) * 0.05                          # ends at different levels
    kw = dict(f_hz=5e6, order=4)
    y = ST.apply_filter(x, FS, "butter", zero_phase=True, **kw)
    big = np.r_[np.full(20000, x[0]), x, np.full(20000, x[-1])]
    ref = ST.apply_filter(big, FS, "butter", zero_phase=True, **kw)[20000:20000 + N]
    assert np.abs(y - ref).max() < 1e-9


def test_a_real_filter_acts_on_both_parts_of_a_complex_record_alike():
    rng = np.random.default_rng(3)
    i, q = rng.standard_normal(N), rng.standard_normal(N)
    for kind, base in ((k, v[0]) for k, v in FILTERS.items()):
        for zp in (False, True) if "zero_phase" in base else (False,):
            kw = {**base, **({"zero_phase": zp} if "zero_phase" in base else {})}
            y = ST.apply_filter(i + 1j * q, FS, kind, **kw)
            want = ST.apply_filter(i, FS, kind, **kw) + 1j * ST.apply_filter(q, FS, kind, **kw)
            assert np.iscomplexobj(y) and np.abs(y - want).max() < 1e-9 * np.abs(want).max(), kind


# ------------------------------------------------------------------------------- refusals
@pytest.mark.parametrize("call,exc,match", [
    (lambda s: s.law("clip", lo=0), ValueError, "unknown kind 'clip'.*law kinds: tanh"),
    (lambda s: s.law("compression"), ValueError, r"nonlinearity\(compression"),
    (lambda s: s.law("tanh", hysteresis=0.1), TypeError, "does not read 'hysteresis'.*comparator"),
    (lambda s: s.law("rectify"), TypeError, "needs mode"),
    (lambda s: s.law("rectify", mode="both"), ValueError, "'half'.*'full'"),
    (lambda s: s.law("limiter"), TypeError, "lo=, hi= or both"),
    (lambda s: s.law("quantize", bits=8), TypeError, "needs lo, hi"),
    (lambda s: s.law("quantize", bits=2.5, lo=0, hi=1), ValueError, "whole number of bits"),
    (lambda s: s.law("comparator", hysteresis=-1), ValueError, "cannot be negative"),
    (lambda s: s.law("diode"), TypeError, "needs r_load_ohm"),
    (lambda s: s.law("tanh", gian=2.0), TypeError, "does not read 'gian' -- did you mean 'gain'"),
    (lambda s: s.filter("gaussian", f_hz=1e6), ValueError, r"scope\(kind='gaussian'"),
    (lambda s: s.filter("ctle"), ValueError, r"ctle\(fz_ghz"),
    (lambda s: s.filter("butter"), TypeError, "needs f_hz"),
    (lambda s: s.filter("butter", f_hz=FS / 2), ValueError, "not resolved on this grid.*Nyquist"),
    (lambda s: s.filter("butter", f_hz=[2e6, 1e6], btype="bandpass"), ValueError, "lowest first"),
    (lambda s: s.filter("butter", f_hz=1e6, btype="lowpass"), ValueError, "btype is low, high"),
    (lambda s: s.filter("butter", f_hz=1e6, ripple_db=1.0), TypeError,
     "does not read 'ripple_db'.*cheby1"),
    (lambda s: s.filter("bessel", f_hz=1.0, order=12), ValueError, "not realisable in double"),
    (lambda s: s.filter("elliptic", f_hz=1e6, ripple_db=3.0, stop_db=2.0), ValueError,
     "must exceed ripple_db"),
    (lambda s: s.filter("rrc", beta=0.5, baud=0.9e9), ValueError, "above this grid's Nyquist"),
    (lambda s: s.filter("sos", sos=[[1, 0, 0, 1, -1.01, 0]]), ValueError, "unstable.*1.01"),
    (lambda s: s.filter("sos", sos=[[1, 0, 0, 1, -1.0, 0]]), ValueError, "filter\\('integrate'\\)"),
    (lambda s: s.filter("sos", sos=[[1, 0, 0, 2, 0, 0]]), ValueError, "a0 must be 1"),
    (lambda s: s.filter("zpk", poles=[1e6]), ValueError, "unstable.*NEGATIVE real part"),
    (lambda s: s.filter("zpk", poles=[[-1e6, 5e6]]), ValueError, "conjugate pairs"),
    (lambda s: s.filter("zpk", poles=[-1e6], zeros=[-1e5, -2e5]), ValueError, "not a proper"),
    (lambda s: s.filter("zpk", poles=[-2 * np.pi * 0.6e9]), ValueError, "not resolved"),
    (lambda s: s.filter("fir", taps=[1.0, 0.5], zero_phase=True), ValueError,
     "SYMMETRIC, odd-length"),
    (lambda s: s.filter("integrate", zero_phase=True), TypeError, "no\\s+zero-phase form"),
    (lambda s: s.filter("differentiate", tau_s=0.0), ValueError, "must be positive"),
])
def test_what_cannot_be_done_honestly_is_refused_on_the_line_that_wrote_it(call, exc, match):
    s = _sine()
    with pytest.raises(exc, match="(?s)" + match):
        call(s)
    assert len(s.ops) == 1                                   # and nothing was recorded


def test_a_recipe_carrying_a_bad_stage_is_refused_when_it_renders():
    # the builder is not the only way in: a recipe is data and may have been written by hand
    r = _sine().law("rectify", mode="half").recipe()
    r["ops"][-1]["mode"] = "quarter"
    with pytest.raises(ValueError, match="'half'"):
        Signal.from_recipe(r).waveform()
    r = _sine().filter("butter", f_hz=1e6).recipe()
    r["ops"][-1]["f_hz"] = 9e9
    with pytest.raises(ValueError, match="not resolved on this grid"):
        Signal.from_recipe(r).waveform()


def test_a_law_refuses_a_complex_field_rather_than_dropping_a_part():
    with pytest.raises(ValueError, match="complex.*photodetect"):
        ST.apply_law(np.ones(8) + 1j, "tanh")


def test_a_filter_needs_a_rate_and_a_waveform():
    with pytest.raises(ValueError, match="known sample rate"):
        Signal(seed=1).symbols([1, -1, 1, 1]).filter("butter", f_hz=1e6).waveform()
    with pytest.raises(ValueError, match="put a carrier before it"):
        Signal(seed=1, grid=G).law("rectify", mode="full").waveform()
    with pytest.raises(ValueError, match="needs the sample rate"):
        Signal(seed=1).symbols([1, -1]).filter_delays()


# ------------------------------------------------------------------ how they sit in a chain
def test_general_stages_are_classified_everywhere_an_op_must_be():
    for op in ("law", "filter"):
        assert op in C._EXEC and C.OP_KIND[op] == "general" and op in ws.opkeys.OP_KEYS
        assert (op in C._LEAD_LTI) + (op in C._LEAD_SKIP) + (op in C._LEAD_REJECT) \
            + (op in C._LEAD_ANALYTIC) == 1
    assert set(ws.opkeys.OP_KEYS["law"]) == set(ST.LAW_KEYS)
    assert set(ws.opkeys.OP_KEYS["filter"]) == set(ST.FILTER_KEYS)


def test_canonical_order_refuses_a_stage_with_no_home_and_leaves_other_chains_alone():
    s = _sine().lossy(loss_db=3.0, loss_at_ghz=0.2).law("rectify", mode="full")
    assert s.stage_kinds() == ["source", "channel", "general"]
    with pytest.raises(ValueError, match="general stage with no home"):
        s.canonical()
    plain = _sine().scope(bw_hz=100e6).lossy(loss_db=3.0, loss_at_ghz=0.2)
    assert [o["op"] for o in plain.canonical().ops] == ["carrier", "lossy", "scope"]


def test_a_lead_in_sizes_its_guard_from_a_filter_and_refuses_an_integrator():
    g = Grid(fs=256e9, baud=32e9, n=1 << 12)
    src = lambda **kw: Signal(seed=2, grid=g, **kw).carrier("nrz", n_ui=32, pattern="prbs7")   # noqa: E731
    slow = src(lead_in=True).filter("butter", f_hz=0.5e9, order=2)
    fast = src(lead_in=True).filter("butter", f_hz=20e9, order=2)
    assert slow.lead_plan().measured > 10 * fast.lead_plan().measured > 0
    y = slow.law("limiter", hi=0.5).waveform()
    assert len(y) == g.n and y.max() == 0.5
    with pytest.raises(ValueError, match="running integral FROM THE RECORD'S FIRST SAMPLE"):
        src(lead_in=True).filter("integrate").waveform()


def test_zero_phase_refuses_a_response_too_long_to_hold_both_sides_of():
    # accepted as a design (a 100 Hz corner at 1 GS/s is a legitimate causal baseline filter)
    # and refused when it renders, because a zero-phase render pads for the whole response
    s = _sine().filter("butter", f_hz=100.0, order=2, zero_phase=True)
    with pytest.raises(ValueError, match="still ringing.*apply it causally"):
        s.waveform()
    y = _sine().filter("butter", f_hz=100.0, order=2, btype="high").waveform()
    assert np.abs(y - _sine().waveform()).max() < 1e-3      # 19 cycles of 2.3 MHz pass


@pytest.mark.parametrize("kind,kw", [
    ("butter", {}), ("bessel", {}), ("cheby1", dict(ripple_db=1.0)),
    ("elliptic", dict(ripple_db=1.0, stop_db=40.0))])
def test_a_corner_too_far_below_the_sample_rate_is_refused_not_approximated(kind, kw):
    # MEASURED: at corner / fs = 1e-8 the sections' DC gain is already 1 to 9 % off the
    # design's (the poles are within rounding of z = 1); at 1e-6 every one of these designs
    # holds its corner and its passband to better than 0.1 %.
    with pytest.raises(ValueError, match="unresolved design.*not realisable in double"):
        ST.design_filter(FS, kind, f_hz=FS * 1e-8, **kw)
    sos = ST.design_filter(FS, kind, f_hz=FS * 1e-6, **kw)["sos"]
    dc = float(np.prod(sos[:, :3].sum(1) / sos[:, 3:].sum(1)))
    assert dc == pytest.approx(10 ** (-1 / 20) if kw else 1.0, rel=1e-3)   # order 4: even


def test_sections_with_no_settled_state_are_refused_when_they_render():
    with pytest.raises(ValueError, match="unresolved -- the DC gain"):
        ST.design_filter(FS, "zpk", poles=[-10.0, -10.0], gain=100.0)    # 1.6 Hz at 1 GS/s
    # a double pole at z = 1 - 3e-9: inside the unit circle, and 1 + a1 + a2 rounds to zero
    near = [[1.0, 0.0, 0.0, *np.poly([1 - 3e-9, 1 - 3e-9])]]
    assert np.abs(np.roots(near[0][3:])).max() < 1.0 and sum(near[0][3:]) == 0.0
    with pytest.raises(ValueError, match="no settled state"):
        ST.apply_filter(np.ones(16), FS, "sos", sos=near)


def test_a_fast_edge_arrives_at_the_step_delay_and_a_slow_one_at_the_group_delay():
    """The two delays a filter reports are for the two regimes, and each is the one measured:
    a step's 50 % crossing against `step_delay_s`, a ramp's against `group_delay_s`."""
    fs, n = 1e9, 20000

    def crossing(y):                                    # fractional sample where y reaches 0.5
        j = int(np.flatnonzero(y >= 0.5)[0])
        return j - 1 + (0.5 - y[j - 1]) / (y[j] - y[j - 1])

    step = (np.arange(n) >= 5000).astype(float)         # its midpoint is at 4999.5
    ramp = np.clip((np.arange(n) - 2000) / 6000.0, 0.0, 1.0)
    for kind, kw in (("butter", dict(order=1)), ("butter", dict(order=4)),
                     ("bessel", dict(order=4))):
        rep = ST.describe_filter(fs, kind, f_hz=5e6, **kw)
        fast = crossing(ST.apply_filter(step, fs, kind, f_hz=5e6, **kw)) - 4999.5
        slow = crossing(ST.apply_filter(ramp, fs, kind, f_hz=5e6, **kw)) - crossing(ramp)
        print(f"{kind} {kw}: step moved {fast:.3f} (reported {rep['step_delay_s'] * fs:.3f}), "
              f"ramp moved {slow:.3f} (reported {rep['group_delay_s'] * fs:.3f}) samples")
        assert abs(fast - rep["step_delay_s"] * fs) < 1e-6
        assert abs(slow - rep["group_delay_s"] * fs) < 1e-3
    one = ST.describe_filter(fs, "butter", f_hz=5e6, order=1)
    assert one["step_delay_s"] / one["group_delay_s"] == pytest.approx(np.log(2.0), rel=2e-3)
    # no midpoint to speak of: a filter that blocks DC, an integrator; none at all: zero phase
    assert ST.describe_filter(fs, "butter", f_hz=5e6, btype="high")["step_delay_s"] is None
    assert ST.describe_filter(fs, "integrate")["step_delay_s"] is None
    assert ST.describe_filter(fs, "butter", f_hz=5e6, zero_phase=True)["step_delay_s"] == 0.0
