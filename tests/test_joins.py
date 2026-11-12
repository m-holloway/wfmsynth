"""The joins (GitHub #76, #79): `combine`, and `gain` / `offset` / `gate` / `delay` driven by a
number or by a sub-chain.

Every assertion is on the EFFECT: an analytic result, an existing op that must agree, or the
band-limited interpolation computed a second way. Tolerances are stated where they are used.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

import wfmsynth as ws
from wfmsynth import Grid, Signal, joins as JN

G = Grid(fs=1e6, baud=1e3, n=1 << 14)
T = np.arange(G.n) / G.fs
EDGE = JN.DELAY_TAPS // 2            # samples from either end where the kernel sees the edge


def sine(f_hz, amp=1.0, phase_rad=0.0, offset=0.0, seed=0):
    return Signal(seed=seed, grid=G).carrier("sine", f_hz=f_hz, amp=amp, phase_rad=phase_rad,
                                             offset=offset)


def square01(f_hz, duty=0.5):
    return Signal(grid=G).carrier("square", f_hz=f_hz, amp=0.5, offset=0.5, duty=duty)


def tone(f_hz, amp=1.0, phase_rad=0.0, t=T):
    return amp * np.sin(2 * np.pi * f_hz * t + phase_rad)


# ------------------------------------------------------------------------------- combine
def test_the_sum_of_two_sines_is_the_analytic_sum():
    y = sine(1e3).combine(sine(7e3, amp=0.3, phase_rad=0.4)).waveform()
    assert np.array_equal(y, tone(1e3) + tone(7e3, 0.3, 0.4))


def test_gains_scale_each_other_signal_and_leave_the_chain_alone():
    y = sine(1e3).combine([sine(7e3), sine(9e3)], gains=[0.25, -2.0]).waveform()
    want = tone(1e3) + 0.25 * tone(7e3) - 2.0 * tone(9e3)
    assert np.max(np.abs(y - want)) < 1e-15          # one rounding of a three-term sum


@pytest.mark.parametrize("how, fn", [("product", lambda a, b: a * b), ("min", np.minimum),
                                     ("max", np.maximum)])
def test_product_min_and_max_are_the_sample_by_sample_operation(how, fn):
    y = sine(1e3).combine([sine(7e3, amp=0.6)], how=how).waveform()
    assert np.array_equal(y, fn(tone(1e3), tone(7e3, 0.6)))


def test_a_product_with_a_0_1_square_is_the_modulate_ook_path():
    """`modulate(kind="ook", suppressed=True)` is `depth * message * carrier` with depth 1, so
    the product is bit-identical: bound 0. The unsuppressed form, `(1 + m) * carrier` with a
    +/-1 message, is the same product with a gain of 2, also to the bit here."""
    car = lambda: sine(50e3, seed=1)                                        # noqa: E731
    a = car().combine([square01(500.0)], how="product").waveform()
    b = car().modulate(kind="ook", suppressed=True, message=square01(500.0)).waveform()
    assert np.array_equal(a, b)
    assert 0.35 < np.mean(np.abs(a) < 1e-3) < 0.5        # it really is keyed off
    c = car().combine([square01(500.0)], how="product", gains=[2.0]).waveform()
    d = car().modulate(kind="ook",
                       message=Signal(grid=G).carrier("square", f_hz=500.0)).waveform()
    assert np.max(np.abs(c - d)) <= 4 * np.finfo(float).eps


def test_one_sub_chain_needs_no_list_and_every_form_is_the_same_record():
    other = sine(7e3, amp=0.3)
    a = sine(1e3).combine(other)
    b = sine(1e3).combine([{"ops": sine(7e3, amp=0.3).ops}])
    c = sine(1e3).combine({"from": sine(7e3, amp=0.3).recipe()})
    assert a.ops[1]["others"] == [{"ops": other.ops}] and a.sha256() == b.sha256()
    assert np.array_equal(a.waveform(), c.waveform()) and a.sha256() != c.sha256()


def test_a_combine_replays_from_its_recipe_alone():
    s = (sine(1e3, seed=4).combine([sine(7e3).rx_noise(rms=0.1)], gains=[0.5])
         .gain(by=sine(50.0, amp=0.05, offset=1.0)).delay(by=sine(300.0, amp=2e-6, offset=4e-6)))
    r = Signal.from_recipe(json.loads(s.to_json()))
    assert r.sha256() == s.sha256() and np.array_equal(r.waveform(), s.waveform())


def test_a_sub_chains_randomness_is_namespaced_and_rerolls_alone():
    noisy = lambda s: s.digitize(noise_rms=0.05)                            # noqa: E731
    s = noisy(sine(1e3, seed=3)).combine([noisy(sine(7e3))])
    role = "combine/2/others/0/noise/1"
    assert role in s.roles() and "noise/1" in s.roles()
    y = s.waveform()
    own = noisy(sine(1e3, seed=3)).waveform()
    assert np.array_equal(y, own + s.subchain("combine/2/others/0"))
    other = s.contrast(role) - own                         # re-roll the other signal's noise:
    assert np.std(other - (y - own)) > 0.03                # it moved,
    assert np.max(np.abs(other - tone(7e3))) < 0.5         # about the same tone, and ours did not


def test_a_sub_chains_events_surface_under_its_namespace():
    g = Grid(fs=64e9, baud=1e9, n=1 << 13)
    other = Signal(grid=g).carrier("nrz", n_ui=128).events(kind="glitch", count=3)
    s = Signal(seed=2, grid=g).carrier("nrz", n_ui=128).combine([other], gains=[0.2])
    ev = s.event_list()
    assert len(ev) == 3 and {e.at for e in ev} == {"combine/1/others/0"}
    assert len(ev.on_record()) == 0 and ev.mask().sum() == 0     # they did not modify this record


@pytest.mark.parametrize("build, words", [
    (lambda: sine(1e3).combine([sine(7e3)], how="add"), "is not one of"),
    (lambda: sine(1e3).combine([]), "`others` is empty"),
    (lambda: sine(1e3).combine([sine(7e3)], gains=[1.0, 2.0]), "2 gains for 1 entries"),
    (lambda: sine(1e3).combine([sine(7e3)], gains=0.5), "one number per entry"),
    (lambda: sine(1e3).combine([sine(7e3)], gains=[float("nan")]), "finite"),
    (lambda: sine(1e3).combine(np.zeros(G.n)), "ndarray"),
    (lambda: sine(1e3).combine([Signal(grid=Grid(fs=2e6, n=G.n)).carrier("sine", f_hz=7e3)]),
     "grid"),
    (lambda: Signal(grid=G).combine([sine(7e3)]).waveform(), "put a carrier before it"),
    (lambda: Signal(seed=1, grid=G, lead_in="auto").carrier("sine", f_hz=1e3)
     .combine([sine(7e3)]).waveform(), "lead_in"),
    (lambda: sine(1e3).combine([sine(7e3)], gain=2.0), "gain"),
])
def test_combine_refuses_by_name(build, words):
    with pytest.raises((ValueError, TypeError), match=words):
        build()


def test_a_recipe_written_by_hand_is_refused_the_same_way():
    ops = [{"op": "carrier", "kind": "sine", "f_hz": 1e3}]
    sub = {"ops": [{"op": "carrier", "kind": "sine", "f_hz": 7e3}]}
    base = sine(1e3).recipe()
    for bad, words in [({"op": "combine", "others": [sub], "how": "xor"}, "is not one of"),
                       ({"op": "combine", "others": [], "how": "sum"}, "empty"),
                       ({"op": "combine", "others": [sub], "gains": [1, 2]}, "2 gains"),
                       ({"op": "gain"}, "needs by="),
                       ({"op": "gain", "by": True}, "needs by="),
                       ({"op": "gate", "by": 1.0, "off": "float"}, "off is"),
                       ({"op": "delay", "by": 1e-6, "max_rate": 1.5}, "between 0 and 1")]:
        with pytest.raises(ValueError, match=words):
            Signal.from_recipe({**base, "ops": ops + [bad]}).waveform()


def test_a_different_domain_is_refused_by_name():
    """Which joins mean something on an I/Q record is `wfmsynth.domain`'s table, checked both
    ways in tests/test_iq_seams.py. Here: the wrong KIND of other signal is refused where it is
    rendered, and the joins with no meaning on an envelope where they are written."""
    gi = Grid(fs=16e6, baud=1e6, n=1 << 12, domain="iq", fc_hz=4e6)
    gr = Grid(fs=16e6, baud=1e6, n=1 << 12)
    env = Signal(seed=1, grid=gi).iq("qpsk", prbs=7)
    with pytest.raises(ValueError, match="REAL waveform"):
        env.combine([{"ops": [{"op": "carrier", "kind": "sine", "f_hz": 1e6}]}]).waveform()
    real = Signal(seed=1, grid=gr).carrier("sine", f_hz=1e6)
    with pytest.raises(ValueError, match="COMPLEX"):
        real.combine({"from": Signal(seed=1, grid=gi).iq("qpsk", prbs=7).recipe()}).waveform()
    for stage, instead in (("offset", "iq_offset"), ("delay", "carrier")):
        with pytest.raises(ValueError, match="complex baseband") as e:
            getattr(Signal(seed=1, grid=gi).iq("qpsk", prbs=7), stage)(by=1.0)
        assert instead in str(e.value)


def test_the_joins_have_no_home_in_canonical_order():
    with pytest.raises(ValueError, match="general stage"):
        sine(1e3).gain(by=2.0).canonical()


# ------------------------------------------------------------------------- gain and offset
def test_gain_and_offset_take_a_number():
    assert np.array_equal(sine(1e3).gain(by=2.5).waveform(), 2.5 * tone(1e3))
    s = sine(1e3).gain(by=np.int64(2)).offset(by=np.float32(0.5))      # stored as plain floats
    assert [type(o["by"]) for o in s.ops[1:]] == [float, float]
    assert np.array_equal(s.waveform(), 2.0 * tone(1e3) + 0.5)
    assert np.array_equal(sine(1e3).offset(by=-0.4).waveform(), tone(1e3) - 0.4)


def test_gain_and_offset_by_a_chain_are_the_product_and_the_sum():
    ripple = lambda: sine(120.0, amp=0.03, offset=1.0)                      # noqa: E731
    y = sine(1e3).gain(by=ripple()).waveform()
    assert np.array_equal(y, tone(1e3) * (1.0 + tone(120.0, 0.03)))
    assert np.array_equal(y, sine(1e3).combine([ripple()], how="product").waveform())
    z = sine(1e3).offset(by=ripple()).waveform()
    assert np.array_equal(z, sine(1e3).combine([ripple()]).waveform())


def test_drift_is_gain_by_a_ramp():
    """`drift(kind="gain", shape="linear")` multiplies by 1 + amount * (0..1 across the
    record); a triangle sub-chain cannot be that ramp exactly, so the ramp comes in as a
    capture. Bound: one rounding."""
    ramp = 1.0 + 0.2 * np.arange(G.n) / (G.n - 1)
    by = Signal(grid=G).capture(values=ramp)
    a = sine(1e3).gain(by=by).waveform()
    b = sine(1e3).drift(kind="gain", amount=0.2, shape="linear").waveform()
    assert np.max(np.abs(a - b)) <= 2 * np.finfo(float).eps


def test_a_toleranced_number_is_a_unit_parameter():
    s = sine(1e3, seed=5).gain(by=ws.tol(2.0, 10))
    by = s.ops[1]["by"]
    assert 1.8 <= by <= 2.2 and by != 2.0 and "unit/1/by" in s.roles()
    assert np.array_equal(s.waveform(), by * tone(1e3))


# ---------------------------------------------------------------------------------- gate
def test_gate_by_a_square_is_burst_at_that_cadence():
    """`burst` shapes each on/off edge with a raised-cosine ramp at least one sample long, so
    with edge_frac=0 it zeroes the first and last sample of every burst; `gate` is an ideal
    switch. Everywhere else the two are the same samples: bound 0 off those two per burst."""
    car = lambda: sine(50e3, seed=1)                                        # noqa: E731
    a = car().gate(by=square01(500.0)).waveform()
    b = car().burst(t_on_s=1e-3, t_off_s=1e-3, edge_frac=0.0).waveform()
    n_bursts = int(np.ceil(G.n / 2000))
    differ = np.flatnonzero(a != b)
    assert len(differ) <= 2 * n_bursts
    assert set(differ % 2000) <= {0, 999}                # only a burst's first or last sample
    assert np.array_equal(b[differ], np.zeros(len(differ)))
    assert np.mean(a == 0.0) > 0.45                      # and half of it really is gated off
    assert np.max(np.abs(a - b)) < 0.31                  # sin(2*pi*50e3/1e6): one sample of it


def test_gate_reports_each_switching_instant():
    y, ev = sine(50e3, seed=1).gate(by=sine(500.0)).realize()         # threshold 0.5 on a sine
    want = []                                             # sin(2*pi*500*t) = 0.5, analytically
    for k in range(9):
        want += [(k + 1 / 12) / 500.0, (k + 5 / 12) / 500.0]
    want = [t for t in want if t < G.n / G.fs]
    got = [e.t_s for e in ev]
    assert len(got) == len(want)
    assert np.max(np.abs(np.array(got) - want)) < 2e-9    # linear interpolation over 1 us
    assert [e.direction for e in ev][:2] == [1.0, -1.0]
    assert all(e.placement == "gate/1" and e.kind == "transition" for e in ev)
    first = ev.events[0]
    assert y[first.sample] != 0.0 and y[first.sample - 1] == 0.0


def test_threshold_and_off_do_what_they_say():
    by = lambda: sine(500.0)                                                # noqa: E731
    x = tone(50e3)
    for thr in (-0.3, 0.0, 0.7):
        y = sine(50e3).gate(by=by(), threshold=thr).waveform()
        assert np.array_equal(y, np.where(tone(500.0) > thr, x, 0.0))
    h = sine(50e3).gate(by=by(), off="hold").waveform()
    on = tone(500.0) > 0.5
    k = np.flatnonzero(~on & (np.arange(G.n) > np.flatnonzero(on)[0]))
    last_on = np.maximum.accumulate(np.where(on, np.arange(G.n), 0))
    assert np.array_equal(h[on], x[on]) and np.array_equal(h[k], x[last_on[k]])
    assert np.all(h[:np.flatnonzero(on)[0]] == x[0])       # nothing let through yet
    assert np.array_equal(sine(50e3).gate(by=1.0).waveform(), x)
    assert not sine(50e3).gate(by=0.0).waveform().any()


# --------------------------------------------------------------------------------- delay
def _exact(x, pos):
    """The band-limited (trigonometric) interpolant of the periodic record `x` at fractional
    sample positions `pos`: an independent method, straight from the DFT."""
    n = len(x)
    X = np.fft.rfft(x)
    k = np.arange(len(X))
    w = np.where((k == 0) | ((n % 2 == 0) & (k == n // 2)), 1.0, 2.0)
    return np.array([np.sum(w * (X * np.exp(2j * np.pi * k * p / n)).real) for p in pos]) / n


def test_a_whole_number_of_samples_is_an_exact_shift_and_dc_stays_dc():
    x = tone(20e3)
    y = sine(20e3).delay(by=7 / G.fs).waveform()
    assert np.array_equal(y[7:], x[:-7]) and np.all(y[:7] == x[0])
    assert np.array_equal(sine(20e3).delay(by=0.0).waveform(), x)
    dc = Signal(grid=G).carrier("dc", level=0.7).delay(by=sine(300.0, amp=2e-6, offset=4e-6))
    assert np.max(np.abs(dc.waveform() - 0.7)) < 1e-15


@pytest.mark.parametrize("f0", [2e3, 50e3, 200e3, 390e3])
def test_a_tone_under_a_moving_delay_is_within_the_reported_error(f0):
    d = 4e-6 + tone(300.0, 2e-6)
    y = sine(f0).delay(by=sine(300.0, amp=2e-6, offset=4e-6)).waveform()
    err = np.max(np.abs(y - tone(f0, t=T - d))[EDGE + 8:-EDGE])
    bound = JN.delay_error(f0, G.fs)
    assert err <= bound + 1e-13, (err, bound)
    assert bound <= JN.describe_delay(G.fs)["error"] < 2.5e-5


def test_the_reported_error_bounds_a_wideband_record():
    """Six tones across the band with a moving delay, against the exact interpolant of the
    same samples. The bound is the sum over the tones of amplitude times `delay_error`."""
    n = 2048
    g = Grid(fs=1e6, n=n)
    bins = [3, 41, 160, 333, 600, 790]                    # up to 0.386 fs, whole cycles
    amps = [1.0, 0.5, 0.4, 0.3, 0.3, 0.2]
    x = sum(a * np.sin(2 * np.pi * b * np.arange(n) / n + b) for a, b in zip(amps, bins))
    d = 30e-6 + 20e-6 * np.sin(2 * np.pi * 5 * np.arange(n) / n)
    y = JN.delay(x, d, g.fs)
    sl = slice(64, n - 64)
    err = np.max(np.abs(y - _exact(x, np.arange(n) - d * g.fs))[sl])
    bound = sum(a * JN.delay_error(b * g.fs / n, g.fs) for a, b in zip(amps, bins))
    assert 1e-7 < err <= bound < 5e-5, (err, bound)


def test_delay_by_a_sine_reproduces_periodic_jitter():
    """`timing(pj=)` warps by A*sin(2*pi*fj*t) about zero with LINEAR interpolation; `delay`
    needs a non-negative delay, so it carries the same sine on a whole-sample offset D0 and
    the comparison is one shifted by D0. The two agree within delay's reported error plus the
    linear interpolator's own, (2*pi*f0/fs)**2 / 8 for a unit tone."""
    f0, fj, amp, k = 2e3, 300.0, 2e-6, 4
    d0 = k / G.fs
    by = sine(fj, amp=amp, offset=d0, phase_rad=-2 * np.pi * fj * d0)
    a = sine(f0, seed=1).delay(by=by).waveform()
    b = sine(f0, seed=1).timing(pj=dict(amp_ps=amp * 1e12, f_hz=fj)).waveform()
    diff = np.max(np.abs(a[k:] - b[:-k])[EDGE:-EDGE])
    reported = JN.delay_error(f0, G.fs)
    linear = (2 * np.pi * f0 / G.fs) ** 2 / 8
    assert diff <= reported + linear, (diff, reported, linear)
    exact = tone(f0, t=T - amp * np.sin(2 * np.pi * fj * T))
    assert np.max(np.abs(a[k:] - exact[:-k])[EDGE:-EDGE]) <= reported
    assert np.ptp(a[k:] - tone(f0)[:-k]) > 0.02          # and the jitter is really there


def test_delay_states_what_it_promises():
    s = sine(2e3).delay(by=1e-6).gain(by=2.0).delay(by=2e-6, max_rate=0.1)
    rep = s.delay_errors()
    assert [r["op"] for r in rep] == [1, 3] and [r["max_rate"] for r in rep] == [0.5, 0.1]
    assert rep[0]["taps"] == 32 and rep[0]["below_hz"] == 0.4 * G.fs
    f = np.linspace(0, rep[0]["below_hz"], 97)
    assert np.all(JN.delay_error(f, G.fs) <= rep[0]["error"] * (1 + 1e-3))   # scan resolution
    assert JN.delay_error(0.0, G.fs) < 1e-15 and JN.delay_error(0.5 * G.fs, G.fs) > 0.99


@pytest.mark.parametrize("build, words", [
    (lambda: sine(2e3).delay(by=sine(300.0, amp=2e-6)).waveform(), "is an advance"),
    (lambda: sine(2e3).delay(by=-1e-6).waveform(), "is an advance"),
    (lambda: sine(2e3).delay(by=sine(100e3, amp=1e-6, offset=1e-6)).waveform(), "max_rate"),
    (lambda: sine(2e3).delay(by=1e-6, max_rate=1.0), "between 0 and 1"),
    (lambda: sine(2e3).delay(by=1e-6, rate=0.1), "rate"),
    (lambda: Signal().carrier("sine", cycles=5, n=256).delay(by=1e-6).waveform(), "sample rate"),
    (lambda: Signal().carrier("sine", cycles=5, n=256).delay(by=1e-6).delay_errors(), "grid"),
    (lambda: sine(2e3).gate(by=1.0, off="float"), "off is"),
    (lambda: sine(2e3).gain(by=None), "needs by="),
    (lambda: sine(2e3).offset(by=[1.0, 2.0]).waveform(), "sub-chain"),
])
def test_the_stages_refuse_by_name(build, words):
    with pytest.raises((ValueError, TypeError), match=words):
        build()


def test_the_rate_limit_is_the_one_stated():
    """A 1 us sine at 100 kHz steps by up to 2*sin(pi/10) = 0.588 us per 1 us sample: refused
    at the default 0.5, accepted at 0.7, and the tone still lands within its reported error."""
    by = lambda: sine(100e3, amp=1e-6, offset=1e-6)                         # noqa: E731
    with pytest.raises(ValueError, match="0.5878 seconds per second"):
        sine(2e3).delay(by=by()).waveform()
    y = sine(2e3).delay(by=by(), max_rate=0.7).waveform()
    d = 1e-6 + tone(100e3, 1e-6)
    assert np.max(np.abs(y - tone(2e3, t=T - d))[EDGE:-EDGE]) <= JN.delay_error(2e3, G.fs)


def test_a_constant_delay_takes_its_history_from_a_lead_in():
    """Without a lead-in the line holds the record's first sample; with one, the head of the
    record is the delayed guard -- the delay is classified LTI and sizes the guard."""
    g = Grid(fs=64e9, baud=1e9, n=1 << 12)
    k = 640                                               # 10 UI
    mk = lambda **kw: Signal(seed=1, grid=g, **kw).carrier("nrz", n_ui=64)  # noqa: E731
    plain = mk().delay(by=k / g.fs).waveform()
    assert np.all(plain[:k] == plain[0])
    y = mk(lead_in=1024).delay(by=k / g.fs).waveform()
    ref = mk(lead_in=1024).waveform()                      # the same guard, so the same bits
    assert np.array_equal(y[k:], ref[:-k]) and np.ptp(y[:k]) > 1.0
    auto = mk(lead_in="auto").delay(by=k / g.fs).lead_plan()
    assert auto.measured == k + 1 and auto.lead >= k       # the delay itself sized the guard


# ------------------------------------------------------------------- every knob moves it
@pytest.mark.parametrize("a, b", [
    (lambda s: s.combine([sine(7e3)]), lambda s: s.combine([sine(8e3)])),
    (lambda s: s.combine([sine(7e3)]), lambda s: s.combine([sine(7e3)], how="max")),
    (lambda s: s.combine([sine(7e3)]), lambda s: s.combine([sine(7e3)], gains=[0.5])),
    (lambda s: s.gain(by=2.0), lambda s: s.gain(by=2.1)),
    (lambda s: s.offset(by=0.1), lambda s: s.offset(by=sine(7e3, amp=0.1))),
    (lambda s: s.gate(by=sine(500.0)), lambda s: s.gate(by=sine(500.0), threshold=0.0)),
    (lambda s: s.gate(by=sine(500.0)), lambda s: s.gate(by=sine(500.0), off="hold")),
    (lambda s: s.delay(by=1e-6), lambda s: s.delay(by=1.5e-6)),
])
def test_every_parameter_moves_the_samples_and_the_digest(a, b):
    sa, sb = a(sine(1e3, seed=1)), b(sine(1e3, seed=1))
    assert sa.sha256() != sb.sha256()
    assert np.max(np.abs(sa.waveform() - sb.waveform())) > 1e-3


def test_max_rate_is_in_the_digest_and_decides_what_is_refused():
    a, b = sine(1e3).delay(by=1e-6), sine(1e3).delay(by=1e-6, max_rate=0.1)
    assert a.sha256() != b.sha256() and np.array_equal(a.waveform(), b.waveform())
