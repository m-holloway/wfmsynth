"""GitHub #85: a triggered acquisition -- render long, find a condition, keep the window.

Every claim is checked against something the trigger did not compute:

  * crossing times against the CLOSED FORM of a sine and of a hand-built trapezoid train;
  * the window against the LONG RENDER, sample for sample (a window is a slice, not a resample);
  * the scope's arming against a brute-force loop written out here;
  * timestamps against the transitions `stages`' comparator reports for the same long render,
    which is a separate implementation;
  * trigger jitter against its stated rms, and against what it must NOT do (smear one record);
  * averaging against 1/sqrt(n), and against the mean of the same triggers taken as segments.
"""
from __future__ import annotations

import json
import re

import numpy as np
import pytest

from wfmsynth import stages
from wfmsynth import trigger as TRIG
from wfmsynth.compose import Signal, _EXEC, _LEAD_REJECT
from wfmsynth.grid import Grid
from wfmsynth.streams import Streams

FS = 1e9
F0 = 5e6                       # 200 samples per period, 5 ns of rise on the trapezoids below
PERIOD = 1.0 / F0


def sine(n=40000, phase=0.3, amp=1.0):
    t = np.arange(n) / FS
    return amp * np.sin(2 * np.pi * F0 * t + phase)


def pulses(widths, heights=None, gap=400e-9, tr=5e-9, n=None):
    """Positive trapezoid pulses 0 -> height -> 0, given by their width AT HALF HEIGHT.
    Returns (x, [(t_rise_50, t_fall_50), ...])."""
    heights = [1.0] * len(widths) if heights is None else heights
    pts_t, pts_v, truth = [0.0], [0.0], []
    t0 = gap
    for w, h in zip(widths, heights):
        pts_t += [t0 - tr / 2, t0 + tr / 2, t0 + w - tr / 2, t0 + w + tr / 2]
        pts_v += [0.0, h, h, 0.0]
        truth.append((t0, t0 + w))
        t0 += w + gap
    n = int(round((t0 + gap) * FS)) if n is None else n
    return np.interp(np.arange(n) / FS, pts_t, pts_v), truth


def crossing(x, level, lo, hi, rising=True):
    """Independent sub-sample crossing of `level` in x[lo:hi]: np.interp on the monotone edge."""
    seg = x[lo:hi]
    k = np.arange(lo, hi)
    return float(np.interp(level, seg if rising else seg[::-1], k if rising else k[::-1]))


# ------------------------------------------------------------------ the condition, found
def test_edge_times_match_the_closed_form_of_a_sine():
    x = sine()
    f = TRIG.find(x, FS, level=0.0, slope="rising")
    k = np.arange(1, len(f["t_s"]) + 1)
    exact = (2 * np.pi * k - 0.3) / (2 * np.pi * F0)
    err = np.abs(f["t_s"] - exact).max() * FS
    assert len(f["t_s"]) == 200 and err < 1e-5, err          # samples; measured 4.0e-6
    assert (f["direction"] == 1.0).all()


def test_slope_selects_the_direction_and_either_takes_both():
    x = sine()
    up = TRIG.find(x, FS, level=0.2, slope="rising")
    dn = TRIG.find(x, FS, level=0.2, slope="falling")
    both = TRIG.find(x, FS, level=0.2, slope="either")
    assert (dn["direction"] == -1.0).all() and len(dn["t_s"]) in (len(up["t_s"]), len(up["t_s"]) + 1)
    assert np.array_equal(both["t_s"], np.sort(np.concatenate([up["t_s"], dn["t_s"]])))
    # a falling crossing of +0.2 is at pi - asin(0.2), closed form
    exact = (np.pi - np.arcsin(0.2) - 0.3) / (2 * np.pi * F0)
    assert abs(dn["t_s"][0] - exact) * FS < 1e-3        # linear interpolation: 1.2e-4


def test_level_none_is_the_midpoint_of_the_record():
    x = 1.5 + 0.5 * sine()
    f = TRIG.find(x, FS)
    assert f["level"] == pytest.approx(1.5, abs=1e-6)


def test_hysteresis_rejects_noise_chatter_and_still_times_the_level_itself():
    rng = np.random.default_rng(0)
    clean = sine(amp=1.0)
    noisy = clean + rng.normal(0, 0.02, len(clean))
    raw = TRIG.find(noisy, FS, level=0.0)
    hyst = TRIG.find(noisy, FS, level=0.0, hysteresis=0.3)
    truth = TRIG.find(clean, FS, level=0.0)
    assert len(raw["t_s"]) > len(truth["t_s"])                     # chatter: several per edge
    assert len(hyst["t_s"]) == len(truth["t_s"])                   # one per edge
    # timed at `level`, not at the band's edge: a band-edge time would be 0.15/slope = 4.8
    # samples late; the noise alone moves the crossing by sigma/slope = 0.64 samples rms
    late = (hyst["t_s"] - truth["t_s"]) * FS
    assert abs(late.mean()) < 0.5 and late.std() < 1.5, (late.mean(), late.std())


def test_width_trigger_measures_the_pulse_and_fires_on_its_trailing_edge():
    widths = [50e-9, 120e-9, 30e-9, 200e-9, 80e-9]
    x, truth = pulses(widths)
    f = TRIG.find(x, FS, condition="width", level=0.5, width_s=[None, 1.0])
    assert np.allclose(f["width_s"], widths, atol=1e-12)
    assert np.allclose(f["t_s"], [b for _, b in truth], atol=1e-12)
    assert (f["direction"] == -1.0).all() and np.allclose(f["peak"], 1.0)
    mid = TRIG.find(x, FS, condition="width", level=0.5, width_s=[60e-9, 150e-9])
    assert np.allclose(mid["width_s"], [120e-9, 80e-9], atol=1e-12)
    short = TRIG.find(x, FS, condition="width", level=0.5, width_s=[None, 40e-9])
    assert np.allclose(short["t_s"], [truth[2][1]], atol=1e-12)
    long_ = TRIG.find(x, FS, condition="width", level=0.5, width_s=[150e-9, None])
    assert np.allclose(long_["width_s"], [200e-9], atol=1e-12)


def test_width_polarity_follows_slope():
    x, truth = pulses([50e-9, 120e-9], gap=300e-9)
    neg = TRIG.find(x, FS, condition="width", level=0.5, slope="falling", width_s=[None, 1.0])
    # the one negative pulse is the gap between the two positive ones
    assert np.allclose(neg["width_s"], [300e-9], atol=1e-12) and neg["direction"][0] == 1.0
    assert np.allclose(neg["t_s"], [truth[1][0]], atol=1e-12)
    both = TRIG.find(x, FS, condition="width", level=0.5, slope="either", width_s=[None, 1.0])
    assert len(both["t_s"]) == 3


def test_runt_trigger_fires_only_on_the_pulse_that_falls_short():
    x, truth = pulses([60e-9] * 5, heights=[1.0, 1.0, 0.55, 1.0, 1.0])
    f = TRIG.find(x, FS, condition="runt", level=0.3, runt_level=0.8)
    assert len(f["t_s"]) == 1 and f["peak"][0] == pytest.approx(0.55)
    # its trailing edge crosses 0.3 where the 0.55-high trapezoid does, closed form
    t_fall = truth[2][1] - 2.5e-9 + 5e-9 * (1 - 0.3 / 0.55)
    assert f["t_s"][0] == pytest.approx(t_fall, abs=1e-12)
    # negative runt: the same record upside down
    g = TRIG.find(1.0 - x, FS, condition="runt", level=0.7, runt_level=0.2, slope="falling")
    assert g["t_s"][0] == pytest.approx(t_fall, abs=1e-12) and g["direction"][0] == 1.0
    # a width bound applies to a runt as well
    assert len(TRIG.find(x, FS, condition="runt", level=0.3, runt_level=0.8,
                         width_s=[100e-9, None])["t_s"]) == 0


# ------------------------------------------------------------------ the window
def test_the_window_is_a_slice_and_the_edge_sits_at_pre_s():
    x = sine()
    ev = []
    y = TRIG.apply(x, FS, pre_s=120e-9, post_s=300e-9, level=0.0, events=ev)
    e = ev[0]
    assert len(y) == 420 and np.array_equal(y, x[e.params["start"]:e.params["start"] + 420])
    # independent sub-sample measurement of where the crossing sits in the delivered record
    meas = crossing(y, 0.0, 100, 140) / FS
    assert abs(meas - e.t_s) * FS < 1e-4                      # the event says where, exactly
    assert abs(meas - 120e-9) * FS <= 0.5                     # at pre_s to half a sample
    assert e.sample == int(np.ceil(e.t_s * FS)) and y[e.sample - 1] < 0.0 <= y[e.sample]
    assert e.kind == "trigger" and e.direction == 1.0 and e.params["jitter_s"] == 0.0


def test_event_fields_are_json_and_stay_out_of_the_mask():
    g = Grid(fs=FS, n=20000)
    sig = Signal(seed=1, grid=g).carrier("sine", f_hz=F0).trigger(pre_s=50e-9, post_s=350e-9)
    x, ev = sig.realize()
    d = json.loads(json.dumps(ev.to_dicts()))[0]
    assert d["at"] == "trigger/1" and d["placement"] == "trigger/1"
    assert set(d["params"]) == {"segment", "timestamp_s", "jitter_s", "start", "level",
                                "condition", "dropped"}
    assert ev.on_record() == [] and ev.mask().sum() == 0 and ev.n == len(x) == 400


# ------------------------------------------------------------------ the scope's arming
def _brute_force(times, pre, post, holdoff):
    out, ready = [], pre
    for i, t in enumerate(times):
        if t >= ready:
            out.append(i)
            ready = t + max(holdoff, pre + post) * (1 - TRIG.TIE)
    return out


@pytest.mark.parametrize("holdoff", [0.0, 450e-9, 1.1e-6, 2.5 * PERIOD, 3.1e-6,
                                     5 * PERIOD, 15 * PERIOD])    # the last two are exact ties
def test_holdoff_skips_exactly_the_events_it_should(holdoff):
    x = sine(n=60000)
    every = TRIG.find(x, FS, level=0.0)["t_s"]
    ev = []
    TRIG.apply(x, FS, pre_s=100e-9, post_s=200e-9, level=0.0, holdoff_s=holdoff,
               n_segments=12, events=ev)
    got = [e.params["timestamp_s"] for e in ev]
    want = every[_brute_force(every, 100e-9, 200e-9, holdoff)[:12]]
    assert np.array_equal(got, want)
    gaps = np.diff(got) / PERIOD
    k = max(1, int(np.ceil(max(holdoff, 300e-9) / PERIOD - 1e-9)))
    assert np.allclose(gaps, k, atol=1e-6), (gaps, k)       # every k-th edge, none between


def test_nothing_fires_before_the_pre_trigger_buffer_is_full():
    x = sine(n=20000)
    every = TRIG.find(x, FS, level=0.0)["t_s"]
    ev = []
    TRIG.apply(x, FS, pre_s=1.0e-6, post_s=100e-9, level=0.0, events=ev)
    assert (every < 1.0e-6).sum() == 5                           # five edges it must pass over
    assert ev[0].params["timestamp_s"] == every[5] and ev[0].params["start"] >= 0


def test_segments_never_overlap_and_are_laid_end_to_end():
    x = sine(n=60000)
    ev = []
    y = TRIG.apply(x, FS, pre_s=100e-9, post_s=200e-9, level=0.0, n_segments=7, events=ev)
    rows = y.reshape(7, -1)
    starts = np.array([e.params["start"] for e in ev])
    assert (np.diff(starts) >= 300).all()
    for k, e in enumerate(ev):
        assert e.params["segment"] == k
        assert np.array_equal(rows[k], x[starts[k]:starts[k] + 300])
        assert k * 300 <= e.sample < (k + 1) * 300


def test_segment_timestamps_match_the_comparators_transitions_in_the_long_render():
    g = Grid(fs=FS, n=80000)
    chain = lambda: Signal(seed=4, grid=g).carrier("sine", f_hz=F0, amp=1.0).rx_noise(rms=0.01)
    long_x = chain().waveform()
    seen = []
    stages.apply_law(long_x, "comparator", fs=FS, events=seen, threshold=0.1)
    rising = np.array([e.t_s for e in seen if e.direction > 0])
    _, ev = chain().trigger(pre_s=80e-9, post_s=300e-9, level=0.1, holdoff_s=1e-6,
                            n_segments=20).realize()
    stamps = np.array([e.params["timestamp_s"] for e in ev])
    idx = np.searchsorted(rising, stamps - 1e-15)
    assert len(stamps) == 20 and np.abs(rising[idx] - stamps).max() < 1e-15
    # and the stamp is where the edge is in the LONG render, measured independently
    for e in ev:
        j = int(np.floor(e.params["timestamp_s"] * FS))
        frac = (0.1 - long_x[j]) / (long_x[j + 1] - long_x[j])
        assert abs((j + frac) / FS - e.params["timestamp_s"]) < 1e-15


# ------------------------------------------------------------------ trigger jitter
def test_trigger_jitter_has_its_stated_rms_and_moves_only_the_window():
    x = sine(n=402 * 400)
    ev = []
    y = TRIG.apply(x, FS, pre_s=100e-9, post_s=200e-9, level=0.0, jitter_s=4e-9,
                   n_segments=400, rng=np.random.default_rng(3), events=ev)
    jit = np.array([e.params["jitter_s"] for e in ev])
    assert abs(jit.std() / 4e-9 - 1) < 0.12, jit.std()          # 400 draws: 3.5 % one sigma
    rows = y.reshape(400, -1)
    for k, e in enumerate(ev):
        # the crossing sits at pre_s - jitter to half a sample, and the event says where
        where = e.t_s - k * 300 / FS
        assert abs(where - (100e-9 - e.params["jitter_s"])) * FS <= 0.5 + 1e-6
        # every sample is the long render's: the window moved, nothing in it was smeared
        assert np.array_equal(rows[k], x[e.params["start"]:e.params["start"] + 300])
    pos = np.array([e.t_s - k * 300 / FS for k, e in enumerate(ev)])
    want = np.sqrt(4e-9 ** 2 + (1 / FS) ** 2 / 12)              # jitter and sample rounding
    assert abs(pos.std() / want - 1) < 0.12


def test_rerolling_trigger_jitter_keeps_the_same_edges():
    g = Grid(fs=FS, n=60000)
    sig = (Signal(seed=2, grid=g).carrier("sine", f_hz=F0).rx_noise(rms=0.01)
           .trigger(pre_s=100e-9, post_s=200e-9, level=0.0, jitter_s=3e-9, n_segments=8))
    assert sig.roles() == ["rx_noise/1", "trigger/2"]
    a = sig.event_list()
    b = sig.event_list(Streams(sig.seed).reroll("trigger/2"))
    assert [e.params["timestamp_s"] for e in a] == [e.params["timestamp_s"] for e in b]
    assert [e.params["start"] for e in a] != [e.params["start"] for e in b]
    assert not np.array_equal(sig.contrast("trigger/2"), sig.waveform())
    # and the timebase is a different role: re-rolling it leaves every trigger draw alone
    tb = (Signal(seed=2, grid=g).carrier("sine", f_hz=F0).timebase(rms_ps=50.0)
          .trigger(pre_s=100e-9, post_s=200e-9, level=0.0, jitter_s=3e-9, n_segments=8))
    c = tb.event_list()
    d = tb.event_list(Streams(tb.seed).reroll("timebase/1"))
    assert [e.params["jitter_s"] for e in c] == [e.params["jitter_s"] for e in d]


def test_no_jitter_means_no_role_and_no_draw():
    g = Grid(fs=FS, n=20000)
    sig = Signal(seed=2, grid=g).carrier("sine", f_hz=F0).trigger(pre_s=1e-7, post_s=2e-7)
    assert sig.roles() == []
    assert np.array_equal(sig.waveform(), sig.waveform(Streams(99)))


def _rise_10_90(y, lo, hi):
    a, b = y[lo], y[hi - 1]
    return (crossing(y, a + 0.9 * (b - a), lo, hi) - crossing(y, a + 0.1 * (b - a), lo, hi)) / FS


def test_averaging_jittered_triggers_smears_the_edge_and_one_record_is_not_smeared():
    n_avg, sigma, tr = 256, 6e-9, 20e-9
    x, _ = pulses([300e-9] * (n_avg + 2), gap=300e-9, tr=tr)
    kw = dict(pre_s=100e-9, post_s=100e-9, level=0.5, jitter_s=sigma)
    one = TRIG.apply(x, FS, rng=np.random.default_rng(1), **kw)
    avg = TRIG.apply(x, FS, rng=np.random.default_rng(1), average=n_avg, **kw)
    assert _rise_10_90(one, 60, 140) == pytest.approx(0.8 * tr, rel=1e-9)     # the ramp's own
    # the average is the ramp convolved with the jitter's Gaussian: integrate it numerically
    t = (np.arange(200) - 100) / FS
    u = np.linspace(-6, 6, 2401)
    w = np.exp(-u * u / 2); w /= w.sum()
    ramp = lambda tt: np.clip(tt / tr + 0.5, 0.0, 1.0)
    want = np.array([(w * ramp(ti - u * sigma)).sum() for ti in t])
    r_avg, r_want = _rise_10_90(avg, 40, 160), _rise_10_90(want, 40, 160)
    assert r_want > 1.2 * 0.8 * tr and abs(r_avg / r_want - 1) < 0.10, (r_avg, r_want)


# ------------------------------------------------------------------ averaging
def test_average_is_the_mean_of_the_same_triggers_and_cuts_noise_by_root_n():
    g = Grid(fs=FS, n=200 * 2000)
    base = lambda: (Signal(seed=7, grid=g).carrier("cmos", v_lo=0.0, v_hi=1.0, f_hz=1e6,
                                                   tr_s=5e-9).rx_noise(rms=0.05))
    kw = dict(pre_s=50e-9, post_s=400e-9, level=0.5, hysteresis=0.4)
    n_avg = 64
    segs = base().trigger(n_segments=n_avg, **kw).waveform().reshape(n_avg, -1)
    avg, ev = base().trigger(average=n_avg, **kw).realize()
    assert np.allclose(avg, segs.mean(0), rtol=0, atol=1e-15)
    assert [e.params["shot"] for e in ev] == list(range(n_avg))
    assert all(e.params["segment"] == 0 and e.sample < len(avg) for e in ev)
    flat = slice(100, 440)                                   # the settled high level
    s1 = segs[:, flat].std(1).mean()
    sn = avg[flat].std()
    assert abs(s1 / 0.05 - 1) < 0.05
    assert abs(sn / (0.05 / np.sqrt(n_avg)) - 1) < 0.15, (sn, 0.05 / 8)     # 340 samples: 3.8 %


def test_segments_of_averages():
    x = sine(n=60000)
    ev = []
    y = TRIG.apply(x, FS, pre_s=100e-9, post_s=200e-9, level=0.0, n_segments=3, average=4,
                   events=ev)
    each = TRIG.apply(x, FS, pre_s=100e-9, post_s=200e-9, level=0.0, n_segments=12)
    assert np.allclose(y.reshape(3, -1), each.reshape(3, 4, -1).mean(1), rtol=0, atol=1e-15)
    assert [(e.params["segment"], e.params["shot"]) for e in ev] == \
        [(s, k) for s in range(3) for k in range(4)]


# ------------------------------------------------------------------ events of the long render
def _with_glitches(times, **trig):
    g = Grid(fs=FS, n=40000)
    return (Signal(seed=1, grid=g).carrier("sine", f_hz=F0)
            .events("glitch", on="times", times=times)
            .trigger(level=0.0, **trig))


def test_events_are_retimed_into_the_window_and_the_rest_are_dropped_and_counted():
    times = [0.5e-6, 1.25e-6, 1.30e-6, 30e-6]
    plain = Signal(seed=1, grid=Grid(fs=FS, n=40000)).carrier("sine", f_hz=F0).events(
        "glitch", on="times", times=times)
    long_x, long_ev = plain.realize()
    sig = _with_glitches(times, pre_s=1.0e-6, post_s=200e-9)
    x, ev = sig.realize()
    trig = [e for e in ev if e.kind == "trigger"][0]
    kept = [e for e in ev if e.kind == "glitch"]
    start = trig.params["start"]
    inside = [e for e in long_ev if start <= e.sample < start + len(x)]
    assert len(kept) == len(inside) >= 1
    assert trig.params["dropped"] == len(long_ev) - len(inside) >= 1
    for a, b in zip(kept, inside):
        assert a.sample == b.sample - start and a.t_s == pytest.approx(b.t_s - start / FS)
        assert 0 <= a.sample < len(x)                       # nothing points past the record
        # the glitch really is at that sample of the delivered record
        assert np.array_equal(x[a.sample:a.sample + 8], long_x[b.sample:b.sample + 8])
    want = np.zeros(len(x))
    for e in kept:
        want[e.sample:e.sample + e.width] = 1.0
    assert np.array_equal(ev.mask(), want) and want.sum() > 0      # the trigger marks nothing


def test_events_land_in_their_own_segment():
    sig = _with_glitches([3e-6, 5.45e-6, 10.6e-6], pre_s=100e-9, post_s=200e-9,
                         holdoff_s=5.05e-6, n_segments=4)
    x, ev = sig.realize()
    trig = [e for e in ev if e.kind == "trigger"]
    kept = [e for e in ev if e.kind == "glitch"]
    long_ev = Signal.from_recipe({**sig.recipe(), "ops": sig.recipe()["ops"][:2]}).event_list()
    segs = []
    for e in long_ev:
        for t in trig:
            if t.params["start"] <= e.sample < t.params["start"] + 300:
                segs.append((t.params["segment"], e.sample - t.params["start"]))
    assert [(e.sample // 300, e.sample % 300) for e in kept] == segs
    assert sorted(s for s, _ in segs) == [1, 2]              # one each, in different windows
    assert trig[0].params["dropped"] == len(long_ev) - len(kept)


def test_a_comparators_transitions_are_retimed_too():
    g = Grid(fs=FS, n=20000)
    sig = (Signal(seed=1, grid=g).carrier("sine", f_hz=F0).law("comparator", threshold=0.0)
           .trigger(pre_s=150e-9, post_s=250e-9, level=0.5))
    x, ev = sig.realize()
    tr = [e for e in ev if e.kind == "transition"]
    assert len(tr) == 4 and all(0 <= e.sample < len(x) for e in tr)
    for e in tr:                                            # each sits on a step of the window
        assert x[e.sample] != x[e.sample - 1]


# ------------------------------------------------------------------ with acquire
PROFILE = dict(sample_rate_hz=250e6, record_length=5000, input_bandwidth_hz=60e6)


def test_after_acquire_it_triggers_at_the_acquisition_rate():
    g = Grid(fs=FS, n=20000)
    sig = (Signal(seed=3, grid=g).carrier("sine", f_hz=F0)
           .events("glitch", on="times", times=[1.05e-6, 9e-6])
           .acquire(PROFILE).trigger(pre_s=1.0e-6, post_s=400e-9, level=0.0))
    x, ev = sig.realize()
    acquired = Signal.from_recipe({**sig.recipe(), "ops": sig.recipe()["ops"][:3]}).waveform()
    trig = [e for e in ev if e.kind == "trigger"][0]
    assert len(x) == 350 and np.array_equal(x, acquired[trig.params["start"]:][:350])
    sg = sig.stored_grid(x)
    assert sg.fs == 250e6 and sg.n == 350
    assert abs(trig.t_s - 1.0e-6) * 250e6 <= 0.5
    # the glitch placed at 1.05 us of the long render is ~50 ns after the window's 1 us mark
    kept = [e for e in ev if e.kind == "glitch"]
    start_s = trig.params["start"] / 250e6
    assert len(kept) == 1 and trig.params["dropped"] == 1
    assert abs(kept[0].sample / 250e6 - (1.05e-6 - start_s)) <= 1 / 250e6
    assert kept[0].t_s == pytest.approx(1.05e-6 - start_s, abs=1e-12)


def test_before_acquire_the_window_is_what_gets_acquired():
    g = Grid(fs=FS, n=20000)
    prof = dict(sample_rate_hz=250e6, record_length=100)
    sig = (Signal(seed=3, grid=g).carrier("sine", f_hz=F0)
           .trigger(pre_s=100e-9, post_s=300e-9, level=0.0).acquire(prof))
    x = sig.waveform()
    assert len(x) == 100 and sig.stored_grid(x).fs == 250e6
    k = np.flatnonzero((x[:-1] < 0) & (x[1:] >= 0))[0]
    t = (k + (0 - x[k]) / (x[k + 1] - x[k])) / 250e6
    assert abs(t - 100e-9) * FS <= 0.5 + 0.05               # half a SIMULATION sample


# ------------------------------------------------------------------ the op and its recipe
def _sig(n=40000, rms=0.02, **kw):
    g = Grid(fs=FS, n=n)
    p = dict(pre_s=100e-9, post_s=300e-9, level=0.0, hysteresis=0.1)
    p.update(kw)
    s = Signal(seed=5, grid=g).carrier("sine", f_hz=F0)
    return (s.rx_noise(rms=rms) if rms else s).trigger(**p)


def test_recipe_replays_bit_for_bit_from_json():
    sig = _sig(jitter_s=2e-9, n_segments=4, holdoff_s=1e-6)
    again = Signal.from_recipe(json.loads(json.dumps(sig.recipe())))
    assert np.array_equal(again.waveform(), sig.waveform()) and again.sha256() == sig.sha256()
    assert again.event_list().to_dicts() == sig.event_list().to_dicts()
    assert sig.recipe()["ops"][-1] == {
        "op": "trigger", "pre_s": 100e-9, "post_s": 300e-9, "level": 0.0, "hysteresis": 0.1,
        "jitter_s": 2e-9, "n_segments": 4, "holdoff_s": 1e-6}


@pytest.mark.parametrize("knob", [
    dict(pre_s=150e-9), dict(post_s=350e-9), dict(level=0.3), dict(slope="falling"),
    dict(hysteresis=1.5, rms=0.3), dict(holdoff_s=2e-6, n_segments=3), dict(jitter_s=5e-9),
    dict(n_segments=2), dict(average=4), dict(condition="width", width_s=[90e-9, 110e-9]),
    dict(condition="runt", level=-0.5, runt_level=1.2, hysteresis=0.0),
])
def test_every_knob_moves_the_samples_and_the_digest(knob):
    base_kw = dict(n_segments=3) if "holdoff_s" in knob else {}
    if "rms" in knob:                    # hysteresis only matters where the noise chatters
        base_kw = dict(rms=knob["rms"])
    a, b = _sig(**base_kw), _sig(**knob)
    xa, xb = a.waveform(), b.waveform()
    assert a.sha256() != b.sha256()
    assert len(xa) != len(xb) or not np.array_equal(xa, xb)


def test_a_hand_written_recipe_is_checked_at_render():
    r = _sig().recipe()
    r["ops"][-1]["slope"] = "upward"
    with pytest.raises(ValueError, match="slope is one of"):
        Signal.from_recipe(r).waveform()
    r["ops"][-1] = {"op": "trigger", "pre_s": 1e-7}
    with pytest.raises(ValueError, match="post_s is required"):
        Signal.from_recipe(r).waveform()


def test_a_toleranced_trigger_parameter_is_drawn_like_any_other():
    import wfmsynth as ws
    g = Grid(fs=FS, n=40000)
    sig = Signal(seed=5, grid=g).carrier("sine", f_hz=F0).trigger(
        pre_s=100e-9, post_s=300e-9, level=ws.tol(0.0, tol_abs=0.5))
    level = sig.ops[-1]["level"]
    assert sig.roles() == ["unit/1/level"] and 0 < abs(level) <= 0.5
    assert sig.event_list().events[0].params["level"] == level
    assert sig.unit(9).ops[-1]["level"] != level


def test_trigger_is_classified_everywhere_an_op_must_be():
    from wfmsynth import domain, opkeys
    from wfmsynth.compose import OP_KIND
    assert "trigger" in _EXEC and OP_KIND["trigger"] == "instrument"
    assert "trigger" in _LEAD_REJECT and domain.accepts("trigger") == (domain.REAL,)
    assert opkeys.OP_KEYS["trigger"] >= {"pre_s", "post_s", "holdoff_s", "jitter_s",
                                         "n_segments", "average", "runt_level", "width_s"}


def test_trigger_on_another_channel_at_the_function_level():
    data = sine(phase=1.1, amp=0.4) + 2.0
    clock = sine(phase=0.3)
    ev = []
    y = TRIG.apply(data, FS, pre_s=100e-9, post_s=200e-9, level=0.0, on=clock, events=ev)
    start = ev[0].params["start"]
    assert np.array_equal(y, data[start:start + 300])                 # the data is what is cut
    exact = (2 * np.pi - 0.3) / (2 * np.pi * F0)                      # the CLOCK's first edge
    assert abs(ev[0].params["timestamp_s"] - exact) * FS < 1e-5
    with pytest.raises(ValueError, match="shares the record's timebase"):
        TRIG.apply(data, FS, pre_s=1e-7, post_s=2e-7, on=clock[:-1])


# ------------------------------------------------------------------ refusals
def test_no_trigger_is_refused_and_says_why():
    x = sine()
    with pytest.raises(ValueError, match="no rising edge through 5 .* outside that range"):
        TRIG.apply(x, FS, pre_s=1e-7, post_s=2e-7, level=5.0)
    ramp = np.linspace(1.0, -1.0, 5000)                                # only ever falls
    with pytest.raises(ValueError, match="no rising edge through 0 anywhere") as e:
        TRIG.apply(ramp, FS, pre_s=1e-7, post_s=2e-7, level=0.0)
    assert "outside that range" not in str(e.value)
    x, _ = pulses([50e-9, 60e-9])
    with pytest.raises(ValueError, match="no positive pulse at 0.5 between 1e-07 and 2e-07 s wide"):
        TRIG.apply(x, FS, pre_s=1e-7, post_s=2e-7, condition="width", level=0.5,
                   width_s=[100e-9, 200e-9])


def test_the_refusal_names_an_open_ended_width_in_words():
    x, _ = pulses([50e-9, 60e-9])
    kw = dict(pre_s=1e-7, post_s=2e-7, condition="width", level=0.5)
    with pytest.raises(ValueError, match="at least 1e-07 s wide"):
        TRIG.apply(x, FS, width_s=[100e-9, None], **kw)
    with pytest.raises(ValueError, match="at most 1e-08 s wide"):
        TRIG.apply(x, FS, width_s=[None, 10e-9], **kw)
    with pytest.raises(ValueError, match="no positive pulse at 0.5 that does not reach 0.9"):
        TRIG.apply(x, FS, pre_s=1e-7, post_s=2e-7, condition="runt", level=0.5,
                   runt_level=0.9)


def _needed(msg):
    return int(re.search(r"needs (?:at least|about) (\d+) samples", msg).group(1))


@pytest.mark.parametrize("kw", [dict(n_segments=60), dict(pre_s=30e-6),
                                dict(n_segments=5, average=10, holdoff_s=3.1e-6)])
def test_too_few_triggers_is_refused_with_a_render_length_that_is_enough(kw):
    """The refusal's number is checked by using it: a render that long triggers."""
    with pytest.raises(ValueError) as e:
        _sig(n=20000, rms=0, **kw).waveform()
    msg = str(e.value)
    need = _needed(msg)
    assert "this render is 20000 samples" in msg and need > 20000
    assert "Render longer (Grid(n=)" in msg
    sig = _sig(n=need, rms=0, **kw)
    assert len(sig.waveform()) == TRIG.record_length(sig.ops[-1], FS)
    assert need < 20000 * 40                                 # and it is not a wild overestimate


def test_the_exact_render_length_is_tight_when_the_render_merely_stops_short():
    x = sine(n=40000)
    with pytest.raises(ValueError, match="needs at least") as e:
        TRIG.apply(x[:1150], FS, pre_s=100e-9, post_s=1e-6, level=0.0)
    need = _needed(str(e.value))
    TRIG.apply(x[:need], FS, pre_s=100e-9, post_s=1e-6, level=0.0)
    with pytest.raises(ValueError):
        TRIG.apply(x[:need - 3], FS, pre_s=100e-9, post_s=1e-6, level=0.0)


@pytest.mark.parametrize("kw, match", [
    (dict(post_s=1e-7), "pre_s is required"),
    (dict(pre_s=1e-7), "post_s is required"),
    (dict(pre_s=-1e-9, post_s=1e-7), "pre_s is a number >= 0"),
    (dict(pre_s=1e-7, post_s=1e-7, condition="pattern"), "condition is one of"),
    (dict(pre_s=1e-7, post_s=1e-7, slope="up"), "slope is one of"),
    (dict(pre_s=1e-7, post_s=1e-7, n_segments=0), "n_segments is a whole number >= 1"),
    (dict(pre_s=1e-7, post_s=1e-7, average=2.5), "average is a whole number >= 1"),
    (dict(pre_s=1e-7, post_s=1e-7, jitter_s=-1e-12), "jitter_s is a number >= 0"),
    (dict(pre_s=1e-7, post_s=1e-7, level="mid"), "level is a number"),
    (dict(pre_s=1e-7, post_s=1e-7, width_s=[1e-9, 2e-9]), "not by an edge trigger"),
    (dict(pre_s=1e-7, post_s=1e-7, runt_level=0.5), "only by condition='runt'"),
    (dict(pre_s=1e-7, post_s=1e-7, condition="width"), "needs width_s"),
    (dict(pre_s=1e-7, post_s=1e-7, condition="width", width_s=5e-9), r"width_s is \[min, max\]"),
    (dict(pre_s=1e-7, post_s=1e-7, condition="width", width_s=[None, None]), "at least one"),
    (dict(pre_s=1e-7, post_s=1e-7, condition="width", width_s=[3e-9, 2e-9]), "min < max"),
    (dict(pre_s=1e-7, post_s=1e-7, condition="runt", level=0.2), "needs both thresholds"),
    (dict(pre_s=1e-7, post_s=1e-7, condition="runt", level=0.2, runt_level=0.8,
          slope="either"), "no single runt_level"),
    (dict(pre_s=1e-7, post_s=1e-7, condition="runt", level=0.8, runt_level=0.2),
     "runt_level above level"),
])
def test_a_bad_parameter_is_refused_where_it_is_written(kw, match):
    g = Grid(fs=FS, n=4000)
    with pytest.raises(ValueError, match=match):
        Signal(grid=g).carrier("sine", f_hz=F0).trigger(**kw)


def test_a_misspelled_parameter_is_refused_with_the_nearest_name():
    g = Grid(fs=FS, n=4000)
    with pytest.raises(TypeError, match="did you mean 'holdoff_s'"):
        Signal(grid=g).carrier("sine", f_hz=F0).trigger(pre_s=1e-7, post_s=1e-7, holdoff=1e-6)


def test_chain_refusals():
    g = Grid(fs=FS, n=40000)
    one = lambda **kw: Signal(grid=g).carrier("sine", f_hz=F0).trigger(
        pre_s=1e-7, post_s=2e-7, **kw)
    with pytest.raises(ValueError, match="already has a trigger at op 1"):
        one().trigger(pre_s=1e-8, post_s=1e-8)
    with pytest.raises(ValueError, match="digitize .* across the seams .* before `trigger`"):
        one(n_segments=3).digitize(enob=8)
    one().digitize(enob=8).waveform()                         # one window is an ordinary record
    one(average=4).digitize(enob=8).waveform()                # and so is an averaged one
    with pytest.raises(ValueError, match="lead_in: the 'trigger' op cannot"):
        one().with_lead_in(True).waveform()
    r = one(n_segments=2).recipe()
    r["ops"].append({"op": "scope", "bw_hz": 1e8})
    with pytest.raises(ValueError, match="across the seams"):
        Signal.from_recipe(r).waveform()
    with pytest.raises(ValueError, match="at least 2"):
        Signal(grid=g).carrier("sine", f_hz=F0).trigger(pre_s=1e-10, post_s=1e-10).waveform()


def test_a_complex_or_two_channel_record_is_refused():
    with pytest.raises(ValueError, match="is complex"):
        TRIG.apply(sine() + 0j, FS, pre_s=1e-7, post_s=2e-7)
    with pytest.raises(ValueError, match="not one waveform on a timebase"):
        TRIG.apply(np.stack([sine(), sine()]), FS, pre_s=1e-7, post_s=2e-7)
    with pytest.raises(ValueError, match="NaN"):
        TRIG.apply(np.where(np.arange(40000) == 7, np.nan, sine()), FS, pre_s=1e-7, post_s=2e-7)
    gi = Grid(fs=1e9, baud=25e6, n=1 << 14, domain="iq", fc_hz=100e6)
    with pytest.raises(ValueError, match="complex baseband record"):
        Signal(grid=gi).iq("qpsk", prbs=7).trigger(pre_s=1e-7, post_s=2e-7)


def test_a_chain_without_a_trigger_never_reaches_this_module(monkeypatch):
    """Existing records are untouched: with no trigger op the executor is not called, and the
    chain check returns before reading anything."""
    def boom(*a, **k):
        raise AssertionError("trigger code ran on a chain with no trigger")
    monkeypatch.setattr(TRIG, "apply", boom)
    monkeypatch.setattr(TRIG, "check", boom)
    g = Grid(fs=64e9, baud=16e9, n=4096)
    sig = Signal(seed=3, grid=g).carrier("nrz", n_ui=1024).lossy(loss_db=6.0).digitize(snr_db=30)
    x, ev = sig.realize()
    assert len(x) == 4096 and sig.stored_grid(x).n == 4096 and sig.roles() == ["noise/2"]
