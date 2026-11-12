"""`ws.Segments`: a segmented acquisition of a rare event, each occurrence rendered on its own
(#97). What is asserted is the EFFECT -- against `trigger(n_segments=)`, against the values
the recipe states, and against the arrival distribution -- with each bound stated beside it."""
from __future__ import annotations

import json
import math
import time

import numpy as np
import pytest
from scipy import stats

import wfmsynth as ws
from wfmsynth import Signal, compose
from wfmsynth import tolerance as TOL

FS = 1e9
G = ws.Grid(fs=FS, n=3000)
TRIG = dict(pre_s=200e-9, post_s=1.5e-6, level=0.5)


def step(amp=None, sag=None, noise=0.0):
    """A step from 0 to 1 at 1 us: the occurrence. `amp` and `sag` are gains on it."""
    s = Signal().carrier("step", t_step_s=1e-6, tr_s=20e-9, amp=0.5, offset=0.5)
    if amp is not None:
        s = s.gain(by=amp)
    if sag is not None:
        s = s.gain(by=sag)
    return s.rx_noise(rms=noise) if noise else s


def seg(chain=None, n=40, arrival=None, **kw):
    return ws.Segments(step(ws.tol(1.0, 10, per="segment"), noise=0.01) if chain is None
                       else chain, n=n,
                       arrival=arrival or dict(dist="exponential", mean_s=1e-3),
                       trigger=kw.pop("trigger", TRIG), seed=kw.pop("seed", 1), grid=G, **kw)


def rows(x, n):
    return x.reshape(n, -1)


# ---------------------------------------------------------------- the commonest case
def test_output_is_windows_end_to_end_with_trigger_events():
    s = seg()
    x, ev = s.realize()
    n_win = int(round((TRIG["pre_s"] + TRIG["post_s"]) * FS))
    fired = [e for e in ev if e.kind == "trigger"]
    assert x.shape == (40 * n_win,) and len(fired) == 40
    assert [e.params["segment"] for e in fired] == list(range(40))
    # the same fields `trigger(n_segments=)` emits, with the same meaning
    want = set(Signal(seed=1, grid=ws.Grid(fs=FS, n=20000)).carrier("sine", f_hz=1e6)
               .trigger(pre_s=200e-9, post_s=300e-9, level=0.0, n_segments=2)
               .realize()[1].events[0].params)
    assert set(fired[0].params) == want
    for k, e in enumerate(fired):              # the crossing sits pre_s in, to half a sample
        assert abs(e.t_s - k * n_win / FS - TRIG["pre_s"]) * FS <= 0.5 + 1e-9
    assert s.stored_grid(x).fs == FS and s.stored_grid(x).n == len(x)
    stamps = np.array([e.params["timestamp_s"] for e in fired])
    arrive = np.array([o["arrival_s"] for o in s.occurrences()])
    # the crossing is 1 us + half the 20 ns edge into each occurrence's render
    assert np.all(np.abs(stamps - arrive - 1e-6) < 30e-9)


def test_replays_from_its_recipe_alone_and_digest_is_stable():
    s = seg()
    x = s.waveform()
    again = ws.Segments.from_recipe(json.loads(s.to_json()))
    assert np.array_equal(again.waveform(), x)
    assert again.sha256() == s.sha256() == seg().sha256()
    with pytest.raises(ValueError, match="Segments.from_recipe"):
        Signal.from_recipe(s.recipe())


# ---------------------------------------------------------------- against trigger(n_segments=)
def test_stationary_chain_places_the_trigger_as_trigger_does():
    """A clock through Segments and through trigger(n_segments=): the crossing's position in
    its window is pre_s - jitter plus the half-sample rounding, uniform because the scope's
    clock is free-running. Two-sample KS on the positions, p > 0.01, and each spread within
    15 % of sqrt(dt^2/12 + jitter^2) (400 windows: an rms known to about 4 %)."""
    n, jit = 400, 0.4e-9
    trig = dict(pre_s=100e-9, post_s=200e-9, level=0.0, jitter_s=jit)

    def clock():
        return Signal().carrier("square", f_hz=3.7e6, tr_frac=0.02).rx_noise(rms=0.01)

    long = Signal(seed=2, grid=ws.Grid(fs=FS, n=n * 600))
    long.ops = clock().ops
    xt, et = long.trigger(**trig, n_segments=n).realize()
    s = ws.Segments(clock(), n=n, arrival=dict(dist="exponential", mean_s=1e-3), trigger=trig,
                    seed=2, grid=ws.Grid(fs=FS, n=600))
    xs, es = s.realize()

    def placed(x, ev):
        out = []
        for k, r in enumerate(rows(x, n)):
            i = 95 + int(np.flatnonzero((r[95:-1] < 0) & (r[96:] >= 0))[0])
            out.append(i + (0 - r[i]) / (r[i + 1] - r[i]) - 100.0)
        return np.array(out)

    pt, ps = placed(xt, et), placed(xs, es)
    want = math.sqrt(1 / 12 + (jit * FS) ** 2)
    p = stats.ks_2samp(pt, ps).pvalue
    print(f"placement rms: trigger {pt.std():.4f}, Segments {ps.std():.4f}, predicted "
          f"{want:.4f} samples; KS p = {p:.3f}")
    assert p > 0.01
    assert abs(pt.std() / want - 1) < 0.15 and abs(ps.std() / want - 1) < 0.15
    # and the jitter each window was moved by is the one its event reports
    js = np.array([e.params["jitter_s"] for e in es if e.kind == "trigger"])
    assert np.abs(ps + js * FS).max() <= 0.5 + 0.05      # 0.05: the noise moves the crossing


def test_an_event_is_retimed_into_its_window_as_trigger_does():
    """A glitch stated at 1.3 us into the occurrence sits, in its window, that far after the
    crossing as the crossing is after the occurrence's start -- to one sample, the glitch's
    own placement."""
    chain = step().events("glitch", on="times", times=[1.3e-6])
    s = seg(chain, n=12)
    x, ev = s.realize()
    occ = s.occurrences()
    trig = {e.params["segment"]: e for e in ev if e.kind == "trigger"}
    glitch = [e for e in ev if e.kind == "glitch"]
    assert len(glitch) == 12 and trig[0].params["dropped"] == 0
    for g_ev in glitch:
        k = int(g_ev.sample // (len(x) // 12))
        cross_local = trig[k].params["timestamp_s"] - math.floor(occ[k]["arrival_s"] * FS) / FS
        assert abs((g_ev.t_s - trig[k].t_s) - (1.3e-6 - cross_local)) * FS <= 1.0


# ---------------------------------------------------------------- per-occurrence values
def test_drawn_amplitude_is_what_each_window_shows():
    s = seg(step(ws.tol(1.0, 10, per="segment"), ws.trend("ramp", start=1.0, stop=0.8)), n=30)
    x = rows(s.waveform(), 30)
    amp = [p["by"] for p in ws.segment_params(s.recipe(), "gain", which=0)]
    sag = [p["by"] for p in ws.segment_params(s.recipe(), "gain", which=1)]
    occ = s.occurrences()
    assert amp == [o["values"]["vary/1/by"] for o in occ]
    assert sag == pytest.approx(np.linspace(1.0, 0.8, 30), abs=1e-15)
    plateau = x[:, -200:].mean(1)
    err = np.abs(plateau / (np.array(amp) * np.array(sag)) - 1).max()
    print(f"plateau against drawn x trend: worst {err:.2e}")
    assert err < 1e-9
    assert all(0.9 <= a <= 1.1 for a in amp) and len(set(amp)) == 30


def test_per_segment_value_outside_segments_is_refused_by_name():
    chain = step(ws.tol(1.0, 10, per="segment"))
    with pytest.raises(ValueError, match="per occurrence.*ws.Segments"):
        chain.waveform()
    sig = Signal(seed=1, grid=G).carrier("step", t_step_s=1e-6).gain(by=ws.trend(
        "step", before=1.0, after=0.5, at=3))
    with pytest.raises(ValueError, match="trend.*ws.Segments"):
        sig.waveform()
    rec = ws.Channels({"a": chain}, seed=1, grid=G)
    with pytest.raises(ValueError, match="ws.Segments"):
        rec.waveforms()


def test_a_tolerance_without_per_is_the_dict_it_always_was():
    assert ws.tol(1.0, 5) == {"nom": 1.0, "tol_pct": 5}
    assert ws.corners({"a": 1.0}) == {"corners": {"a": 1.0}}
    s = Signal(seed=3, grid=G).carrier("step", t_step_s=1e-6).gain(by=ws.tol(1.0, 5))
    assert "_segment" not in s.ops[1] and set(s.ops[1]["_unit"]) == {"by"}


# ---------------------------------------------------------------- roles and contrast
def test_roles_are_exactly_what_is_drawn():
    s = seg(n=5, trigger=dict(TRIG, jitter_s=1e-9))
    r = s.roles()
    assert r[:3] == ["arrival", "vary/1/by", "trigger"]
    assert r[3:] == [f"segment/{k}/rx_noise/2" for k in range(5)]
    base = s.waveform()
    for name in r:
        assert not np.array_equal(s.contrast(name, seed=99), base), name
    with pytest.raises(ValueError, match="names no randomness"):
        s.contrast("variant")
    fixed = seg(n=5, arrival=dict(dist="fixed", t_s=1e-3))
    assert "arrival" not in fixed.roles()


def test_contrast_moves_one_thing():
    s = seg(n=20)
    occ = s.occurrences()
    # re-roll the draws: arrivals and every noise sample stay; only amplitudes move
    xa = rows(s.contrast("vary/1/by"), 20)
    x = rows(s.waveform(), 20)
    st = ws.Streams(s.seed).reroll("vary/1/by")
    occ_a = s.occurrences(st)
    assert [o["arrival_s"] for o in occ_a] == [o["arrival_s"] for o in occ]
    assert all(a["values"]["vary/1/by"] != b["values"]["vary/1/by"] for a, b in zip(occ, occ_a))
    assert np.abs(xa[:, -200:].mean(1) / x[:, -200:].mean(1) - np.array(
        [b["values"]["vary/1/by"] / a["values"]["vary/1/by"] for a, b in zip(occ, occ_a)])
    ).max() < 2e-3
    # re-roll every occurrence's noise: the draws and arrivals are the record's
    xn = rows(s.contrast("segment/*/rx_noise/2"), 20)
    assert not np.array_equal(xn, x)
    assert np.abs(xn[:, -200:].mean(1) - x[:, -200:].mean(1)).max() < 5e-3


def test_removing_the_trend_leaves_every_draw_and_noise_sample():
    """`without("trend/*")` is the cause taken out: the arrivals, the drawn amplitudes and the
    noise are the acquisition's own. Each window is aligned on its `start` in the
    occurrence's render; before the step, where the trend scales only the source's own 2e-15
    baseline and the edge's far tail, the samples (the 0.01 rms noise) agree to 1e-13 over the first 20; after it, the plateau ratio is the
    trend's value to 2e-3 (0.01 rms noise averaged over 300 samples)."""
    s = seg(step(ws.tol(1.0, 10, per="segment"), ws.trend("ramp", start=1.0, stop=0.7),
                 noise=0.01), n=25)
    flat = s.without("trend/*")
    assert s.trends() == ["trend/2/by"] and flat.trends() == []
    a, b = s.occurrences(), flat.occurrences()
    assert [o["arrival_s"] for o in a] == [o["arrival_s"] for o in b]
    assert [o["values"]["vary/1/by"] for o in a] == [o["values"]["vary/1/by"] for o in b]
    assert all(set(o["values"]) == {"vary/1/by"} for o in b)
    (x, ex), (y, ey) = s.realize(), flat.realize()
    n_win = len(x) // 25
    sx = [e.params["start"] for e in ex if e.kind == "trigger"]
    sy = [e.params["start"] for e in ey if e.kind == "trigger"]
    ratio = []
    for k in range(25):
        d = sy[k] - sx[k]                         # the windows of one render, aligned
        u = x[k * n_win:(k + 1) * n_win][max(d, 0):][:n_win - abs(d)]
        v = y[k * n_win:(k + 1) * n_win][max(-d, 0):][:n_win - abs(d)]
        assert np.abs(u[:20] - v[:20]).max() < 1e-13     # 180 ns before the step: noise
        ratio.append(np.mean(u[-300:]) / np.mean(v[-300:]))
    trend = np.linspace(1.0, 0.7, 25)
    print(f"plateau ratio against the removed trend: worst "
          f"{np.abs(np.array(ratio) / trend - 1).max():.2e}")
    assert np.abs(np.array(ratio) / trend - 1).max() < 2e-3   # 0.01 noise over 300 samples
    with pytest.raises(ValueError, match="not a trend"):
        s.without("vary/1/by")


# ---------------------------------------------------------------- arrival and cost
def test_arrivals_follow_the_stated_distribution():
    spec = dict(dist="lognormal", median_s=2e-3, sigma=0.6)
    s = seg(n=300, arrival=spec)
    t = np.array([o["arrival_s"] for o in s.occurrences()])
    gaps = np.diff(np.concatenate([[0.0], t]))
    p = stats.kstest(gaps, stats.lognorm(s=0.6, scale=2e-3).cdf).pvalue
    print(f"300 lognormal gaps: KS p = {p:.3f}")
    assert p > 0.01
    fixed = seg(n=10, arrival=dict(dist="fixed", t_s=7e-6))
    occ = fixed.occurrences()
    assert np.allclose(np.diff([o["arrival_s"] for o in occ]), 7e-6, rtol=0, atol=1e-15)
    # a period shorter than the 3 us render: every other occurrence arrives while the last
    # is still being stored, and is missed and counted
    short = seg(n=6, arrival=dict(dist="fixed", t_s=2e-6))
    assert [o["missed"] for o in short.occurrences()] == [0, 1, 1, 1, 1, 1]


def test_render_cost_is_n_times_window_not_the_arrival_span(monkeypatch):
    rendered = []
    run = compose.Signal._run

    def counting(self, *a, **k):
        out = run(self, *a, **k)
        rendered.append(len(out[0] if isinstance(out, tuple) else out))
        return out

    monkeypatch.setattr(compose.Signal, "_run", counting)
    took = []
    for mean_s in (1e-3, 1e3):                   # spans of about 0.1 s and 1e5 s
        rendered.clear()
        t0 = time.perf_counter()
        seg(n=100, arrival=dict(dist="exponential", mean_s=mean_s)).waveform()
        took.append(time.perf_counter() - t0)
        assert rendered == [G.n] * 100
    print(f"render time ratio across a 1e6 x longer span: {took[1] / took[0]:.2f}")
    assert took[1] / took[0] < 2.0


# ---------------------------------------------------------------- variants
def test_variants_are_drawn_under_their_own_role_and_recorded():
    late = Signal().carrier("step", t_step_s=1.3e-6, tr_s=20e-9).rx_noise(rms=0.01)
    s = seg(n=400, variants={"late": dict(p=0.1, chain=late)})
    occ = s.occurrences()
    x, ev = s.realize()
    names = [o["variant"] for o in occ]
    got = sum(v == "late" for v in names)
    assert abs(got - 40) < 4 * math.sqrt(400 * 0.1 * 0.9)                  # binomial, 4 sigma
    assert [e.params["variant"] for e in ev if e.kind == "trigger"] == names
    stamps = np.array([e.params["timestamp_s"] - o["arrival_s"]
                       for e, o in zip([e for e in ev if e.kind == "trigger"], occ)])
    assert np.all(np.isclose(stamps[np.array(names) == "late"], 1.31e-6, atol=30e-9))
    assert "variant" in s.roles()
    occ_b = s.occurrences(ws.Streams(1).reroll("variant"))
    assert [o["arrival_s"] for o in occ_b] == [o["arrival_s"] for o in occ]
    assert [o["variant"] for o in occ_b] != names
    stated = seg(n=10, variants={"late": dict(at=[3, 7], chain=late)})
    assert [o["variant"] for o in stated.occurrences()].count("late") == 2
    assert "variant" not in stated.roles()


def test_a_variants_parts_are_named_under_it():
    late = step(ws.tol(1.0, 20, per="segment")).gain(by=ws.tol(1.0, 5))
    s = seg(step(ws.tol(1.0, 10, per="segment")), n=50,
            variants={"late": dict(p=0.2, chain=late)})
    r = s.roles()
    assert "vary/1/by" in r and "late/vary/1/by" in r and "late/unit/2/by" in r
    rec = s.recipe()
    assert [e["role"] for e in TOL.resolved(rec)] == ["late/unit/2/by"]


# ---------------------------------------------------------------- units nest
def test_a_unit_holds_for_the_acquisition_and_segments_draw_inside_it():
    chain = step(ws.tol(1.0, 10, per="segment")).gain(by=ws.tol(1.0, 20))
    s = seg(chain, n=20)
    other = s.unit(7)
    assert ws.unit_of(s.recipe()) == 1 and ws.unit_of(other.recipe()) == 7
    a, b = s.occurrences(), other.occurrences()
    assert [o["values"] for o in a] == [o["values"] for o in b]
    ua = TOL.resolved(s.recipe())[0]["value"]
    ub = TOL.resolved(other.recipe())[0]["value"]
    x, y = rows(s.waveform(), 20), rows(other.waveform(), 20)
    assert ua != ub
    assert np.allclose(x[:, -200:].mean(1) / y[:, -200:].mean(1), ua / ub, rtol=2e-3)
    assert not np.array_equal(s.contrast("unit/*", seed=5), s.waveform())


# ---------------------------------------------------------------- refusals
@pytest.mark.parametrize("kw, said", [
    (dict(trigger=dict(pre_s=1e-7, post_s=1e-6)), "needs level="),
    (dict(arrival=1e-3), "a period or a mean"),
    (dict(trigger=dict(TRIG, holdoff_s=1e-6)), "missed and counted"),
    (dict(trigger=dict(TRIG, n_segments=4)), "how many occurrences"),
    (dict(variants={"late": dict(p=1.0, chain=step())}), "strictly between 0"),
    (dict(variants={"main": dict(p=0.1, chain=step())}), "not 'main'"),
    (dict(arrival=dict(dist="poisson", mean_s=1.0)), "'dist' is one of"),
])
def test_refused_by_name(kw, said):
    with pytest.raises(ValueError, match=said):
        seg(**kw)


def test_an_occurrence_whose_render_is_too_short_is_refused_with_the_length():
    s = ws.Segments(step(), n=3, arrival=dict(dist="fixed", t_s=1e-3),
                    trigger=dict(pre_s=200e-9, post_s=3e-6, level=0.5), seed=1, grid=G)
    with pytest.raises(ValueError, match=r"occurrence 0.*needs at least (\d+) samples") as e:
        s.waveform()
    need = int(__import__("re").search(r"needs at least (\d+) samples", str(e.value)).group(1))
    ok = ws.Segments(step(), n=3, arrival=dict(dist="fixed", t_s=1e-3),
                     trigger=dict(pre_s=200e-9, post_s=3e-6, level=0.5), seed=1,
                     grid=ws.Grid(fs=FS, n=need))
    assert len(ok.waveform()) == 3 * 3200


def test_a_ramp_on_a_count_is_refused():
    with pytest.raises(ValueError, match="whole number"):
        Signal().carrier("nrz", n_ui=ws.trend("ramp", start=8, stop=16))
    with pytest.raises(ValueError, match="takes"):
        ws.trend("ramp", start=1.0)
    with pytest.raises(ValueError, match="per="):
        ws.tol(1.0, 5, per="capture")
