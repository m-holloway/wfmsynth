"""`carrier("states")`: a level that dwells in one of a few states and jumps (issue #90).

What is asserted is the EFFECT, each against something that shares nothing with how the
source draws: dwell times against `scipy.stats` distributions, occupancy against the
stationary vector from an eigen-decomposition, edge times against crossings read off the
samples, a telegraph spectrum against its Lorentzian, and a gated carrier against the
intervals the source said it had. Every figure is printed (`pytest -s`).
"""
import json
import warnings

import numpy as np
import pytest
from scipy import signal as sps
from scipy import stats

import wfmsynth as ws
from wfmsynth import Grid, Signal
from wfmsynth import states as S
from wfmsynth.streams import Streams

G = Grid(fs=1e8, n=1 << 18)               # 10 ns per sample, 2.6 ms
KS_1PCT = 1.628                           # Kolmogorov-Smirnov: D * sqrt(N) at the 1 % level


def lengths(iv, state=None):
    """The dwells the record holds whole, in seconds."""
    return np.array([r["t_stop_s"] - r["t_start_s"] for r in iv
                     if r["complete"] and (state is None or r["state"] == state)])


def crossings(x, level):
    """Where `x` crosses `level`, in samples, by a cubic through the four samples about each
    crossing -- an instrument that does not know how the edge was made."""
    s = np.asarray(x, float) - level
    out = []
    for k in np.flatnonzero((s[:-1] < 0) != (s[1:] < 0)):
        if k < 1 or k + 2 >= s.size:
            continue
        r = np.polynomial.Polynomial.fit(np.arange(-1, 3), s[k - 1:k + 3], 3).convert().roots()
        r = r[np.abs(r.imag) < 1e-9].real
        r = r[(r >= -1e-9) & (r <= 1 + 1e-9)]
        if r.size:
            out.append(k + r[0])
    return np.array(out)


# ------------------------------------------------------------------ the simplest call
def test_one_line_is_a_two_state_telegraph_with_its_truth_and_its_role():
    sig = Signal(seed=1, grid=G).carrier("states", dwell=2e-6)
    st = Streams(sig.seed)
    x = sig.waveform(streams=st)
    ev = sig.event_list()
    iv = S.intervals(ev)
    assert sig.roles() == ["states/0"] and st.drawn() == {"states/0"}
    assert not np.array_equal(sig.contrast("states/0", seed=9), x)
    assert {e.kind for e in ev} == {"state"} and all(e.at == "" and e.on_record for e in ev)
    # the intervals tile the record, alternate, and the samples sit on the stated level
    assert iv[0]["t_start_s"] == 0.0 and iv[-1]["t_stop_s"] == G.n / G.fs
    assert all(a["t_stop_s"] == b["t_start_s"] and a["state"] != b["state"]
               for a, b in zip(iv, iv[1:]))
    assert not iv[0]["complete"] and not iv[-1]["complete"]
    t = np.arange(G.n) / G.fs
    worst = 0.0
    for r in iv:
        m = (t >= r["t_start_s"] + 8 / G.fs) & (t <= r["t_stop_s"] - 8 / G.fs)
        if m.any():
            worst = max(worst, np.abs(x[m] - r["level"]).max())
    print(f"{len(iv)} intervals; worst |sample - stated level| 8 samples inside one: {worst:.2e}")
    assert worst < 5e-3
    # the recipe is data and replays with nothing else
    again = Signal.from_recipe(json.loads(json.dumps(sig.recipe())))
    assert np.array_equal(again.waveform(), x) and again.sha256() == sig.sha256()
    assert Signal(seed=1, grid=G).carrier("states", dwell=2.1e-6).sha256() != sig.sha256()


# ------------------------------------------------------------------ dwell distributions
DISTS = {
    "exponential": (3e-6, stats.expon(scale=3e-6)),
    "lognormal": ({"dist": "lognormal", "median_s": 2e-6, "sigma": 0.7},
                  stats.lognorm(s=0.7, scale=2e-6)),
    "uniform": ({"dist": "uniform", "lo_s": 1e-6, "hi_s": 5e-6},
                stats.uniform(loc=1e-6, scale=4e-6)),
}


@pytest.mark.parametrize("name", sorted(DISTS))
def test_dwell_times_follow_the_stated_distribution(name):
    """Kolmogorov-Smirnov against scipy's CDF, at the 1 % level. State 0 has the distribution
    under test and state 1 a fixed dwell, so a mix-up between the states would fail it."""
    spec, dist = DISTS[name]
    sig = Signal(seed=11, grid=G).carrier("states", dwell=[spec, {"dist": "fixed", "t_s": 1e-6}])
    iv = S.intervals(sig.event_list())
    d0, d1 = lengths(iv, 0), lengths(iv, 1)
    ks = stats.kstest(d0, dist.cdf)
    crit = KS_1PCT / np.sqrt(d0.size)
    print(f"{name}: N={d0.size} mean {d0.mean():.4g} s (stated {dist.mean():.4g}); "
          f"KS D={ks.statistic:.4f} against a 1 % critical value of {crit:.4f}; the fixed "
          f"state is off by at most {np.abs(d1 - 1e-6).max():.1e} s")
    assert d0.size > 400 and ks.statistic < crit
    assert np.abs(d1 - 1e-6).max() < 1e-15
    # and the instrument can tell: the same dwells against a distribution 15 % longer fail
    assert stats.kstest(d0 * 1.15, dist.cdf).statistic > crit


def test_a_table_redraws_measured_dwells_in_their_stated_proportions():
    t_s, w = [1e-6, 2.5e-6, 7e-6], [5, 3, 2]
    sig = Signal(seed=3, grid=G).carrier(
        "states", dwell=[{"dist": "table", "t_s": t_s, "weights": w}, 2e-6])
    d = lengths(S.intervals(sig.event_list()), 0)
    got = np.array([np.sum(np.abs(d - v) < 1e-12) for v in t_s])
    chi = stats.chisquare(got, d.size * np.array(w) / 10.0)
    print(f"table: N={d.size} counts {got.tolist()} for weights {w}; chi-square p={chi.pvalue:.3f}")
    assert got.sum() == d.size and chi.pvalue > 0.01


def test_a_censored_interval_says_so_and_is_not_a_draw():
    """The interval in force at t = 0 is already part-way through: with a FIXED dwell its
    remainder is uniform on (0, T), mean T/2, which the stated distribution never gives."""
    rem = []
    for seed in range(400):
        ev = Signal(seed=seed, grid=Grid(fs=1e8, n=2048)).carrier(
            "states", dwell={"dist": "fixed", "t_s": 4e-6}).event_list()
        first = S.intervals(ev)[0]
        assert not first["complete"]
        rem.append(first["t_stop_s"])
    rem = np.array(rem)
    ks = stats.kstest(rem, stats.uniform(scale=4e-6).cdf)
    print(f"remainder of the first interval over 400 seeds: mean {rem.mean():.3e} s "
          f"(T/2 = 2e-6), KS D={ks.statistic:.3f} against uniform(0, T), 1 % critical "
          f"{KS_1PCT / 20:.3f}")
    assert ks.statistic < KS_1PCT / 20


# ------------------------------------------------------------------ occupancy
P3 = [[0, 0.3, 0.7], [0.5, 0, 0.5], [1, 0, 0]]
D3 = [5e-6, {"dist": "lognormal", "median_s": 1e-6, "sigma": 0.8},
      {"dist": "uniform", "lo_s": 2e-6, "hi_s": 6e-6}]


def theory3():
    """Time fractions from the left eigenvector of the transition table and the mean dwells
    written out by hand -- not through `states.occupancy`."""
    w, v = np.linalg.eig(np.asarray(P3, float).T)
    nu = np.real(v[:, np.argmin(np.abs(w - 1))])
    nu = nu / nu.sum()
    t = nu * np.array([5e-6, 1e-6 * np.exp(0.5 * 0.8 ** 2), 4e-6])
    return t / t.sum()


def test_stationary_occupancy_matches_theory_in_time_and_at_t0():
    want = theory3()
    assert np.allclose(S.occupancy(D3, P3), want, atol=1e-12)
    g = Grid(fs=1e8, n=1 << 21)                                      # 21 ms
    sig = Signal(seed=4, grid=g).carrier("states", levels=[0.0, 0.4, 1.0], dwell=D3,
                                         transitions=P3, tr_s=50e-9)
    x, ev = sig.realize()
    iv = S.intervals(ev)
    occ = np.zeros(3)
    for r in iv:
        occ[r["state"]] += r["t_stop_s"] - r["t_start_s"]
    occ /= occ.sum()
    # the samples agree with the events: time nearest each level
    near = np.bincount(np.argmin(np.abs(x[:, None] - np.array([0.0, 0.4, 1.0])), axis=1),
                       minlength=3) / x.size
    # which state follows which, counted
    n_ij = np.zeros((3, 3))
    for a, b in zip(iv, iv[1:]):
        n_ij[a["state"], b["state"]] += 1
    p_hat = n_ij / n_ij.sum(axis=1, keepdims=True)
    p3 = np.asarray(P3, float)
    sd = np.sqrt(p3 * (1 - p3) / n_ij.sum(axis=1, keepdims=True))
    z = np.abs(p_hat - p3)[sd > 0] / sd[sd > 0]
    print(f"occupancy over {len(iv)} intervals: events {np.round(occ, 4).tolist()}, samples "
          f"{np.round(near, 4).tolist()}, theory {np.round(want, 4).tolist()}; transition "
          f"frequencies within {z.max():.2f} sigma of the table (worst {np.abs(p_hat - p3).max():.4f})")
    assert np.abs(occ - want).max() < 0.015 and np.abs(near - want).max() < 0.02
    assert z.max() < 3.5 and np.array_equal(n_ij == 0, p3 == 0)
    # a stationary start: the state at t = 0 is drawn in proportion to those time fractions
    _, specs, p, start = S._plan(levels=[0, 1, 2], dwell=D3, transitions=P3)
    trials = 4000
    first = np.array([S.draw(np.random.default_rng(k), 1e-9, specs, p, start)[0][0]
                      for k in range(trials)])
    frac = np.bincount(first, minlength=3) / trials
    sigma = np.sqrt(want * (1 - want) / trials)
    print(f"state at t=0 over {trials} draws: {np.round(frac, 4).tolist()}; worst deviation "
          f"{np.abs((frac - want) / sigma).max():.2f} sigma")
    assert np.abs((frac - want) / sigma).max() < 3.5


def test_start_names_the_state_entered_at_t0():
    sig = Signal(seed=2, grid=G).carrier("states", levels=[0.0, 1.0, 2.0], dwell=4e-6, start=2)
    iv = S.intervals(sig.event_list())
    assert iv[0]["state"] == 2 and iv[0]["t_start_s"] == 0.0 and iv[0]["complete"]
    first = [S.intervals(Signal(seed=k, grid=Grid(fs=1e8, n=1 << 14)).carrier(
        "states", dwell=4e-6, start=1).event_list())[0] for k in range(300)]
    d = np.array([r["t_stop_s"] for r in first if r["complete"]])
    # censored at the record's 163.84 us, which an exponential of mean 4 us never reaches
    assert d.size == 300 and all(r["state"] == 1 for r in first)
    ks = stats.kstest(d, stats.expon(scale=4e-6).cdf)
    print(f"first dwell under start=1, 300 seeds: mean {d.mean():.3e} s, KS D={ks.statistic:.3f}")
    assert ks.statistic < KS_1PCT / np.sqrt(300)


# ------------------------------------------------------------------ edges between samples
def test_a_jump_is_at_its_own_time_not_on_the_grid():
    g = Grid(fs=1e8, n=1 << 17)
    sig = Signal(seed=6, grid=g).carrier("states", levels=[-1.0, 1.0], dwell=8e-6, tr_s=160e-9)
    x, ev = sig.realize()
    edges = np.array([e.t_s for e in ev][1:]) * g.fs
    isolated = edges[np.r_[True, np.diff(edges) > 64] & np.r_[np.diff(edges) > 64, True]]
    isolated = isolated[(isolated > 64) & (isolated < g.n - 64)]
    got = crossings(x, 0.0)
    err = np.array([np.abs(got - e).min() for e in isolated])
    frac = isolated % 1.0
    print(f"{isolated.size} isolated jumps at fractions spread {frac.min():.3f}..{frac.max():.3f} "
          f"of a sample; |crossing - stated time| max {err.max():.2e} sample; snapped to the "
          f"grid it would be up to {np.abs(np.round(isolated) - isolated).max():.2f}")
    assert isolated.size > 100 and err.max() < 2e-3
    # the process is the same on a faster grid and on a longer record: the same intervals
    fine = S.intervals(Signal(seed=6, grid=Grid(fs=4e8, n=1 << 19)).carrier(
        "states", levels=[-1.0, 1.0], dwell=8e-6, tr_s=160e-9).event_list())
    here = S.intervals(ev)
    assert [r["t_start_s"] for r in fine] == [r["t_start_s"] for r in here]
    longer = Signal(seed=6, grid=Grid(fs=1e8, n=1 << 18)).carrier(
        "states", levels=[-1.0, 1.0], dwell=8e-6, tr_s=160e-9).waveform()
    assert np.abs(longer[:g.n - 256] - x[:-256]).max() < 1e-9


def test_the_edge_is_the_rise_time_asked_for_and_the_default_is_four_samples():
    g = Grid(fs=1e8, n=1 << 14)
    for tr_s, want in ((None, 4.0), (200e-9, 20.0)):
        kw = {} if tr_s is None else {"tr_s": tr_s}
        x = Signal(seed=1, grid=g).carrier(
            "states", dwell={"dist": "fixed", "t_s": 20e-6}, start=0, **kw).waveform()
        a, b = crossings(x[1500:2500], 0.1), crossings(x[1500:2500], 0.9)
        print(f"tr_s={tr_s}: 10-90 % measured {b[0] - a[0]:.3f} samples, stated {want}")
        assert abs((b[0] - a[0]) / want - 1) < 0.03
    with pytest.warns(RuntimeWarning, match="edge time clamped"):
        Signal(seed=1, grid=g).carrier("states", dwell=20e-6, tr_s=1e-9).waveform()


# ------------------------------------------------------------------ used directly: burst noise
def test_burst_noise_is_added_with_combine_and_has_a_lorentzian_spectrum():
    """An asymmetric telegraph signal, mean dwells t0 and t1, step dv: variance
    dv^2 * p0 * p1 and one-sided PSD 4 dv^2 p0 p1 r / (r^2 + (2 pi f)^2), r = 1/t0 + 1/t1."""
    g = Grid(fs=1e7, n=1 << 21)
    t0, t1, dv = 40e-6, 10e-6, 0.05
    rts = Signal().carrier("states", levels=[0.0, dv], dwell=[t0, t1])
    clean = Signal(seed=8, grid=g).carrier("sine", f_hz=1e5, amp=0.5)
    sig = clean.copy().combine(rts)
    x, ev = sig.realize()
    assert sig.roles() == ["combine/1/others/0/states/0"]
    noise = sig.subchain("combine/1/others/0")
    assert np.array_equal(x, clean.waveform() + noise)
    mine = [e for e in ev if e.kind == "state"]
    assert mine and all(e.at == "combine/1/others/0" and not e.on_record for e in mine)
    p0, p1, r = t0 / (t0 + t1), t1 / (t0 + t1), 1 / t0 + 1 / t1
    f, pxx = sps.welch(noise, fs=g.fs, nperseg=1 << 14)
    band = (f >= 1e3) & (f <= 2e5)                       # the corner is r / 2 pi = 19.9 kHz
    want = 4 * dv ** 2 * p0 * p1 * r / (r ** 2 + (2 * np.pi * f[band]) ** 2)
    err_db = 10 * np.log10(pxx[band] / want)
    print(f"burst noise: variance {noise.var():.4e} (theory {dv ** 2 * p0 * p1:.4e}), mean "
          f"{noise.mean():.5f} (theory {dv * p1:.5f}); PSD against the Lorentzian over "
          f"1 kHz..200 kHz: mean {err_db.mean():+.2f} dB, rms {err_db.std():.2f} dB")
    assert abs(noise.var() / (dv ** 2 * p0 * p1) - 1) < 0.08
    assert abs(noise.mean() / (dv * p1) - 1) < 0.06
    assert abs(err_db.mean()) < 0.5 and err_db.std() < 1.0
    # two levels, not a Gaussian: nearly every sample is on one of them
    on_level = np.mean(np.minimum(np.abs(noise), np.abs(noise - dv)) < 0.02 * dv)
    print(f"fraction of samples within 2 % of a level: {on_level:.3f}")
    assert on_level > 0.9


# ------------------------------------------------------------------ as a schedule
SCHEDULE = dict(dwell=[5e-6, {"dist": "lognormal", "median_s": 2e-6, "sigma": 0.5}])


@pytest.mark.parametrize("tr_s,causal", [(None, False), (100e-9, False), (100e-9, True)])
def test_a_gated_carrier_is_idle_exactly_where_the_intervals_say(tr_s, causal):
    tr = 4 / G.fs if tr_s is None else tr_s
    edge = {} if tr_s is None else {"tr_s": tr_s}
    busy = Signal().carrier("states", causal=causal, **SCHEDULE, **edge)
    sig = Signal(seed=5, grid=G).carrier("dc", level=1.0).gate(by=busy)
    x, ev = sig.realize()
    assert sig.roles() == ["gate/1/by/states/0"]
    iv = S.intervals(ev, at="gate/1/by")
    t = np.arange(G.n) / G.fs
    checked = wrong = 0
    for r in iv:
        m = (t >= r["t_start_s"] + tr) & (t <= r["t_stop_s"] - tr)
        checked += int(m.sum())
        wrong += int(np.sum(x[m] != float(r["state"])))
    opened = np.array([e.t_s for e in ev if e.kind == "transition"])
    stated = np.array([r["t_start_s"] for r in iv[1:]])
    off = np.array([np.abs(stated - c).min() for c in opened])
    print(f"tr_s={tr_s} causal={causal}: {len(iv)} intervals, {checked} of {G.n} samples lie "
          f"more than tr_s inside one and {wrong} of them are wrong; the gate's {opened.size} "
          f"switching instants are within {off.max() / tr:.2f} tr_s of a stated boundary")
    assert checked > 0.9 * G.n and wrong == 0
    assert off.max() <= tr
    # a dwell shorter than the edge is a runt the gate does not see: fewer switchings, never more
    assert opened.size <= stated.size


def test_a_switch_selector_is_an_activity_schedule():
    idle = Signal().carrier("dc", level=0.0)
    data = Signal().carrier("nrz", n_ui=2048, seed=3)
    busy = Signal().carrier("states", dwell=[20e-6, 60e-6], tr_s=200e-9)
    sig = Signal(seed=2, grid=G).switch(select=busy, inputs=[idle, data])
    x, ev = sig.realize()
    payload = Signal(seed=2, grid=G).carrier("nrz", n_ui=2048, seed=3).waveform()
    iv = S.intervals(ev, at="switch/0/select")
    t = np.arange(G.n) / G.fs
    wrong = 0
    for r in iv:
        m = (t >= r["t_start_s"] + 200e-9) & (t <= r["t_stop_s"] - 200e-9)
        wrong += int(np.sum(x[m] != (payload[m] if r["state"] else 0.0)))
    changes = np.array([e.t_s for e in ev if e.kind == "switch"])
    stated = np.array([r["t_start_s"] for r in iv[1:]])
    print(f"switch: {len(iv)} intervals, {wrong} wrong samples; its changes are within "
          f"{np.abs(changes - stated).max() * G.fs:.3f} sample of the stated boundaries")
    assert wrong == 0 and changes.size == stated.size
    assert np.abs(changes - stated).max() < 0.5 / G.fs
    busy_frac = sum(r["t_stop_s"] - r["t_start_s"] for r in iv if r["state"]) / (G.n / G.fs)
    print(f"busy {busy_frac:.3f} of the record; long-run {S.occupancy([20e-6, 60e-6])[1]:.3f}")


def test_two_channels_share_one_schedule():
    rec = ws.Channels(
        {"a": Signal().carrier("dc", level=1.0).gate(by=Signal().use("busy")),
         "b": Signal().carrier("dc", level=2.0).gate(by=Signal().use("busy"))},
        shared={"busy": Signal().carrier("states", dwell=20e-6)}, seed=4, grid=G)
    w = rec.waveforms()
    assert rec.roles() == ["busy/states/0"]
    assert np.array_equal(w["a"] != 0, w["b"] != 0) and 0 < np.mean(w["a"] != 0) < 1


# ------------------------------------------------------------------ 1/f, as a composition
def test_flicker_is_a_visible_sub_chain_with_a_one_over_f_slope():
    g = Grid(fs=1e6, n=1 << 20)
    noise = ws.recipes.flicker(f_lo_hz=30.0, f_hi_hz=1e5, rms=0.1)
    assert {o["op"] for o in noise.ops} == {"carrier", "combine"}
    assert all(o["ops"][0]["kind"] == "states" for o in noise.ops[1]["others"])
    sig = Signal(seed=3, grid=g).carrier("dc", level=0.0).combine(noise)
    assert len(sig.roles()) == 1 + len(noise.ops[1]["others"]) == 8
    x = sig.waveform()
    f, pxx = sps.welch(x, fs=g.fs, nperseg=1 << 15)
    band = (f >= 300.0) & (f <= 1e4)                 # a decade inside each end of the corners
    slope = np.polyfit(np.log10(f[band]), np.log10(pxx[band]), 1)[0]
    want = ws.recipes.flicker_psd(f[band], 30.0, 1e5, rms=0.1)
    err_db = 10 * np.log10(pxx[band] / want)
    print(f"flicker 30 Hz..100 kHz, 8 processes: slope over 300 Hz..10 kHz {slope:.3f} "
          f"(the Lorentzian sum has "
          f"{np.polyfit(np.log10(f[band]), np.log10(want), 1)[0]:.3f}); PSD error mean "
          f"{err_db.mean():+.2f} dB, rms {err_db.std():.2f} dB; rms {x.std():.4f} (stated 0.1)")
    assert abs(slope + 1.0) < 0.1
    assert abs(err_db.mean()) < 0.5 and err_db.std() < 1.0
    assert abs(x.std() / 0.1 - 1) < 0.1
    # one process moves alone
    one = "combine/1/others/0/combine/1/others/3/states/0"
    assert one in sig.roles() and not np.array_equal(sig.contrast(one, seed=5), x)
    with pytest.raises(ValueError, match="f_lo_hz < f_hi_hz"):
        ws.recipes.flicker(1e5, 30.0)
    with pytest.raises(ValueError, match="at least one sample|of a sample"):
        Signal(seed=1, grid=g).carrier("dc", level=0.0).combine(
            ws.recipes.flicker(30.0, 1e6)).waveform()


# ------------------------------------------------------------------ refusals
def build(**kw):
    return Signal(seed=1, grid=G).carrier("states", **kw)


REFUSED = {
    "no dwell": (lambda: build(), "needs dwell="),
    "a callable": (lambda: build(dwell=lambda rng: 1e-6), "callable.*Tabulate"),
    "an unknown distribution": (lambda: build(dwell={"dist": "pareto", "alpha": 2.0}),
                                "one of .*exponential.*table"),
    "a key the distribution does not read": (
        lambda: build(dwell={"dist": "exponential", "mean_s": 1e-6, "sigma": 1.0}),
        "'sigma'.*nothing reads"),
    "a dwell of zero": (lambda: build(dwell=0.0), "positive"),
    "uniform with lo above hi": (
        lambda: build(dwell={"dist": "uniform", "lo_s": 2e-6, "hi_s": 1e-6}), "lo_s < hi_s"),
    "one level": (lambda: build(levels=[1.0], dwell=1e-6), "at least two"),
    "dwells that do not match the levels": (lambda: build(dwell=[1e-6, 2e-6, 3e-6]),
                                            "3 entries and there are 2 levels"),
    "a self-transition": (lambda: build(dwell=1e-6, transitions=[[0.5, 0.5], [1, 0]]),
                          r"transitions\[0\]\[0\].*longer dwell"),
    "a row that does not sum to 1": (
        lambda: build(levels=[0, 1, 2], dwell=1e-6,
                      transitions=[[0, 0.5, 0.4], [1, 0, 0], [1, 0, 0]]), "sums to 0.9"),
    "a start that is not a state": (lambda: build(dwell=1e-6, start=2), "start is"),
    "a parameter of another carrier": (lambda: build(dwell=1e-6, f_hz=1e6),
                                       "does not read f_hz=.*dwell=<seconds>"),
    "the issue's word for the edge": (lambda: build(dwell=1e-6, transition_s=1e-7),
                                      "tr_s="),
    "a dwell under one sample": (lambda: build(dwell=1e-9).waveform(),
                                 "0.1 of a sample.*raise Grid"),
    "no single stationary state": (
        lambda: build(levels=[0, 1, 2, 3], dwell=1e-6,
                      transitions=[[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]]
                      ).waveform(), "no single stationary state.*start="),
    "a state never entered": (
        lambda: build(levels=[0, 1, 2], dwell=1e-6,
                      transitions=[[0, 1, 0], [1, 0, 0], [0.5, 0.5, 0]]).waveform(),
        "state 2 is never entered"),
    "no grid": (lambda: Signal(seed=1).carrier("states", dwell=1e-6).waveform(), "Grid"),
    "a bare carrier spec": (
        lambda: Signal(seed=1, grid=G).carrier("nrz", n_ui=64).crosstalk(
            aggressor=dict(kind="states", dwell=1e-6), coupling=0.1).waveform(),
        r"bare carrier spec.*Signal\(\).carrier\('states'"),
    "a lead-in": (
        lambda: Signal(seed=1, grid=G, lead_in=True).carrier("states", dwell=1e-6).lossy(
            loss_db=3, loss_at_ghz=0.01).waveform(), "lead_in: a 'states' source"),
    "an I/Q grid": (
        lambda: Signal(seed=1, grid=Grid(fs=1e8, n=4096, domain="iq", fc_hz=1e9)).carrier(
            "states", dwell=1e-6), "I/Q grid"),
}


@pytest.mark.parametrize("name", sorted(REFUSED))
def test_what_it_cannot_do_honestly_is_refused_by_name(name):
    fn, says = REFUSED[name]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises((ValueError, TypeError), match=says) as e:
            fn()
    print(f"{name}: {str(e.value)[:150]}")


def test_a_state_never_entered_is_allowed_as_the_stated_start():
    """Power-up then steady state: state 2 is where the record begins and is never returned to."""
    sig = build(levels=[0.0, 1.0, 2.0], dwell=1e-6, start=2,
                transitions=[[0, 1, 0], [1, 0, 0], [0.5, 0.5, 0]])
    seen = [r["state"] for r in S.intervals(sig.event_list())]
    assert seen[0] == 2 and 2 not in seen[1:]


def test_a_toleranced_dwell_is_a_part_of_the_unit():
    sig = build(dwell={"dist": "exponential", "mean_s": ws.tol(1e-6, 10)})
    mean = sig.recipe()["ops"][0]["dwell"]["mean_s"]
    other = sig.unit(8).recipe()["ops"][0]["dwell"]["mean_s"]
    assert isinstance(mean, float) and 0.9e-6 <= mean <= 1.1e-6 and other != mean
