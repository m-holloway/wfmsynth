"""GitHub #86: unit-to-unit variation -- parameters drawn from stated tolerances and corners.

A seed re-rolls noise and patterns; it does not make a second board. `wfmsynth.tolerance` lets a
numeric parameter be written as a tolerance or a corner set, draws it once per record under a
`unit/...` role, and writes the NUMBER into the recipe with what it was drawn from beside it.

What is established here, each against something independent of the code under test:

  * the resolved values FOLLOW THE STATED DISTRIBUTION over a population of records -- a
    Kolmogorov-Smirnov test against the stated CDF (chi-square for the discrete forms), each
    with a negative control so the test is known to be able to fail;
  * re-rolling `unit/*` changes ONLY the toleranced parameters, and leaves the noise where it
    was -- and re-rolling the noise leaves the unit where it was;
  * a recipe with resolved values replays BIT-IDENTICALLY and draws nothing in doing so;
  * the resolved number is the one the waveform was made with: a measured RC rise time equals
    ln(9) R C computed from the recipe;
  * a record that uses no tolerance is the same object it always was;
  * everything that cannot be drawn honestly is refused by name.

THRESHOLDS. The populations are seeds 0..N-1, so every statistic here is deterministic. A
distribution test passes at p > P_MIN = 1e-3 and its negative control must fail below 1e-9.
"""
from __future__ import annotations

import copy
import json

import numpy as np
import pytest
from scipy import stats

import wfmsynth as ws
from wfmsynth import tolerance as TOL
from wfmsynth.compose import Signal
from wfmsynth.grid import Grid
from wfmsynth.streams import Streams

G = Grid(fs=100e9, n=8000)             # 80 ns: two cycles of the 25 MHz source below
N_POP = 10000
P_MIN = 1e-3
R_NOM, R_PCT = 1e3, 20.0
C_NOM, C_PCT = 2e-12, 20.0


def rc_record(seed, noise_rms=0.0):
    """A CMOS edge into a toleranced RC. 10 time constants per half period, so the edge
    settles, and a 100 ps source edge against a 4.4 ns RC one, so the RC is what is measured."""
    extra = {"noise_rms": noise_rms} if noise_rms else {}
    return (Signal(seed=seed, grid=G)
            .carrier("cmos", v_lo=0.0, v_hi=3.3, f_hz=25e6, duty=0.5, tr_s=1e-10)
            .probe(r_source=ws.tol(R_NOM, R_PCT),
                   c_load_f=ws.tol(C_NOM, C_PCT, dist="normal"), **extra))


def rise_s(y, fs, lo=0.0, hi=3.3):
    """10-90 % rise time of the first rising edge, by linear interpolation of the crossings."""
    a, b = lo + 0.1 * (hi - lo), lo + 0.9 * (hi - lo)

    def ups(th):
        return np.flatnonzero((y[:-1] < th) & (y[1:] >= th))

    def at(k, th):
        return (k + (th - y[k]) / (y[k + 1] - y[k])) / fs

    i = ups(a)[0]
    j = ups(b)
    j = j[j > i][0]
    return at(j, b) - at(i, a)


# ---- the population -----------------------------------------------------------------------------
@pytest.fixture(scope="module")
def population():
    """N_POP records, four toleranced parameters each, read back from the recipes."""
    rows = []
    for seed in range(N_POP):
        s = Signal(seed=seed, grid=G).probe(
            r_source=ws.tol(1e3, 5),
            c_load_f=ws.tol(1e-11, 10, dist="normal"),
            atten=ws.tol(0.1, 2, dist="corner"),
            bw_hz=ws.corners({"slow": 1e9, "typ": 2e9, "fast": 3e9}))
        p = s.recipe()["ops"][0]
        rows.append((p["r_source"], p["c_load_f"], p["atten"], p["bw_hz"]))
    return np.array(rows)


def test_uniform_is_uniform_over_the_tolerance(population):
    r = population[:, 0]
    assert r.min() >= 950.0 and r.max() <= 1050.0
    u = (r - 950.0) / 100.0
    p = stats.kstest(u, "uniform").pvalue
    # the control: the same numbers are NOT a normal with tol = 3 sigma
    p_wrong = stats.kstest((r - 1000.0) / (50.0 / 3), stats.truncnorm(-3, 3).cdf).pvalue
    print(f"uniform: KS p = {p:.3f} against U[950, 1050]; against the normal form p = {p_wrong:.1e}")
    assert p > P_MIN
    assert p_wrong < 1e-9


def test_normal_means_tol_is_three_sigma_truncated(population):
    """The convention, measured: sigma = tol / 3, nothing outside +/- tol, and so a realised
    standard deviation of 0.9866 sigma (the truncation costs 1.3 %)."""
    c = population[:, 1]
    z = (c - 1e-11) / (1e-12 / TOL.NORMAL_SIGMAS)
    assert np.abs(z).max() <= TOL.NORMAL_SIGMAS
    want_sd = stats.truncnorm(-3, 3).std()
    assert want_sd == pytest.approx(0.98658, abs=1e-5)
    p = stats.kstest(z, stats.truncnorm(-3, 3).cdf).pvalue
    p_wrong = stats.kstest(z, stats.uniform(-3, 6).cdf).pvalue
    # the reading someone might assume instead: tol = 1 sigma. Ruled out by measurement.
    p_one_sigma = stats.kstest(z / 3.0, stats.truncnorm(-1, 1).cdf).pvalue
    print(f"normal: KS p = {p:.3f} against N(nom, tol/3) truncated at tol; sd = {z.std():.4f} "
          f"sigma (expected {want_sd:.4f}); max |z| = {np.abs(z).max():.3f}; "
          f"uniform p = {p_wrong:.1e}; tol = 1 sigma p = {p_one_sigma:.1e}")
    assert p > P_MIN
    assert z.std() == pytest.approx(want_sd, rel=0.03)      # the sd's own standard error is 0.7 %
    assert p_wrong < 1e-9 and p_one_sigma < 1e-9


def test_corner_is_one_of_the_two_extremes_equally(population):
    a = population[:, 2]
    vals, counts = np.unique(a, return_counts=True)
    assert vals == pytest.approx([0.098, 0.102], rel=1e-12)
    p = stats.chisquare(counts).pvalue
    print(f"corner: {dict(zip(vals.round(4).tolist(), counts.tolist()))}, chi-square p = {p:.3f}")
    assert p > P_MIN


def test_named_corners_are_equally_likely(population):
    vals, counts = np.unique(population[:, 3], return_counts=True)
    assert vals.tolist() == [1e9, 2e9, 3e9]
    p = stats.chisquare(counts).pvalue
    p_wrong = stats.chisquare(counts, f_exp=N_POP * np.array([0.25, 0.5, 0.25])).pvalue
    print(f"corners: {counts.tolist()}, chi-square p = {p:.3f}; against 1:2:1, p = {p_wrong:.1e}")
    assert p > P_MIN
    assert p_wrong < 1e-9


def test_parameters_in_one_record_are_independent(population):
    """Each parameter has its own role, so knowing one unit's resistor says nothing about its
    capacitor. |r| < 0.04 is 4 standard errors at N = 10000."""
    r = np.corrcoef(population[:, 0], population[:, 1])[0, 1]
    print(f"correlation between two ungrouped parameters over {N_POP} records: {r:+.4f}")
    assert abs(r) < 0.04


# ---- the number in the recipe is the number the waveform was made with -----------------------
def test_the_resolved_value_is_the_one_in_the_waveform():
    worst = 0.0
    seen = []
    for seed in range(8):
        s = rc_record(seed)
        p = ws.op_params(s.recipe(), "probe")
        want = np.log(9.0) * p["r_source"] * p["c_load_f"]
        got = rise_s(s.waveform(), G.fs)
        worst = max(worst, abs(got / want - 1.0))
        seen.append(got)
    print(f"RC rise time vs ln(9)*R*C from the recipe, 8 units: worst {worst:.1e}; "
          f"the units span {min(seen) * 1e9:.2f}..{max(seen) * 1e9:.2f} ns")
    assert worst < 1e-3
    assert max(seen) / min(seen) > 1.2          # and they really are different units


# ---- what the recipe carries -------------------------------------------------------------------
def test_the_recipe_holds_the_number_and_what_it_was_drawn_from():
    s = rc_record(3)
    p = s.recipe()["ops"][1]
    assert isinstance(p["r_source"], float) and isinstance(p["c_load_f"], float)
    assert abs(p["r_source"] - R_NOM) <= R_NOM * R_PCT / 100
    assert p["_unit"] == {
        "r_source": {"nom": R_NOM, "tol_pct": R_PCT, "seed": 3},
        "c_load_f": {"nom": C_NOM, "tol_pct": C_PCT, "dist": "normal", "seed": 3}}
    rows = TOL.resolved(s.recipe())
    assert [r["role"] for r in rows] == ["unit/1/r_source", "unit/1/c_load_f"]
    assert rows[0]["value"] == p["r_source"] and rows[0]["op"] == "probe"
    assert ws.unit_of(s.recipe()) == 3
    assert s.roles() == ["unit/1/r_source", "unit/1/c_load_f"]


def test_a_resolved_recipe_replays_bit_identically_and_draws_nothing(monkeypatch):
    s = rc_record(5, noise_rms=2e-3)
    y, digest = s.waveform(), s.sha256()
    recipe = json.loads(s.to_json())

    def no_draw(*a, **k):
        raise AssertionError("a resolved recipe must not draw")

    monkeypatch.setattr(TOL, "_u", no_draw)
    back = Signal.from_recipe(recipe)
    assert back.sha256() == digest
    assert np.array_equal(back.waveform(), y)


def test_a_recipe_written_with_tolerances_in_place_is_the_builder_record():
    s = rc_record(7)
    template = {"seed": 7, "grid": s.recipe()["grid"], "ops": [
        dict(s.ops[0]),
        {"op": "probe",
         "r_source": {"nom": R_NOM, "tol_pct": R_PCT},
         "c_load_f": {"nom": C_NOM, "tol_pct": C_PCT, "dist": "normal"}}]}
    before = copy.deepcopy(template)
    back = Signal.from_recipe(template)
    assert back.sha256() == s.sha256()
    assert template == before                    # the caller's template is not written into
    # and the same template under another seed is another unit
    other = Signal.from_recipe({**template, "seed": 8})
    assert other.ops[1]["r_source"] != s.ops[1]["r_source"]


def test_the_digest_covers_the_tolerance_as_well_as_the_value():
    s = rc_record(3)
    r = s.recipe()
    r2 = copy.deepcopy(r)
    r2["ops"][1]["_unit"]["r_source"]["tol_pct"] = 1.0
    a, b = Signal.from_recipe(r), Signal.from_recipe(r2)
    assert a.sha256() != b.sha256()
    assert np.array_equal(a.waveform(), b.waveform())      # same unit, a different population


# ---- re-rolling ---------------------------------------------------------------------------------
def _diff_paths(a, b, path=()):
    if isinstance(a, dict) and isinstance(b, dict) and a.keys() == b.keys():
        return [p for k in a for p in _diff_paths(a[k], b[k], path + (k,))]
    if isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        return [p for k in range(len(a)) for p in _diff_paths(a[k], b[k], path + (k,))]
    return [] if a == b else [path]


def test_rerolling_the_unit_changes_only_the_toleranced_parameters():
    s = rc_record(3, noise_rms=2e-3)
    u = s.unit(40)
    changed = _diff_paths(s.recipe(), u.recipe())
    assert sorted(changed) == sorted([
        ("ops", 1, "r_source"), ("ops", 1, "c_load_f"),
        ("ops", 1, "_unit", "r_source", "seed"), ("ops", 1, "_unit", "c_load_f", "seed")])
    assert ws.unit_of(u.recipe()) == 40 and u.seed == s.seed


def test_contrast_rerolls_the_unit_and_keeps_the_noise_or_the_reverse():
    """The noise is isolated by subtracting the same unit rendered without it; the probe adds
    it after the pole, so the difference IS the noise record, to float rounding."""
    s, clean = rc_record(3, noise_rms=2e-3), rc_record(3)
    y = s.waveform()
    noise = y - clean.waveform()
    assert noise.std() == pytest.approx(2e-3, rel=0.05)

    # the unit re-rolled: it is exactly the sibling unit, and it carries the SAME noise
    y_unit = s.contrast("unit/*", seed=40)
    assert np.array_equal(y_unit, s.unit(40).waveform())
    noise_unit = y_unit - clean.unit(40).waveform()
    d_unit = float(np.abs(y_unit - y).max())
    d_noise = float(np.abs(noise_unit - noise).max())
    assert d_unit > 0.02 and d_noise < 1e-12        # ten times the noise rms; float rounding

    # the noise re-rolled: the same unit (same rise time under the noise), other noise
    y_noise = s.contrast("probe/1")
    other = y_noise - clean.waveform()
    r = float(np.corrcoef(other, noise)[0, 1])
    print(f"unit re-rolled: waveform moves {d_unit:.3f} V, its noise by {d_noise:.1e} V; "
          f"noise re-rolled: correlation with the old noise {r:+.4f}, rms {other.std():.2e}")
    assert other.std() == pytest.approx(2e-3, rel=0.05) and abs(r) < 0.05

    # one role alone moves one parameter
    y_r = s.contrast("unit/1/r_source", seed=40)
    only_r = TOL.redrawn(s, Streams(s.seed).reroll("unit/1/r_source", seed=40))
    assert _diff_paths(s.recipe(), only_r.recipe()) == [
        ("ops", 1, "r_source"), ("ops", 1, "_unit", "r_source", "seed")]
    assert np.array_equal(y_r, only_r.waveform())
    with pytest.raises(ValueError, match="2 different seeds"):
        ws.unit_of(only_r.recipe())


def test_a_unit_role_in_streams_is_honoured_by_every_render_path():
    s = (Signal(seed=2, grid=Grid(fs=64e9, baud=16e9, n=4096), lead_in=True)
         .carrier("nrz", n_ui=64, pattern="prbs7", seed=3)
         .lossy(loss_db=ws.tol(8.0, 25), loss_at_ghz=8.0, causal=True)
         .digitize(snr_db=30))
    st = lambda: Streams(2).reroll("unit/*", seed=11)          # noqa: E731
    want = s.unit(11)
    assert not np.array_equal(want.waveform(), s.waveform())
    assert np.array_equal(s.waveform(streams=st()), want.waveform())
    assert np.array_equal(s.realize(streams=st())[0], want.waveform())
    ext, plan = s.rendered_lead(streams=st())
    assert np.array_equal(ext[plan.window], want.waveform())
    assert plan.summary() == want.lead_plan().summary()


# ---- groups, corners, nesting ------------------------------------------------------------------
def test_a_group_is_one_draw():
    """A process corner moves the edge AND the output resistance together."""
    names = []
    for seed in range(40):
        s = (Signal(seed=seed, grid=G)
             .carrier("cmos", v_lo=0.0, v_hi=3.3, f_hz=25e6,
                      tr_s=ws.corners({"fast": 0.8e-10, "typ": 1e-10, "slow": 1.3e-10},
                                      group="process"))
             .probe(r_source=ws.corners({"fast": 800.0, "typ": 1e3, "slow": 1300.0},
                                        group="process"),
                    c_load_f=ws.tol(2e-12, 10, group="process"),
                    atten=ws.tol(0.1, 10, dist="normal", group="process")))
        assert s.roles() == ["unit/process"]
        rows = {r["param"]: r for r in TOL.resolved(s.recipe())}
        assert rows["tr_s"]["corner"] == rows["r_source"]["corner"]
        names.append(rows["tr_s"]["corner"])
        # a toleranced pair in one group sits at the same quantile of each distribution
        u = (rows["c_load_f"]["value"] - 1.8e-12) / 0.4e-12
        z = (rows["atten"]["value"] - 0.1) / (0.01 / 3)
        assert stats.truncnorm(-3, 3).cdf(z) == pytest.approx(u, abs=1e-9)
        # and the corner index is that quantile's third, in sorted-name order
        assert names[-1] == sorted(["fast", "typ", "slow"])[int(u * 3)]
    assert set(names) == {"fast", "typ", "slow"}


def test_corner_selection_does_not_depend_on_dict_order_and_keeps_the_type():
    a = Signal(seed=9, grid=G).reflect(n_bounce=ws.corners({"few": 2, "many": 6}), td_ps=50.0)
    b = Signal(seed=9, grid=G).reflect(n_bounce=ws.corners({"many": 6, "few": 2}), td_ps=50.0)
    assert a.ops[0]["n_bounce"] == b.ops[0]["n_bounce"]
    assert type(a.ops[0]["n_bounce"]) is int
    # sort_keys in to_json reorders the corner dict; the replay must not move
    assert Signal.from_recipe(json.loads(a.to_json())).sha256() == a.sha256()
    pinned = Signal(seed=9, grid=G).reflect(
        n_bounce=ws.corners({"few": 2, "many": 6}, pick="many"), td_ps=50.0)
    assert pinned.ops[0]["n_bounce"] == 6
    assert pinned.unit(1).ops[0]["n_bounce"] == 6           # a stated corner is not drawn
    assert pinned.ops[0]["_unit"]["n_bounce"]["corner"] == "many"


def test_tol_abs_varies_a_parameter_whose_nominal_is_zero():
    v = np.array([Signal(seed=k, grid=G).intra_pair_skew(
        skew_ps=ws.tol(0.0, tol_abs=5.0)).ops[0]["skew_ps"] for k in range(2000)])
    assert v.min() >= -5.0 and v.max() <= 5.0
    assert stats.kstest((v + 5.0) / 10.0, "uniform").pvalue > P_MIN


def _net(seed):
    return Signal(seed=seed, grid=Grid(fs=20e9, n=4000)).net(
        nodes=["bus"], rails={"vdd": 3.3},
        elements=[{"id": "rp", "kind": "r", "a": "bus", "b": "vdd", "ohm": ws.tol(885.0, 5)},
                  {"id": "cb", "kind": "c", "a": "bus", "b": "gnd",
                   "farad": ws.tol(40e-12, 10, dist="normal")}],
        drivers=[{"id": "d0", "node": "bus", "volts": 0.0, "ohm": 20.0, "state": "on"}],
        schedule=[{"t_s": 20e-9, "id": "d0", "state": "hiz"}])


def test_a_tolerance_nested_in_a_net_element_is_resolved_and_addressed():
    """A pull-up and a bus capacitance are the resistor and capacitor people mean. The release
    is an RC charge toward the rail, so its 10-90 % time is ln(9) R C from the recipe."""
    s = _net(4)
    assert s.roles() == ["unit/0/elements.0.ohm", "unit/0/elements.1.farad"]
    el = s.recipe()["ops"][0]["elements"]
    r, c = el[0]["ohm"], el[1]["farad"]
    assert abs(r - 885.0) <= 885.0 * 0.05 and abs(c - 40e-12) <= 4e-12
    y = s.waveform()
    v0 = 3.3 * 20.0 / (20.0 + r)                    # the divider the driver holds before it lets go
    got = rise_s(y, 20e9, lo=v0, hi=3.3)
    assert got == pytest.approx(np.log(9.0) * r * c, rel=2e-3)
    assert np.array_equal(Signal.from_recipe(json.loads(s.to_json())).waveform(), y)
    only_c = TOL.redrawn(s, Streams(4).reroll("unit/0/elements.1.farad", seed=77))
    el2 = only_c.recipe()["ops"][0]["elements"]
    assert el2[0]["ohm"] == r and el2[1]["farad"] != c


# ---- nothing moves for a record that uses none ---------------------------------------------------
def test_a_record_without_tolerances_is_untouched():
    p = {"loss_db": 8.0, "loss_at_ghz": 8.0, "trend": (1.0, 2.0, 3.0),
         "jitter": {"rj": 0.01}, "taps": [0.1, 0.2]}
    assert TOL.resolve(3, p, 1) is p
    ops = [{"op": "lossy", **p}]
    assert TOL.resolve_ops(ops, 1)[0] is ops[0]
    assert TOL.roles(ops) == []
    s = (Signal(seed=1, grid=Grid(fs=64e9, baud=16e9, n=4096))
         .carrier("nrz", n_ui=64, jitter={"rj": 0.01}).lossy(loss_db=8.0, loss_at_ghz=8.0)
         .events("glitch", count=2).probe(noise_rms=1e-3).digitize(snr_db=30))
    assert s.roles() == ["jitter/0", "events/2", "probe/3", "noise/4"]
    assert all("_unit" not in o for o in s.recipe()["ops"])


# ---- the dataset helper ------------------------------------------------------------------------
def _capture(rng):
    return rc_record(int(rng.integers(1 << 30)), noise_rms=2e-3)


def test_unit_dataset_is_units_by_captures_and_splits_on_the_recipe():
    n_units, n_cap = 6, 4
    X, recipes = ws.unit_dataset(_capture, n_units, n_cap, seed=1)
    assert X.shape == (n_units * n_cap, G.n) and X.dtype == np.float32
    unit = np.array([ws.unit_of(r) for r in recipes])
    r_val = np.array([ws.op_params(r, "probe")["r_source"] for r in recipes])
    assert len(np.unique(unit)) == n_units and len(np.unique(r_val)) == n_units
    for u in range(n_units):
        rows = slice(u * n_cap, (u + 1) * n_cap)                 # unit-major
        assert len(set(unit[rows])) == 1 and len(set(r_val[rows])) == 1
        assert len({r["seed"] for r in recipes[rows]}) == n_cap  # and the captures differ
    # every row is its recipe's record
    for i in (0, 5, 23):
        assert np.array_equal(Signal.from_recipe(recipes[i]).waveform().astype(np.float32), X[i])
    # WITHIN a unit the measured rise time moves only by noise; BETWEEN units by the tolerance
    t = np.array([rise_s(x.astype(float), G.fs) for x in X]).reshape(n_units, n_cap)
    within, between = t.std(axis=1).mean(), t.mean(axis=1).std()
    print(f"rise time: within-unit sd {within * 1e12:.1f} ps, between-unit sd "
          f"{between * 1e12:.1f} ps ({between / within:.0f}x)")
    assert between > 10 * within

    # the units do not depend on how many captures each has, or on what a capture draws
    _, more = ws.unit_dataset(_capture, n_units, 2, seed=1)
    assert [ws.unit_of(r) for r in more[::2]] == unit[::n_cap].tolist()
    assert ([ws.op_params(r, "probe")["r_source"] for r in more[::2]]
            == r_val[::n_cap].tolist())
    _, again = ws.unit_dataset(_capture, n_units, n_cap, seed=1)
    assert again == recipes


def test_unit_dataset_refuses_what_would_not_be_a_unit_dataset():
    with pytest.raises(ValueError, match="identical to an earlier record"):
        ws.unit_dataset(lambda rng: rc_record(1), 2, 2)
    with pytest.raises(ValueError, match="no toleranced parameter"):
        ws.unit_dataset(lambda rng: Signal(seed=int(rng.integers(99)), grid=G)
                        .carrier("cmos", v_lo=0.0, v_hi=3.3, f_hz=25e6, tr_s=1e-10), 2, 2)
    with pytest.raises(ValueError, match="at least 1"):
        ws.unit_dataset(_capture, 0, 2)


# ---- refusals ------------------------------------------------------------------------------------
@pytest.mark.parametrize("spec, match", [
    ({"nom": "nrz", "tol_pct": 5}, "not a real number"),
    ({"nom": True, "tol_pct": 5}, "not a real number"),
    ({"nom": 1 + 2j, "tol_pct": 5}, "not a real number"),
    ({"nom": float("nan"), "tol_pct": 5}, "not a real number"),
    ({"nom": {"nom": 1.0, "tol_pct": 5}, "tol_pct": 5}, "inside a tolerance"),
    ({"nom": 50.0}, "exactly one of tol_pct"),
    ({"nom": 50.0, "tol_pct": 5, "tol_abs": 1.0}, "exactly one of tol_pct"),
    ({"nom": 50.0, "tol_pct": -1}, ">= 0"),
    ({"nom": 50.0, "tol_pct": 100}, "reaches zero or changes the sign"),
    ({"nom": 0.0, "tol_pct": 5}, "takes tol_abs"),
    ({"nom": 50.0, "tol_pct": 5, "dist": "gaussian"}, "not one of"),
    ({"nom": 50.0, "tol_pct": 5, "sigma": 3}, "does not take"),
    ({"nom": 50.0, "tol_pc": 5}, "did you mean 'tol_pct'"),
    ({"tol_pct": 5}, "needs 'nom'"),
    ({"corners": {}}, "non-empty"),
    ({"corners": {"a": "x"}}, "not real numbers"),
    ({"corners": {"a": 1.0}, "pick": "b"}, "not one of the corners"),
    ({"corners": {"a": 1.0}, "nom": 1.0}, "does not take"),
    ({"nom": 50.0, "tol_pct": 5, "group": "a/b"}, "plain name"),
    ({"nom": 50.0, "tol_pct": 5, "group": "*"}, "plain name"),
])
def test_a_tolerance_that_cannot_be_drawn_honestly_is_refused(spec, match):
    with pytest.raises(ValueError, match=match):
        Signal(seed=1, grid=G).probe(r_source=spec)
    with pytest.raises(ValueError, match=match):          # and the same on the recipe route
        Signal.from_recipe({"seed": 1, "ops": [{"op": "probe", "r_source": spec}]})


def test_the_refusal_names_the_op_and_the_parameter():
    with pytest.raises(ValueError, match=r"op 1, parameter 'elements\.0\.ohm'"):
        (Signal(seed=1, grid=G).carrier("cmos", v_lo=0.0, v_hi=3.3, f_hz=25e6, tr_s=1e-10)
         .net(nodes=["bus"], elements=[{"id": "r", "kind": "r", "a": "bus", "b": "gnd",
                                        "ohm": {"nom": "big", "tol_pct": 5}}]))


def test_helpers_check_what_they_build():
    assert ws.tol(50, 5) == {"nom": 50, "tol_pct": 5}
    assert ws.tol(50.0, 5, dist="normal", group="lot") == {
        "nom": 50.0, "tol_pct": 5, "dist": "normal", "group": "lot"}
    assert ws.corners({"a": 1.0}, pick="a") == {"corners": {"a": 1.0}, "pick": "a"}
    with pytest.raises(ValueError, match="not a real number"):
        ws.tol("fifty", 5)


def test_a_signal_used_inside_another_record_hands_its_tolerances_to_that_record():
    """A Signal nested as a net driver's source or as a sub-chain is part of the outer unit:
    its tolerance is drawn under the OUTER seed and role, and `unit()` moves it."""
    inner = Signal(grid=G).carrier(
        "cmos", v_lo=0.0, v_hi=3.3, f_hz=25e6, tr_s=ws.tol(1e-10, 10))
    net = Signal(seed=1, grid=G).net(
        nodes=["bus"],
        elements=[{"id": "c", "kind": "c", "a": "bus", "b": "gnd", "farad": 1e-12}],
        drivers=[{"id": "d", "node": "bus", "ohm": 20.0, "state": "on",
                  "source": {"ops": inner, "volts": True}}])
    assert net.roles()[-1] == "net/0/d/unit/0/tr_s"
    assert "_unit" not in net.ops[0]["drivers"][0]["source"]["ops"][0]

    victim = Signal(seed=1, grid=G).carrier("cmos", v_lo=0.0, v_hi=3.3, f_hz=10e6, tr_s=1e-10)
    s = victim.crosstalk(aggressor=inner, coupling=0.1)
    role = "crosstalk/1/aggressor/unit/0/tr_s"
    assert s.roles() == [role]
    tr = s.ops[1]["aggressor"]["ops"][0]["tr_s"]
    assert tr == TOL.quantile(ws.tol(1e-10, 10), TOL._u(1, role))[0]      # the outer seed and role
    assert tr != inner.ops[0]["tr_s"]                                     # not the inner draw
    # the drawn edge is the one in the coupled waveform: the sub-chain as rendered inside the
    # record is the cmos carrier with exactly that rise time
    want = Signal(grid=G).carrier("cmos", v_lo=0.0, v_hi=3.3, f_hz=25e6, tr_s=tr).waveform()
    assert np.array_equal(s.subchain("crosstalk/1/aggressor"), want)
    other = s.unit(9)
    assert other.ops[1]["aggressor"]["ops"][0]["tr_s"] != tr
    assert np.array_equal(s.contrast("unit/*", seed=9), other.waveform())
    assert np.array_equal(Signal.from_recipe(json.loads(s.to_json())).waveform(), s.waveform())


def test_a_pinned_record_keeps_its_own_unit_and_a_hand_written_resolved_chain_is_refused():
    lane = Signal(seed=5, grid=G).carrier(
        "cmos", v_lo=0.0, v_hi=3.3, f_hz=25e6, tr_s=ws.tol(1e-10, 10))
    victim = Signal(seed=1, grid=G).carrier("cmos", v_lo=0.0, v_hi=3.3, f_hz=10e6, tr_s=1e-10)
    s = victim.crosstalk(aggressor={"from": lane}, coupling=0.1)
    assert s.ops[1]["aggressor"]["from"]["ops"] == lane.ops        # embedded as recorded
    assert np.array_equal(s.subchain("crosstalk/1/aggressor"), lane.waveform())
    assert "_unit" not in s.ops[1]                                 # nothing drawn by the outer
    with pytest.raises(ValueError, match="arrives with its own resolved tolerances"):
        Signal(seed=1, grid=G).carrier(
            "cmos", v_lo=0.0, v_hi=3.3, f_hz=10e6, tr_s=1e-10).crosstalk(
            aggressor={"ops": [dict(o) for o in lane.ops]}, coupling=0.1)


def test_a_tolerance_under_a_key_a_role_cannot_address_is_refused():
    with pytest.raises(ValueError, match="cannot be addressed"):
        TOL.resolve(0, {"agents": {"a.b": {"gain": {"nom": 1.0, "tol_pct": 5}}}}, 1)


def test_rerolling_a_unit_that_is_not_there_is_refused():
    plain = Signal(seed=1, grid=G).carrier("cmos", v_lo=0.0, v_hi=3.3, f_hz=25e6, tr_s=1e-10)
    with pytest.raises(ValueError, match="no toleranced parameter"):
        plain.unit(2)
    with pytest.raises(ValueError, match="no toleranced parameter"):
        plain.contrast("unit/*")
    with pytest.raises(ValueError, match="no toleranced parameter"):
        ws.unit_of(plain.recipe())
    s = rc_record(3)
    with pytest.raises(ValueError, match=r"no unit role \['unit/1/r_sourc'\]"):
        s.contrast("unit/1/r_sourc")
    # "the unit it already is" is not another unit
    with pytest.raises(ValueError, match="seed they already have"):
        s.unit(4).contrast("unit/*")                 # contrast's default seed is s.seed + 1 = 4
    assert s.unit(3).sha256() == s.sha256()          # unit() is a statement, so it may repeat


def test_a_tolerance_inside_a_drivers_own_chain_belongs_to_the_outer_record():
    """A driver's source chain written as an op list is data in the net op, so a tolerance in
    it is resolved, addressed and re-rolled as a parameter of that op -- and the nested chain
    is then run through a prefixed view of the record's streams, which must not trip on it."""
    g = Grid(fs=20e9, n=4000)

    def build(seed):
        return Signal(seed=seed, grid=g).net(
            nodes=["bus"],
            elements=[{"id": "c", "kind": "c", "a": "bus", "b": "gnd", "farad": 10e-12}],
            drivers=[{"id": "d", "node": "bus", "ohm": 50.0, "state": "on",
                      "source": {"volts": True, "ops": [
                          {"op": "carrier", "kind": "cmos", "v_lo": 0.0, "f_hz": 25e6,
                           "tr_s": 1e-9, "v_hi": {"nom": 3.3, "tol_pct": 10}}]}}])

    s = build(6)
    role = "net/0/d/unit/0/v_hi"
    assert s.roles() == [role]
    v_hi = s.ops[0]["drivers"][0]["source"]["ops"][0]["v_hi"]
    y = s.waveform()
    high = lambda w: float(np.median(w[w > 0.5 * w.max()]))      # noqa: E731  the settled level
    assert high(y) == pytest.approx(v_hi, rel=1e-4)          # the bus settles to this unit's rail
    y2 = s.contrast(role, seed=50)
    v2 = s.unit(50).ops[0]["drivers"][0]["source"]["ops"][0]["v_hi"]
    assert v2 != v_hi and high(y2) == pytest.approx(v2, rel=1e-4)
