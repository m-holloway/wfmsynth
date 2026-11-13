"""A recipe written before an op's numerics changed says so on replay.

This library's central claim is that a record's ground truth is not its samples but the ordered
list of ops that produced them. That claim holds only while an op's numerics stay put, and once
they did not: 0.41.0 (#53) inverted `de_emphasis_taps`' sign convention, so `de_emphasis(db=3.5)`
rendered as de-emphasis under 0.40.0 and as PRE-emphasis afterwards. The recipe replayed without
complaint and the transmitter shaping was upside down.

`recipe()` had recorded `wfmsynth_version` all along, and `from_recipe()` never read it. These
tests exist because a warning nobody can trust is worse than none: the silent cases below matter
as much as the loud one.
"""
from __future__ import annotations

import warnings

import pytest

from wfmsynth import compose as C
from wfmsynth.compose import Signal
from wfmsynth.grid import Grid


@pytest.fixture(autouse=True)
def _the_build_that_carries_every_change(monkeypatch):
    """`wfmsynth.__version__` is read from installed metadata, which lags a version bump until
    the package is reinstalled; these tests are about recipes written before or after a change,
    so the running build is, here, the one that carries the newest entry."""
    newest = max(C.NUMERIC_CHANGES, key=lambda e: C._version_tuple(e[0]))[0]
    monkeypatch.setattr("wfmsynth.__version__", newest)


def _recipe(with_de_emphasis=True):
    s = Signal(seed=1, grid=Grid(fs=80e9, baud=10e9, n=2048)).carrier("nrz", n_ui=256)
    if with_de_emphasis:
        s = s.de_emphasis(db=-3.5)
    return s.recipe()


def _replay(r):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        Signal.from_recipe(r)
    return [str(c.message) for c in caught if "numerics changed" in str(c.message)]


def test_a_recipe_predating_the_change_warns_and_names_the_op():
    r = _recipe()
    r["wfmsynth_version"] = "0.40.0"
    msgs = [m for m in _replay(r) if "changed in 0.41.0" in m]   # 0.48.0 moved it again
    assert len(msgs) == 1
    assert "de_emphasis" in msgs[0]
    assert "0.40.0" in msgs[0] and "0.41.0" in msgs[0]
    assert "negate" in msgs[0].lower()          # says what to DO, not only what happened


def test_a_current_recipe_is_silent():
    """The bar that keeps the warning credible: the ordinary case must not fire."""
    assert _replay(_recipe()) == []


def test_a_recipe_predating_the_change_but_NOT_using_the_op_is_silent():
    r = _recipe(with_de_emphasis=False)
    r["wfmsynth_version"] = "0.40.0"
    assert _replay(r) == []


def test_a_recipe_with_no_version_field_is_silent():
    """Recipes predating the field are not evidence of anything. Warning on all of them would
    teach people to filter this category out, which costs more than it buys."""
    r = _recipe()
    r.pop("wfmsynth_version", None)
    assert _replay(r) == []


def test_an_unparseable_version_is_treated_as_unknown_not_as_old():
    r = _recipe()
    r["wfmsynth_version"] = "not-a-version"
    assert _replay(r) == []


@pytest.mark.parametrize("v,want", [
    ("0.41.0", (0, 41, 0)), ("0.40.0", (0, 40, 0)), ("1.2.3.4", (1, 2, 3)),
    ("0.41.0rc1", (0, 41, 0)), ("0.41", (0, 41)), ("", None), (None, None),
    ("garbage", None), ("0.0.0+unknown", (0, 0, 0)),
])
def test_version_parsing(v, want):
    assert C._version_tuple(v) == want


def test_an_older_recipe_still_replays_rather_than_being_refused():
    """A warning, not a refusal. The record may be exactly what the caller wants -- they may be
    reproducing the old behaviour deliberately -- so this reports and renders."""
    r = _recipe()
    r["wfmsynth_version"] = "0.40.0"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        x = Signal.from_recipe(r).waveform()
    assert x.shape == (2048,)


def test_the_table_is_well_formed():
    """Each entry drives a warning message, so a malformed one would produce nonsense at the
    moment someone most needs to trust it."""
    assert C.NUMERIC_CHANGES, "the table is empty; #53 should be its first entry"
    for changed_in, op, what in C.NUMERIC_CHANGES:
        assert C._version_tuple(changed_in) is not None, changed_in
        # `carrier/<kind>` names one kind of carrier, wherever in the recipe it sits
        assert op.split("/")[0] in C._EXEC, f"{op!r} is not an executable op"
        if "/" in op:
            assert op.split("/")[1], op          # `carrier/<kind>` or `<op>/<kind>`
        assert len(what) > 40 and what.strip().endswith("."), f"{op}: explain what changed"


def _square_recipe(version):
    g = Grid(fs=64e9, n=4096)
    r = Signal(seed=1, grid=g).carrier("sine", f_hz=1e9).pass_fet(
        gate=dict(f_hz=3e8), vth=0.0).recipe()
    r["wfmsynth_version"] = version
    return r


def test_a_change_to_one_kind_of_carrier_warns_only_recipes_holding_that_kind():
    """0.46.0 moved `square` and `cmos` edges. Every recipe has a carrier, so an entry keyed on
    the op would warn all of them; it is keyed on the kind, found wherever it sits -- here as
    the default kind of a `pass_fet` gate, two levels down and unnamed."""
    got = _replay(_square_recipe("0.45.0"))
    assert len(got) == 1 and "carrier/square" in got[0] and "0.46.0" in got[0]
    assert _replay(_square_recipe("0.46.0")) == []
    r = Signal(seed=1, grid=Grid(fs=64e9, baud=1e9, n=4096)).carrier("nrz", n_ui=64).recipe()
    r["wfmsynth_version"] = "0.45.0"
    assert _replay(r) == []
    r = Signal(seed=1, grid=Grid(fs=10e9, n=4096)).carrier(
        "cmos", v_lo=0.0, v_hi=3.3, f_hz=10e6, tr_s=2e-9).recipe()
    r["wfmsynth_version"] = "0.45.0"
    got = _replay(r)
    assert len(got) == 1 and "carrier/cmos" in got[0]


def _idle_recipe(version, **kw):
    r = Signal(seed=1, grid=Grid(fs=1e9, n=2048)).carrier("dc", level=3.3).lossy(
        loss_db=3, loss_at_ghz=0.1).recipe()
    r["wfmsynth_version"] = version
    return r


def test_a_record_that_starts_from_a_held_level_warns_a_pre_0_48_recipe_for_that_op_only():
    """0.48.0 (#100): `lossy`, and the other stages that began at zero, hold the first sample.
    An older recipe holding one is told; one holding none of them, or already current, is not."""
    got = [m for m in _replay(_idle_recipe("0.47.0")) if "changed in 0.48.0" in m]
    assert len(got) == 1 and "'lossy'" in got[0] and "false edge" in got[0]
    assert _replay(_idle_recipe("0.48.0")) == []
    r = Signal(seed=1, grid=Grid(fs=64e9, baud=1e9, n=4096)).carrier("nrz", n_ui=64).recipe()
    r["wfmsynth_version"] = "0.47.0"
    assert _replay(r) == []


def test_a_change_to_one_kind_of_an_op_warns_only_recipes_that_choose_it():
    """`crosstalk` is moved only for `kind="next"` (a delayed copy of the aggressor's level);
    the derivative form `fext`, the default, is not, so it must not be warned."""
    def recipe(kind):
        s = Signal(seed=1, grid=Grid(fs=80e9, baud=10e9, n=2048)).carrier("nrz", n_ui=256)
        r = s.crosstalk(kind=kind).recipe()
        r["wfmsynth_version"] = "0.47.0"
        return r
    assert len(_replay(recipe("next"))) == 1
    assert _replay(recipe("fext")) == []


def test_an_entry_newer_than_the_running_build_is_silent(monkeypatch):
    """A build older than a change cannot have rendered it differently, so it does not tell a
    recipe it predates that change: the entries for an unreleased version say nothing yet."""
    monkeypatch.setattr("wfmsynth.__version__", "0.47.0")
    assert _replay(_idle_recipe("0.46.0")) == []
    monkeypatch.setattr("wfmsynth.__version__", "0.48.0")
    assert len(_replay(_idle_recipe("0.46.0"))) == 1
