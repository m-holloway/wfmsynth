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
    msgs = _replay(r)
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
        assert op in C._EXEC, f"{op!r} is not an executable op"
        assert len(what) > 40 and what.strip().endswith("."), f"{op}: explain what changed"
