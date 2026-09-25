"""Where `y` comes from.

`dataset()` returns `(X, recipes)` and deliberately no labels, because the sampled knob values
are already baked into the ops — so a label is READ BACK from the same artifact that reproduces
the record. That is the whole argument for not having a `label=` parameter, and it is only worth
anything if the route actually works and stays aligned.

Reviewing the library as an ML engineer would use it, this was the dealbreaker: the good route
existed and was documented nowhere, and the obvious alternative (appending to a list inside
`build`) is order-dependent and misaligns silently the day anything filters or reshards the set.
"""
from __future__ import annotations

import numpy as np
import pytest

import wfmsynth as ws
from wfmsynth.compose import Signal, op_params
from wfmsynth.grid import Grid

L = 1 << 13


def _grid():
    return Grid(fs=160e9, baud=10e9, n=L)


def _build(rng):
    """The documented shape: the sampled knob IS the label."""
    loss = float(rng.uniform(4.0, 16.0))
    s = int(rng.integers(1 << 30))
    return (Signal(seed=s, grid=_grid())
            .carrier("nrz", n_ui=L // 16, pattern="prbs13", tr_frac=0.5, causal=True, seed=s)
            .lossy(loss_db=loss, loss_at_ghz=12.5, causal=True))


def test_the_label_is_recoverable_from_the_recipe():
    X, recipes = ws.dataset(_build, 12, seed=0)
    y = np.array([op_params(r, "lossy")["loss_db"] for r in recipes])
    assert y.shape == (12,)
    assert np.all((y >= 4.0) & (y <= 16.0))
    assert len(np.unique(y)) == 12, "the knob was supposed to vary per record"


def test_the_label_is_aligned_with_its_row_by_construction():
    """The property a side-list cannot offer. Re-rendering row i from recipe i must reproduce
    row i — so the label read from recipe i belongs to X[i] and nothing else can drift."""
    X, recipes = ws.dataset(_build, 5, seed=0)
    for i, r in enumerate(recipes):
        again = Signal.from_recipe(r).waveform()[:X.shape[1]]
        assert np.allclose(again, X[i], atol=0), f"row {i} does not match its own recipe"


def test_the_label_survives_a_reshard():
    """The failure a side list has and this does not: filter or reorder the set and the labels
    still follow their records, because they are carried by them."""
    X, recipes = ws.dataset(_build, 10, seed=0)
    y = np.array([op_params(r, "lossy")["loss_db"] for r in recipes])
    keep = y > np.median(y)
    Xk, rk = X[keep], [r for r, k in zip(recipes, keep) if k]
    yk = np.array([op_params(r, "lossy")["loss_db"] for r in rk])
    assert np.array_equal(yk, y[keep])
    assert Xk.shape[0] == len(rk) == int(keep.sum())


def test_a_repeated_op_is_an_error_not_a_silent_first_match():
    """THE reason this is a function and not a list comprehension. A chain with a package loss
    and a board loss is ordinary; `[...][0]` would label every record from whichever came first
    and the labels would look perfectly reasonable."""
    g = _grid()
    sig = (Signal(seed=1, grid=g)
           .carrier("nrz", n_ui=L // 16, pattern="prbs13", tr_frac=0.5, causal=True)
           .lossy(loss_db=3.0, loss_at_ghz=12.5, causal=True)      # package
           .lossy(loss_db=11.0, loss_at_ghz=12.5, causal=True))    # board
    r = sig.recipe()
    with pytest.raises(ValueError, match="2 'lossy' ops"):
        op_params(r, "lossy")
    assert op_params(r, "lossy", which=0)["loss_db"] == 3.0
    assert op_params(r, "lossy", which=1)["loss_db"] == 11.0


def test_a_missing_op_raises_rather_than_returning_none():
    """A missing label should stop a dataset build, not become a null in a training set."""
    r = Signal(seed=1, grid=_grid()).carrier("nrz", n_ui=L // 16, tr_frac=0.5).recipe()
    with pytest.raises(KeyError, match="no 'lossy' op"):
        op_params(r, "lossy")
    with pytest.raises(KeyError, match="carrier"):       # names what IS there
        op_params(r, "lossy")


def test_a_measured_label_is_a_different_thing_from_a_requested_one():
    """The distinction `examples/ground_truth.py` exists for, pinned here because the docstring
    now sends people to it: the knob you asked for and the property the record actually has are
    not the same number, and which one you want depends on the task."""
    X, recipes = ws.dataset(_build, 6, seed=1)
    requested = np.array([op_params(r, "lossy")["loss_db"] for r in recipes])
    g = Grid(fs=160e9, baud=10e9, n=X.shape[1])
    measured = np.array([ws.eye_height(X[i], g, levels=2) for i in range(len(X))])
    assert np.corrcoef(requested, measured)[0, 1] < -0.5, "more loss should close the eye"
    assert not np.allclose(requested, measured), "they are different quantities"
