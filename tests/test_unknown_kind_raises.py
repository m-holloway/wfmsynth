"""GitHub #55: several `kind`/`shape`-style parameters were `if x == "a": ... else: <fallback>`,
so a typo silently selected the fallback branch instead of raising -- a wrong answer, not a
cosmetic one, since e.g. FEXT and NEXT are physically different couplings. The fix is the pattern
already used elsewhere in this library (`physics.carrier_symbols`'s unknown-pattern error):
validate against the allowed set and raise, naming the value and the choices.
"""
from __future__ import annotations

import numpy as np
import pytest

from wfmsynth import physics as P
from wfmsynth import impairments as IMP
from wfmsynth import instrument as INST
from wfmsynth.grid import Grid

B = P.nrz(n_ui=256, seed=1, n=4096, causal=True)
A = P.nrz(n_ui=256, seed=2, n=4096, causal=True)


def test_crosstalk_unknown_kind_raises_instead_of_becoming_next():
    with pytest.raises(ValueError, match="fext.*next|next.*fext"):
        P.crosstalk(B, A, coupling=0.2, kind="FEXT")


def test_crosstalk_valid_kinds_still_work():
    fext = P.crosstalk(B, A, coupling=0.2, kind="fext")
    next_ = P.crosstalk(B, A, coupling=0.2, kind="next")
    assert not np.allclose(fext, next_)


def test_crosstalk_sum_unknown_kind_raises():
    with pytest.raises(ValueError, match="fext.*next|next.*fext"):
        P.crosstalk_sum(B, [A], [0.2], kinds="bogus")


def test_crosstalk_matrix_unknown_kind_raises():
    g = Grid(fs=256e9, baud=28e9, n=4096)
    with pytest.raises(ValueError, match="fext.*next|next.*fext"):
        P.crosstalk_matrix(B, g, [0.2], kind="bogus")


def test_drift_unknown_shape_raises_instead_of_becoming_sine():
    with pytest.raises(ValueError, match="linear.*sine|sine.*linear"):
        IMP.drift(B, kind="gain", amount=0.1, shape="LINEAR")


def test_drift_valid_shapes_still_work():
    lin = IMP.drift(B, kind="gain", amount=0.2, shape="linear")
    sine = IMP.drift(B, kind="gain", amount=0.2, shape="sine")
    assert not np.allclose(lin, sine)


def test_scope_bandwidth_unknown_kind_already_raises():
    """Already fixed (not reproducible against the current source) -- pinned here so a future
    change cannot quietly reopen GitHub #55 for this call site."""
    g = Grid(fs=256e9, n=4096)
    x = np.random.default_rng(0).standard_normal(4096)
    with pytest.raises(ValueError, match="bessel"):
        INST.scope_bandwidth(x, g, 32e9, kind="BESSEL")
