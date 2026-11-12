"""A `{"from": recipe}` sub-chain is a pinned record, and `unit()` does not move it.

That is the statement; `roles()` and `contrast()` have to agree with it. Before 0.46.0's
release `roles()` listed a pinned record's toleranced part (`crosstalk/1/aggressor/unit/1/
r_source`) and `contrast()` refused the same name, saying the record had no toleranced
parameter: the list offered a role that could not be used.
"""
import numpy as np
import pytest

import wfmsynth as ws
from wfmsynth import Grid, Signal

G = Grid(fs=64e9, baud=16e9, n=4096)
AT = "crosstalk/2/aggressor"


def _lane():
    return (Signal(seed=9, grid=G).carrier("nrz", n_ui=1024, jitter=dict(rj=0.4))
            .probe(r_source=ws.tol(50, 10), c_load_f=1e-12))


def _record(pinned):
    return (Signal(seed=1, grid=G).carrier("nrz", n_ui=1024)
            .probe(r_source=ws.tol(50, 10), c_load_f=1e-12)
            .crosstalk(aggressor={"from": pinned.recipe()}, coupling=0.2))


def test_unit_moves_the_records_own_part_and_not_the_pinned_one():
    lane = _lane()
    sig = _record(lane)
    other = sig.unit(3)
    assert other.ops[1]["r_source"] != sig.ops[1]["r_source"]              # its own part moved
    assert other.ops[2]["aggressor"] == sig.ops[2]["aggressor"]            # the recipe held
    assert np.array_equal(other.subchain(AT), sig.subchain(AT))            # ... and its samples
    assert np.array_equal(sig.subchain(AT), lane.waveform())               # pinned as rendered
    assert not np.array_equal(other.waveform(), sig.waveform())


def test_roles_lists_what_can_move_and_every_one_of_them_does():
    sig = _record(_lane())
    assert sig.roles() == [f"{AT}/jitter/0", "unit/1/r_source"]            # no pinned unit role
    x = sig.waveform()
    for role in sig.roles() + ["unit/*"]:
        assert not np.array_equal(x, sig.contrast(role, seed=5)), role
    assert f"({AT}/unit/1/r_source)" in sig.summary()                      # still shown, as data


def test_a_record_whose_only_toleranced_part_is_pinned_has_no_unit():
    sig = (Signal(seed=1, grid=G).carrier("nrz", n_ui=1024)
           .crosstalk(aggressor={"from": _lane().recipe()}, coupling=0.2))
    assert sig.roles() == ["crosstalk/1/aggressor/jitter/0"]
    with pytest.raises(ValueError, match="no toleranced parameter"):
        sig.unit(3)
    with pytest.raises(ValueError, match="no toleranced parameter"):
        sig.contrast("unit/*", seed=3)


def test_the_same_chain_as_a_sub_chain_is_part_of_the_record_and_moves():
    """The contrast that makes "pinned" mean something: written `Signal()`, the lane's part
    belongs to this record's unit."""
    lane = Signal().carrier("nrz", n_ui=1024).probe(r_source=ws.tol(50, 10), c_load_f=1e-12)
    sig = Signal(seed=1, grid=G).carrier("nrz", n_ui=1024).crosstalk(aggressor=lane, coupling=0.2)
    assert sig.roles() == ["crosstalk/1/aggressor/unit/1/r_source"]
    other = sig.unit(3)
    assert not np.array_equal(other.subchain("crosstalk/1/aggressor"),
                              sig.subchain("crosstalk/1/aggressor"))
