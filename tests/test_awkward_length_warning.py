"""A circular stage transforms at the RECORD's own length and cannot pick a friendlier one,
so the record length itself decides the price -- and an FFT of a length with a large prime
factor is not a little slower, it is up to 12x slower.

`physics.warn_if_awkward_length` names that instead of silently charging for it. The two
properties worth gating are opposite ones: it has to fire when it matters, and it has to stay
SILENT for every length anyone actually picks, or it becomes noise people filter out.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

from wfmsynth import physics as P
from wfmsynth import instrument as INST
from wfmsynth.grid import Grid


# every round length a caller plausibly chooses, including a deep capture's 40 M = 2**9 * 5**7
@pytest.mark.parametrize("n", [1 << 16, 1 << 20, 1 << 22, 100000, 1000000, 2500000,
                               40000000, 8192000, 2000000])
def test_it_is_silent_for_lengths_people_actually_pick(n):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        P.warn_if_awkward_length(n, "test")
    assert not w, f"{n} warned; leftover={P._small_radix_leftover(n)}"


@pytest.mark.parametrize("n", [1999993, 2097143, 2097153, 1999992])
def test_it_fires_for_a_length_with_a_large_prime_factor(n):
    with pytest.warns(RuntimeWarning, match="large prime factor"):
        P.warn_if_awkward_length(n, "test")


def test_it_names_a_better_length_that_is_actually_better():
    """Advice has to be actionable, so the length it suggests must itself be friendly."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        P.warn_if_awkward_length(2097143, "test")
    msg = str(w[0].message)
    better = int(msg.split("the next one is ")[1].split()[0])
    assert better >= 2097143
    assert P._small_radix_leftover(better) == 1


def test_a_short_record_does_not_warn_however_awkward():
    """Under the threshold the whole transform is cheap, so the advice would cost more
    attention than it saves."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        P.warn_if_awkward_length(9973, "test")      # prime, but small
    assert not w


def test_the_circular_stages_reach_it():
    """The point of the helper is the call sites, so check one end to end rather than trusting
    that it was wired in."""
    n = 1999993
    g = Grid(fs=80e9, baud=10e9, n=n)
    x = np.zeros(n)
    with pytest.warns(RuntimeWarning, match="probe_loading"):
        INST.probe_loading(x, g, c_load_f=0.5e-12, r_source=50.0)
    with pytest.warns(RuntimeWarning, match="brickwall"):
        INST.scope_bandwidth(x, g, 25e9, kind="brickwall")


def test_a_linear_stage_never_warns_because_it_picks_its_own_length():
    """`apply_transfer`'s default path pads to a 5-smooth length, so the record's own factors
    do not reach the transform at all. Warning there would be advice about nothing."""
    n = 1999993
    x = np.zeros(n)

    def mk(nfft):
        return np.ones(nfft // 2 + 1, complex)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        P.apply_transfer(x, mk, linear=True)
    assert not [c for c in w if "large prime factor" in str(c.message)]
