"""Phase 2's other bus extension: a second sink on an open-drain line -- wired-AND resolved at
the ANALOG level (a resistor-divider mid-level when sinks with different on-resistance fight),
not the logic-level `bus.open_drain`/`combine_drivers` OR. `tests/test_open_drain_line.py`
covers the existing single-sink physics; these calibrate the new second-sink physics against
closed-form divider answers, and pin that the single-sink default is untouched.
"""
from __future__ import annotations

import numpy as np
import pytest

from wfmsynth import physics as P
from wfmsynth.compose import Signal
from wfmsynth.grid import Grid

FS = 1e9
RP = 1000.0
CB = 200e-12
VDD = 3.3


def _settle(sink_on, sink_b_on=None, r_sink_ohm=20.0, r_sink_b_ohm=None, n=20000):
    y = P.open_drain_line(sink_on, FS, RP, CB, v_dd=VDD, r_sink_ohm=r_sink_ohm,
                          sink_b_on=sink_b_on, r_sink_b_ohm=r_sink_b_ohm)
    return float(y[-1])            # steady state, held constant long enough to settle


def test_no_second_sink_is_bit_identical_to_the_original_signature():
    sink = np.zeros(4096, dtype=bool)
    sink[1000:] = True
    a = P.open_drain_line(sink, FS, RP, CB, v_dd=VDD, r_sink_ohm=25.0)
    b = P.open_drain_line(sink, FS, RP, CB, v_dd=VDD, r_sink_ohm=25.0, sink_b_on=None)
    assert np.array_equal(a, b)


def test_both_released_settles_to_v_dd():
    n = 20000
    off = np.zeros(n, dtype=bool)
    assert _settle(off, off, n=n) == pytest.approx(VDD, rel=1e-3)


def test_only_a_matches_the_single_sink_formula():
    n = 20000
    a_on = np.ones(n, dtype=bool)
    b_off = np.zeros(n, dtype=bool)
    r_a = 30.0
    got = _settle(a_on, b_off, r_sink_ohm=r_a, n=n)
    expect = VDD * r_a / (RP + r_a)
    assert got == pytest.approx(expect, rel=1e-3)


def test_only_b_uses_bs_own_on_resistance():
    n = 20000
    a_off = np.zeros(n, dtype=bool)
    b_on = np.ones(n, dtype=bool)
    r_a, r_b = 20.0, 80.0
    got = _settle(a_off, b_on, r_sink_ohm=r_a, r_sink_b_ohm=r_b, n=n)
    expect = VDD * r_b / (RP + r_b)
    assert got == pytest.approx(expect, rel=1e-3)


def test_wired_and_when_both_sink_uses_the_parallel_resistance():
    n = 20000
    both = np.ones(n, dtype=bool)
    r_a, r_b = 20.0, 80.0
    r_par = (r_a * r_b) / (r_a + r_b)
    got = _settle(both, both, r_sink_ohm=r_a, r_sink_b_ohm=r_b, n=n)
    expect = VDD * r_par / (RP + r_par)
    assert got == pytest.approx(expect, rel=1e-3)


def test_the_stronger_sink_dominates_the_combined_level():
    """A strong (low-Ron) sink pulls the wired-AND level much closer to its OWN level than to
    the weak sink's -- the physical content of 'wired-AND', not just a number in between."""
    n = 20000
    both = np.ones(n, dtype=bool)
    r_strong, r_weak = 5.0, 500.0
    combined = _settle(both, both, r_sink_ohm=r_strong, r_sink_b_ohm=r_weak, n=n)
    strong_alone = VDD * r_strong / (RP + r_strong)
    weak_alone = VDD * r_weak / (RP + r_weak)
    assert abs(combined - strong_alone) < abs(combined - weak_alone) / 10


def test_second_sink_needs_the_same_shape_as_the_first():
    with pytest.raises(ValueError, match="shape"):
        P.open_drain_line(np.zeros(100, bool), FS, RP, CB, sink_b_on=np.zeros(50, bool))


# --------------------------------------------------------------------------------- through the composer
def test_open_drain_second_sink_composes_through_a_recipe_and_round_trips():
    g = Grid(fs=FS, n=4096)
    sig = (Signal(seed=1, grid=g).carrier("nrz", n_ui=32, seed=3)
           .open_drain(r_pullup_ohm=RP, c_bus_f=CB, v_dd=VDD, r_sink_ohm=25.0,
                       second=dict(kind="nrz", n_ui=32, seed=9), r_sink_b_ohm=90.0))
    x = sig.waveform()
    x2 = Signal.from_recipe(sig.recipe()).waveform()
    assert np.array_equal(x, x2)
    assert np.all(np.isfinite(x))


def test_a_carrier_that_never_toggles_never_sinks_and_the_op_says_so():
    """Found by dogfooding the skill on a fresh agent (2026-09): `open_drain` thresholds the
    incoming carrier at the MIDPOINT of its own excursion, so a constant `dc` carrier -- any
    level, any sign -- straddles nothing and never sinks. The line then does exactly nothing
    for the whole record, which is the same silent-impairment failure mode
    docs/ARCHITECTURE.md's "impairments can raise" rule exists to catch, so the op warns."""
    g = Grid(fs=FS, n=4096)
    with pytest.warns(RuntimeWarning, match="never sink"):
        (Signal(seed=1, grid=g).carrier("dc", level=0.3)
         .open_drain(r_pullup_ohm=RP, c_bus_f=CB, v_dd=VDD).waveform())


def test_a_toggling_carrier_does_not_warn():
    import warnings
    g = Grid(fs=FS, n=4096)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        (Signal(seed=1, grid=g).carrier("nrz", n_ui=32, seed=3)
         .open_drain(r_pullup_ohm=RP, c_bus_f=CB, v_dd=VDD).waveform())
    assert not any("never sink" in str(w.message) for w in caught)


def test_only_the_second_sink_toggling_does_not_warn():
    """One device sinking and the other not is ordinary wired-AND, not degenerate -- only
    NEITHER device ever sinking is worth a warning."""
    import warnings
    g = Grid(fs=FS, n=4096)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        (Signal(seed=1, grid=g).carrier("dc", level=0.3)
         .open_drain(r_pullup_ohm=RP, c_bus_f=CB, v_dd=VDD,
                     second=dict(kind="nrz", n_ui=32, seed=9)).waveform())
    assert not any("never sink" in str(w.message) for w in caught)


def test_second_sink_knob_moves_the_output_through_the_composer():
    g = Grid(fs=FS, n=4096)

    def render(**kw):
        return (Signal(seed=1, grid=g).carrier("nrz", n_ui=32, seed=3)
                .open_drain(r_pullup_ohm=RP, c_bus_f=CB, v_dd=VDD, r_sink_ohm=25.0, **kw)
                .waveform())

    a = render()
    b = render(second=dict(kind="nrz", n_ui=32, seed=9), r_sink_b_ohm=90.0)
    rel = float(np.sqrt(((a - b) ** 2).mean())) / (float(np.sqrt((a**2).mean())) + 1e-12)
    assert rel > 1e-4
