"""Phase 3, item 2: a pass-FET / analog switch. Mechanism, not a protocol: a gate-controlled
series element with an on-resistance and, when off, either high-Z or a body-diode clamp -- the
two properties the ratified plan named as the acceptance bar ("pass-FET off is high-Z / body
diode").
"""
from __future__ import annotations

import numpy as np
import pytest

from wfmsynth import physics as P
from wfmsynth.compose import Signal
from wfmsynth.grid import Grid

FS = 1e9
G = Grid(fs=FS, n=8192)


# --------------------------------------------------------------------------------- physics
def test_on_state_is_a_resistive_divider_against_the_load():
    n = 100
    x = np.full(n, 2.0)
    gate = np.ones(n, dtype=bool)
    rds, rload = 10.0, 990.0
    y = P.pass_fet(x, gate, rds_on_ohm=rds, r_load_ohm=rload)
    expect = 2.0 * rload / (rload + rds)
    assert np.allclose(y, expect)


def test_off_state_within_rails_is_high_z_reads_as_zero():
    n = 100
    x = np.full(n, 2.0)
    gate = np.zeros(n, dtype=bool)
    y = P.pass_fet(x, gate, rds_on_ohm=5.0, r_load_ohm=1e6, v_rail_hi=5.0, v_rail_lo=-5.0)
    assert np.allclose(y, 0.0)


def test_off_state_with_no_rails_given_is_high_z_regardless_of_input():
    n = 50
    x = np.full(n, 50.0)             # far outside any sane rail
    gate = np.zeros(n, dtype=bool)
    y = P.pass_fet(x, gate, rds_on_ohm=5.0, r_load_ohm=1e6)
    assert np.allclose(y, 0.0)


def test_body_diode_clamps_above_the_high_rail_when_off():
    n = 50
    x = np.full(n, 20.0)
    gate = np.zeros(n, dtype=bool)
    v_rail_hi, drop = 5.0, 0.6
    y = P.pass_fet(x, gate, rds_on_ohm=5.0, r_load_ohm=1e6, v_rail_hi=v_rail_hi,
                   diode_drop=drop, diode_on_ohm=1.0)
    expect = (v_rail_hi + drop) * (1e6 / (1e6 + 1.0))
    assert np.allclose(y, expect, rtol=1e-6)


def test_body_diode_clamps_below_the_low_rail_when_off():
    n = 50
    x = np.full(n, -20.0)
    gate = np.zeros(n, dtype=bool)
    v_rail_lo, drop = -5.0, 0.6
    y = P.pass_fet(x, gate, rds_on_ohm=5.0, r_load_ohm=1e6, v_rail_lo=v_rail_lo,
                   diode_drop=drop, diode_on_ohm=1.0)
    expect = (v_rail_lo - drop) * (1e6 / (1e6 + 1.0))
    assert np.allclose(y, expect, rtol=1e-6)


def test_body_diode_does_not_conduct_within_the_rails_while_off():
    n = 50
    x = np.full(n, 1.0)          # well inside +/-5 V + 0.6 V drop
    gate = np.zeros(n, dtype=bool)
    y = P.pass_fet(x, gate, rds_on_ohm=5.0, r_load_ohm=1e6, v_rail_hi=5.0, v_rail_lo=-5.0)
    assert np.allclose(y, 0.0)


def test_gate_toggling_switches_between_the_two_regimes_sample_by_sample():
    x = np.array([2.0, 20.0, 2.0, 20.0])
    gate = np.array([True, True, False, False])
    y = P.pass_fet(x, gate, rds_on_ohm=10.0, r_load_ohm=990.0, v_rail_hi=5.0, diode_drop=0.6,
                   diode_on_ohm=1.0)
    assert y[0] == pytest.approx(2.0 * 990.0 / 1000.0)
    assert y[1] == pytest.approx(20.0 * 990.0 / 1000.0)
    assert y[2] == pytest.approx(0.0)                              # off, within rail
    assert y[3] == pytest.approx(5.6 * 990.0 / (990.0 + 1.0), rel=1e-3)  # off, clamped


# --------------------------------------------------------------------------------- through the composer
def test_pass_fet_composes_through_a_recipe_and_round_trips():
    sig = (Signal(seed=1, grid=G).carrier("sine", f_hz=10e6, amp=8.0)
           .pass_fet(gate=dict(kind="square", f_hz=1e6, duty=0.5), vth=0.0,
                     rds_on_ohm=8.0, r_load_ohm=1e4, v_rail_hi=5.0, v_rail_lo=-5.0))
    x = sig.waveform()
    x2 = Signal.from_recipe(sig.recipe()).waveform()
    assert np.array_equal(x, x2)
    assert np.all(np.isfinite(x))


def test_pass_fet_off_regions_are_bounded_by_the_rails_plus_diode_drop():
    """The clamp only protects while OFF -- an ON switch just conducts, so the ON regions are
    free to carry the full drive amplitude (here deliberately above the rail, to make the OFF
    clamp's job non-trivial)."""
    gate = P.square(f_hz=1e6, duty=0.5, n=G.n, fs=G.fs) > 0.0
    sig = (Signal(seed=1, grid=G).carrier("sine", f_hz=10e6, amp=8.0)
           .pass_fet(gate=dict(kind="square", f_hz=1e6, duty=0.5), vth=0.0,
                     rds_on_ohm=8.0, r_load_ohm=1e4, v_rail_hi=5.0, v_rail_lo=-5.0,
                     diode_drop=0.6))
    x = sig.waveform()
    off = x[~gate]
    assert off.max() <= 5.6 + 1e-3
    assert off.min() >= -5.6 - 1e-3
    assert x.max() > 5.6                # the ON regions are NOT clamped


def test_pass_fet_emits_only_the_arguments_it_was_given():
    op = (Signal(seed=1, grid=G).carrier("sine")
          .pass_fet(gate=dict(kind="square", f_hz=1e6), rds_on_ohm=8.0)
          .recipe()["ops"][1])
    assert set(op) == {"op", "gate", "rds_on_ohm"}


def test_pass_fet_is_classified_for_the_lead_in():
    from wfmsynth import compose as C
    assert "pass_fet" in C._EXEC and "pass_fet" in C.OP_KIND
    assert (("pass_fet" in C._LEAD_LTI) + ("pass_fet" in C._LEAD_SKIP)
            + ("pass_fet" in C._LEAD_REJECT) + ("pass_fet" in C._LEAD_ANALYTIC)) == 1


@pytest.mark.parametrize("knob,base,alt", [
    ("rds_on_ohm", dict(gate=dict(kind="square", f_hz=1e6), r_load_ohm=1e4), 200.0),
    ("r_load_ohm", dict(gate=dict(kind="square", f_hz=1e6), rds_on_ohm=8.0), 100.0),
    ("v_rail_hi", dict(gate=dict(kind="square", f_hz=1e6), rds_on_ohm=8.0), 2.0),
    ("diode_drop", dict(gate=dict(kind="square", f_hz=1e6), rds_on_ohm=8.0, v_rail_hi=2.0), 1.5),
    ("vth", dict(gate=dict(kind="square", f_hz=1e6), rds_on_ohm=8.0), 0.5),
])
def test_every_pass_fet_knob_moves_the_output(knob, base, alt):
    def render(**kw):
        return Signal(seed=1, grid=G).carrier("sine", f_hz=10e6, amp=8.0).pass_fet(**kw).waveform()

    a = render(**base)
    b = render(**{**base, knob: alt})
    rel = float(np.sqrt(((a - b) ** 2).mean())) / (float(np.sqrt((a**2).mean())) + 1e-12)
    assert rel > 1e-4, f"pass_fet.{knob} did not move the output"
