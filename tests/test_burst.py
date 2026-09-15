"""Phase 2's last piece: burst/idle as a proper op, promoting `impairments.burst_gate`
(docs/DATASET-METHODOLOGY.md's own rule: "if a consumer will ever want to select on it, make
it an op"). Composable onto ANY carrier -- analog or digital, `carrier` or `cmos` -- since it
is a gate applied to whatever samples arrive, not something that reads the carrier's kind.
"""
from __future__ import annotations

import numpy as np
import pytest

from wfmsynth import physics as P
from wfmsynth.compose import Signal
from wfmsynth.grid import Grid

FS = 1e6
G = Grid(fs=FS, n=100000)


def _render(**kw):
    return (Signal(seed=1, grid=G).carrier("sine", f_hz=50e3, amp=1.0)
            .burst(**kw).waveform())


def test_duty_cycle_matches_the_request():
    # a DC carrier isolates the gate's own duty cycle from a sinusoid's natural zero crossings.
    # A finer grid than the module's `G` so the on-interval (200 samples) is many samples
    # across the edge_frac=0.05 ramp (10 samples) -- at `G`'s coarser 20-samples-on the ramp
    # rounds to a single sample and reads as a floor/discretisation effect, not the duty cycle.
    g = Grid(fs=10e6, n=1_000_000)
    y = (Signal(seed=1, grid=g).carrier("dc", level=1.0)
         .burst(t_on_s=2e-5, t_off_s=8e-5, off="zero", edge_frac=0.05).waveform())
    on = np.abs(y) > 1e-6
    assert on.mean() == pytest.approx(0.2, abs=0.02)


def test_off_zero_is_actually_zero_in_the_gap_interior():
    y = _render(t_on_s=2e-5, t_off_s=8e-5, off="zero", edge_frac=0.02)
    # well inside a gap (skip the ramp regions near the edges)
    interior = y[int(3e-5 * FS) : int(9e-5 * FS)]
    assert np.allclose(interior, 0.0, atol=1e-9)


def test_off_hold_carries_the_last_on_value_through_the_gap():
    y = _render(t_on_s=2e-5, t_off_s=8e-5, off="hold", edge_frac=0.02)
    last_on = y[int(1.9e-5 * FS)]
    interior = y[int(3e-5 * FS) : int(9e-5 * FS)]
    assert np.allclose(interior, last_on, atol=1e-6)


def test_the_gate_edges_are_soft_not_a_hard_step():
    """A hard on/off step would itself read as a signal edge; `edge_frac` raised-cosines it,
    same as `impairments.burst_gate` already does."""
    y = _render(t_on_s=2e-5, t_off_s=8e-5, off="zero", edge_frac=0.1)
    # samples right at the nominal edge are neither fully on nor fully off
    i_edge = int(2e-5 * FS)
    ramp = y[i_edge - 3 : i_edge + 3]
    assert 0.0 < np.abs(ramp).max() < 1.0


def test_burst_is_identical_off_off_when_the_carrier_is_never_gated():
    """A degenerate all-on burst (t_off_s=0) must not move the samples at all."""
    x = Signal(seed=1, grid=G).carrier("sine", f_hz=50e3, amp=1.0).waveform()
    y = _render(t_on_s=1e-4, t_off_s=0.0, off="zero")
    assert np.allclose(x, y, atol=1e-9)


def test_burst_composes_on_a_unipolar_cmos_carrier_too():
    """Burst does not read the carrier's kind -- it gates whatever samples arrive, which is
    the whole point of it being one mechanism rather than a per-source knob."""
    y = (Signal(seed=1, grid=Grid(fs=FS, n=100000))
         .carrier("cmos", v_lo=0.0, v_hi=3.3, duty=0.5, f_hz=100e3, tr_s=50e-9)
         .burst(t_on_s=2e-5, t_off_s=8e-5, off="zero", edge_frac=0.02).waveform())
    interior = y[int(3e-5 * FS) : int(9e-5 * FS)]
    assert np.allclose(interior, 0.0, atol=1e-6)


def test_phase_s_shifts_where_the_first_gap_starts():
    a = _render(t_on_s=2e-5, t_off_s=8e-5, off="zero", edge_frac=0.02, phase_s=0.0)
    b = _render(t_on_s=2e-5, t_off_s=8e-5, off="zero", edge_frac=0.02, phase_s=3e-5)
    assert not np.allclose(a, b)


# --------------------------------------------------------------------------------- gates
def test_burst_emits_only_the_arguments_it_was_given():
    sig = Signal(seed=1, grid=G).carrier("sine").burst(t_on_s=1e-5, t_off_s=4e-5)
    op = sig.recipe()["ops"][1]
    assert set(op) == {"op", "t_on_s", "t_off_s"}


def test_burst_recipe_round_trips_bit_identical():
    sig = (Signal(seed=1, grid=G).carrier("sine", f_hz=50e3)
           .burst(t_on_s=2e-5, t_off_s=8e-5, off="hold").lossy(loss_db=3.0, loss_at_ghz=1e-3))
    x = sig.waveform()
    x2 = Signal.from_recipe(sig.recipe()).waveform()
    assert np.array_equal(x, x2)


def test_burst_is_classified_for_the_lead_in():
    from wfmsynth import compose as C
    assert "burst" in C._EXEC and "burst" in C.OP_KIND
    assert (("burst" in C._LEAD_LTI) + ("burst" in C._LEAD_SKIP)
            + ("burst" in C._LEAD_REJECT) + ("burst" in C._LEAD_ANALYTIC)) == 1


def test_burst_survives_a_lead_in_render():
    sig = (Signal(seed=1, grid=Grid(fs=FS, n=20000)).carrier("sine", f_hz=50e3)
           .burst(t_on_s=2e-5, t_off_s=8e-5).lossy(loss_db=3.0, loss_at_ghz=1e-3)
           .with_lead_in(True))
    x = sig.waveform()
    assert x.size == 20000 and np.all(np.isfinite(x))


def test_unknown_off_mode_is_a_named_error():
    with pytest.raises(ValueError, match="off"):
        _render(t_on_s=1e-5, t_off_s=1e-5, off="bogus")


@pytest.mark.parametrize("knob,base,alt", [
    ("t_on_s", dict(t_on_s=2e-5, t_off_s=8e-5), 5e-5),
    ("t_off_s", dict(t_on_s=2e-5, t_off_s=8e-5), 2e-5),
    ("off", dict(t_on_s=2e-5, t_off_s=8e-5), "hold"),
    ("edge_frac", dict(t_on_s=2e-5, t_off_s=8e-5), 0.3),
    ("phase_s", dict(t_on_s=2e-5, t_off_s=8e-5), 3e-5),
])
def test_every_burst_knob_moves_the_output(knob, base, alt):
    a = _render(**base)
    b = _render(**{**base, knob: alt})
    rel = float(np.sqrt(((a - b) ** 2).mean())) / (float(np.sqrt((a**2).mean())) + 1e-12)
    assert rel > 1e-4, f"burst.{knob} did not move the output"
