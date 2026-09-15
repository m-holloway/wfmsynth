"""Phase 2 of the analog-instrument extension (BACKLOG.md): the probe pack.

Every new knob defaults to leaving `probe()`'s existing output bit-identical (pinned in
tests/test_byte_identity.py['probe']) -- these tests calibrate the NEW physics against a known
answer once each knob is actually engaged.
"""
from __future__ import annotations

import numpy as np
import pytest

from wfmsynth import instrument as INST
from wfmsynth import physics as P
from wfmsynth.compose import Signal
from wfmsynth.grid import Grid

FS = 20e9
G = Grid(fs=FS, baud=1e9, n=1 << 14)
X = P.nrz(n_ui=64, n=1 << 14, seed=3, tr_frac=0.2)


def test_default_probe_output_is_unchanged_by_the_new_knobs():
    """The new keyword-only knobs must not move the existing default path at all."""
    a = INST.probe(X, G, c_load_f=1e-12, r_source=50.0)
    b = INST.probe(X, G, c_load_f=1e-12, r_source=50.0, compensate=1.0, l_gnd_h=0.0,
                   coupling="dc", r_term_ohm=None, overload_range=None)
    assert np.array_equal(a, b)


# --------------------------------------------------------------------------------- compensation
def test_correctly_compensated_probe_is_flat_the_identity():
    y = INST.probe(X, G, c_load_f=2e-12, r_source=1000.0, compensate=1.0)
    y_bare = INST.probe(X, G, c_load_f=2e-12, r_source=1000.0)
    assert np.array_equal(y, y_bare)


def _tr_1090(y, dt):
    y = np.asarray(y, float)
    lo, hi = y.min(), y.max()

    def cross(frac):
        lvl = lo + frac * (hi - lo)
        i = int(np.argmax(y > lvl))
        assert i > 0, "the record starts above the crossing level"
        return (i - 1) + (lvl - y[i - 1]) / (y[i] - y[i - 1])

    return (cross(0.9) - cross(0.1)) * dt


def test_undercompensated_probe_rounds_off_more_than_bare_loading():
    """Under-compensation (< 1) is EXTRA roll-off on top of the bare RC pole: a step comes out
    with a SLOWER delivered rise time, not faster (a comb-spectrum metric on a periodic square
    wave is the wrong instrument here -- see docs/DATASET-METHODOLOGY.md Sec 3.5 -- so this
    calibrates on an isolated step's 10-90 % rise time instead).

    No lead-in is used here (this calls `instrument.probe` directly, below `Signal`), so the
    record's first samples carry the same turn-on transient every LTI stage in this library has
    without one (docs/BACKLOG.md #52) -- the window below sits around the step, away from it."""
    n = 1 << 14
    step = P.step(amp=1.0, t_step_frac=0.5, tr_s=0.01e-9, n=n, fs=FS)
    under = INST.probe(step, G, c_load_f=5e-12, r_source=1000.0, compensate=0.3)
    bare = INST.probe(step, G, c_load_f=5e-12, r_source=1000.0)
    win = slice(n // 2 - 3000, n // 2 + 3000)
    assert _tr_1090(under[win], 1.0 / FS) > _tr_1090(bare[win], 1.0 / FS)


def test_overcompensated_probe_peaks_above_the_bare_response():
    edge = P.square(f_hz=50e6, tr_frac=0.01, n=1 << 14, fs=FS)
    over = INST.probe(edge, G, c_load_f=5e-12, r_source=1000.0, compensate=3.0)
    bare = INST.probe(edge, G, c_load_f=5e-12, r_source=1000.0)
    assert over.max() > bare.max() * 1.02


# --------------------------------------------------------------------------------- ground lead L
def test_ground_lead_inductance_rings_a_step():
    step = P.step(amp=1.0, t_step_frac=0.5, tr_s=0.05e-9, n=1 << 14, fs=FS)
    rung = INST.probe(step, G, c_load_f=5e-12, r_source=50.0, l_gnd_h=50e-9)
    bare = INST.probe(step, G, c_load_f=5e-12, r_source=50.0)
    # ringing shows up as excursion beyond the settled level, well beyond whatever small
    # residual ripple the (already band-limited) bare edge carries on its own
    settled_rung = rung[-100:].mean()
    settled_bare = bare[-100:].mean()
    overshoot_rung = rung.max() - settled_rung
    overshoot_bare = bare.max() - settled_bare
    assert overshoot_rung > 5 * overshoot_bare
    assert overshoot_rung > 0.02


def test_ground_lead_ring_frequency_matches_the_lc_resonance():
    L, C = 80e-9, 5e-12
    wn_expected = 1.0 / np.sqrt(L * C)
    step = P.step(amp=1.0, t_step_frac=0.1, tr_s=0.02e-9, n=1 << 15, fs=200e9)
    g = Grid(fs=200e9, n=1 << 15)
    y = INST.probe(step, g, c_load_f=C, r_source=10.0, l_gnd_h=L)
    ring = y[int(0.1 * (1 << 15)) :] - y[-100:].mean()
    spec = np.abs(np.fft.rfft(ring))
    f = np.fft.rfftfreq(len(ring), d=1.0 / 200e9)
    f_peak = f[np.argmax(spec[1:]) + 1]
    assert f_peak == pytest.approx(wn_expected / (2 * np.pi), rel=0.15)


# --------------------------------------------------------------------------------- termination
def test_fifty_ohm_termination_attenuates_more_than_high_z():
    hiz = INST.probe(X, G, c_load_f=1e-12, r_source=50.0, r_term_ohm=None)
    fifty = INST.probe(X, G, c_load_f=1e-12, r_source=50.0, r_term_ohm=50.0)
    assert np.sqrt((fifty**2).mean()) < np.sqrt((hiz**2).mean())


def test_termination_resistance_sets_the_expected_divider_gain():
    r_source, r_term = 450.0, 50.0
    expect_gain = r_term / (r_source + r_term)
    dc = P.dc(level=1.0, n=1024)
    g = Grid(fs=FS, n=1024)
    y = INST.probe(dc, g, c_load_f=1e-15, r_source=r_source, r_term_ohm=r_term)
    assert y[-1] == pytest.approx(expect_gain, rel=1e-3)


# --------------------------------------------------------------------------------- coupling
def test_ac_coupling_removes_the_dc_level():
    # the highpass corner needs several cycles inside the record to settle; 20 MHz over an
    # 819 ns record is ~16 cycles, comfortably enough (1 MHz would be under one cycle and is
    # a record-length problem, not an AC-coupling one -- see docs/DATASET-METHODOLOGY.md Sec 3.1
    # on sizing a grid/record from the effect's own time constant).
    x = P.square(f_hz=100e6, n=1 << 14, fs=FS) + 0.7
    y = INST.probe(x, G, c_load_f=1e-15, r_source=1.0, coupling="ac", ac_fc_hz=20e6)
    assert abs(y[1000:].mean()) < 0.05
    y_dc = INST.probe(x, G, c_load_f=1e-15, r_source=1.0, coupling="dc")
    assert y_dc[1000:].mean() > 0.5


# --------------------------------------------------------------------------------- overload recovery
def test_no_overload_recovery_by_default_even_past_a_stated_range():
    y = INST.probe(X * 10, G, c_load_f=1e-15, r_source=1.0)
    assert y.max() > 5.0            # nothing clips without the knob


def test_overload_clips_to_the_stated_range():
    # a SUSTAINED overload, not a fast-toggling one -- a real front end cannot exceed its own
    # rail while still being driven past it, whatever the recovery tail does afterward
    x = np.full(4000, 8.0)
    g = Grid(fs=FS, n=len(x))
    y = INST.probe(x, g, c_load_f=1e-15, r_source=1.0, overload_range=1.0, overload_tau_s=1e-9)
    assert y.max() <= 1.0 + 1e-9


def test_overload_recovery_settles_back_with_the_stated_time_constant():
    tau_s = 5e-9
    x = np.concatenate([np.full(2000, 5.0), np.zeros(4000)])   # a hard overload, then quiet
    g = Grid(fs=FS, n=len(x))
    y = INST.probe(x, g, c_load_f=1e-15, r_source=1.0, overload_range=1.0, overload_tau_s=tau_s)
    tail = y[2000:] - 0.0
    i_tau = int(round(tau_s * FS))
    # a first-order recovery is within 1/e of its initial excess after one tau
    assert abs(tail[i_tau]) < abs(tail[0]) * (1.0 / np.e + 0.15)
    assert abs(tail[10 * i_tau]) < 0.02


# --------------------------------------------------------------------------------- through the composer
def test_probe_pack_composes_through_a_recipe_and_round_trips():
    sig = (Signal(seed=1, grid=G).carrier("nrz", n_ui=64, tr_frac=0.2)
           .probe(c_load_f=2e-12, r_source=200.0, compensate=1.3, l_gnd_h=10e-9,
                  r_term_ohm=1e6, coupling="ac", ac_fc_hz=5e6,
                  overload_range=0.9, overload_tau_s=2e-9))
    x = sig.waveform()
    x2 = Signal.from_recipe(sig.recipe()).waveform()
    assert np.array_equal(x, x2)
    assert np.all(np.isfinite(x))


def test_probe_with_overload_recovery_is_excused_from_the_lti_lead_in_probe():
    """The lead-in extent probe must strip the nonlinear knob before measuring an impulse
    response, exactly as it already does for `noise_rms` -- otherwise sizing a guard on a
    chain using overload_range either crashes or silently mis-sizes."""
    sig = (Signal(seed=1, grid=Grid(fs=FS, baud=1e9, n=4096))
           .carrier("nrz", n_ui=64, tr_frac=0.2)
           .probe(c_load_f=1e-12, r_source=50.0, overload_range=0.5, overload_tau_s=1e-9)
           .with_lead_in(True))
    x = sig.waveform()          # must not raise
    assert x.size == 4096


@pytest.mark.parametrize("op,kw,knob,alt", [
    ("probe", dict(c_load_f=1e-12, r_source=200.0), "compensate", 2.0),
    ("probe", dict(c_load_f=5e-12, r_source=50.0), "l_gnd_h", 40e-9),
    ("probe", dict(c_load_f=1e-12, r_source=200.0), "r_term_ohm", 50.0),
    ("probe", dict(c_load_f=1e-15, r_source=1.0), "coupling", "ac"),
    ("probe", dict(c_load_f=1e-15, r_source=1.0, coupling="ac"), "ac_fc_hz", 100e6),
    ("probe", dict(c_load_f=1e-15, r_source=1.0), "overload_range", 0.3),
    ("probe", dict(c_load_f=1e-15, r_source=1.0, overload_range=0.3), "overload_tau_s", 5e-9),
])
def test_every_probe_pack_knob_moves_the_output(op, kw, knob, alt):
    a = INST.probe(X, G, **kw)
    b = INST.probe(X, G, **{**kw, knob: alt})
    rel = float(np.sqrt(((a - b) ** 2).mean())) / (float(np.sqrt((a**2).mean())) + 1e-12)
    assert rel > 1e-4, f"{knob} did not move probe()'s output"
