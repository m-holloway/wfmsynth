"""Phase 1 of the analog-instrument extension (BACKLOG.md): new `carrier(kind=...)` sources.

Each test calibrates on a KNOWN answer rather than asserting "an array came back" (see
docs/DATASET-METHODOLOGY.md §3.5) -- a step's measured rise time, a chirp's instantaneous
frequency, a CMOS carrier's rail voltages, a band-limited noise carrier's occupied band.
"""
from __future__ import annotations

import numpy as np
import pytest

from wfmsynth import physics as P
from wfmsynth.compose import Signal
from wfmsynth.grid import Grid

FS = 256e9
BAUD = 16e9
G = Grid(fs=FS, baud=BAUD, n=1 << 14)


def _tr_1090(y, dt):
    """10-90 % rise time by interpolated threshold crossings (see test_edge_rise_time.py)."""
    y = np.asarray(y, float)
    lo, hi = y.min(), y.max()

    def cross(frac):
        lvl = lo + frac * (hi - lo)
        i = int(np.argmax(y > lvl))
        assert i > 0, "the record starts above the crossing level"
        return (i - 1) + (lvl - y[i - 1]) / (y[i] - y[i - 1])

    return (cross(0.9) - cross(0.1)) * dt


def _first_rising_window(y, half=400):
    """A window bracketing the first rising mid-crossing, for a periodic carrier whose phase
    at sample 0 is not known in advance (e.g. `cmos`, where the record can start high, low, or
    mid-edge depending on `phase_rad`)."""
    y = np.asarray(y, float)
    mid = 0.5 * (y.min() + y.max())
    above = y > mid
    idx = np.where(~above[:-1] & above[1:])[0]
    assert idx.size > 0, "no rising edge found in the record"
    c = int(idx[0]) + 1
    lo, hi = max(c - half, 0), min(c + half, len(y))
    return y[lo:hi]


# --------------------------------------------------------------------------------- step
def test_step_settles_to_the_requested_levels():
    y = P.step(amp=1.0, offset=0.0, t_step_frac=0.5, tr_s=1e-9, n=1 << 14, fs=FS)
    assert np.allclose(y[: (1 << 14) // 4], -1.0, atol=1e-6)
    assert np.allclose(y[3 * (1 << 14) // 4 :], 1.0, atol=1e-6)


def test_step_delivers_the_requested_rise_time():
    tr_s = 2e-9
    y = P.step(amp=1.0, t_step_frac=0.5, tr_s=tr_s, n=1 << 14, fs=FS)
    got = _tr_1090(y, 1.0 / FS)
    assert got == pytest.approx(tr_s, rel=0.05)


def test_step_moves_with_offset_and_amplitude():
    y = P.step(amp=2.0, offset=1.0, t_step_frac=0.5, tr_s=1e-9, n=1 << 12, fs=FS)
    assert y[: 1 << 10].mean() == pytest.approx(-1.0, abs=1e-3)
    assert y[-(1 << 10) :].mean() == pytest.approx(3.0, abs=1e-3)


# --------------------------------------------------------------------------------- pulse
def test_pulse_width_matches_the_request():
    width_s = 5e-9
    y = P.pulse(amp=1.0, offset=0.0, t_start_frac=0.2, width_s=width_s, tr_s=0.2e-9,
               n=1 << 14, fs=FS)
    above = y > 0.0
    measured = above.sum() / FS
    assert measured == pytest.approx(width_s, rel=0.05)


def test_pulse_is_flat_outside_its_window():
    y = P.pulse(amp=1.0, offset=0.0, t_start_frac=0.4, width_frac=0.1, tr_frac=0.005,
               n=1 << 14, fs=FS)
    assert np.allclose(y[: (1 << 14) // 10], -1.0, atol=1e-6)
    assert np.allclose(y[-(1 << 14) // 10 :], -1.0, atol=1e-6)


# --------------------------------------------------------------------------------- exp
def test_exp_rise_reaches_one_time_constant_at_1_minus_1_over_e():
    tau_s = 3e-9
    y = P.exp(amp=1.0, offset=0.0, t_start_frac=0.0, tau_s=tau_s, n=1 << 14, fs=FS)
    i_tau = int(round(tau_s * FS))
    assert y[i_tau] == pytest.approx(1.0 - np.exp(-1.0), abs=0.01)
    # 5 tau is 1 - e^-5 = 0.99326 settled, not 1.0 -- an exact single-pole answer, not a floor.
    assert y[5 * i_tau] == pytest.approx(1.0 - np.exp(-5.0), abs=1e-3)


def test_exp_decay_falls_from_the_target_toward_offset():
    tau_s = 2e-9
    y = P.exp(amp=1.0, offset=0.0, t_start_frac=0.0, tau_s=tau_s, decay=True, n=1 << 14, fs=FS)
    assert y[0] == pytest.approx(1.0, abs=1e-3)
    i_tau = int(round(tau_s * FS))
    assert y[i_tau] == pytest.approx(np.exp(-1.0), abs=0.01)


# --------------------------------------------------------------------------------- chirp (Hz)
def test_chirp_instantaneous_frequency_sweeps_in_hz_linearly():
    f0_hz, f1_hz = 1e9, 5e9
    n = 1 << 16
    y = P.chirp_sweep(f0_hz=f0_hz, f1_hz=f1_hz, n=n, fs=FS)
    # instantaneous frequency from the analytic signal's unwrapped phase derivative
    from scipy.signal import hilbert
    phase = np.unwrap(np.angle(hilbert(y)))
    inst_f = np.diff(phase) / (2 * np.pi) * FS
    t = np.arange(n - 1) / FS
    span = n / FS
    expect = f0_hz + (f1_hz - f0_hz) * t / span
    # edges of a Hilbert-derived instantaneous frequency are unreliable; check the interior
    mid = slice(n // 8, -n // 8)
    rel = np.abs(inst_f[mid] - expect[mid]) / expect[mid]
    assert np.median(rel) < 0.02


def test_chirp_is_on_the_grid_not_the_legacy_ramp():
    """`physics.chirp` (legacy, normalised) must stay untouched; the new sweep is a
    different function so it cannot silently move the pinned legacy behaviour."""
    assert P.chirp is not P.chirp_sweep


# --------------------------------------------------------------------------------- two_tone
def test_two_tone_spectrum_has_exactly_two_peaks_at_the_requested_frequencies():
    f1_hz, f2_hz = 2e9, 7e9
    n = 1 << 16
    y = P.two_tone(f1_hz=f1_hz, f2_hz=f2_hz, amp1=1.0, amp2=0.5, n=n, fs=FS)
    spec = np.abs(np.fft.rfft(y))
    freqs = np.fft.rfftfreq(n, d=1.0 / FS)
    i1 = int(np.argmin(np.abs(freqs - f1_hz)))
    i2 = int(np.argmin(np.abs(freqs - f2_hz)))
    top2 = np.argsort(spec)[-2:]
    assert set(top2) == {i1, i2}
    assert spec[i1] / spec[i2] == pytest.approx(2.0, rel=0.05)


# --------------------------------------------------------------------------------- noise
def test_band_noise_has_no_energy_outside_the_requested_band():
    lo_hz, hi_hz = 20e9, 40e9
    n = 1 << 15
    y = P.analog_noise(rms=0.3, band_lo_hz=lo_hz, band_hi_hz=hi_hz, n=n, fs=FS,
                       rng=np.random.default_rng(0))
    spec = np.abs(np.fft.rfft(y)) ** 2
    freqs = np.fft.rfftfreq(n, d=1.0 / FS)
    in_band = spec[(freqs >= lo_hz) & (freqs <= hi_hz)].sum()
    out_band = spec.sum() - in_band
    assert out_band / spec.sum() < 1e-6


def test_noise_rms_matches_the_request():
    y = P.analog_noise(rms=0.4, n=1 << 16, fs=FS, rng=np.random.default_rng(1))
    assert float(np.sqrt((y * y).mean())) == pytest.approx(0.4, rel=0.05)


def test_pink_noise_carries_more_low_frequency_power_than_white():
    n = 1 << 16
    white = P.analog_noise(rms=1.0, pink_frac=0.0, n=n, fs=FS, rng=np.random.default_rng(2))
    pink = P.analog_noise(rms=1.0, pink_frac=1.0, n=n, fs=FS, rng=np.random.default_rng(2))
    freqs = np.fft.rfftfreq(n, d=1.0 / FS)
    low = (freqs > 0) & (freqs < freqs.max() * 0.01)
    p_white = (np.abs(np.fft.rfft(white)) ** 2)[low].sum()
    p_pink = (np.abs(np.fft.rfft(pink)) ** 2)[low].sum()
    assert p_pink > 5 * p_white


# --------------------------------------------------------------------------------- cmos
def test_cmos_rails_sit_at_v_lo_and_v_hi_not_normalised_plusminus_one():
    y = P.cmos(v_lo=0.0, v_hi=3.3, duty=0.5, f_hz=100e6, tr_s=1e-9, n=1 << 14, fs=FS)
    assert y.min() == pytest.approx(0.0, abs=0.02)
    assert y.max() == pytest.approx(3.3, abs=0.02)


def test_cmos_zero_volts_low_rail_is_exactly_zero_not_stretched_to_minus_one():
    # a band-limited edge legitimately undershoots a fraction of the rail-to-rail span (the
    # same Bessel edge shaping `square` uses); the claim under test is "near 0 V", not "-1.8 V"
    # (which is what stretching the unipolar excursion onto +/-1 would have produced).
    y = P.cmos(v_lo=0.0, v_hi=1.8, duty=0.3, f_hz=50e6, tr_s=2e-9, n=1 << 14, fs=FS)
    lo_samples = y[y < 0.9]
    assert lo_samples.min() >= -0.05 * 1.8
    assert lo_samples.min() < 0.05


def test_cmos_duty_cycle_matches_the_request():
    n = 1 << 16
    y = P.cmos(v_lo=0.0, v_hi=3.3, duty=0.25, f_hz=50e6, tr_s=0.1e-9, n=n, fs=FS)
    mid = 0.5 * (y.min() + y.max())
    duty = float((y > mid).mean())
    assert duty == pytest.approx(0.25, abs=0.02)


def test_cmos_edge_time_is_absolute_seconds_not_a_fraction_of_the_period():
    """Unlike `square`'s tr_frac, `cmos`'s tr_s must not scale with frequency: a real gate's
    edge is fixed by its output stage, not by how fast it is being clocked."""
    tr_s = 2e-9
    y_slow = P.cmos(v_lo=0.0, v_hi=1.0, duty=0.5, f_hz=10e6, tr_s=tr_s, n=1 << 16, fs=FS)
    y_fast = P.cmos(v_lo=0.0, v_hi=1.0, duty=0.5, f_hz=100e6, tr_s=tr_s, n=1 << 16, fs=FS)
    got_slow = _tr_1090(_first_rising_window(y_slow), 1.0 / FS)
    got_fast = _tr_1090(_first_rising_window(y_fast), 1.0 / FS)
    assert got_slow == pytest.approx(tr_s, rel=0.1)
    assert got_fast == pytest.approx(tr_s, rel=0.1)


def test_cmos_composing_with_probe_loading_needs_no_new_op():
    """The 'Ron if it actually loads something' half of the CMOS request is already the
    existing `probe(r_source=..., c_load_f=...)` RC pole -- composed, not a new knob."""
    from wfmsynth import instrument as INST
    y = P.cmos(v_lo=0.0, v_hi=3.3, duty=0.5, f_hz=50e6, tr_s=0.2e-9, n=1 << 14, fs=FS)
    loaded = INST.probe(y, G, c_load_f=20e-12, r_source=1000.0)
    assert (_tr_1090(_first_rising_window(loaded), 1.0 / FS)
            > _tr_1090(_first_rising_window(y), 1.0 / FS))


# --------------------------------------------------------------------------------- through the composer
NEW_KINDS = ["step", "pulse", "exp", "chirp", "two_tone", "noise"]


@pytest.mark.parametrize("kind", NEW_KINDS)
def test_every_new_analog_kind_runs_through_a_recipe_with_an_impairment(kind):
    sig = Signal(seed=3, grid=G).carrier(kind).lossy(loss_db=6.0, loss_at_ghz=1.0)
    x = sig.waveform()
    assert x.size == G.n and np.all(np.isfinite(x))
    assert [o["op"] for o in sig.recipe()["ops"]] == ["carrier", "lossy"]


def test_cmos_runs_through_a_recipe_and_round_trips():
    sig = Signal(seed=3, grid=G).carrier("cmos", v_lo=0.0, v_hi=3.3, duty=0.4, f_hz=100e6,
                                         tr_s=1e-9).probe(c_load_f=1e-12)
    x = sig.waveform()
    r = sig.recipe()
    x2 = Signal.from_recipe(r).waveform()
    assert np.array_equal(x, x2)
    assert x.min() >= -0.05 * 3.3          # near 0 V, not stretched down toward -1 V


def test_every_new_kind_emits_only_the_arguments_it_was_given():
    """The recipe must not restate a default: only what the caller passed appears in the op."""
    sig = Signal(seed=1, grid=G).carrier("step", t_step_frac=0.5, tr_s=1e-9)
    op = sig.recipe()["ops"][0]
    assert set(op) == {"op", "kind", "t_step_frac", "tr_s"}


# --------------------------------------------------------------------------------- clamp warning
# Found by dogfooding the skill on a fresh agent (2026-09): step/pulse/cmos silently clamped an
# under-resolved edge to the 2-sample floor with no warning, unlike every other edge-shaping
# primitive in this library (`physics.resolve_rise_time`). `_resolve_abs_edge` is the same floor,
# made visible the same way.
@pytest.mark.parametrize("kind,kw", [
    ("step", dict(t_step_frac=0.5, tr_s=1e-12)),
    ("pulse", dict(t_start_frac=0.3, width_frac=0.2, tr_s=1e-12)),
    ("cmos", dict(v_hi=3.3, duty=0.5, f_hz=50e6, tr_s=1e-12)),
])
def test_an_under_resolved_edge_warns_the_same_way_resolve_rise_time_does(kind, kw):
    with pytest.warns(RuntimeWarning, match="edge time clamped"):
        Signal(seed=1, grid=G).carrier(kind, **kw).waveform()


def test_cmos_reference_example_does_not_silently_clamp():
    """The flagship cmos example this library ships (SKILL.md/REFERENCE.md/README.md) must
    itself respect the k >= 8 sampling rule it teaches -- shipping an example that silently
    violates the library's own headline rule is exactly the kind of silent mismatch this repo
    tests against everywhere else."""
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        g = Grid(fs=10e9, n=4096, v_full=3.3)
        Signal(seed=1, grid=g).carrier("cmos", v_lo=0.0, v_hi=3.3, duty=0.5, f_hz=10e6,
                                       tr_s=2e-9).waveform()


@pytest.mark.parametrize("kind,knob,base,alt", [
    ("step", "amp", dict(t_step_frac=0.5, tr_s=1e-9), 2.0),
    ("step", "t_step_frac", dict(amp=1.0, tr_s=1e-9), 0.2),
    ("pulse", "width_frac", dict(t_start_frac=0.3), 0.3),
    ("exp", "tau_frac", dict(t_start_frac=0.0), 0.4),
    ("chirp", "f1_hz", dict(f0_hz=1e9), 20e9),
    ("two_tone", "f2_hz", dict(f1_hz=1e9), 15e9),
    ("noise", "rms", dict(), 0.9),
    ("cmos", "duty", dict(v_hi=3.3, f_hz=50e6, tr_s=1e-9), 0.7),
    ("cmos", "v_hi", dict(duty=0.5, f_hz=50e6, tr_s=1e-9), 5.0),
])
def test_every_new_kind_knob_moves_the_output(kind, knob, base, alt):
    def render(**kw):
        return Signal(seed=1, grid=G).carrier(kind, **kw).waveform()

    a = render(**base)
    b = render(**{**base, knob: alt})
    rel = float(np.sqrt(((a - b) ** 2).mean())) / (float(np.sqrt((a ** 2).mean())) + 1e-12)
    assert rel > 1e-4, f"{kind}.{knob} did not move the output"
