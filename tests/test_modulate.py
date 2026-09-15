"""Phase 3's last piece: AM/ASK/OOK, FM/FSK, PM as one `modulate` op, the message a nested
carrier spec (the same shape `crosstalk`'s `aggressor` and `open_drain`'s `second` already
take -- analog or CMOS). New, Grid-native physics: `physics.am_modulate`/`fm_modulate`/
`pm_modulate` are NOT `physics.am`/`fm`, which the skill file requires stay bit-identical on
the legacy normalised ramp.
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy.signal import hilbert

from wfmsynth import physics as P
from wfmsynth.compose import Signal
from wfmsynth.grid import Grid

FS = 1e9
N = 1 << 16
G = Grid(fs=FS, n=N)


# --------------------------------------------------------------------------------- AM
def test_am_sidebands_sit_at_fc_plus_minus_fm():
    fc, fm = 20e6, 1e6
    y = (Signal(seed=1, grid=G).carrier("sine", f_hz=fc, amp=1.0)
         .modulate(kind="am", depth=0.5, message=dict(kind="sine", f_hz=fm)).waveform())
    spec = np.abs(np.fft.rfft(y))
    freqs = np.fft.rfftfreq(N, d=1.0 / FS)

    def bin_at(f):
        return spec[int(np.argmin(np.abs(freqs - f)))]

    carrier_p = bin_at(fc)
    upper = bin_at(fc + fm)
    lower = bin_at(fc - fm)
    floor = np.median(spec)
    assert upper > 20 * floor
    assert lower > 20 * floor
    assert carrier_p > upper


def test_am_depth_zero_leaves_the_carrier_unmodulated():
    fc = 20e6
    y = (Signal(seed=1, grid=G).carrier("sine", f_hz=fc, amp=1.0)
         .modulate(kind="am", depth=0.0, message=dict(kind="sine", f_hz=1e6)).waveform())
    x = P.sine(f_hz=fc, amp=1.0, n=N, fs=FS)
    assert np.allclose(y, x, atol=1e-9)


def test_ook_message_gates_the_carrier_fully_on_and_off():
    # OOK is suppressed-carrier AM with a UNIPOLAR (0/1) message: (1+depth*msg) never reaches
    # 0 for a message that only goes 0..1, but depth*msg*carrier does -- full suppression when
    # the message is 0, full amplitude when it is 1.
    fc = 20e6
    y = (Signal(seed=1, grid=G).carrier("sine", f_hz=fc, amp=1.0)
         .modulate(kind="ook", depth=1.0, suppressed=True,
                   message=dict(kind="cmos", v_lo=0.0, v_hi=1.0, duty=0.5, f_hz=1e6,
                               tr_s=1e-9)).waveform())
    n_win = int(FS / 1e6 / 4)                # a quarter-period window
    assert np.abs(y[:n_win]).max() < 1e-6 or np.abs(y[:n_win]).max() > 0.5  # not ambiguous
    # somewhere the carrier is fully suppressed and somewhere it is at full amplitude
    usable = (len(y) // n_win) * n_win
    windows = y[:usable].reshape(-1, n_win)
    peak = np.abs(windows).max(axis=1)
    assert peak.min() < 1e-2       # a band-limited cmos low rail carries a small residual ripple
    assert peak.max() > 0.9


def test_suppressed_carrier_am_has_no_energy_at_fc_itself():
    fc, fm = 20e6, 1e6
    y = (Signal(seed=1, grid=G).carrier("sine", f_hz=fc, amp=1.0)
         .modulate(kind="am", depth=1.0, suppressed=True,
                   message=dict(kind="sine", f_hz=fm)).waveform())
    freqs = np.fft.rfftfreq(N, d=1.0 / FS)
    spec = np.abs(np.fft.rfft(y))
    i_fc = int(np.argmin(np.abs(freqs - fc)))
    i_side = int(np.argmin(np.abs(freqs - (fc + fm))))
    assert spec[i_fc] < spec[i_side] / 10


# --------------------------------------------------------------------------------- FM
def test_fm_instantaneous_frequency_follows_the_message_in_hz():
    fc, fm_hz, dev_hz = 50e6, 1e6, 5e6
    y = (Signal(seed=1, grid=G).carrier("dc", level=0.0)
         .modulate(kind="fm", fc_hz=fc, dev_hz=dev_hz,
                   message=dict(kind="sine", f_hz=fm_hz)).waveform())
    phase = np.unwrap(np.angle(hilbert(y)))
    inst_f = np.diff(phase) / (2 * np.pi) * FS
    # at message zero-crossings (rising) the deviation should be ~0 -> inst_f ~ fc
    t = np.arange(N - 1) / FS
    msg = np.sin(2 * np.pi * fm_hz * t)
    near_zero = np.abs(msg) < 0.02
    mid = slice(N // 8, -N // 8)
    sel = near_zero[mid]
    assert np.median(inst_f[mid][sel]) == pytest.approx(fc, rel=0.02)
    # at message peaks (~+1) the deviation should be close to +dev_hz
    near_peak = msg > 0.98
    if near_peak[mid].any():
        assert np.median(inst_f[mid][near_peak[mid]]) == pytest.approx(fc + dev_hz, rel=0.1)


def test_fm_ignores_the_upstream_carrier_and_needs_fc_hz():
    with pytest.raises(ValueError, match="fc_hz"):
        (Signal(seed=1, grid=G).carrier("dc", level=0.0)
         .modulate(kind="fm", dev_hz=1e5, message=dict(kind="sine", f_hz=1e5)).waveform())


# --------------------------------------------------------------------------------- PM
def test_pm_phase_deviation_matches_dev_rad_at_the_message_peak():
    fc, fm_hz, dev_rad = 50e6, 1e6, 1.2
    y_mod = (Signal(seed=1, grid=G).carrier("dc", level=0.0)
             .modulate(kind="pm", fc_hz=fc, dev_rad=dev_rad,
                       message=dict(kind="sine", f_hz=fm_hz)).waveform())
    y_ref = (Signal(seed=1, grid=G).carrier("dc", level=0.0)
             .modulate(kind="pm", fc_hz=fc, dev_rad=0.0,
                       message=dict(kind="sine", f_hz=fm_hz)).waveform())
    phase_mod = np.unwrap(np.angle(hilbert(y_mod)))
    phase_ref = np.unwrap(np.angle(hilbert(y_ref)))
    dphi = phase_mod - phase_ref
    mid = slice(N // 8, -N // 8)
    assert np.abs(dphi[mid]).max() == pytest.approx(dev_rad, rel=0.1)


# --------------------------------------------------------------------------------- gates
def test_modulate_message_can_be_a_cmos_carrier_not_just_analog():
    y = (Signal(seed=1, grid=G).carrier("sine", f_hz=20e6, amp=1.0)
         .modulate(kind="am", depth=1.0,
                   message=dict(kind="cmos", v_lo=0.0, v_hi=1.0, duty=0.5, f_hz=1e6,
                               tr_s=1e-9)).waveform())
    assert y.size == N and np.all(np.isfinite(y))


def test_modulate_has_no_new_public_protocol_names():
    import wfmsynth as ws
    for bad in ("i2c", "uart", "iso7637", "ask_modulate", "fsk_modulate"):
        assert not hasattr(ws, bad)
        assert not hasattr(Signal, bad)


def test_modulate_composes_through_a_recipe_and_round_trips():
    sig = (Signal(seed=1, grid=G).carrier("sine", f_hz=20e6, amp=1.0)
           .modulate(kind="am", depth=0.4, message=dict(kind="sine", f_hz=1e6))
           .lossy(loss_db=3.0, loss_at_ghz=1.0))
    x = sig.waveform()
    x2 = Signal.from_recipe(sig.recipe()).waveform()
    assert np.array_equal(x, x2)


def test_modulate_emits_only_the_arguments_it_was_given():
    op = (Signal(seed=1, grid=G).carrier("sine", f_hz=20e6)
          .modulate(kind="am", depth=0.4, message=dict(kind="sine", f_hz=1e6))
          .recipe()["ops"][1])
    assert set(op) == {"op", "kind", "depth", "message"}


def test_modulate_is_classified_for_the_lead_in():
    from wfmsynth import compose as C
    assert "modulate" in C._EXEC and "modulate" in C.OP_KIND
    assert (("modulate" in C._LEAD_LTI) + ("modulate" in C._LEAD_SKIP)
            + ("modulate" in C._LEAD_REJECT) + ("modulate" in C._LEAD_ANALYTIC)) == 1


def test_unknown_modulate_kind_is_a_named_error():
    with pytest.raises(ValueError, match="kind"):
        (Signal(seed=1, grid=G).carrier("sine", f_hz=20e6)
         .modulate(kind="bogus", message=dict(kind="sine", f_hz=1e6)).waveform())


@pytest.mark.parametrize("knob,base,alt", [
    ("depth", dict(kind="am", message=dict(kind="sine", f_hz=1e6)), 0.9),
    ("message", dict(kind="am", depth=0.5), dict(kind="sine", f_hz=3e6)),
    ("suppressed", dict(kind="am", depth=0.5, message=dict(kind="sine", f_hz=1e6)), True),
])
def test_every_am_knob_moves_the_output(knob, base, alt):
    def render(**kw):
        return (Signal(seed=1, grid=G).carrier("sine", f_hz=20e6, amp=1.0)
                .modulate(**kw).waveform())

    base2 = {"message": dict(kind="sine", f_hz=1e6), **base}
    a = render(**base2)
    b = render(**{**base2, knob: alt})
    rel = float(np.sqrt(((a - b) ** 2).mean())) / (float(np.sqrt((a**2).mean())) + 1e-12)
    assert rel > 1e-4, f"modulate.{knob} did not move the output"
