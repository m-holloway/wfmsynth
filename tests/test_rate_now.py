"""The general stages are stated in Hz and seconds, so they run at the rate the record is at
WHEN THEY RUN -- after an `acquire` or a decimating `downconvert`, not the synthesis grid's.
Every assertion is the effect, against the stated number."""
import numpy as np
import pytest

import wfmsynth as ws
from wfmsynth import Signal

FS_SYN, FS_ACQ = 80e9, 20e9
PROFILE = ws.AcquisitionProfile(sample_rate_hz=FS_ACQ, record_length=4096)


def acquired(f_hz):
    return (Signal(seed=1, grid=ws.Grid(fs=FS_SYN, n=1 << 15))
            .carrier("sine", f_hz=f_hz).acquire(PROFILE))


def gain_db(tone_hz, **kw):
    x = acquired(tone_hz).waveform()
    y = acquired(tone_hz).filter("butter", zero_phase=True, **kw).waveform()
    return 20 * np.log10(y[500:-500].std() / x[500:-500].std())


def test_filter_corner_after_acquire_is_the_stated_hz():
    # zero-phase keeps the stated magnitude: -3.01 dB at the corner, -10 log10(1 + (f/fc)^2)
    # elsewhere (the bilinear warp is under 0.05 dB this far below the 10 GHz Nyquist)
    at_corner, below = gain_db(2e9, f_hz=2e9, order=1), gain_db(0.5e9, f_hz=2e9, order=1)
    print(f"gain at the 2 GHz corner {at_corner:.3f} dB (-3.01), at 0.5 GHz {below:.3f} dB (-0.26)")
    assert abs(at_corner - -3.01) < 0.05
    assert abs(below - -10 * np.log10(1 + 0.25 ** 2)) < 0.05


def test_delay_after_acquire_is_the_stated_seconds():
    x = acquired(0.3e9).waveform()
    y = acquired(0.3e9).delay(by=0.5e-9).waveform()      # 10 samples at 20 GSa/s, 40 at 80
    assert np.array_equal(y[100:-100], x[90:-110])


def test_comparator_time_after_acquire_is_on_the_stored_clock():
    s = acquired(0.3e9).law("comparator", threshold=0.0)
    x, ev = s.realize()
    first = [e for e in ev if e.kind == "transition"]
    assert first
    for e in first:                                      # the crossing lies in the sample before
        assert (e.sample - 1) / FS_ACQ <= e.t_s <= e.sample / FS_ACQ


def test_gate_time_and_sub_chain_after_acquire_are_on_the_stored_clock():
    enable = Signal().carrier("sine", f_hz=50e6)
    s = acquired(0.3e9).gate(by=enable, threshold=0.0)
    x, ev = s.realize()
    t = np.array([e.t_s for e in ev if e.kind == "transition"])
    # a 50 MHz enable crosses zero every 10 ns: stated in Hz, so rendered at the stored rate
    print("gate transitions [ns]:", np.round(t * 1e9, 3)[:4])
    assert len(t) >= 15 and np.allclose(np.diff(t), 10e-9, atol=1e-12)


def test_matched_filter_after_a_decimating_downconvert():
    g = ws.Grid(fs=8e9, baud=25e6, n=1 << 17, domain="iq", fc_hz=1e9)
    s = (Signal(seed=1, grid=g).iq("qam16", prbs=15).upconvert()
         .downconvert(fc_hz=1e9, lpf=dict(pass_hz=20e6, stop_hz=60e6), fs_out=200e6)
         .filter("rrc", beta=0.35))
    x = s.waveform()
    sent, ts = s.transmitted()
    d = s.filter_delays()[0]["group_delay_s"]
    i = np.rint((ts + d) * 200e6).astype(int)
    ok = i < len(x)
    err = np.abs(x[i[ok]] - sent[ok])[10:-10].max()
    print(f"worst symbol error through the matched filter at 200 MSa/s: {err:.4f}")
    assert err < 0.02          # measured 0.008; 1.83 when the taps were designed at 8 GSa/s


def test_reports_are_at_the_rate_the_op_runs_at():
    s = acquired(0.3e9).filter("butter", f_hz=1e9, order=1).delay(by=1e-10)
    # first-order low-pass: group delay at DC is 1/(2 pi fc), whatever the rate
    assert s.filter_delays()[0]["group_delay_s"] == pytest.approx(1 / (2 * np.pi * 1e9), rel=2e-2)
    assert s.delay_errors()[0]["below_hz"] == pytest.approx(0.4 * FS_ACQ)


def test_a_corner_above_the_stored_nyquist_is_refused_where_it_is_written():
    with pytest.raises(ValueError, match="filter"):
        acquired(0.3e9).filter("butter", f_hz=15e9)      # fine at 80 GSa/s, not at 20


def test_unchanged_rate_is_bit_identical_to_the_grid():
    g = ws.Grid(fs=64e9, baud=1e9, n=1 << 12)
    s = Signal(seed=1, grid=g).carrier("nrz", n_ui=64).filter("butter", f_hz=2e9).delay(by=3e-11)
    from wfmsynth import stages as ST, joins as JN
    x = Signal(seed=1, grid=g).carrier("nrz", n_ui=64).waveform()
    want = JN.delay(ST.apply_filter(x, g.fs, "butter", f_hz=2e9, grid_baud=g.baud), 3e-11, g.fs)
    assert np.array_equal(s.waveform(), want)
