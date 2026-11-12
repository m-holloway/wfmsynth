"""Where the last pieces of 0.46.0 meet each other.

Each was built and tested alone: the OFDM source and its receiver (#88), the delay and complex
gain on an envelope and `recipes.multipath` (#89), the state-dwell source (#90). These are the
checks that only exist once they are in one tree -- one piece measured by another that shares
no code with it. (A complex gain on an optical field, #89 meeting #92, is in
tests/test_optical_domain.py with the rest of the field's stages.)
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

import wfmsynth as ws
from wfmsynth import Grid, Signal

SCS = 312.5e3
G = Grid(fs=20e6, n=1 << 14, fc_hz=2.4e9 + 1e6)
OFDM = dict(n_fft=64, scs_hz=SCS, cp=16, active=52, constellation="qpsk", prbs=15,
            prbs_seed=0x2F6E15)


@pytest.fixture(autouse=True)
def _quiet():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


def tx():
    return Signal(seed=1, grid=G).ofdm(**OFDM)


def test_an_ofdm_equaliser_reads_back_a_multipath_channel_carrier_phase_included():
    """The second path is 6.6 samples late (between samples, inside the 16-sample prefix), so
    subcarrier k sees H(k) = 1 + g*exp(-j*2*pi*(fc + k*scs)*tau). The OFDM receiver's
    per-subcarrier equaliser returns it; leaving the carrier's turn out of the reference is
    off by more than half the echo."""
    tau, gain = 0.33e-6, 0.4
    rx = ws.recipes.multipath(tx(), delays_s=[0.0, tau], gains=[1.0, gain])
    sent, _ = tx().transmitted()
    raw = ws.rf.evm(rx.waveform(), G, sent, ofdm=OFDM)
    eq = ws.rf.evm(rx.waveform(), G, sent, ofdm=OFDM, correct=("equalizer",))
    k = np.asarray(eq["subcarriers"])
    h = 1.0 + gain * np.exp(-2j * np.pi * (G.fc_hz + k * SCS) * tau)
    no_turn = 1.0 + gain * np.exp(-2j * np.pi * k * SCS * tau)
    err = np.abs(eq["removed"]["equalizer"] - h).max()
    print(f"raw EVM {raw['evm']:.6f} (|H - 1| = {gain}); equalised {eq['evm']:.2e}; "
          f"|equaliser - H(k)| <= {err:.2e}; without the carrier's turn "
          f"{np.abs(eq['removed']['equalizer'] - no_turn).max():.3f}")
    assert abs(raw["evm"] - gain) < 1e-3              # the echo, counted whole
    assert eq["evm"] < 3e-3 and err < 1e-3            # the fractional-delay interpolator's error
    assert np.abs(eq["removed"]["equalizer"] - no_turn).max() > 0.5 * gain
    assert rx.transmitted()[0].shape == sent.shape    # the truth survives the arrangement


def test_an_ofdm_burst_follows_a_state_schedule_and_the_intervals_say_when():
    """`gate(by=<states>)` on an OFDM envelope: inside every stated on interval the samples are
    the source's, inside every off interval nothing, to the sample."""
    sig = tx().gate(by=Signal().carrier("states", dwell=[80e-6, 30e-6]))
    x, events = sig.realize()
    spans = ws.states.intervals(events, at="gate/1/by")
    assert len(spans) > 6 and {s["state"] for s in spans} == {0, 1}
    t, ref = np.arange(G.n) / G.fs, tx().waveform()
    edge = ws.states.DEFAULT_TR_SAMPLES / G.fs
    on, inside = np.zeros(G.n, bool), np.zeros(G.n, bool)
    for s in spans:
        m = (t > s["t_start_s"] + edge) & (t < s["t_stop_s"] - edge)
        inside |= m
        on |= m & (s["state"] == 1)
    assert inside.mean() > 0.95 and 0.1 < on.mean() < 0.9
    assert np.array_equal(x[on], ref[on]) and not np.any(x[inside & ~on])
    assert sig.roles() == ["gate/1/by/states/0"]      # the schedule is the one thing drawn
    assert [e.kind for e in events if e.kind == "ofdm_symbol"]    # both truths in one list
