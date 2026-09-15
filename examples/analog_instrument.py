"""Analog sources: a general-purpose instrument, not just a serial-link generator.

Run from the repository root after ``pip install -e .``:

    python examples/analog_instrument.py
"""
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import wfmsynth as ws

grid = ws.Grid(fs=10e9, n=4096, v_full=3.3)

# An analog carrier takes the same later ops a serial carrier does.
sine = ws.Signal(seed=1, grid=grid).carrier("sine", f_hz=10e6).probe(bw_hz=200e6).waveform()

# 'cmos' is unipolar and stays in real volts (0 V is 0, not stretched onto -1); a clock is
# duty=0.5, PWM is any other duty. tr_s is an absolute edge time -- fs must give it at least
# 8 samples across the transition (k = tr_s * fs = 20 here), the same k >= 8 sizing rule as
# everywhere else in this library.
clock = (ws.Signal(seed=1, grid=grid)
         .carrier("cmos", v_lo=0.0, v_hi=3.3, duty=0.5, f_hz=10e6, tr_s=2e-9)
         .probe(r_source=200.0, c_load_f=5e-12)   # a loaded-output RC pole, not a new knob
         .waveform())

# A capture starts a chain from a real file instead of a synthesized source, then takes the
# same ops: here a one-shot exponential stands in for "samples someone handed you". `fs_hz=`
# alone is provenance/resample-source; pass a matching `grid=Grid(fs=...)` too, or a
# downstream Hz-denominated knob like `loss_at_ghz` has no real rate to mean anything against.
fs_cap = 5e9
with tempfile.TemporaryDirectory() as tmp:
    path = os.path.join(tmp, "capture.npy")
    np.save(path, ws.physics.exp(tau_s=2e-9, n=4096, fs=fs_cap))
    captured = (ws.Signal(seed=1, grid=ws.Grid(fs=fs_cap, n=4096))
                .capture(path=path, fs_hz=fs_cap)
                .lossy(loss_db=3.0, loss_at_ghz=1.0).waveform())

print("sine  ", sine.shape, "cmos rails", round(clock.min(), 2), round(clock.max(), 2),
      "capture", captured.shape)
