"""
Measured S-parameter (Touchstone) channels.

The analytic loss model is smooth and monotonic; real channels resonate. Read a `.sNp`
file (here a synthetic one with a notch) and drive synthesis through its S21.

    python examples/touchstone_channel.py
"""
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import wfmsynth as ws

# a synthetic 2-port with a resonant S21 notch at 20 GHz (stands in for a measured .s2p)
f = np.linspace(1e8, 40e9, 800)
S = np.zeros((len(f), 2, 2), complex)
S[:, 1, 0] = (1 - 0.9 * np.exp(-((f - 20e9) / 1.5e9) ** 2)) * np.exp(-1j * 2 * np.pi * f * 20e-12)
S[:, 0, 1] = S[:, 1, 0]
S[:, 0, 0] = S[:, 1, 1] = 0.05
path = os.path.join(tempfile.mkdtemp(), "thru.s2p")
ws.write_touchstone(path, f, S, fmt="RI")

g = ws.Grid(fs=80e9, baud=20e9, n=1 << 13)
sig = (ws.Signal(seed=1, grid=g)
       .carrier("pam4", n_ui=g.n // 4, pattern="prbs13q", causal=True)
       .sparam(path=path))               # drive PAM4 through the measured channel
y = sig.waveform()

# show the resonance in the output spectrum
probe = ws.touchstone_channel(np.random.default_rng(0).standard_normal(1 << 14), path,
                              grid=ws.Grid(fs=80e9, n=1 << 14))
fg = np.fft.rfftfreq(1 << 14, d=1 / 80e9)
Y = np.abs(np.fft.rfft(probe))
print("read Touchstone:", path)
print(f"PAM4 through the measured channel: {y.shape}")
print(f"spectral notch at 20 GHz: |Y(20GHz)|/|Y(10GHz)| = "
      f"{Y[np.argmin(abs(fg - 20e9))] / Y[np.argmin(abs(fg - 10e9))]:.3f}  (a resonance the "
      f"analytic model can't make)")
