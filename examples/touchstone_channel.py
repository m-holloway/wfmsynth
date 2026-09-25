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

# The grid: 16 samples/UI, so a tr_frac=0.5 edge gets k = 8 across the transition. That sets
# Nyquist at 160 GHz, and a measured response must COVER the band it is applied to -- otherwise
# `sparam` refuses rather than silently zeroing the power above the file's last frequency. So
# the synthetic response below runs to 160 GHz too; the notch it exists to show is still at
# 20 GHz. It also starts at DC, so the low-band guard has nothing to hold down either.
FS, BAUD = 320e9, 20e9

# a synthetic 2-port with a resonant S21 notch at 20 GHz (stands in for a measured .s2p)
f = np.linspace(0.0, FS / 2, 3200)      # from DC: a response must cover the band
                                        # it is applied to, at BOTH ends
S = np.zeros((len(f), 2, 2), complex)
S[:, 1, 0] = (1 - 0.9 * np.exp(-((f - 20e9) / 1.5e9) ** 2)) * np.exp(-1j * 2 * np.pi * f * 20e-12)
S[:, 0, 1] = S[:, 1, 0]
S[:, 0, 0] = S[:, 1, 1] = 0.05
path = os.path.join(tempfile.mkdtemp(), "thru.s2p")
ws.write_touchstone(path, f, S, fmt="RI")

g = ws.Grid(fs=FS, baud=BAUD, n=1 << 13)
n_ui = int(g.n // g.samples_per_ui)      # derived, not a hardcoded divisor
sig = (ws.Signal(seed=1, grid=g)
       .carrier("pam4", n_ui=n_ui, pattern="prbs13q", tr_frac=0.5, causal=True)
       .sparam(path=path))               # drive PAM4 through the measured channel
y = sig.waveform()

# show the resonance in the output spectrum
probe = ws.touchstone_channel(np.random.default_rng(0).standard_normal(1 << 14), path,
                              grid=ws.Grid(fs=FS, n=1 << 14))
fg = np.fft.rfftfreq(1 << 14, d=1 / FS)
Y = np.abs(np.fft.rfft(probe))
print("read Touchstone:", path)
print(f"PAM4 through the measured channel: {y.shape}")
print(f"spectral notch at 20 GHz: |Y(20GHz)|/|Y(10GHz)| = "
      f"{Y[np.argmin(abs(fg - 20e9))] / Y[np.argmin(abs(fg - 10e9))]:.3f}  (a resonance the "
      f"analytic model can't make)")
