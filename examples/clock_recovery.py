"""
Clock recovery — what the scope actually records.

A scope folds the raw record by a recovered clock (a PLL CDR). It tracks OUT timing jitter
slower than its loop bandwidth and passes jitter faster than it, so the recorded eye depends
on the recovery: an emitted eye is only meaningful alongside the CDR that produced it.

    python examples/clock_recovery.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import wfmsynth as ws

baud = 50e9
N = 1 << 16
t = np.arange(N) / baud

print("residual (eye) jitter vs transmitted jitter, by loop bandwidth:")
print("  jitter freq      BW=300kHz   BW=1MHz    BW=3MHz")
for fm in (1e5, 3e5, 1e6, 3e6, 1e7):
    phase = np.sin(2 * np.pi * fm * t)
    row = []
    for bw in (3e5, 1e6, 3e6):
        res = ws.jitter_transfer(phase, baud, bw)
        row.append(np.ptp(res[N // 2:]) / np.ptp(phase[N // 2:]))
    print(f"  {fm/1e6:6.2f} MHz     " + "   ".join(f"{v:7.3f}" for v in row))

print("\n-> low-frequency jitter is tracked out (small residual); high-frequency passes"
      " through.\n   A wider loop bandwidth tracks out more, but peaks near its own corner.")

# order matters under a frequency offset (a phase ramp): type-2 tracks it to zero
ramp = np.arange(N) * 2e-4
for order in (1, 2):
    res = ws.recover_clock(ramp, baud, 1e6, order=order)[1]
    print(f"\norder-{order} CDR, frequency-offset residual (steady) = "
          f"{np.mean(np.abs(res[N//2:])):.4f}")
