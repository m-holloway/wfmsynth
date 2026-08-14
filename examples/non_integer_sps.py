"""
Non-integer samples-per-UI — the realistic default. The sample clock and the symbol
clock are unrelated, so a real acquisition is almost never an integer number of samples
per symbol. This shows the exact *fractional* pattern period, and why a tool that
assumes an integer samples-per-UI drifts and silently mis-recovers symbols.

    python examples/non_integer_sps.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import wfmsynth as ws

g = ws.Grid(fs=256e9, baud=112e9, n=1 << 14)         # 256 GSa/s, 112 GBd
sps = g.samples_per_ui
print(f"grid: {g.fs/1e9:.0f} GSa/s, {g.baud/1e9:.0f} GBd  ->  samples_per_UI = {sps:.6f}  (non-integer)")

# IEEE PRBS13Q repeats every 8191 symbols; its exact period in SAMPLES is fractional:
period_ui = 8191
print(f"PRBS13Q period: {period_ui} UI = {g.pattern_period_samples(period_ui):.3f} samples "
      f"(fractional — folding needs sub-sample realignment)")

# a known PAM4 symbol stream at the true (fractional) sps, recovered two ways
n_sym = 3000
n = int(round(n_sym * sps))
syms = np.random.default_rng(0).integers(0, 4, n_sym)
lv = np.array([-1.0, -1 / 3, 1 / 3, 1.0])
wav = lv[syms[np.clip((np.arange(n) / sps).astype(int), 0, n_sym - 1)]]


def recover(sps_used):
    pos = np.round(np.arange(n_sym) * sps_used + sps_used / 2).astype(int)
    pos = pos[pos < n]
    rec = np.argmin(np.abs(wav[pos][:, None] - lv[None, :]), axis=1)
    return (rec == syms[:len(pos)]).mean()


print(f"symbol recovery @ true sps={sps:.4f}: {100*recover(sps):5.1f}%   "
      f"@ assumed-integer sps={round(sps)}: {100*recover(round(sps)):5.1f}%   <- the trap")

# the ground truth you would emit alongside the waveform
gt = dict(samples_per_ui=round(sps, 6), n_symbols=n_sym,
          pattern_period_ui=period_ui,
          pattern_period_samples=round(g.pattern_period_samples(period_ui), 3))
print("ground truth:", gt)
