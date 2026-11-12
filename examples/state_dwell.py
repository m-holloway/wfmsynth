"""A level that dwells in a state and jumps: burst noise, and a bursty link's schedule.

Run from the repository root after ``pip install -e .``:

    python examples/state_dwell.py

WHAT THIS IS AN EXAMPLE OF
--------------------------
One source, `carrier("states", ...)`, used three ways:

    .combine(telegraph)                    burst (random telegraph) noise on a signal
    .gate(by=busy)                         a link that is idle and active for irregular times
    .combine(ws.recipes.flicker(...))      1/f noise: many two-state processes superposed

A state lasts a time drawn from a stated distribution -- a number is the mean of an
exponential; a dict names a lognormal, a uniform, a fixed time or a table of measured dwells
-- and a transition table says which state follows which. Every interval comes back as a
`state` event, so each step below prints what was asked next to what the record holds.
"""
import numpy as np
from scipy import signal as sps

import wfmsynth as ws
from wfmsynth import states

# 1. BURST NOISE. A trap that is empty 40 us and full 10 us on average moves a sensor line by
#    2 mV. The telegraph process is its own chain; `combine` adds it.
g = ws.Grid(fs=1e7, n=1 << 20)                                      # 0.1 us per sample, 105 ms
telegraph = ws.Signal().carrier("states", levels=[0.0, 2e-3], dwell=[40e-6, 10e-6])
sig = ws.Signal(seed=1, grid=g).carrier("sine", f_hz=1e5, amp=0.5).combine(telegraph)
x, events = sig.realize()
iv = states.intervals(events, at="combine/1/others/0")
full = sum(r["t_stop_s"] - r["t_start_s"] for r in iv if r["state"] == 1) / (g.n / g.fs)
print(f"burst noise   {len(iv)} intervals; the trap is full {full:.3f} of the time "
      f"(long-run {states.occupancy([40e-6, 10e-6])[1]:.3f}); its role is {sig.roles()}")

# 2. AN ACTIVITY SCHEDULE. Idle gaps are exponential, bursts are lognormal about 30 us, and
#    the enable takes 200 ns to change. `gate` lets the data through while the level is high.
busy = ws.Signal().carrier(
    "states", dwell=[80e-6, {"dist": "lognormal", "median_s": 30e-6, "sigma": 0.4}],
    tr_s=200e-9)
link = ws.Signal(seed=2, grid=g).carrier("nrz", n_ui=1 << 14, seed=5).gate(by=busy)
y, events = link.realize()
iv = states.intervals(events, at="gate/1/by")
bursts = np.array([r["t_stop_s"] - r["t_start_s"] for r in iv if r["state"] and r["complete"]])
t = np.arange(g.n) / g.fs
idle = np.zeros(g.n, bool)
for r in iv:
    if r["state"] == 0:
        idle |= (t >= r["t_start_s"] + 200e-9) & (t <= r["t_stop_s"] - 200e-9)
print(f"bursty link   {bursts.size} whole bursts, median {np.median(bursts) * 1e6:.1f} us "
      f"(asked 30); of {idle.sum()} samples the schedule calls idle, "
      f"{int(np.count_nonzero(y[idle]))} are not zero")

# 3. 1/f NOISE is a composition, so it is a function that returns the chain, not an op.
gf = ws.Grid(fs=1e6, n=1 << 20)
noise = ws.recipes.flicker(f_lo_hz=30.0, f_hi_hz=1e5, rms=0.01)
z = ws.Signal(seed=3, grid=gf).carrier("dc", level=0.0).combine(noise).waveform()
f, pxx = sps.welch(z, fs=gf.fs, nperseg=1 << 15)
band = (f >= 300.0) & (f <= 1e4)
slope = np.polyfit(np.log10(f[band]), np.log10(pxx[band]), 1)[0]
print(f"flicker       {len(noise.ops[1]['others']) + 1} two-state processes; spectral slope "
      f"over 300 Hz..10 kHz {slope:.2f} (1/f is -1); rms {z.std() * 1e3:.2f} mV (asked 10)")
