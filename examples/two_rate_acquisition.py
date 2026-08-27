"""Two-rate synthesis: fine simulation -> instrument sampling -> stored record.

Use this pattern when an ML example should resemble data exported by a scope or
digitizer. Run from the repository root:

    python examples/two_rate_acquisition.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import wfmsynth as ws


# Simulate the source and interconnect on a fine grid so fast edge/channel effects
# exist before the instrument removes information.
sim_grid = ws.Grid(fs=200e9, baud=25e9, n=32_768)
n_ui = int(sim_grid.n / sim_grid.samples_per_ui)
signal = (
    ws.Signal(seed=3, grid=sim_grid)
    .carrier("nrz", n_ui=n_ui, causal=True)
    .lossy(loss_db=8.0, loss_at_ghz=12.5, causal=True)
    .reflect(td_ps=60.0, gamma_s=0.16)
)

# Describe the acquisition independently of the source. This instrument samples at
# one eighth of the simulation rate, has a finite analog bandwidth, a noisy 7-bit
# ADC, and stores a 512-bin peak-detect record.
profile = ws.AcquisitionProfile(
    sample_rate_hz=25e9,
    record_length=4000,
    input_bandwidth_hz=10e9,
    sample_clock_jitter_rms_s=250e-15,
    noise_floor={"rms": 0.002, "shape": "pink"},
    interleave={"m_cores": 4, "offset_v": 0.001},
    clip_full_scale=0.8,
    enob=7,
    decimation={"mode": "peak_hold", "depth": 512},
)

# Taps make information loss inspectable. "conditioned" is still on the fine
# simulation grid; "digitized" is sampled at the instrument rate; "stored" is the
# final record an ML pipeline would consume.
taps = signal.acquire_taps(profile)

for name in ("simulated", "conditioned", "digitized", "stored"):
    value = taps[name]
    print(f"{name:11s} shape={value.shape!s:12s} span={float(np.ptp(value)):.3f}")

print("realized acquisition:", taps["info"])
print(
    "peak-hold stores two channels:",
    "minimum and maximum in each time bin",
    f"({taps['stored'].shape[1]} bins)",
)

# For a one-line pipeline when intermediate taps are unnecessary:
stored_direct = signal.acquire(profile).waveform()
print("direct stored output:", stored_direct.shape, "(use this form when taps are unnecessary)")
