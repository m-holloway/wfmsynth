"""Guided quickstart: source -> channel -> acquisition -> ML variations.

Run from the repository root after ``pip install -e .``:

    python examples/quickstart.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import wfmsynth as ws


# Grid gives the array real units. Here each symbol lasts 100 ps and is represented
# by five simulation samples. Real sample and symbol clocks need not divide evenly.
# The quickstart uses the legacy 4096-point size because domain_randomize currently
# expects that default length (see BACKLOG #51).
grid = ws.Grid(fs=50e9, baud=10e9, n=4096)
n_ui = int(grid.n / grid.samples_per_ui)

# Build in physical order. causal=True prevents future samples from influencing an
# earlier edge. Channel loss acts like frequency-dependent blur; the reflection is
# a delayed connector/via echo.
signal = (
    ws.Signal(seed=7, grid=grid)
    .carrier("nrz", n_ui=n_ui, tr_frac=0.4, causal=True)
    .nonlinearity(compression=0.03, rise_fall_ratio=1.1)
    .lossy(loss_db=6.0, loss_at_ghz=5.0, causal=True)
    .reflect(td_ps=120.0, gamma_s=0.12)
)

simulated = signal.waveform()
print(
    "1) simulated link:",
    simulated.shape,
    "| samples/UI",
    round(grid.samples_per_ui, 3),
    "| span",
    round(float(np.ptp(simulated)), 3),
)

# An acquisition profile answers a different question: what would the instrument
# store? It limits bandwidth, samples at a lower rate, adds ADC effects, and then
# compresses the record. Peak-hold stores min/max channels so narrow events survive.
profile = ws.AcquisitionProfile(
    sample_rate_hz=20e9,
    record_length=1600,
    input_bandwidth_hz=8e9,
    enob=7,
    noise_floor={"rms": 0.002, "shape": "pink"},
    decimation={"mode": "peak_hold", "depth": 512},
)
taps = signal.acquire_taps(profile)
print(
    "2) acquisition:",
    "conditioned",
    taps["conditioned"].shape,
    "-> digitized",
    taps["digitized"].shape,
    "-> stored",
    taps["stored"].shape,
)

# A recipe is exact generation provenance. It can be serialized beside each ML
# sample and reconstructed later without relying on hidden random state.
recipe = signal.recipe()
replayed = ws.Signal.from_recipe(recipe).waveform()
print("3) recipe reproduces bit-for-bit:", np.array_equal(simulated, replayed))

# Keep class labels separate from nuisance variation:
#   impairment       -> the labelled effect a model should detect
#   domain_randomize -> harmless capture variation a model should ignore
rng = np.random.default_rng(11)
faulty = ws.apply_impairment("glitch", simulated, rng)
varied_normal = ws.domain_randomize(simulated, rng)
print(
    "4) ML variants:",
    "labelled glitch changed",
    int(np.count_nonzero(faulty != simulated)),
    "samples; varied normal shape",
    varied_normal.shape,
)

print("\nNext: python examples/realistic_scenario.py")
