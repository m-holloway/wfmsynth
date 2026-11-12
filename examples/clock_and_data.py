"""
Several correlated channels from ONE recipe -- a clock and its data off the same timing source.

Two recipes that happen to share a seed are correlated by convention. Here the timing source is
declared once, by name, and both channels `use` it, so they are correlated by construction: the
same displacement, sample for sample, and one role that moves both.

    python examples/clock_and_data.py
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import wfmsynth as ws
from wfmsynth import Signal

g = ws.Grid(fs=320e9, baud=10e9, n=32 * 512)             # 32 samples/UI, 512 UI
n_ui = 512


def clock():
    return Signal().carrier("square", f_hz=5e9, tr_frac=0.3)


def data():
    return Signal().carrier("nrz", n_ui=n_ui, pattern="prbs7", tr_frac=0.3)


# 1. One record, three named chains. `pll` starts with an op that acts on a waveform, so it is
#    a STAGE and use("pll") applies it: both channels get the same realisation of it.
rec = ws.Channels({"clk": clock().use("pll"),
                   "data": data().use("pll").lossy(loss_db=3.0, loss_at_ghz=5.0)},
                  shared={"pll": Signal().timing(phase_noise=dict(rms_ps=2.0))},
                  seed=1, grid=g)
x = rec.waveforms()                                      # {"clk": array, "data": array}
print(rec.summary())
print("roles:", rec.roles())

# 2. Measure it: each channel's edge times against the same record with no timing source.
ideal = ws.Channels({"clk": clock(), "data": data().lossy(loss_db=3.0, loss_at_ghz=5.0)},
                    seed=1, grid=g).waveforms()


def crossings(y):
    s = np.signbit(y)
    i = np.nonzero(s[1:] != s[:-1])[0]
    return i + y[i] / (y[i] - y[i + 1])


def tie(name):                                           # edge time, and its error in samples
    t = crossings(ideal[name])
    return t, crossings(x[name]) - t


(tc, jc), (td, jd) = tie("clk"), tie("data")
at_data = np.interp(td, tc, jc)                          # the clock's error at each data edge
print(f"\nclock TIE {jc.std() / g.fs * 1e12:.2f} ps rms, data TIE {jd.std() / g.fs * 1e12:.2f} "
      f"ps rms; correlation at the {len(td)} data edges {np.corrcoef(at_data, jd)[0, 1]:.4f}")

# 3. Re-roll the ONE shared role: both channels move.
y = rec.contrast("pll/timing/0")
print("re-rolled pll/timing/0, largest change:",
      {name: round(float(np.abs(y[name] - x[name]).max()), 3) for name in x})

# 4. One recipe, one digest, every channel back bit for bit.
again = ws.Channels.from_recipe(json.loads(rec.to_json())).waveforms()
print("replayed bit-identically:", all(np.array_equal(again[name], x[name]) for name in x),
      "| sha256", rec.sha256()[:16])

# 5. The two legs of a pair are one bit stream: the second leg USES the first.
pair = ws.Channels({"p": Signal().carrier("nrz", n_ui=n_ui, jitter=dict(rj=0.3)),
                    "n": Signal().use("p").law("poly", coeffs=[0.0, -1.0])}, seed=1, grid=g)
legs = pair.waveforms()
print("pair: largest common-mode sample", float(np.abs(legs["p"] + legs["n"]).max()))
