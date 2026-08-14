"""
Provenance-first composable synthesis — every waveform carries a complete, serializable
recipe (exact ground truth for training, fully reproducible).

    python examples/provenance.py
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import wfmsynth as ws

g = ws.Grid(fs=256e9, baud=112e9, n=1 << 14)             # 256 GSa/s, 112 GBd

# compose a real link in real units, as an explicit component graph
sig = (ws.Signal(seed=42, grid=g)
       .carrier("pam4", n_ui=g.n // 8, pattern="prbs13q", causal=True, jitter=dict(rj=0.4, pj=0.2))
       .lossy(loss_db=15.0, loss_at_ghz=26.0, causal=True)     # 15 dB @ 26 GHz
       .reflect(td_ps=55.0, gamma_s=0.4, gamma_l=0.4)          # echo at 55 ps
       .digitize(snr_db=32.0, enob=5.5, interleave=dict(m_cores=4, offset_mm=0.01)))

x = sig.waveform()
recipe = sig.recipe()
print("waveform:", x.shape)
print("recipe (the ground truth):")
print(json.dumps(recipe, indent=2))

# the recipe round-trips exactly — this is what makes a dataset trustworthy
x2 = ws.Signal.from_recipe(json.loads(json.dumps(recipe))).waveform()
print("reproduced bit-for-bit from the recipe:", np.array_equal(x2, x))

# a ground-truth dataset: sample knobs however you like; the sampled values are recorded
def build(rng):
    return (ws.Signal(seed=int(rng.integers(1e9)), grid=g)
            .carrier("pam4", n_ui=g.n // 8, seed=int(rng.integers(1e6)))
            .lossy(loss_db=float(rng.uniform(8, 22)), loss_at_ghz=26.0, causal=True)
            .reflect(td_ps=float(rng.uniform(20, 80)), gamma_s=float(rng.uniform(0.2, 0.5))))

X, recipes = ws.dataset(build, 8, seed=0)
print(f"\ndataset: {X.shape}, each example labelled to arbitrary depth by its recipe, e.g.:")
for r in recipes[:3]:
    print("  loss_db=%.1f  reflect_ps=%.1f  gamma_s=%.2f"
          % (r["ops"][1]["loss_db"], r["ops"][2]["td_ps"], r["ops"][2]["gamma_s"]))

# contrastive pairs: re-roll exactly one factor, hold everything else bit-identical
pair_a = sig.waveform()
pair_b = sig.contrast("noise/1")          # same symbols/jitter/channel; only the noise differs
print("\nre-rollable factors:", sig.roles())
print("contrastive pair differs only in noise:", not np.array_equal(pair_a, pair_b))
