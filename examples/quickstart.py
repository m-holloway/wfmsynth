"""
Quickstart — generate physics-grounded waveforms and a realistic PAM4 capture.

    python examples/quickstart.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # run from source, no install needed
import wfmsynth as ws
from wfmsynth import physics as P

rng = np.random.default_rng(0)

# 1) a low-level primitive: a causal (minimum-phase) lossy channel on an NRZ signal
x = P.lossy_channel(P.nrz(n_ui=32, seed=3), length_in=10.0, tand=0.02, causal=True)
print("1) NRZ through a causal lossy channel:", x.shape, "| span", round(float(np.ptp(x)), 3))

# 2) apply a named impairment, then label-preserving domain randomization
y = ws.apply_impairment("crosstalk", P.pam4(n_ui=32, seed=5), rng)
y = ws.domain_randomize(y, rng)
print("2) PAM4 + crosstalk + domain randomization:", y.shape)

# 3) compositional grammar: many physics-grounded waveforms, each labeled with
#    its carrier kind and the impairments applied
g = ws.generate(16, seed=0)
print("3) grammar batch:", g["X"].shape, "| carriers:", list(g["carrier"][:6]))

# 4) a realistic segmented PAM4 "deep memory" scope capture with injected SI defects,
#    channel-class defects grouped across segments (shared physical link) — ground truth
cap = ws.deep_capture(n_segments=1000, needle_rate=0.03, seed=1, group_size=8)
defect_types = sorted(set(cap["labels"][i] for i in cap["needle_idx"]))
print("4) PAM4 capture:", cap["X"].shape, "| defects:", len(cap["needle_idx"]), "| types:", defect_types)

print("\nAll ground truth is returned alongside the waveforms — labels, needle indices, group ids.")
