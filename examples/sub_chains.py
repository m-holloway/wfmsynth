"""
A sub-chain is a value -- the aggressor is "the neighbouring lane AFTER ITS OWN CHANNEL".

Every op that takes a second signal (`crosstalk(aggressor=)`, `crosstalk_matrix(aggressors=)`,
`modulate(message=)`, `open_drain(second=)`, a `net` driver's `source`) takes it the same way:
build it as a Signal and hand it over.

    python examples/sub_chains.py
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import wfmsynth as ws

g = ws.Grid(fs=64e9, baud=4e9, n=1 << 13)               # 64 GSa/s, 4 GBd, 16 samples/UI
n_ui = int(g.n // g.samples_per_ui)

# 1. The neighbouring lane is a chain of its own: its pattern, its jitter, its channel. No seed
#    and the same grid, so it becomes PART of the record it is handed to.
lane = (ws.Signal(grid=g)
        .carrier("nrz", n_ui=n_ui, seed=9, jitter=dict(rj=0.4))
        .lossy(loss_db=12.0, loss_at_ghz=2.0)
        .digitize(noise_rms=0.02))

sig = (ws.Signal(seed=1, grid=g)
       .carrier("nrz", n_ui=n_ui, jitter=dict(rj=0.4))
       .crosstalk(aggressor=lane, coupling=0.1)           # <- the whole lane, in one argument
       .digitize(noise_rms=0.01))
x = sig.waveform()

# 2. The chain reads as a tree. The [namespace] is how everything else refers to the sub-chain.
print(sig.summary())

# 3. The second signal is ground truth too: exactly the aggressor that was coupled.
at = "crosstalk/1/aggressor"
aggressor = sig.subchain(at)
print(f"\nthe aggressor as coupled: {aggressor.shape}, {np.ptp(aggressor):.3f} peak-to-peak")

# 4. Its randomness is its own, under that namespace -- re-roll ONE factor inside it.
print("re-rollable factors:", sig.roles())
sibling = sig.contrast(f"{at}/noise/2")                 # only the aggressor's noise differs
print(f"re-rolling the aggressor's noise moves the record by {np.abs(sibling - x).max():.4f} "
      f"peak (its own noise and both jitters are bit-identical)")

# 5. The recipe embeds the sub-chain, so the record replays from JSON with no extra files.
replayed = ws.Signal.from_recipe(json.loads(sig.to_json())).waveform()
print("replayed bit-for-bit from the nested recipe:", np.array_equal(replayed, x))

# 6. ANOTHER RECORD, as recorded -- its own seed, here its own symbol rate -- goes under "from".
#    Same sample rate and length, or it is refused: nothing is ever resampled to fit.
other = (ws.Signal(seed=4, grid=ws.Grid(fs=g.fs, baud=2.5e9, n=g.n))
         .carrier("pam4", n_ui=320, jitter=dict(rj=0.4)))
two = (ws.Signal(seed=1, grid=g)
       .carrier("nrz", n_ui=n_ui)
       .crosstalk_matrix(couplings=[0.08, 0.03], aggressors=[lane, {"from": other}]))
print()
print(two.summary())
print("the embedded record is the standalone record:",
      np.array_equal(two.subchain("crosstalk_matrix/1/aggressors/1"), other.waveform()))

try:
    ws.Signal(seed=1, grid=g).carrier("nrz", n_ui=n_ui).crosstalk(
        aggressor=ws.Signal(grid=ws.Grid(fs=32e9, baud=4e9, n=g.n)).carrier("nrz", n_ui=n_ui))
except ValueError as e:
    print("\na lane on another grid is refused, with the fix:\n ", e)
