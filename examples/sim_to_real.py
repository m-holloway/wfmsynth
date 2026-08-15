"""
Sim-to-real separability harness.

The question that tells you whether your realism work matters: can a trivial classifier tell
synthetic from real, and WHICH feature does it use? The separating feature names the missing
physics. (Here "real" is stood in for by a perturbed synthetic set.)

    python examples/sim_to_real.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import wfmsynth as ws

g = ws.Grid(fs=200e9, baud=50e9, n=1 << 12)
nui = g.n // 4


def mk(seed, noise_rms=0.0):
    s = (ws.Signal(seed=seed, grid=g).carrier("pam4", n_ui=nui, pattern="prbs13q",
                                              causal=True, seed=seed)
         .lossy(loss_db=6.0, loss_at_ghz=25.0, causal=True))
    return (s.digitize(noise_rms=noise_rms) if noise_rms else s).waveform()


synthetic = [mk(s) for s in range(60)]
also_synth = [mk(s + 5000) for s in range(60)]          # same generator, different seeds
real_like = [mk(s, noise_rms=0.08) for s in range(60)]  # stand-in for real captures

print("synthetic vs synthetic (same distribution):")
r = ws.separability(synthetic, also_synth, g)
print(f"  best feature = {r['best_feature']}  AUC = {r['best_auc']:.3f}  -> not separable (good)")

print("\nsynthetic vs 'real' (extra noise):")
r = ws.separability(synthetic, real_like, g)
print(f"  best feature = {r['best_feature']}  AUC = {r['best_auc']:.3f}  -> separable")
print("  full diagnosis (AUC per feature):")
for feat, auc in sorted(r["auc"].items(), key=lambda kv: -kv[1]):
    print(f"    {feat:18s} {auc:.3f}")
print("  -> the top feature names the physics to fix next.")
