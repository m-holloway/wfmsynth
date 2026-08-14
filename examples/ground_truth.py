"""
Ground truth as measured, not as ideal.

Labels are measured FROM the generated output (they differ from the knobs, and that
difference is where silent label noise comes from). Eye height is reported under two named
definitions that diverge under deterministic ISI, and per-symbol labels carry the realized
integer-symbol alignment (a causal channel has group delay).

    python examples/ground_truth.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import wfmsynth as ws

g = ws.Grid(fs=200e9, baud=50e9, n=1 << 13)
n_ui = int(g.n // g.samples_per_ui)

# a link with a causal channel (group delay) and a discrete echo (deterministic ISI)
sig = (ws.Signal(seed=1, grid=g)
       .carrier("pam4", n_ui=n_ui, pattern="prbs13q", causal=True)
       .lossy(loss_db=3.0, loss_at_ghz=25.0, causal=True)
       .reflect(td_ps=40.0, gamma_s=0.3, gamma_l=0.3))

gt = sig.ground_truth()
print("measured ground truth:")
print(f"  eye height (contour) = {gt['eye_contour']:.3f}   <- measured vertical opening")
print(f"  eye height (sigma)   = {gt['eye_sigma']:.3f}   <- 3-sigma construction (diverges under ISI)")
print(f"  sampling phase       = {gt['best_phase']:.2f} samples")
print(f"  symbol alignment     = {gt['align_offset']} symbol(s)   "
      f"(corr {gt['align_corr']:.3f}; only {gt['align_corr_at_zero']:.3f} if you skip it)")

# the two eye definitions AGREE under Gaussian noise, DIVERGE under deterministic ISI
isi = sig.waveform()
gau = (ws.Signal(seed=1, grid=g).carrier("pam4", n_ui=n_ui, pattern="prbs13q", causal=True)
       .digitize(noise_rms=0.06)).waveform()
print("\neye-definition divergence:")
print(f"  deterministic ISI : |contour - sigma| = "
      f"{abs(ws.eye_height(isi, g, defn='contour') - ws.eye_height(isi, g, defn='sigma')):.3f}")
print(f"  Gaussian noise    : |contour - sigma| = "
      f"{abs(ws.eye_height(gau, g, defn='contour') - ws.eye_height(gau, g, defn='sigma')):.3f}")
