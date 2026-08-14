"""
Confounder-controlled sweeps + realized labels.

A naive reflection sweep is *also* an eye-height sweep, so a model trained on it can read
eye height instead of ISI structure. Hold the eye height fixed by solving insertion loss,
and label every example with attributes MEASURED from the output.

    python examples/confounder_sweep.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import wfmsynth as ws

g = ws.Grid(fs=200e9, baud=50e9, n=1 << 13)          # 4 samples/UI
n_ui = int(g.n // g.samples_per_ui)


def build(gamma=0.05, loss_db=2.0):
    return (ws.Signal(seed=1, grid=g)
            .carrier("pam4", n_ui=n_ui, pattern="prbs13q", causal=True)
            .lossy(loss_db=loss_db, loss_at_ghz=25.0, causal=True)
            .reflect(td_ps=30.0, gamma_s=gamma, gamma_l=gamma))


# 1) the confound, made visible: reflection alone closes the eye
print("naive reflection sweep (loss fixed) — eye height moves with the knob:")
for gm in (0.0, 0.15, 0.30, 0.40):
    print(f"  gamma={gm:.2f}  measured eye={ws.eye_height(build(gm, 0.0).waveform(), g):.3f}")

# 2) hold the eye height fixed by solving insertion loss as reflection is swept
target = ws.eye_height(build(0.05, 2.0).waveform(), g)
print(f"\nhold eye height = {target:.3f} while sweeping reflection (solve insertion loss):")
recs = ws.hold_constant(build, "gamma", [0.05, 0.15, 0.25, 0.35], "eye", target,
                        "loss_db", (0.0, 4.0), g, ws.eye_height, tol=0.004)
for r in recs:
    print(f"  gamma={r['gamma']:.2f} -> loss_db={r['loss_db']:.2f}  "
          f"realized eye={r['realized_eye']:.3f}")
print("  (loss must DROP as reflection rises — you cannot vary reflection, loss AND eye "
      "independently)")

# 3) realized-vs-requested labels + correlation matrix (a leak shows up here)
sets = [dict(gamma=gm, loss_db=1.0) for gm in np.linspace(0.0, 0.4, 6)]
_, corr, names = ws.realized_table(build, sets, g, ws.attributes)
print("\nrealized attribute correlation matrix (", names, "):")
for row in corr:
    print("  " + "  ".join(f"{v:+.2f}" for v in row))
