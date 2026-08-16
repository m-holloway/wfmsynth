"""
Realistic scenarios composed from primitives.

The point of the Realism-v2 primitive set: any realistic scenario a scope might capture is a
COMPOSITION of small, validated primitives. Here are two end-to-end examples — a 50 GBd
electrical PAM4 link and an 800G-style optical lane — plus a multi-lane scene with a shared
supply rail. Everything carries a provenance recipe.

    python examples/realistic_scenario.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import wfmsynth as ws

g = ws.Grid(fs=256e9, baud=64e9, n=1 << 14)
n_ui = int(g.n // g.samples_per_ui)

# ---- 1) a realistic electrical PAM4 link: Tx FFE -> lossy+resonant channel -> crosstalk ->
#         supply coupling -> combined jitter (SSC + phase noise) -> Rx CTLE -> scope -> ADC ----
electrical = (ws.Signal(seed=1, grid=g)
              .carrier("pam4", n_ui=n_ui, pattern="prbs13q", causal=True)
              .tx_ffe(taps=[-0.12, 1.0, -0.2], pre=1)
              .lossy(loss_db=14.0, loss_at_ghz=32.0, causal=True)
              .resonant_reflect(td_ps=45.0, f0_ghz=28.0, q=8.0, gamma0=0.3)
              .crosstalk_matrix(couplings=[0.06, 0.04])
              .supply_coupling(f_ripple_hz=2e6, am_depth=0.02, psij_ps=1.5)
              .timing(ssc=dict(f_ssc=32e3, spread=0.004),
                      phase_noise=dict(rms_ps=0.4, slope=2.0), rj_ps=0.2)
              .ctle(fz_ghz=8.0, fp1_ghz=25.0, fp2_ghz=45.0)
              .scope(bw_hz=40e9).timebase(rms_ps=0.3)
              .digitize(snr_db=32.0, enob=6.5, interleave=dict(m_cores=4, offset_v=1e-3)))
xe = electrical.waveform()
print(f"electrical PAM4 link:  {xe.shape},  measured eye = {ws.eye_height(xe, g):.3f}")
print(f"  recipe: {len(electrical.recipe()['ops'])} composed ops, fully reproducible")

# ---- 2) an optical lane: PAM4 -> intensity (finite ER) + RIN + shot -> dispersion -> scope ----
optical = (ws.Signal(seed=2, grid=g)
           .carrier("pam4", n_ui=n_ui, pattern="prbs13q", causal=True)
           .optical(er_db=4.5, rin_db_per_hz=-140, shot=True, photons_per_unit=5e3)
           .dispersion(strength=8.0)
           .scope(bw_hz=45e9))
xo = optical.waveform()
print(f"\noptical PAM4 lane:     {xo.shape},  power >= 0: {xo.min() >= -1e-9}")

# ---- 3) a multi-lane scene: two lanes sharing a supply rail + lane-to-lane crosstalk ----
w0 = ws.Signal(seed=3, grid=g).carrier("pam4", n_ui=n_ui, pattern="prbs13q", causal=True, seed=3).waveform()
w1 = ws.Signal(seed=4, grid=g).carrier("pam4", n_ui=n_ui, pattern="prbs13q", causal=True, seed=9).waveform()
scene = (ws.Scene(g).add("lane0", w0).add("lane1", w1)
         .shared_supply(f_ripple_hz=2e6, am_depth=0.03)
         .couple(into="lane1", frm="lane0", coupling=0.05))
print(f"\nmulti-lane scene:      lanes {list(scene.lanes())}, correlated by a shared supply rail")

# every scenario is reproducible from its recipe
assert np.array_equal(ws.Signal.from_recipe(electrical.recipe()).waveform(), xe)
print("\nall scenarios reproduce bit-for-bit from their recipes.")
