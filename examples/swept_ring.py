"""A ring resonator measured on a swept-wavelength bench, across a few dies, slow and fast.

Run from the repository root after ``pip install -e .``:

    python examples/swept_ring.py

WHAT THIS IS AN EXAMPLE OF
--------------------------
The measurement behind most passive photonic tests: a tunable laser sweeps, the device
filters it, a detector is sampled, and a second detector behind a reference interferometer
says what the wavelength was at each sample. Here it is a record with known ground truth:

    laser                 a swept source. Its field is held in the frame of its own moving
                          carrier, and that carrier IS the time-to-wavelength map
    resonator             the ring, solved in that frame (coupled-mode theory)
    recipes.mzi           the reference: the field and a delayed copy, delay + gain + combine
    recipes.stray_reflection   a setup artefact: a weak copy 40 ps late, the same three ops
    photodetect, tia      the detector and its bandwidth, as in any optical chain
    ws.tol, unit()        die-to-die variation: a tolerance where the number goes
    measure.swept_map, transmission, resonances      read the record back, knowing no recipe

One laser feeds both channels (`Channels`, `use`), so the reference sees the sweep the device
saw, nonlinearity included.

WHAT IT FINDS
-------------
Section 3: the stray reflection's fringe moves each die's measured Q by up to 4 % and its
extinction by up to 0.2 dB, by an amount set by where that die's resonance happens to sit on
the fringe. Section 4: the same dies swept at 100 nm/s into a 50 kHz detector read 10 to
20 % less Q and 3 to 19 dB less extinction, the best die losing the most. Section 5: a mode
hop moves every wavelength after it by 28 pm on the bench's own axis. None of these is the
ring; all three are in real campaigns.

WHAT THE LIBRARY DOES NOT DO HERE
---------------------------------
No laser linewidth, no polarisation, no Bragg grating, and nothing here was compared with a
measured sweep. The resonator is the coupled-mode model: a high-finesse ring.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import wfmsynth as ws
from wfmsynth import Signal

FS = 1e6
REF_DELAY_S = 2e-9                     # reference MZI: fringes 0.5 GHz (4 pm) apart
STRAY_DELAY_S, STRAY_R = 40e-12, 0.02  # a 6 mm air gap between two weak reflections
C_NM = 299792458.0e9
SLOW = dict(grid=ws.Grid(fs=FS, n=1 << 18), laser=dict(start_nm=1549.8, stop_nm=1550.2))
FAST = dict(grid=ws.Grid(fs=FS, n=1 << 12), laser=dict(start_nm=1549.8, speed_nm_per_s=100.0))
DIES = range(5)


def bench(grid, laser, stray=True, detector_hz=50e3):
    """The design: a laser whose sweep bows and ripples, a ring whose resonance, loss and
    coupling vary from die to die, a stray reflection after it, a detector; and the
    reference interferometer on a second detector."""
    src = Signal().laser(bow_nm=0.004, ripple_nm=0.5e-3, ripple_hz=90.0, **laser)
    ring = Signal().use("laser").resonator(
        wavelength_nm=ws.tol(1550.0, tol_abs=0.1),            # +/- 100 pm of resonance
        intrinsic_hz=ws.tol(110e6, 25), coupling_hz=ws.tol(80e6, 25))
    dut = Signal().use("ring")
    if stray:
        dut = ws.recipes.stray_reflection(dut, delay_s=STRAY_DELAY_S, reflectivity=STRAY_R)
    return ws.Channels(
        {"dut": dut.photodetect(shot=False).tia(bw_hz=detector_hz),
         "ref": ws.recipes.mzi(Signal().use("laser"), delay_s=REF_DELAY_S).photodetect(shot=False)},
        shared={"laser": src, "ring": ring}, seed=1, grid=grid)


def truth(rec):
    """What this die's ring IS, from the recipe: wavelength, loaded Q, extinction."""
    op = rec.recipe()["shared"]["ring"]["ops"][1]
    ki, ke = op["intrinsic_hz"], op["coupling_hz"]
    return dict(wavelength_nm=op["wavelength_nm"],
                q=C_NM / op["wavelength_nm"] / (ki + ke),
                extinction_db=-20 * np.log10(abs(ki - ke) / (ki + ke)))


def measure(rec, on="recovered"):
    """What the bench READS: the map from the reference channel, the device's record on
    it, and the dip's numbers. Nothing here looks at the recipe except the start wavelength."""
    w = rec.waveforms()
    lam = (ws.swept.wavelength_map(rec.signal("ref"))["wavelength_nm"] if on == "truth"
           else ws.measure.swept_map(w["ref"], rec.grid, REF_DELAY_S, at_s=0.0,
                                     wavelength_nm=1549.8))
    wl, t = ws.measure.transmission(w["dut"], lam)
    return max(ws.measure.resonances(wl, t), key=lambda d: d["extinction_db"]), (wl, t)


print("1. THE RECORD\n")
design = bench(**SLOW)
print(design.summary())
map_true = ws.swept.wavelength_map(design.signal("ref"))
w = design.waveforms()
map_got = ws.measure.swept_map(w["ref"], design.grid, REF_DELAY_S, at_s=0.0, wavelength_nm=1549.8)
linear = 1549.8 + 0.4 * map_true["t_s"] / design.grid.duration
print(f"\n   sweep: 1549.8 -> 1550.2 nm in {design.grid.duration * 1e3:.0f} ms. It is not "
      f"linear: {np.abs(map_true['wavelength_nm'] - linear).max() * 1e3:.2f} pm off a straight "
      f"line at worst.")
print(f"   map recovered from the reference fringes: "
      f"{np.abs(map_got - map_true['wavelength_nm']).max() * 1e3:.4f} pm from the truth at "
      f"worst (one fringe is {1550.0 ** 2 / C_NM / REF_DELAY_S * 1e3:.2f} pm).")

print("\n2. FIVE DIES, SLOW SWEEP, NO ARTEFACT: the bench reads what each die is\n")
print("   die   resonance nm (true)      Q (true)             extinction dB (true)")
clean = {}
for die in DIES:
    rec = bench(stray=False, **SLOW).unit(die)
    tr, (got, _) = truth(rec), measure(rec)
    clean[die] = got
    print(f"   {die}     {got['wavelength_nm']:.5f} ({tr['wavelength_nm']:.5f})    "
          f"{got['q']:9.0f} ({tr['q']:9.0f})    {got['extinction_db']:6.2f} "
          f"({tr['extinction_db']:6.2f})")

print("\n3. THE SAME DIES WITH A STRAY REFLECTION (r = 0.02, 40 ps: a 0.2 nm fringe)\n")
print("   die   fringe at resonance   extinction dB   change    Q change")
for die in DIES:
    rec = bench(**SLOW).unit(die)
    tr, (got, _) = truth(rec), measure(rec)
    # the artefact's own ground truth: the reflected copy's phase at this die's resonance
    fringe = 2 * STRAY_R * np.cos(2 * np.pi * C_NM / tr["wavelength_nm"] * STRAY_DELAY_S)
    print(f"   {die}     {fringe:+.4f}             {got['extinction_db']:6.2f}        "
          f"{got['extinction_db'] - clean[die]['extinction_db']:+.2f}    "
          f"{100 * (got['q'] / clean[die]['q'] - 1):+.2f} %")
print("   The level away from resonance ripples by +/- 4 %, and the dip rides on it: the "
      "error follows the\n   fringe's phase at each die's resonance, which the die-to-die "
      "spread in wavelength randomises.")

print("\n4. THE SAME DIES AT 100 nm/s INTO THE SAME 50 kHz DETECTOR\n")
print("   die   Q slow     Q fast    extinction slow   fast     dip found late by")
for die in DIES:
    fast = bench(stray=False, **FAST).unit(die)
    got, _ = measure(fast)
    late = got["wavelength_nm"] - truth(fast)["wavelength_nm"]
    print(f"   {die}   {clean[die]['q']:8.0f}  {got['q']:8.0f}     "
          f"{clean[die]['extinction_db']:6.2f}        {got['extinction_db']:6.2f}     "
          f"{late * 1e3:+.2f} pm")
wide = bench(stray=False, detector_hz=None, **FAST).unit(0)
got, _ = measure(wide)
print(f"   The ring did not change: die 0 at the same speed with no detector limit reads "
      f"Q {got['q']:.0f},\n   {got['extinction_db']:.2f} dB. At 100 nm/s a "
      f"{C_NM / 1550.0 ** 2 * 1550.0 / clean[0]['q'] / 1e6:.0f} MHz linewidth passes in "
      f"{C_NM / 1550.0 / clean[0]['q'] / (100.0 * C_NM / 1550.0 ** 2) * 1e6:.0f} us, and a "
      f"50 kHz detector takes about 7 us to respond.")

print("\n5. A MODE HOP: the reference loses count, and the record says where\n")
hop = dict(SLOW["laser"], hops=[{"t_s": 0.1, "by_nm": 0.03}])
rec = bench(grid=SLOW["grid"], laser=hop, stray=False).unit(0)
w, events = rec.realize()
at = [e for e in events["dut"] if e.kind == "mode_hop"][0]
(got, _), (onmap, _) = measure(rec), measure(rec, on="truth")
print(f"   event: {at.kind} at {at.t_s * 1e3:.0f} ms, {at.params['by_nm'] * 1e3:+.0f} pm "
      f"(to {at.params['wavelength_nm']:.4f} nm)")
print(f"   die 0 resonance is at {truth(rec)['wavelength_nm']:.5f} nm; the bench, counting "
      f"fringes through the hop, puts it at {got['wavelength_nm']:.5f} nm;\n   on the true "
      f"map it is at {onmap['wavelength_nm']:.5f} nm.")
