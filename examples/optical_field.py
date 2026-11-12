"""An optical link, and what the record IS at each point along it.

Run from the repository root after ``pip install -e .``:

    python examples/optical_field.py

WHAT THIS IS AN EXAMPLE OF
--------------------------
Between `eo` and `photodetect` a record is an OPTICAL FIELD -- E = sqrt(P) exp(j phi), complex,
with |E|^2 the power -- and the chain knows it:

    link = (ws.Signal(seed=1, grid=g).carrier("nrz", n_ui=256)   # a voltage: the drive
            .eo(kind="dml", alpha=3.0, er_db=8)                  # -> an optical field
            .fiber(length_km=10.0)                               #    ... still a field
            .photodetect()                                       # -> a photocurrent, real
            .tia(bw_hz=8e9))

The grid states no domain and the recipe does not mention one: the chain starts as a voltage,
and `eo` is where it becomes light. `stage_kinds(domains=True)` and `summary()` say where.

THE FOUR THINGS WORTH READING OFF THE OUTPUT
--------------------------------------------
1. An electrical op placed on the field is refused by name and told `photodetect` is the
   bridge. (It used to be cast to a float under a numpy warning.)
2. `combine` on a field is a COUPLER: fields add, powers do not. One laser split and rejoined
   detects as 4 P or as nothing; two lasers, each with its own linewidth, as 2 P.
3. `filter` on a field is an OPTICAL filter: after an amplifier it removes the noise outside
   its band, and the detected noise falls by the share of the band it kept.
4. Polarisation is not part of any of this: two fields are two named chains of one record.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import wfmsynth as ws
from wfmsynth import Signal

FS, BAUD, N_UI = 320e9, 10e9, 256                 # 32 samples per UI: 4.8 across an edge
g = ws.Grid(fs=FS, baud=BAUD, n=N_UI * 32)
field_grid = ws.Grid(fs=FS, baud=BAUD, n=g.n, domain="optical")   # a chain that STARTS as light


def drive(seed=1):
    return Signal(seed=seed, grid=g).carrier("nrz", n_ui=N_UI)


def power(sig):
    """What an ideal square-law detector reads."""
    return sig.photodetect(shot=False).waveform()


# 0. the link, and where it is light
link = (drive().eo(kind="dml", alpha=3.0, er_db=8).fiber(length_km=10.0)
        .photodetect().tia(bw_hz=8e9))
print(link.summary())
print("domains  ", [d for _, d in link.stage_kinds(domains=True)])
print("recipe   grid keys", sorted(link.recipe()["grid"]), "-- no domain: it starts as a voltage")
again = Signal.from_recipe(link.recipe())
print("replays bit for bit from its recipe:", np.array_equal(again.waveform(), link.waveform()))

# 1. a mistake is a refusal
try:
    drive().eo().lossy(loss_db=3.0, loss_at_ghz=5.0)
except ValueError as err:
    print(f"\nrefused  {str(err)[:211]}")

# 2. a coupler: one laser against itself, and against another laser
one = power(drive().eo()).mean()
same = power(drive().eo().combine(drive().eo())).mean()
dark = np.abs(power(drive().eo().combine(drive().eo(), gains=[-1.0]))).max()
DV = 200e6                                         # each laser's linewidth
two = np.mean([power(drive(s).eo(linewidth_hz=DV)
                     .combine(Signal().carrier("nrz", n_ui=N_UI).eo(linewidth_hz=DV))).mean()
               for s in range(1, 17)])
a = 2 * np.pi * DV * g.n / FS                      # the record, in coherence times
print(f"\ncoupler  one laser, split and rejoined: {same / one:.3f} P in phase (4), "
      f"{dark:.1e} out of phase (0)")
print(f"         two lasers of {DV / 1e6:.0f} MHz linewidth: {two / one:.2f} P over 16 records "
      f"(2 (1 + (1 - exp(-a)) / a) = {2 * (1 + (1 - np.exp(-a)) / a):.2f}, a = {a:.0f})")
beat = drive().eo(linewidth_hz=DV).combine(Signal().carrier("nrz", n_ui=N_UI).eo(linewidth_hz=DV))
print("         roles:", beat.roles())

# 3. an optical filter after an amplifier. A steady field, so what moves is the noise.
F_HZ = 20e9
steady = np.ones(g.n, complex)


def amplified(*, filtered):
    s = Signal(seed=3, grid=field_grid).capture(values=steady).edfa(gain_db=15.0, nf_db=5.0)
    if filtered:
        s = s.filter("butter", f_hz=F_HZ, order=4)
    return power(s)[g.n // 8:]                    # past the filter's settling


wide, narrow = amplified(filtered=False), amplified(filtered=True)
kept = 2 * F_HZ * (np.pi / 8) / np.sin(np.pi / 8) / FS     # butter-4 noise bandwidth, two-sided
print(f"\nfilter   detected noise: {wide.std() / wide.mean() * 100:.2f} % of the light unfiltered, "
      f"{narrow.std() / narrow.mean() * 100:.2f} % through a {F_HZ / 1e9:.0f} GHz optical filter")
print(f"         variance ratio {narrow.var() / wide.var():.3f}; the filter keeps "
      f"{kept:.3f} of the band the amplifier's noise fills")

# 4. polarisation: two fields are two chains. Both through one fibre; a detector that does not
#    resolve polarisation sees the two powers, added.
rec = ws.Channels(
    {"x": Signal().carrier("nrz", n_ui=N_UI).eo().use("span"),
     "y": Signal().carrier("nrz", n_ui=N_UI, pattern="prbs9").eo().use("span"),
     "pd": Signal().use("x").photodetect(shot=False)
     .combine(Signal().use("y").photodetect(shot=False))},
    shared={"span": Signal().fiber(length_km=10.0)}, seed=1, grid=g)
w = rec.waveforms()
print(f"\npolarisation  x, y are {w['x'].dtype} fields through one span; "
      f"|pd - (|Ex|^2 + |Ey|^2)| <= {np.abs(w['pd'] - np.abs(w['x']) ** 2 - np.abs(w['y']) ** 2).max():.0e}")
print(rec.summary())
