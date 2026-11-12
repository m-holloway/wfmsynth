"""Many units of one design, and why a model must be tested on units it has never seen.

Run from the repository root after ``pip install -e .``:

    python examples/unit_variation.py

WHAT THIS IS AN EXAMPLE OF
--------------------------
A seed re-rolls noise. It does not make a second board: the resistor is still exactly 1 kohm.
Write the tolerance where the number would go and each record becomes one UNIT of a population:

    .probe(r_source=ws.tol(1e3, 5),                       # 1 kohm +/- 5 %, uniform
           c_load_f=ws.tol(2e-12, 10, dist="normal"))     # 2 pF, 10 % = 3 sigma, truncated

The value is drawn once and the recipe holds the NUMBER -- this unit's actual resistor -- with
the tolerance it came from beside it. Nothing is drawn on replay.

THE PART
--------
A CMOS output seen through a series resistor into a capacitive load: a 5 % resistor, a 10 %
capacitor, and a few millivolts of probe noise that differ on every capture.

WHAT IT SHOWS
-------------
1. One unit, its recipe, and the two re-rolls: another unit under the same noise, and the same
   unit under other noise.
2. 24 units x 6 captures, and where the spread in a measured rise time comes from.
3. The reason for the helper. A nearest-neighbour "model" is asked for a record's resistor.
   Split at random, it looks excellent -- it has seen other captures of the same unit and is
   recalling the unit. Tested on held-out units it is no better than guessing the nominal,
   which is the honest answer: the waveform shows R*C, and R cannot be read from R*C alone.
"""
import numpy as np

import wfmsynth as ws

GRID = ws.Grid(fs=100e9, n=8000)          # 80 ns: two cycles of a 25 MHz output, 10 ps a sample


def capture(rng):
    """One CAPTURE of the design. What a capture varies -- here the noise -- comes from `rng`;
    what a UNIT varies is written as a tolerance."""
    return (ws.Signal(seed=int(rng.integers(1 << 30)), grid=GRID)
            .carrier("cmos", v_lo=0.0, v_hi=3.3, f_hz=25e6, duty=0.5, tr_s=1e-10)
            .probe(r_source=ws.tol(1e3, 5),
                   c_load_f=ws.tol(2e-12, 10, dist="normal"),
                   noise_rms=3e-3))


def rise_ns(y):
    """10-90 % rise time of the first rising edge, in ns."""
    def crossing(th, after=0):
        k = np.flatnonzero((y[:-1] < th) & (y[1:] >= th))
        k = k[k >= after][0]
        return k, (k + (th - y[k]) / (y[k + 1] - y[k])) / GRID.fs
    k, t10 = crossing(0.33)
    return (crossing(2.97, after=k)[1] - t10) * 1e9


# ---- 1. one unit ------------------------------------------------------------------------------
s = capture(np.random.default_rng(0))
probe = ws.op_params(s.recipe(), "probe")
print("one unit")
print(f"  roles                 {s.roles()}")
for row in ws.tolerance.resolved(s.recipe()):
    stated = (f"{row['nom']:g} +/- {row['tol_pct']} % {row.get('dist', 'uniform')}")
    print(f"  {row['role']:<21} {row['value']:.6g}    drawn from {stated}")
print(f"  rise time             {rise_ns(s.waveform()):.3f} ns measured, "
      f"{np.log(9) * probe['r_source'] * probe['c_load_f'] * 1e9:.3f} ns = ln(9) R C from the recipe")

y = s.waveform()
other_unit = s.contrast("unit/*")           # another unit, the SAME noise
other_noise = s.contrast("probe/1")         # the same unit, other noise
print(f"  contrast('unit/*')    rise {rise_ns(other_unit):.3f} ns   (another unit)")
print(f"  contrast('probe/1')   rise {rise_ns(other_noise):.3f} ns   (this unit, other noise)")

replay = ws.Signal.from_recipe(s.recipe())
assert replay.sha256() == s.sha256() and np.array_equal(replay.waveform(), y)
print(f"  replayed from its recipe bit-identically, digest {s.sha256()[:16]}")

# ---- 2. a population --------------------------------------------------------------------------
N_UNITS, N_CAPTURES = 24, 6
X, recipes = ws.unit_dataset(capture, N_UNITS, N_CAPTURES, seed=1)
unit = np.array([ws.unit_of(r) for r in recipes])                       # the group key
r_ohm = np.array([ws.op_params(r, "probe")["r_source"] for r in recipes])  # the label
t = np.array([rise_ns(x.astype(float)) for x in X]).reshape(N_UNITS, N_CAPTURES)

print(f"\n{N_UNITS} units x {N_CAPTURES} captures")
print(f"  r_source across units {r_ohm.min():.1f} .. {r_ohm.max():.1f} ohm  (stated 950 .. 1050)")
print(f"  rise time, one unit   sd {t.std(axis=1).mean() * 1e3:.1f} ps   (noise)")
print(f"  rise time, the units  sd {t.mean(axis=1).std() * 1e3:.1f} ps   (the tolerances)")

# ---- 3. why the split is by unit ----------------------------------------------------------------
def nearest_neighbour_error(test):
    """RMS error, in ohms, of predicting each test record's resistor as that of the closest
    training waveform."""
    train = ~test
    err = []
    for i in np.flatnonzero(test):
        d = ((X[train] - X[i]) ** 2).sum(axis=1)
        err.append(r_ohm[train][int(np.argmin(d))] - r_ohm[i])
    return float(np.sqrt(np.mean(np.square(err))))


random_test = np.random.default_rng(2).permutation(len(X)) < len(X) // 4
held_out = np.isin(unit, np.unique(unit)[:N_UNITS // 4])        # six whole units
guess = float(np.sqrt(np.mean((r_ohm - 1e3) ** 2)))             # always answering "1 kohm"

e_random, e_units = nearest_neighbour_error(random_test), nearest_neighbour_error(held_out)
print("\npredicting r_source from the waveform (nearest neighbour), RMS error")
print(f"  random split          {e_random:6.1f} ohm   <- it has seen this unit before")
print(f"  held-out units        {e_units:6.1f} ohm   <- what it does on a new unit")
print(f"  answering '1 kohm'    {guess:6.1f} ohm")
assert e_random < 0.2 * guess < e_units
