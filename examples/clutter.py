"""A clean record against the same record with what a real capture carries.

Run from the repository root after ``pip install -e .``:

    python examples/clutter.py

WHAT THIS IS AN EXAMPLE OF
--------------------------
`wfmsynth.recipes`: arrangements of ops that almost nobody writes from scratch, each a plain
function that returns the chain it is made of.

    hum        50 Hz and its third harmonic, from the mains
    ripple     the board's own 50 kHz converter: a triangle, and a ring at each switching edge
    pickup     a medium-wave broadcast carrier, amplitude modulated
    flicker    1/f noise, as a sum of two-state traps

THE QUESTION
------------
A 3.3 V rail dips by 30 mV for a couple of microseconds, three times in 65 ms. A glitch
detector is set half-way: it fires when the rail falls through 15 mV below nominal. On a
synthetic rail with a millivolt of white noise that detector is perfect, and a model trained
on such records learns that a dip is the only thing that ever goes down. A real rail is not
like that. None of the four things above reaches 15 mV on its own, which is why each looks
harmless in a design review -- and the detector is measured here against each, and against
all of them at once.

WHAT THE LIBRARY DOES NOT DO HERE
---------------------------------
It does not model how each of these gets onto the rail: the coupling is the amplitude you
state. And it has no opinion on the detector. The last section shows what one obvious fix
does and what it costs; choosing it is engineering.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import wfmsynth as ws
from wfmsynth import Signal

FS, N = 8e6, 1 << 19                             # 65.5 ms: 3.3 mains periods
GRID = ws.Grid(fs=FS, n=N)
V_RAIL, DIP_V, DIP_SAMPLES = 3.3, 0.030, 16      # the dip: 30 mV, 2 us
T_DIP = [11e-3, 37e-3, 52e-3]
LEVEL = V_RAIL - 0.5 * DIP_V                     # the detector's threshold
F_SW, F_RING, F_RADIO = 50e3, 500e3, 1.2e6       # 16 samples on a ring period

CLUTTER = {
    "hum": lambda: ws.recipes.hum(50.0, amp=6e-3, harmonics={3: 0.3}, phase_rad=0.5),
    "ripple": lambda: ws.recipes.ripple(F_SW, amp=3e-3, duty=0.35, spike=9e-3,
                                        f_ring_hz=F_RING, q=3.0),
    "pickup": lambda: ws.recipes.pickup(F_RADIO, amp=2e-3, depth=0.6,
                                        message=Signal().carrier("sine", f_hz=1e3)),
    "flicker": lambda: ws.recipes.flicker(f_lo_hz=100.0, f_hi_hz=100e3, rms=1e-3),
}


def rail(*names, band=None, width=DIP_SAMPLES):
    """The rail, its three dips, the named clutter, and the instrument's own noise. `band`
    is a high-pass corner in Hz in front of the detector and `width` the dip's length in
    samples, both for section 3."""
    sig = (Signal(seed=1, grid=GRID).carrier("dc", level=V_RAIL)
           .events("glitch", on="times", times=T_DIP, amp=-DIP_V, width=width))
    if names:
        sig = sig.combine([CLUTTER[k]() for k in names])
    sig = sig.rx_noise(rms=1e-3)
    return sig if band is None else sig.filter("butter", btype="high", f_hz=band, order=2)


def detect(sig, level=LEVEL, within_s=2 * DIP_SAMPLES / FS):
    """``(found, false, rms_v, pp_v)``: how many of the stated dips the detector fired on,
    how many times it fired on anything else, and the record's rms and peak-to-peak about its
    mean. The truth is the record's own event list, not this file's constants."""
    x, events = sig.realize()
    truth = np.array([e.t_s for e in events.on_record() if e.kind == "glitch"])
    fired = ws.trigger.find(x, FS, level=level, slope="falling")["t_s"]
    near = np.abs(fired[:, None] - truth[None, :]) < within_s
    found = int(near.any(axis=0).sum())
    return found, int((~near.any(axis=1)).sum()), float(np.std(x)), float(np.ptp(x))


def main():
    everything = tuple(CLUTTER)
    print("1. The clutter is chains, not a noise knob")
    print("-----------------------------------------")
    full = rail(*everything)
    print(full.summary())
    print(f"\nroles() -- everything this record draws: {full.roles()}")
    print("Hum, ripple and pickup draw nothing: they are stated. The flicker's traps and the\n"
          "instrument's noise are the randomness, each under its own name.\n")

    print("2. The same detector, one thing added at a time")
    print("-----------------------------------------------")
    print(f"threshold {1e3 * (V_RAIL - LEVEL):.0f} mV below the rail; {len(T_DIP)} dips of "
          f"{1e3 * DIP_V:.0f} mV in {1e3 * N / FS:.1f} ms\n")
    print(f"  {'record':<22}{'rms mV':>8}{'pk-pk mV':>10}{'dips found':>12}{'false':>7}")
    rows = {}
    for label, names in [("clean", ())] + [("+ " + k, (k,)) for k in CLUTTER] + [
            ("+ all four", everything)]:
        rows[label] = detect(rail(*names))
        found, false, rms, pp = rows[label]
        print(f"  {label:<22}{1e3 * rms:>8.2f}{1e3 * pp:>10.1f}{found:>9}/{len(T_DIP)}{false:>7}")
    assert rows["clean"][:2] == (3, 0)
    assert all(rows["+ " + k][:2] == (3, 0) for k in CLUTTER), "each alone is harmless"
    assert rows["+ all four"][1] > 100
    print(f"\nEach alone leaves the detector perfect. Together it fires "
          f"{rows['+ all four'][1]} times on nothing:\nthe ring at a switching edge reaches "
          f"the threshold only while the mains is near its trough.")

    x = full.waveform()
    spec = 2.0 * np.abs(np.fft.rfft((x - x.mean()) * np.hanning(N))) / (0.5 * N)
    f = np.fft.rfftfreq(N, 1.0 / FS)
    owner = {50.0: "hum (3.3 periods in the record, so the window spreads it)",
             150.0: "hum, 3rd harmonic", F_SW: "ripple", F_RING: "the ring, a comb about it",
             F_RADIO: "pickup carrier", F_RADIO + 1e3: "pickup sideband"}
    print("\n  what a spectrum of the cluttered record shows (Hann window):")
    for f0, who in owner.items():
        k = int(np.argmin(np.abs(f - f0)))
        print(f"    {f0:>11.0f} Hz  {1e3 * spec[k - 2:k + 3].max():6.2f} mV   {who}")
    false_t = ws.trigger.find(x, FS, level=LEVEL, slope="falling")["t_s"]
    phase = (false_t * 50.0) % 1.0
    print(f"  the firings sit at {100 * phase.min():.0f} .. {100 * phase.max():.0f} % of the "
          f"mains period, {len(np.unique(np.floor(false_t * F_SW)))} switching periods of "
          f"{int(N / FS * F_SW)}")
    assert phase.max() - phase.min() < 0.5

    print("\n3. One fix, and what it costs")
    print("----------------------------")
    sag = 64000                                   # the same 30 mV, lasting 8 ms
    print(f"  {'in front of the detector':<26}{'2 us dips':>12}{'false':>7}{'8 ms sags':>12}")
    cost = {}
    for label, corner in (("nothing", None), ("a 1 kHz high-pass", 1e3)):
        level = LEVEL if corner is None else -0.5 * DIP_V
        cost[label] = (detect(rail(*everything, band=corner), level)[:2]
                       + detect(rail(*everything, band=corner, width=sag), level,
                                within_s=sag / FS)[:1])
        a, fa, b = cost[label]
        print(f"  {label:<26}{a:>10}/{len(T_DIP)}{fa:>7}{b:>10}/{len(T_DIP)}")
    assert cost["nothing"] == (3, rows["+ all four"][1], 3)
    assert cost["a 1 kHz high-pass"] == (3, 0, 0)
    print("Taking the mains out stops the false firings, because the ring alone never "
          "reached\nthe threshold. It also takes out every sag slow enough to sit under the "
          "corner:\na filter is a statement about which defects you no longer want to see.")

    print("\n4. What the dataset should hold")
    print("-------------------------------")
    again = Signal.from_recipe(full.recipe())
    assert np.array_equal(again.waveform(), x)
    sibling = full.contrast("rx_noise/3", seed=2)
    moved = float(np.std(sibling - x))
    quiet = rail(*everything).ops
    assert quiet[-1]["op"] == "rx_noise" and abs(moved / (np.sqrt(2) * quiet[-1]["rms"]) - 1) < 0.01
    print(f"The cluttered record replays from its recipe alone ({len(full.to_json())} bytes, "
          f"digest {full.sha256()[:12]}).\n`full.contrast('rx_noise/3')` is the same capture "
          f"with only the instrument's noise re-drawn:\nit differs by {1e3 * moved:.3f} mV rms, "
          f"sqrt(2) times that noise, with every dip, the hum, the ripple\nand each flicker "
          f"trap exactly where it was.")


if __name__ == "__main__":
    main()
