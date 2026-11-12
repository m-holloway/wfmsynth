"""A triggered capture: render long, let the scope's trigger pick the window.

Run from the repository root after ``pip install -e .``:

    python examples/triggered_capture.py

WHAT THIS IS AN EXAMPLE OF
--------------------------
A real capture is placed by a trigger: the event of interest sits a fixed time into the record,
the trigger has its own jitter, and holdoff decides which occurrence fires. One op does that:

    .trigger(pre_s=200e-9, post_s=600e-9)                 an edge, at the 50 % level

The grid is the LONG render -- the stretch the scope was watching -- and `trigger` keeps the
window. Four things to see, each a number a model trained on captures would otherwise guess:

  1. the window is a slice of the long render, and the triggering edge sits at `pre_s` to within
     half a sample; the event says exactly where;
  2. a PULSE-WIDTH trigger synchronises to the pattern: PRBS7 holds one run of seven ones, so
     triggering on a pulse wider than 6.5 UI fires once per 127 UI;
  3. SEGMENTED memory with a holdoff, and the timestamp of every segment;
  4. AVERAGING pattern-synchronous triggers cuts the noise by sqrt(n).

and a fifth that matters more than the rest: asking for more triggers than the render holds is
REFUSED, with the render length that would do, instead of returning an untriggered record.
"""
import numpy as np

import wfmsynth as ws

N_UI, SPU = 8192, 128
long_grid = ws.Grid(fs=2e9, baud=2e9 / SPU, n=N_UI * SPU)          # 524 us of a 15.6 MBd link
UI = 1.0 / long_grid.baud
SCOPE = dict(sample_rate_hz=500e6, record_length=long_grid.n // 4,  # the scope watches all of it
             input_bandwidth_hz=150e6, bits=8, enob=7.0, clip_full_scale=1.5)
NOISE = 0.03


def acquired():
    """The link and the scope looking at it, before any trigger: a new chain each call."""
    return (ws.Signal(seed=3, grid=long_grid)
            .carrier("nrz", n_ui=N_UI, pattern="prbs7", jitter=dict(rj=1.0))
            .lossy(loss_db=3.0, loss_at_ghz=0.02)
            .rx_noise(rms=NOISE)
            .acquire(SCOPE))


def triggers(events):
    return [e for e in events if e.kind == "trigger"]


long_x = acquired().waveform()
fs = SCOPE["sample_rate_hz"]
print(f"long render: {len(long_x)} samples at {fs / 1e6:.0f} MSa/s = {len(long_x) / fs * 1e6:.0f} us")

# 1. The simplest call. The record is the window; its edge is where a scope would put it.
sig = acquired().trigger(pre_s=200e-9, post_s=600e-9)
x, events = sig.realize()
t = triggers(events)[0]
start = t.params["start"]
print("\n1. edge trigger")
print(f"   record: {len(x)} samples, grid {sig.stored_grid(x).fs / 1e6:.0f} MSa/s")
print(f"   a slice of the long render: {np.array_equal(x, long_x[start:start + len(x)])}")
print(f"   level {t.params['level']:+.3f} (50 %), fired {t.params['timestamp_s'] * 1e6:.4f} us "
      f"into the long render")
print(f"   the crossing sits at {t.t_s * 1e9:.3f} ns; pre_s is 200 ns "
      f"(half a sample is {0.5 / fs * 1e9:.1f} ns)")

# 2. Pulse width: the one run of seven ones in PRBS7. It fires at the pulse's trailing edge.
sync = dict(condition="width", level=0.0, hysteresis=0.2, width_s=[6.5 * UI, None],
            pre_s=8 * UI, post_s=4 * UI)
_, events = acquired().trigger(n_segments=6, **sync).realize()
stamps = np.array([e.params["timestamp_s"] for e in triggers(events)])
print("\n2. pulse-width trigger on the pattern's longest run")
print(f"   widths {[round(e.params['width_s'] / UI, 2) for e in triggers(events)]} UI")
print(f"   spacing {np.round(np.diff(stamps) / UI, 2)} UI (PRBS7 repeats every 127)")

# 3. Segmented memory with a holdoff: every third repetition.
seg = acquired().trigger(n_segments=5, holdoff_s=300 * UI, **sync)
x, events = seg.realize()
rows = x.reshape(5, -1)
print("\n3. segmented memory, holdoff 300 UI")
print(f"   {rows.shape[0]} segments of {rows.shape[1]} samples")
for e in triggers(events):
    print(f"   segment {e.params['segment']}: timestamp {e.params['timestamp_s'] * 1e6:8.3f} us, "
          f"window starts at long-render sample {e.params['start']}")

# 4. Averaging. The same 32 triggers, taken as segments and as two averages of 16.
singles = acquired().trigger(n_segments=32, **sync).waveform().reshape(32, -1)
averaged = acquired().trigger(n_segments=2, average=16, **sync).waveform().reshape(2, -1)
n_win = singles.shape[1]
top = slice(int(2.5 * UI * fs), int(6.5 * UI * fs))             # inside the run of ones
rms_1 = (singles[:16, top] - singles[16:, top]).std() / np.sqrt(2)
rms_16 = (averaged[0, top] - averaged[1, top]).std() / np.sqrt(2)
print("\n4. averaging 16 pattern-synchronous triggers")
print(f"   noise on the settled level: {rms_1 * 1e3:.2f} mV single, {rms_16 * 1e3:.2f} mV "
      f"averaged -> {rms_1 / rms_16:.2f}x (sqrt(16) = 4)")

# 5. More triggers than the render holds: refused, with the length that would do.
print("\n5. asking for too much")
try:
    acquired().trigger(n_segments=200, **sync).waveform()
except ValueError as err:
    print("   " + str(err))

# The recipe is the whole record: the trigger is one more op, and it replays.
again = ws.Signal.from_recipe(seg.recipe())
assert np.array_equal(again.waveform(), x) and again.sha256() == seg.sha256()
assert abs(rms_1 / rms_16 - 4.0) < 1.0 and n_win == int(round(12 * UI * fs))
print(f"\nrecipe replays bit for bit; digest {seg.sha256()[:16]}")
