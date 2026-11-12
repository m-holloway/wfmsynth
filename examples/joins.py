"""A second signal meets the chain: add it, or let it drive a parameter.

Run from the repository root after ``pip install -e .``:

    python examples/joins.py

WHAT THIS IS AN EXAMPLE OF
--------------------------
Five ops, each one line, each taking another chain as a value:

    .combine(hum)              a second signal ADDED (also how="product" | "min" | "max")
    .gain(by=ripple)           a gain that follows another signal
    .offset(by=baseline)       a baseline that follows another signal
    .gate(by=enable)           an enable from somewhere else
    .delay(by=path)            a delay that moves, in seconds

`by` is a number or a chain. Whatever is inside the other chain -- its own noise, its own
filter, its own events -- comes along under its own namespace, and the recipe is still one JSON
document. Each step below prints the number the physics says it should produce next to the
number measured from the record.

WHICH OP IS WHICH. A sum is right where the sources do not load one another; two drivers on one
node is `net`. `crosstalk` (a coupled sum), `modulate` (a modulation), `burst` (a periodic gate
with ramped edges), `drift` (a ramp on gain or offset) and `timing` (clock jitter about zero)
stay the named ops for those effects; see `wfmsynth.joins` for the line between each.
"""
import numpy as np

import wfmsynth as ws

FS, N = 1e6, 100_000                                  # 0.1 s, so one FFT bin is 10 Hz
g = ws.Grid(fs=FS, n=N)
t = np.arange(N) / FS


def sensor(f_hz=1e3):
    return ws.Signal(seed=7, grid=g).carrier("sine", f_hz=f_hz, amp=1.0)


def line(x, f_hz):
    """Amplitude of the spectral line at `f_hz` (every tone here sits on a bin)."""
    return 2 * abs(np.fft.rfft(x)[int(round(f_hz * N / FS))]) / N


# 1. COMBINE: mains hum on the sensor. The hum is its own chain and keeps its own level.
hum = ws.Signal(grid=g).carrier("sine", f_hz=50.0, amp=0.02)
y = sensor().combine(hum).waveform()
print(f"combine   hum line at 50 Hz: {line(y, 50.0) * 1e3:.3f} mV (asked 20), "
      f"sensor line: {line(y, 1e3):.6f} (asked 1)")

# 2. GAIN: the amplifier's gain rides on a 3 % supply ripple at 120 Hz. That is AM, so it puts
#    a sideband of half the ripple either side of the tone.
ripple = ws.Signal(grid=g).carrier("sine", f_hz=120.0, amp=0.03, offset=1.0)
y = sensor().gain(by=ripple).waveform()
print(f"gain      sidebands at 880 / 1120 Hz: {line(y, 880.0) * 1e3:.3f} / "
      f"{line(y, 1120.0) * 1e3:.3f} mV (0.03 / 2 = 15)")

# 3. GATE: an enable that is itself a chain. The switching instants are ground truth, emitted
#    as events with the sub-sample time the enable crossed the threshold.
enable = ws.Signal(grid=g).carrier("square", f_hz=100.0, amp=0.5, offset=0.5, duty=0.25)
y, events = sensor().gate(by=enable).realize()
opened = [e.t_s for e in events if e.direction > 0]
print(f"gate      on {np.mean(y != 0) * 100:.1f} % of the record (duty 25 %), "
      f"{len(events)} transitions, opens every {np.mean(np.diff(opened)) * 1e3:.2f} ms (10)")

# 4. DELAY: a path that lengthens by 1 ns every microsecond -- a receding source. A delay
#    growing at a rate r plays the input back slower by (1 - r): a Doppler shift. The delay
#    here is a measured-style array, so it comes in through `capture`.
RATE = 1e-3
path = ws.Signal(grid=g).capture(values=RATE * t)
sig = sensor(10e3).delay(by=path)
y = sig.waveform()
spec = np.abs(np.fft.rfft(y * np.hanning(N)))
print(f"delay     10 kHz tone arrives at {np.argmax(spec) * FS / N:.0f} Hz "
      f"(10 kHz * (1 - {RATE:g}) = {10e3 * (1 - RATE):.0f})")
exact = np.sin(2 * np.pi * 10e3 * (t - RATE * t))
promise = sig.delay_errors()[0]
print(f"          worst sample error {np.abs(y - exact)[32:-32].max():.1e}; the op states "
      f"{ws.joins.delay_error(10e3, FS):.1e} at 10 kHz and {promise['error']:.1e} anywhere "
      f"below {promise['below_hz'] / 1e3:.0f} kHz")

# What the op will not do, it says. A delay cannot go negative; `timing` is the op for jitter
# about zero.
try:
    sensor().delay(by=ws.Signal(grid=g).carrier("sine", f_hz=300.0, amp=2e-6)).waveform()
except ValueError as err:
    print(f"refused   {str(err)[:98]}...")

# The whole thing is one record: one recipe, one tree, and every inner factor has a name.
bench = (sensor().combine(ws.Signal(grid=g).carrier("sine", f_hz=50.0, amp=0.02)
                          .digitize(noise_rms=0.005))
         .gain(by=ws.Signal(grid=g).carrier("sine", f_hz=120.0, amp=0.03, offset=1.0)))
print(bench.summary())
print("roles:", bench.roles())
again = ws.Signal.from_recipe(bench.recipe())
print("replays bit for bit from its recipe:", np.array_equal(again.waveform(), bench.waveform()))
