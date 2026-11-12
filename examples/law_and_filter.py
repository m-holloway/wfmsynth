"""From an analog signal to a digital one: a filter, a comparator, and the times they emit.

Run from the repository root after ``pip install -e .``:

    python examples/law_and_filter.py

WHAT THIS IS AN EXAMPLE OF
--------------------------
The two GENERAL stages, for a shape that has no op of its own:

    .filter("butter", f_hz=20e3, order=2)                 a stated linear filter
    .law("comparator", threshold=0.0, hysteresis=0.1)     a named transfer law

The circuit is a zero-crossing detector: a noisy 200 Hz sensor signal, an anti-alias filter, and
a comparator that turns it into a logic level. Three things about it are worth seeing, because
each is a number a model trained on this record would otherwise have to guess:

  1. without hysteresis the comparator CHATTERS on the noise -- many transitions per crossing;
     with hysteresis wider than the noise it makes exactly one;
  2. the comparator's transitions are GROUND TRUTH, emitted as events with the sub-sample time
     of each crossing, so nobody has to find the edges again;
  3. a causal filter DELAYS, and it REPORTS by how much, so the time of the logic edge can be
     traced back to the zero crossing that caused it.

WHICH OP IS WHICH. `scope`, `probe`, `ctle`, `nonlinearity`, `digitize` and `pass_fet` stay the
named ops for an instrument, an equaliser, a transmitter, a converter and a switch. `law` and
`filter` are for the shape that is none of those; `ws.LAW_KINDS` and `ws.FILTER_KINDS` list them.
"""
import numpy as np

import wfmsynth as ws

FS, F_SIG, AMP = 1e6, 200.0, 1.0
F_CORNER, HYST, NOISE = 20e3, 0.1, 0.05
g = ws.Grid(fs=FS, n=1 << 17)


def detector(hysteresis):
    return (ws.Signal(seed=11, grid=g)
            .carrier("sine", f_hz=F_SIG, amp=AMP)
            .rx_noise(rms=NOISE)                              # wideband sensor noise
            .filter("butter", f_hz=F_CORNER, order=2)         # the anti-alias filter
            .law("comparator", threshold=0.0, hysteresis=hysteresis, v_lo=0.0, v_hi=3.3))


crossings = int(2 * F_SIG * g.n / FS)                         # zero crossings in the record
print(f"a {F_SIG:g} Hz sine crosses zero {crossings} times in {g.n / FS * 1e3:.1f} ms")

# 1. Hysteresis. The count is read from the events the comparator emitted, not from the samples.
for hyst in (0.0, HYST):
    n_tr = len(detector(hyst).event_list())
    print(f"  hysteresis {hyst:4.2f} V -> {n_tr:4d} transitions"
          + ("   (chatter: the noise re-crosses the threshold)" if n_tr > crossings else
             "   (one per crossing)"))

# 2. The transitions, as data. `sample` is the first sample at the new level; `t_s` is when the
#    comparator's INPUT crossed the trip level, between two samples.
sig = detector(HYST)
y, events = sig.realize()
first = events.events[0]
print(f"\nfirst transition: sample {first.sample}, t = {first.t_s * 1e6:.3f} us, "
      f"direction {first.direction:+.0f}, crossed {first.params['trip']:+.2f} V, "
      f"output now {first.params['level']:.1f} V")
assert y[first.sample] == first.params["level"]

# 3. Where did each edge come from? Three known delays stand between a zero crossing of the
#    sensor signal and the logic edge, and none of them has to be measured off the record:
#      - the filter's group delay, which the filter reports;
#      - the hysteresis: the input has to travel half the band past the threshold;
#      - and nothing else.
delay = sig.filter_delays()[0]
t_hyst = np.arcsin(0.5 * HYST / AMP) / (2 * np.pi * F_SIG)
t_edge = np.array([e.t_s for e in events])
t_zero = np.round(t_edge * 2 * F_SIG) / (2 * F_SIG)           # the crossing each edge belongs to
raw = t_edge - t_zero
corrected = raw - delay["group_delay_s"] - t_hyst
print(f"\nthe filter reports {delay['group_delay_s'] * 1e6:.3f} us of group delay at "
      f"{delay['at_hz']:g} Hz; half the hysteresis band costs {t_hyst * 1e6:.3f} us")
print(f"  edge - zero crossing, as emitted : {raw.mean() * 1e6:+7.3f} us mean")
print(f"  with both delays taken back out  : {corrected.mean() * 1e6:+7.3f} us mean, "
      f"{corrected.std() * 1e6:.3f} us rms (the noise, as jitter)")
assert abs(corrected.mean()) < 0.05 * raw.mean()

# The recipe is data, the two stages included, and it replays to the same samples.
print(f"\nrecipe ops: {[o['op'] + ':' + o['kind'] for o in sig.recipe()['ops'][2:]]}")
assert np.array_equal(ws.Signal.from_recipe(sig.recipe()).waveform(), y)
print(f"digest {sig.sha256()[:16]}  replays bit for bit")
