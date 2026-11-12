"""A record made of parts: idle, preamble, payload -- and two streams sharing one line.

Run from the repository root after ``pip install -e .``:

    python examples/sequence_and_switch.py

WHAT THIS IS AN EXAMPLE OF
--------------------------
Two SOURCE ops whose parts are ordinary chains (sub-chains, see examples/sub_chains.py):

    .sequence([{"chain": idle, "duration_s": 1e-6}, {"chain": preamble, "duration_s": 1e-6}, {"chain": data}])
    .switch(select=gate, inputs=[lane_a, lane_b], threshold=0.0, t_switch_s=20e-9)

`sequence` joins chains in TIME: the durations are written in the recipe. `switch` selects
between chains with a SIGNAL: the instants are an outcome, and are emitted. Four things are
worth seeing, because each is something a model trained on the record would otherwise guess:

  1. the record IS the parts, spliced: outside the stated join window not one sample differs
     from the chains rendered on their own;
  2. where each part is comes back as data -- `layout()` and `segment` events -- so a
     per-sample label costs one line;
  3. whether a part's phase runs on across a boundary ("continue") or starts again
     ("restart") is stated per segment, and the two differ exactly where you would expect;
  4. a switch reports when its selector crossed the threshold, between samples.
"""
import numpy as np

import wfmsynth as ws

g = ws.Grid(fs=1e9, baud=50e6, n=8000)                        # 20 samples/UI, 8 us
T_IDLE, T_PRE, T_JOIN = 1e-6, 1e-6, 100e-9


def part():
    return ws.Signal(grid=g)


idle = part().carrier("dc", level=-1.0)                       # the line parked low
preamble = part().carrier("square", f_hz=25e6)                # a 1010 clock at the baud rate
data = part().carrier("nrz", n_ui=400, jitter=dict(rj=0.02))  # the payload


def burst(join="cut", t_join_s=None):
    return ws.Signal(seed=5, grid=g).sequence(
        [{"chain": idle, "duration_s": T_IDLE}, {"chain": preamble, "duration_s": T_PRE}, {"chain": data}],
        join=join, t_join_s=t_join_s)


# 1. The record is the parts, spliced. `subchain()` is each chain as this record rendered it.
sig = burst()
y = sig.waveform()
lay = ws.sequence.layout(sig.ops[0]["segments"], g)
alone = [sig.subchain(f"sequence/0/segments/{s['segment']}/chain") for s in lay]
spliced = np.concatenate([w[:s["stop"] - s["start"]] for w, s in zip(alone, lay)])
print("a sequence of idle, preamble and payload")
print(f"  samples that differ from the three chains spliced by hand: "
      f"{int(np.sum(y != spliced))} of {g.n}")

# 2. Where the parts are, as data. The label is built from the layout, not found in the samples.
label = np.zeros(g.n, int)
for s in lay:
    label[s["start"]:s["stop"]] = s["segment"]
    print(f"  segment {s['segment']}: samples {s['start']:5d}..{s['stop']:5d}  "
          f"{s['t_start_s'] * 1e6:.1f}..{s['t_stop_s'] * 1e6:.1f} us  ({s['state']})")
for e in sig.event_list():
    print(f"  event: {e.kind} {e.params['segment']} begins at sample {e.sample} "
          f"({e.t_s * 1e6:.1f} us)")
print(f"  mean level by label: " + ", ".join(
    f"{k}: {y[label == k].mean():+.2f}" for k in range(3)))

# A join that takes time is a modelled transient. Outside its window nothing has changed.
soft = burst("crossfade", T_JOIN).waveform()
m = int(round(T_JOIN * g.fs))
inside = np.zeros(g.n, bool)
for s in lay[1:]:
    inside[s["start"]:s["start"] + m] = True
print(f"  with a {T_JOIN * 1e9:.0f} ns crossfade: {int(np.sum(soft[inside] != y[inside]))} "
      f"samples differ from the cut inside the two {m}-sample join windows, "
      f"{int(np.sum(soft[~inside] != y[~inside]))} outside them")

# One channel and one receiver for the whole burst: ops after `sequence` act on the joined record.
rx = burst("crossfade", T_JOIN).lossy(loss_db=4.0, loss_at_ghz=0.05).rx_noise(rms=0.01)
print(f"  through one channel: {rx.waveform().std():.3f} V rms, digest {rx.sha256()[:12]}, "
      f"roles {rx.roles()}")

# 3. State across a boundary. 5.3 MHz is not a whole number of cycles in 2 us.
tone = part().carrier("sine", f_hz=5.3e6)
print("\nthe same tone on both sides of a boundary at 2 us")
for state in ("continue", "restart"):
    z = ws.Signal(grid=g).sequence([{"chain": tone, "duration_s": 2e-6},
                                    {"chain": tone, "state": state}]).waveform()
    same = np.array_equal(z, tone.waveform())
    print(f"  {state:8s}: step at the boundary {abs(z[2000] - z[1999]):.3f} V"
          + ("   (the uninterrupted tone, sample for sample)" if same else
             "   (the tone from its own t = 0: a phase jump)"))

# 4. A switch: two lanes on one line, chosen by a 500 kHz gate, 20 ns to change over.
gate = part().carrier("sine", f_hz=500e3, phase_rad=0.3)
lane_a = part().carrier("sine", f_hz=10e6, amp=0.5)
lane_b = part().carrier("nrz", n_ui=400)
mux = ws.Signal(seed=5, grid=g).switch(select=gate, inputs=[lane_a, lane_b], threshold=0.0,
                                       t_switch_s=20e-9)
w, events = mux.realize()
print(f"\na switch between two lanes: {len(events)} changes")
for e in events.events[:3]:
    print(f"  input {e.params['was']} -> {e.params['input']}: selector crossed "
          f"{e.params['threshold']:g} at {e.t_s * 1e9:9.3f} ns, first sample of the new "
          f"input {e.sample}, settled {e.width} samples later")
busy = events.mask().astype(bool)
ideal = np.where(gate.waveform() > 0.0, lane_b.waveform(), lane_a.waveform())
print(f"  outside the {int(busy.sum())} switching samples the line is the selected lane "
      f"exactly: {int(np.sum(w[~busy] != ideal[~busy]))} differ")
print(mux.summary())
