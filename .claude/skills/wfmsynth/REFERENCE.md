# wfmsynth reference

Worked examples for the shape and the traps in `SKILL.md`. Every snippet here has been executed
against the library with warnings as errors. Run them before building on them.

## A record end to end

```python
import numpy as np, wfmsynth as ws

baud, tr_s = 16e9, 20e-12            # symbol rate, and the transmitter's rise time
fs_synth = 10 / tr_s                 # 10 samples across the edge -> 500 GSa/s
fs_store = 256e9                     # what the instrument actually samples at
n_ui = 4096
n = int(round(n_ui * fs_synth / baud))

grid = ws.Grid(fs=fs_synth, baud=baud, n=n, v_full=0.8)
scope = ws.AcquisitionProfile(sample_rate_hz=fs_store,
                              record_length=int(round(n * fs_store / fs_synth)),
                              input_bandwidth_hz=33e9, enob=5.9)

sig = (ws.Signal(seed=7, grid=grid)
       .carrier("nrz", pattern="prbs13", n_ui=n_ui, tr_frac=0.35, causal=True, seed=11)
       .timing(rj_ps=0.9)
       .lossy(loss_db=12.0, loss_at_ghz=8.0, causal=True)
       .events("glitch", severity=1.0, indices=[1024])
       .acquire(scope))

x, events = sig.realize()            # normalised amplitude, plus what was injected
volts = np.asarray(x, float) * 0.5 * grid.v_full
```

`waveform()` returns the samples alone; `realize()` returns them with an `EventList`.
`carrier(seed=)` is separate from `Signal(seed=)` and is what moves the pattern phase.

## The two rates

The synthesis grid is not a sample rate anyone owns: sizing from the edge routinely asks for
hundreds of gigasamples per second, and storing at that rate describes an instrument that does not
exist. So a record has two rates, and they must be separated deliberately.

`acquire(AcquisitionProfile(...))` is the mechanism. It takes `sample_rate_hz`, `record_length`,
and optionally `input_bandwidth_hz`, `enob`, `sample_clock_jitter_rms_s`, `clip_full_scale`,
`interleave`, `noise_floor`, `decimation`.

Measured on the chain above — a glitch injected at a known symbol, located by differencing against
an otherwise identical clean record:

| symbol | `t_s` | `int(t_s * fs_store)` | measured | error |
|---|---|---|---|---|
| 1024 | 64.0 ns | 16,384 | 16,388 | +4 |
| 3000 | 187.5 ns | 48,000 | 48,004 | +4 |
| 4000 | 250.0 ns | 64,000 | 64,004 | +4 |

The +4 is the 33 GHz front end's group delay, not an error in the rule.

**The raw op, if you need it.** `sample_clock`'s `ppm` is a rate scaling relative to the synthesis
grid rather than a small offset, and `n_out` is a record length. To store at `fs_store` off a grid at `fs_synth`
with a genuine `ppm_real` of timebase offset:

```python
ppm_real = 55.0                      # the timebase offset you actually want
n_out = int(round(n * fs_store / fs_synth))
ppm_op = ((fs_store / fs_synth) * (1 + ppm_real * 1e-6) - 1.0) * 1e6

raw = (ws.Signal(seed=7, grid=grid)          # a chain that has NOT been through acquire()
       .carrier("nrz", pattern="prbs13", n_ui=n_ui, tr_frac=0.35, causal=True, seed=11)
       .scope(bw_hz=33e9)
       .sample_clock(ppm=ppm_op, n_out=n_out, span="strict"))
```

For the numbers above that is −487,972 ppm. `span="strict"` raises rather than holding the last
sample when the clock would need data past the end of the render.

Passing a genuine `ppm` of 55 together with an `n_out` computed from the rate ratio does **not**
resample. It truncates the record to that many samples at the grid's own rate: 65,536 samples
spanning 131 ns of a 256 ns render, 49 % of it gone and nothing said. A defect in the discarded
half then has a label with no signal behind it.

`drift_ppm_per_s` walks the clock; `phase0_s` sets where it starts. Resampling is asynchronous, so
samples-per-symbol need not be a whole number and should not be forced to one.

Keep the front end below the sampler's Nyquist. A declared analog bandwidth above about 0.45 × fs
is not an instrument, and content above Nyquist folds into the record. The library does not
enforce this — it clamps the filter corner and carries on — so check it yourself.

## Sizing

```python
k = 10                               # samples across the transition; 8 is the floor
fs = k / tr_s                        # 500 GSa/s for a 20 ps edge
```

The bandwidth a given edge needs, or the edge a bandwidth produces, is `BW × tr = 0.3497` for
10–90 % and `0.1348` for 30–70 %. Both are single-pole constants, confirmed analytically and on a
20 M-point step response. On the library's default 4th-order Bessel front end the 10–90 % figure
holds to 0.15 % (0.3502) while 30–70 % is 0.149, about 10 % off.

**The delivered edge is the one requested**, to a fraction of a percent once the grid resolves it.
Measured on one isolated step, 10–90 % by interpolated crossings:

| | k=8 | k=10 | k=32 |
|---|---|---|---|
| `causal=True` | 1.0106× | 1.0065× | 1.0004× |
| `causal=False` (default) | 1.0124× | 1.0078× | 1.0007× |

The residual falls as the grid gets finer, which is how you know it is the grid's and not the
model's. Below `tr_frac × samples_per_ui = 2` the library clamps to two samples and warns, and past
that point you are measuring the grid — so measure `tr` on the record you actually built.

### Record length

```python
repeats = 256                        # buys sqrt(256) = 16x; measured 15.8x. 128 buys 11.4x
period = ws.describe_pattern("prbs13", length=8191)["period"]    # 8191
n_ui_folded = repeats * period       # 2,096,896 symbols
```

Take the period from the registry when you know the pattern. For a stream you did not generate,
`ws.pattern_period(symbols)` returns `(lag, score, autocorrelation)` and the lag is *a* period
rather than the fundamental — the autocorrelation ties exactly at every multiple and float rounding
picks one. Measured on clean PRBS, every row scoring 1.000:

| pattern | period | 4 reps | 8 reps | 16 reps |
|---|---|---|---|---|
| prbs7 | 127 | 127 (1×) | 381 (3×) | 889 (7×) |
| prbs9 | 511 | 511 (1×) | 511 (1×) | 2044 (4×) |

Which multiple you get shifts with the pattern phase, not just the record length. Fold on a
multiple and you get proportionally fewer folds. Divide the lag by small integers and keep the
smallest that still correlates.

There is no coherent-averaging fold in the library. `ws.recover_and_fold` splits a record into
blocks, measures eye height per block and returns the median — it does not average the waveform.
Write the fold yourself if you want the √N.

## Sub-sample timing

Every timing impairment moves the signal by a fraction of a sample, and all of them are
interpolated with the same bandlimited kernel rather than linearly — jitter, duty-cycle
distortion, intra-pair skew, the free-running sample clock, and a reflection's delay. So a
reflection delay is continuous: `td_ps=100.0` and `td_ps=100.25` give different records, which
matters because the echo's phase within the eye is what decides whether it lands on a crossing or
in the middle.

The SOURCE's own edges are a separate matter. `square` and `cmos` place each edge at its own
time between samples (measured at 0.997 GHz on 64 GSa/s, where snapping to the next sample gave
15.6 ps of edge-time error: 3e-5 ps with a 9.6-sample edge, 0.011 ps at `square`'s default
3.2-sample edge, 0.052 ps for a 2.2-sample `cmos` edge). An `nrz` / `pam<N>` / `symbols` carrier places its
edges ON samples, so its clock is clean only when samples per UI is a whole number: choose `n`
and `n_ui` so `n / n_ui` is whole. `carrier(jitter=)` displaces edges by whole samples for the
same reason; jitter that means what it says is `timing(rj_ps=)`. `carrier` warns once when
`n_ui` is left out on a grid whose `baud` the default 32 symbols do not match.

Two places still interpolate linearly, and both act on the synthesis grid where the sizing rule
keeps the oversample high: the FFE's fractional tap spacing and interleaved-converter channel
skew. The cost there is bounded and measured (under 0.1 % of peak-to-peak at 20+ samples/UI).

## Recipes

```python
r = sig.recipe()                     # plain data: {"ops": [...], "grid": {...}, "seed": ...}
digest = sig.sha256()                # content address of the chain

with open("recipe.json", "w") as fh:
    fh.write(sig.to_json())

again = ws.Signal.from_recipe(r)
assert again.sha256() == digest
assert np.array_equal(again.waveform(), sig.waveform())      # bit-identical, not merely close
```

A recipe op is **flat** — `{"op": ..., <parameters>, "_prov": {...}}`. `op` is reserved, anything
beginning with an underscore is annotation, and everything else has to be a parameter the op
reads. `annotate(stage=..., node=...)` attaches provenance without touching the samples.

Keep the recipe, not the waveforms. Tens of kilobytes expand to hundreds of megabytes.

## Events and labels

`events()` places a localised mechanism and reports what it injected. Placement is `indices=[...]`
in symbols with the default `on="symbols"`, or `count=`/`fraction=`/`every=` to scatter, or
`times=`/`samples=` to place directly. `ws.MECHANISMS` lists the kinds.

```python
for e in events.events:
    stored_index = int(e.t_s * fs_store)     # correct in the record you stored
```

Label from `events.on_record()`. `e.on_record` is False for an event that reached the record
only through another op (an aggressor's glitch, an enable's edge) and for a trigger firing;
`mask()` and `overlapping()` skip those, a loop over `events` does not. `e.at` says which
sub-chain an event came from: a chain that IS the record (a `sequence` segment, a `switch`
input while selected, a chain brought in by `use`) keeps its events on it. Labels follow the
samples through `acquire`, `downconvert(fs_out_hz=)`, `trigger` and `delay`; a `filter` moves
them by `filter_delays()`, which you add. A trigger's crossing is at `pre_s` only to half a
sample; the exact time is the trigger event's `t_s`.

`Event.sample` indexes the grid the event was placed on. After any resampling it points somewhere
else, and where the synthesis grid outruns the sampler it can exceed the whole stored record. A
label silently dropped for being out of range does not read as missing data — `truth` tables are
usually treated as authoritative, so it reads as a positive claim that a defective record is
clean, and a model trained on it learns the defect is normal.

## Patterns

```python
def fn(length, **kw):                # every generator takes `length`
    return list(np.resize([3.0, 1.0, -1.0, -3.0], length))

ws.register_pattern(
    "my_stress", fn,
    period=65535,                            # symbols before it repeats
    levels=4,                                # which carrier these symbols belong on
    source="IEEE 802.3-2022 120.5.11.2.3",   # the document and clause: makes the name auditable
    marker=(3.0, -3.0, 3.0),                 # a level subsequence unique within the period
    hash_source=True,                        # your code, so a changed generator is named
)
symbols = ws.resolve_pattern("my_stress", length=65535)
sig = ws.Signal(seed=1, grid=grid).pattern("my_stress", length=65535, tr_frac=0.35)
```

`carrier(pattern=...)` will not take a registered name — that argument is the closed built-in set
(`legacy`, `prbs7/9/11/13/15/23/31`, `clock`) and never consults the registry. `levels=` declares
which carrier the symbols belong on, so a `levels=4` pattern cannot drive an NRZ carrier.

Register rather than passing a bare callable: a function cannot be serialised or digested, so a
recipe holding one cannot be replayed. If a consumer may not have your registration, put the
symbols in the recipe as data — `Signal.symbols([...])`, or
`Signal.pattern(name, length=..., embed=True)` to render from the registry and embed the result.
`source` and `marker` are empty on the entries the library ships, because which document specifies
a polynomial is the caller's knowledge.

## Analog, CMOS, and captured sources

A chain does not have to start with a digital carrier. Analog kinds (`sine`, `square`,
`triangle`, `sawtooth`, `dc`, `step`, `pulse`, `exp`, `chirp`, `two_tone`, `noise`) and the
unipolar `cmos` kind take the SAME later ops a serial carrier does:

```python
g = ws.Grid(fs=10e9, n=4096, v_full=3.3)          # k = tr_s * fs = 20, the same k >= 8 rule
clock = (ws.Signal(seed=1, grid=g)                # as everywhere else in this library
         .carrier("cmos", v_lo=0.0, v_hi=3.3, duty=0.5, f_hz=10e6, tr_s=2e-9)
         .probe(r_source=200.0, c_load_f=5e-12)      # a loaded-output RC pole, not a new knob
         .waveform())
assert clock.min() >= -0.2 and clock.max() <= 3.5     # real volts, not stretched onto +/-1
```

`cmos` stays in real volts (0 V is 0, not -1) and `tr_s` is an absolute edge time in seconds,
unlike `square`'s `tr_frac` — a real gate's edge does not shrink as it is clocked faster. Like
every other edge in this library, an under-resolved `tr_s` (fewer than ~2 samples) clamps to the
floor and warns rather than silently inflating the edge (`physics.resolve_rise_time`'s clamp,
made visible the same way for an absolute edge). `physics.am`/`fm`/`chirp` predate absolute
units, take a normalised ramp, and stay bit-identical; `am_modulate`/`fm_modulate`/`pm_modulate`
(below) are the Grid-native ones.

`capture()` starts a chain from a real file instead of a synthesized one:

```python
fs_cap = 5e9
np.save("scope.npy", ws.physics.exp(tau_s=2e-9, n=4096, fs=fs_cap))
captured = (ws.Signal(seed=1, grid=ws.Grid(fs=fs_cap, n=4096))
            .capture(path="scope.npy", fs_hz=fs_cap)
            .lossy(loss_db=3.0, loss_at_ghz=1.0).waveform())
```

The recipe records the path AND a sha256 digest over the sample VALUES (not the file's bytes),
taken when `.capture()` is called. A file that no longer matches that digest is a named error
at replay time; a missing file is a named error, not zeros. `embed=True` reads the file now and
stores its values inline, so the recipe no longer depends on the file at all.

`fs_hz=` on `capture()` is provenance and what `resample=True` resamples FROM; it does **not**,
by itself, make a downstream Hz-denominated knob (`lossy(loss_at_ghz=...)`, `probe(bw_hz=...)`)
mean anything real. Those read `grid.fs`, so pass a `grid=Grid(fs=..., n=...)` matching the
file's own rate — as above — or a `capture()`-only chain with no `Grid` renders against no real
rate at all. If the file's own length does not equal `Grid.n`, that is a named error unless
`resample=True` is passed (which then resamples onto `Grid.n`, or onto the rate implied by
`Grid.fs` vs `fs_hz` if both are given) — silently truncating or padding a captured record is
exactly the failure `sample_clock(n_out=)`'s trap above exists to prevent.

`modulate()` is AM/ASK/OOK/FM/FSK/PM, the message a nested carrier spec:

```python
am = (ws.Signal(seed=1, grid=ws.Grid(fs=1e9, n=1 << 14))
      .carrier("sine", f_hz=100e6, amp=1.0)
      .modulate(kind="am", depth=0.5, message=dict(kind="sine", f_hz=1e6))
      .waveform())
fm = (ws.Signal(seed=1, grid=ws.Grid(fs=1e9, n=1 << 14))
      .carrier("dc", level=0.0)                       # unused by FM/PM -- see below
      .modulate(kind="fm", fc_hz=100e6, dev_hz=5e6, message=dict(kind="sine", f_hz=1e6))
      .waveform())
```

AM/ASK/OOK COMBINE with the upstream carrier; FM/FSK/PM GENERATE their own carrier at `fc_hz`
and never read the upstream signal — frequency/phase modulation is not a per-sample transform
of an already-rendered fixed-frequency carrier the way AM is. OOK with a unipolar (0/1) message
needs `suppressed=True`: `(1 + depth*message)` never reaches 0 for a message that only goes
0..1, while `depth*message` does.

The instrument/bus additions compose the same way:

```python
g2 = ws.Grid(fs=1e9, n=8192)
x = (ws.Signal(seed=1, grid=g2)
     .carrier("nrz", n_ui=64)
     .open_drain(r_pullup_ohm=1000.0, c_bus_f=200e-12, v_dd=3.3, r_sink_ohm=20.0,
                 second=dict(kind="nrz", n_ui=64, seed=9), r_sink_b_ohm=90.0)  # wired-AND 2nd sink
     .burst(t_on_s=2e-6, t_off_s=6e-6, off="hold")           # gate any carrier on/off
     .pass_fet(gate=dict(kind="square", f_hz=1e6), rds_on_ohm=8.0,
               r_load_ohm=1e4, v_rail_hi=5.0, v_rail_lo=-5.0)  # off = high-Z / body diode
     .events("clamped_exp", on="times", times=[1e-6], amp=1.5, tau_s=5e-9, v_clamp=1.0)
     .waveform())
```

`r_sink_ohm` (default 20.0) is the FIRST device's on-resistance, shown explicitly here — it is
easy to reach for `r_sink_a_ohm` by symmetry with `second`'s `r_sink_b_ohm` and get a
did-you-mean error instead. The upstream carrier's absolute LEVEL never decides sinking: the
line is thresholded at the MIDPOINT of that carrier's own excursion (`open_drain`'s own
docstring), so a `dc` carrier — any level, any sign — straddles nothing and never sinks; only a
carrier that actually toggles (`nrz`, `square`, `cmos`, ...) drives the bus.

None of these add a protocol name: a wired-AND second sink is `open_drain(second=...)`, not
`i2c()`; `clamped_exp` is the generic rail-transient shape, and which standard's pulse numbers
apply (amplitude, tau, clamp voltage) is the caller's own registry — the same rule a PRBS
polynomial already follows.

## A shared bus, and the protocol layer you write above it

`open_drain` is a one-line bus. When several parties share one net and hand it over —
turnaround, contention, a weaker driver holding a level, a switched termination, a level
translator — use `Signal.net()`. It is a channel-rank op that solves a whole lumped R/C
network in closed form between switching events, so there is no step size and no accuracy that
decays over a long record.

Writing its event schedule by hand does not scale past one net. `wfmsynth.scenario` is the
layer that produces it:

```python
from wfmsynth.scenario import Hold, Agent, merge_holds, hold_windows

holds  = [Hold("host", 1e-6, 2.5e-6, field="addr", bit=0), ...]   # who was on the net, when
agents = {"host": Agent(during={"pad_h": "driving"},              # what being on it DOES
                        after={"pad_h": "hiz"}, lead=20e-9, lag=60e-9)}
x = sig.net(**bus, holds=[h.to_dict() for h in holds],
            agents={k: a.to_dict() for k, a in agents.items()}, probe="bus").waveform()
labels = hold_windows(holds, fs, n)        # -> events.label_windows, carrying `who` + tags
```

The division of labour is the point, and it is the same boundary the pattern registry draws:

```
bytes --> [ YOUR protocol encoder ] --> holds --> [ scenario ] --> events --> [ net ] --> volts
                  caller code                       library           library
```

The encoder knows start conditions and acknowledge slots and nothing about volts. The agents
know about output stages and nothing about the protocol. **Do not look for `i2c()`, `spi()` or
`can()` — they are deliberately absent**, like `nrz_prbs31_per_annex_C()`: which bits a standard
puts on a wire has a revision date and a document behind it. Write the ~40-line encoder;
`examples/i2c_bus.py` is a complete worked one (I2C across a 3.3 V/1.8 V MOSFET level
translator) and is the file to copy from.

A driver's Thevenin level may itself be a waveform — its own supply, edge rate and timing
impairments — injected through its own scheduled output conductance:

```python
drivers=[{"id": "ctrl", "node": "bus", "ohm": 20.0, "state": "hiz",
          "source": {"ops": Signal(grid=g).carrier("cmos", v_lo=0.0, v_hi=3.3,
                                                   f_hz=20e6, tr_s=2e-9)
                                          .timing(rj_ps=12.0),
                     "volts": True}},                       # already volts
         {"id": "n2", "node": "bus", "ohm": 55.0, "state": "hiz",
          "source": {"kind": "nrz", "n_ui": 328, "seed": 37,
                     "volts": [0.0, 1.8]}}]                 # map ±1 onto these volts
```

A body is exactly one of `from="chain"` (the waveform arriving at the op), `ops=` (its own op
list, or a `Signal`), or a carrier spec by `kind=`. With every driver carrying one, the net can
be the FIRST op — nothing has to arrive at it. `examples/multi_driver_bus.py` is the worked one.

Things that bite:

- **`volts` is required on a driver source, and it is the trap.** A net is in volts; a carrier
  returns normalised ±1 (rule 2). `volts=True` asserts the chain is already volts, `[lo, hi]`
  maps ±1 onto those two. Omitted, it refuses — because a source passed through unconverted is
  a driver a thousand times too weak and the solve would be perfectly correct about it.
- **A released driver contributes nothing, whatever its chain is doing.** It is out of the
  stamp, so its level is absent rather than small. That is the whole reason to put the sources
  inside the solve: summing each party's contribution by hand leaves the level it was holding
  in the sum, and the baseline drifts by one held offset at every change of hands.
- **Size the grid from the fastest MODE on the board, not from the bit rate.** A 60 ohm switch
  between two bus capacitances is a nanosecond-scale redistribution on a microsecond-scale bus.
  The solver warns when a mode finishes inside one sample — the levels stay exact, but an edge
  measured off that record is the grid's.
- **Merging is required.** Consecutive identical bits abut, so without `merge_holds` the bus
  releases and re-acquires at every bit boundary.
- **The holds are the labels.** A fault is an edit to the holds, which makes the edit its own
  label — there is no second list to keep in step.
- **Hand-overs need no rule.** Two agents holding at once are two parallel drivers; events on
  one sample are applied as one configuration change, so there is no moment with nobody driving.
- Lumped R, L and C only in this tier, and a driver's SOURCE may be a waveform but its output
  CONDUCTANCE may not (`gate`, `relax` are refused by name). Transmission lines, S-parameter
  blocks inside a net, sub-nets and voltage-dependent laws are deferred — `BACKLOG.md` has the
  order.

An inductance is what makes a hand-over RING rather than settle, which is the one place a
synthetic record differed from a measured one by more than noise:

```python
elements=[{"id": "lp", "kind": "l", "a": "pad", "b": "bus", "henry": 8e-9, "ohm": 0.3},
          {"id": "cp", "kind": "c", "a": "pad", "b": "gnd", "farad": 2e-12}, ...]
out = ws.net.solve(spec, fs=fs, n=n, schedule=sched, probes=["bus", "i(lp)"],
                   v0={"pad": 0.0, "bus": 0.0, "i(lp)": 0.0})    # i(...) is AMPS
```

- **`ohm` on an inductor is the series resistance of the same wire**, not a parallel resistor
  (absent means ideal). One element with two numbers, because the node between a series R and a
  series L has no capacitance and this tier solves states — so writing them as two elements is
  refused with that explanation.
- **An inductor is not switchable**, nor is a capacitor, and the schedule refuses both by name.
  A current and a charge are the states carried across every switch; removing either from the
  stamp would zero it within one sample. Switch what is in SERIES with it, and the node
  capacitance takes the current — which is what rings. (Both used to render as though the event
  had not been written.)
- **A ring the grid cannot show ALIASES**, unlike a decay it cannot show, which is simply
  absent. Below 8 samples per ring period the solver says so and past Nyquist it says the
  record's ring is not the circuit's. The solve is exact at every damping including exactly
  critical (3.5e-14 V against the closed form; a modal solve of that case returns 368 mV of
  nonsense, which is why the inductive path is a Schur form).
- **A line** is reachable as `net.line_ladder(a, b, z0_ohm, delay_s, tr_s)`, a lumped ladder
  sized by a measured rule — `ceil(6·(delay/tr)**1.5)` sections for ~2 % of the swing. It is the
  one approximation in the module, it refuses a line longer than ~12 rise times, and a real line
  still belongs to `lossy`/`sparam`/`cascade`. `examples/inductive_handover.py` is the worked one.

## Chains that meet

Each block below starts from these lines.

```python
import json, numpy as np, wfmsynth as ws
from wfmsynth import Signal

g = ws.Grid(fs=64e9, baud=1e9, n=1 << 16)            # 64 samples/UI, 1024 UI
N_UI = 1024

def data(seed=3):                                    # a fresh Signal per call
    return Signal(seed=1, grid=g).carrier("nrz", n_ui=N_UI, pattern="prbs7", seed=seed)
```

A builder method appends to the Signal it is called on and returns it, so reusing one base
builds ONE chain. `base.copy()` is an independent Signal with the same recipe and samples:
`short, long = base.copy().lossy(loss_db=3, loss_at_ghz=8), base.copy().lossy(loss_db=12, loss_at_ghz=8)`.

### A sub-chain: three ways to write one, and how to reach inside

```python
lane = Signal().carrier("nrz", n_ui=N_UI, seed=9).timing(rj_ps=1.0).lossy(loss_db=6.0, loss_at_ghz=0.5)

a = data().crosstalk(aggressor=lane, coupling=0.1)                       # a Signal: part of this record
b = data().crosstalk(aggressor={"ops": lane.ops}, coupling=0.1)          # the same thing, as data
assert a.sha256() == b.sha256()

standalone = Signal(seed=5, grid=g).carrier("nrz", n_ui=N_UI, seed=9).timing(rj_ps=1.0)
c = data().crosstalk(aggressor={"from": standalone.recipe()}, coupling=0.1)   # another record, pinned
assert np.array_equal(c.subchain("crosstalk/1/aggressor"), standalone.waveform())

print(a.summary())                                   # the tree; [crosstalk/1/aggressor] is the namespace
assert a.roles() == ["crosstalk/1/aggressor/timing/1"]
used = a.subchain("crosstalk/1/aggressor")           # the aggressor exactly as it was coupled
moved = a.contrast("crosstalk/1/aggressor/timing/1") # re-roll ONLY the aggressor's jitter
assert not np.array_equal(moved, a.waveform())
assert Signal.from_recipe(json.loads(a.to_json())).sha256() == a.sha256()
```

`roles()` is exactly what the record draws from: a role is listed only when the op's
parameters make it draw (`store` with `dither_lsb`, `acquire` with a noise floor, a carrier
with `rj`), so every listed name moves samples when re-rolled. A `{"from": recipe}` record is
pinned, unit included: `unit()` does not move its toleranced parts and `roles()` lists only
its noise.

A sub-chain must land on the record's own grid as a real waveform, or it is refused: nothing is
resampled. Parameters that take one: `crosstalk(aggressor=)`,
`crosstalk_matrix(couplings=[...], aggressors=[...])`, `modulate(message=)`,
`open_drain(second=)`, `hybrid_echo(own=)`, `pass_fet(gate=)`, `trigger(on=)`, a `net` driver's
`source`, `combine`, `by=`, `sequence` segments and `switch` inputs. A record with a lead-in
cannot hold one.

### combine, and a parameter driven by a signal

```python
def tone(**k):
    return Signal().carrier("sine", **k)

data().combine(tone(f_hz=133e6, amp=0.03))                        # x + other
data().combine([tone(f_hz=1e6), tone(f_hz=3e6)], gains=[0.1, 0.05])
data().combine(tone(f_hz=5e9), how="product")                     # also "min", "max"
data().gain(by=tone(f_hz=700e3, amp=0.04, offset=1.0))            # x * by: by is the factor itself
data().offset(by=0.1)                                             # x + by; a number or a chain
gated, ev = data().gate(by=tone(f_hz=2e6), threshold=0.5, off="zero").realize()   # off="hold" too
[e.t_s for e in ev if e.kind == "transition"]                     # when the enable crossed
late = data().delay(by=300e-12)                                   # seconds; >= 0 everywhere
late.delay_errors()                                               # [{'op': 1, 'error': 2.31e-05, 'below_hz': 0.4*fs, ...}]
```

`delay(by=<chain>)` is a moving delay (Doppler, a wobbling path): causal, band-limited, error
stated per tone below 0.4 fs. Its rate of change per sample must stay under `max_rate` (0.5), so
white jitter as `by` is refused; filter it in its sub-chain, or use `timing` for jitter about
zero. A gate with a noisy enable chatters: end the enable's sub-chain in
`law("comparator", hysteresis=)`.

### sequence and switch

```python
tx = (Signal(seed=1, grid=g)
      .sequence([{"chain": Signal().carrier("dc", level=-1.0), "duration_s": 64e-9},
                 {"chain": Signal().carrier("nrz", n_ui=N_UI, pattern="clock"), "duration_s": 448e-9,
                  "state": "continue"},                           # as if running since t = 0
                 {"chain": Signal().carrier("nrz", n_ui=N_UI, pattern="prbs7"),
                  "state": "continue"}],                          # last one takes the rest
                join="crossfade", t_join_s=1e-9)                  # or "hold"; default "cut"
      .lossy(loss_db=6.0, loss_at_ghz=0.5))                       # ops after it act on the joined record
x, ev = tx.realize()
assert [e.sample for e in ev if e.kind == "segment"] == [4096, 32768]
plan = ws.sequence.layout(tx.ops[0]["segments"], g)               # [{'segment', 'start', 'stop', ...}]

mux = Signal(seed=1, grid=g).switch(select=Signal().carrier("square", f_hz=2e6),
                                    inputs=[Signal().carrier("dc", level=0.0),
                                            Signal().carrier("nrz", n_ui=N_UI)],
                                    threshold=0.0, t_switch_s=1e-9)   # N inputs take N-1 thresholds
```

Both are sources: first in the chain. Durations are whole samples and must fill the record; an
off-grid `duration_s` is refused naming the two nearest that fit. The default `"state": "restart"`
begins the chain at its own t = 0 (turn-on included). Nothing is carried from one segment's
chain into the next; to share a filter put it after `sequence`.

### Several channels from one recipe

```python
def clock():
    return Signal().carrier("nrz", n_ui=N_UI, pattern="clock")

rec = ws.Channels(
    {"clk":  Signal().use("clk_line").rx_noise(rms=0.01).use("scope"),
     "data": (Signal().carrier("nrz", n_ui=N_UI, pattern="prbs7").use("pll")
              .lossy(loss_db=6.0, loss_at_ghz=0.5).rx_noise(rms=0.01).use("scope"))},
    shared={"pll": Signal().timing(phase_noise=dict(rms_ps=2.0)),      # a STAGE: applied
            "clk_line": clock().use("pll"),                            # a SIGNAL: starts with a source
            "scope": Signal().trigger(on=Signal().use("clk_line"), pre_s=1e-9, post_s=3e-9,
                                      level=0.0, slope="rising", n_segments=16, holdoff_s=20e-9)},
    seed=1, grid=g)

x, events = rec.realize()                             # {"clk": array, "data": array}, {name: EventList}
assert x["clk"].shape == x["data"].shape == (16 * 256,)          # the same 16 windows on both
assert rec.roles() == ["clk/rx_noise/1", "data/rx_noise/3", "pll/timing/0"]
other_pll = rec.contrast("pll/timing/0")              # every channel that uses the PLL moves
assert ws.Channels.from_recipe(json.loads(rec.to_json())).sha256() == rec.sha256()
print(rec.summary())
```

Chains are written `Signal()` with no seed and no grid; the record has one of each.
`use(name)` first in a chain (or inside any signal-taking parameter) IS that chain's waveform;
after a source it APPLIES the named stage. A chain draws under its own name wherever it is used,
so two channels through `pll` see one realisation, and the same chain under two names is two.
The trigger above is on `clk_line`, not `clk`: `clk` uses `scope`, and a chain cannot reach
itself (a cycle is refused, naming it). `rec.waveform(name)`, `rec.stored_grid(name, x)`,
`rec.unit(seed)` and `rec.write_hdf5(path)` are the per-record counterparts of the `Signal`
methods; `rec.signal(name)` is one channel as a `Signal` that renders exactly as it does in the
record, for `transmitted()`, `rf_report()`, `subchain()` and the rest. A differential pair is `{"p": chain, "n": Signal().use("p").law("poly", coeffs=[0.0, -1.0])}`.

### law and filter

```python
out = (Signal(seed=1, grid=ws.Grid(fs=1e6, n=50000))
       .carrier("sine", f_hz=200.0).rx_noise(rms=0.05)
       .filter("butter", f_hz=5e3, order=2)                       # causal: it delays what follows
       .law("comparator", threshold=0.0, hysteresis=0.3, v_lo=0.0, v_hi=3.3))
x, ev = out.realize()
edges = [e for e in ev if e.kind == "transition"]     # e.t_s is the sub-sample crossing, e.direction +-1
out.filter_delays()                                   # [{'op': 2, 'kind': 'butter', 'group_delay_s': 4.5e-05, ...}]
```

| `law(kind, ...)` | Parameters |
|---|---|
| `tanh` | `gain=1, sat=1` |
| `poly` | `coeffs=[c0, c1, ...]` ascending |
| `limiter` | `lo`, `hi` (at least one) |
| `deadzone` | `width, center=0` |
| `rectify` | `mode="half"` or `"full"` |
| `quantize` | `bits, lo, hi`; saturates; top code is `hi` less one step |
| `comparator` | `threshold=0, hysteresis=0, v_lo=0, v_hi=1, initial=None` |
| `diode` | `r_load_ohm, i_sat=1e-12, ideality=1`; needs volts |

| `filter(kind, ...)` | Parameters |
|---|---|
| `butter`, `bessel`, `cheby1`, `elliptic` | `f_hz, order=4, btype="low"`; `[lo, hi]` for `"bandpass"`/`"bandstop"`; `ripple_db`, `stop_db` |
| `rrc` | `beta, span_ui=16`; unit DC gain; the `iq` source's matched filter |
| `fir`, `sos` | your `taps` / digital `sos` for this grid |
| `zpk` | analog `poles`, `zeros` in rad/s (a 1 MHz pole is `-2*pi*1e6`), `gain` |
| `integrate`, `differentiate` | `tau_s=1` |

`f_hz` is the -3 dB point for `butter` and `bessel` but the passband EDGE for `cheby1` and
`elliptic`. Every filter takes `zero_phase=True` except `integrate`/`differentiate`.
`filter_delays()` gives each filter's `group_delay_s` (what a feature slow against the filter
is delayed by) and `step_delay_s` (the midpoint of an edge fast against it: 22.1 against 31.8
samples for one pole at 5 MHz and 1 GSa/s); correct an upstream event by the one that matches
its edge. It covers `filter` ops only; `scope`, `probe` and `lossy` have delays of their own. A chain holding `law`, `filter`, a join or `use` cannot be `canonical()`-sorted: its order
is its meaning.

### A triggered acquisition

```python
shot = (data().lossy(loss_db=3.0, loss_at_ghz=0.5).rx_noise(rms=0.01)
        .trigger(pre_s=1e-9, post_s=3e-9,             # required; the window about each firing
                 condition="edge",                    # or "width" (width_s=[min, max]), "runt" (runt_level=)
                 level=0.0, slope="rising",           # level=None is "set to 50 %" of THIS record
                 hysteresis=0.0, holdoff_s=20e-9, jitter_s=0.0, n_segments=8, average=1))
x, ev = shot.realize()
rows = x.reshape(8, -1)                               # segments are end to end in one 1-D record
stored = shot.stored_grid(x)                          # n = 2048; shot.grid is still the long render
for e in ev:                                          # one Event(kind="trigger") per firing
    e.t_s, e.params["segment"], e.params["timestamp_s"], e.params["jitter_s"]
```

Render long, then cut: the trigger acts on the realised waveform, so where it fires is known only
after the chain has run. Nothing fires before `pre_s`, and it re-arms after
`max(holdoff_s, pre_s + post_s)`. Too few firings is refused with the render length that would
hold them. Put it after `acquire` to trigger at the acquisition rate. With `n_segments > 1` it
must be the last op. Width and runt triggers fire on the pulse's TRAILING edge.

An event of the long render that falls outside every window is no longer on the record and is
DROPPED from the list; each trigger event's `params["dropped"]` says how many, counted over the
whole record (the same number on every trigger event). To put a window AT a known instant,
trigger on a step placed there:

```python
t0 = sig.event_list().events[0].t_s                   # the instant to look at
win = sig.trigger(on=Signal().carrier("step", t_step_s=t0), pre_s=1e-9, post_s=3e-9)
```

`trigger(level=)` is a scope's word; `gate`, `switch` and `law("comparator")` call theirs
`threshold=`, and each refusal names the other. Their defaults differ on purpose: a trigger
with no level sets itself to 50 % of the record, a comparator compares against 0 V, and a
gate or switch separates an enable's two states at 0.5 (right for 0/1 and for -1/+1).

### Toleranced parts

```python
def unit_under_test(seed):
    return (Signal(seed=seed, grid=ws.Grid(fs=100e9, n=8000))
            .carrier("cmos", v_lo=0.0, v_hi=3.3, f_hz=25e6, tr_s=1e-9)
            .probe(r_source=ws.tol(1e3, 5),                        # +/-5 %, uniform
                   c_load_f=ws.tol(2e-12, 10, dist="normal"))      # 10 % = 3 sigma; "corner" = an end
            .digitize(bits=ws.corners({"coarse": 6, "fine": 10}))  # a whole number takes corners
            .rx_noise(rms=0.005))

s = unit_under_test(3)
r = s.recipe()
ws.op_params(r, "probe")["r_source"]                  # 1019.1...: the resolved number, in its own place
ws.tolerance.resolved(r)                              # [{'role': 'unit/1/r_source', 'value': ..., 'nom': ...}, ...]
s.unit(8)                                             # a sibling Signal: another unit, the SAME noise
s.contrast("unit/*", seed=8)                          # its waveform
X, recipes = ws.unit_dataset(lambda rng: unit_under_test(int(rng.integers(1 << 30))),
                             n_units=6, n_captures=4, seed=1)
group = np.array([ws.unit_of(q) for q in recipes])    # the key for a held-out-unit split
```

`ws.tol(nom, tol_pct)` or `ws.tol(nom, tol_abs=)` (a percentage of a zero nominal is refused).
`group="board"` makes several parameters one draw, in one chain or across the chains of a
`Channels` record. A hand-written recipe may carry `{"nom": 1e3, "tol_pct": 5}` where a number
goes; `from_recipe` draws it under the recipe's seed. A COUNT (`bits`, `n_ui`, `order`,
`n_segments`, ...) takes `ws.corners`: `ws.tol` on one is refused, because the fraction it
draws would be truncated. `rec.contrast(role)` takes every role `rec.roles()` lists, a part's
included, on a `Signal` and a `Channels` alike; a name that matches nothing is refused.

Split a unit dataset on `group`, not at random. MEASURED: 1.7 ohm error on a random split
against 31.3 ohm on held-out units, where answering the nominal scores 30.2 -- the random
split had captures of every unit on both sides.

### A segmented acquisition of a rare event

```python
closure = (Signal().carrier("step", t_step_s=1e-6, tr_s=20e-9, amp=0.5, offset=0.5)
           .gain(by=ws.tol(1.0, 10, per="segment"))         # drawn again per occurrence: vary/1/by
           .gain(by=ws.trend("ramp", start=1.0, rate_per_s=-2e-3))  # or stop=; "step" with at= / at_s=
           .gain(by=ws.tol(1.0, 5))                         # a unit part: one draw, variants included
           .rx_noise(rms=0.01))                             # each occurrence's own: segment/<k>/rx_noise/4
seg = ws.Segments(closure, n=100, seed=1,
                  grid=ws.Grid(fs=1e9, n=3000),             # ONE occurrence's render, not the span
                  arrival=dict(dist="lognormal", median_s=0.2, sigma=0.8),  # a states dwell; "fixed" = period
                  trigger=dict(pre_s=200e-9, post_s=1.5e-6, level=0.5, jitter_s=0.5e-9),
                  variants={"late": dict(p=0.05, chain=closure.copy().delay(by=300e-9))})
x, ev = seg.realize()                                 # x.reshape(100, -1); trigger events as trigger(n_segments=)
seg.stored_grid(x)                                    # the windows' Grid, as Signal.stored_grid after trigger
seg.occurrences()[0]                                  # arrival_s, missed, variant, values{location: number}
ws.segment_params(seg.recipe(), "gain", which=0)      # op_params, one dict per occurrence
seg.roles()                                           # arrival, variant, vary/..., trigger, segment/<k>/..., unit/...
seg.without("trend/*")                                # the cause removed; draws, arrivals, noise kept
seg.unit(8)                                           # another unit (tol without per=), same occurrences
ws.Segments.from_recipe(seg.recipe())                 # bit for bit
```

Each occurrence starts at its arrival and is cut where the trigger first fires after `pre_s`;
its source is delayed by the fraction of a sample the arrival falls after one, so placement
spreads as on a free-running scope. The scope is busy for `post_s` plus `rearm_s=` (in the
trigger dict, default 0) after each trigger; an arrival then is missed and counted, and the
render's length changes no arrival. A variant is the same unit: the variant above, a copy of
`closure`, holds the same `unit/3/by` value as the main chain; its `per="segment"` draws are
its own (`late/vary/1/by`). Refused by name: `level=None`, a bare-number `arrival`, `holdoff_s`
/ `average` / `n_segments` / `on` in `trigger`, a render too short for the window (with the
length that would do), a trigger that fires only before `pre_s` (move the event later), a
variant stored at another rate, and a `per="segment"` value or a trend in a chain rendered on
its own.

## I/Q and RF

```python
import os, tempfile, numpy as np, wfmsynth as ws
from wfmsynth import Signal, rf

g = ws.Grid(fs=6.4e9, baud=25e6, n=1 << 18, fc_hz=1e9)   # the centre; the iq source makes it I/Q

src = Signal(seed=1, grid=g).iq("qam16", prbs=15, prbs_phase=100)   # or iq(symbols=[1+0j, 1j, ...])
# ONE source: prbs=<order> (prbs_seed= is its starting state, 1 unless stated; prbs_phase= steps
# it that many bits further), bits=[...], or symbols=[...]. Signal(seed=) does NOT move the
# data: a dataset loop varies prbs_seed (or prbs_phase), as carrier(seed=) does for a voltage.

sent, t_s = src.transmitted()                          # ground truth: complex symbols, centre times

rx = (src.copy()                                       # branch: `src` itself stays the source
      .upconvert()                                     # iq -> real passband volts, same grid
      .law("tanh", sat=2.5)                            # an amplifier acts on the voltage
      .probe(bw_hz=3e9)                                # every real op applies here
      .downconvert(lpf=dict(pass_hz=40e6, stop_hz=120e6, atten_db=80), fs_out_hz=400e6))   # fc_hz: the grid's
y = rx.waveform()                                      # complex128
gy = rx.stored_grid(y)                                 # 400 MSa/s, domain "iq": measure with THIS

raw = rf.evm(y, gy, sent, skip=8)                      # nothing fitted: 26.7 %
fit = rf.evm(y, gy, sent, skip=8, correct=("gain", "phase"))      # 1.8 %
fit["removed"]                                         # {'gain_db': -0.62, 'phase_rad': -0.267}
rf.aclr(y, gy, channel_bw_hz=33.75e6, offsets_hz=[40e6])["aclr_db"]    # [42.4]
rf.ccdf(y[1024:-1024])["papr_db"]                      # 4.96; slice off the burst's turn-on
spec = rf.spectrum(y, gy, rbw_hz=1e6)                  # {'f_hz', 'psd_per_hz', 'rbw_hz', ...}
rf.band_power(spec, -16.875e6, 16.875e6)
```

`upconvert` and `downconvert` never resample: the grid must hold the carrier with the envelope's
bandwidth clear of DC and Nyquist (refused otherwise, stating the folded power), and `lpf` has no
default (`lpf=dict(pass_hz=)`; `stop_hz` defaults to twice it, `atten_db` to 80). `fs_out_hz` is
whole-number decimation only. A record's domain follows its source: `iq(...)` or an I/Q capture
makes it complex baseband without `Grid(domain="iq")`, and a grid that states `domain="real"`
refuses them. `rf.evm` corrections, each returned in `removed`: `"timing"` (within half a
symbol of `delay_s`), `"frequency"`, `"offset"`, `"gain"`, `"phase"`, `"equalizer"` (with
`equalizer_taps=`). `skip=` drops symbols at each end, which lack their neighbours' tails.
With `reference="decided"` the carrier is acquired blind before any decision (the line in the
4th power of the symbols for QPSK and QAM): an offset within +/- baud/8 is found at any phase,
and a record it cannot lock to is refused instead of measured. Past that range the estimate
is an alias (it repeats every baud/4): `["acquisition"]` gives `pull_in_hz` and
`alias_spacing_hz`, and passing `sent` makes a wrong alias a refusal. Measured, QAM-16 2 kHz off at
Es/N0 24 dB: 6.29 % decided against 6.29 % truth.
`rf.evm` is THE EVM of a record: it applies the RRC matched filter itself and returns percent, dB
and a fraction. `ws.iq.evm(received, sent)` is only its last step, a fraction, for symbols you
already hold.

```python
gi = ws.Grid(fs=400e6, baud=25e6, n=1 << 16, fc_hz=1e9)

tx = (Signal(seed=1, grid=gi).iq("qam16", prbs=15, prbs_phase=100)
      .dac(bits=8, headroom_pct=5)                     # or full_scale=; clip="hard" to allow clipping
      .iq_imbalance(gain_db=0.3, phase_deg=2.0)        # Q against I, modulator convention
      .iq_offset(i=0.02, q=-0.01)                      # carrier leak, record units
      .phase_noise(profile=[(1e4, -75), (1e5, -95), (1e6, -115)])   # (offset Hz, dBc/Hz)
      .freq_offset(f_hz=2e3)                           # also drift_hz_per_s=
      .spur(f_hz=3.1e6, dbc=-45)
      .awgn(es_n0_db=25, ref_power=1.0))               # or eb_n0_db=, or snr_db= WITH bw_hz=
x = tx.waveform()
tx.rf_report()                                         # what each request came to, per op
phase = tx.lo_phase()                                  # rad per sample: phase_noise + freq_offset
assert tx.roles() == ["phase_noise/4", "awgn/7"]

work = tempfile.mkdtemp()
ws.to_sigmf(os.path.join(work, "rec"), x, gi)                       # float32 by default
again = ws.read_iq(os.path.join(work, "rec"), format="sigmf", baud=25e6)   # a Signal; grid from the file
wrote = ws.to_interleaved(os.path.join(work, "rec.iq"), x, dtype="float32")
raw = ws.read_iq(wrote["path"], format="iq_interleaved", dtype="float32",
                 fs_hz=gi.fs, fc_hz=gi.fc_hz, baud=25e6, embed=True)       # embed: replay needs no file
matched = raw.filter("rrc", beta=0.35, zero_phase=True)             # a recording is a chain like any other
```

The order of the impairments is physical and is kept: these are `general` stages. All seven
refuse a real record. `eb_n0_db` needs the source's constellation (or `bits_per_symbol=`).
`awgn`'s noise is white over the whole sampled band: `snr_db` is in `bw_hz`, and the SNR of the
stored record is 10*log10(fs/bw_hz) lower. An I/Q file's byte order is checked on read; its
`scale`, and a wrong dtype of the same width, cannot be, since every scale gives a
self-consistent record.
`ws.iqfile.describe(path, format, ...)` reports a file's length, peak counts and full-scale
fraction without building anything. A file states no symbol rate; pass `baud=` or a matched
filter has nothing to read. On an envelope `gain(by=)` (real, or complex: a number or an I/Q
chain), `gate(by=)` (real `by`), `combine` (sum, I/Q peers), `sequence`, `switch`, `filter`,
`use`, `delay(by=)` (needs `Grid(fc_hz=)`: it turns the carrier too), `lossy` / `sparam` (read
at `fc_hz + f`) and `law("saleh" | "rapp")` work; `offset`, the voltage laws and `trigger` are
refused and name `iq_offset`, the passband record or `dac`, and a voltage.

### OFDM, a channel and an amplifier on the envelope

```python
go = ws.Grid(fs=20e6, n=1 << 16, fc_hz=2.4e9)          # fs = n_fft * scs_hz, or a WHOLE multiple
NUM = dict(n_fft=64, scs_hz=312.5e3, cp=16, active=52, pilots=[-21, -7, 7, 21])   # cp in FFT-rate samples

def ofdm():                                            # ALWAYS state prbs_seed (see SKILL #36)
    return Signal(seed=1, grid=go).ofdm(constellation="qam16", prbs=15, prbs_seed=0x2F6E, **NUM)

par = ofdm().ops[0]                                    # the receiver takes the op's own parameters
grid_sent, t_useful = ofdm().transmitted()             # [symbol, subcarrier] (819 x 52), each FFT's start
plan = ws.ofdm.plan(par, go)
bw, ref = plan["noise_bw_hz"], plan["ref_power"]       # 16.25 MHz; amp**2. An OFDM grid has no baud
# awgn(snr_db=s, bw_hz=bw, ref_power=ref) is s dB per resource element, burst or not

TWT = dict(alpha_a=2.1587, beta_a=1.1517, alpha_phi_rad=4.0033, beta_phi=9.1040)
rx = (ws.recipes.multipath(ofdm(), delays_s=[0.0, 150e-9], gains=[1.0, 0.5])   # delay + gain + combine
      .gain(by=0.1).law("saleh", **TWT)                # 20 dB back-off, AM/AM + AM/PM
      .awgn(snr_db=30, bw_hz=bw))                      # against the record's own power here
y = rx.waveform()
fit = rf.evm(y, go, grid_sent, ofdm=par, correct=("common_phase", "gain", "equalizer"))
fit["evm_pct"]                                         # 7.1 %; without "equalizer" 47 %
fit["removed"]["equalizer"]                            # the channel, one complex number per subcarrier
fit["evm_per_subcarrier"], fit["evm_per_symbol"]       # where the error is
# symbols=<bool mask per OFDM symbol> counts only those; block=<symbols> refits per block (a fade)

# a transmitter that DRAWS (noise, a fade of its own) is a named chain, the same on every path:
noisy = ws.Channels(
    {"rx": ws.recipes.multipath(Signal().use("tx"), delays_s=[0.0, 150e-9], gains=[1.0, 0.5])
           .gain(by=0.1).law("saleh", **TWT)},
    shared={"tx": Signal().ofdm(constellation="qam16", prbs=15, prbs_seed=0x2F6E, **NUM)
                  .awgn(snr_db=30, bw_hz=bw, ref_power=ref)}, seed=1, grid=go)

fade = ws.recipes.fading(doppler_hz=200.0)             # unit mean power; k_db= adds a line of sight
# sig.gain(by=fade): the record must hold 4 Doppler periods or more (refused under that)
```

`rf.evm(..., ofdm=)` corrections: `"timing"`, `"common_phase"` (per symbol), `"gain"`,
`"equalizer"`; the single-carrier names are refused. A frame is a `sequence` of `ofdm` chains;
a burst schedule is `gate(by=Signal().carrier("states", levels=[1.0, 0.0], dwell=[t_on,
t_off]))` -- `dwell[i]` belongs to `levels[i]`, and the DEFAULT levels are `[0.0, 1.0]`, off
first -- and then EVM is taken over the symbols inside an on interval:
`rf.evm(..., ofdm=par, symbols=mask)` with the mask built from
`ws.states.intervals(events, at="gate/<i>/by")` (COOKBOOK, "OFDM"). In a `Channels` record a
chain brought in by `use` keeps its name in front (`at="tx/gate/1/by"`); an `at` nothing is
at is refused with the list of those that are. An `ofdm_symbol` event's useful part starts at
`t_s + params["cp_s"]`; `complete: False` marks one a segment cut. `ws.ofdm.register(name,
source=, **numbers)` names a parameter set; the recipe keeps the numbers. `multipath`'s
transmitter is rendered once per path, so one that draws anything (after a `use` as well) is
refused: make the whole transmitter a named chain of a `Channels` record and pass
`Signal().use("tx")`, as above. Ops after `use` are checked when the record renders.

## Sources that dwell, and shipped compositions (`ws.recipes`)

```python
import numpy as np, wfmsynth as ws
from wfmsynth import Signal
R = ws.recipes
g = ws.Grid(fs=1e6, n=1 << 20)

busy = Signal().carrier("states", dwell=[800e-6, 300e-6])        # dwell[i] is levels[i]'s mean: 0 (off) 0.8 ms, 1 (on) 0.3 ms
sig = Signal(seed=1, grid=g).carrier("sine", f_hz=10e3).gate(by=busy)
x, events = sig.realize()
spans = ws.states.intervals(events, at="gate/1/by")    # [{t_start_s, t_stop_s, state, level, complete}, ...]
ws.states.occupancy([800e-6, 300e-6])                  # [0.727 0.273]
# levels=[...], dwell={"dist": "lognormal"|"uniform"|"fixed"|"table"|"exponential", ...} per state,
# transitions=[[...]] (zero diagonal, rows sum to 1), start="stationary"|<index>, tr_s=<10-90 % s>

rail = (Signal(seed=1, grid=g).carrier("dc", level=3.3)
        .combine([R.hum(50.0, amp=0.02, harmonics={3: 0.3}),                 # each is a visible sub-chain
                  R.ripple(2e3, amp=0.01, spike=0.03, f_ring_hz=50e3),
                  R.pickup(200e3, amp=0.005),
                  R.flicker(10.0, 1e4, rms=0.003)]))
print(rail.summary())                                  # every op of every one
closing = Signal(seed=1, grid=g).carrier("dc", level=3.3).gate(by=R.bounce(t_s=2e-3, gap_s=400e-6))
osc = R.oscillator_startup(100e3, startup_s=8e-3, sig=Signal(seed=1, grid=g))   # sig=: it IS the record
```

`ws.recipes` also has `intermittent`, `multitone` (+ `multitone_plan`), `buck` (+ `buck_ripple`),
`fading`, `multipath`, and the photonic `mzi`, `stray_reflection`, `ring`, `gas_cell`. Each
converts its arguments, so each takes numbers: `ws.tol(...)` goes on the op's own parameter in
the chain written out.

## The optical field, and a swept-wavelength bench

```python
gs = ws.Grid(fs=1e6, n=1 << 17)                        # 0.131 s
bench = ws.Channels(
    {"dut": R.ring(Signal().use("laser"), wavelength_nm=1550.0, q=1e5, extinction_db=15.0)
            .photodetect(shot=False),                  # shot=False: photodetect() draws shot noise by default
     "ref": R.mzi(Signal().use("laser"), delay_s=2e-9).photodetect(shot=False)},
    shared={"laser": Signal().laser(start_nm=1549.0, stop_nm=1551.0)},   # also bow_nm=, ripple_nm=, hops=[...]
    seed=1, grid=gs)
w = bench.waveforms()
truth = ws.swept.wavelength_map(bench.signal("dut"))["wavelength_nm"]       # from the recipe
nm = ws.measure.swept_map(w["ref"], gs, delay_s=2e-9, at_s=0.0, wavelength_nm=1549.0)   # from the fringes
dip = ws.measure.resonances(*ws.measure.transmission(w["dut"], nm), level=1.0)[0]
dip["wavelength_nm"], dip["q"], dip["extinction_db"]   # 1550.00000, 99999.98, 15.000
```

`level=` is the record's OFF-RESONANCE baseline: 1.0 here (unit laser, unit detector, no other
loss). On a bench with insertion loss, a responsivity or another power, pass
`transmission(x, nm, power=<the detector's reading with no device>)` or leave `level` out
(the spectrum's median). `level` is not checked against the record: only one it never comes
HALF WAY up to (over 3 dB of loss) is refused, and 1 dB left in with `level=1.0` reads the
width 30 % wide (Q 76000 for 1e5) and the dip 1 dB too deep. `resonator`'s
`intrinsic_hz` and `coupling_hz` are its two loss rates as linewidths in Hz:
`Q = nu0 / (intrinsic_hz + coupling_hz)` (`nu0 = ws.swept.hz(wavelength_nm)`), the dip reaches
`((ki - ke) / (ki + ke))**2`, equal rates are critical coupling. The die a unit drew is
`ws.op_params(bench.unit(k).recipe(), "resonator", chain="dut")`.

A field is `eo()` .. `photodetect()` (or `laser(...)` ..), its own domain: `filter`, `gain`
(real, or a complex NUMBER), `gate`, a sum `combine` (a coupler: coherent), `sequence`,
`switch`, `use`, `fiber`, `edfa`, `optical_mpi` act on it and everything else is refused
naming `photodetect`. Only a `laser`'s field states its carrier, so only there do `delay` and
`resonator` work (and `optical_mpi` is refused for `recipes.stray_reflection`). A toleranced
ring is `resonator(wavelength_nm=ws.tol(...), intrinsic_hz=..., coupling_hz=...)`, and
`bench.unit(k)` another die.

## Writing records out

There is no archive writer in this library — it produces waveforms, and the container is yours.

### HDF5, for one record a bench instrument or any HDF5 reader should open

```python
from wfmsynth import hdf5
span = float(volts.max() - volts.min())      # what THIS record spans, after the channel
hdf5.write_hdf5("record.h5", volts, fs=fs_store, t0=-1e-9, full_scale=span * 1.05)
volts_back, t = hdf5.read_hdf5("record.h5", channel=1)
```

Samples go out as `int16` codes with the scale that inverts them: `volts = code * YInc + YOrg`,
`seconds = index * XInc + XOrg`. Verified independently of the library's own reader: exact to one
half-code, and the time axis to the last digit. Storing floats reads fine in Python and does not
load on an instrument.

Pass `volts`, not `waveform()`'s normalised output. Omit `full_scale` and the window is taken from
the record; pass it when several records must share one vertical scale, and pass a window that
contains them — a narrower one is refused, naming the span it found, rather than clipping the peaks
into a file that opens and looks plausible.

`like="a_capture.h5"` copies the metadata that marks a file as belonging to a particular
instrument out of a capture you already have; `identity_from` reads it. Needs `h5py`.

**If the file is going onto a bench instrument, model that instrument first.** A record exported
straight off the synthesis grid has a full-rate, full-bandwidth, infinite-resolution edge, and
loaded onto a scope it looks like nothing that scope could ever have captured — so anything read
off it about margin or rise time is about the simulation, not the measurement. Put the target
instrument in the chain before exporting, and ask what it actually is:

- **the probe**, if there is one: `probe(bw_hz=, c_load_f=, r_source=, atten=)`. Its input
  capacitance loads the node, which changes the very rise time the link is graded on — a real
  effect, not a correction factor.
- **the analog front end**: `scope(bw_hz=)` at the instrument's bandwidth, or the setting it will
  be used at, which is often narrower than the hardware.
- **the sample rate and record length**: `acquire(AcquisitionProfile(sample_rate_hz=,
  record_length=, input_bandwidth_hz=, enob=, sample_clock_jitter_rms_s=))`. Its rate is the
  instrument's, not the grid's.
- **the converter**: `enob` rather than the headline bit count — effective bits fall with
  bandwidth, and it is the honest figure. `digitize(bits=, enob=)` and `store(bits=)` if you are
  driving the ops directly.

Then pass the STORED rate to the exporter — `write_hdf5(..., fs=fs_store)` — because `XInc` is
what the instrument will use to lay the record out in time, and the synthesis rate would stretch
it. `Signal.grid` still holds the synthesis rate after `acquire()`, so do not read it from there.

The point of the exercise is that the file and a real capture of the same link should be
comparable. If you cannot say which instrument the file represents, it does not represent one.

### Zarr, for a corpus you will train on

Use `zarr` directly, and get the layout right the first time.

1. **Array count is what breaks at scale**, not chunk count. Sharding packs chunks inside an
   array; nothing packs arrays inside a store. A record dimension inside one array keeps working;
   one array per record is fine only while records are few, large and of differing length.
2. **Chunk is the unit of read; shard is the unit of file.** Chunk along time, around a megabyte,
   so a training loop reading windows pays for roughly what it reads. Size shards to bound how
   many files exist.
3. **Consolidated metadata is a cache.** It goes stale silently on any mutation, so nothing that
   validates a store may read it.
4. **Choose fill values the data cannot hold** — not-a-number for floats, the dtype minimum for
   integers, never a boolean. Absence then stays distinguishable from zero, which is what lets a
   store ship with arrays declared and unwritten and filled on arrival.
5. **Digest array values, not files**, in a canonical order and byte order, so recompression or
   resharding is not a corruption alarm.
6. **Key every per-record column by record identity.**

Two more for training specifically. Keep a **group id** per record and split on it, or a defect and
its own matched control land on opposite sides and the model learns the pair. And do not store
per-sample columns you can recompute — a derived column costs the same bytes as the signal and can
disagree with it.

## Checking what you built

Assert the outcome. A mechanism that ran is not an effect that landed.

```python
def link(loss_db):
    return (ws.Signal(seed=3, grid=grid)
            .carrier("nrz", pattern="prbs13", n_ui=n_ui, tr_frac=0.35, causal=True, seed=11)
            .lossy(loss_db=loss_db, loss_at_ghz=8.0, causal=True))

a, b = link(2.0).waveform(), link(16.0).waveform()
assert not np.allclose(a, b)                 # the knob moved the output at all
assert fs_synth * tr_s >= 8                  # the grid resolves the edge you asked for

def centroid(v, fs):                         # a lossy channel lowers the spectral centroid
    f = np.fft.rfftfreq(len(v), 1 / fs)
    pw = np.abs(np.fft.rfft(v)) ** 2
    return float((f * pw).sum() / pw.sum())

assert centroid(b, fs_synth) < centroid(a, fs_synth)
assert ws.Signal.from_recipe(sig.recipe()).sha256() == sig.sha256()
```

`ws.measure`, `ws.eye`, `ws.eye_height` and `ws.recover_and_fold` cover the usual measurements, and
`ws.ground_truth(...)` returns measured eye geometry — its keys are `eye_contour` and `eye_sigma`.
Note that `eye_contour` floors out once the eye closes, so it is not a usable regression target
across a loss sweep that goes past closure; `eye_sigma` stays monotone. `examples/` in the
repository holds runnable end-to-end scripts.
