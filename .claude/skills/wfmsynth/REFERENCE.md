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
grid, not a small offset, and `n_out` is a length. To store at `fs_store` off a grid at `fs_synth`
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
sample when the clock would need data past the end of the render. Passing a real `ppm` of 55 with
an `n_out` computed from the rate ratio does **not** resample: it truncates the record to that many
samples at the grid rate, and 49 % of it disappears with no warning.

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
