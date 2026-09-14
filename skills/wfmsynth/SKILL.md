---
name: wfmsynth
description: Synthesise oscilloscope-realistic waveforms with known ground truth using the wfmsynth library — compose impairment chains, size a record from its edge, model the acquisition instrument, and emit replayable recipes with content digests. Use when asked to generate or extend synthetic signal-integrity data, build training sets with defect labels, model a link or an instrument, or reproduce a waveform from a recipe.
version: 2
---

# wfmsynth

A waveform is an ordered list of operations. Each op owns one physical effect and is a function
from a waveform to a waveform, so a record is their composition and the list of ops is data.

Everything below is executable against the library as it stands. Run the snippets before trusting
a claim you make on top of them.

## Get it

```bash
git clone https://github.com/m-holloway/wfmsynth.git
cd wfmsynth && pip install -e .          # or: export PYTHONPATH=$PWD
python -c "import wfmsynth; print(wfmsynth.__version__)"
```

Python 3.12+, numpy, scipy. Nothing else is required.

## The chain

```
 bits ─► line code ─► symbols ─► levels ─► waveform
                                              │
  transmitter   tx_ffe · nonlinearity · dcd   │
  timing        ssc · timing · intra_pair_skew┤
  channel       lossy · reflect · crosstalk   │
                cascade · sparam              │
  supply        supply_coupling               │
                                              ▼
  instrument    probe ─► scope ─► sample_clock ─► digitize ─► scope ─► store
```

Order is physical. A band limit before the sampler is a measurement; after it, an artefact. A
defect placed upstream of a filter that removes it produces a label with no signal behind it.
Build in the order the diagram gives unless you have a reason.

`Signal.stage_kinds()` reports which stage each op belongs to. `wfmsynth.IMPAIRMENTS` and
`wfmsynth.PATTERNS` list what is available.

## A record, end to end

```python
import numpy as np, wfmsynth as ws

baud, tr_s = 16e9, 20e-12           # symbol rate, and the transmitter's rise time
fs_synth = 10 / tr_s                # 10 samples across the edge -> 500 GSa/s
fs_store = 256e9                    # what the instrument actually samples at
n_ui = 4096
n = int(round(n_ui * fs_synth / baud))

grid = ws.Grid(fs=fs_synth, baud=baud, n=n, v_full=0.8)
sig = (ws.Signal(seed=7, grid=grid)
       .carrier("nrz", pattern="prbs13", n_ui=n_ui, tr_frac=0.35, causal=True)
       .timing(rj_ps=0.9)
       .lossy(loss_db=12.0, loss_at_ghz=8.0, causal=True)
       .events("glitch", severity=1.0, indices=[1024])
       .scope(bw_hz=33e9)
       .sample_clock(ppm=55.0, n_out=int(round(n * fs_store / fs_synth)))
       .digitize(bits=10)
       .store(bits=11))

x, events = sig.realize()           # waveform plus what was actually injected
```

`waveform()` returns the samples alone; `realize()` returns them with an `EventList`.

## Size the record before you build it

Two numbers decide whether the data means anything, and both have a rule.

**The sample rate follows the fastest edge, not the symbol rate.** A grid chosen from
samples-per-symbol can leave a fraction of a sample across the transition, and then every rise
time, jitter and slew measurement on that record is measuring the grid.

```python
k = 10                              # samples across the transition; 8 is the floor
fs = k / tr_s                       # 500 GSa/s for a 20 ps edge
```

For the other direction — the bandwidth a given edge needs, or the edge a given bandwidth
produces — the relationship is `BW × tr = 0.3497` for 10–90 % and `0.1348` for 30–70 %.

**The record length follows the pattern period.** Folding averages N repetitions of a periodic
pattern and reduces uncorrelated noise by √N while preserving the pattern-dependent part. Choose
the length from the period, not from a round number of symbols.

```python
repeats = 256                                # buys sqrt(256) = 16x; 128 buys 11.3x
period = ws.describe_pattern("prbs13", length=8191)["period"]    # 8191
n_ui_folded = repeats * period               # 2,096,896 symbols
```

Take the period from the registry whenever you know which pattern it is. For a symbol stream you
did not generate, `ws.pattern_period(symbols)` returns `(lag, score, autocorrelation)` — and the
lag it returns is *a* period rather than the fundamental one. Measured on clean PRBS with no
noise, it is an exact integer multiple that varies with how much signal you give it:

| pattern | period | 4 reps | 8 reps | 16 reps |
|---|---|---|---|---|
| prbs7 | 127 | 127 | 381 (3x) | 889 (7x) |
| prbs9 | 511 | 1022 (2x) | 511 | 2555 (5x) |

Every one of those scored 1.000, so the score does not tell you which you got. Fold on a multiple
and you get correspondingly fewer folds than the record could give. Divide the returned lag by
small integers and keep the smallest lag that still correlates.

A maximal-length sequence whose period exceeds what you can store cannot be folded at all. That
is a property of the pattern, so decide it when you choose the pattern.

## The instrument is part of the physics

A record with no instrument in it is a plot. The synthesis grid is not a sample rate anyone owns:
sizing from the edge routinely asks for hundreds of gigasamples per second, and storing a record
at that rate describes an instrument that does not exist.

Separate the two rates explicitly:

- the **synthesis** grid — fast enough that the render does not limit the edge
- the **stored** record — what the instrument samples at, set by `sample_clock(n_out=...)`

`sample_clock` is a free-running timebase: `ppm` offsets it from the link, `drift_ppm_per_s` walks
it, `phase0_s` sets where it starts. It resamples asynchronously, so samples-per-symbol does not
need to be a whole number, and it should not be forced to one.

Keep the front end below the sampler's Nyquist. A declared analog bandwidth above about 0.45 × fs
is not an instrument; it is two numbers written side by side, and content above Nyquist folds into
the record.

## The recipe is the artifact

```python
r = sig.recipe()                            # plain data: {"ops": [...], "grid": {...}, ...}
digest = sig.sha256()                       # content address of the chain

with open("recipe.json", "w") as fh:
    fh.write(sig.to_json())

again = ws.Signal.from_recipe(r)
assert again.sha256() == digest             # and the samples are bit-identical
```

Keep the recipe, not the waveforms. It is what you review, diff and version, and it expands back
to the samples on demand. Record the resolved parameters alongside any name you resolved, so the
recipe replays for someone who does not hold your registry.

## Events and labels

`events()` places a localised mechanism and reports what it injected. Placement is by
`indices=[...]` in symbols (with the default `on="symbols"`), or `count=`/`fraction=`/`every=` to
scatter, or `times=`/`samples=` to place directly.

**Read an event's time, not its sample index.** `Event.sample` indexes the grid the event was
placed on. Anything downstream that resamples — `sample_clock` above all — leaves that index
pointing at a different instant, and on a chain whose synthesis grid outruns the sampler the index
can exceed the whole stored record. `Event.t_s` stays correct.

```python
for e in events.events:
    stored_index = int(e.t_s * fs_store)    # correct in the record you stored
```

Getting this wrong is expensive in a specific way: a label silently dropped for being "out of
range" does not read as missing data. It reads as a positive claim that a defective record is
clean, and a model trained on it learns that the defect is normal.

## Patterns: mechanisms here, standards knowledge in your code

A pseudo-random sequence is a polynomial, a seed and a length, and that arithmetic is in the
library — `ws.PATTERNS` lists what ships, `physics.PRBS_TAPS` holds the polynomials.

Which sequence a given standard names for a given test is a different kind of fact. It has a
revision date, a document behind it, and it changes. Register it from your own code:

```python
def fn(length, **kw):                        # every generator takes `length`, and that is the
    ...                                      # whole calling contract

ws.register_pattern(
    "my_stress", fn,
    period=65535,                            # symbols before it repeats
    levels=4,                                # which carrier these symbols belong on
    source="IEEE 802.3-2022 120.5.11.2.3",   # the document and clause: what makes the name auditable
    marker=(3.0, -3.0, 3.0),                 # a level subsequence unique within the period, so a
                                             # consumer can find where a capture starts
    hash_source=True,                         # your code, so a changed generator is named
)
symbols = ws.resolve_pattern("my_stress", length=65535)
```

`source` and `marker` are deliberately empty on the entries the library ships: which document
specifies a polynomial, and which word it designates for alignment, is knowledge that belongs with
the caller. `hash_source=True` records a digest of your generator, so a consumer holding a
different function under the same name gets a named mismatch instead of different samples.

Register rather than passing a bare callable: a function cannot be serialised or digested, so a
recipe holding one cannot be replayed. If a consumer may not have your registration, embed the
symbols as data (`carrier(..., symbols=[...])`, or `pattern(..., embed=True)`), which needs no
code at all on the other side.

## Traps

**An op refuses a parameter it does not read.** `sample_clock(fs_hz=...)` and
`events(..., at_ui=...)` raise, naming the nearest valid key and the whole accepted set, because
a parameter the physics ignores would sit in the recipe describing a record it did not shape. Read
the message rather than guessing again: it lists what the op takes.

That gate is recent, and the habit it replaces is still the right one. **Assert the effect, never
the call.** A knob that is accepted, recorded and ignored is the most convincing kind of wrong, so
check the output moved.

**Measure with an instrument that can see the effect.** Several plausible measurements cannot:
sub-sample displacement read with integer sample counts returns exactly zero, so duty-cycle
distortion measures as absent when it is present and correct. Calibrate on a known answer before
trusting a measurement on an unknown one.

**A channel code is not a line modulation.** `128b/130b` is a code applied to the bits;
NRZ and PAM4 are how symbols become levels. Reading one as the other changes the symbol rate and
every timing figure that follows from it.

**Index per-record data by record identity, never by row position.** One filter or reshard
upstream and every record wears another record's parameters.

## Writing many records out

There is no archive writer in this library — it produces waveforms, and the container is yours to
choose. Two formats carry the weight in practice.

**HDF5, for one record that a bench instrument or any HDF5 reader should open.**

```python
from wfmsynth import hdf5
hdf5.write_hdf5("record.h5", volts, fs=fs_store, t0=-1e-9, full_scale=0.8)
volts_back, t = hdf5.read_hdf5("record.h5", channel=1)
```

Samples go out as `int16` codes with the scale that inverts them (`volts = code * YInc + YOrg`,
`seconds = index * XInc + XOrg`). Storing floats reads fine in Python and does not load on an
instrument. Pass `full_scale=` when several records must share one vertical scale, or each
record's window follows its own range and two records stop being comparable code for code. An
instrument also looks for metadata identifying a file as its own; `like="a_capture.h5"` copies
that out of a capture you already have. Needs `h5py`.

**Zarr, for a corpus you will train on.** Use `zarr` directly, and get the layout right the first
time — it is expensive to change once the data exists.

1. **Array count is what breaks at scale**, not chunk count. Sharding packs chunks inside an
   array; nothing packs arrays inside a store. One array per record is the natural layout and it
   stops opening in reasonable time in the thousands. A record dimension inside one array keeps
   working. One array per record is fine while records are few, large, and of differing length.
2. **Chunk is the unit of read; shard is the unit of file.** Chunk along time, around a megabyte,
   so a training loop reading windows pays for roughly what it reads. Size shards to bound how
   many files exist.
3. **Consolidated metadata is a cache.** It goes stale silently on any mutation, so nothing that
   validates a store may read it.
4. **Choose fill values the data cannot hold** — not-a-number for floats, the dtype minimum for
   integers, never a boolean. Absence then stays distinguishable from zero, which is what lets a
   store ship with its arrays declared and unwritten and filled on arrival.
5. **Digest array values, not files**, in a canonical order and byte order, so recompression or
   resharding is not a corruption alarm.
6. **Key every per-record column by record identity**, never by row position. One filter or
   reshard upstream and every record wears another record's labels.

Two more that matter only for training. Keep a **group id** per record and split on it, or a
defect and its own matched control land on opposite sides of the split and the model learns the
pair. And do not store per-sample columns you can recompute — a derived column costs the same
bytes as the signal and can disagree with it.

## Check what you built

Assert the outcome, because a mechanism that ran is not an effect that landed.

```python
def link(loss_db):
    return (ws.Signal(seed=3, grid=grid)
            .carrier("nrz", pattern="prbs13", n_ui=n_ui, tr_frac=0.35, causal=True)
            .lossy(loss_db=loss_db, loss_at_ghz=8.0, causal=True))

a, b = link(2.0).waveform(), link(16.0).waveform()
assert not np.allclose(a, b)                 # the knob moved the output at all
assert fs_synth * tr_s >= 8                  # the grid resolves the edge you asked for

def centroid(v, fs):                         # a lossy channel lowers the spectral centroid
    f = np.fft.rfftfreq(len(v), 1 / fs)
    pw = np.abs(np.fft.rfft(v)) ** 2
    return float((f * pw).sum() / pw.sum())

assert centroid(b, fs_synth) < centroid(a, fs_synth)

# and the record replays to the same bytes
assert ws.Signal.from_recipe(sig.recipe()).sha256() == sig.sha256()
```

`ws.measure`, `ws.eye`, `ws.eye_height` and `ws.recover_and_fold` cover the usual measurements.
`examples/` in the repository holds runnable end-to-end scripts.

## Staying current

This file is `version: 2` in its own frontmatter. The copy in the repository is the source of
truth, so an installed copy can fall behind it.

Check the installed version against the published one without cloning anything:

```bash
sed -n 's/^version: *//p' ~/.claude/skills/wfmsynth/SKILL.md | head -1
curl -fsSL https://raw.githubusercontent.com/m-holloway/wfmsynth/main/skills/wfmsynth/SKILL.md \
  | sed -n 's/^version: *//p' | head -1
```

From a clone of the repository, ask git instead:

```bash
git -C /path/to/wfmsynth fetch -q origin
git -C /path/to/wfmsynth diff --stat HEAD origin/main -- skills/wfmsynth/SKILL.md
```

`skills/install.sh --check` in the repository does the comparison and prints both numbers.
`skills/install.sh` installs or updates in place; it is idempotent.

If the published version is higher than the installed one, say so plainly and offer to update.
Do not update a user's installed skill without asking.

When you change this file, raise `version` in the frontmatter. An unchanged number on changed
content is what makes an installed copy undetectably stale.

## Read next

- `docs/ARCHITECTURE.md` — why the library is shaped this way, the ordering gates, and the rules
  for laying out an archive of many records
- `docs/DATASET-METHODOLOGY.md` — building a defensible dataset: sourcing parameters, marking how
  good each one is, the sizing arithmetic in full, and the traps in more depth
