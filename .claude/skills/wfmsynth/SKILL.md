---
name: wfmsynth
description: Synthesise oscilloscope-realistic waveforms with known ground truth using the wfmsynth library — compose impairment chains, size a record from its edge, model the acquisition instrument, export to HDF5 or Zarr, and emit replayable recipes with content digests. Use when asked to generate or extend synthetic signal-integrity data, build training sets with defect labels, model a link or an instrument, or reproduce a waveform from a recipe.
version: 1
---

# wfmsynth

A waveform is an ordered list of operations. Each op owns one physical effect and is a function
from a waveform to a waveform, so a record is their composition and the list of ops is data.

```bash
git clone https://github.com/m-holloway/wfmsynth.git
cd wfmsynth && pip install -e .          # or: export PYTHONPATH=$PWD
python -c "import wfmsynth; print(wfmsynth.__version__)"
```

Python 3.9+, numpy, scipy. `h5py` only for HDF5 export.

**`REFERENCE.md`, beside this file, has the worked examples** — a record end to end, the sizing
arithmetic, recipes and digests, the pattern registry, export, and validation snippets. Read it
before writing code. This file is the shape of the thing and the mistakes that cost time.

## The shape

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
                   or: acquire(AcquisitionProfile(...))
```

Order is physical. A band limit before the sampler is a measurement; after it, an artefact. A
defect placed upstream of a filter that removes it produces a label with no signal behind it.

`ws.IMPAIRMENTS` and `ws.PATTERNS` list what exists; `Signal.stage_kinds()` says which stage each
op in a chain belongs to.

## Two numbers decide whether the data means anything

**The sample rate follows the fastest edge, not the symbol rate.** `fs = k / tr_s` with k ≥ 8. A
grid chosen from samples-per-symbol can leave a fraction of a sample across the transition, and
then every rise time, jitter and slew measurement on that record is measuring the grid.

**The record length follows the pattern period.** Folding averages N repetitions and cuts
uncorrelated noise by √N — measured: 256 repeats buys 15.8×, 128 buys 11.4×. Choose the length
from the period, not from a round number of symbols.

## What people get wrong

Each of these is silent. The record renders, the digest is stable, and it is not the record that
was asked for.

1. **`sample_clock(n_out=)` is a record LENGTH, not a rate.** To store at a rate below the
   synthesis grid, use `acquire(AcquisitionProfile(sample_rate_hz=, record_length=, ...))`.
   Passing `n_out` with a real `ppm` truncates the record instead of resampling it — half the
   record silently disappears, and any defect in the discarded half becomes a label asserting a
   defect that is not there. The raw op's `ppm` is a rate *scaling* against the grid, not a small
   offset; see `REFERENCE.md`.
2. **`waveform()` returns normalised amplitude, about ±1 — not volts.** `Grid(v_full=)` is
   recorded in the recipe and does not scale the output. Convert with `x * 0.5 * v_full`.
3. **`Signal(seed=)` does not seed the pattern phase.** `Signal(seed=7)` and `Signal(seed=99)`
   give bit-identical waveforms unless you pass `carrier(..., seed=)`. A thousand records from
   one template otherwise share one PRBS phase, and a model learns the position.
4. **Below about 8 samples across the transition you are measuring the grid, not the signal.**
   The delivered 10–90 % rise time is the one requested to within 1.2 % at k = 8 and 0.04 % at
   k = 32. Under `tr_frac × samples_per_ui = 2` the library clamps to two samples and warns, and
   from there down every rise time, jitter and slew figure is the grid's. Measure `tr` on the
   record rather than trusting the request.
5. **An op refuses a parameter it does not read**, naming the nearest key. Read the message. The
   habit it does not excuse: **assert the effect, never the call.**
6. **`carrier(pattern=)` takes only built-in sequences.** A registered name renders through
   `Signal.pattern(name, length=...)`; symbols you already hold go through `Signal.symbols([...])`.
   `carrier(symbols=...)` does not exist.
7. **Read an event's `t_s`, not its `sample`.** `Event.sample` indexes the grid the event was
   placed on, and anything that resamples leaves it pointing elsewhere — possibly past the end of
   the stored record. `int(e.t_s * fs_stored)` is the index, good to a few samples of front-end
   group delay.
8. **HDF5 `full_scale` must contain the record's own excursion**, which is wider than the
   transmitter's amplitude once a channel, a defect and a supply have added overshoot. It is not
   `v_full`. A window narrower than the record is refused rather than clipped.
9. **`pattern_period` returns *a* period, not the fundamental** — an integer multiple, scoring
   1.000 either way, because the autocorrelation ties exactly at every multiple and float noise
   breaks the tie. Take the period from the registry when you know the pattern.
10. **`BW × tr = 0.3497` (10–90 %) and `0.1348` (30–70 %) are single-pole constants.** 10–90 %
    holds within 0.2 % on the library's default Bessel-4 front end; 30–70 % is 0.149 there, 10 %
    off. Convert a 30–70 % specification with care.
11. **Index per-record data by record identity, never by row position.** One filter or reshard
    upstream and every record wears another record's labels.
12. **Measure with an instrument that can see the effect.** Sub-sample displacement read with
    integer sample counts returns exactly zero, so duty-cycle distortion measures as absent when
    it is present and correct. Calibrate on a known answer first.

13. **`eye_height` requires `levels` and has no default.** 2 for NRZ, 4 for PAM4, N for PAM-N.
    Reading a binary record as four levels measures an eye that is not there, and does it
    silently: 0.0017 against the 0.0432 one such record actually has. The level count is a
    property of the record rather than a preference, so there is nothing sensible to default to.
14. **`eye_contour` floors once the eye closes**, so it is not a health check on a closed eye:
    past closure it sits near zero whatever you do to the channel. `eye_sigma` goes negative
    there, which is the eye being shut rather than an error, and still separates one closed eye
    from another.
15. **`eye_density` returning zeros means the clock did not lock, not that the record is empty.**
    Check `meta['traces']`: zero, with `meta['clock']['reason']` one of `short`, `edges`,
    `nolock`, `nowindow`, `toofast` or `still`. The threshold is `eye.LOCK_RMS_UI = 0.25` against
    `clock['lockRmsUi']`, and on a failure the clock dict carries only five keys — every lock
    diagnostic, `residualRmsUi` included, is absent rather than zero. `reason == 'fromgrid'` is
    NOT a failure: the period was fitted to the edges but seeded by the grid's rate, so the
    picture is real and only the clock is partly the grid's. A mid-reach lossy chain failing to
    lock at the pad is ordinary. To see the impairment anyway, fold on nominal centres with
    `eye.density(x, centres, ui)` — the lower-level call that takes centres instead of recovering
    them — and say on the plot that it is a nominal-centre fold rather than a recovered eye.
16. **Exporting for a bench instrument means modelling that instrument first.** A record taken
    straight off the synthesis grid has a full-rate, full-bandwidth, infinite-resolution edge;
    loaded onto a scope it looks like nothing that scope could have captured, so any margin or
    rise time read off it describes the simulation. Put the probe, the front end, the sample rate
    and the converter in the chain before writing the file, and pass the STORED rate to the
    exporter. `REFERENCE.md` has the parameters.
17. **After `acquire()`, `Signal.grid` is the SYNTHESIS grid.** It reports the rate and length
    the chain was built on rather than the record in your hand, so handing it to a measurement
    measures the wrong thing. Build one for the stored record:
    `Grid(fs=fs_store, baud=baud, n=len(x))`.

## Where the boundary sits

Mechanisms live in the library; standards knowledge does not. A pseudo-random sequence is a
polynomial, a seed and a length — that is arithmetic. Which sequence a given standard names for a
given test has a revision date and a document behind it, so it goes in the registry the caller
populates, and a recipe records both the name and the resolved parameters.

## Staying current

This file is `version: 1`. The copy in the repository is the source of truth, so an installed copy
can fall behind it. Run this with the same `HOME` the install used:

```bash
sed -n 's/^version: *//p' ~/.claude/skills/wfmsynth/SKILL.md | head -1
curl -fsSL https://raw.githubusercontent.com/m-holloway/wfmsynth/main/.claude/skills/wfmsynth/SKILL.md \
  | sed -n 's/^version: *//p' | head -1
```

On Windows, where there is no `sed`:

```powershell
Select-String -Path "$HOME\.claude\skills\wfmsynth\SKILL.md" -Pattern '^version:' | Select-Object -First 1
```

`skills/install.sh --check` in a clone prints both and changes nothing; `skills/install.sh`
updates in place, and `skills/README.md` has the PowerShell equivalent. Report a difference plainly and offer to update; do not update without asking.
Raise `version` whenever this file changes, or an installed copy is undetectably stale.

## Read next

- `REFERENCE.md` — worked examples for everything above
- `docs/ARCHITECTURE.md` in the repository — why the library is shaped this way, and the rules for
  laying out an archive of many records
- `docs/DATASET-METHODOLOGY.md` — sourcing parameters, marking how good each one is, and the
  sizing arithmetic in full
