---
name: wfmsynth
description: Synthesise oscilloscope-realistic waveforms with known ground truth using the wfmsynth library — compose impairment chains, size a record from its edge, model the acquisition instrument, export to HDF5 or Zarr, and emit replayable recipes with content digests. Chains can start from an analog source, a CMOS/PWM source, or a real capture, not just a serial link; chains can meet (a chain as another op's parameter, several correlated channels in one recipe, a triggered acquisition, toleranced parts); a record can be complex baseband (I/Q, OFDM, fading and multipath) with RF impairments and measures, or an optical field with a swept laser and resonators; and wfmsynth.recipes ships realistic clutter (hum, ripple, 1/f, contact bounce) as visible chains. Use when asked to generate or extend synthetic signal-integrity data, build training sets with defect labels, model a link, a bus, a multi-channel capture, an RF signal or an instrument, or reproduce a waveform from a recipe.
version: 11
---

# wfmsynth

A waveform is an ordered list of operations. Each op owns one physical effect and is a function
from a waveform to a waveform, so a record is their composition and the list of ops is data.

```bash
git clone https://github.com/m-holloway/wfmsynth.git
cd wfmsynth && pip install -e .          # or: export PYTHONPATH=$PWD
python -c "import wfmsynth; print(wfmsynth.__version__)"
```

Python 3.9+, numpy 2.0+, scipy 1.13+. `h5py` only for HDF5 export.

**`REFERENCE.md`, beside this file, has the worked examples** — a record end to end, the sizing
arithmetic, recipes and digests, the pattern registry, chains that meet, I/Q, export, and
validation snippets. Read it before writing code. This file is the shape of the thing and the
mistakes that cost time.

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
                   then, optionally: trigger(pre_s=, post_s=)   keeps the windows a scope would

  a parameter that takes a signal takes a chain:   .crosstalk(aggressor=Signal().carrier(...).lossy(...))
  several named chains are one record:             ws.Channels({...}, shared={...})  +  .use("name")
  a record may be a complex envelope:              .iq("qam16", prbs=15) | .ofdm(...)  ...  .upconvert(fc_hz=)
  or light:                                        .eo() | .laser(start_nm=, stop_nm=)  ...  .photodetect()
  a useful arrangement is a function, not an op:   ws.recipes.hum(...) · multipath(...) · ring(...)
```

Order is physical. A band limit before the sampler is a measurement; after it, an artefact. A
defect placed upstream of a filter that removes it produces a label with no signal behind it.

A chain is a SERIES path with one source and one probe. When several parties share one net and
hand it over — a bus, a backplane, a level translator — that has no expression as a chain:
`Signal.net()` solves the whole network, and `wfmsynth.scenario` compiles its schedule from
"who held the net when". Each driver may bring its OWN chain as its level (`source=`, with a
required `volts=`), so parties with different supplies, output impedances and edge rates share
one solve instead of being summed by hand — and a released driver then contributes nothing
rather than leaving its last level in the next owner's baseline. Elements are lumped R, L and
C, so a hand-over RINGS rather than merely settling: package inductance against node
capacitance is what a scope shows at turn-on and release, and it was the last visible
difference between a synthetic record and a measured one. The protocol above it is yours to
write (`examples/i2c_bus.py`, `examples/multi_driver_bus.py` and
`examples/inductive_handover.py` are worked ones); see `REFERENCE.md`.

## When chains meet

A chain is a line; these let lines meet, and all of them are data in the recipe. `REFERENCE.md`
has each one worked.

- **A sub-chain is a value.** Write it `Signal()` — no seed, no grid — and pass it wherever a
  signal goes (`aggressor=`, `message=`, `by=`, `on=`, `others=`, a `sequence` segment). It
  joins the record under a namespace (`crosstalk/1/aggressor`). `{"from": other.recipe()}`
  embeds another record as rendered instead.
- **Joins.** `combine(other)`; `gain`/`offset`/`gate`/`delay` with `by=` a number or a chain;
  `sequence([...])` and `switch(select=, inputs=)` are sources made of chains.
- **`ws.Channels({name: chain}, shared={name: chain}, seed=, grid=)`** is several channels from
  one recipe; `.use(name)` applies a shared stage or stands for a shared signal.
- **`law(kind, ...)` / `filter(kind, ...)`** are the shapes with no named op; **`trigger(pre_s=,
  post_s=)`** cuts windows from a LONG render; **`ws.tol(nom, pct)`** in place of a number is
  a toleranced part and `sig.unit(seed)` another unit.
- **I/Q.** `.iq("qam16", prbs=15)` starts a complex envelope (the record's domain follows its
  source; `Grid(fc_hz=)` states the centre). `upconvert()` / `downconvert(lpf=dict(pass_hz=))`
  cross to a real passband voltage and back; `dac`, `iq_imbalance`, `iq_offset`,
  `phase_noise`, `freq_offset`, `awgn`, `spur` impair it; `ws.rf` measures it; `ws.read_iq` /
  `ws.to_sigmf` are files. `.ofdm(n_fft=, scs_hz=, cp=, active=)` is the multicarrier source
  (`rf.evm(..., ofdm=)`). `delay`, a complex `gain`, `lossy`, `sparam` and
  `law("saleh" | "rapp")` act on the envelope about `fc_hz`. An op or a measure handed the
  wrong kind of record refuses by name and says what to use.
- **The optical field** is a third domain between `eo()` (or `laser(...)`) and `photodetect()`.
  A `laser`'s field is held in the frame of its own swept carrier, where `delay` makes
  interferometer fringes and `resonator` is a ring; `ws.swept.wavelength_map` is the truth and
  `ws.measure.swept_map` / `transmission` / `resonances` read a record as a bench does.
- **`carrier("states", dwell=[...])`** dwells at levels for random times (burst noise, an
  activity schedule for `gate`/`switch`); `ws.states.intervals(events, at=)` is what it drew.
- **`ws.recipes`** is the shipped compositions, each a function returning a visible chain:
  `hum`, `ripple`, `pickup`, `flicker` (for `combine`); `bounce`, `intermittent` (for
  `gate`); `oscillator_startup`, `multitone`, `buck` (sources); `fading`, `multipath` (I/Q);
  `mzi`, `stray_reflection`, `ring`, `gas_cell` (a laser's field).

`sig.summary()` prints any of this as a tree. `sig.roles()` is exactly the randomness the
record draws from, and `sig.contrast(name)` re-rolls one of them alone.

**Do not add an op for an arrangement.** If it can be written as a chain, write a function
that returns the `Signal()` sub-chain. An op is one mechanism nothing else expresses
(`docs/ARCHITECTURE.md`, "What becomes an op").

`ws.IMPAIRMENTS` and `ws.PATTERNS` list what exists; `Signal.stage_kinds()` says which stage each
op in a chain belongs to.

## Not just a serial link

`carrier()` also takes analog and CMOS/PWM sources, and `capture()` starts a chain from a real
file instead of a synthesized one — both take the SAME later ops (`probe`, `lossy`, `events`, …).

```python
g = ws.Grid(fs=10e9, n=4096)   # k = tr_s * fs = 20 below, the k >= 8 rule again
ws.Signal(seed=1, grid=g).carrier("cmos", v_lo=0.0, v_hi=3.3, duty=0.5, f_hz=10e6, tr_s=2e-9)
ws.Signal(seed=1).capture(path="scope.npy")   # a real capture, first op instead of a carrier
```

`probe()` also takes a fuller instrument pack (`compensate`, `l_gnd_h`, `r_term_ohm`,
`coupling="ac"`, `overload_range`/`overload_tau_s`); `open_drain()` takes a `second=` wired-AND
sink; `burst()` gates any carrier on/off (`off="zero"` or `"hold"`); `pass_fet()` is a
gate-controlled switch (off is high-Z unless a body diode clamps it to a stated rail);
`modulate()` is AM/ASK/OOK/FM/FSK/PM, its message a nested carrier spec; `events("clamped_exp",
...)` is a generic rail transient (which standard's pulse numbers apply is the caller's own
registry, same as a PRBS polynomial). None of these add a protocol name or a second builder —
same `Signal`, same one-op-one-effect rule.

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
   one template otherwise share one PRBS phase, and a model learns the position. The same
   holds for `iq(..., prbs=)`: seeds 1 and 99 transmit identical symbols; vary `prbs_seed=`.
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
17. **After `acquire()`, `trigger` or `downconvert(fs_out_hz=)`, `Signal.grid` is the SYNTHESIS
    grid — use `Signal.stored_grid(x)`.** `Signal.grid` is the rate and length the chain was
    built on, not the record in your hand, and handing it to a measurement raises nothing:
    `measure` folds at the wrong symbol period and returns a plausible number. MEASURED on an
    80 → 40 GSa/s acquisition, `eye_height` reads 0.6184 against the true 0.5896 — 4.9 % out,
    silently. The same rule for ops: the NAMED link and instrument ops (`scope`, `lossy`,
    `events`, ...) placed AFTER an acquisition still read the synthesis rate, so put them
    before it; `filter`, `law`, the joins and the I/Q impairments run at the stored rate.
18. **A unipolar source (`cmos`, and any future one) stays in real volts, not the ±1 the rest of
    `carrier` uses.** `x * 0.5 * v_full` is the wrong conversion once a chain contains one — the
    array already IS volts. `cmos`'s `tr_s` is also an absolute edge time in seconds, unlike
    `square`'s `tr_frac`: a real gate's edge does not shrink as it is clocked faster.
19. **`modulate(kind="ook", ...)` needs `suppressed=True` for a unipolar (0/1) message.**
    `(1 + depth * message)` never reaches 0 for a message that only goes 0..1; `depth * message`
    does. The same trap applies to any two-level message that is not centred on 0.
20. **`capture(path=...)` digests the sample VALUES, not the file's bytes**, at the moment
    `.capture()` is called — as long as the VALUES a different codec or dtype round-trips are
    unchanged, the digest still matches (a lossy dtype, e.g. float64 rounded through float32,
    changes the values and correctly fails to match). A file that has since changed is a named
    error at render time, not a silently different waveform under the same recipe. `embed=True`
    drops the file dependency entirely.
21. **`capture()`'s `fs_hz=` is provenance, not a rate for the rest of the chain.** It is what
    `resample=True` resamples FROM; it does not make `lossy(loss_at_ghz=...)`, `probe(bw_hz=...)`
    or any other Hz-denominated knob downstream mean anything real. Those read `grid.fs`, so a
    `capture()` chain needs its own `Grid(fs=..., n=...)` matching the file's rate — a
    `capture()`-only `Signal()` with no `Grid` renders against no real rate at all, silently.
22. **`open_drain`'s sink decision thresholds the incoming carrier at the MIDPOINT of its OWN
    excursion, not at an absolute level.** A `dc` carrier — any level, any sign — straddles a
    single value and never sinks; only a carrier that actually toggles (`nrz`, `square`, `cmos`,
    ...) drives the bus. The first device's on-resistance is the existing `r_sink_ohm` (default
    20.0); a second device (`second=...`) adds its own `r_sink_b_ohm` — there is no
    `r_sink_a_ohm`.
23. **An under-resolved `step`/`pulse`/`cmos` edge clamps to the 2-sample floor and WARNS**,
    the same visible clamp `physics.resolve_rise_time` gives every UI-fraction edge in this
    library (`carrier(tr_frac=...)`). `tr_s`/`tr_frac` here are absolute or a fraction of the
    RECORD, not of a UI, but the floor and the warning are the same mechanism.

24. **On a shared net, size the grid from the fastest MODE, not the bit rate.** A 60 ohm
    switch between two bus capacitances is a nanosecond redistribution on a microsecond bus —
    three decades apart, and a bit-rate-derived grid measures none of it. `Signal.net()` says
    so rather than returning a staircase. The same rule as #4, but the fast thing is no longer
    the edge you asked for.
25. **An under-sampled RING does not go missing, it ALIASES.** A decay the grid cannot show is
    simply absent from the record; a ring the grid cannot show comes back at a frequency that
    is not the circuit's and looks entirely plausible. With an inductance on the net, size the
    grid for 8 samples per ring period (`f0 = 1/(2*pi*sqrt(L*C))`), not just for the edge. The
    solver warns below that and says explicitly when the record's ring is an alias.

Composition and I/Q added these. Each renders, replays and digests cleanly, and is wrong.

26. **A builder method MUTATES the Signal and returns it.** `base.lossy(...)` then
    `base.rx_noise(...)` is one chain with both ops. Branch with `base.copy().lossy(...)`.
27. **A `Signal()` sub-chain is drawn under the PARENT's seed and namespace**, so it is not
    `that_signal.waveform()`; what was used is `sig.subchain("crosstalk/1/aggressor")`.
    `{"from": recipe}` is the opposite trap: pinned, unit included, so a loop that varies the
    parent's seed (or calls `unit()`) embeds the same lane in every record.
28. **Keep samples per UI a whole number, and take jitter from `timing`.** An `nrz` / `pam<N>` /
    `symbols` carrier puts each edge on a sample: MEASURED at 64 GSa/s, 1000 UI over 32768
    samples carries 15.5 ps peak to peak that is the grid's, against under 1e-9 ps at 1024
    UI, and `carrier(jitter=)` moves edges by whole samples. (`square` and `cmos` do not.)
29. **Randomness inside a SHARED chain is the same draw on every channel that uses it.** Right
    for a PLL, wrong for noise: put `rx_noise` on each channel.
30. **`by=` is the quantity itself.** A 3 % ripple is `gain(by=)` a sine of amplitude 0.03 with
    `offset=1.0`; without the offset the record is multiplied by a signal that crosses zero.
    `delay(by=)` is seconds. And `combine` ADDS: two drivers on one node are `Signal.net()`.
31. **Label from `events.on_record()`.** A loop over `events` also sees an aggressor's glitch,
    an enable's edge and each trigger firing, none a defect on this record.
32. **Split a unit dataset on `ws.unit_of(recipe)`**, never at random. And a tolerance is drawn
    when the op is ADDED: changing `sig.seed` later redraws nothing, `sig.unit(seed)` does.
33. **Raw EVM after any front end is a rotation, not distortion.** `rf.evm` fits nothing unless
    asked: name `correct=("gain", "phase")` (and `"frequency"`), and read `["removed"]`. A
    `reference="decided"` EVM reads LOW once decisions fail in noise.
34. **`awgn`'s reference power is the record as it reaches the op**, start-up, leak and spurs
    included: "Es/N0 = 25 dB" against a unit-power constellation needs `ref_power=1.0`.
35. **An I/Q file's `scale` is the value of ONE COUNT, and a wrong one cannot be detected.**
    Read with exactly what the writer returned.

The last round (OFDM, envelope channels, dwelling sources, recipes, the optical field) added
these.

36. **State `prbs_seed=` on `ofdm` (and `iq`).** The default LFSR state emits almost only
    zeros for its first few hundred bits, so the first OFDM symbols carry one constellation
    point on every subcarrier and are impulses: MEASURED 15.5 dB peak-to-average on 52 QPSK
    subcarriers with `prbs=31`, against 11.6 dB with a stated seed.
37. **`ofdm`'s `cp=` and `window=` are samples at the FFT rate** (`n_fft * scs_hz`), not
    seconds and not the grid's samples (a time is refused, with the count it comes to). And
    the `"equalizer"` correction is ONE coefficient per subcarrier for the whole record: on a
    channel that moves within it (fading), measure in blocks: `rf.evm(..., ofdm=, block=16)`.
38. **EVM of a gated or bursty record counts what was never sent.** MEASURED on an OFDM burst
    schedule: 76.5 % over every symbol against 5.6 % over the symbols inside an on interval.
    Select the symbols from the schedule's own intervals: `rf.evm(..., ofdm=, symbols=mask)`.
    And `dwell[i]` belongs to `levels[i]`, whose default is `[0.0, 1.0]`: `dwell=[a, b]` alone
    is OFF for `a`, on for `b`.
39. **A sub-chain that draws nothing is the SAME signal every time it is written.** Two
    `Signal().carrier(...).eo()` peers in a `combine` are one laser, fully coherent (4x the
    power, or none); two `recipes` paths of one transmitter are one transmission. Independence
    needs something drawn (`eo(linewidth_hz=)`, a fade) under its own role.
40. **`photodetect()` draws shot noise unless `shot=False`**, and a swept record through it is
    noisy before any device is in the chain.
41. **A deterministic recipe does not vary with the seed.** `hum`, `ripple`, `pickup`,
    `bounce`, `multitone` and `oscillator_startup` draw nothing, so a thousand records from
    one template carry the same hum at the same phase. Vary a number: `phase_rad=` on `hum`,
    `ripple` and `pickup`; `t_s=` on `bounce`; `phase="random", seed=` on `multitone`;
    `startup_s=` on `oscillator_startup`.
42. **A state source's cut intervals are not draws.** The interval in force at t = 0 and the
    last one carry `complete: False`; in a dwell histogram they bias it short. So does one
    that a `sequence` segment, a `trigger` / `acquire` window or a `delay` ended: it stops at
    the cut. After a moving `delay(by=<chain>)` every `dwell_s` is the delayed record's
    (stretched or squeezed with the samples), so none is the draw. After `trigger`, or in a restarting `sequence` segment, the state in force at
    the window's START is missing from the event list altogether.
43. **`recipes.fading` averages to unit power only over many Doppler periods**: MEASURED
    0.971 over 52 periods, and a first-path power of 1.34 over 10. Size the record from the
    Doppler, as from the pattern period.
44. **A mode hop moves the truth, and the reference interferometer cannot see it**: the
    fringe count misses a step of whole fringes, so `measure.swept_map` is wrong after the
    hop by about its size (MEASURED 28 pm for a 30 pm hop). That is the bench's failure
    reproduced, not a bug: read `swept.wavelength_map` for the truth and the `mode_hop` event
    for where.
45. **`filter` on a laser's field tracks the moving carrier**; it is not a fixed optical
    passband. `fiber` there still uses its own `wavelength_nm`.

## Where the boundary sits

Mechanisms live in the library; standards knowledge does not. A pseudo-random sequence is a
polynomial, a seed and a length — that is arithmetic. Which sequence a given standard names for a
given test has a revision date and a document behind it, so it goes in the registry the caller
populates, and a recipe records both the name and the resolved parameters.

## Staying current

The `version:` in this file's own front matter is the number to compare; the copy in the
repository is the source of truth, so an installed copy can fall behind it. (Do not restate that
number in this paragraph — it was written out here once and was two behind within a month.)
Run this with the same `HOME` the install used:

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
