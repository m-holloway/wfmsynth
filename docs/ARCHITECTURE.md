# How the library is put together

This file explains the structure of the library, the reasoning behind it, and where the
boundaries sit. Read it before changing a recipe or adding an op.

## The shape

A waveform is an ordered list of operations. Each one owns a single physical effect.

```
 bits ─► line code ─► symbols ─► levels ─► waveform
                                              │
  transmitter   tx_ffe · nonlinearity · dcd   │
  timing        ssc · jitter · skew       ────┤
  channel       loss · reflection · crosstalk │
  supply        supply_coupling · noise       │
                                              ▼
  instrument    probe ─► scope ─► sample_clock ─► digitize ─► scope ─► store
                                              │
                                              ▼   trigger (cuts the windows a scope keeps)

  an op's parameter may hold another chain:   crosstalk(aggressor=<chain>) · gain(by=<chain>)
  several named chains are one record:        Channels({...}, shared={...}) · use("name")
  a record is a voltage, an envelope or light: iq · ofdm ↔ upconvert/downconvert,  eo · laser ↔ photodetect
```

The ops are independent of each other. `jitter` displaces edges. `lossy` applies a frequency
response. `sample_clock` resamples onto an independent timebase. Each is a function from a waveform
to a waveform, and a record is their composition.

```python
s = Signal(seed=1, grid=Grid(fs=256e9, baud=16e9, n=1<<16, v_full=0.8))
s.carrier("nrz", pattern="prbs13", n_ui=4096, tr_frac=0.35)
s.jitter(rj_ps=0.9)
s.lossy(loss_db=12.0, loss_at_ghz=8.0)
s.scope(bw_hz=33e9)
```

## Why a chain rather than a configuration schema

The alternative design is a generator driven by a configuration file: a schema of named scenarios,
each a bundle of settings. It is easier to write, and easier to use for the cases its author
anticipated.

A chain also covers combinations nobody enumerated in advance. Three that come up in practice:
a duty-cycle error on a bus whose specification never mentions one; a probe's loading changing the
rise time the bus is graded on; an asynchronous sampler interacting with spread-spectrum clocking.
Each is an existing pair of ops applied together, and none needs new library code.

This changes who does the work. Under a configuration schema, a new case is a change request to
whoever owns the schema, and throughput is limited by that team's capacity. Under a chain, a new case
is a line the consumer writes.

## The recipe

Every op is data, so a chain serialises.

```json
{"op": "lossy", "loss_db": 12.0, "loss_at_ghz": 8.0,
 "_prov": {"stage": "channel", "node": "TP2", "impairment": "loss"}}
```

Consequences worth knowing about:

- A dataset is distributable as a document. Tens of kilobytes of recipe expand to hundreds of
  megabytes of waveforms, and the ratio grows with record length. The recipe is the thing to
  review, diff, email and keep in version control.
- Reproducibility is checkable. Digests are taken over array values, so a rebuilt record either
  matches or it does not. A codec or library upgrade that changed no number leaves the digests alone.
- Provenance has somewhere to live. Each op records the stage it belongs to and the effect it
  represents, so a record explains itself without a companion spreadsheet.
- Changing a dataset means editing a document. Retune a parameter, rerun, and the waveforms follow.

## Composition needs gates

Order is physical. A band limit applied before a resampler is a measurement; applied after it, an
artefact. A defect applied upstream of a filter that removes it produces a label with no signal
behind it. A configuration schema fixes the order once, in the author's code. A chain lets the caller
get it wrong, silently. The library carries checks to cover that.

- **Op classification.** Every op declares whether a warm-up region can host it: it has an impulse
  response, has none, has one known in closed form, or cannot be guarded. An op that skips
  classification raises rather than defaulting into the permissive case.
- **Swept knobs.** Every parameter is swept in the test suite and must move the output. A parameter
  the recipe records and the physics ignores is a hard failure to notice any other way.
- **Impairments can raise.** An impairment that changes nothing at the plane the record is observed
  from raises, instead of producing a labelled record with no signal in it.
- **Property checks** where the physics allows one: a lossy channel must lower the spectral centroid;
  a causal channel must put its energy after t=0; crosstalk must couple the aggressor's derivative
  rather than its level.

## Where chains meet

A chain is a line, and real records are not: an aggressor has its own channel, a clock and its
data share a PLL, a transmission is an idle, a preamble and a payload. Four things let chains
meet, and each is data in the recipe.

- **A sub-chain is a value** (`subchain`). Any parameter that takes a signal takes a chain:
  `{"ops": [...]}` is part of this record and moves with its seed; `{"from": recipe}` is another
  record, pinned as rendered. One resolver serves every op, the parameter is found by its shape,
  and its randomness is namespaced by where it sits (`crosstalk/1/aggressor/jitter/0`), so
  `roles()`, `contrast()`, `summary()` and the event list reach inside with nothing registered.
- **Joins** (`joins`, `sequence`). `combine` puts peers together sample by sample;
  `gain`/`offset`/`gate`/`delay` take a number or a chain as `by`; `sequence` and `switch` join
  chains in time.
- **Named chains** (`channels`). A `Channels` record holds several chains under one seed and
  grid, and `use(name)` is the only way one refers to another. Each chain draws under its OWN
  name wherever it is used from, which is the whole reason two channels through one `pll` move
  together, and why the correlation survives serialisation where object identity would not.
- **A toleranced value** (`tolerance`). `ws.tol(1e3, 5)` in place of a number is resolved on the
  way INTO the recipe, so every executor still sees a number and replay draws nothing; the
  stated tolerance stays beside it so `unit(seed)` can draw another unit under the same noise.

Three rules keep these honest. Nothing is ever resampled, truncated or stripped of an imaginary
part to make it fit: a sub-chain off the record's grid is refused by name. A record is in one
**domain** (below), and an op is real-only unless it has opted in with a test of what it means
elsewhere; a chain that did not state its grid starts in the domain its own ops need. And the
ops whose meaning is their position (`law`, `filter`, the joins, `use`, the I/Q impairments)
are stage kind `general`: `canonical()` refuses to sort a chain that contains one.

`trigger` is the one op that shortens a record: the grid is the long render and the op keeps the
windows a scope would. `Signal.stored_grid(x)` accounts for it, as it does for `acquire`.

## Three domains, and what a record is an envelope about

A record is in one domain at each point of its chain, and `wfmsynth.domain` is the whole of
the mechanism: a table of which ops accept which, and a trace that refuses by name.

| Domain | A sample is | Starts with | Crosses by |
|---|---|---|---|
| `real` | a voltage | `carrier`, `symbols`, `coded`, a real `capture` | |
| `iq` | I + jQ, the envelope about `fc_hz` | `iq`, `ofdm`, a first `awgn` or `spur`, a complex `capture` | `upconvert` / `downconvert` |
| `optical` | a field E, with \|E\|² the power | `eo` after a real drive, `laser`, a capture on `Grid(domain="optical")` | `eo` / `photodetect` |

Being complex is never taken as being I/Q. The two complex domains differ in their bridge (a
mixer pair, a modulator and a square-law detector), their unit and the ops that mean anything,
so an envelope offered where a field belongs is refused, and the reverse. The general stages
(`filter`, `gain`, `gate`, a sum `combine`, `sequence`, `switch`, `use`) mean the same on all
three. An op that reads a frequency stated at RF has to know what the envelope is about, and
that is the second thing the trace carries:

- **I/Q carries `fc_hz`.** `delay` turns the carrier by -2π·fc·delay as it shifts the
  envelope, and `lossy` / `sparam` read their response at `fc_hz + f`. Without a stated centre
  `delay` is refused: shifting the envelope alone is a different signal.
- **A field from `eo` or a capture states no carrier.** `fiber(wavelength_nm=)` states one
  where it acts, and `delay` is refused there.
- **A field from `laser` carries the laser op itself**: the swept frame. A tunable laser
  crossing 2 nm moves 250 GHz, which no grid holds, so the record is held in the frame of its
  own moving carrier (the samples are the constant `sqrt(power)`) and ν(t) is recipe data.
  `delay` and `resonator` read it. A delayed copy beats against the direct path at (sweep
  rate) × delay, which is every interferometer, and a ring is solved in that frame, ring-down
  included. The record stays a time record: `swept.wavelength_map` is the ground-truth axis
  and no op exposes a wavelength axis.

Because a chain that reaches the field through `eo` still starts real, its grid, recipe JSON
and digest are what they were before the field was a domain.

## Where the boundary sits: mechanisms in, knowledge out

A pseudo-random sequence is a polynomial, a seed and a length. That is arithmetic, and it belongs in
a physics library.

Which sequence a given standard names for a given test is a different kind of fact. It has a revision
date, a document behind it, and a tendency to change. It lives in a registry the caller populates.
A recipe records both the registered name and the resolved parameters, so the recipe stays readable
and stays replayable by someone who does not hold the registry entry.

The same line governs extension. A user will need their own patterns, and the obvious mechanism is to
accept a callable. A function cannot be serialised or digested, so a recipe holding one cannot be
replayed — which costs reproducibility, the property the recipe exists to provide. The workable form
is a named registration whose source is hashed, with symbols-as-data as the always-available
fallback. A consumer who lacks the registry entry is then told exactly what is missing.

## What becomes an op

The library had half of this rule already (one op, one effect; mechanisms in, knowledge out).
Issue #87 asked for the other half, and this is it.

1. **An op is a mechanism**: one physical effect that no arrangement of existing ops expresses.
   The test is to try to write it as a chain. If it can be written, it is not an op.
2. **A useful arrangement ships as a composition**: a plain function that returns a `Signal()`
   sub-chain, with a COOKBOOK entry. It stays visible, because `summary()`, the recipe and
   `roles()` show the ops it is made of, and a caller can change one of them. The shipped ones
   are `wfmsynth.recipes`; `examples/composed_capture.py` has two more (`clock()` and `pll()`).
3. **Standards knowledge goes in a registry the caller populates**, as test patterns do. A named
   parameter set has a document and a revision date behind it; the recipe records the name and
   the resolved numbers.
4. **An op that overlaps a general primitive stays, as the named shorthand** for the effect it
   names, and the docs say which is canonical (the table is in
   [COOKBOOK.md](COOKBOOK.md#which-name-is-canonical)). The named op is canonical for its own
   effect; the general stage is for the shape that has no name.
5. **Existing calls, recipes and outputs do not change.** Proven by digest against the previous
   release, not assumed ([COMPATIBILITY.md](COMPATIBILITY.md)).

The rule was applied while the primitives were built, and it cut things:

| Asked for | Ruling | What it is instead |
|---|---|---|
| an `image(rejection_db)` op | declined | an image is what imbalance produces: `iq_imbalance(**rf.imbalance_for(30))` |
| an amplifier op for RF | declined | `law("tanh")` on the passband record, between `upconvert` and `downconvert` |
| hysteresis and edge shaping on `gate`, hysteresis on `switch` | declined | a `law("comparator", hysteresis=)` or a `filter` at the end of the enable's own sub-chain |
| a pattern trigger | declined | a pulse-width trigger on the pattern's longest run |
| `fork()` / `tap()` / a shared-op registry for multi-channel records | collapsed | one op, `use(name)` |
| a second `phase_noise` shared with `timing(phase_noise=)` | declined | different mechanisms: one moves a waveform in time, the other rotates an envelope |

**Which of #87's candidate primitives were worth having.** All eight, and none became
unnecessary, but they are fewer concepts than issues. The sub-chain (#75) came first because
every other one is a parameter that holds one. Signal-driven parameters (#79) and `combine`
(#76) are one module, and `gain(by=chain)` is the same samples as a product `combine`: both
names are kept because one says "a property of this signal" and the other "a second signal".
`sequence` and `switch` (#77) stayed two ops, since one states its instants and the other emits
them. Tolerances (#86) and the trigger (#85) are independent of the rest. `law`/`filter` (#80)
are the general stages rule 4 is about.

**The rule applied again (#88–#94).** Seven issues asked for OFDM, an envelope channel,
dwelling sources, clutter, the optical field and a swept-wavelength bench. Each was first
written as a chain. Three ops came out (`ofdm`, `laser`, `resonator`), four kinds of an
existing op (`carrier("states")`, `filter("doppler")`, `law("saleh")`, `law("rapp")`) and a
handful of opt-ins; the rest are functions in `wfmsynth.recipes`.

| Asked for | Ruling | Why, or what it is instead |
|---|---|---|
| OFDM (#88) | op `ofdm` | an inverse FFT per symbol and a cyclic prefix is one mechanism; a chain of `iq` per subcarrier has no rectangular pulse and no prefix |
| frames, preambles, gaps, per-block constellations (#88) | composition | `sequence` of `ofdm` chains; a preamble is all pilots; disjoint `active=` joined by `combine` |
| a standard's OFDM numerology (#88) | registry | `ofdm.register(name, source=, ...)`; the recipe holds the name and the numbers |
| `delay`, complex `gain`, `lossy`, `sparam` on an envelope (#89) | opted in | each has one meaning about `fc_hz`, checked against the passband route |
| AM/PM (#89) | `law("saleh")`, `law("rapp")` | measured first: a memoryless passband law turns the carrier by under 1e-5 rad, so AM/PM does not compose |
| fading (#89) | composition, plus two small mechanisms | `recipes.fading` is complex noise through a Doppler filter. Neither part existed: `filter("doppler")` (a FIR that resolves a Doppler of fs/1000 is thousands of taps of recipe data) and `awgn` as a first op |
| multipath, a reflection on an envelope (#89) | composition | `recipes.multipath`: `delay` + `gain` per path, `combine` |
| receiver-side I/Q imbalance (#89) | declined | `iq_imbalance` after the channel, between a turn and its inverse for a phase error |
| a dwelling source (#90) | `carrier("states")`, a source KIND | every source seam already existed for `carrier`; a renewal process is not a chain |
| a 1/f noise op (#90) | composition | `recipes.flicker`: two-state processes with corners spaced in log frequency |
| heavy-tailed event amplitudes (#90) | declined | a second randomness per event needs its own role and draw order; several `events(on="poisson")` ops, one per amplitude, express a table |
| hum, ripple, pickup, bounce, intermittent contact, oscillator start-up, multitone, a converter (#91) | compositions | all eight are chains. `spur` may now start a chain, as `awgn` may, because an I/Q multitone had no honest source |
| the optical field as a domain (#92) | domain, no op | `eo` and `photodetect` became bridges; polarisation is a `Channels` pair |
| a swept source (#93) | op `laser` | the moving carrier is the mechanism; a mode hop is a parameter of it, since a `sequence` of lasers loses the carrier |
| a ring (#94) | op `resonator` | coupled-mode dynamics in the laser's frame; every order summed in closed form |
| MZI, stray reflection, etalon, gas-cell reference (#93, #94) | compositions | `delay` + `gain` + `combine`, or one `resonator` per line. A low-finesse etalon is `stray_reflection`, a high-finesse one `resonator` |
| thermal drift, detector range switching, die-to-die campaigns (#94) | compositions | `resonator(by=<chain>)`; `photodetect().gain(by=<step>)`; `ws.tol` and `unit()` |
| a wavelength axis on the record (#93) | declined | the record stays in time; the map is ground truth and `measure.swept_map` recovers it as a bench does |

Three things a function in `recipes` could not be, each named in its module docstring: a
contact that opens fast and recovers slowly (one draw used twice, so a `Channels` shared
chain), an oscillator as a feedback loop (a chain has none; `oscillator_startup` is the loop's
closed-form solution), and a capacitor's series resistance in `buck` (`net` needs capacitance
on every node). The mechanism the first one points at, a tap that reuses a chain's own samples
inside a sub-chain, is deferred (BACKLOG).

## Working with agents

Four properties of this shape matter when an agent is doing the work.

**A small orthogonal surface.** An agent reasoning about twenty composable ops has fewer things to
learn, and more things it can express, than one reasoning about two hundred configuration keys.
Combinations come free.

**Each op is independently verifiable.** An agent can assert that a requested four picoseconds of
duty-cycle distortion measures as four picoseconds, on one op, in isolation, and build trust in a
chain a stage at a time. This also catches a failure mode agents are prone to: reporting success from
a mechanism rather than from an outcome. A duty-cycle request measures as exactly zero if you measure
it with integer sample counts, because the displacement is a fraction of a sample. The op was right
and the measurement was wrong, and isolated ops make that discoverable.

**The recipe is a readable intermediate representation.** An agent can emit a chain, inspect it, diff
it against a previous one, and explain what changed, without running anything.

**Research maps onto structure.** Reading a specification produces parameters; parameters go into
ops. An agent can go from a clause to a record with no human translating in between. The provenance
markers give it somewhere honest to record what it could not establish, which is what stops a
research pass from inventing figures.

A capability file describing the chain, the sizing rules and the traps is enough for an agent to pick
up dataset work with no other context.

## Archive format

An archive holding these records has to answer a scaling question up front, because the answer is
expensive to change later.

1. **Array count is what breaks at scale.** Sharding packs chunks inside an array; nothing packs
   arrays inside a store. One array per record is the natural layout and it fails at scale. A record
   dimension inside one array keeps working.
2. **Chunk is the unit of read; shard is the unit of file.** Size chunks around a megabyte, and size
   shards to bound how many files exist.
3. **Consolidated metadata is a cache.** It goes stale silently, so nothing that validates an archive
   may read it.
4. **Fill values the data cannot hold** — not-a-number for floats, the dtype minimum for integers,
   and never a boolean. This makes absence detectable where zero would be ambiguous.
5. **Digests over values**, in canonical order and byte order, so a codec change does not read as
   corruption.
6. **Every table column carries its own key.** Index by identifier; row position is not stable.

Rule 4 has a second effect. An unwritten array is valid and reads as fill, so an archive can ship at
any fill level. A small file carrying every label and parameter, with the waveform arrays declared
and empty, expands into the full dataset where it lands. That falls out of choosing fill values
honestly, and needed no separate mechanism.

## Applying this to other shared libraries

The question worth asking of a shared library: can a consumer answer a question the author did not
anticipate, without asking the author?

A configuration-shaped library answers only the questions its schema enumerates. A composable one has
a wider range, at the cost of asking more of the consumer and requiring the author to build gates
against misuse. What that cost buys:

- **Independence.** A consuming team moves at its own pace, with no request queue between a question
  and an answer.
- **Reuse past the original purpose.** Primitives built for one dataset serve the next project,
  because they were never specialised to the first.
- **Agent leverage.** Small composable primitives with verifiable behaviour are what an agent can
  reason over, extend and check.

The shape is not always the right one. For a narrow, stable, well-understood problem, a configuration
schema is the better engineering — it is less code and harder to misuse. Composition earns its keep
where the interesting cases are the ones nobody has thought of yet, which covers most synthesis work
and most research tooling, and it keeps a library useful after its authors stop maintaining it.
