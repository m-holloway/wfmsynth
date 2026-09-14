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
{"op": "lossy", "args": {"loss_db": 12.0, "loss_at_ghz": 8.0},
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
