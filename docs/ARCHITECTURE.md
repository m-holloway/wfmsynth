# Why this library is a chain of small ops

An argument for composable, graph-shaped libraries — for synthesising data in particular, and for
libraries shared between teams in general.

## The shape

A waveform here is an ordered list of operations. Each one owns exactly one physical thing.

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

Nothing in that list knows about the others. `jitter` displaces edges. `lossy` applies a frequency
response. `sample_clock` resamples onto an independent timebase. Each is a function from a waveform to
a waveform, and a record is their composition.

```python
s = Signal(seed=1, grid=Grid(fs=256e9, baud=16e9, n=1<<16, v_full=0.8))
s.carrier("nrz", pattern="prbs13", n_ui=4096, tr_frac=0.35)
s.jitter(rj_ps=0.9)
s.lossy(loss_db=12.0, loss_at_ghz=8.0)
s.scope(bw_hz=33e9)
```

## Composable beats configurable

The alternative shape is a generator driven by a configuration file: a schema of named scenarios, each
a bundle of settings. It is easier to write and easier to use for the cases its author anticipated.

That is also its ceiling. A configuration enumerates combinations someone thought of. A chain
expresses combinations nobody thought of, including the ones that turn out to matter — a duty-cycle
error on a bus whose specification never mentions one, a probe's loading changing the very rise time
the bus is graded on, an asynchronous sampler interacting with spread-spectrum clocking. Each of those
was an unanticipated combination of two existing ops, and each needed no new code.

The economics follow. In a configuration-shaped library, every new case is a change request to whoever
owns the schema. In a composable one, a new case is a line the consumer writes. The first scales with
the owning team's capacity; the second does not depend on it.

## The recipe is the artifact

Because every op is data rather than a function call, the chain serialises.

```json
{"op": "lossy", "args": {"loss_db": 12.0, "loss_at_ghz": 8.0},
 "_prov": {"stage": "channel", "node": "TP2", "impairment": "loss"}}
```

That one property carries most of the practical value:

- **A dataset becomes a document.** Ninety-five kilobytes of recipe expands to half a gigabyte of
  waveforms. The recipe is what you review, diff, email and keep in version control.
- **Reproducibility is checkable rather than promised.** Digests are taken over array values, so a
  rebuilt record either matches or it does not, and a codec or library upgrade that changed no number
  does not go red.
- **Provenance has somewhere to live.** Each op records what stage it belongs to and what it
  represents, so a record explains itself without a companion spreadsheet.
- **Changing a dataset is editing a document.** Retune a parameter, rerun, and the waveforms follow.

## Composition needs gates, or it is just freedom

Order is physical. A band limit before a resampler is a measurement; after it, an artefact. A defect
applied upstream of a filter that removes it is a label with no signal behind it. In a configuration
schema the author fixes the order once. In a chain, the caller can get it wrong, silently.

So the library carries checks that a monolith would not need:

- **Every op is classified** for whether a warm-up region can host it — has an impulse response, has
  none, has one known in closed form, or cannot be guarded. A new op that skips classification is
  refused rather than defaulting into the permissive case.
- **Every knob is swept** and must move the output. A parameter a recipe records and the physics
  ignores is the most convincing kind of wrong.
- **Impairments can refuse.** An impairment that changes nothing at the plane a record is observed
  from raises instead of producing a labelled record with no signal in it.
- **Property checks over unit tests** where the physics allows: a lossy channel must lower the
  spectral centroid; a causal channel must put its energy after t=0; crosstalk must couple the
  aggressor's derivative and not its level.

Composability and rigour are not in tension here. The gates are what make the freedom usable.

## Where the boundary sits: mechanisms in, knowledge out

A pseudo-random sequence is a polynomial, a seed and a length. That is arithmetic, and it belongs in a
physics library.

"Which sequence a given standard names for a given test" is a fact with a revision date, a document
behind it, and a tendency to change. It does not belong here. It lives in a registry the caller
populates, and a recipe records both the name and the resolved parameters — so the document stays
readable and stays replayable by someone who does not have the registry entry.

The same line applies to extension. A user needs their own patterns, and the obvious mechanism is to
accept a callable. That breaks reproducibility outright, because a function cannot be serialised or
digested, so a recipe holding one cannot be replayed. The workable form is a named registration whose
source is hashed, plus symbols-as-data as the always-available fallback: a consumer without the entry
is told exactly what they need rather than getting a silent mismatch.

## Why this suits agents particularly well

This is the argument that has grown fastest in practice.

**A small orthogonal surface beats a large specific one.** An agent reasoning about twenty composable
ops has fewer things to learn and more things it can express than one reasoning about two hundred
configuration keys. Combinations come free.

**Each op is independently verifiable.** An agent can assert that a requested four picoseconds of
duty-cycle distortion measures as four picoseconds, on one op, in isolation. That is how a chain gets
trusted a stage at a time instead of judged as a whole. It also catches the failure mode agents are
most prone to: reporting success from a mechanism rather than an outcome. A duty-cycle request reads as
exactly zero if you measure it with integer sample counts, because the displacement is a fraction of a
sample — the op was right and the instrument was wrong. Isolated, measurable ops make that discoverable.

**The recipe is a readable intermediate representation.** An agent can emit a chain, inspect it, diff
it against a previous one, and explain what changed, without running anything. A configuration bundle
hides the same information behind a name.

**Research maps onto structure.** Reading a specification produces parameters. Parameters go into ops.
An agent can go from "read this clause" to "emit this record" with no human translating in between —
and the provenance markers give it somewhere honest to put what it could not establish, which is what
keeps a research pass from quietly inventing figures.

**A capability file is enough to hand over.** A short description of the chain, the sizing rules and
the traps lets an agent pick up dataset work with no other context. That is a different kind of
deliverable from a dataset: it is the ability to make the next one.

## The format follows the same logic

An archive holding these records has to answer a scaling question that is not obvious, and getting it
wrong is expensive later:

1. **Array count is what breaks, not chunk count.** One array per record is natural and fails at
   scale. A record dimension inside one array is the shape that keeps working.
2. **Chunk is the unit of read; shard is the unit of file.** Chunks around a megabyte, shards sized to
   bound how many files exist.
3. **Consolidated metadata is a cache, not a source.** It goes stale silently, so nothing that
   validates an archive may read it.
4. **Fill values the data cannot hold** — not-a-number for floats, the dtype minimum for integers,
   never a boolean. This is what makes absence detectable rather than indistinguishable from zero.
5. **Digests over values, not files**, in canonical order and byte order, so a codec change is not a
   corruption alarm.
6. **Every table column carries its own key.** Index by identifier, never by row position.

Rule 4 is the one that pays unexpectedly. Because an unwritten array is valid and reads as fill, an
archive can ship at any fill level — so a small file carrying every label and parameter, with the
waveform arrays declared and empty, expands into the full dataset where it lands. That is not a
separate feature. It falls out of choosing fill values honestly.

## Generalising, for libraries shared between teams

The test worth applying to any shared library is: **can a consumer answer a question the author did
not anticipate, without asking the author?**

A configuration-shaped library fails that by construction — the set of answerable questions is the set
the schema enumerates. A composable one passes, at the cost of asking more of the consumer and
requiring the author to build gates against misuse.

That cost buys three things worth more than the convenience it gives up:

- **Independence.** A consuming team maneuvers at its own pace. No request queue sits between a
  question and an answer.
- **Reuse beyond the original purpose.** Primitives built for one dataset serve the next project,
  because they were never specialised to the first.
- **Agent leverage.** Small composable primitives with verifiable behaviour are what an agent can
  actually reason over, extend and check. A library shaped this way is one an agent can be handed.

The shape is not free and it is not always right. For a narrow, stable, well-understood problem, a
configuration schema is the better engineering. For a problem where the interesting cases are the ones
nobody has thought of yet — which is most synthesis, and most research tooling — composition is what
keeps the library useful after its authors stop paying attention to it.
