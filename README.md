# wfmsynth

**Physics-informed waveform synthesis for ML datasets, benchmarks, and instrument-like
captures.** `wfmsynth` builds voltage-versus-time signals from validated physical effects:
sources, channels, reflections, interference, clock error, receiver equalization, and the
scope/ADC that records the result.

The library is useful when ideal sine waves or perfect digital edges are too clean for the
problem you are testing. It helps you:

- generate reproducible training examples with exact recipes;
- vary one physical factor while holding the others fixed;
- emit labels measured from the resulting waveform, not merely copied from input knobs;
- model the difference between an ideal simulated signal and a stored scope record; and
- compare synthetic and measured sets to identify missing realism.

Only NumPy and SciPy are required. Each physical primitive has a validation assertion that
checks the behavior it claims to model.

## Install and verify

```bash
git clone https://github.com/m-holloway/wfmsynth.git
cd wfmsynth
python -m pip install -e ".[test]"
python -m wfmsynth.validate
```

The final command is a physics sanity gate: it checks properties such as channel loss,
reflection delay, jitter transfer, ADC artifacts, and recipe round trips. It validates the
synthesizer, not an ML model trained from its output.

## Your first realistic waveform

Start with the composable `Signal` API. A `Grid` gives the waveform real sample-rate and
symbol-rate units; each chained operation represents one stage in the signal path.

```python
import wfmsynth as ws

# fs is chosen from the EDGE, not the symbol rate: 400 GSa/s is 16 samples/UI at 25 Gbaud,
# so a tr_frac=0.5 edge gets 8 samples across the transition. Under k = 8 you are measuring
# the grid rather than the signal, and the library says so -- see "Two numbers" below.
grid = ws.Grid(fs=400e9, baud=25e9, n=16_384)          # 16 samples/UI

signal = (
    ws.Signal(seed=7, grid=grid)
    .carrier("nrz", n_ui=1024, tr_frac=0.5, causal=True)  # transmitted data; k = 8 samples/edge
    .lossy(loss_db=8.0, loss_at_ghz=12.5, causal=True) # PCB/cable bandwidth
    .reflect(td_ps=80.0, gamma_s=0.15)                 # echo; gamma_s = SOURCE-end Γ, not seconds
    .scope(bw_hz=30e9)                                 # instrument front end
)

waveform = signal.waveform()
recipe = signal.recipe()  # JSON-serializable, reproducible ground truth
```

For the complete path from a fine simulation grid to a stored acquisition record, use an
`AcquisitionProfile`:

```python
profile = ws.AcquisitionProfile(
    sample_rate_hz=25e9,
    record_length=4096,
    input_bandwidth_hz=10e9,
    enob=7,
)
stored = signal.acquire(profile).waveform()
```

Run [`examples/quickstart.py`](examples/quickstart.py) for a guided first example, then see
[`examples/README.md`](examples/README.md) for the full learning path.

## Working with an agent

`.claude/skills/wfmsynth/SKILL.md` is an installable skill for Claude Code and other agent CLIs.
It carries the op chain and its ordering rules, the two numbers that decide whether a record means
anything, and the list of things that fail silently; `REFERENCE.md` beside it holds the worked
examples for sizing, the instrument, recipes and digests, the pattern registry, and export to HDF5
or Zarr.

The skill lives at `.claude/skills/wfmsynth/`, so an agent working inside a clone of this
repository finds it with nothing to install. To use it from your own project, where wfmsynth is a
dependency rather than the working directory:

```bash
./skills/install.sh
```

Or without a clone:

```bash
bash <(curl -fsSL https://raw.githubusercontent.com/m-holloway/wfmsynth/main/skills/install.sh)
```

`./skills/install.sh --check` compares what is installed against what is published and changes
nothing. `--local` installs into one project's `.claude/skills` instead. `SKILL.md` carries a
version in its frontmatter, so an agent can tell you when an installed copy has fallen behind.

On Windows, where the installer needs Git Bash or WSL, fetch the two files straight into place
with PowerShell — `skills/README.md` has the commands.

Then ask for the work in your own words.

**Generating data**
- Generate 200 NRZ records at 32 GBd, half with excess insertion loss, labelled — and tell me
  what the labels mean.
- I need a set where crosstalk and loss vary independently, so a model cannot learn one from the
  other.

**Getting the record right**
- My measured rise time is coming back equal to the sample interval. Size the grid properly for a
  15 ps edge.
- How long must this record be to fold 256 repetitions of PRBS13Q, and does that fit in
  64 Mpoint?

**The instrument**
- Store this as a 33 GHz 10-bit scope would actually capture it, with a 50 ppm sampler offset.
- Is the front end I have declared physically possible at this sample rate?

**Writing it out**
- Put these 500 records in a Zarr store laid out for training — chunked so a window read does not
  pull a whole record, and keyed so I can split without leaking a defect against its own control.
- Export this record to HDF5 so it opens both in Python and on the bench.

**Recipes**
- Here is a recipe.json — rebuild the waveform and confirm it matches the digest.
- Change the loss to 18 dB and tell me which records' digests change.

**Extending it**
- Read this clause of the specification and add its stress pattern to the registry, marked with
  where it came from.

See `skills/README.md`. Point an agent at this repository and ask it to install the skill and it
has what it needs to do so.

## Which API should I use?

| Goal | Start with | Why |
|---|---|---|
| Build a realistic, reproducible signal path | `Signal` + `Grid` | Recommended high-level API; records each stage in a recipe |
| Model what a scope stores | `Signal.acquire(AcquisitionProfile)` | Separates fine-grid simulation from front end, sampling, ADC, and record decimation |
| Apply or study one physical operation | `wfmsynth.physics` | Low-level NumPy-in/NumPy-out primitives for custom pipelines |
| Add one named fault to an existing array | `apply_impairment` | Convenient fixed vocabulary for class-labelled augmentation |
| Plant a rare, localized defect in a long record | `Signal.events` / `place_events` | Targeting (symbols, edges, pattern, aggressor, poisson, …) is independent of mechanism (runt, glitch, ring, droop, …); emits event times for an external segmenter |
| Add harmless capture variation | `domain_randomize` | Adds sub-threshold gain, offset, bandwidth, and noise changes without changing the class; currently use the default 4096-point record length |
| Sample many waveform shapes broadly | `generate` | Grammar-based morphology coverage; useful for broad pretraining, not a protocol-accurate link |
| Build a segmented PAM4 defect benchmark | `deep_capture` | Specialized legacy preset with segment, defect, and shared-link labels |

Do not treat these as interchangeable:

- A **labelled impairment** is the effect your model should detect.
- **Domain randomization** is nuisance variation the model should ignore.
- A **recipe** records requested causes; **measured ground truth** records what those causes
  actually produced after the complete chain.

## From ideal to real-world-like

A useful pipeline follows the order of the physical system:

```text
source data and clock
  -> transmitter imperfections / FFE
  -> channel loss and reflections
  -> coupled interference and supply effects
  -> receiver equalization
  -> scope/probe bandwidth and timebase
  -> ADC noise, clipping, interleave artifacts, and quantization
  -> stored record / decimation
```

| Effect | Plain-language model | Use it when |
|---|---|---|
| Channel loss | Frequency-dependent blur that smears neighboring symbols | Modeling a PCB trace, cable, package, or other bandwidth-limited path |
| Reflection | A delayed echo from an impedance discontinuity | Modeling connectors, vias, stubs, poor termination, or damaged interconnect |
| Crosstalk | Interference coupled from another active signal | Modeling adjacent traces or lanes; use asynchronous aggressors unless lock is intentional |
| Jitter / timing | Random, periodic, or slowly varying movement of edge times | Modeling transmitter clocks, SSC, phase noise, or timing margin |
| Tx FFE / Rx CTLE / DFE | Compensation applied before or after the channel | Modeling links that use real transmitter or receiver equalization |
| Supply coupling | Correlated amplitude and timing modulation from a power rail | Modeling ripple, switching activity, or power-supply-induced jitter |
| Scope / probe | Bandwidth and loading imposed before digitization | Matching what an instrument sees rather than an ideal node |
| ADC effects | Noise, interleave mismatch, clipping, and finite resolution | Matching stored sample statistics and converter artifacts |
| Localized event | A defect with finite time support (one UI, one edge, a Poisson arrival) | Planting a needle — runt, glitch, ring, droop — rather than a whole-record fault |

For jitter, prefer source timing (`carrier(..., jitter=...)` or `.timing(...)`) so the
shifted edges propagate through the channel. `physics.inject_jitter()` remains available for
legacy array-warp workflows but should not be the default for a new physical chain.

## Small glossary

| Term | Meaning |
|---|---|
| **NRZ / PAM4** | Two-level / four-level digital signaling |
| **UI (unit interval)** | One transmitted symbol period |
| **GSa/s / GBd** | Billions of samples per second / symbols per second |
| **ISI** | Inter-symbol interference: one symbol smears into its neighbors |
| **Eye height** | Vertical decision margin after repeated symbols are overlaid |
| **Rj / Pj / DCD** | Random jitter / periodic jitter / duty-cycle distortion |
| **FFE / CTLE / DFE** | Transmitter feed-forward / receiver analog / receiver feedback equalizers |
| **CDR** | Clock and data recovery; tracks some timing movement and leaves the rest visible |
| **ENOB** | Effective number of bits — a **SINAD** figure (noise *and* distortion), *not* the converter's bit depth. A 10-bit converter with ENOB 5.9 still has a 10-bit lattice; the missing four bits are noise. |
| **Touchstone / S-parameters** | Standard measured frequency-response files for channels |

## Learning path

1. **Start:** [`examples/quickstart.py`](examples/quickstart.py) — source, channel, capture
   variation, and reproducibility.
2. **Build a complete chain:** [`examples/realistic_scenario.py`](examples/realistic_scenario.py)
   — electrical, optical, and multi-lane scenarios.
3. **Create ML data safely:** [`examples/provenance.py`](examples/provenance.py),
   [`examples/ground_truth.py`](examples/ground_truth.py), and
   [`examples/confounder_sweep.py`](examples/confounder_sweep.py).
4. **Check realism:** [`examples/sim_to_real.py`](examples/sim_to_real.py) — find features
   that separate synthetic from measured data.
5. **Use specialist features as needed:** clock recovery, Touchstone channels, non-integer
   samples/UI, two-rate acquisition, and localized events (needles in a long record)
   are indexed in [`examples/README.md`](examples/README.md).

## Capability map

| Area | Modules | Main capabilities |
|---|---|---|
| Composition and units | `compose`, `grid`, `streams` | `Signal`, recipes, deterministic factor streams, contrastive pairs |
| Sources and effects | `physics`, `capture`, `impairments`, `events`, `grammar` | Digital/RF/analog/CMOS sources, a real capture from disk, channels, reflections, jitter, named faults, localized needles (incl. a generic clamped-exponential rail transient), broad shape generation |
| Acquisition | `acquire`, `instrument`, `pam4` | Two-rate captures, a probe pack (loading, compensation, ground-lead ring, termination, AC coupling, overload recovery), scope/ADC effects, segmented PAM4 datasets |
| Links and systems | `rx`, `cdr`, `sparam`, `scene`, `optical`, `coding`, `bus` | Equalization, clock recovery, measured channels, multi-lane, optical, coding, UART/open-drain (incl. a wired-AND second sink), a pass-FET/analog switch, AM/ASK/OOK/FM/FSK/PM modulation |
| Dataset quality | `measure`, `sweep`, `simreal` | Measured labels, confounder control, synthetic-vs-real separability |
| Scale and trust | `stream`, `validate` | Bounded-memory channel processing and physical-property assertions |
| Long-record performance | `tools/benchmark.py`, `response_cache` | A benchmark suite gated in CI on peak memory and the scaling ratio `t(2n)/t(n)`; `apply_transfer(method="overlap")` for a block path that holds ~1.1x the record instead of ~5x; `response_cache()` to reuse a channel's minimum-phase response across a batch (`dataset()` already does) |

## Design principles
- **Physics-grounded, not hand-drawn.** Real formulas (skin+dielectric insertion loss,
  transmission-line reflection lattices, decomposed jitter), on a normalized/unitless time
  grid that's rate-parameterizable.
- **Validated.** `python -m wfmsynth.validate` checks each effect actually holds (loss lowers
  the spectral centroid, echoes decay geometrically, injected jitter RMS is recovered, the
  causal channel is minimum-phase, …). It's the "don't fool yourself" gate.
- **Label-preserving augmentation.** `domain_randomize` adds capture-condition nuisances
  (gain, DC, bandwidth, AWGN, 1/f, quantization, ambient coupling) kept *sub-threshold* to the
  fault magnitudes, so it never masquerades as a labeled impairment.
- **Composition over enumeration.** The grammar spans signal morphologies by composing
  primitives, so a model/tool tested on it generalizes beyond named protocols.

## Rates and record lengths
Primitives infer the grid from the array they're given, and generators take an optional
`n`, so nothing is locked to the default 4096-point grid — a multi-megapoint deep-memory
record works the same as a 4096-point toy. `N`/`T` remain the defaults, and default-grid
output is bit-identical to before.

```python
n = 1 << 20                                                # any record length
x = P.pam4(n_ui=n // 8, seed=1, n=n, pattern="prbs13q")    # IEEE PRBS13Q
y = P.multi_reflection(P.lossy_channel(x, length_in=8.0, causal=True),
                       td_samples=64)                      # absolute, not a fraction
```

Two notes on fidelity. `pattern="prbs13q"` gives the **IEEE 802.3 Clause 120.5.11.2.1**
sequence — use it when a capture has to pattern-lock on an instrument, since a
non-standard degree-13 polynomial has the right level statistics and still will not lock.
And `causal=True` on the carrier generators uses forward-only edge shaping; the default
zero-phase shaping is symmetric and therefore adds pre-cursor content, which is worth
knowing about in a library whose channel model is otherwise strictly causal.

## Choosing the pattern (it is not a cosmetic knob)

Channel ISI is a function of pattern **history** — the long runs and low-frequency content
are what actually close an eye through a lossy or reflective channel. `nrz` takes a
`pattern` too:

```python
x = P.nrz(n_ui=n // 8, seed=1, n=n, pattern="prbs31")   # compliance/SI: period 2**31-1
x = P.nrz(n_ui=n // 8, seed=1, n=n, pattern="clock")    # 1010...: the ISI-free contrast
```

| carrier | patterns |
|---|---|
| `nrz` | `legacy` (== `prbs7`, the default), `prbs7`, `prbs9`, `prbs11`, `prbs13`, `prbs15`, `prbs23`, `prbs31`, `clock` |
| `pam4` | `legacy` (the default), `prbs13q`, `prbs31q` |

Binary orders are spelled out and the quaternary sequences keep IEEE's `Q` suffix, so
`prbs13` and `prbs13q` can never be confused. Crossing them raises rather than coercing.

The default stays `prbs7` because this kernel is pinned by SHA and its output diffed
sample-for-sample downstream — it is a **compatibility default, not a recommendation**.
PRBS7's 127-bit period repeats 23 622 times inside a 3 M UI record and carries almost no
low-frequency content: through a 14-inch lossy channel the validation suite measures its
eye **~40 % more open** than PRBS31's on the same link. Pick `prbs31` for anything meant
to resemble compliance or SI work; a dataset built on PRBS7 teaches an impairment
signature that does not occur on a real link.

## The pattern registry — mechanisms here, standards knowledge in the caller

`wfmsynth.patterns` maps a **name** to a symbol generator. The mechanisms ship: a general
LFSR for an arbitrary feedback polynomial (the one `physics.prbs` itself runs on, so a
polynomial nobody tabulated goes through the same arithmetic as one that is), and a
block-repeat generator for a sequence a document prints rather than specifies. *Which*
pattern a given standard names is **not** here — that changes with the revision of a
document, so the caller registers it with a `source=` citation.

```python
import wfmsynth.patterns as PAT

PAT.resolve("prbs13", length=8191, seed=9)          # a shipped polynomial
PAT.resolve("lfsr", length=1023, taps=[10, 7])      # any polynomial, nothing to register

@PAT.pattern("my_link_test", levels=2, period=2048, source="<designation, clause, table>")
def _my_link_test(length, seed=1):                   # named, so a recipe can carry it
    return PAT.block_repeat(WORD, length=length)

sig = ws.Signal(seed=1, grid=g).pattern("prbs13", length=g.n // 8, seed=9).lossy(length_in=8.0)
```

**The recipe records the name *and* the resolved parameters.** Name alone cannot be replayed
by anyone without the registry entry; a bare polynomial cannot be read or diffed against the
document it came from. Both makes the JSON self-describing *and* replayable:

```json
{"op": "symbols",
 "pattern": {"name": "prbs13", "generator": "lfsr", "levels": 2, "period": 8191,
             "params": {"taps": [13, 12, 2, 1], "seed": 9, "length": 4096}}}
```

A consumer missing the `prbs13` entry still renders that record, from the `lfsr` mechanism the
block names. For caller-owned generators the block also carries a hash of the generator's
source, so a consumer with a *different* version under the same name is told
`needs pattern 'x' @ <hash>` instead of quietly getting other samples. And
`pattern(..., embed=True)` writes the resolved symbols into the recipe — **symbols as data**,
reproducible with no generator code at all, which is the route that always works.

There are therefore no lambdas in a recipe. `physics.arbitrary(fn=...)` is the exception that
proves it: a callable cannot be serialised, hashed or replayed, `compose` refuses to store one,
and its docstring points here.

## Absolute units (real ps / Hz / dB / V)
Bind the abstract grid to real units with `Grid`, then specify parameters the way an
engineer would — delays in ps, jitter in seconds, corner/periodic-jitter frequencies in
Hz, channel loss in dB at a stated frequency. Omit the grid and the fraction/sample forms
are unchanged.

```python
g = ws.Grid(fs=256e9, baud=112e9, n=1 << 16)     # 256 GSa/s, 112 GBd, ~65k points
g.samples_per_ui                                  # 2.286  (non-integer, as in reality)
x = P.pam4(n_ui=g.n // 8, seed=1, n=g.n, pattern="prbs13q")
x = P.lossy_channel(x, loss_db=15.0, loss_at_ghz=26.0, grid=g, causal=True)  # 15 dB @ 26 GHz
x = P.multi_reflection(x, td_ps=55.0, grid=g)                                 # echo at 55 ps
x = P.inject_jitter(x, sigma_rj_s=300e-15, f_pj_hz=4e6, grid=g)              # 300 fs Rj + 4 MHz Pj
```

## Provenance & reproducible datasets
Compose a signal as an explicit component graph; every waveform carries a serializable
**recipe** — exact ground truth, fully reproducible.

```python
sig = (ws.Signal(seed=42, grid=g)
       .carrier("pam4", n_ui=g.n // 8, pattern="prbs13q", jitter=dict(rj=0.4))
       .lossy(loss_db=15.0, loss_at_ghz=26.0, causal=True)
       .reflect(td_ps=55.0).digitize(snr_db=32.0, enob=5.5, interleave=dict(m_cores=4)))
x, recipe = sig.waveform(), sig.recipe()          # samples + JSON-able provenance
assert (ws.Signal.from_recipe(recipe).waveform() == x).all()   # bit-for-bit round-trip

X, recipes = ws.dataset(build, n=10_000)          # each example labelled to arbitrary depth
```

**Contrastive pairs / ablations.** Each random factor is a separate stream, so you can
re-roll exactly one and hold the rest bit-identical:

```python
a = sig.waveform()
b = sig.contrast("noise/1")     # same symbols, same jitter, same channel — only noise differs
sig.roles()                     # -> the re-rollable factors, e.g. ["jitter/0", "noise/1"]
```

## Confounder-controlled sweeps & measured labels
A naive reflection sweep is also an eye-height sweep — hold the shortcut constant so a
model learns ISI structure, not eye height. Metrics are **measured from the output**.

```python
from wfmsynth import eye_height, hold_constant, attributes, realized_table
target = eye_height(build(gamma=0.05, loss_db=2.0).waveform(), g)
recs = hold_constant(build, "gamma", [0.05,0.15,0.25,0.35], "eye", target,
                     "loss_db", (0.0,4.0), g, eye_height)   # solve loss to pin the eye
# each record carries realized_eye; loss must drop as reflection rises (a real constraint)
recs, corr, names = realized_table(build, sets, g, attributes)   # realized labels + leak matrix
```

## Ground truth as measured
Labels are **measured from the output**, not read off the knobs. Eye height comes in two
named definitions (they diverge under deterministic ISI), and per-symbol labels carry the
realized integer-symbol alignment (a causal channel has group delay).

```python
gt = sig.ground_truth()
# {eye_contour, eye_sigma, best_phase, align_offset, align_corr, align_corr_at_zero, ...}
ws.eye_height(x, g, defn="contour")   # measured opening
ws.eye_height(x, g, defn="sigma")     # 3-sigma construction
```

## Cookbook — one recipe per capability

Everything above is orientation. The per-feature recipes — the instrument and converter model,
equalisation, optical, events, crosstalk, coding, buses, acquisition, and the rest — live in
**[docs/COOKBOOK.md](docs/COOKBOOK.md)**, one short section each with runnable code.

They were in this file, which had grown to 867 lines and 47 sections: a reference manual wearing
a README. Splitting them leaves this page as the thing someone reads once, and the cookbook as
the thing they come back to.

## Roadmap and backlog

**[docs/COMPATIBILITY.md](docs/COMPATIBILITY.md)** states what will not break, and what a
change that moves rendered samples has to do before it ships. Read it before upgrading a
pipeline that has already generated data.

**[docs/FIDELITY.md](docs/FIDELITY.md)** says which of this library's numbers are calibrated
against real captured hardware, which are closed form, and which are fitted shapes — using the
evidence markers `DATASET-METHODOLOGY.md` defines. Read it before trusting a figure.

**[ROADMAP.md](ROADMAP.md)** describes future direction from the current composable,
provenance-first architecture. **[BACKLOG.md](BACKLOG.md)** contains only active work,
known limitations, and concise delivered milestones. Current priorities are consolidating
parallel acquisition/digitization paths, reporting realized rise time, and closing specific
standards-fidelity gaps.

## Tests
```bash
python -m pip install -e ".[test]"
pytest                                       # ~1250 tests
python -m wfmsynth.validate                  # hard physics-property assertions
python tools/benchmark.py --quick --check    # peak memory and complexity, not wall time
```

## Contributing

**[CONTRIBUTING.md](CONTRIBUTING.md)** has the gates, the definition of done, and the rules that
are specific to a library whose output is training data — chiefly what to do when a change moves
rendered samples. **[CHANGELOG.md](CHANGELOG.md)** records what moved between versions; releases
are tagged `v*`. See also [SECURITY.md](SECURITY.md) and
[CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).

## License
**0BSD** (Zero-Clause BSD) — see [LICENSE](LICENSE). Maximally permissive: use, copy, modify, and
distribute for any purpose, with **no attribution requirement** and no conditions. Public-domain-equivalent.
