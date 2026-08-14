# wfmsynth — prioritized backlog

`ROADMAP.md` is the **breadth** map: everything this engine could eventually model.
This file is the **depth** list: the next actionable pieces, in priority order, each
with why it matters and what "done" looks like.

**Definition of done, for anything here:** a hard assertion in `wfmsynth.validate`
that the physical property actually holds. That gate is the reason this library is
trustworthy; an item without one isn't finished.

**Priority principle:** fidelity to real captures first, then ground-truth quality,
then breadth. An effect that shows up the moment you compare synthetic against a
real acquisition outranks an effect that is merely absent.

**Compatibility principle:** this is used as a dependency. Prefer additive changes
behind existing defaults; when output could change, add an opt-in flag, say so in the
docstring, and add a test that the legacy path is bit-identical.

---

## P0 — blocks realistic use

### 1. Absolute units and rate binding — ✅ DONE (v0.2)
`wfmsynth.grid.Grid(fs, baud, n, v_full)` binds the abstract grid to real units;
primitives take it via `grid=` plus an absolute-unit kwarg: `multi_reflection(td_ps=)`,
`lossy_channel(loss_db=, loss_at_ghz=)`, `inject_jitter(sigma_rj_s=/a_pj_s=/f_pj_hz=/dcd_s=)`,
`ac_couple(fc_hz=)`. Fraction/sample/cycles-per-record forms remain the defaults
(bit-identical). `validate` asserts a ps delay and a dB@freq loss realize as requested.

Everything lives on a normalized, unitless grid: amplitudes are ±1, time is [0,1),
delays are fractions of the record, frequencies are cycles-per-record, and insertion
loss is expressed as inches × loss tangent. Record length is now parameterizable, but
a caller still cannot state a symbol rate, a sample rate, a capture duration or a
full-scale voltage — so a real link cannot be expressed, and every derived quantity
has to be reinterpreted by hand.

`multi_reflection(td_samples=...)` is a first step: an absolute delay, because a
discontinuity sits a fixed number of picoseconds away regardless of how long you
acquire for. The same treatment is needed throughout.

**Proposal.** A small immutable descriptor threaded through the primitives:

```python
Grid(fs=..., baud=..., n=..., v_full=...)
# derived: dt, ui_seconds, samples_per_ui (often non-integer), f_nyquist
```

Then parameters can be given in the units engineers use — delays in ps, jitter in fs,
periodic-jitter frequency in Hz, bandwidths in GHz, loss in dB at a stated frequency —
with the fraction-of-record forms kept as defaults so nothing breaks.

**Why it is P0 and blocks other items:** provenance recipes (#5) are only portable
between tools if the knobs are in real units, and every fidelity item below wants to
be specified physically rather than as a fraction of an arbitrary window.

**Done when:** a primitive accepts a delay in ps and a loss in dB at a stated
frequency, and validate asserts the realized values match the requested ones after
round-tripping through the grid.

### 2. Non-integer samples-per-UI as the norm — ✅ DONE (v0.2)
`Grid.pattern_period_samples(n_ui)` gives the exact fractional pattern period; carriers already sample-and-hold correctly at non-integer sps (see `examples/non_integer_sps.py`); `validate` proves the true-sps recovery is exact while an integer-sps assumption drifts.

`pam4.py` uses `SPS_INT = 16` internally and 4 samples/UI at the scope. Real
acquisitions are almost never an integer number of samples per symbol — the sample
clock and the symbol clock are unrelated. Tools developed against integer-sps
synthetic data acquire hidden assumptions and then break on measured data, which is
exactly the failure this library exists to prevent.

Consequence worth exposing explicitly: with non-integer sps the **repeating pattern
period is fractional** in samples. Anything that averages or folds repetitions needs
sub-sample realignment, and a downstream tool that assumes an integer period will be
subtly wrong rather than obviously wrong.

**Done when:** carriers accept non-integer samples-per-UI, it is the default in at
least one worked example, and any emitted ground truth includes the exact fractional
pattern period.

### 3. Instrument model: time-interleaved ADC artifacts — ✅ DONE (v0.2)
`wfmsynth.instrument.interleave_adc(m_cores, gain_mm, offset_mm, skew_mm)` injects per-core mismatch -> spurs at k*fs/M (offset) and images at k*fs/M±f_in (gain/skew); plus `shaped_noise_floor` (coloured floor) and `clip_adc` (saturation + clipped mask). `validate` asserts the spurs appear with mismatch and vanish at zero. (Timebase/trigger jitter: follow-on.)

`pam4._digitize` models thermal noise + finite ENOB. A real high-speed front end is a
**time-interleaved** ADC, and interleaving is much of what makes a real capture look
real:

- per-core **gain**, **offset** and **timing skew** mismatch → spurs at `f_s/M` and
  image tones mirrored about `f_s/(2M)`
- a **frequency-shaped** noise floor rather than a flat one
- **timebase / trigger jitter** on the sampling clock itself
- amplitude **clipping** behaviour at full scale, and how clipped samples are marked

If you train on synthetic and test on measured, perfectly white Gaussian noise plus
perfectly uniform sampling is a strong candidate for the first thing a model notices.
Cheap to add, large fidelity payoff.

**Done when:** validate asserts that with `M` cores and injected mismatch, spurs
appear at the predicted frequencies and vanish when mismatch is zero.

### 4. Apply jitter at the source, not by warping the output — ✅ DONE (v0.2)
`physics.Jitter(rj, pj, f_pj, dcd)` (or `Jitter.at(grid, rj_s=, ...)` for seconds/Hz); carriers take `jitter=` and displace the symbol EDGE TIMES before shaping, so DDJ emerges from the channel and post-channel noise is untouched. Legacy `inject_jitter` (output warp) kept for back-compat. `validate` asserts recovered edge-RMS ~ injected AND post-channel noise stays exactly additive (with a contrast showing the output-warp path corrupts it).

`inject_jitter` warps the time axis of an already-generated waveform and resamples.
Two problems:

- run after additive noise, it **jitters the noise too**, which is not physical
- run after the channel, it **smears the ISI the channel just created** rather than
  displacing the transmitted edges that then propagate through it

Physically, jitter belongs to the transmitter's edge times: perturb the symbol edge
positions, *then* shape and propagate. DDJ then emerges from the channel for free
instead of needing to be injected separately.

**Done when:** carriers accept jitter parameters directly, and validate asserts that
recovered edge-time RMS matches the injected value **and** that noise added after the
channel is uncorrelated with the injected jitter.

---

## P1 — needed for ground-truth-grade datasets

### 5. Provenance-first composable synthesis — ✅ DONE (v0.3)
`wfmsynth.compose.Signal(seed, grid).carrier(...).lossy(...).reflect(...).digitize(...)` builds a waveform + a JSON-serializable `recipe()` (engine version stamped). `Signal.from_recipe(r).waveform()` reproduces bit-for-bit (asserted in validate). `dataset(build, n)` returns (waveforms, recipes[]) with each sampled knob recorded.

The `ROADMAP.md` flagship; the design sketched there is the right shape. Two additions:

- It becomes materially more valuable after #1, because a recipe in real units is
  portable and a recipe in fractions-of-record is not.
- **Round-trip must be a test, not an aspiration:** `from_recipe(recipe).waveform()`
  reproducing bit-for-bit, including across an engine version bump, is the property
  that makes datasets trustworthy. Stamp the engine version in the recipe and assert
  the round trip in validate.

### 6. RNG stream roles — ✅ DONE (v0.4)
`wfmsynth.streams.Streams(seed)` gives each factor (symbols/jitter/noise/per-impairment/
interleave) an independent stream keyed by `(seed, role-name)` — order-free, so adding a
role never disturbs existing ones. `Signal.contrast(*factors)` returns a sibling with only
those factors re-rolled and everything else bit-identical (valid contrastive pairs /
clean ablations). Also adds an absolute `noise_rms` floor. validate asserts a changed
upstream factor leaves the downstream noise realization bit-for-bit identical.

Currently a single `rng` threads through generation, so any change to one factor
re-rolls everything downstream of it. For dataset construction that is a correctness
problem, not an inconvenience.

Tag randomness by **role** — symbols, jitter, thermal noise, per-impairment,
capture nuisances — each with its own stream. Then a sibling waveform can re-roll
exactly one factor and leave everything else bit-identical.

This is what makes valid **contrastive pairs** and clean **ablations** possible: "same
data, same jitter, same noise, different channel" is only true if the streams are
separated. Without it, a pair that looks controlled differs in every factor at once,
and any conclusion drawn from it is confounded.

**Done when:** two waveforms differing in one declared factor are bit-identical
outside that factor's contribution, asserted in validate.

### 7. Confounder-controlled sweeps, and realized-vs-requested labels
The single most valuable addition if the output is training data, and not currently
anywhere in the roadmap.

When you sweep an attribute to build a labeled set you have to hold the obvious
shortcut constant, or a model learns the shortcut and scores well for the wrong
reason. Concretely: adding a reflection closes the eye, so a naive reflection sweep is
*also* an eye-height sweep, and anything trained on it can read eye height instead of
ISI structure.

Two capabilities:

- **Hold-constant sweeps.** Declare a metric to pin and a free parameter to solve for:
  *"sweep the reflection coefficient across its range, bisecting insertion loss to
  hold eye height fixed."* This exposes a real physical constraint — you cannot
  independently vary reflection, loss **and** eye height, only two of the three — so
  the API should make the caller choose which two are free rather than silently
  picking one.
- **Realized, not requested, labels.** With deep composition the knob → outcome map is
  many-to-many, and nominally orthogonal knobs produce correlated *realized*
  attributes. Emit measured attribute values alongside the requested ones, plus the
  realized correlation matrix across a generated set, so a leak is visible instead of
  latent.

**Done when:** a sweep builder returns realized values, and validate asserts a
hold-constant sweep keeps its pinned metric within tolerance while the swept attribute
moves monotonically.

### 8. Ground truth as measured, not as ideal
In `ROADMAP.md` under "Ground-truth co-generation". Three sharpenings:

- Emit values **measured from the generated output**, not computed from the knobs.
  They differ, and the difference is precisely where silent label noise comes from.
- **State the metric definition, and offer more than one.** Eye height from a σ-based
  formula and eye height from a measured contour agree for roughly Gaussian vertical
  distributions and **diverge badly** when ISI is deterministic — a discrete echo
  produces distinct ISI branches, not a Gaussian smear, so a 3σ construction
  overstates the closure. A library emitting eye metrics as ground truth should name
  the definition and ideally provide both.
- Emit the **realized symbol alignment**. A minimum-phase channel has group delay, so
  any per-symbol statistic needs an integer-symbol realignment between the transmitted
  stream and the sampled output. A consumer that skips it computes closed eyes while
  everything looks reasonable, so the offset belongs in the ground truth.

### 9. Impairment mixing at constant total power
A general, reusable primitive: combine several impairment kinds in declared
proportions while holding **total** impairment power fixed, by summing in quadrature
with weights `√w_i`. That separates *how much* impairment there is from *what kind* it
is, which is the distinction any character-versus-magnitude attribute needs.

Useful well beyond one project: it is the natural API for "same SNR, different noise
character" datasets, and doing it by hand is easy to get wrong at the endpoints.

**Done when:** validate asserts total impairment RMS is invariant across the mixing
sweep while a character statistic moves monotonically.

### 10. Time-varying and intermittent impairments
`ROADMAP.md` lists "burst/intermittent faults". Worth promoting, because intermittency
is where stationary analysis genuinely fails rather than merely being imprecise:

- A record-**average** analysis dilutes an intermittent defect roughly in proportion
  to its duty cycle, and below a few percent duty it disappears into the noise floor.
- Worse, standard decompositions tend to **misattribute** time variation into the
  *random noise* bucket, because variation across repetitions is what those estimators
  interpret as randomness. A defect that is deterministic but intermittent can be
  reported as a large noise figure, which points a user at entirely the wrong fix.

Implementation note worth recording: a gated impairment can be applied **exactly**
rather than approximately, provided the impairment is linear and additive noise is
applied after the channel — build the impairment-free response once, then add gated,
integer-sample-delayed copies. Use a smooth (raised-cosine) gate so the onset does not
itself read as an edge.

**Done when:** validate asserts the impairment is confined to its gate (negligible
leakage outside), and per-sample defect masks are emitted as ground truth.

### 11. Tx FFE / Rx CTLE / DFE
In `ROADMAP.md`. Priority note: **Tx FFE is the highest-fidelity one.** Real
high-speed transmitters almost always run multi-tap FFE, which puts a deliberate
*pre-cursor* in the pulse response. A synthetic waveform with no pre-emphasis has a
qualitatively different pulse-response shape from anything on a real link — and pulse
response is the first thing an engineer looks at.

### 12. Clock recovery as part of "what the scope records"
An instrument does not show you the raw record; it shows you the record folded by a
recovered clock. The CDR choice — constant-frequency versus first/second/third-order
PLL, and its loop bandwidth — materially changes what an eye looks like and how much
low-frequency jitter is tracked out. If the library models the front end, the recovery
that follows it belongs in the same model, and an emitted eye is only meaningful
alongside the recovery that produced it.

---

## P2 — breadth and fidelity

### 13. Measured S-parameter channels (Touchstone)
In `ROADMAP.md`. Largest realism-per-effort item after the instrument model: the
analytic √f + f loss model cannot produce resonances, fibre-weave periodicity,
connector structure, or interacting multiple bounces. Real `.sNp` files give all of it.
Keep the analytic model as the dependency-free default.

### 14. Reflection realism beyond a flat coefficient
`multi_reflection` uses a real, frequency-flat Γ. Real discontinuities are **resonant**
— frequency-dependent magnitude *and* phase — and a stub behaves like a resonator
rather than a mirror. Multiple interacting discontinuities also produce products the
single-lattice model does not.

### 15. Level nonlinearity in the nominal case
`deep_capture` has an `rlm_compression` *fault*, but the base carrier is exactly
linear — level ratios are perfect to numerical precision. Real transmitters are
imperfect **always**, not only when faulted. Add nominal level nonlinearity,
level-dependent noise, and rise/fall asymmetry so "nominal" is not suspiciously
perfect. A too-perfect nominal class is itself a giveaway.

### 16. Multiple aggressors from a coupling matrix
In `ROADMAP.md`. Add: aggressors should be **asynchronous by default**. A synchronous
aggressor produces interference locked to the victim's clock, which a receiver's CDR
partly tracks out and which looks like ISI rather than crosstalk — so a synchronous
default quietly makes crosstalk easier to detect than it really is.

### 17. Noise realism beyond white Gaussian
Non-Gaussian tails, 1/f at low frequency, level-dependent and shot-noise terms. Real
noise is not one flat Gaussian, and anything learning a noise-character axis will key
on the difference.

### 18. Composition-level causality assertion
Causality is asserted for the channel primitive, but not for a **composed** chain —
and the default zero-phase edge shaping reintroduces pre-cursor content *after* the
causal channel has been applied. That is a composition hazard: each stage looks fine
in isolation while the pipeline is not causal end to end.

**Done when:** validate asserts near-zero pre-cursor energy for a full composed chain,
not just for `lossy_channel` alone.

### 19. Pattern lock-ability as a validated property
PRBS statistics are now asserted, which catches a wrong polynomial. Going further: for
any carrier claiming a standard pattern, assert the symbol sequence autocorrelates to a
single sharp peak at the declared period. That is the property an analyser actually
depends on, and it is cheap to check.

### 20. Streaming / chunked generation
A multi-megapoint record needs several float64 arrays live at once, and the
frequency-domain channel needs a full-length FFT — the memory bottleneck. Overlap-add
or overlap-save convolution plus chunked or memory-mapped output would remove it.
Deep-memory captures are the headline use case, so this matters before anyone reaches
for a hundred-megapoint record.

### 21. Sim-to-real separability harness
A tool rather than a physics primitive, but it is what tells you whether any of the
above matters. Given a set of synthetic waveforms and a set of real captures, compute a
common feature vector and ask whether a trivial classifier can separate them — and if
so, **which feature does it use?** That names the missing physics instead of guessing
at it, and turns this whole list into an evidence-ordered work queue.

Listed at P2 only because it needs real captures to be useful. In value terms it
outranks most of this file.

### 22. Critical sampling — document the hazard
*(Documentation, not code.)* Deserves a README paragraph because it is non-obvious and
it silently degrades downstream analysis: if the modelled front-end bandwidth sits at
or near the Nyquist frequency of the sample rate, the capture is **critically
sampled**, and any analysis needing sub-sample interpolation becomes lossy.

The case that bites: pattern-averaging has to realign repetitions to a common grid, and
with non-integer samples-per-UI (#2) the period is fractional. Sub-sample realignment
is exact only well below Nyquist, so near critical sampling the deterministic part
fails to cancel and the residual is dominated by leftover *signal* rather than by the
impairment you were trying to isolate — which reads as a mysteriously large noise
floor rather than as an aliasing problem.

### 23. CI across platforms
The validation suite is the trust anchor, so it should be proven to run everywhere. It
previously died partway through on a stock Windows console — cp1252 could not encode a
maths glyph in a detail string, which looks like a hang and hides every check after it.
A CI matrix over Linux/macOS/Windows would have caught it.

---

## Explicitly out of scope

Kept out to keep this package *pure synthesis*. From `ROADMAP.md`, plus one addition:

- **Signal-integrity measurement/analysis** (eye, level-ratio, SNR, TIE extraction from
  a waveform) — an analysis library. Note the tension with #8: co-generating measured
  ground truth needs *some* measurement. Keep only what is needed to label what this
  library generates, and resist growing a general analyzer.
- **Eye-diagram / density rendering** — a visualization utility; adds a plotting dep.
- **Vendor instrument file containers.** Reading and writing a specific instrument's
  native capture format is coupled to that vendor's undocumented layout and versioning.
  Export generic, well-specified formats (`.npz`, Parquet, Touchstone, plain HDF5) and
  leave vendor containers to whatever tool talks to that instrument.

---

## Contributing an item

1. Add it here with a priority, a rationale, and a "done when".
2. Implement behind existing defaults where possible; if output could change, add an
   opt-in flag and say so in the docstring.
3. Add the validation assertion, plus a test that the legacy path is bit-identical.
