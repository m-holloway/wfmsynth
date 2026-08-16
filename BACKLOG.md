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

### 7. Confounder-controlled sweeps, and realized-vs-requested labels — ✅ DONE (v0.5)
`wfmsynth.measure` measures attributes FROM the output (eye height under two named
definitions: 'contour' and 'sigma'). `wfmsynth.sweep.hold_constant(...)` sweeps one knob
while pinning a measured metric by solving a second knob (bisection), returning REALIZED
values — the caller chooses which two of {swept, solved, pinned} are free. `realized_table`
emits realized labels + the realized correlation matrix so a leak is visible. validate
asserts the pin holds within tolerance while the compensation moves monotonically.
(Alternate-eye-definition divergence and symbol alignment are sharpened in #8.)

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

### 8. Ground truth as measured, not as ideal — ✅ DONE (v0.6)
`wfmsynth.measure` emits labels measured FROM the output: eye height under two named
definitions ('contour' = measured opening, 'sigma' = 3-sigma construction) that agree
under Gaussian noise and diverge under deterministic ISI; `align_symbols`/`Signal.
ground_truth()` recover the realized integer-symbol offset (a causal channel has group
delay) by reconstructing the transmitted stream via `physics.carrier_symbols`. validate
asserts the definitions diverge under ISI, and that the recovered offset is nonzero and
essential (skipping it collapses tx/output correlation).

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

**Follow-on, from using it:** `eye_height(defn='sigma')` can return a NEGATIVE
value while `defn='contour'` returns a healthy positive one on the same waveform
(measured -0.086 vs +0.107). That is not a bug -- it is the definitional
divergence this item is about, and it is sharper than expected: the two
definitions can disagree in SIGN, not merely in magnitude, because a sigma
construction folds deterministic ISI into a Gaussian width and can drive the
result below zero. Worth stating in the docstring, since a consumer treating
eye height as necessarily positive will mis-handle it, and worth having
`ground_truth` carry both definitions side by side rather than a single number.

### 9. Impairment mixing at constant total power — ✅ DONE (v0.8)
`impairments.mix_at_constant_power(components, weights, total_rms)` combines impairment
components in declared proportions (powers add in quadrature) and rescales to an exact
total RMS — separating *how much* impairment from *what kind*. validate asserts total RMS
is invariant across a mixing sweep while a character statistic (spectral centroid) moves
monotonically.

A general, reusable primitive: combine several impairment kinds in declared
proportions while holding **total** impairment power fixed, by summing in quadrature
with weights `√w_i`. That separates *how much* impairment there is from *what kind* it
is, which is the distinction any character-versus-magnitude attribute needs.

Useful well beyond one project: it is the natural API for "same SNR, different noise
character" datasets, and doing it by hand is easy to get wrong at the endpoints.

**Done when:** validate asserts total impairment RMS is invariant across the mixing
sweep while a character statistic moves monotonically.

### 10. Time-varying and intermittent impairments — ✅ DONE (v0.9)
`impairments.burst_gate(n, intervals, edge_frac)` builds a raised-cosine per-sample gate;
`impairments.apply_gated(x, impairment, intervals)` applies an impairment (array or
callable) only within the gate and returns `(y, mask)`. The gate is exact — `y` is
bit-identical to `x` outside it — and the mask is the ground-truth 'where the defect is
active'. validate asserts zero leakage outside the gate, the defect present inside, and a
smooth (non-step) onset. (Ties into #6 masks-as-ground-truth for intermittency.)

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

### 11. Tx FFE / Rx CTLE / DFE — ✅ Tx FFE DONE (v0.10); CTLE/DFE remain
`physics.tx_ffe(x, taps, spb, pre=)` + `Signal.tx_ffe(taps, pre=)` apply a T-spaced
transmitter FFE (non-integer spb via fractional delay), placed after the carrier and
before the channel. Puts a deliberate pre-cursor in the pulse response and de-emphasizes
post-cursor ISI. validate asserts a pre-cursor appears one UI before the main pulse and
that FFE opens a lossy-channel eye. Rx CTLE (analog peaking) and DFE (decision feedback)
still to add — logged as #28.

ORIG:
In `ROADMAP.md`. Priority note: **Tx FFE is the highest-fidelity one.** Real
high-speed transmitters almost always run multi-tap FFE, which puts a deliberate
*pre-cursor* in the pulse response. A synthetic waveform with no pre-emphasis has a
qualitatively different pulse-response shape from anything on a real link — and pulse
response is the first thing an engineer looks at.

### 12. Clock recovery as part of "what the scope records" — ✅ DONE (v0.13)
`wfmsynth.cdr.recover_clock(phase, baud, loop_bw, order, damping)` is the standard
linearized-PLL jitter transfer: returns the recovered-clock phase (low-pass) and the
residual timing the eye actually shows (high-pass, corner ~loop_bw). `jitter_transfer`
and `tracked_out_fraction` are conveniences. validate asserts the transfer is high-pass
(sub-BW jitter tracked out, supra-BW passed), a wider loop BW tracks out more, and a
2nd-order (type-2) loop tracks a frequency offset to ~zero while 1st-order leaves a lag.
Waveform-level fold/resample-onto-recovered-clock logged as #29.

An instrument does not show you the raw record; it shows you the record folded by a
recovered clock. The CDR choice — constant-frequency versus first/second/third-order
PLL, and its loop bandwidth — materially changes what an eye looks like and how much
low-frequency jitter is tracked out. If the library models the front end, the recovery
that follows it belongs in the same model, and an emitted eye is only meaningful
alongside the recovery that produced it.

---

## P2 — breadth and fidelity

### 13. Measured S-parameter channels (Touchstone) — ✅ DONE (v0.14)
`wfmsynth.sparam`: `read_touchstone`/`write_touchstone` (.sNp, RI/MA/DB, Hz..GHz, the
2-port ordering quirk handled), `sparam_channel(x, freqs, s21, grid)` and
`touchstone_channel(x, path, grid)` apply a measured Sij as a frequency-domain channel;
`Signal.sparam(path=... | freqs=,s21=)` composes it. Reproduces resonances/structure the
analytic model can't. Analytic model stays the dependency-free default. validate asserts
a .s2p round-trip and a resonant notch in the channel output.

In `ROADMAP.md`. Largest realism-per-effort item after the instrument model: the
analytic √f + f loss model cannot produce resonances, fibre-weave periodicity,
connector structure, or interacting multiple bounces. Real `.sNp` files give all of it.
Keep the analytic model as the dependency-free default.

### 14. Reflection realism beyond a flat coefficient — ✅ DONE (v0.15)
`physics.resonant_reflection(x, grid, td_ps, f0_ghz, q, gamma0)` + `Signal.resonant_reflect()`
model a discontinuity whose reflection coefficient Γ(f) is frequency-dependent (a 2nd-order
band-pass magnitude peaking at f0 with quality q, plus phase) — a stub/open that resonates
rather than mirroring flatly. validate asserts the reflected content peaks near f0, unlike
the flat-Γ multi_reflection. (Interacting multi-discontinuity products still via #13 S-params.)

`multi_reflection` uses a real, frequency-flat Γ. Real discontinuities are **resonant**
— frequency-dependent magnitude *and* phase — and a stub behaves like a resonator
rather than a mirror. Multiple interacting discontinuities also produce products the
single-lattice model does not.

### 15. Level nonlinearity in the nominal case — ✅ DONE (v0.16)
`physics.nominal_nonlinearity(x, compression, level_noise, rise_fall_ratio)` +
`Signal.nonlinearity()` add always-on transmitter imperfections: soft level compression
(PAM4 spacing non-uniform, RLM<1), level-dependent noise (outer levels noisier), and
rise/fall-time asymmetry — so the unfaulted class isn't suspiciously perfect. validate
asserts all three (mild RLM<1, outer>inner noise, rise!=fall).

`deep_capture` has an `rlm_compression` *fault*, but the base carrier is exactly
linear — level ratios are perfect to numerical precision. Real transmitters are
imperfect **always**, not only when faulted. Add nominal level nonlinearity,
level-dependent noise, and rise/fall asymmetry so "nominal" is not suspiciously
perfect. A too-perfect nominal class is itself a giveaway.

### 16. Multiple aggressors from a coupling matrix — ✅ DONE (v0.17)
`physics.crosstalk_matrix(x, grid, couplings, ...)` + `Signal.crosstalk_matrix()` add
crosstalk from several aggressors weighted by a coupling vector, ASYNCHRONOUS by default
(each at a slightly offset baud so its timing isn't locked to the victim clock). Pass
synchronous=True for the locked case. validate asserts power scales with the coupling
vector and that the async default is NOT locked to the victim UI while synchronous is.

In `ROADMAP.md`. Add: aggressors should be **asynchronous by default**. A synchronous
aggressor produces interference locked to the victim's clock, which a receiver's CDR
partly tracks out and which looks like ISI rather than crosstalk — so a synchronous
default quietly makes crosstalk easier to detect than it really is.

### 17. Noise realism beyond white Gaussian — ✅ DONE (v0.18)
`impairments.realistic_noise(n, rms, df, pink_frac)` — Student-t HEAVY TAILS (df -> large
recovers Gaussian) optionally mixed with a 1/f (pink) fraction for low-frequency structure.
validate asserts clear excess kurtosis vs Gaussian and a 1/f low>>high power split.
(Level-dependent noise is in physics.nominal_nonlinearity #15; a Poisson shot term logged #30.)

Non-Gaussian tails, 1/f at low frequency, level-dependent and shot-noise terms. Real
noise is not one flat Gaussian, and anything learning a noise-character axis will key
on the difference.

### 18. Composition-level causality assertion — ✅ DONE (v0.11)
validate now asserts near-zero pre-cursor energy for a FULL composed chain (causal edge
shaping + causal channel), not just `lossy_channel` alone, and demonstrates the hazard:
default zero-phase edge shaping leaks pre-cursor behind a causal channel. Build causal
chains with `causal=True` on the carrier (README notes the hazard).

ORIG:
Causality is asserted for the channel primitive, but not for a **composed** chain —
and the default zero-phase edge shaping reintroduces pre-cursor content *after* the
causal channel has been applied. That is a composition hazard: each stage looks fine
in isolation while the pipeline is not causal end to end.

**Done when:** validate asserts near-zero pre-cursor energy for a full composed chain,
not just for `lossy_channel` alone.

### 19. Pattern lock-ability as a validated property — ✅ DONE (v0.12)
`measure.pattern_period(symbols, max_lag)` returns the detected period via an overlap-
normalized FFT autocorrelation (the peak an analyser pattern-locks onto). validate asserts
the PRBS13Q symbol sequence locks to a single sharp peak at its declared period (8191),
dominant over the next off-peak.

PRBS statistics are now asserted, which catches a wrong polynomial. Going further: for
any carrier claiming a standard pattern, assert the symbol sequence autocorrelates to a
single sharp peak at the declared period. That is the property an analyser actually
depends on, and it is cheap to check.

### 20. Streaming / chunked generation — ✅ DONE (v0.19)
`wfmsynth.stream`: `stream_convolve(x, h, chunk)` and `stream_blocks(x, h, chunk)` apply a
channel FIR by overlap-save FFT convolution, holding only ~chunk+len(h) samples live (no
full-length FFT). `channel_fir(apply_fn, n_taps)` extracts a linear channel's impulse
response for use with them. validate asserts the streamed result equals the full linear
convolution, is produced in chunks, and reproduces a lossy channel in the interior.

A multi-megapoint record needs several float64 arrays live at once, and the
frequency-domain channel needs a full-length FFT — the memory bottleneck. Overlap-add
or overlap-save convolution plus chunked or memory-mapped output would remove it.
Deep-memory captures are the headline use case, so this matters before anyone reaches
for a hundred-megapoint record.

### 21. Sim-to-real separability harness — ✅ DONE (v0.20)
`wfmsynth.simreal`: `feature_vector(x)` (amplitude/shape/spectral descriptors) and
`separability(set_a, set_b)` — a trivial per-feature rank-AUC classifier (numpy only, no
sklearn) that reports whether two sets are distinguishable AND which feature separates them
(hence the missing physics). validate asserts same-distribution sets are ~indistinguishable
(best AUC<0.75) while an added-noise difference is separable and a culprit is named. Feed it
real captures to evidence-order the rest of this backlog.

A tool rather than a physics primitive, but it is what tells you whether any of the
above matters. Given a set of synthetic waveforms and a set of real captures, compute a
common feature vector and ask whether a trivial classifier can separate them — and if
so, **which feature does it use?** That names the missing physics instead of guessing
at it, and turns this whole list into an evidence-ordered work queue.

Listed at P2 only because it needs real captures to be useful. In value terms it
outranks most of this file.

### 22. Critical sampling — document the hazard — ✅ DONE (v0.20)
README now has a **Critical sampling** section: if the modelled front-end bandwidth sits at
or near Nyquist, sub-sample interpolation (needed to realign fractional-period repetitions,
#2) becomes lossy, so pattern-averaging leaves a residual dominated by leftover signal that
reads as a mysteriously large noise floor rather than an aliasing problem.

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

### 23. CI across platforms — ✅ DONE (v0.21)
`.github/workflows/ci.yml` runs pytest + `python -m wfmsynth.validate` on a matrix of
{ubuntu, macos, windows} × Python {3.9, 3.11, 3.12} on every push/PR. The validate gate is
the declared trust anchor, so it is proven to run everywhere (the Windows cp1252 glyph issue
that once hid every check after it is already handled by a UTF-8 stdout reconfigure).

The validation suite is the trust anchor, so it should be proven to run everywhere. It
previously died partway through on a stock Windows console — cp1252 could not encode a
maths glyph in a detail string, which looks like a hang and hides every check after it.
A CI matrix over Linux/macOS/Windows would have caught it.

---

## Known bugs

### B2. `tr_frac` is silently ignored at realistic samples-per-UI
`_shape_edges` is called with `max(tr_frac * spb, 2)` samples. At scope-realistic
sample rates `spb` is small, so `tr_frac * spb` falls below the floor of 2 and the
requested rise time has **no effect at all**. Measured at 4.8188 samples/UI:

| `tr_frac` | requested | realized |
|---|---|---|
| 0.02 | 0.10 samples | 2.39 samples (0.496 UI) |
| 0.05 | 0.24 samples | 2.39 samples (0.496 UI) |
| 0.10 | 0.48 samples | 2.39 samples (0.496 UI) |
| 0.15 | 0.72 samples | 2.39 samples (0.496 UI) |
| 0.30 | 1.45 samples | 2.39 samples (0.496 UI) |

A 15x range of the parameter produces byte-identical rise times, with no warning.

**Why this matters more than it looks.** The realized rise time is ~0.5 UI, which
is an enormous transmitter band limit, and it *dominates the ISI* — so a caller who
asks for sharp edges and a 3 dB channel gets a waveform whose post-cursor is set by
the edge shaping instead. Measured on a matched comparison at a requested 3 dB
insertion loss: first post-cursor tap 0.599 and an essentially closed eye
(contour eye height 8e-06), where the same requested loss with near-ideal edges
gives a post-cursor of ~1e-04. The channel model is being swamped by an artifact of
the shaping stage.

It also interacts badly with **#2**: making non-integer and low samples-per-UI a
first-class case is exactly what pushes `tr_frac * spb` under the floor, so the two
features fight each other.

The floor itself is legitimate — a 0.1-sample rise time is not realizable. The
problems are that it is **silent**, and that the parameter has a large dead range
whose extent depends on `samples_per_ui` with no feedback to the caller.

**Suggested shape of a fix:** express rise time in absolute units via `Grid`
(`tr_ps=`), clamp explicitly rather than implicitly, and **report the realized rise
time** — in the return value or the `#5` provenance recipe — so a caller can see
that the request was not honoured. A `validate` assertion that realized rise time
tracks the request over the usable range, and that clamping is visible outside it,
would have caught this.

### B1. `main` suite red — missing import in the source-jitter test — ✅ FIXED (PR #17)

### B1. `main` suite red — missing import in the source-jitter test — ✅ FIXED (PR #17)
`test_source_jitter_edge_rms_and_independent_noise` raised
`NameError: name 'ws' is not defined` on `ws.Grid(fs=256e9, n=4096)`. It was never a
physics problem: every physical assertion in the test passed and
`python -m wfmsynth.validate` passed completely — only the final Grid-conversion
lines could not run.

Kept as a record for one reason: it reached `main` and sat there red, which is the
argument for **#23 (CI across platforms)**. A suite that is the declared trust
anchor has to be verified before merge, or it quietly stops being one. Verified
green again at v0.3.0 (18 tests + full validate).

---

## P1 (cont.) — gaps found while integrating against the library

### 24. Standalone ENOB quantisation in `instrument` — ✅ DONE (v0.7)
`instrument.quantize_adc(x, enob, full_scale=None)` — finite-ENOB lattice, no longer
buried in the deep_capture pipeline. validate asserts ~2^enob distinct codes with each
sample moving at most half an LSB.

ORIG:
`instrument` exposes `clip_adc`, `interleave_adc` and `shaped_noise_floor`, but
finite-ENOB quantisation exists only *inside* `pam4._digitize`, bundled with
resampling and thermal noise. So a caller who wants "quantise this waveform to N
effective bits" — arguably the single most characteristic thing an ADC does — has
to either reimplement it or accept the whole `deep_capture` pipeline.

```python
quantize_adc(x, enob=5.8, full_scale=None)   # full_scale=None -> from the signal
```

Pairs naturally with the existing `clip_adc`, and makes the quantisation grid
available to anyone comparing synthetic against measured data — where the presence
or absence of an ADC lattice is one of the most obvious differences.

**Done when:** validate asserts the distinct-value count collapses to ~2^enob
while the waveform moves by at most half an LSB.

### 25. A composed `digitize()` with the stages in the right order — ✅ DONE (v0.7)
`instrument.digitize(x, grid=, interleave=, clip_full_scale=, enob=, noise_floor=, rng=)`
-> (y, info), composing noise -> interleave -> clip -> quantise (the physically correct
order) once so a caller cannot get it wrong; returns the applied settings + clipped
fraction. validate asserts it matches the manual stage-by-stage path and that reordering
(quantise-before-noise) changes the result.

ORIG:
The instrument primitives are individually clean but the caller has to know the
physically correct order, and getting it wrong is silent. Interleave mismatch
happens at the sampling instant, clipping happens at the ADC input, quantisation
happens last, and **all** of it follows the channel and the additive impairment.
Reversing quantisation and clipping, or quantising before adding noise, produces
a plausible-looking waveform with the wrong noise floor.

```python
digitize(x, grid=None, interleave=None, clip_full_scale=None, enob=None,
         noise_floor=None, rng=None)   # -> (y, info)
```

Returning the applied settings alongside the samples also feeds #5 (provenance)
and #8 (measured ground truth).

**Done when:** validate asserts the composed path matches the manual stage-by-stage
application, and that reordering changes the result — i.e. that the order matters
and is therefore worth fixing in one place.

### 26. `interleave_adc` offset mismatch in absolute units — ✅ DONE (v0.7)
`interleave_adc(..., offset_v=)` expresses per-core offset in absolute volts (a property
of the converter, not the signal); `shaped_noise_floor(rms=)` was already absolute.
validate asserts the offset tone is invariant to input scaling in volts and proportional
as a fraction.

ORIG:
`offset_mm` is a standard deviation expressed as a fraction of the signal span,
so the artifact scales with the input. A real per-core offset error is a property
of the converter, not of the signal: it stays put when the signal shrinks, which
is exactly when it matters most. With `Grid` now available, an absolute-volts
option would be more physical:

```python
interleave_adc(x, ..., offset_v=1.5e-3, grid=g)   # alongside offset_mm
```

Same argument applies to `shaped_noise_floor(rms=...)` — an absolute noise floor
in volts is the specification an instrument datasheet actually gives.

**Done when:** validate asserts the offset-induced tone amplitude is invariant to
scaling the input when specified in volts, and proportional to it when specified
as a fraction.

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

### 27. Unify compose `Signal.digitize()` onto `instrument.digitize()`
The composer's `_op_digitize` and the new `instrument.digitize()` now model the same ADC
stages independently. `Signal.digitize()` should delegate to `instrument.digitize()` so
there is one canonical, correctly-ordered pipeline. The wrinkle: compose draws noise and
interleave from SEPARATE role streams (#6), while `instrument.digitize` takes a single
rng — so the delegation must thread per-role streams through, and it will change composed
waveform values (a deliberate minor-version break). Kept separate here to land #24-26
without touching the provenance/round-trip guarantees.

**Done when:** `Signal.digitize()` calls `instrument.digitize()`; provenance round-trip
and the #6 contrastive-pair assertions still hold.

### 28. Rx CTLE and DFE (companions to #11 Tx FFE) — ✅ DONE (v0.22)
`wfmsynth.rx`: `ctle(x, grid, fz_ghz, fp1_ghz, fp2_ghz, dc_gain)` — a high-frequency-peaking
analog EQ (zero below poles) that flattens a lossy channel, composable via `Signal.ctle()`;
`dfe(samples, taps, levels)` — decision-feedback EQ that cancels post-cursor ISI from sliced
past symbols. validate asserts CTLE peaks above DC and opens a lossy eye, and DFE cancels a
discrete post-cursor (symbol errors collapse). Completes the Tx-FFE/Rx-EQ equalization story.

Tx FFE landed in v0.10. Add the receive-side equalizers: **CTLE** (a continuous-time
linear peaking filter — a zero/pole boost that flattens the channel) and **DFE**
(decision-feedback — cancels post-cursor ISI using sliced past symbols; nonlinear, and
should model error propagation). CTLE is linear and composes cleanly before the slicer;
DFE needs the transmitted/decided symbol stream, so it pairs with `carrier_symbols` and
the #8 alignment.

**Done when:** validate asserts CTLE peaking opens a lossy-channel
eye, and DFE removes a discrete post-cursor echo it is tuned to.

### 29. Waveform-level clock-recovery fold (companion to #12) — ✅ DONE (v0.38)
`cdr.recover_and_fold(x, grid, n_blocks, levels)` folds a waveform onto the clock a CDR
recovers from it (tracking the local optimal sampling phase across n_blocks ~ loop bandwidth)
and returns the recorded eye height. validate asserts the folded eye is more open than the
fixed-grid eye for low-frequency jitter (tracked out) and ~equal for high-frequency jitter.
Complements the phase-domain jitter transfer (#12 recover_clock).

#12 delivers the CDR jitter transfer on a per-symbol phase sequence. Add a waveform-level
`cdr.recover_and_fold(x, grid, loop_bw, order)` that detects per-symbol timing error from
mid-level crossings, runs it through the clock transfer, and RESAMPLES x onto the recovered
symbol instants -- the literal 'record folded by the recovered clock' -- so a measured eye
can be reported alongside the recovery that produced it (ties to #8 ground_truth).

**Done when:** validate asserts the CDR-folded eye of a waveform with low-frequency jitter
is more open than the fixed-grid eye, and equal for high-frequency jitter.

### 30. Shot-noise term — ✅ DONE (v0.28, folded into #34 optical.shot_noise)

ORIG:
#17 covers heavy tails + 1/f. Add a Poisson shot-noise term (discrete arrivals -> skew, and
variance proportional to rate) as an optional component, for detectors/optical front ends.

**Done when:** validate asserts the shot component is skewed and its variance scales with rate.

---

## Realism v2 — broaden the composable primitive set

**North-star:** a composable set of primitives that can construct ANY realistic scenario a
scope might capture — fast optical AND electrical serial, down to slow buses, plus real-world
design effects (drift, supply coupling, environmental). The library stays pure synthesis:
analysis / root-cause tooling COMPOSES these primitives, it does not live here. Prioritized by
ubiquity-in-real-captures × signature size × discriminative value — run the #21 sim-to-real
harness against real captures to re-order against evidence.

### Tier 1 — near-universal in real links; a synthetic set without them is trivially separable

### 31. Spread-spectrum clocking (SSC) — ✅ DONE (v0.23)
`cdr.ssc_phase(n, fs, f_ssc, spread, profile)` produces the triangular clock-timing wander;
`cdr.apply_ssc(x, fs, ...)` warps a waveform onto the SSC-modulated clock; `Signal.ssc()`
composes it. validate asserts down-spread deviation stays in [-spread,0] and is periodic at
f_ssc, a wide-loop CDR tracks the wander out while a narrow one doesn't, and SSC spreads a
tone's spectrum (its EMI-reduction purpose).

Triangular ~30–33 kHz, ~0.5% down-spread FM of the symbol clock — mandatory/ubiquitous in
PCIe/USB/SATA/DisplayPort for EMI. A large, structured spectral+timing signature that directly
stresses the CDR (#12). Compose as a timing-modulation source (#35) feeding the carrier.
**Done when:** validate asserts the CDR tracks the SSC profile within its loop bandwidth while a
fixed clock shows the triangular wander, and the spectrum shows the spread.

### 32. Differential-pair primitives (intra-pair skew, common-mode, mode conversion) — ✅ DONE (v0.24)
`physics.differential_pair(x, grid, skew_ps, gain_imbalance, cm)` -> `(p, n)` with
`differential_mode`/`common_mode` helpers; `Signal.intra_pair_skew()` composes the
differential-mode closure in-chain. Intra-pair skew closes the differential eye and converts
energy to common-mode; gain imbalance is direct differential->common-mode conversion. validate
asserts ideal recovery with ~zero common-mode, skew-induced closure + common-mode (zero at
skew=0), and imbalance producing data-correlated common-mode.

Real links are differential; we model a single-ended trace. Intra-pair (P/N) skew, a common-mode
term, and differential↔common-mode conversion are first-order — a real differential probe sees all
three.
**Done when:** validate asserts skew closes the differential eye and generates common-mode energy
that vanishes at skew=0.

### 33. Power-supply / PDN coupling (AM + PSIJ) — ✅ DONE (v0.25)
`physics.supply_coupling(x, grid, f_ripple_hz, am_depth, psij_ps, supply)` + 
`Signal.supply_coupling()` couple a supply rail onto the signal as BOTH amplitude modulation
and power-supply-induced jitter from the SAME supply waveform (so the artifacts are
correlated and attributable to the rail); an optional `supply` array carries switching-
correlated activity. validate asserts AM and PSIJ each produce a sideband at f_ripple scaling
with the coupling, and both appear together from the shared rail.

A composable "supply aggressor": a supply rail (ripple tone + switching activity) couples onto the
signal as BOTH amplitude modulation and power-supply-induced jitter, correlated to switching. We
only have a PDN-transient *fault* today. A rich, structured, real artifact — and a strong target
for downstream root-cause tools (which compose it; it does not live here).
**Done when:** validate asserts a supply tone appears as correlated AM + PM sidebands scaling with
the coupling coefficient.

### 34. Optical-link primitives (extinction ratio, RIN, laser chirp, MPI, dispersion) — ✅ DONE (v0.28)
`wfmsynth.optical`: `to_optical(x, er_db)` (bipolar -> intensity with finite extinction ratio),
`rin_noise` (power-proportional relative intensity noise), `shot_noise` (Poisson, variance ∝
power — folds #30), `chromatic_dispersion` (pulse spreading), `mpi` (delayed ghost). Composes
via `Signal.optical(...)` and `Signal.dispersion(...)`. validate asserts ER sets the low/high
power ratio, RIN/shot scale with power, and dispersion spreads a pulse. (Laser chirp logged #44.)

Optical PAM4 (400G/800G) is a whole realism dimension: finite extinction ratio (nonzero low level),
relative intensity noise (RIN), laser chirp (amplitude-dependent phase), multipath interference
(MPI), and chromatic dispersion. Folds in the logged Poisson shot-noise term (#30).
**Done when:** validate asserts ER sets the low/high power ratio, RIN scales intensity-proportional
noise, and dispersion spreads a pulse.

### Composability enablers — what lets these (and anything else) compose into real scenarios

### 35. Timing-modulation source (arbitrary phase into any carrier) — ✅ DONE (v0.26)
`cdr.timing_source(n, grid, ssc, pj, wander, rj_ps, phase_noise)` sums named clock effects
into one per-sample timing-phase sequence; `cdr.apply_phase(x, phase)` warps a waveform by
it; `Signal.timing(...)` composes it into any carrier. The enabler for realistic COMBINED
jitter and for exercising the CDR. validate asserts a Pj source lands a phase tone at its
frequency, a composed SSC+Pj+Rj source carries each component, and apply_phase warps.

A composable phase-modulation source that sums SSC (#31) + oscillator phase noise (#37) + long-term
wander (#38) + Pj/Rj/DCD into one per-symbol timing-phase sequence, injectable into any carrier.
Today jitter is parametric Rj/Pj/DCD only; arbitrary phase injection is the enabler for realistic
combined jitter and for exercising the CDR (#12).
**Done when:** validate asserts a composed phase source (SSC+PN+Pj) reproduces each component in the
carrier's realized edge timing.

### 36. Multi-signal "scene" composition (shared aggressors / supply / clock) — ✅ DONE (v0.27)
`wfmsynth.scene.Scene(grid)` composes several named lanes with SHARED sources: `shared_supply`
(one rail -> correlated artifact across lanes), `couple(into, frm, coupling)` (arbitrary
lane-to-lane crosstalk), `differential(name, skew_ps, ...)` (split a lane into P/N sharing
timing). The enabler for realistic multi-lane scenarios the single-Signal chain can't express.
validate asserts a shared rail gives an identical artifact across independent lanes, coupling
injects the aggressor, and a diff pair yields P/N with skew-induced common-mode.

Compose MULTIPLE Signals that share correlated sources — a supply rail coupling into several lanes,
a differential pair as coupled P/N, user-defined aggressors coupling into user-defined victims.
Today crosstalk aggressors are generated internally; a scene primitive lets arbitrary lanes/rails
couple, which is what "construct a realistic multi-lane scenario" needs.
**Done when:** validate asserts a shared supply rail induces correlated artifacts across two lanes,
and a differential pair's P/N share timing with controlled skew.

### Tier 2 — high value

### 37. Oscillator phase noise (colored jitter spectrum) — ✅ DONE (v0.29)
`cdr.phase_noise(n, grid, rms_ps, slope)` generates colored clock phase noise whose PSD falls
as 1/f**slope (slope~2 = free-running oscillator close-in); it's the phase-noise source for
`timing_source`. validate asserts the PSD follows the requested slope and the RMS is honored.

Real clocks have a Leeson-shaped phase-noise profile; jitter measurement and CDR tracking depend on
the spectrum, not a single Pj tone. Provides a phase-noise source for #35.
**Done when:** validate asserts the generated jitter PSD follows the requested slope.

### 38. Long-term wander / drift (real-world designs) — ✅ DONE (v0.30)
`impairments.drift(x, grid, kind, amount, shape)` applies slow sub-record nonstationarity
(thermal/VGA/DC drift, laser aging) as a slowly-varying gain/amplitude/DC across the record;
`Signal.drift()` composes it. validate asserts a measured attribute moves monotonically across
the record while any short window stays ~stationary.

Sub-Hz nonstationarity over the record: thermal drift, VGA/AGC settling, DC-offset drift, laser
aging. Real long captures are nonstationary; today only intermittent gates (#10) break stationarity.
**Done when:** validate asserts a slow drift moves a measured attribute monotonically across the
record while short-window statistics stay ~unchanged.

### 39. Line coding & scrambling (8b/10b, 64b/66b, 128b/130b) — ✅ DONE (v0.31)
`wfmsynth.coding`: `dc_balanced(bits, block)` (8b/10b-style running-disparity block inversion
-> bounded disparity / DC balance), `scramble_64b66b(bits)` (the real x^58+x^39+1 scrambler +
sync header), `running_disparity`/`max_run` measures. validate asserts dc_balanced keeps
disparity bounded while raw PRBS random-walks, stays balanced, and the scrambler whitens.
(Full 8b/10b tables for tight run-length bounding logged #45.)

PRBS is a stand-in; real DC balance and run length depend on the code, which drives baseline-wander
interaction with AC-coupling.
**Done when:** validate asserts an 8b/10b stream is DC-balanced with bounded run length vs raw PRBS.

### 40. Acquisition-chain primitives (scope/probe transfer, loading, trigger/timebase jitter) — ✅ DONE (v0.32)
`instrument.scope_bandwidth(x, grid, bw_hz, kind)` (Bessel/Gaussian band-limited front end),
`probe_loading(x, grid, c_load_f, r_source)` (capacitive DUT loading -> RC low-pass), and
`timebase_jitter(x, grid, rms_ps)` (sample-clock jitter smears the eye horizontally); compose
via `Signal.scope()`/`Signal.timebase()`. validate asserts scope BW rolls off HF, probe
loading attenuates HF, and timebase jitter closes the eye.

Beyond the ADC (#3/#24/#25): the scope's own bandwidth-limiting response (Bessel/Gaussian), probe
capacitive loading of the DUT, and trigger/timebase jitter — present in every real capture.
**Done when:** validate asserts the scope bandwidth rolls off HF and timebase jitter smears the eye
horizontally.

### Tier 3 — breadth / polish

### 41. Low-speed bus signaling primitives — ✅ DONE (v0.36)
`wfmsynth.bus`: `open_drain(drivers)` (wired-AND -- the line is high unless any driver pulls
low; bus contention / CAN arbitration fall out of it), `uart_frame(bytes)` / `uart_decode`
(idle-high start/stop framing). validate asserts the wired-AND truth table and a UART frame
round-trip. (SPI/CAN-differential compose from these + the differential/channel primitives.)

The range a scope measures extends below serial: I2C (open-drain / wired-AND, ACK), SPI, UART
(start/stop framing, idle), CAN / RS-485 (differential, arbitration, contention), variable bit rate.
Primitives for single-ended open-drain, framed slow signaling, and bus contention.
**Done when:** validate asserts an open-drain wired-AND pulls low when any driver is active and
floats to the pull-up otherwise.

### 42. Standardized Tx de-emphasis presets — ✅ DONE (v0.33)
`physics.de_emphasis_taps(db)` returns the 2-tap FFE weights for a dB de-emphasis preset;
`Signal.de_emphasis(db=)` composes it. validate asserts a 3.5 dB preset yields ~3.5 dB
transition-to-steady ratio.

Extend Tx FFE (#11) with the standardized de-emphasis presets (in dB) real transmitters expose.
**Done when:** validate asserts a preset yields the specified de-emphasis ratio.

### 43. Electrical-idle / LFPS / training sequences — ✅ DONE (v0.34)
`impairments.electrical_idle(x, intervals, grid, lfps_hz, amp)` squelches the signal to
electrical idle over the given intervals, or fills them with an LFPS (low-frequency periodic
signaling) burst. validate asserts an idle interval carries ~no data energy and LFPS shows a
low-frequency periodic burst near f_lfps. (Training-sequence patterns can compose via #45
carrier symbols.)

Protocol non-data intervals: electrical idle, low-frequency periodic signaling (PCIe/USB LFPS),
squelch, training patterns. Real full captures contain these between data bursts.
**Done when:** validate asserts an idle interval carries no data energy and LFPS shows its periodic
low-frequency burst.

### 44. Laser chirp (companion to #34 optical) — ✅ DONE (v0.35)
`optical.laser_chirp(power, alpha, grid)` returns the transient chirp -- the instantaneous
optical frequency excursion during intensity transitions, (alpha/4pi)*d(ln P)/dt -- which
occurs at edges and scales with alpha; with dispersion it causes transition-dependent
distortion. validate asserts chirp concentrates at transitions and scales with alpha.

Amplitude-dependent phase (frequency chirp during intensity transitions): alpha-parameter
transient chirp that, with dispersion, causes transition-dependent pulse distortion.

**Done when:** validate asserts chirp adds a transition-edge frequency excursion scaling with
the alpha parameter.

### 45. Carrier arbitrary-symbol input (companion to #39) — ✅ DONE (v0.37)
`physics.from_symbols(symbols, n, ...)` + `Signal.symbols(...)` build a carrier from an
arbitrary per-UI symbol sequence (same edge-shaping/jitter as nrz/pam4), so a line-coded /
scrambled / custom stream (wfmsynth.coding) drives synthesis end-to-end. validate asserts a
carrier built from a coded bit stream reproduces those bits. Full 8b/10b lookup tables (tight
RLL<=5) split to #46.
#39 bounds disparity (DC balance) but not run length as tightly as real 8b/10b; add the 5b/6b
+ 3b/4b lookup tables (RLL<=5). Also let the carrier accept an arbitrary symbol/bit sequence
(`symbols=`) so a coded/scrambled stream can drive synthesis end-to-end (today carrier takes
seed/pattern only).

**Done when:** validate asserts an 8b/10b stream has max run <=5, and a
carrier built from coded bits reproduces them.

### 46. Full 8b/10b encode tables (companion to #39/#45)
#39's dc_balanced bounds disparity but not run length to 5; add the real 5b/6b + 3b/4b 8b/10b
lookup tables with running-disparity selection (and K control characters). Feed the coded
symbols to the carrier via #45's from_symbols.

**Done when:** validate asserts an 8b/10b
stream has max run <=5 and is DC-balanced, and decodes back to the input bytes.
