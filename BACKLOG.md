# wfmsynth — prioritized backlog

`ROADMAP.md` is the **breadth** map: everything this engine could eventually model.
This file is the **depth** list: the next actionable pieces, in priority order, each
with why it matters and what "done" looks like.

**Definition of done, for anything here:** a hard assertion in `wfmsynth.validate`
that the physical property actually holds. That gate is the reason this library is
trustworthy; an item without one isn't finished.

**Priorities** are about *fidelity to real captures* first, then ground-truth
quality, then breadth. An effect that shows up immediately when you compare against
a real acquisition outranks an effect that is merely absent.

---

## P0 — blocks realistic use

### 1. Absolute units and rate binding
Everything currently lives on a normalized, unitless grid: amplitudes are ±1, time
is [0,1), delays are fractions of the record, and frequencies are cycles-per-record.
Record length is now parameterizable, but there is still no way to state a symbol
rate, a sample rate, a capture duration or a full-scale voltage — so a caller cannot
express a real link, and every derived quantity has to be reinterpreted by hand.

`multi_reflection(td_samples=...)` is a first step in this direction: an absolute
delay, because a discontinuity sits a fixed number of picoseconds away regardless of
how long you acquire for. The same treatment is needed throughout.

**Proposal.** A small immutable descriptor threaded through the primitives:

```python
Grid(fs=..., baud=..., n=..., v_full=...)
# derived: dt, ui_seconds, samples_per_ui (often non-integer!), f_nyquist
```

Then parameters can be given in the units engineers actually use — delays in ps,
jitter in fs, periodic-jitter frequency in Hz, bandwidths in GHz — with the
fraction-of-record forms retained as the default so nothing breaks.

**Done when:** a primitive accepts a delay in ps and an insertion loss in dB at a
stated frequency, and validate asserts the realized values match the requested ones
after round-tripping through the grid.

### 2. Instrument model: time-interleaved ADC artifacts
`pam4._digitize` models thermal noise + finite ENOB. A real high-speed scope front
end is a **time-interleaved** ADC, and interleaving is what makes a real capture look
like a real capture:

- per-core **gain**, **offset** and **timing skew** mismatch → spurs at `f_s/M` and
  image tones mirrored about `f_s/(2M)`
- a **frequency-shaped** noise floor, not a flat one
- **timebase / trigger jitter** on the sampling clock itself

This is my best guess at the single most likely giveaway if you train on synthetic
and test on measured: perfectly white Gaussian noise plus perfect uniform sampling
is not what any instrument records. Cheap to add, high fidelity payoff.

**Done when:** validate asserts that with `M` interleaved cores and injected
mismatch, spurs appear at the predicted frequencies and vanish when mismatch is zero.

### 3. Apply jitter at the source, not by warping the output
`inject_jitter` warps the time axis of an already-generated waveform and resamples.
Two problems:

- if it runs after additive noise, it **jitters the noise too**, which is not physical
- warping *after* the channel **smears the ISI the channel just created**, rather
  than displacing the transmitted edges that then propagate through it

Physically, jitter belongs to the transmitter's edge times: perturb the symbol edge
positions, *then* shape and propagate. That also makes DDJ emerge naturally from the
channel instead of needing to be injected.

**Done when:** the carrier generators accept jitter parameters directly, and validate
asserts that (a) recovered edge-time RMS matches the injected value, and (b) additive
noise added after the channel is uncorrelated with the injected jitter.

---

## P1 — needed for ground-truth-grade datasets

### 4. Provenance-first composable synthesis
The `ROADMAP.md` flagship. Nothing to add to the design there; it is the right shape.
Worth noting it becomes *more* valuable once (1) lands, because a recipe in real units
is portable between tools, whereas a recipe in fractions-of-record is not.

### 5. Confounder-controlled sweeps, and realized-vs-requested reporting
This is the one I would most want if the output is training data, and it is not in
the roadmap.

When you sweep an attribute to build a labeled set, you have to hold the obvious
shortcut constant, or a model learns the shortcut and scores beautifully for the
wrong reason. Concretely: adding a reflection closes the eye, so a naive
reflection sweep is also an eye-height sweep, and anything trained on it can read
eye height instead of ISI structure.

Two capabilities:

- **Hold-constant sweeps.** Declare a target metric to pin and a free parameter to
  solve for: *"sweep the reflection coefficient across its range, bisecting insertion
  loss to hold eye height fixed."* Note the physical constraint this exposes — you cannot
  independently vary reflection, loss **and** eye height, only two of the three — so
  the API should make the caller choose which two are free rather than silently
  picking.
- **Realized, not requested, labels.** With deep composition the knob → outcome map
  is many-to-many, and nominally orthogonal knobs produce correlated realized
  attributes. Emit the *measured* attribute values next to the requested ones, plus
  the realized correlation matrix across a generated set.

**Done when:** a sweep builder returns realized values, and validate asserts that a
hold-constant sweep keeps its pinned metric within a stated tolerance while the swept
attribute moves monotonically.

### 6. Ground truth as measured, not as ideal
In `ROADMAP.md` under "Ground-truth co-generation". One sharpening: emit values
**measured from the generated output**, not computed from the knobs. They differ, and
the difference is exactly where silent label noise comes from.

### 7. Tx FFE / Rx CTLE / DFE
In `ROADMAP.md`. Priority note: **Tx FFE is the highest-value one for fidelity.** Real
PAM4 transmitters almost always run 3-tap or longer FFE, which puts a deliberate
*pre-cursor* in the pulse response. A synthetic waveform with no pre-emphasis has a
qualitatively different pulse-response shape from anything on a real link, and pulse
response is the first thing an SI engineer looks at.

---

## P2 — breadth and fidelity

### 8. Measured S-parameter channels (Touchstone)
In `ROADMAP.md`. Biggest realism-per-effort item after the instrument model: the
analytic √f + f loss model cannot produce resonances, fibre-weave periodicity,
connector structure, or interacting multiple bounces. Real `.s4p` files give all of
it. Keep the analytic model as the dependency-free default.

### 9. Level nonlinearity beyond RLM compression
`pam4.deep_capture` has `rlm_compression`, but the base carrier is exactly linear —
RLM = 1.000 to numerical precision. Real transmitters sit around 0.95–0.99 *always*,
not only when faulted. Add a nominal level-nonlinearity, level-dependent noise, and
rise/fall asymmetry so "nominal" is not suspiciously perfect.

### 10. Multiple aggressors from a coupling matrix
In `ROADMAP.md`. Add: aggressors should be **asynchronous** to the victim by default.
A synchronous aggressor produces interference locked to the victim's clock, which a
receiver's CDR partly tracks out and which looks like ISI rather than crosstalk.

### 11. Streaming / chunked generation
A multi-megapoint record currently needs several float64 arrays live at once, and the
frequency-domain channel needs a full-length FFT. Deep-memory captures are the headline
use case, so chunked or memory-mapped generation is worth having before anyone reaches
for a hundred-megapoint record.

### 12. Sim-to-real separability harness
A tool, not a physics primitive, but it is what tells you whether any of the above
matters: given a set of synthetic waveforms and a set of real captures, compute a
common feature vector and ask whether a trivial classifier can separate them — and
if so, **which feature does it use?** That names the missing physics instead of
guessing at it. Ranked here as P2 only because it needs real captures to be useful;
in value terms it outranks most of this list.

### 13. Note on critical sampling *(documentation, not code)*
Worth a paragraph in the README, because it is non-obvious and it silently degrades
downstream analysis: if the modelled front-end bandwidth sits at or near the Nyquist
frequency of the sample rate, the capture is **critically sampled**, and any analysis
that needs sub-sample interpolation becomes lossy.

The case that bites: pattern-averaging analyses have to realign repetitions to a
common grid, and whenever samples-per-UI is not an integer the pattern period is
**fractional** — an odd-length pattern almost guarantees it. Sub-sample realignment is
only exact well below Nyquist, so near critical sampling the deterministic part fails
to cancel and the residual ends up dominated by leftover signal rather than by the
impairment you were trying to isolate. If the library emits ground truth it should
emit the exact fractional pattern period, so downstream tools do not assume an integer.

### 14. CI across platforms
The validation suite is the trust anchor, so it should be proven to run everywhere.
It previously died partway through on a stock Windows console (cp1252 could not encode
a maths glyph in a detail string, which looks like a hang and hides every check after
it). A CI matrix over Linux/macOS/Windows would have caught that.

---

## Contributing an item

1. Add it here with a priority, a rationale, and a "done when".
2. Implement behind existing defaults where possible — this library is used as a
   dependency, and bit-identical legacy output is worth protecting. When a change
   could alter output, add an opt-in flag and say so in the docstring.
3. Add the validation assertion, and a test that the legacy path is unchanged.
