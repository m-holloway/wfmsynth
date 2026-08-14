# wfmsynth — Roadmap / Wishlist

v1 is a compact, **validated** physics-synthesis engine: causal channels, reflections,
crosstalk, decomposed jitter, AC-coupling, NRZ/PAM4/RF carriers, a compositional grammar,
and realistic segmented deep-memory PAM4 captures. This is where it could go. Nothing here
is required to use v1; it's the map for pushing the engine toward a broad, high-fidelity,
ground-truth-grade signal generator.

---

## ⭐ Flagship: provenance-first composable procedural synthesis

**The idea.** Build a signal as an explicit, composable *graph of components* — each
recording its type and the exact knob values used — so every waveform comes with a
complete, serializable **recipe**. Randomize freely for coverage, but *record every sampled
value*, so each generated waveform is (a) exact ground truth for training, (b) fully
reproducible, and (c) auditable ("what produced this artifact?").

Why it matters: for training data, you don't just want a waveform and a coarse label — you
want to know *precisely* which channel length, which reflection coefficient at which delay,
which Rj/Pj split, which EQ setting, etc. produced it. That turns synthetic data into
labeled-to-arbitrary-depth ground truth, and makes datasets diffable and versionable.

**Sketched API (v0.2):**
```python
from wfmsynth.compose import Signal, Carrier, Channel, Impair

sig = (Signal(seed=42)
       .carrier(Carrier.PAM4, n_ui=64, baud=112e9)
       .channel(Channel.LOSSY,      length_in=8.0, tand=0.02, causal=True)
       .channel(Channel.REFLECTION, td_frac=0.09,  gamma=0.4)
       .impair(Impair.JITTER,       rj=0.3, pj=(0.05, 4e6))
       .digitize(gsa=256e9, enob=5.5))

x      = sig.waveform()   # the samples
recipe = sig.recipe()     # serializable dict: every component + every knob + seeds + versions
```
A `dataset(spec, n)` builder samples knobs from declared distributions and returns
`(waveforms, recipes[])` — each example carrying its own exact recipe. Recipes round-trip:
`Signal.from_recipe(recipe).waveform()` reproduces the sample bit-for-bit.

**Forward-compatibility:** v1's primitives (`physics`, `impairments`, `pam4`) are exactly
the building blocks such a composer orchestrates — adding the provenance layer on top is
*additive*, not a breaking change to the primitive API. (v1 already returns partial
provenance: `grammar.generate` reports the carrier kind and impairment names per waveform;
the full-knob recipe is the v0.2 step.)

---

## Aberrations / impairments (breadth)
- **Jitter, fully decomposed**: DDJ (data-dependent, ISI-driven), DCD, sub-rate/bounded
  uncorrelated jitter, and a proper dual-Dirac / random+deterministic split with target BER.
- **Spread-spectrum clocking (SSC)** (down/center-spread, profile + frequency).
- **PAM4 level nonlinearity / RLM**, level-dependent noise, thermal + shot noise models.
- **Duty-cycle distortion, rise/fall asymmetry, slew-rate limiting.**
- **PDN / power-integrity**: ripple, droop, load-step transients, supply-noise coupling
  onto the signal (AM/PM), ground bounce.
- **EMI / periodic interference**, spurs, intermodulation.
- **Amplitude/timing drift** over the capture (thermal), and burst/intermittent faults.

## Channel realism
- **Measured / behavioral S-parameters** (Touchstone `.sNp`) as the channel, incl. return
  loss and mixed-mode (SDD/SDC/SCD) for differential pairs; convolution with the true impulse.
- **Causal Djordjević–Sarkar** dielectric model (vs the current linked min-phase form).
- **Differential signaling**: true P/N pair, common-mode, skew, mode conversion.
- **Vias/connectors/stubs** as parameterized discontinuities at specified locations.

## Signal types / standards
- NRZ variants, clocks (with wander/SSC), sawtooth/PWM/switching-regulator waveforms.
- Standard-flavored generators (PCIe, DDR, USB, UCIe, Ethernet/OIF-CEI PAM4) — coding,
  scrambling, framing, compliance patterns (PRBS7/13/31Q), preambles.

## Equalization
- **Tx FFE**, **Rx CTLE**, **DFE** (and reference EQ) so captures can be modeled pre- or
  post-EQ at a chosen probe point — this changes waveform character fundamentally.

## Crosstalk / multi-channel
- Multiple aggressors from a **coupling matrix** (NEXT + FEXT), realistic aggressor activity,
  victim/aggressor alignment, and multi-lane captures with cross-lane structure.

## Instrument / scope modeling
- Front-end bandwidth (Bessel-Thomson), ENOB/noise floor, interleave spurs, timebase jitter,
  trigger jitter, probe loading — model *what the scope records*, per instrument class.

## Ground-truth co-generation
- Emit the **true** derived measurements alongside each waveform: ideal eye (height/width per
  eye), RLM, decomposed jitter components, TIE series, per-defect location/extent masks,
  channel IL(f) — so a training set has exact regression/segmentation targets, not just class labels.

## Reproducibility & tooling
- Deterministic seeding end-to-end; **dataset manifests** (spec + recipes + engine version)
  for versionable, diffable datasets; export to `.npz`/`.parquet`/`.wfm`/Touchstone.
- Optional **eye-diagram / density rendering** and quick-look plots (kept as an optional extra
  so the core stays numpy/scipy-only).

## Performance & scale
- Vectorized/batched generation; optional GPU (CuPy/torch) backend; streaming generation for
  very large "deep memory" captures (billions of points / hundreds of thousands of segments).

## Fidelity & validation
- Expand the assertion suite as primitives are added; where real captures exist, add
  distribution-level fidelity checks (does synthetic match measured eye/jitter statistics?).

---

### Companion extractions (separate repos, not part of this one)
To keep this package *pure synthesis*, these belong elsewhere:
- **Signal-integrity measurements** (eye/RLM/SNR/TIE analysis of a waveform) — an analysis lib.
- **Eye-diagram / density-eye rendering** — a visualization utility (adds a plotting dep).
