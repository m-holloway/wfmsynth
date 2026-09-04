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

grid = ws.Grid(fs=100e9, baud=25e9, n=16_384)

signal = (
    ws.Signal(seed=7, grid=grid)
    .carrier("nrz", n_ui=4096, causal=True)             # transmitted data
    .lossy(loss_db=8.0, loss_at_ghz=12.5, causal=True) # PCB/cable bandwidth
    .reflect(td_ps=80.0, gamma_s=0.15)                 # connector/discontinuity echo
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
| **ENOB** | Effective ADC resolution in bits |
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
| Sources and effects | `physics`, `impairments`, `events`, `grammar` | Digital/RF sources, channels, reflections, jitter, named faults, localized needles, broad shape generation |
| Acquisition | `acquire`, `instrument`, `pam4` | Two-rate captures, scope/probe/ADC effects, segmented PAM4 datasets |
| Links and systems | `rx`, `cdr`, `sparam`, `scene`, `optical`, `coding`, `bus` | Equalization, clock recovery, measured channels, multi-lane, optical, coding, UART/open-drain |
| Dataset quality | `measure`, `sweep`, `simreal` | Measured labels, confounder control, synthetic-vs-real separability |
| Scale and trust | `stream`, `validate` | Bounded-memory channel processing and physical-property assertions |

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

## Instrument / ADC model
The ADC stages compose in one place, in the physically correct order — getting it wrong
is silent (quantising before the noise gives the wrong noise floor).

```python
from wfmsynth import digitize_adc, quantize_adc
y, info = digitize_adc(x, noise_floor={"rms":1e-3,"shape":"pink"},
                       interleave={"m_cores":4,"offset_v":1e-3},   # offset in absolute volts
                       clip_full_scale=0.7, enob=6)                # -> noise->interleave->clip->quantise
q = quantize_adc(x, enob=5.8)     # standalone finite-ENOB lattice
```

## Impairment mixing at constant power
Separate impairment *magnitude* from *character* — "same SNR, different noise character":

```python
from wfmsynth import mix_at_constant_power
y = mix_at_constant_power([white, pink], weights=[1-a, a], total_rms=0.05)
# total RMS is exactly 0.05 for any a; only the spectral character changes with a
```

## Intermittent impairments (with defect masks)
Intermittency is where stationary analysis fails — apply a defect on a duty cycle and
emit the per-sample mask as ground truth:

```python
from wfmsynth import apply_gated
y, mask = apply_gated(x, glitch_fn, intervals=[(2000,180),(6000,140)])
# y is bit-identical to x outside the gate; mask marks where the defect is active
```

## Localized events (needles in a long record)
Systemic stages (loss, reflections, ADC) apply to every sample. A runt bit, an
edge-excited ring, a PDN sag on a long run, or an asynchronous glitch has **finite
time support**. Placement (when) is independent of mechanism (what):

```python
sig = (ws.Signal(seed=1, grid=g).carrier("nrz", n_ui=n_ui, causal=True)
       .events("runt", on="symbols", count=3, severity=0.6)      # data-locked
       .events("ring", on="edges", which="rising", count=2,      # transition-locked
               f0_hz=220e6, tau_s=12e-9)
       .events("droop", on="pattern", min_run=8, severity=0.5)   # run-locked
       .events("glitch", on="poisson", rate_hz=1e4, severity=0.4))  # not bus-locked
x, events = sig.realize()   # samples + [{sample, ui, t_s, kind, severity}, ...]
```

`on` is the targeting policy: `symbols`, `edges`, `pattern`, `aggressor` (another
lane's edges), `intervals`, `poisson`, `times`. `kind` is the mechanism: `runt`,
`glitch`, `ring`, `overshoot`, `undershoot`, `nonmonotonic`, `droop`, `slow_edge`.

Eye-mask failures and height at a sampling instant are **measured labels**, not
mechanisms — many causes produce them:

```python
windows = ws.nominal_ui_windows(len(x), g, half_ui=1.0)   # ±1 UI on the known baud
rows = ws.label_windows(windows, events, x=x, eye_low=0.12)
# each row: requested event kinds + measured height / slope reversals / eye_violation
```

This module does not recover a clock or cut segments. Pass recovered instants to
`windows_from_centers` and the same join still works. Systemic effects that were
never placed as events (a resonant stub that rings only on some patterns) still
appear in `measured` when they trip a flag. See [`examples/events.py`](examples/events.py).

## Tx FFE (pre-emphasis)
Real transmitters run multi-tap FFE — a deliberate pre-cursor de-emphasizing ISI:

```python
sig = (ws.Signal(seed=1, grid=g).carrier("pam4", n_ui=n_ui, pattern="prbs13q", causal=True)
       .tx_ffe(taps=[-0.15, 1.0, -0.25], pre=1)      # place before the channel
       .lossy(loss_db=8.0, loss_at_ghz=25.0, causal=True))
```

## Full RX equalization (CTLE / RX FFE / DFE)
The receiver side is composable end-to-end — equalize before slicing, just like real hardware:

```python
sig = (ws.Signal(seed=1, grid=g).carrier("pam4", n_ui=n_ui, pattern="prbs13q", causal=True)
       .tx_ffe([1.0, -0.2]).lossy(loss_db=10, loss_at_ghz=25, causal=True)
       .ctle(fz_ghz=2, fp1_ghz=5, fp2_ghz=10)   # analog HF peaking
       .rx_ffe([1.0, -0.3, 0.1], spacing_ui=0.5) # fractionally-spaced FIR
       .dfe([0.25, 0.1])                          # cancel post-cursors (taps = channel post-cursors)
       .digitize(snr_db=40))
```

## Optical E/O/E link (electrical → optical → electrical)
A true optical link on a **complex field** — so laser chirp and fibre dispersion interact correctly,
and square-law detection closes the loop back to electrical:

```python
det = (ws.Signal(seed=1, grid=g).carrier("pam4", n_ui=n_ui, pattern="prbs13q", causal=True)
       .eo(kind="dml", alpha=3.0, er_db=8)        # E→O: laser + chirp → complex optical field
       .fiber(length_km=4, D_ps_nm_km=17)         # physical β2·L dispersion + attenuation
       .edfa(gain_db=12, nf_db=5)                 # optical amp + ASE (optional)
       .photodetect(responsivity=1.0)             # O→E: i = R·|E|²  (square-law, closes the loop)
       .tia(gain=1.0, bw_hz=50e9)                 # transimpedance front end
       .digitize(snr_db=40))
```
`eo(kind="mzm", ...)` gives a Mach-Zehnder cos² modulator; `optical`/`dispersion` remain for the
simpler intensity-only model. Because the field carries phase, DML chirp through fibre dispersion
distorts the detected eye — the interaction an intensity-only model can't show.

## Causality hazard (build causal chains end-to-end)
Each stage can be causal in isolation while the *pipeline* is not: the default zero-phase
edge shaping reintroduces pre-cursor **after** a causal channel. For an end-to-end causal
chain, set `causal=True` on the carrier (and any lossy stage):

```python
ws.Signal(seed=1, grid=g).carrier("pam4", n_ui=n_ui, pattern="prbs13q", causal=True) \
  .lossy(loss_db=8.0, loss_at_ghz=25.0, causal=True)   # causal shaping AND causal channel
```

## Clock recovery (what the scope records)
A scope folds the record by a recovered clock; the CDR tracks out jitter below its loop
bandwidth. The recorded eye depends on the recovery:

```python
clock, residual = ws.recover_clock(jitter_phase, baud=50e9, loop_bw=1e6, order=2)
# residual is the timing jitter the eye actually shows (low-freq wander tracked out)
ws.tracked_out_fraction(jitter_phase, baud=50e9, loop_bw=1e6)   # 0..1
```

## Measured S-parameter channels (Touchstone)
Drive synthesis through a real `.sNp` channel — resonances the analytic model can't make:

```python
f, S = ws.read_touchstone("thru.s2p")                     # (nf, n, n) complex
sig = ws.Signal(seed=1, grid=g).carrier("pam4", n_ui=n_ui, pattern="prbs13q", causal=True) \
        .sparam(path="thru.s2p")                          # apply S21 as the channel
```

## Resonant reflections
Real discontinuities resonate — frequency-dependent Γ, not a flat mirror:

```python
ws.Signal(seed=1, grid=g).carrier("pam4", n_ui=n_ui, pattern="prbs13q", causal=True) \
  .resonant_reflect(td_ps=40.0, f0_ghz=30.0, q=10.0, gamma0=0.4)   # |Γ| peaks at f0
```

## Nominal imperfections (no class is perfect)
Real transmitters are imperfect always — a too-perfect nominal class is a giveaway:

```python
ws.Signal(seed=1, grid=g).carrier("pam4", n_ui=n_ui, pattern="prbs13q", causal=True) \
  .nonlinearity(compression=0.04, level_noise=0.005, rise_fall_ratio=1.3)
```

## Multiple aggressors (asynchronous by default)
Crosstalk from a coupling vector; aggressors run at offset bauds so they are not clock-
locked to the victim (a synchronous default makes crosstalk artificially easy to detect):

```python
ws.Signal(seed=1, grid=g).carrier("pam4", n_ui=n_ui, pattern="prbs13q", causal=True) \
  .crosstalk_matrix(couplings=[0.12, 0.08, 0.05])   # async by default; synchronous=True to lock
```

## Noise beyond white Gaussian
Heavy tails and 1/f structure — real noise is not one flat Gaussian:

```python
n = ws.realistic_noise(1<<16, rms=1e-3, df=5, pink_frac=0.3)   # heavy tails + some 1/f
```

## Streaming (deep-memory records)
Apply a channel to a multi-megapoint record in bounded memory (overlap-save):

```python
h = ws.channel_fir(lambda a: ws.physics.lossy_channel(a, length_in=8, causal=True), n_taps=512)
for block in ws.stream_blocks(x, h, chunk=1<<16):   # never a full-length FFT/output
    write(block)
```

## Sim-to-real separability harness
Ask whether a trivial classifier can tell synthetic from real — and which feature it uses
(that names the missing physics):

```python
rep = ws.separability(synthetic_set, real_set, grid)
rep["best_feature"], rep["best_auc"]   # e.g. ("hf_fraction", 0.98) -> HF content is wrong
rep["auc"]                             # AUC per feature = the full diagnosis
```

## Critical sampling (a non-obvious hazard)
If the modelled front-end bandwidth sits at or near the **Nyquist frequency** of the sample
rate, the capture is *critically sampled* and any analysis needing sub-sample interpolation
becomes lossy. The case that bites: pattern-averaging must realign repetitions onto a common
grid, and with non-integer samples-per-UI the period is fractional. Sub-sample realignment is
exact only well below Nyquist, so near critical sampling the deterministic part fails to
cancel and the residual is dominated by leftover **signal** rather than by the impairment you
were trying to isolate — which reads as a mysteriously large noise floor rather than as an
aliasing problem. Keep the front-end bandwidth comfortably below Nyquist (oversample), or
treat near-Nyquist residuals with suspicion.

## Receiver equalization (CTLE + DFE)
Real receivers equalize before slicing — CTLE (analog peaking) and DFE (decision feedback):

```python
sig = ws.Signal(seed=1, grid=g).carrier("pam4", n_ui=nui, pattern="prbs13q", causal=True) \
        .lossy(loss_db=9, loss_at_ghz=25, causal=True).ctle(fz_ghz=6, fp1_ghz=22, fp2_ghz=45)
eq, decisions = ws.dfe(per_symbol_samples, taps=[0.35])   # cancel a post-cursor
```

## Spread-spectrum clocking (SSC)
Triangular clock-frequency modulation (ubiquitous for EMI) — spreads the spectrum and adds
low-frequency wander a CDR must track:

```python
ws.Signal(seed=1, grid=g).carrier("pam4", n_ui=nui, pattern="prbs13q", causal=True) \
  .ssc(f_ssc=32e3, spread=0.005, profile="down")
```

## Differential-pair non-idealities
Split to P/N with intra-pair skew, common-mode and mode conversion (real links are differential):

```python
p, n = ws.differential_pair(x, grid=g, skew_ps=6.0, gain_imbalance=0.05)
diff, cm = ws.differential_mode(p, n), ws.common_mode(p, n)   # skew closes diff eye, makes cm
```

## Power-supply / PDN coupling
A supply rail coupling onto the signal as correlated amplitude modulation + supply-induced
jitter (a rich root-cause target that downstream tools compose):

```python
ws.Signal(seed=1, grid=g).carrier("pam4", n_ui=nui, pattern="prbs13q", causal=True) \
  .supply_coupling(f_ripple_hz=1e6, am_depth=0.03, psij_ps=2.0)   # AM + PSIJ from one rail
```

## Timing-modulation source (composable jitter)
Compose any mix of clock effects into one phase and inject it into a carrier:

```python
ws.Signal(seed=1, grid=g).carrier("pam4", n_ui=nui, pattern="prbs13q", causal=True) \
  .timing(ssc=dict(f_ssc=32e3, spread=0.005), pj=dict(amp_ps=2, f_hz=5e9), rj_ps=0.3)
```

## Multi-signal scenes
Compose several correlated lanes — a shared supply rail, lane-to-lane crosstalk, diff pairs:

```python
sc = (ws.Scene(g).add("lane0", w0).add("lane1", w1)
      .shared_supply(f_ripple_hz=1e6, am_depth=0.03)      # one rail, correlated across lanes
      .couple(into="lane1", frm="lane0", coupling=0.08))  # arbitrary lane-to-lane crosstalk
```

## Optical-link primitives
Optical intensity with extinction ratio, RIN, shot noise, and chromatic dispersion:

```python
ws.Signal(seed=1, grid=g).carrier("pam4", n_ui=nui, pattern="prbs13q", causal=True) \
  .optical(er_db=6.0, rin_db_per_hz=-135, shot=True).dispersion(strength=10.0)
```

## Acquisition chain (scope / probe)
Model the instrument beyond the ADC — band-limited front end and timebase jitter:

```python
ws.Signal(seed=1, grid=g).carrier("pam4", n_ui=nui, pattern="prbs13q", causal=True) \
  .scope(bw_hz=33e9).timebase(rms_ps=0.5)   # scope bandwidth + sample-clock jitter
```

## Low-speed buses
Embedded-bus signaling (open-drain wired-AND, UART framing):

```python
bus = ws.open_drain([driver_a, driver_b])     # low if any driver pulls; else pull-up high
wave = ws.uart_frame([0x55, 0xA3], samples_per_bit=16)   # idle-high start/stop framing
```

## Two-rate acquisition (simulation grid -> acquisition -> stored record)
Simulate physics on a fine grid, then model the acquisition system (front end + digitizer +
optional record decimation) onto the sample rate / record length that actually gets stored:

```python
from wfmsynth import AcquisitionProfile
prof = AcquisitionProfile(sample_rate_hz=2.5e9, record_length=16384, input_bandwidth_hz=800e6,
                          enob=7, decimation=dict(mode="peak_hold", depth=1024))
sig = ws.Signal(seed=1, grid=ws.Grid(fs=200e9, baud=25e9, n=1<<14)).carrier("nrz", n_ui=256, causal=True)
stored = sig.acquire(prof).waveform()             # what an acquisition system would store
taps   = sig.acquire_taps(prof)                   # simulated / conditioned / digitized / stored
```
Carrier-agnostic (NRZ/PAM4/clocks/buses/optical). `peak_hold` decimation keeps a 2-channel
(min,max) record so narrow transients survive time compression.

## Roadmap and backlog

**[ROADMAP.md](ROADMAP.md)** describes future direction from the current composable,
provenance-first architecture. **[BACKLOG.md](BACKLOG.md)** contains only active work,
known limitations, and concise delivered milestones. Current priorities are consolidating
parallel acquisition/digitization paths, reporting realized rise time, and closing specific
standards-fidelity gaps.

## Tests
```bash
python -m pip install -e ".[test]"
pytest
python -m wfmsynth.validate
```

## License
**0BSD** (Zero-Clause BSD) — see [LICENSE](LICENSE). Maximally permissive: use, copy, modify, and
distribute for any purpose, with **no attribution requirement** and no conditions. Public-domain-equivalent.
