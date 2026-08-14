# wfmsynth

**Physics-informed synthetic waveform generation.** Generate realistic voltage-vs-time
signals — grounded in real signal-integrity physics — for testing, benchmarking, and
training/ground-truth data. numpy/scipy only; every physics primitive is validated by a
hard assertion.

It models the things that actually shape a high-speed signal: frequency-dependent **causal
channels** (skin + dielectric loss), transmission-line **reflections**, **crosstalk**
(NEXT/FEXT), **AC-coupling** wander, and physically **decomposed jitter** (Rj/Pj/DCD) —
plus digital (NRZ/PAM4) and RF (AM/FM/PSK/QAM) carriers, a compositional grammar of
signals, and full **"deep-memory" scope captures** with injected defects and ground truth.

```python
import numpy as np
import wfmsynth as ws
from wfmsynth import physics as P
rng = np.random.default_rng(0)

# a causal (minimum-phase) lossy channel on an NRZ signal
x = P.lossy_channel(P.nrz(n_ui=32, seed=3), length_in=10.0, tand=0.02, causal=True)

# apply a named impairment, then label-preserving domain randomization
y = ws.domain_randomize(ws.apply_impairment("crosstalk", P.pam4(n_ui=32, seed=5), rng), rng)

# a realistic segmented PAM4 deep-memory capture with injected defects (+ ground truth)
cap = ws.deep_capture(n_segments=2000, needle_rate=0.02, seed=1, group_size=8)
# cap["X"] -> (2000, 1024) scope segments ; cap["labels"], cap["needle_idx"], cap["group_id"]
```

## Install
```bash
pip install numpy scipy          # runtime deps
pip install -e .                 # from a clone
```

## What's inside (`wfmsynth/`)
| module | what it provides |
|---|---|
| `physics` | low-level primitives: `lossy_channel` (causal option), `multi_reflection`, `crosstalk`, `ac_couple`, `inject_jitter`, `nrz`, `pam4`, `am/fm/psk/qam`, `chirp`, `pdn_transient`, `ecg_like`, `family_bank` |
| `impairments` | `apply_impairment(name, x, rng)`, `domain_randomize(x, rng)`, and the `IMPAIRMENTS` vocabulary (14 grounded faults) |
| `grammar` | compositional signals — `carrier()` × `envelope()` → `sample()` / `generate()`; covers a broad "shape manifold" by composition rather than enumeration |
| `pam4` | `deep_capture()` — realistic segmented PAM4 scope captures (internal high-res → scope-rate digitization with thermal noise + finite ENOB), with channel-class defects grouped across a shared link |
| `validate` | `python -m wfmsynth.validate` — hard assertions that each primitive does what it claims |

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

## Roadmap and backlog
**[ROADMAP.md](ROADMAP.md)** is the breadth map — everything this engine could model. The
flagship next step is **provenance-first composable synthesis**: building each signal as a
component graph that records every knob value, so a generated waveform carries a complete,
serializable *recipe* (exact ground truth for training, fully reproducible). v1's
primitives are the building blocks that layer sits on, so it's additive, not a breaking
change.

**[BACKLOG.md](BACKLOG.md)** is the prioritized, actionable list — what to build next, why,
and what "done" looks like for each (a hard assertion in `wfmsynth.validate`).

## Tests
```bash
pip install pytest && pytest        # runs the physics validation + smoke tests
```

## License
MIT — see [LICENSE](LICENSE).
