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
