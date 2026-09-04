# wfmsynth examples

These examples form a learning path. You do not need to understand signal integrity before
starting; each step introduces only the concepts needed for the next one.

Run commands from the repository root after installing the project:

```bash
python -m pip install -e ".[test]"
python examples/quickstart.py
```

## Recommended order

### 1. Start with a signal path

[`quickstart.py`](quickstart.py) builds an NRZ source, passes it through a bandwidth-limited
channel, models a scope acquisition, and shows how the same recipe reproduces exactly.

Learn:

- a waveform is a NumPy array of voltage samples;
- `Grid` binds those samples to real time and symbol-rate units;
- a `Signal` chain follows the physical order of the system; and
- `recipe()` is the exact provenance for an ML example.

### 2. See what a realistic system contains

[`realistic_scenario.py`](realistic_scenario.py) composes:

- an electrical PAM4 link;
- an optical PAM4 lane; and
- a two-lane scene with shared power noise and crosstalk.

This is the end-to-end reference, not the first API tutorial. Read the comments stage by
stage and remove one operation at a time to see what changes.

### 3. Build trustworthy ML datasets

Run these in order:

1. [`provenance.py`](provenance.py) — store exact generation recipes and make controlled
   contrastive pairs.
2. [`ground_truth.py`](ground_truth.py) — measure labels from the generated output instead
   of assuming requested knobs equal realized behavior.
3. [`confounder_sweep.py`](confounder_sweep.py) — prevent an ML model from learning an easy
   correlated shortcut.
4. [`sim_to_real.py`](sim_to_real.py) — identify which simple feature distinguishes a
   synthetic set from measured captures.

The distinction that matters:

- **Recipe values** say what was requested.
- **Measured labels** say what the complete pipeline produced.
- **Controlled pairs** change one factor while preserving the rest.
- **Confounder control** prevents a target factor from being predictable through an
  unintended correlated feature.

### 4. Model the acquisition explicitly

[`two_rate_acquisition.py`](two_rate_acquisition.py) separates:

1. fine-grid physical simulation;
2. analog input bandwidth;
3. sampling onto an instrument-rate grid;
4. ADC noise and quantization; and
5. stored-record decimation.

Use this pattern when training data should resemble what a scope or digitizer stores rather
than an ideal simulated node.

### 5. Specialist topics

| Example | Use it when |
|---|---|
| [`non_integer_sps.py`](non_integer_sps.py) | Sample and symbol clocks are unrelated, as they normally are in hardware |
| [`clock_recovery.py`](clock_recovery.py) | The receiver or scope CDR changes which jitter remains visible |
| [`touchstone_channel.py`](touchstone_channel.py) | You have measured `.sNp` channel data or need resonances absent from an analytic model |
| [`events.py`](events.py) | Rare localized defects (runt, glitch, ring, droop) in a long record, labelled per UI window |

## Choosing a dataset generator

| Need | Recommended entry |
|---|---|
| Reproducible realistic links with detailed labels | `Signal` + `Grid` + `dataset()` |
| Existing array plus one named labelled fault | `apply_impairment()` |
| Label-preserving variation across capture conditions | `domain_randomize()` |
| Broad, protocol-agnostic morphology pretraining | `generate()` |
| A ready-made segmented PAM4 defect benchmark | `deep_capture()` |
| Rare needles in a long record, labelled per window | `Signal.events()` + `label_windows()` |

`deep_capture()` and `generate()` are specialized dataset presets. For a new system model,
prefer a `Signal` recipe so the physical stages and their ordering are explicit.

## Practical realism checklist

Before calling synthetic data “realistic,” check:

- Are sample rate, symbol rate, and record length plausible for the target instrument?
- Is the source causal, and are timing effects applied at the source?
- Does the signal include ordinary nominal imperfections, not only labelled faults?
- Are channel, receiver, scope/probe, and ADC stages in physical order?
- Are nuisance ranges smaller than the labelled fault ranges?
- Are labels measured from the final output where possible?
- Can `separability()` distinguish synthetic from measured data using an obvious shortcut?

For definitions of UI, ISI, jitter, equalizers, ENOB, and other terms, see the glossary in
the root [`README.md`](../README.md).
