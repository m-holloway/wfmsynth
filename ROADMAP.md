# wfmsynth roadmap

This document describes where `wfmsynth` is going. For the prioritized work with
definitions of done, see [`BACKLOG.md`](BACKLOG.md). For user-facing capabilities and
examples, start with [`README.md`](README.md).

## Where the library is today

`wfmsynth` 0.39 is a validated, composable synthesis engine rather than an early collection
of waveform helpers. The current architecture already provides:

- real-unit `Grid` configuration for sample rate, symbol rate, duration, and delay;
- reproducible `Signal` recipes and independent random streams for controlled pairs;
- electrical and optical carriers, source timing, transmitter imperfections, and drift;
- causal analytic channels, reflections, Touchstone channels, and crosstalk;
- transmitter and receiver equalization, CDR behavior, and multi-lane scenes;
- scope/probe/ADC effects and generalized two-rate acquisition;
- measured ground truth, confounder-controlled sweeps, and sim-to-real separability checks;
- localized events (runt, glitch, ring, droop, …) with independent placement and
  per-window labels for an external segmenter;
- streaming channel application for long records; and
- a physics validation gate plus automated tests.

Provenance-first composition was the original flagship roadmap item. It is now the primary
shipped API (`Signal`, `recipe()`, `dataset()`, and `contrast()`), not future work.

## Roadmap principles

1. **Measured fidelity before feature count.** Compare synthetic and real captures, then
   address the feature that separates them most strongly.
2. **One canonical physical path.** Equivalent operations should not have subtly different
   stage order or semantics across APIs.
3. **Ground truth must survive scrutiny.** Record requested causes, independent random
   factors, realized outputs, and any clamping or approximation.
4. **Real units and physical order by default.** Keep legacy normalized forms compatible,
   but teach and validate the paths users need for real instruments.
5. **Every completed physics item needs an assertion.** Plausible-looking output is not
   sufficient evidence that an effect behaves correctly.

## Near term: consolidate the acquisition path

The highest-value work is reducing parallel implementations, not adding another effect.

- Unify `Signal.digitize()` with the validated `instrument.digitize()` pipeline while
  preserving independent role-based random streams.
- Refactor `deep_capture()` into a PAM4 segmented-data preset over the generalized
  `Signal.acquire()` path without changing default outputs unexpectedly.
- Add absolute rise-time input (`tr_ps`) and record requested, realized, and clamped rise
  time in provenance.
- Separate trigger jitter from sample-clock/timebase jitter and model it only where the
  distinction is observable.

See backlog items #27, #48, and the rise-time follow-up.

## Next: standards and protocol fidelity

The library has useful building blocks, but it does not yet claim complete protocol
implementations.

- Implement full 8b/10b tables, running-disparity selection, K characters, decoding, and
  run-length validation.
- Expose additional standard PRBS patterns through the carrier API.
- Decide whether 128b/130b belongs in this package before advertising it as implemented.
- Add focused low-speed building blocks only where they improve waveform realism; current
  bus support is open-drain composition and UART framing, not complete SPI/CAN stacks.
- Add standards-flavored presets as transparent recipes, not opaque monolithic generators.

## Evidence-driven channel and instrument depth

Use `separability(synthetic, measured, grid)` and measured attribute comparisons to order
this work. Candidate gaps include:

- trigger behavior and other acquisition-specific timing effects;
- a fuller causal dielectric model;
- mixed-mode S-parameters for differential channels;
- parameterized connector, via, and stub discontinuities;
- explicit dual-Dirac / target-BER jitter products;
- ground bounce, EMI, and intermodulation; and
- distribution-level fidelity benchmarks against versioned measured datasets.

These are candidates, not promises in priority order. Real-capture evidence should decide.

## Dataset tooling and scale

Longer-term work that supports ML production workflows:

- dataset manifests containing the generation spec, recipes, engine version, and schema;
- export helpers for well-specified formats such as NPZ, Parquet, and HDF5;
- batched or parallel dataset generation;
- memory-mapped output for records larger than RAM; and
- optional accelerated backends only after profiling shows generation is the bottleneck.

## Intentionally outside the core

The package should stay focused on synthesis and the minimum measurements needed to validate
and label its output.

- General-purpose SI analysis and root-cause diagnosis belong in downstream tools.
- Interactive eye/density visualization should remain optional and not add a plotting
  dependency to the runtime core.
- Vendor-native instrument containers belong in instrument-specific adapters; prefer open,
  documented interchange formats in this package.

## How to propose roadmap work

A proposal should state:

1. the real capture or downstream failure that motivates it;
2. why existing composition cannot represent the effect;
3. the public API and compatibility impact;
4. a measurable physical property; and
5. the validation assertion that will define completion.
