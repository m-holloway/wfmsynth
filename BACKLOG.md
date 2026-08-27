# wfmsynth prioritized backlog

This is the actionable work list for `wfmsynth`. The broader direction lives in
[`ROADMAP.md`](ROADMAP.md); usage guidance lives in [`README.md`](README.md).

The previous backlog mixed active work with long essays for more than forty completed
items. Git history preserves those implementation narratives. This file now keeps only
current work, known limitations, and a compact delivered-milestone record.

## Definition of done

An item is complete when:

1. its public behavior and physical assumptions are documented;
2. a hard assertion in `wfmsynth.validate` checks the claimed physical property;
3. automated tests cover API behavior and compatibility;
4. provenance records any new randomized or approximated values; and
5. existing defaults remain bit-identical unless a deliberate versioned change is stated.

Priority order is: measured fidelity, ground-truth correctness, consolidation, then breadth.

## P0 — canonical pipelines and trustworthy provenance

### #27 Unify `Signal.digitize()` with `instrument.digitize()`

**Status:** Open.

The composer's `_op_digitize` and `instrument.digitize()` independently implement ADC
stages. The instrument path has the canonical order:

```text
noise -> interleave mismatch -> clipping -> quantization
```

The composer uses separate role-based random streams and currently omits clipping. Delegation
must retain those independent streams so `Signal.contrast()` remains a valid controlled
ablation.

**Done when:**

- `Signal.digitize()` delegates to one canonical instrument pipeline;
- noise and interleave role streams remain independently rerollable;
- clipping is available through the composed path;
- recipe round trips and contrastive-pair assertions pass; and
- any intentional output change is documented as a versioned compatibility change.

### #48 Refactor `deep_capture()` onto generalized acquisition

**Status:** Open.

`pam4.deep_capture()` still owns a fixed PAM4-specific digitization path. General acquisition
now exists in `AcquisitionProfile`, `Signal.acquire()`, and `acquire_record()`.

**Done when:**

- `deep_capture()` is a segmented PAM4 dataset preset over the generalized acquisition
  stages;
- segment labels, `needle_idx`, `group_id`, and default array shapes remain stable;
- the shared-link defect model remains intact;
- acquisition settings and realized rates appear in provenance; and
- compatibility tests prove whether defaults are bit-identical or explicitly migrated.

### #49 Absolute and realized transmitter rise time

**Status:** Partial. The former B2 silent-clamp bug is fixed: low samples/UI now produces a
visible warning and validation covers the floor. Absolute rise-time input and recipe
reporting are still missing.

At low samples/UI, a requested fractional rise time can be below the representable limit.
Clamping is physically necessary; hiding the realized value is not.

**Done when:**

- carriers accept `tr_ps` when a `Grid` is present;
- recipes record requested rise time, realized rise time, and whether clamping occurred;
- the fractional `tr_frac` path remains compatible;
- validation shows realized rise time follows requests over the usable range; and
- unrepresentable requests remain visible and never produce a response below the sampling
  limit.

### #51 Make named impairments and domain randomization length-aware

**Status:** Open.

`domain_randomize()` still draws some nuisance arrays at the legacy global length `N=4096`.
Passing a valid waveform with another record length can fail with a shape mismatch, which
conflicts with the library's general rate/record-length parameterization.

**Done when:**

- every nuisance and named impairment infers length from its input array;
- default 4096-point output remains bit-identical;
- tests cover shorter, default, and deep-memory inputs; and
- `generate()` behavior and labels remain unchanged.

## P1 — standards and missing instrument distinctions

### #46 Full 8b/10b coding

**Status:** Open.

Current `dc_balanced()` is an 8b/10b-style block-inversion helper, not a standards-complete
8b/10b codec. Current 64b/66b scrambling is implemented; 128b/130b is not.

**Done when:**

- 5b/6b and 3b/4b tables are implemented;
- running disparity and K control characters are supported;
- encoded streams satisfy run length <= 5 and remain DC-balanced;
- decoding round-trips valid input bytes and control symbols; and
- arbitrary encoded symbols drive `Signal.symbols()` end to end.

### #50 Trigger jitter distinct from timebase jitter

**Status:** Open.

`timebase_jitter()` models sample-clock uncertainty. Trigger jitter is not currently a
separate primitive, despite older acquisition-chain descriptions grouping the two.

**Done when:**

- the API states exactly which alignment varies and which samples remain unchanged;
- repeated acquisitions exhibit the requested trigger-position distribution;
- single-record sample-clock jitter remains a separate effect; and
- validation distinguishes trigger movement from edge smearing.

## P2 — evidence-ordered fidelity work

Do not implement these in list order by intuition. First compare synthetic and measured
captures with `separability()` and realized metrics, then promote the strongest observed gap
into a scoped item with a definition of done.

Candidate areas:

- mixed-mode Touchstone handling for differential channels;
- a fuller causal dielectric model;
- explicit connector/via/stub networks;
- dual-Dirac and target-BER jitter products;
- ground bounce, EMI, and intermodulation;
- additional standards-flavored carrier recipes; and
- distribution-level regression against versioned measured datasets.

## P3 — dataset tooling and scale

These improve production workflows but do not outrank demonstrated fidelity gaps:

- dataset manifest and schema versioning;
- NPZ, Parquet, and HDF5 export helpers;
- batched/parallel generation;
- memory-mapped long-record output; and
- optional accelerated backends after profiling.

## Known limitations and claim boundaries

Keep these statements synchronized with user-facing documentation:

- `Signal.digitize()` and `instrument.digitize()` are not unified yet (#27).
- `deep_capture()` is a specialized parallel acquisition path (#48).
- Rise-time clamping is visible, but absolute `tr_ps` and realized recipe fields are not
  implemented (#49).
- `domain_randomize()` currently expects the legacy 4096-point record length (#51).
- `dc_balanced()` is not full 8b/10b; 128b/130b is not implemented (#46).
- Acquisition models timebase/sample-clock jitter, not a distinct trigger-jitter process
  (#50).
- Low-speed bus support currently covers open-drain composition and UART framing/decoding.
  It is not a full SPI, CAN, or RS-485 protocol stack.
- Touchstone support applies selected single-ended S-parameters; mixed-mode conversion is
  future work.
- `simreal.separability()` provides a diagnostic method, but the repository does not ship a
  measured-capture benchmark corpus.

## Delivered milestones

The following capabilities are implemented, tested, and represented in the validation gate.
Version labels refer to the historical development sequence, not separate supported release
branches.

| Backlog items | Delivered capability |
|---|---|
| #1–4 | Real-unit grids, non-integer samples/UI, ADC artifacts, source-applied jitter |
| #5–10 | Reproducible recipes, role-based RNG, controlled sweeps, measured labels, constant-power mixing, intermittent masks |
| #11–12 | Tx FFE and clock-recovery transfer |
| #13–20 | Touchstone channels, resonant reflections, nominal nonlinearity, multiple aggressors, realistic noise, causal-chain checks, pattern lock, streaming |
| #21–26 | Sim-to-real diagnostics, critical-sampling guidance, CI foundation, canonical instrument quantization/digitization/absolute offsets |
| #28–38 | Rx CTLE/DFE, waveform CDR fold, shot noise, SSC, differential pairs, supply coupling, optical primitives, timing sources, scenes, phase noise, drift |
| #39 | DC-balance helper and 64b/66b scrambling; standards-complete 8b/10b remains #46 |
| #40 | Scope bandwidth, probe loading, and timebase jitter; trigger jitter remains #50 |
| #41–45 | Open-drain/UART primitives, de-emphasis presets, electrical idle/LFPS, laser chirp, arbitrary-symbol carriers |
| #47 | Generalized two-rate acquisition and record decimation |

## CI status

The current workflow uses a fast Linux/Python 3.12 gate for pull requests and a scheduled
Linux/Windows matrix for Python 3.9 and 3.12. It does not currently run macOS or a full
cross-platform matrix on every push. Update this section if the workflow policy changes.

## Contributing an item

Before adding work:

1. show the measured-data gap or downstream failure;
2. explain why composition of existing primitives is insufficient;
3. state compatibility and provenance effects;
4. define the measurable property and acceptable tolerance; and
5. write the validation assertion that will mark the item complete.

Completed items should move into the milestone table rather than accumulating historical
design essays in the active queue.
