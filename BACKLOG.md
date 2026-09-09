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
| events | Localized placeable needles (`Signal.events` / `place_events`) with per-window labels; clock recovery stays external |

## Follow-ups owed by the acquisition-path work (recovered-clock DFE + probe op)

Left undone deliberately, because each needs a file outside that change's scope.

### `probe_loading`'s default is not the pole it documents

`probe_loading(causal=False)` — still the default, still bit-identical — routes through
`scope_bandwidth`'s zero-phase Bessel, which runs the single pole forwards and backwards. Its
magnitude is therefore the closed form SQUARED: measured 6.027 dB at `fc` where
`10*log10(1 + (f/fc)**2)` says 3.014, and 14.014 where the closed form says 6.989 — a ratio of
1.9995 and 2.0050. Its group delay is zero, where an RC pole's is not. `causal=True` and the
new `probe()` are the closed form to 5e-15 dB and 7e-15 degrees.

**Done when:** the default is the pole, the change is stated as a versioned compatibility
break, and `wfmsynth/validate.py:1124` — which asserts only that HF is attenuated *at all*, a
check that passes on a response twice as steep as the physics — asserts the closed form
instead.

### `wfmsynth/validate.py` has no entry for either mechanism

Both are covered by `tests/test_acquisition_path.py` (41 assertions, incl. 17 closed-form probe
rows) but neither appears in `validate.py`, so `python -m wfmsynth.validate` still reports a
green physics run for a chain with no probe and a DFE that diverges.

**Done when:** `validate.py` carries (a) the RC pole against `1/(2*pi*R*C)` in magnitude and
phase, and (b) a constructed-symbol round trip showing the recovered-clock DFE at zero symbol
errors and the fixed stride at the constellation's chance rate.

### `rx_ffe`'s tap spacing is quantised to whole samples

Not the DFE's defect — a time-invariant FIR cannot accumulate — but a real fixed error:
`int(round(samples_per_ui * spacing_ui))` misplaces every tap by up to half a sample, measured
at 0.020 UI for 7.143 samples/UI, constant from the first block of a record to the last.
Fractionally-spaced taps by interpolation would remove it, and would need `wfmsynth/rx.py`.

### `_op_dfe` renders equalised symbols onto the nominal timebase

`P.from_symbols(eq, n=len(x))` lays the equalised symbols back down at a uniform rate even
when the decisions were taken on a clock that was not uniform. The decisions and their instants
are right (`compose.dfe_decisions` returns both); only the waveform rendering is nominal.

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
