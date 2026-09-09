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

### #52 The record's head after U-16 — a lead-in the composer renders and discards

**Status:** Open. Owner: whoever owns `compose.py` / `instrument.py`; NOT a `physics.py` change.

U-16 made every frequency-domain stage a LINEAR convolution: the response to the record's last
samples no longer wraps onto its first. That is the correct convolution, and it has a
consequence that has to be owned somewhere. Before sample 0 there is now a quiescent line, so a
record that begins mid-pattern begins with a TURN-ON EDGE — the line was never high before the
capture started. A real deep-memory capture is a window on a link that was already running and
has no such edge.

Three measured consequences, on the shipped full-chain recipe (carrier -> channel -> reflection
-> timing -> supply -> crosstalk -> front end -> converter -> export):

- the turn-on is a step in the record's periodic extension, and an unwindowed whole-record FFT
  spreads it flat: the 11-bit stored record's stop band reads **−165.3 dB/Hz instead of its
  lattice's −182.1**. `validate.py`'s floor instrument now windows, which recovers −182.3 and
  still recovers both constructed quantisation floors to 0.03 dB — but a downstream consumer
  measuring the record's PSD without a window will see the edge.
- `scope(kind="brickwall")` is still a CIRCULAR frequency-domain stage, so it rings on that step.
  The overshoot lands on the record's LAST samples at **6.8 % above** anything the link does, and
  the ranged 11-bit code count falls from **1949 to 1830**, outside the 1851–2035 band the three
  real captures set. `validate.py` trims 1024 samples off each end to measure the interior.
- a zero-phase (non-causal) stage additionally rolls the record's TAIL off, because its
  pre-cursor reaches for samples past the end that are not there.

**Done when:** a chain renders `guard` extra samples of the same pattern before the record (and,
for a non-causal stage, after it), applies the channel, and discards the lead-in — so the
delivered record is a window on a running link and carries neither a turn-on nor its ringing;
and `validate.py`'s edge trims and windows are no longer load-bearing.

Sizing it needs no new estimate: `physics.response_extent` already measures exactly how many
samples the lead-in has to be.

### #53 `_op_lossy` and `_op_resonant` drop the `linear` and `guard` keywords

**Status:** Open. Owner: whoever owns `compose.py`.

`compose._op_lossy` whitelists the keys it forwards to `physics.lossy_channel`, so a recipe
cannot ask for `linear=False` (the pinned-length circular convolution) or set `guard=`. The
same is true of the other frequency-domain ops. That makes the U-16 comparison impossible to do
from a recipe, and it means a caller who knows their channel's impulse-response length cannot
skip the probe. Add both keys to the whitelist and to the recipe schema.

### #54 `scope`/`brickwall` and the other frequency-domain stages outside `physics`/`sparam`
still wrap

**Status:** Open. Owner: whoever owns `instrument.py`.

U-16 covered `physics.lossy_channel`, `physics.resonant_reflection`, `sparam.sparam_channel`
and `sparam.cascade_channel`. `instrument.scope`'s frequency-domain kinds (`brickwall`, the
Gaussian mask) are still `irfft(rfft(x) * H)` at the record length and still wrap. They are
also the stages that now ring on the head that #52 describes. `physics.apply_transfer` is the
shared implementation; pointing them at it is the whole change.

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
- Frequency-domain stages in `physics` and `sparam` apply a LINEAR convolution and pad to a
  5-smooth transform length; the guard is MEASURED (`physics.response_extent`, -100 dB below
  the impulse response's peak) and capped at twice the record, so a channel with an algebraic
  tail keeps a residual bounded at that level -- ~3e-5 of a record's span, about a fifteenth of
  one LSB at 11 bits. `guard=` overrides it.
- A record now begins on a quiescent line, so it carries a turn-on edge at its head (#52), and
  the frequency-domain stages outside `physics`/`sparam` still wrap (#54).

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

### The DFE's default amplitude normalisation is the 99th percentile of ISI-corrupted samples

Separate from the timing, and exposed by the same constructed answer. `_op_dfe` normalises the
sampled magnitudes by their own 99th percentile, which post-cursor ISI inflates above the
constellation's full scale, so the slicer's levels sit wrong even on a record with a perfect
clock: measured 12.3 % symbol error on a chain that reads perfectly with `scale=1.0`. The
`scale=` parameter is in; the default is unchanged because changing it is a compatibility
break. An AGC that estimates the level scale from the DECIDED levels rather than a percentile
of the raw samples would fix it without a parameter.

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
