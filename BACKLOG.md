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

**Status:** DELIVERED in `compose.py` as `Signal(lead_in=...)` / `.with_lead_in()`, default OFF
(off is byte-identical: 7 of 7 sampled chains, waveform and recipe hash). See the closing
notes at the end of this item for what it did and did not fix.

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

**What was delivered, measured.** `Signal(lead_in=True)` renders `guard` extra samples of the same
pattern before the record and after it, runs the chain on the longer record and delivers the middle;
the guard is `physics.response_extent` on the chain's own combined impulse response (the LTI ops
applied to a unit impulse), rounded UP to a whole UI so samples-per-UI (`n/n_ui`, exact rational)
cannot move. On an exactly periodic constructed record whose steady-state answer is `np.convolve`'s
and nothing of ours, a 5-UI lead-in recovers the WHOLE record to 4.4e-16, where the same record
without one is wrong by 0.696 of a +/-1 signal in 106 samples split across its two ends. The sizer
recovers a closed-form extent exactly (a lumped reflection's echo train: measured 433 = 2*td*k_max+1).
On the shipped chain, with NOTHING TRIMMED: the ranged 11-bit code count is 1950 against 1840
(1851-2035 is the band the three real captures set), the Hann-windowed stop band is -0.02 dB on its own
lattice's `q**2/12` and +3.02 dB with dither, and the brickwall's 6.2 % edge overshoot is gone.
`validate.py`'s 1024-sample edge trim is removed. The guard is also floored at the SOURCE's own
settling (`32 * tr`), which no impulse probe can see because a carrier is not a filter of the record --
without that floor `lead_in='auto'` is zero on a chain whose other ops are all memoryless, and the
record's last edge stays missing.

**What it costs.** The render is `(n + lead + tail)/n` longer, and the extent probe adds a few FFTs.
Measured on a 1 M-sample record at 256 GSa/s: a grid-consistent carrier (16 samples/UI) pays 1.087x
the samples and 0.28 s against 0.10 s; the shipped recipe's 32-UI carrier pays 1.312x and 0.18 s
against 0.09 s -- more, because its rise time is 4,915 samples (`tr_frac` of a 32,768-sample UI) and
the source's own settling floor is what sizes it, not the channel.

**Three things it also fixed that the item did not name.** (a) The record's LAST samples were wrong
before any channel at all: `physics._shape_edges` is `sosfiltfilt`, whose padding invents the samples
past the end, so a constructed record's final falling edge was simply MISSING -- its last four samples
read 1.0/1.0/1.0/1.0 where the running link's are 0.961/0.824/0.568/0.204 (0.796 of full scale, 68
samples across the two ends). A lead-in delivers them to 0.0e+00. (b) The vertical: `store`/`digitize`
with an automatic full scale must range to the DELIVERED WINDOW, not to the guard -- ranging to the
guard gives 1850 codes against 1950, i.e. the same defect wearing a different hat. (c) #54's wrap is
confined to the guard (see #54).

**What it does NOT fix, stated so nobody re-discovers it.** The record's stop band read on an
UNWINDOWED whole-record periodogram is still not `q**2/12`, and a lead-in makes it worse, not better:
a window on a running link starts and ends mid-pattern, so its periodic extension has a step of order
the signal amplitude (measured |x[-1]-x[0]| = 1.13 against 0.15 without a lead-in) and a rectangular
window leaks it flat across the band (+7.4 dB at n = 1 M, +3.1 dB at n = 65 k, against +0.00 / +0.12 dB
Hann-windowed). This is not a defect the lead-in should chase: REAL captures are windows on running
links and are not periodic either, so a window is the correct instrument for any record's PSD. The
only reason a rectangular window ever worked here is that the pre-U-16 circular convolution made
records exactly periodic. `validate.py`'s floor and centroid instruments stay windowed for that
reason, and that is no longer a workaround.

**Not covered, deliberately.** A lead-in REFUSES `events`, `drift`, `acquire`, `digitize(n_out=)` and
every record-fraction knob (`td_frac`, `fc_frac`, `f0_frac`, `dfe(phase=)`), because a lead-in moves
the record's origin and lengthens the record those fractions are fractions of -- see #55.

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

**Measured against the lead-in (#52), because "a lead-in makes it harmless" is a claim, not a fact.**
The shipped 1 M-sample chain through the same 32 GHz brickwall, applied circularly and applied as a
linear convolution, differs by **362 LSB** of an 11-bit record INSIDE the delivered record with no
lead-in. With an auto-sized lead-in (163,840 samples each end) the difference inside the delivered window is
**0.005 LSB**, and 0.027 LSB at 16,384 samples of guard -- a fortieth of a code or less, so it cannot
move a stored code except on a rounding tie. Guard sweep on a second 1 M record on the same grid (a 65,536-UI carrier, so the guard quantum
is 16 samples rather than 32,768): no lead-in 220 LSB; guard 4,096 -> 0.086 LSB, 16,384 -> 0.027,
45,840 -> 0.014, 131,072 -> 0.006. The wrap falls only as ~1/guard, because the stage's impulse
response is a sinc -- 243,269 samples long at -100 dB, a quarter of the record -- which is also why
containing it fully is not the bar; being under one code is.

So: a lead-in CONFINES this defect to the samples that are thrown away; it does not fix the stage.
The 362-LSB error is still there, in the guard, and every caller who uses `scope(kind="brickwall")`
or the Gaussian mask WITHOUT a lead-in still receives it. This item stays open, and it is the single
biggest remaining reason a record needs a lead-in at all.

### #55 The ops a lead-in refuses: shift the record's origin instead of rejecting

**Status:** Open. Owner: whoever owns `compose.py` (`_LEAD_REJECT` / `_LEAD_RELATIVE`).

`Signal(lead_in=...)` refuses five things rather than silently changing what they mean, and each is a
real gap:

- `events` places a mechanism at an absolute position in the record; a lead-in moves the origin, so
  every anchor (`sample`, `t`, `frac`, and the realized times `EventList` reports) needs shifting by
  the lead-in and the mask needs slicing to the window. This is the one that matters most -- a fault
  dataset cannot use a lead-in today.
- `drift`'s profile is defined ACROSS the record ("0 to 1 over the capture"), so a guarded render is a
  different drift; it needs to be told the window rather than the array.
- `acquire` resamples onto a second rate, so the guard's sample count is not the record's; the window
  has to be sliced on the OUTPUT rate.
- `digitize(n_out=)` changes the record length for the same reason.
- the record-fraction knobs (`reflect(td_frac=)`, `resonant_reflect(td_frac=/f0_frac=)`,
  `ac_couple(fc_frac=)`, `dfe(phase=)`) are fractions of a record the lead-in lengthens. These have
  absolute-unit equivalents already; the refusal names them.

Also unfixed, and NOT refused because it only changes an amplitude scale: ops that scale themselves by
`np.ptp(x)` of the whole array -- `physics.crosstalk`'s coupling and `impairments.drift(kind="dc")` --
see the guarded render's span, not the window's, and the guarded span is larger by whatever the turn-on
does. Only `store` and `digitize` are window-ranged today (`_LEAD_WINDOW_RANGED`). Threading the window
into those primitives is a `physics`/`impairments` change, not a composer one.

### #56 A dataset-wide lead-in policy, and whether it should be the default

**Status:** Open. Owner: dataset/authoring layer.

`lead_in` is off by default, because turning it on changes every rendered record and this kernel is
pinned by SHA and diffed sample-for-sample downstream. But a record without one is a record that begins
on a step no link makes, and #52's measurements say the cost of the truth is 6 % of the render for a
typical chain (1.062x-1.125x on the shipped recipe). The decision that has to be made deliberately,
with a version label: which datasets are rendered with a lead-in, and whether v5 flips the default. Note
that a lead-in ADVANCES the pattern by `lead_ui` symbols (the history has to be genuine), so a v4 record
rendered with one is not the same record -- unless the lead-in is a whole multiple of the pattern's
period, in which case the symbols are identical and only the history is added. A dataset that wants both
should size its lead-in up to the next multiple of its pattern period.

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
- A record rendered WITHOUT `Signal(lead_in=...)` begins on a quiescent line, so it carries a
  turn-on edge at its head, its last edge is missing (`sosfiltfilt` padding), and the
  frequency-domain stages outside `physics`/`sparam` still wrap 362 LSB of an 11-bit record onto it
  (#52 delivers the lead-in, #54 is the stage that still wraps). `lead_in` is off by default (#56).
- A record's PSD must be measured through a window. A window on a running link is not periodic, so
  a rectangular whole-record periodogram reads its own edge step, not the record's floor: +7.4 dB
  above `q**2/12` at n = 1 M, against +0.01 dB Hann-windowed (#52).

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
