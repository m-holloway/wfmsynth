# Changelog

Notable changes to `wfmsynth`, newest first.

This file starts at 0.41.0 and reconstructs the earlier versions from git history, because
nothing recorded them at the time — the first three entries are therefore summaries of a commit
range rather than release notes anyone wrote. Tags were added retroactively at the last commit
carrying each version, so `git diff v0.39.0..v0.40.0` is exact even though `v0.40.0` was never
cut as a release.

Dates are the last commit in each range. The project uses [semantic versioning](https://semver.org)
loosely: it is pre-1.0, so a minor bump may change behaviour, and anything that changes rendered
samples is called out explicitly under **Changed output** below.

## 0.46.0 — 2026-11-12

Chains can meet, and a record can be complex baseband. Until now a chain was a line with one
source, and signals met only inside a few special ops, each with its own nested format. This
release adds the small set of primitives that let chains meet (#75–#80, #85, #86, tracked by
#87), and an I/Q domain with its mixers, impairments, file formats and measures (#81–#84).
Every existing call and recipe is unchanged, and so is every rendered sample but those of a
`square` or `cmos` carrier whose edges were not on samples; see **Changed output** and *How it
stays exact*.

On top of those, the same release uses them (#88–#94): an OFDM source, a channel and an
amplifier on the envelope, a source that dwells in states, a module of shipped compositions
(`wfmsynth.recipes`), the optical field as a third domain, and a swept-wavelength bench. Most
of that is compositions, not ops; the entries are grouped under *Added* below.

Read [docs/COOKBOOK.md](docs/COOKBOOK.md#composing-a-realistic-capture) for these in order.
The rule that decided what became an op and what did not, and the table of how each request
was ruled, is in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md#what-becomes-an-op).

### Added

- **A sub-chain is a value** (#75). Every parameter that takes a signal takes a chain, in one
  form: a `Signal()` or `{"ops": [...]}` is part of this record and draws under its seed;
  `{"from": recipe}` is another record, pinned as rendered. Wired into
  `crosstalk(aggressor=)`, the new `crosstalk_matrix(aggressors=[...])`, `modulate(message=)`,
  `open_drain(second=)`, `hybrid_echo(own=)`, `pass_fet(gate=)`, `events(aggressor=)` and a
  `net` driver's `source`. `Signal.summary()` prints a chain as a tree,
  `subchains()` / `subchain(namespace)` return a sub-chain and its waveform exactly as used, and
  `roles()` / `contrast()` reach inside under the namespace (`crosstalk/1/aggressor/jitter/0`).
  An event inside a sub-chain is listed with `Event.at` set and, unless that chain's samples
  are the record's (a `sequence` segment, a selected `switch` input, a `use`d chain),
  `Event.on_record` False; `EventList.on_record()`, which `mask()` goes by, is the
  rest. A sub-chain off the record's grid is refused, never resampled.
- **`combine`, and `gain` / `offset` / `gate` / `delay` driven by a signal** (#76, #79).
  `combine(others, how="sum"|"product"|"min"|"max", gains=)`; the other four take `by=`, a
  number or a chain. `gate` emits its switching instants as events. `delay` is a causal,
  band-limited variable fractional delay whose error is stated (`delay_errors()`: 2.3e-5 of a
  tone below 0.4 fs) and whose rate of change is bounded (`max_rate`).
- **`sequence` and `switch`** (#77). Two sources made of chains: segments end to end with
  stated durations, `restart` or `continue` state and a `cut`, `crossfade` or `hold` join; or N
  inputs selected by a signal crossing thresholds. Boundaries and changes are events;
  `ws.sequence.layout` gives every segment's sample range for labels.
- **Several correlated channels from one recipe** (#78). `ws.Channels({name: chain},
  shared={name: chain}, seed=, grid=)` and one op, `use(name)`: a shared stage is applied, a
  shared signal is referred to. Each chain draws under its own name, so channels through one
  shared PLL see one realisation and `contrast("pll/timing/0")` moves exactly those channels.
  One `recipe()`, one `sha256()`, `Channels.from_recipe`, `write_hdf5` (channel names and the
  recipe travel in the file; `hdf5.read_recipe`, `hdf5.channel_names`). `rec.signal(name)` is
  one channel as a `Signal` that renders exactly as it does in the record, so `transmitted()`,
  `rf_report()`, `lo_phase()`, `subchain()` and `realize()` work per channel.
- **Two general stages, `law` and `filter`** (#80). Eight named transfer laws (`tanh`, `poly`,
  `limiter`, `deadzone`, `rectify`, `quantize`, `comparator`, `diode`) and ten named linear
  filters (`butter`, `bessel`, `cheby1`, `elliptic`, `rrc`, `fir`, `sos`, `zpk`, `integrate`,
  `differentiate`), never callables. A comparator's transitions are events with the sub-sample
  crossing time; `filter_delays()` reports each filter's group delay and, for an edge fast
  against the filter, its 50 % step delay (a different number); `zero_phase=True` removes
  the delay without squaring the response. Asking for a shape a named op owns is refused and
  names that op.
- **A triggered acquisition** (#85, and BACKLOG #50). `trigger(pre_s=, post_s=, ...)` finds an
  edge, pulse-width or runt condition on the realised waveform and keeps a window about each
  firing; `n_segments`, `holdoff_s`, `average`, and `jitter_s` as trigger jitter distinct from
  `timebase`. `trigger(on=<chain>)` triggers from another signal, which in a `Channels` record
  cuts every channel at the same instants. Each firing is an event carrying its exact time.
- **Unit-to-unit variation** (#86). `ws.tol(nom, tol_pct | tol_abs=, dist=, group=)` and
  `ws.corners({...})` go where a number goes. The value is drawn once on the way into the
  recipe, which keeps both it and its stated tolerance. `Signal.unit(seed)` /
  `Channels.unit(seed)` is another unit under the same noise; `ws.unit_dataset` builds units ×
  captures and `ws.unit_of(recipe)` is the key for a held-out-unit split.
- **Complex baseband records** (#81). An `iq` source (BPSK, QPSK, QAM-16/64 or literal
  symbols, root-raised-cosine), `upconvert` to a real passband record and `downconvert` back
  with a stated low-pass (`lpf=dict(pass_hz=, stop_hz=, atten_db=)`, `fs_out_hz=`), and
  `Signal.transmitted()` as ground truth. A record's domain follows its source: `iq(...)`, a
  complex `capture` or a `sequence` of I/Q chains makes it complex baseband with no more
  said, `Grid(fc_hz=)` states the centre frequency, and a grid that states a domain
  (`domain="real"` / `"iq"`) is held to it, a contradicting source being refused by name.
  An op is real-only unless it has opted in (`wfmsynth.domain`), so nothing drops an
  imaginary part.
- **I/Q recordings in open formats** (#82). `ws.read_iq` and `capture(format=)` read SigMF,
  raw interleaved, CSV and MAT; `ws.to_sigmf` and `ws.to_interleaved` write. SigMF annotations
  and events convert both ways; interleaved export can write a marker channel. A byte-swapped
  file is detected and refused.
- **I/Q impairments** (#83). `dac`, `iq_imbalance`, `iq_offset`, `phase_noise` (a stated
  dBc/Hz profile), `freq_offset`, `awgn` (`es_n0_db=`, `eb_n0_db=`, or `snr_db=` in a stated
  `bw_hz=`) and `spur`; `rf_report()` says what each request came to on this record and `lo_phase()` returns
  the oscillator's phase per sample.
- **RF measures against ground truth** (#84). `ws.rf.spectrum`, `band_power`, `aclr`, `ccdf`,
  `constellation_density`, and `evm` against the transmitted symbols, which fits nothing unless
  a correction is named and returns what each removed. With `reference="decided"` a carrier
  is acquired blind before any decision is made (the line in the M-th power of the symbols:
  an offset within &plusmn;baud/(2M) at any phase) and the lock is checked; a record it
  cannot lock to is refused, not measured. Measured against the truth-referenced figure on
  QAM-16 2 kHz off: 0.32 % / 0.32 % clean and 6.29 % / 6.29 % at Es/N0 24 dB, where a
  decision-directed fit alone returned 24.5 % and 24.9 %. The estimate repeats every
  baud/M, and the result says so: `["acquisition"]` always carries `pull_in_hz` and
  `alias_spacing_hz`. Past the range (3.2 MHz on QAM-16 at 25 MBd) a call without `sent`
  returns the alias, -3.05 MHz, with those two numbers; a call with `sent` sees three
  decisions in four wrong as corrected and none with the constellation turned a step per
  symbol, and is refused as the wrong alias, naming the offset. Where there are too few
  symbols for a line (128 of clean QAM-16) the decision-directed fit from zero offset is
  used and kept only if self-consistent (rms error under half the distance to a decision
  boundary, one more pass moving the frequency under 0.01 turn): 0.315 %, the truth's
  figure. At zero offset the figure is that of the fit alone to four decimals of a percent
  from Es/N0 14 dB up, and differs by up to 0.010 points at 10 dB and 0.028 at 6 dB
  (QAM-16, where a quarter to a half of the decisions are wrong either way).
- **`Signal.copy()`** branches a chain. A builder method appends to its Signal and returns
  it, so reusing a base builds one chain holding every variant's ops;
  `base.copy().lossy(...)` is an independent Signal with the same recipe, digest and samples.
- **`tools/digest_compare.py`** digests a tree (31 fixed chains and every render every
  example makes) and compares two digests, so "nothing that existed moved" is a command.
- **The pieces together.** `gain`, `gate`, `combine` (sum), `sequence`, `switch`, `filter` and
  `use` are defined on an envelope and the rest refuse it naming the alternative; a chain that
  did not state its grid starts in the domain its own ops need; tolerances work inside a
  `Channels` record, with `group=` making one part across chains.
- **OFDM** (#88). `Signal.ofdm(n_fft=, scs_hz=, cp=, active=, pilots=, constellation=,
  window=, ...)` is a second I/Q source: an inverse FFT per symbol and a cyclic prefix, on a
  grid at the FFT rate or a whole multiple of it (refused otherwise, never resampled).
  `transmitted()` returns the resource grid; `ws.ofdm.plan`, `resource_grid` and `demodulate`
  are its layout and receiver; `rf.evm(..., ofdm=)` is EVM per resource element with
  `"timing"`, `"common_phase"`, `"gain"` and `"equalizer"` corrections and
  `evm_per_subcarrier` / `evm_per_symbol`. `ws.ofdm.register(name, source=, ...)` names a
  parameter set; the recipe keeps the numbers. Frames, preambles and per-block constellations
  are `sequence` and `combine` of `ofdm` chains. Measured: round trip 6.9e-16; EVM in noise
  within 1 % of analytic.
- **A channel on the envelope** (#89). On an I/Q record `delay(by=)` shifts the envelope and
  turns the carrier (-2π·fc·delay; `fc_hz` must be stated), `gain(by=)` takes a complex number
  or an I/Q chain, and `lossy` / `sparam` read their response at `fc_hz + f`
  (`wfmsynth.baseband`), each within 1e-4 of peak of `upconvert` → the op → `downconvert`.
  `law("saleh")` (AM/AM and AM/PM) and `law("rapp")` act on an envelope's magnitude;
  `stages.am_am_pm` is their curves. `filter("doppler", f_hz=, shape="jakes"|"flat")` and
  `awgn` as a first op (a complex noise source, `ref_power` required) are what fading is
  made of.
- **A source that dwells in states** (#90). `carrier("states", levels=, dwell=, transitions=,
  start=, tr_s=)`: a renewal process with exponential, lognormal, uniform, fixed or tabulated
  dwells, sub-sample jumps, and one `state` event per interval. `ws.states.intervals(events,
  at=)` and `ws.states.occupancy(...)` read and predict it.
- **`wfmsynth.recipes`: shipped compositions** (#89, #90, #91, #94). Functions that return a
  chain of existing ops, visible in `summary()`: `hum`, `ripple`, `pickup`, `flicker` (with
  `flicker_psd`); `bounce` (with `bounce_times`), `intermittent`; `oscillator_startup`,
  `multitone` (with `multitone_plan`), `buck` (with `buck_ripple`; 5.4345 mV of ripple
  against the textbook 5.4375); `fading`, `multipath`; `mzi`, `stray_reflection`, `ring`,
  `gas_cell`. A recipe takes numbers and refuses a `ws.tol` by name. `spur` may be a first op
  (an unmodulated carrier), which is what an I/Q `multitone` is.
- **The optical field is a domain** (#92). Between `eo` and `photodetect` a record is
  `"optical"`: `stage_kinds(domains=True)` and `summary()` say so, `filter`, `gain`, `gate`,
  a sum `combine` (a coupler, coherent), `sequence`, `switch` and `use` act on the field, and
  every other op refuses it naming `photodetect`. `Grid(domain="optical")` starts a chain in
  the field from a capture. A chain through `eo` still starts real: recipes and digests of
  every existing optical link are unchanged.
- **A swept-wavelength measurement** (#93, #94). `Signal.laser(start_nm=, stop_nm=,
  speed_nm_per_s=, bow_nm=, ripple_nm=, ripple_hz=, hops=)` is a field held in the frame of
  its own moving carrier; `resonator(wavelength_nm=, intrinsic_hz=, coupling_hz=, fsr_hz=,
  split_hz=, by=)` is a coupled-mode ring solved in that frame (ring-down against a closed
  form to 5.8e-6); `delay` on a laser's field makes interferometer fringes.
  `ws.swept.wavelength_map(sig)` is the ground-truth axis and `ws.swept.steady` a resonator's
  settled response; `ws.measure.swept_map(ref, grid, ...)`, `transmission` and `resonances`
  read a record as a bench does, without the recipe.
- **Eighteen examples**, all run by the test suite: `state_dwell`, `clutter`, `ofdm_frame`,
  `fading_multipath`, `optical_field`, `swept_ring`, and: `sub_chains`, `joins`,
  `sequence_and_switch`, `clock_and_data`, `law_and_filter`, `triggered_capture`,
  `unit_variation`, `composed_capture`, `iq_passband`, `iq_recordings`, `iq_impairments`,
  `rf_measures`. `composed_capture.py` uses the composition primitives on one two-channel
  triggered capture and isolates each cause of its jitter by changing one named thing.

### Changed output

- **`carrier("square")` and `carrier("cmos")` place each edge at its own time, to a fraction
  of a sample.** Both built their ideal train by testing the phase on the samples, which did
  two things that are the grid and not the signal. An edge between two samples went to the
  next one, so every edge was late by a different amount: one sample peak to peak of
  deterministic jitter. And an edge exactly on a sample was decided by the rounding of
  `f * t % 1.0`: 1 GHz on a 64 GSa/s grid, 64 samples a period, had 35 of its 510 periods
  rendered as 63 or 65. Edge times are now computed directly, and an edge between samples is
  the on-sample edge delayed by its fraction (an 8-tap Lagrange fractional delay, applied
  before the edge shaper). Measured at 0.997 GHz on 64 GSa/s, edge-time error peak to peak
  by polynomial crossing: **15.58 ps before; after, 3.0e-5 ps with a 9.6-sample edge**
  (`tr_frac=0.15`, and `cmos` at the same edge), **1.1e-2 ps for `square` at its default
  3.2-sample edge** (`tr_frac=0.05`) **and 5.2e-2 ps for `cmos` at a 2.2-sample edge**
  (`tr_s=35e-12`): the shorter the edge, the less the interpolator has to work with; a duty of 0.3
  of a 256-sample period reads 0.299999999982 where it read 0.30078 (77 samples of 256). The residual is
  5.7e-6 sample for an edge of 8 samples and 5.6e-3 at the 2-sample floor.
  A record whose edges were all on samples *and* rightly placed is the same bits
  (`tests/test_edge_placement.py`); any other moves by up to one sample at each affected
  edge. Recorded in `compose.NUMERIC_CHANGES` as `carrier/square` and `carrier/cmos`, found
  wherever such a carrier sits in a recipe (a sub-chain, a `pass_fet` gate, a `net` driver),
  so replaying an affected pre-0.46.0 recipe says so and one holding only other carriers
  stays silent. Of the 31 examples, one printed figure moved: `sequence_and_switch.py`'s
  mean preamble level, `-0.00` to `+0.00`, a 25 MHz clock over whole periods now averaging
  to zero; 155 of their 482 renders move, in `clock_and_data`, `joins`, `multi_driver_bus`,
  `sequence_and_switch` and `unit_variation`, each holding one of the two carriers. `step`, `pulse`, `sawtooth`, and an `nrz`/`pam` carrier whose samples per UI is
  not a whole number still place edges on samples; see *Known gaps*.

### Changed

None of these moves a rendered sample.

- **`Signal.roles()` lists exactly what the record draws from.** Added, each only when the
  op's parameters make it draw: `timing/<i>`, `rx_noise/<i>`, `timebase/<i>`, `acquire/<i>`,
  `store/<i>`, `nl_noise/<i>`, `rin/<i>`, `shot/<i>`, `linewidth/<i>`, `ase/<i>`, `tia/<i>`,
  `carrier_noise/<i>`, `jitter/<i>` for `symbols` and `coded`, the jitter of a bare carrier
  spec in six ops and a `net` driver, and sub-chain roles. Removed, because nothing was
  drawn under them: `jitter/xtalk<i>` for an aggressor with no jitter, `jitter/<i>` for a
  carrier whose `jitter=` has no `rj`, `events/<i>` for events at stated positions, and a
  pinned (`{"from": recipe}`) record's toleranced parts. Every listed role moves samples when
  re-rolled and nothing is drawn under an unlisted one, checked against the generators over
  41 chains. Code comparing `roles()` against a stored list will differ.
- **`contrast()` says when a re-roll changed nothing.** A name that is no op's role is
  refused. A real role on an op whose parameters draw nothing from it (`jitter/0` with no
  `rj`) is accepted as before and now warns that the sibling is the record again.
- **`carrier` warns when its default 32 symbols disagree with the grid's baud.** `n_ui`
  still defaults to 32 stretched over the record. On a grid that states a `baud` those 32
  symbols do not come out at, one RuntimeWarning names both rates and the `n_ui` that
  agrees. Stating `n_ui` (any value) is silent.
- **A `net` driver's `{"ops": Signal}` warns when that Signal's seed or grid is dropped and
  differs from the record's.** The output is unchanged; the fix it names is
  `{"from": that_signal.recipe()}`.
- **A measure handed the wrong kind of record refuses by name.** `measure` and `eye` used to
  cast a complex record to float (keeping I, dropping Q); they now name `rf.evm`,
  `upconvert` and `x.real`. The envelope measures already refused a voltage.
- **A complex file read into a real chain is refused.** `capture()` used to keep the real part
  under a numpy warning. HDF5 export of a complex record is likewise refused by name.
- **A malformed `net` driver source is refused**: `{"ops": [...], <unread key>, "volts": ...}`
  used to ignore the extra key.
- **Three parameter shapes are now reserved in a recipe**: a dict with a list under `ops` or a
  dict under `from` is a sub-chain, and a dict holding `nom`, `tol_pct`, `tol_abs` or `corners`
  is a toleranced value. No existing op parameter used them.
- **`dataset()` stacks an I/Q record as complex64** (it was cast to float32, dropping the
  imaginary part under a numpy warning), and a builder that returns a `Channels` gives
  `{name: (n, L)}`; `unit_dataset` takes one too. `op_params(recipe, op, chain=)` reads a
  several-channel recipe.
- **Labels follow the samples through a rate change and a delay.** After `acquire`,
  `digitize(n_out=)`, `sample_clock(n_out=)` or `downconvert(fs_out_hz=)` an upstream event's
  `sample` and `width` are on the stored record (they used to stay on the synthesis grid and
  point past its end: 36 of 40 labels on a 4:1 acquisition), and one past a shortened record
  is dropped. `delay` moves upstream events by the delay where they land. Samples are
  unchanged, so no recipe's waveform moves.
- `stage_kinds()` reports `general` for the ops whose position is their meaning (`law`,
  `filter`, the joins, `use`, the I/Q impairments), and `canonical()` refuses a chain holding
  one rather than sort it.
- **Two chains that used to render are refused, replay included** (#92). An electrical op
  between `eo` and `photodetect` cast the field to float under a numpy warning; a field op
  (`fiber`, `edfa`, `optical_mpi`, `photodetect`) on a voltage with no `eo` treated the
  voltage as a field. Each now names the bridge it lacks. No shipped chain, example or test
  did either ([COMPATIBILITY.md](docs/COMPATIBILITY.md)).
- `summary()` marks the op that changes the domain (`-> optical`, `-> real`) and a chain that
  does not start as a voltage (`domain=iq`); it prints a short dict or list of numbers in
  full (a dwell spec, a pair of levels, a band's edges) where it printed keys or a count. A
  real chain's text is otherwise unchanged.
- A lead-in on an `iq` source is refused for its stated reason (the old message said the
  source was not one).

### How it stays exact

- **What moved is two entries in `NUMERIC_CHANGES`, and nothing else.** Recipe digest,
  samples, JSON replay, event list and a re-rolled sibling were compared with
  `tools/digest_compare.py` against `main` (0.45.0, 52e0d03). Of its 31 fixed chains, 23 are
  identical in every field; `pass_fet` differs in its samples (its gate is a `square`, see
  *Changed output*); seven differ only in the sibling, because the first role `roles()` lists
  for them is one that was drawn and unlisted before. Of the 238 example renders the two
  trees have in common, 2 differ, both in `multi_driver_bus.py`, both holding `cmos`
  carriers. Before the `square` / `cmos` change the same comparison, on the tool's then 28
  chains, was 28 of 28 and 238 of 238. `tests/test_byte_identity.py` pins the default path in the tree.
- **Each effect is asserted against an analytic result or an independent method**, in
  `wfmsynth.validate` (485 checks) and the test suite (2891 tests): a sub-chain against
  the same chain rendered alone and coupled by hand (equal); Bessel-4 `filter` against the
  scope front end (equal); a product `combine` against `modulate` OOK (equal); the delay's
  error against the exact trigonometric interpolant; trigger instants against an independent
  edge list; toleranced draws against their stated distributions over 10 000 seeds; the mixers
  against a closed-form tone; each I/Q impairment against its closed form; EVM in noise against
  the analytic value. The numbers are printed by `python -m wfmsynth.validate`.
- **What a stage cannot do honestly it refuses, by name, with the alternative.** Nothing is
  resampled to fit, truncated, clipped on export, or stripped of an imaginary part.

- **The pieces of #88–#94 were compared the same way**, each against the branch tip it
  started from: 31 fixed chains and every example render in common, 0 differ. Their effects
  are in `wfmsynth.validate` under their issue numbers, and `tests/test_pieces_meet.py`
  holds the checks that only exist once they share a tree (an OFDM equaliser reading back a
  composed multipath channel, carrier phase included; an I/Q multitone through `multipath`
  at every tone; a contact bounce gating an envelope and a field).

- **A review round before release** (four independent reviewers, every finding reproduced
  first) fixed what the new surface got wrong, with the test that would have caught each:
  the general stages after an `acquire` or a decimating `downconvert` ran at the synthesis
  rate (a 2 GHz corner landed at 0.5 GHz); `phase_noise` under-delivered a profile whose
  close-in power fell between FFT bins (0.0019 rad for a stated 0.0263); defects in a
  `sequence` segment, a selected `switch` input or a `use`d chain were left out of `mask()`;
  `contrast()` with a mistyped role returned the record unchanged; `iq()` recorded and
  ignored a second symbol source.
- **A second review round** (three reviewers, every finding reproduced first) fixed, each
  with its test:
  - `resonator`: a mode hop BETWEEN two samples was moved to a sub-step edge, so every sample
    after it had the wrong ring phase (field error 0.14 to 1.7 against the closed form); the
    sample holding a hop is now cut at the hop (6e-14). An input ramping across a sample was
    held at each sub-step's midpoint; it is now integrated as the line it is (1.1e-2 ->
    4e-16 at kappa*dt = 6). A hop that leaves a ring still holding light half the sample
    rate or more from the laser is refused, as a fast sweep is.
  - events that state an extent: a `state` interval ended by a `sequence` segment, a
    `trigger` / `acquire` / `sample_clock(n_out=)` window or a `delay` kept its whole
    `dwell_s` and `complete: True`, so `states.intervals` reported it past the cut; it now
    stops there, incomplete (`events.cut`). An `ofdm_symbol` event carried `t_useful_s`, an
    absolute time nothing re-timed (wrong in every later `sequence` segment and after a
    `delay`): it is gone, the useful part starting at `t_s + cp_s`; a symbol a segment cuts
    has `complete: False`, so the complete events are `transmitted()`'s rows. Under a
    `delay(by=<chain>)` that varies only an event's start was re-timed: its END now is too,
    by the delay at its own arrival, so `dwell_s` (and an `ofdm_symbol`'s `width` and `cp_s`)
    are the delayed record's (the ends were up to 287 us from the samples' edges under a
    delay swinging 2 ms at 50 Hz; now under 1 us).
  - `ofdm(cp=)` truncated a non-whole number (`cp=0.8e-6`, seconds, rendered with no prefix
    at all): refused, with the count it comes to.
  - `law("saleh" | "rapp")` after `Signal().use(...)` or in a shared stage was refused at
    build time as "a real voltage"; a chain that cannot yet know its record is now checked
    when it renders, as `gain` was.
  - `recipes.multipath` / `mzi` / `stray_reflection` accepted a transmitter that drew AFTER
    a `use` (each path then got independent noise): refused.
  - `combine({"from": recipe})` of a field about another carrier was accepted (two lasers
    1.2 THz apart summed coherently): refused, as the sub-chain form is.
  - `states.intervals(at=)` returned `[]` for a namespace nothing is at: refused, listing
    those that are. `measure.resonances(level=)` returned NaN widths when the record never
    came HALF WAY up to the stated baseline (over 3 dB of loss): refused, naming
    `transmission(power=)`. A smaller loss cannot be told from a ring of low finesse and
    is read as stated: 1 dB left in with `level=1.0` gives the width 30 % wide.
  - `measure.swept_map` misread the part fringe at a record's end where the sweep rate was
    changing (up to 0.7 fringe, and every sample when it was the first): it is now read from
    the fringe's own crest and trough (0.02 at worst over the same sweeps). A crest is the
    extreme of the samples averaged over an eighth of a turn, not the highest sample, which
    detector noise places short of a crest the record never reached: with 1 % and 2 % noise
    on a rippled, bowed sweep the ends read to 0.023 and 0.039 fringe (the highest sample
    put them 0.27 out in 5 and 17 draws of 60).
  - `rf.evm(..., ofdm=)` gained `symbols=` (a mask over OFDM symbols) and `block=` (the
    fits refitted per block, for a fade); without `ofdm=` on an OFDM record it names the
    argument. `ofdm.plan()` returns `ref_power`. `recipes.ripple` and `pickup` take
    `phase_rad=`. A `sequence` segment may carry a `label`, which `bounce` uses so an
    off-grid duration is refused in `bounce`'s own argument names.
  - stated, not changed: `recipes.ring`'s Q and extinction are one mode's, read to about
    `(pi*coupling_hz/fsr_hz)**2` with `fsr_nm` (measured at a finesse of 10); a causal
    `lossy` on an envelope far above the sample rate has almost no carrier phase and a
    0.35 ns advance, because its minimum phase is built up to the band's own top
    (`wfmsynth.baseband`); the dwell order of a burst schedule in the docs was backwards.
  Against the branch tip before it (8a26c2d), `tools/digest_compare.py`: 31 of 31 fixed
  chains identical. Of the 588 example renders in common, the 15 of
  `swept_ring.py` differ, by 3e-16 on a slow sweep and 1.1e-13 on a fast one (the
  resonator's exact step for a ramping input; inside the Tier 2 band), and no other.
- **The rulings on what that round left open** are in BACKLOG.md (#75–#87, *Ruled*), each
  with its test: sub-sample edges for `square` / `cmos`, the `n_ui` warning, exact
  `roles()`, the domain following the source, `copy()`, the decided-reference EVM's
  acquisition, the three renamed parameters, measures refusing the wrong domain, and the
  `net` source warning.

### Known gaps

- The named link and instrument ops (`scope`, `probe`, `lossy`, `events`, ...) placed AFTER
  an `acquire` still read the synthesis grid's rate, as on `main`: `scope(bw_hz=2e9)` after
  an 80 -> 20 GSa/s acquisition has its -3 dB point at 0.5 GHz. Put them before the
  acquisition; `filter` after it runs at the stored rate.

- `square` and `cmos` now place edges between samples (*Changed output*). The same grid
  placement remains, unchanged from `main`, in three other places: an `nrz` / `pam<N>` /
  `symbols` carrier whose samples per UI is not a whole number (measured: a clock pattern
  of 1000 UI on 32768 samples, 15.5 ps peak to peak at 64 GSa/s, against under 1e-9 ps at
  1024 UI); `carrier(jitter=)`, which displaces edges by whole samples (BACKLOG, U-13);
  and `step` / `pulse` / `sawtooth`, whose single edge or flyback is rounded to a sample.
  Keep samples per UI whole, and take jitter from `timing`.
- Photonics is one carrier per record, through port only, and checked against closed forms
  and an independent integrator, not measured sweeps. Deferred with reasons in BACKLOG.md
  (#88–#94): Bragg gratings, polarisation-dependent loss, a resonator's drop port, coherent
  detection, a carrier through `sequence` / `switch`.
- A `state` event in force when a `trigger` window or a restarting `sequence` segment begins
  is dropped, not clipped to the window. (One a window or a segment ENDS is cut there.)
- On numpy < 2 a COMPLEX record does not replay to the byte: numpy 1.x's complex product
  rounds differently with the array's alignment (measured on 1.26.4, arm64: three byte
  results for one `x * g`), so 13 to 18 tests that assert byte equality of an I/Q or optical
  replay fail per run there. Real records are unaffected; numpy >= 2.0.2 is byte-stable. The
  declared floor is still `numpy>=1.21`; raising it is undecided
  ([COMPATIBILITY.md](docs/COMPATIBILITY.md), Tier 2).
- The work of #88–#94 was gated on Python 3.14 (macOS). `validate` also passes on 3.9.6 with
  numpy 2.0.2 and scipy 1.13.1, all 485 checks; the test suite was not re-run on 3.9, 3.12,
  Linux or Windows after it.
- Python 3.9, the declared floor, was run on one machine only (macOS arm64, CPython 3.9.6),
  not in CI. With numpy 2.0.2 and scipy 1.13.1: 2470 passed, 2 failed, both
  `test_the_demo_does_not_warn` (`i2c_bus.py`, `inductive_handover.py`), on spurious
  `matmul` floating-point warnings that numpy 2.0 raises with Accelerate inside `net.py`,
  which this release does not touch; `validate` passes. With the floor's numpy 1.21.6 and
  scipy 1.7.3 the suite cannot complete on that machine (numpy's own `linalg.solve` dies
  with a bus error); the 13 test files this release's rulings added or re-pinned, 272
  tests, pass there. No 3.10+ syntax or API was found in the diff against `main`.

## 0.45.0 — 2026-11-04

A shared net gains inductance, so a hand-over rings instead of merely settling — the last place
a synthetic record differed from a measured one by more than noise. A net with **no** inductor
does not take the new code path at all and renders bit for bit as 0.44.0 did.

### Added

- **An inductor element in the shared net** (#74). An R-C net can only settle toward a level.
  What a scope shows when a fast driver takes a bus, changes it or lets go of it is overshoot
  and a ring, at a frequency set by the package and trace inductance against the node
  capacitance and decaying at a rate set by the termination.

  - `{"id": "lp", "kind": "l", "a": "pad", "b": "bus", "henry": 8e-9, "ohm": 0.3}` — `ohm` is
    the series resistance of the *same piece of wire* (absent means ideal), not a parallel
    resistor. It is one element with two numbers because the node *between* a series R and a
    series L has no capacitance, so it is not a state and this tier refuses it; the refusal now
    says that by name.
  - `solve(..., probes=["bus", "i(lp)"])` reads an inductor's branch current in **amps** — the
    state the solve now carries across every switch, and the only way to see it.
    `v0={"bus": 0.0, "i(lp)": 0.0}` sets it; a current `v0` does not name starts at zero, which
    is at rest. `v0` must now name every node, and an unknown `v0` key is refused.
  - `net.line_ladder(a, b, z0_ohm, delay_s, tr_s)` and `net.line_sections(delay_s, tr_s)` — a
    line between two nodes as a lumped L-C ladder, which is the one approximation in the module.
    The section count comes from a measured rule, `ceil(6 * (delay/tr) ** 1.5)`, holding the
    ladder's own error between 1.1 % and 1.9 % of the swing for lines from half a rise time to
    ten rise times long. A smaller explicit `sections=` warns (about 5 % of the swing at half
    the rule's count, 10-16 % at a quarter); past 256 sections it is refused and names
    `lossy` / `sparam` / `cascade` / `reflect`, which represent a line *as* a line at any length.
  - `examples/inductive_handover.py` — one 3.3 V transceiver, a 7 ohm output stage, 8 nH of
    package, 20 pF of bus. The bus reaches +4.396 V and -1.096 V, 796 mV past both of the
    receiver's absolute maxima, while the R-C model of the identical board sits 412 mV *inside*
    both and is wrong about every edge; the two records differ by up to 1.758 V, 57 % of the
    swing. Letting go while carrying 15.4 mA kicks the pad 0.980 V in 195 ps against the
    1.020 V that current into the loop's own 66 ohm accounts for, ringing at a measured
    1339 MHz against 1320 MHz closed form. 12 ohm of series damping is the whole fix, for
    131 ps of fall time.
  - `tests/test_net_inductors.py` (43 tests) and seven new hard checks in `wfmsynth.validate`.

- **A warning for a ring the grid cannot show**, separate from the existing one for a decay it
  cannot show, because the two fail differently: a fast decay is simply absent from the record,
  but a fast ring **aliases** and comes back at a frequency that is not the circuit's and looks
  entirely plausible. It fires below 8 samples per ring period — the library's own k >= 8
  sizing rule — and says explicitly when the record's ring is past Nyquist and therefore an
  alias. Once per distinct frequency, not once per interval.

### Changed

- **A schedule event targeting a capacitor or an inductor is refused**, as is a `state` key on
  either in the description. Both are *state*: a capacitor's charge and an inductor's current
  are what the solve carries across a switch, so neither can be taken out of the stamp without
  a discontinuity. Previously an event on a capacitor rendered as though it had not been
  written — a silent no-op. Refusals only: no rendered samples move, so there is no
  `compose.NUMERIC_CHANGES` entry.
- The unsupported-element-kind message now says `'r', 'c' and 'l'`.

### How it stays exact

An inductor's branch current joins the state vector, and the resulting system matrix is no
longer symmetric — that skew coupling between charge and flux is exactly what makes a mode
oscillate rather than decay — so the R-C tier's symmetric eigensolver does not apply. Nor do
eigenvectors in general: at **exactly** critical damping (`R = 2*sqrt(L/C)`, a damping network
somebody designs on purpose) the matrix is defective, its eigenvector matrix is singular, and a
modal solve returns 368 mV of nonsense on a 1 V step. So the inductive path takes the exact
zero-order-hold pair for one sample out of one matrix exponential and back-substitutes it
through a **Schur** form, whose basis is unitary and therefore well conditioned at every
damping, one `lfilter` call per state.

Measured against the textbook series-RLC step response: **3.5e-14 V** underdamped, **8.2e-14**
critically damped, **4.3e-14** overdamped, on a 1 V step; and 2.0e-10 V against an independent
RK45 integration of a switched two-node board, which is the integrator's own tolerance rather
than the solve's. The one cost is a rounding floor the R-C path
does not have, because the answer is accumulated sample by sample rather than evaluated at
`k dt`: 3.5e-14 V for any mode that decays, and **8.4e-12 V** over 262144 samples of a lossless
tank, which is the worst case there is. A net with no inductors is verified bit-identical to
0.44.0 by digest, and the test suite pins it by refusing to let an R-C net enter the inductive
code at all.

## 0.44.0 — 2026-11-03

A driver on a shared net may now render its own signal chain as its output level. Nothing
existing changes: every path without a driver source is bit-identical, and a source that never
moves is routed back to the closed form so it equals a declared `volts` bit-for-bit.

### Added

- **Drivers whose output follows their own signal chain** (#72 follow-on). Tier 1 let a
  driver's *state* follow a carrier; its output was still a fixed Thevenin level. On a net
  shared by several parties each one usually has its own supply, output impedance, edge rate
  and timing impairments, and adding their contributions by hand outside the solve is where a
  level held by a driver that has let go gets carried into the next owner's baseline.

  - `net.solve(..., sources={driver_id: volts_per_sample})` — a driver's Thevenin level as a
    waveform on the record's own grid.
  - `Signal.net(drivers=[{..., "source": <spec>}])` — the same thing as a nested spec, so it
    survives into the recipe. A body is exactly one of `from="chain"` (the waveform arriving
    at this op), `ops=` (its own op list, or a `Signal` to take it from), or a carrier spec by
    `kind=` — the idiom `crosstalk(aggressor=)` already uses.
  - `volts=` is **required** on a source: `True` when the chain is already volts, or
    `[lo, hi]` to map the ±1 carrier convention onto those two volts. A net is in volts and a
    carrier is not, so a source passed through unconverted would be a driver a thousand times
    too weak and the solve would be perfectly correct about it. The map is affine, so a
    `tx_ffe`'s amplitude or an overshoot survives rather than being renormalised away.
  - A net all of whose drivers carry their own source may be the **first op** in a chain:
    nothing has to arrive at it, and its length comes from the grid.
  - Each driver's randomness is namespaced by its id, so two drivers written from one spec are
    two devices rather than one written twice, and `Signal(seed=)` still moves the whole
    record.

  Still exact. Only `b` moves between events, so the eigenbasis is computed once per interval
  as before and the sampled source is carried through it by the zero-order-hold recurrence
  `physics.open_drain_line` already applies to its target — no step size, no integration
  error. Checked against an independent RK45 integrator written from the node equations
  (1e-11 V on a 2.2 V swing), against the scalar closed form for the one-driver case (5e-15 V),
  and for the reduction by `np.array_equal`. A time-varying *conductance* is still refused by
  name (`gate`, `relax`): that is the one thing the closed form cannot absorb.

- **`examples/multi_driver_bus.py`** — three transceivers of different supplies, output
  impedances and edge rates taking turns on one terminated backplane bus. Every measured level
  and edge lands on its closed form to ~0.2 %; the weakest part misses the receiver's V_IH by
  346 mV because a terminated bus penalises it twice; and the bus returns to its bias exactly
  at every hand-over, against the −1.765 V of drift a by-hand sum would have accumulated.

### Changed

- **A misspelled key on a `net` element or driver description is now refused, naming it.** A
  description is a dict, so `{"id": "d", "node": "bus", "ohms": 60}` was previously accepted
  and did nothing — the record rendered, the digest was stable, and the driver had whatever
  strength the schedule happened to set. The accepted set is per *kind*, because the traps
  differ: a capacitor is never switched out of the stamp (its charge is the state the solve
  carries across every switch), so a `state` on one was honoured by nobody. An element with no
  `kind`, a driver with no `node`, and an element of an unsupported kind are refused the same
  way — the last of those from the description rather than from the stamp, since an inductor
  parked `state="open"` previously rendered as though it were not there. Keys beginning with
  an underscore are annotation and are still ignored, as in a recipe op. These are refusals,
  not output changes; no rendered samples move.
- `examples/i2c_bus.py` no longer puts a decorative `carrier("dc")` in front of its net, now
  that a net can be the first op. Its output is unchanged.

## 0.43.0 — 2026-10-20

Two additions that belong together: a shared net (#72) and the layer that writes its schedule
(#73). One entry under **Changed output**, and it can only reach a recipe rendered from `main`
after 2026-10-19 — `net` itself has never been in a tagged release.

### Added

- **`wfmsynth.scenario` — a scenario layer over the shared net: holds, agents, guarded
  reactions** (#73). `net`'s schedule of `{"t_s", "id", "state"}` is the right interface for a
  solver and the wrong one to write by hand. This is the layer between, and it adds no
  electrical physics — it emits the events a careful person would have written.

  - `Hold(who, t0, t1, **tag)` — an agent on the net over an interval, with free-form tags.
  - `Agent(during, after, lead, lag, then, unless)` — a party configured by *behaviour*:
    its turn-on and idle rows, how far each precedes or follows the logic edge, a release
    reaction, and whose holds suppress that reaction because the net changed hands rather
    than being released.
  - `schedule(holds, agents)` — pure, deterministic, JSON-able.
  - `merge_holds` — unions an agent's touching holds, so consecutive identical bits are one
    hold rather than a release and re-acquire at every bit boundary.
  - `hold_windows` — the same hold list as `events.label_windows` input, so the thing that
    produced the waveform is the thing that labels it and a fault injected by editing the
    holds labels itself.
  - `Signal.net(holds=, agents=)` takes the authored pair and compiles it; the recipe stores
    *that* rather than the events it expands to.

  Two semantics worth knowing. A release reaction is all-or-nothing, decided over its whole
  window: half of a settling sequence would park the thing it was settling in its intermediate
  state permanently. And the `rest` row seeded at t=0 is the agent *settled* — `after` with the
  reaction applied — not `after` alone, which for a switch that stays closed and then opens is
  the opposite state.

- **Protocol layers are caller code, and there is now a worked one.**
  [`examples/i2c_bus.py`](examples/i2c_bus.py) writes a byte to an I2C target across a
  3.3 V/1.8 V MOSFET level translator: a ~40-line encoder that knows I2C and nothing about
  volts, over agents that know about output stages and nothing about I2C. It finds a real
  result — the loaded 1.8 V side rises in 1209 ns against the 300 ns Fast-mode limit, and
  R<sub>ds(on)</sub> at V<sub>gs</sub> = 1.8 V eats 90 % of the 3.3 V side's V<sub>OL</sub>
  budget. No `i2c()` function was added and none will be: which bits a standard puts on a wire
  has a revision date behind it, the same boundary the pattern registry draws.

- **`net` schedule events take `g` and `src`.** `g` is a conductance in siemens (zero *is* an
  open, so strength and state cannot disagree) and `src` a driver's Thevenin level in volts, so
  a weaker driver taking over from a stronger one is two drivers at two strengths. `gate` (a
  turn-on slew) and `relax` (settling toward another strength) are refused *by name* with what
  to do instead, rather than ignored: both are elements that vary *during* an interval, which
  is the one assumption the closed form rests on.

- **`Signal.net(...)` and `wfmsynth.net` — a shared net as one channel stage** (#72, tier 1).
  Several drivers hand over through hi-Z, terminations and pull-ups switch on a schedule, and a
  weaker driver can hold a level against a stronger one. None of that is expressible as the
  series 2-port cascade every other channel op is. Nodes, rails, lumped R and C, Thevenin
  drivers with a scheduled output conductance, a schedule of absolute-time events against stable
  ids, and several probes off one solve — all plain JSON, so a protocol, traffic or fault
  generator can produce a schedule without the op knowing about it.

  Exact rather than integrated: between events the network is constant, so a symmetric
  generalised eigendecomposition gives the interval in closed form. No step size, no guard time,
  no periodicity assumption, and continuity across a switch is the capacitor voltages carried
  forward. `physics.open_drain_line` is reproduced to **2.2e-15** (it is the one-node case of the
  same recurrence), and a two-node coupled net with a switched termination agrees with an
  independently-written RK45 integrator to **5.9e-12**.

  Lumped R and C only in this tier; transmission lines, S-parameter blocks, sub-nets, nonlinear
  laws and event emission are deferred, with the order and reasoning in `BACKLOG.md`.

### Changed output

- **`net` placed a scheduled event one sample late, depending on float representation.**
  `2.5e-6 * 1e8` is `250.00000000000003`, so a generated schedule (`t0 + i*dt`, the normal
  case) had edges landing on either side of a sample boundary according to how the generator
  accumulated. `net.sample_at` now snaps a time within a billionth of a sample onto the
  boundary, and `scenario.hold_windows` uses the same rule — so a hold's label window and the
  instant the solver switched are the same sample rather than neighbours.

  Recorded in `compose.NUMERIC_CHANGES`, so replaying an affected recipe says so. This is
  listed under *Changed output* rather than *Fixed* because it moves samples, which is what a
  consumer scanning this file is looking for — but the blast radius is small: `net` has never
  been in a tagged release, so only a recipe rendered from `main` between 2026-10-19 and the
  next release can carry it, and the shift is one sample at the odd switching instant.

## 0.42.0 — 2026-09-25

Nothing here changes rendered output: every entry is additive, documentation, a test, or a new
warning. Verified by rendering three chains (a full instrument chain, an acquisition, and a
de_emphasis chain) at `v0.41.0` and at HEAD and comparing sha256 digests — all three identical.

**Cut because 0.41.0 had become two different things.** The version was bumped to `0.41.0` on
2026-09-22, *before* the performance work began, so everything from the benchmark suite through
`op_params` also reported `0.41.0`. A checkout taken from `main` on 22–23 September and the
`v0.41.0` tag are materially different code under one label — which is exactly the
one-label-two-meanings defect this release's own `NUMERIC_CHANGES` work exists to prevent, and
it was introduced by not bumping the version once the work landed. See
[`docs/COMPATIBILITY.md`](docs/COMPATIBILITY.md) for how to tell which one you have.

### Added

- [`docs/COOKBOOK.md`](docs/COOKBOOK.md) — the 31 per-feature recipes that were making the
  README a reference manual (867 lines, 47 sections). The README is now 442 lines of
  orientation; the cookbook is what you come back to.
- [`docs/COMPATIBILITY.md`](docs/COMPATIBILITY.md) — what the library promises not to break,
  across its three surfaces: the API, rendered output, and stored recipes. Written because the
  project broke the third one deliberately in 0.41.0 (#53) and had no vocabulary for what it had
  done.

### Fixed

- **A recipe written before an op's numerics changed now says so on replay.** `recipe()` has
  always recorded `wfmsynth_version`; `from_recipe()` never read it, making the field write-only
  provenance. That was not theoretical: 0.41.0 (#53) inverted `de_emphasis_taps`' sign, so a
  stored `de_emphasis(db=3.5)` rendered as de-emphasis under 0.40.0 and as PRE-emphasis
  afterwards — inverted transmitter shaping, replayed without complaint. `compose.NUMERIC_CHANGES`
  is the table that reads the version, with #53 as its first entry.
- **All twelve examples now satisfy the library's own `k >= 8` sizing rule.** README.md's first
  example warned when run verbatim (0.6 samples across an edge), and eight demos were
  undersampled at 2.3–10 samples/UI. `ground_truth.py` was understating the eye divergence it
  exists to demonstrate by 3x (0.034 → 0.103) and printing a sampling phase of 0.00 that was a
  resolution artifact. Three examples also hardcoded a samples-per-UI divisor that silently
  contradicted their own grid; all now derive it. A gate keeps it that way.

### Security

- `capture.load_values` states `allow_pickle=False` explicitly and reports a pickled `.npy`/
  `.npz` as a named refusal. This was never open — numpy has defaulted to `False` since
  1.16.3 — but a capture file is the one input this library reads that a user may not have
  produced, so the restriction is now written down and tested with a real code-execution
  payload rather than inherited from a default that could change.

### Changed

- Five error messages that stated a rule without the value that broke it now name the value,
  and the two identical `node must be 'load' or 'source'` messages say which function raised.
- `capture`, `hdf5` and `resample` are in `__all__`; they were importable as `ws.<name>` but
  `from wfmsynth import *` missed them.
- `Signal.reflect`'s docstring now says that `gamma_s` means **source**, not seconds — the
  suffix means seconds everywhere else in the library, and it sits beside a `td_ps` that is
  genuinely a time.

### Added

- `op_params(recipe, op)` — read a knob back out of a recipe, which is where a `dataset()`
  label comes from. `dataset()` deliberately returns no `y`: the sampled values are already
  baked into the ops, so a label read from the recipe cannot drift out of alignment with the
  record the recipe reproduces. Raises when an op appears more than once, because `[...][0]`
  would silently label every record from the wrong stage.
- `CONTRIBUTING.md`, `SECURITY.md`, `CODE_OF_CONDUCT.md`, issue and PR templates, and a
  dependabot config for GitHub Actions.

## 0.41.0 — 2026-09-24

The long-record performance pass, a batch of S-parameter capabilities, and eleven filed defects.

### Performance

End-to-end at 4.2 M samples, output bit-exact or within 1.4e-15 unless noted:

| Workload | Before | After |
|---|---|---|
| Instrument chain | 0.449 s / 614 MB | **0.285 s / 382 MB** |
| Jitter + supply chain | 2.626 s / 634 MB | **0.959 s / 445 MB** |
| Dataset, 32 records | 0.439 s / 203 MB | **0.224 s / 165 MB** |

- `physics.lfsr` computes a long sequence by jumping over GF(2) rather than stepping per bit:
  738 ms → 18 ms at 4 M bits, uniform across PRBS7–31, **bit-exact**. `phase` is now a matrix
  power, so a phase offset of 1e9 returns in under a millisecond instead of being untenable.
- `resample.resample_at` factors its kernel weights into a polyphase table and takes its
  normaliser from a phase table: ~3.4× faster, agreeing to ~1e-15. This is the hot path for
  jitter, DCD, intra-pair skew, the sampling clock and acquisition.
- `physics._min_phase_H` no longer builds three record-length temporaries, and can return just
  the rfft half that every filtering caller wanted.
- `compose.dataset()` streams each record into the output array instead of collecting them all
  first, which held the whole set twice: peak 3.0× → 1.2× the array.
- `tools/benchmark.py` plus a CI gate on peak memory and the **scaling ratio** `t(2n)/t(n)`,
  which divides the runner's speed out and so catches a complexity regression without depending
  on machine load.

### Added

- `Signal.stored_grid(x)` — the `Grid` describing the record `waveform()` returned, which is not
  `Signal.grid` after an `acquire` or a `digitize(n_out=)`. Handing the synthesis grid to a
  measurement was silently 4.9 % wrong on an 80 → 40 GSa/s acquisition; this removes the footgun
  rather than warning about it, and raises if it cannot account for the chain's length.
- `physics.response_cache()` — reuse a channel's minimum-phase response across a batch of
  records. Bit-exact, opt-in (an entry costs about one record of storage), and `dataset()`
  already wraps its own loop in one. 1.28× on a 12-record batch.
- `physics.apply_transfer(method=)` — `"overlap"`/`"auto"` compute the same linear convolution in
  blocks, holding ~1.1× the record instead of ~5.0×. **Not the default**: see *Changed output*.
- `sparam.read_mdif` / `MdifSweep` — read an MDIF (`.mdf`) sweep of S-matrices over one or more
  outer variables; `.select(**kwargs)` picks a point for `sparam_channel`. (#62)
- `sparam.stub()` and `sparam.measured(left=, right=)` — an open/short transmission-line stub,
  and a measured 4-port block taken as a differential cascade section via its mixed-mode SDD
  block. (#59, #60)
- `sparam.check_response()` — passivity and precursor-energy diagnostics for a measured
  response, reported rather than enforced. (#61)
- `sparam.renormalize_s()`, and Touchstone reference-impedance handling throughout. (#58)
- `physics.warn_if_awkward_length()` — names a record length that costs a circular stage up to
  12× (a large prime factor). Silent for every round length: `2**k`, `10**k`, and the
  `40,000,000 = 2**9 * 5**7` of a deep capture.

### Fixed

- `measure.align_symbols` locked onto noise on an inverted record. (#52)
- `physics.de_emphasis_taps`' sign was inverted from how every specification quotes it; negative
  dB now means de-emphasis, e.g. PCIe's "−3.5 dB". (#53)
- `__version__` is derived from `pyproject.toml` instead of a second literal that had already
  drifted from it. (#54)
- `crosstalk`/`drift` silently took a fallback branch on an unrecognised `kind`/`shape`; both
  now raise. (#55)
- `Signal.sparam` shredded a mixed-mode `ports` pairing given as a string, and dropped `z0`
  entirely. (#57, #63)
- `examples/confounder_sweep.py` crashed on the library's own documented gotcha (`eye_height`
  requires `levels`). Every example now runs in CI, so this class of rot cannot return.
- `stream.stream_blocks` mishandled a single-tap filter (`-(nh-1)` is `-0`, and `tail[-0:]` is
  the whole array).

### Changed output

Read this section before upgrading a pipeline that has already generated data. See
[`docs/COMPATIBILITY.md`](docs/COMPATIBILITY.md) for what each tier of change requires.

- **BREAKING, by default: `de_emphasis` changed sign (#53).** A stored recipe with a POSITIVE
  `db` asked for de-emphasis under 0.40.0 and renders as PRE-emphasis from 0.41.0 — inverted
  transmitter shaping on an existing record. Negate the `db` to preserve the original intent.
  This is the one change in the release that alters a default render, and it is why
  `compose.NUMERIC_CHANGES` and the version check on `from_recipe` now exist: replaying such a
  recipe warns and names the fix instead of quietly producing a different record.
- Everything else is bit-exact or within ~1.4e-15 — inside the byte-identity band — except:
- `apply_transfer(method="overlap")`, which is **opt-in and recorded in the recipe**. It differs
  from the transform by 7.7e-5 of peak-to-peak for a channel stage, 2.5e-4 through a full chain,
  and a whole 8-bit code on 1.16 % of samples once a converter sees it. A recipe rendered on one
  path replays on that path.

### Known limitations

- `instrument.probe_loading(causal=True)` still applies its pole circularly. Making it linear
  was implemented, measured and reverted: it costs 0.05 dB and 1.4° across the whole record to
  fix a wrap confined to ~5 time constants at its head, because an analog pole does not vanish
  at Nyquist and its sampled impulse response has a 1/k tail. See `BACKLOG.md` #58.

## 0.40.0 — 2026-09-22

Seventy-six commits. The largest single release, spanning the analog/instrument extension and
most of the acquisition path.

- Analog, CMOS/PWM and captured-from-disk sources, so a chain is not only a serial link:
  `step`/`pulse`/`exp`/`chirp_sweep`/`two_tone`/`analog_noise`, unipolar `cmos`, and
  `Signal.capture()` reading `.npy`/`.npz`/`.csv` with a sha256 over sample values.
- A fuller instrument pack: probe compensation, ground-lead inductance, termination, AC coupling
  and overload recovery; `burst`, `pass_fet`, `modulate` (AM/ASK/OOK/FM/FSK/PM); an open-drain
  wired-AND second sink.
- Bandlimited sub-sample displacement (`resample`), so jitter, DCD, intra-pair skew and a
  free-running sample clock stop being rounded to whole samples.
- HDF5 export shaped for a bench instrument, a pattern registry, line/level coding ops, and an
  installable agent skill under `.claude/skills/`.
- `compose` refuses a parameter an op does not read, instead of silently recording it.

## 0.39.0 — 2026-09-03

- **Relicensed MIT → 0BSD** (Zero-Clause BSD).
- Receiver-side equalisation as first-class chain ops: FFE and DFE.
- PRBS31Q PAM4 pattern; causal (minimum-phase) lossy channel fixed on odd-length records.

## 0.1.0 — 2026-08-25

Initial public shape: the composable `Signal` op-chain, physics-grounded channel/reflection/
crosstalk/jitter primitives, recipes with content digests, the `wfmsynth.validate` physics gate,
and two-rate acquisition.
