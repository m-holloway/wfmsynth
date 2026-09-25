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
