# Contributing to wfmsynth

This library generates ground-truth training data. A record that is *plausible but wrong* is
worse than one that is obviously broken, because nothing downstream will notice. Most of what
follows exists to make wrongness loud.

Read [`BACKLOG.md`](BACKLOG.md) for what is currently open, and
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for why the library is shaped as a chain of small
ops.

## The gates

Run all four before you open a PR. CI runs them too, but locally they take under a minute.

```bash
python -m pip install -e ".[test]"

python -m pytest -q                                   # ~1250 tests
python -m wfmsynth.validate                           # hard physics-property assertions
python tools/benchmark.py --quick --check             # peak memory and complexity
python examples/quickstart.py                         # the five-minute path still works
```

`wfmsynth.validate` is not a test suite — it is a set of assertions that a claimed *physical*
property actually holds in the rendered samples (a filter's −3 dB point really is where it was
asked for; a quantisation floor really is `q²/12`). If you change physics, expect to change it,
and say in the PR what moved and why.

`tools/benchmark.py --check` gates **peak memory** and the **scaling ratio** `t(2n)/t(n)`, not
wall time — the ratio divides the machine's speed out, so it catches an `O(n log n)` stage
turning `O(n²)` without being flaky on a loaded runner. If you make something legitimately
faster or leaner, re-record the floor so a regression back to the old behaviour is caught:

```bash
python tools/benchmark.py --quick --baseline          # --quick is the mode CI checks
```

## Definition of done

From `BACKLOG.md`, and applied to every change:

1. public behaviour and physical assumptions are documented;
2. a hard assertion in `wfmsynth.validate` checks the claimed physical property;
3. tests cover API behaviour and compatibility;
4. provenance records any new randomised or approximated value; and
5. **existing defaults stay bit-identical unless a deliberate versioned change is stated.**

## Changing rendered output

Point 5 is the one that bites. Records are reproduced from recipes, sometimes long after they
were generated, so output is a compatibility surface.

- `tests/test_byte_identity.py` pins the default path to a **tolerance** (~3.8e-13), not to
  bytes, so ordinary floating-point reassociation is absorbed and a behaviour change is not.
- If your change moves output **within** that band, say so in the commit with the measured
  number. Do not say "no change" — say "1.4e-15, six ULP, from the polyphase resampler".
- If it moves output **beyond** that band, that is a deliberate versioned change. Either put it
  behind a parameter that is recorded in the recipe (see `apply_transfer(method=)`, which stays
  opt-in precisely because it differs by 7.7e-5), or re-pin the references *explicitly* in its
  own commit with the reason.

Never re-pin a reference to make a red test green without understanding why it moved.

## Adding or changing an op

- Every op must appear in `_EXEC`, in `OP_KIND`, and in **exactly one** lead-in bucket
  (`_LEAD_LTI` / `_LEAD_SKIP` / `_LEAD_ANALYTIC` / `_LEAD_REJECT`). A gate fails if an op is
  unclassified, so a new op cannot default into the safe-looking pile.
- After adding or renaming a parameter, regenerate the accepted-key table:

  ```bash
  python tools/derive_op_keys.py > wfmsynth/opkeys.py
  ```

  `tests/test_op_keys.py` re-derives it and fails if the committed table drifted, so forgetting
  this is a test failure rather than someone's script silently ignoring a parameter.
- An op must **refuse** a parameter it does not read. A knob that is accepted and does nothing
  is recorded in the recipe and changes nothing — the exact silent-wrong-answer shape this
  library is built to avoid.
- The knob-sweep gate
  (`tests/test_composition.py::test_the_knob_sweep_covers_every_op_the_composer_can_execute`)
  requires every executable op to be exercised, or explicitly excused with its own test file.

## Style, as this codebase practices it

**State the measurement, not the assertion.** "Faster" and "more accurate" are not reviewable.
`738 ms → 18 ms at 4 M bits, bit-exact` is. Every performance or accuracy claim in this
repository carries its number, and the number was produced by running the thing.

**Name the problem, in the error.** Errors say what was passed, what is allowed, and where
possible what to do:

```python
raise ValueError(f"modulate_field: kind must be 'mzm' or 'dml', got {kind!r}")
```

A rule with the offending value missing sends the reader to go and print it themselves.

**Warn only where you would act on it.** A warning that fires on a legitimate usage gets
filtered out and then protects nobody. If a check cannot distinguish misuse from ordinary use,
prefer removing the footgun (an API that makes the right thing easy) over warning about it —
`Signal.stored_grid()` exists for exactly that reason.

**Derive a filter's length from the filter.** Never from signal properties like symbol rate or
"about ten symbols". `physics.response_extent` measures an impulse response's extent to a stated
relative threshold; use it.

**Record what you measured and rejected.** Several `BACKLOG.md` entries document changes that
were implemented, measured, and backed out, with the numbers. That is not clutter — it stops the
same dead end being re-explored. If you try something and the measurement says no, write the
measurement down.

## Commits and PRs

- Conventional prefixes: `feat:`, `fix:`, `perf:`, `docs:`, `test:`, `bench:`.
- The body carries the reasoning and the numbers. Commit messages here are long on purpose;
  `git log` is where the "why" for this codebase lives.
- Reference a GitHub issue with `Fixes #N` — the parenthetical `(#N)` form does **not**
  auto-close.
- Update `CHANGELOG.md` under an `## Unreleased` heading for anything a user would notice.

## Scope

Mechanisms live in the library; standards knowledge does not. A pseudo-random sequence is a
polynomial, a seed and a length — that is arithmetic, and it belongs here. *Which* sequence a
given standard names for a given test has a revision date and a document behind it, so it goes
in the registry a caller populates (`wfmsynth.patterns`), and a recipe records both the name and
the resolved parameters.

`tests/test_public_hygiene.py` enforces the related rule that the shipped library stays
self-contained and vendor-neutral — no company or product names, and no paths from anyone's
machine.
