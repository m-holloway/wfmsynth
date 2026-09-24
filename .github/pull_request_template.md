## What changed, and why

<!-- The reasoning and the numbers. See CONTRIBUTING.md — "state the measurement, not the
     assertion". "Faster" is not reviewable; "738 ms -> 18 ms at 4 M bits, bit-exact" is. -->

## Effect on rendered output

<!-- Tick one. Point 5 of the definition of done: existing defaults stay bit-identical unless a
     deliberate versioned change is stated. -->

- [ ] Bit-identical — no rendered sample moves.
- [ ] Moves within the byte-identity tolerance (~3.8e-13). Measured delta: `______`
- [ ] Deliberate versioned change beyond that band. It is recorded in the recipe / the
      references are re-pinned in their own commit, and `CHANGELOG.md` says so.

## Gates

- [ ] `python -m pytest -q`
- [ ] `python -m wfmsynth.validate`
- [ ] `python tools/benchmark.py --quick --check`
- [ ] Regenerated `wfmsynth/opkeys.py` if an op gained or renamed a parameter
- [ ] `CHANGELOG.md` updated if a user would notice
