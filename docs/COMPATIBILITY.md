# What this library promises not to break

Most libraries have one compatibility surface: the API. This one has three, and the unusual two
are the ones that bite.

| surface | the promise | how it is enforced |
|---|---|---|
| **API** | a name that exists keeps working, or is deprecated before it goes | review |
| **Rendered output** | the same recipe renders the same samples | `tests/test_byte_identity.py` |
| **Recipes** | a stored recipe keeps replaying, and never silently means something else | `compose.NUMERIC_CHANGES` |

The second and third exist because this library's product is *data*. A record generated last
year may be in a training set today, and its recipe is the only thing that reproduces it. An
output change is therefore not an implementation detail — for a consumer it is indistinguishable
from a corrupted dataset.

This policy is written because the project already broke it once, deliberately and for a good
reason, and had no vocabulary to describe what it had done. See **The precedent** below.

---

## Tier 1 — the API

**Adding** is always fine: a new op, a new parameter with a default, a new module.

**Removing or renaming** requires a deprecation cycle: the old name keeps working and warns for
at least one minor release, and the warning names the replacement. Do not rename a parameter
that is recorded in a recipe without adding it to Tier 3 as well — a recipe carries parameter
names, so renaming one is also an output-compatibility event.

Pre-1.0, a minor version may remove something that has been through a cycle. It may not remove
something without one.

## Tier 2 — rendered output

**The band is the definition.** `tests/test_byte_identity.py` pins the default path to a
tolerance of about `3.8e-13`, not to bytes. Floating-point reassociation — a faster loop order,
a different reduction, a fused multiply — lands inside it. That is deliberate: pinning to bytes
would make every optimisation a breaking change and freeze the library.

**A complex record needs numpy 2 to replay to the byte, which is why `numpy>=2.0` is the
declared floor from 0.46.1.** numpy 1.x forms a complex product in
a vector loop whose rounding depends on where the array happens to be allocated, so the same
I/Q or optical chain rendered twice in one process can differ in its last bit: measured on
numpy 1.26.4 (arm64 macOS, CPython 3.9.6), `x * g` for one complex array and one complex
number gave three distinct byte results in 200 tries, and `iq(...).gain(by=<complex>)` two
sample digests. That is inside the band (4.6e-16), but a sample digest is not a band: on
numpy < 2 a complex record replays to rounding, not to the byte, and the tests that assert
byte equality of a complex replay can fail there (13 to 18 of them per run). Real records are
unaffected, and numpy 2.0.2 and later were byte-stable in every run. A source tree used from
`PYTHONPATH` under numpy 1.x still imports, with one warning saying this.

So there are exactly three kinds of change:

**(a) Inside the band.** Not a compatibility event. Say the measured number in the commit anyway
— "1.4e-15, six ULP, from the polyphase resampler" — because "no change" is a claim nobody can
check and this one can.

**(b) Outside the band, and optional.** Put it behind a parameter that is **recorded in the
recipe**, so a record replays on the path it was rendered with. The precedent is
`apply_transfer(method="overlap")`: it computes the same convolution in blocks for 1.5× the
speed and a fifth of the memory, and it stays opt-in because it differs by 7.7e-5 of
peak-to-peak — a whole 8-bit code on 1.16 % of samples once a converter sees it. Attractive is
not the same as free.

**(c) Outside the band, and necessary.** Sometimes the old behaviour is simply wrong. Then:

1. add an entry to `compose.NUMERIC_CHANGES` naming the op, the version, and what changed;
2. record it under **Changed output** in `CHANGELOG.md`, not under *Fixed* — a consumer
   scanning for things that move their data must find it there;
3. bump the minor version.

Step 1 is what makes the difference between a documented change and a silent one. It is the only
mechanism by which a *stored recipe* can learn that the world moved underneath it.

## Tier 3 — recipes

A recipe must always **replay**. It may warn; it may not be refused, because the caller may be
deliberately reproducing old behaviour and only they can judge that.

`from_recipe` compares the recorded `wfmsynth_version` against `NUMERIC_CHANGES` and warns when
a recipe predates a change to an op it contains, naming the op and what to do. The silent cases
are as important as the loud one and are tested: a current recipe, a recipe not using the
changed op, and a recipe with no version field at all (those predate the field and are not
evidence of anything — warning on all of them would teach people to filter the category out).

**What a recipe may hold grew in 0.46.0, by addition.** A parameter may be a sub-chain
(`{"ops": [...]}` or `{"from": <recipe>}`); a hand-written recipe may state a toleranced value
(`{"nom": ..., "tol_pct": ...}`, `{"corners": {...}}`) where a number goes, and a resolved one
keeps its statement under the op's `_unit` annotation; a several-channel record has `channels`
(and `shared`) where a single one has `ops`; and `grid` carries `domain` / `fc_hz` only when
the record is not real. Those key shapes are reserved: a parameter dict with a list under `ops`,
a dict under `from`, or any of `nom`, `tol_pct`, `tol_abs`, `corners` is read as what it looks
like. A recipe written before 0.46.0 contains none of them and replays unchanged.

**Two kinds of chain that rendered before 0.46.0 are now refused, replay included.** Both are
the optical field becoming a domain, and both were a record that did not mean what its recipe
said. An electrical op placed between `eo` and `photodetect` (`eo().scope(...)`) used to cast
the complex field to float under a numpy warning, dropping its phase; it is refused naming
`photodetect`. A field op (`fiber`, `edfa`, `optical_mpi`, `photodetect`) on a voltage with no
`eo` before it used to treat the voltage as a field; it is refused naming `eo`. No shipped
chain, example or test did either, and every chain that goes `eo` .. `photodetect` is
bit-identical. This is the one place a stored recipe is refused, not warned, and it is stated
here because the rule above says it should not happen: to replay such a recipe, render it
with 0.45.x.

**Adding to `NUMERIC_CHANGES` is the running cost of the guarantee.** It is one tuple, and it is
small beside a record that silently means something other than it says.

---

## The precedent

0.41.0 inverted `de_emphasis_taps`' sign convention (#53), because every specification quotes
de-emphasis as a **negative** dB and this library had it positive. For a caller writing new code
that was unambiguously a fix, and the docstring told them.

It could not tell a stored recipe. `de_emphasis(db=3.5)` rendered as de-emphasis under 0.40.0
and as **pre-emphasis** afterwards: the transmitter shaping of an existing record inverted, and
it replayed without complaint. `recipe()` had recorded the version all along and nothing read it.

That is the change this policy is built around, and it is the first entry in
`NUMERIC_CHANGES`. The fix was right; shipping it without a way for old recipes to notice was
not.

### What has moved since

`NUMERIC_CHANGES` is the authoritative list; this is the same list in prose.

- **0.43.0, `net`.** A schedule time within a billionth of a sample of a boundary is on it.
- **0.46.0, `carrier("square")` and `carrier("cmos")`.** Each edge is at its own time to a
  fraction of a sample, not on the next sample after it, and an edge exactly on a sample is
  no longer decided by floating-point rounding of the phase. A record with an edge between
  samples, or one that rounding had put a sample late, moves by up to one sample at that
  edge: it had up to one sample peak to peak of timing error that was the grid's (15.58 ps
  at 64 GSa/s for a 0.997 GHz clock; now 3.0e-5 ps where the edge spans 9.6 samples, 1.1e-2
  ps at `square`'s default 3.2-sample edge and 5.2e-2 ps for a 2.2-sample `cmos` edge). A record whose edges were all on samples and rightly placed
  is unchanged bit for bit. The entries are keyed `carrier/square` and `carrier/cmos` and
  match that kind of carrier anywhere in a recipe, so a recipe holding only `nrz` or `sine`
  carriers is not warned. To reproduce an old record exactly, render it with 0.45.x.

---

## What is explicitly *not* promised

Being honest about the edges is what makes the rest credible.

- **Bit-identity across numpy, scipy, BLAS or CPU.** The band is the promise, not the bits. A
  different LAPACK can move the last digits; `test_byte_identity.py` is a tolerance test for
  exactly this reason.
- **Bit-identity across Python versions** beyond that same band.
- **Anything under a leading underscore.** `_min_phase_H`, `_LEAD_LTI`, `_EXEC` and their
  neighbours are internal and change without notice. If you depend on one, open an issue and say
  why — that is usually a missing public API.
- **The `JUDGEMENT`-grade models.** Fitted shapes (`resonant_reflection`, the nonlinearity
  defaults, the crosstalk kernels) may be re-fitted as better evidence arrives. Such a re-fit is
  a Tier 2(c) change and goes through the same gate. See [`FIDELITY.md`](FIDELITY.md).
- **The list `roles()` returns.** It names the randomness a chain draws, and it changes when
  the list is made truer: 0.46.0 added every role that draws only under a condition
  (`acquire`, `store`, `tia`, sub-chain roles, ...) and removed the few that were listed and
  drew nothing. The draws themselves, and so the samples, are Tier 2.
- **Warnings.** New ones may appear in any release. They are how this library says "this is
  probably not what you meant", and adding one is not a breaking change. Run with
  `-W error` in CI if you want them to be.

## Telling which build you actually have

`wfmsynth.__version__` alone is not always enough, and once it was actively misleading.

**What happened, recorded so it is not repeated.** The version was bumped to `0.41.0` on
2026-09-22, *before* a large performance pass began. Every commit of that pass also reported
`0.41.0`. A checkout taken from `main` on 22–23 September and the `v0.41.0` tag were therefore
materially different code — an LFSR that took 740 ms versus 18 ms for the same call — under one
label. A user who checked `__version__`, saw `0.41.0`, and concluded nothing had shipped was
reading the evidence correctly.

That is the same one-label-two-meanings defect `NUMERIC_CHANGES` exists to prevent for recipes,
and the cause was procedural: **the version was bumped when work started rather than when it
shipped.** Bump at release.

Two ways to check what you have:

```bash
git describe --tags          # e.g. v0.42.0, or v0.42.0-7-gabc1234 for 7 commits past it
```

```python
# or test the CODE rather than the label
import time, wfmsynth as ws
from wfmsynth import physics as P
t = time.perf_counter(); P.lfsr(P.PRBS_TAPS[13], 1 << 21); dt = time.perf_counter() - t
print(ws.__version__, f"{dt * 1000:.0f} ms")     # ~18 ms with 0.41.0's perf work, ~740 without
```

One more trap, for contributors rather than users: `__version__` reads the **installed
distribution's** metadata first, and an editable install freezes that at install time. After
changing `pyproject.toml`, run `python -m pip install -e .` or `__version__` stays stale.
`tests/test_version_single_source.py` catches it and says so. The precedence is deliberate — a
vendored copy must not pick up the host project's `pyproject.toml` two directories up.

## Versioning

Pre-1.0 and honest about it: `0.MINOR.PATCH`, where a minor bump may include a Tier 2(c) output
change, and a patch may not. Releases are tagged `v*`; `CHANGELOG.md` is the record, and its
**Changed output** heading is the section to read before upgrading a pipeline that has already
generated data.
