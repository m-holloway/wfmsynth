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
- **Warnings.** New ones may appear in any release. They are how this library says "this is
  probably not what you meant", and adding one is not a breaking change. Run with
  `-W error` in CI if you want them to be.

## Versioning

Pre-1.0 and honest about it: `0.MINOR.PATCH`, where a minor bump may include a Tier 2(c) output
change, and a patch may not. Releases are tagged `v*`; `CHANGELOG.md` is the record, and its
**Changed output** heading is the section to read before upgrading a pipeline that has already
generated data.
