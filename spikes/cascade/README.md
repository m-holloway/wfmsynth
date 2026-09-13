# `cascade` — a channel is a path, not a lumped block

    PYTHONPATH=. python3 spikes/cascade/prove.py

## The defect

A shipped recipe read

    carrier -> ssc -> lossy -> reflect -> timing -> supply -> crosstalk -> sparam -> scope -> ...

**one lumped loss, then one lumped reflection.** Two things are wrong with that, and they are
different things.

1. **The echo pays the wrong attenuation.** Applying the whole channel's loss and *then*
   reflecting means the echo arrives having taken the full channel loss exactly once. A real
   near-end echo takes only the loss of the short segment it traverses — **twice**, out and back.
2. **A single lumped Γ has no positional structure.** Every reflection sits at the same
   electrical place, so a near and a far discontinuity of equal Γ produce identical echoes.

§C of `prove.py` is (2) as a table. The `LUMPED` column is **constant to four decimal places from
0.5 in to 10 in**; the `CASCADE` column falls from 50.23 mV to 7.22 mV over the same span, and
tracks the closed form `Γ·H_segment(f)²` to ~1e-14 mV.

That matters more than tidiness: positional structure is what converts an echo's arrival time
into a distance, and the distance is the whole product.

## Where the fix lives, and why

**In `sparam`, as S-matrix cascading.** Not as a "sectioned `lossy`/`reflect`". Cascading two
2-ports is

    D = 1 - A22·B11
    S21 = A21·B21/D        S11 = A11 + A12·A21·B11/D

and that `1/D` is not bookkeeping — it is the closed form of the infinite bounce series between
the two junctions, carrying the round-trip loss of whatever sits between them. Expanded for two
discontinuities separated by a line of one-way transmission `L`:

    S11 = Γ₁ + t₁²·Γ₂·L² / (1 + Γ₁Γ₂L²)

The positional attenuation *falls out of the algebra*: the far echo carries its own Γ, the
segment's loss **twice**, and the near junction's transmission twice. A time-domain sectioned
`lossy`/`reflect` would have to hand-code that series, would be a second implementation of
physics that already has one here, and would get the higher-order bounces — the terms that make a
real board's return loss ripple — wrong.

So the division of labour is:

| piece | role | changed? |
|---|---|---|
| `sparam.cascade` / `cascade_channel` | **carries the topology** | new |
| `physics.insertion_loss_db` | the loss law, shared by a section and a lumped channel | lifted out of `lossy_channel`, byte-identical |
| `physics.lossy_channel` | the lumped channel | unchanged |
| `physics.multi_reflection` | the lumped reflection — still correct for a line mismatched at both ends and nowhere else | unchanged |

A section may also be a **measured** Touchstone file, so a real connector can sit inside an
otherwise synthetic path. That is the "both" half of the answer: `sparam` carries the path, and
the path can be part measured.

## What `prove.py` establishes

| § | claim | result |
|---|---|---|
| A | the algebra against its own closed form | `S11 − Γ·H²` max err **0.0**; two-disc form **2e-17**; steps unitary to 1e-15; cascade passive |
| B | a discontinuity at a **known** distance lands at the arithmetic's lag | zero-phase loss: **−1.9 ps at every distance** (a constant matched-filter bias, not the model) |
| B | causal loss adds excess delay | **+8.8 ps at 1 in … +146.8 ps at 10 in**, growing with the segment's loss — real physics, paid twice |
| C | near vs far at equal Γ | cascade **50.23 → 7.22 mV**; lumped **14.4899 mV, flat** |
| D | both discontinuities in one path | 1.0 in and 10.0 in echoes, **15.96 dB apart at identical Γ** |

## The measured arm

A separate measured-channel study, against IEEE 802.3ap's B12 ATCA backplane
(hash-verified from the manifest; **no measured file is committed to this repository**). On that
board the lumped topology over-charges a 0.20-inch discontinuity by **21.4 dB** and under-charges
a 17.4-inch one by **10.4 dB** — a **39×** spread in the Γ you would infer, from the topology
alone.

## The bias this exposed

Because a lumped echo is a delayed copy of a signal that has *already* taken all its loss, the
echo and the main cursor carry the same dispersion and it cancels out of the lag. A cascaded echo
traverses 2·d of dispersive line the cursor never sees, so it arrives late. Measured end to end
through a lag-to-distance estimator: a **true 1.66 inch** reads back as
**1.66 inch exactly** with zero-phase loss, and as **1.7345 inch (+4.5 %)** at 3.46 dB of segment
loss and **1.7933 inch** at 6.00 dB with causal loss. The shipped 1.66-inch result is exact for
the article it was measured on; what had never been tested is a physically cascaded echo.

## Byte-identity

`tests/test_byte_identity.py` pins 15 default paths, including the exact recipe above, by SHA-256
of their raw float64 bytes. They were captured before this work and are unchanged.
