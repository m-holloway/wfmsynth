# spikes/eye3m — how fast is a 3 M UI eye, and what makes it slow?

48 Mpt at 16 samples/UI, 384 MB as float64. The question was whether the eye render
is efficient at that size and whether it can be made "lightning fast" without
touching the clock recovery, whose jitter transfer IS the measurement.

Run in this order:

    python3 spikes/eye3m/record.py      # build/cache the 48 Mpt record
    python3 spikes/eye3m/profile3m.py   # PART 1: where the wall clock goes, before and after
    python3 spikes/eye3m/ablate.py      # PART 2: what each technique bought, one at a time
    python3 spikes/eye3m/bench.py       # before/after wall clock + peak RSS, one process each
    python3 spikes/eye3m/verify.py      # is it the same eye? and is the CDR untouched?

`shipping.py` loads the demo app's engine kernel — the reduction that actually runs —
pinned at commit `b37483b`, the commit whose ground-truth evidence this work must not
disturb. It is materialised read-only into the scratch directory with `git archive`;
nothing here writes to the demo checkout. The pin is not decoration: that file changed
twice while this was being measured, and one change moved a returned array by 1667
elements. Set `WFMSYNTH_EYE_BASELINE` to compare against another revision.

## Part 1 — what it actually cost (measured, not guessed)

Whole frame, 3 M UI, interactive trace cap (400):

| stage | ms | % |
|---|---:|---:|
| **edge finder** (`_views_crossings`) | **357** | **61 %** |
| period search (4 percentile seeds + 5 least-squares fits) | 108 | 19 % |
| CDR (2 loop filters, then 2 more for the tracked fraction) | 50 | 9 % |
| threshold + volts axis (mean, ptp, min, max) | 24 | 4 % |
| per-symbol phase hold | 11 | 2 % |
| the fold itself | 17 | 3 % |
| **total** | **583** | |

The dominant cost is not the loop and not the fold: it is finding the edges, and
inside that it is not arithmetic but allocation. The exhaustive form marks every
sample that sits outside the hysteresis band — 44.9 M of 48 M "events", of which
1.5 M survive the alternation filter — and then walks back to each straddling sample
pair with two running maxima over an int64 index array: five full-length passes and
about 2.5 GB of intermediates for a 384 MB record. Peak RSS 2.99 GB.

At the full-density cap (every UI folded, 3.0 M traces, 384 M interpolated points)
the fold takes over: 2896 ms of a 3454 ms frame.

## Part 2 — what each technique bought

Measured one at a time on the same record, each row the same computation done two
ways with the two results compared (`ablate.py`):

| stage | technique | before → after | gain | result |
|---|---|---|---|---|
| edges | armed **runs**, not armed samples | 354 → 158 ms | 2.24x | identical |
| edges | … over 8 threads | 157 → 76 ms | 2.06x | identical |
| edges | walk-back as a **merge**, not a binary search | 45.7 → 12.1 ms | 3.79x | identical |
| period | four percentiles in **one** call | 35.3 → 12.4 ms | 2.85x | identical |
| period | the five candidate fits at once | 66.9 → 20.9 ms | 3.20x | identical |
| phase | hold as a run-length, not a NaN fill-forward | 10.4 → 6.9 ms | 1.49x | identical |
| CDR | reuse the residual instead of refiltering for it | 48.5 → 23.7 ms | 2.04x | identical |
| CDR | the loop's two filters at the same time | 24.0 → 12.8 ms | 1.88x | identical |
| threshold | mean/min/max at the same time | 15.1 → 6.7 ms | 2.25x | identical |
| fold (400) | two-tap lerp, not `np.interp` on an `arange(n)` axis | 16.3 → 3.4 ms | 4.78x | identical |
| fold (all) | … and spread over 8 threads | 2164 → 569 ms | 3.81x | identical |
| fold | in-place ufuncs instead of a ten-temporary expression | 2347 → 2160 ms | 1.09x | identical |

Negative results, kept because they were tried:

| technique | effect | why it was rejected |
|---|---|---|
| `np.histogram2d` instead of `np.bincount` | **3.8x slower** (2344 → 8959 ms) | it bins floats against edge arrays; bincount counts integers |
| float32 through the accumulate path | **5 % slower** *and* not identical: max &#124;d&#124; = 4 counts | the gather is not bandwidth-bound, and 24-bit mantissa flips rows at bin boundaries |
| threading the fold at 400 traces | no change (3.4 → 3.4 ms) | 400 traces is one block; there is nothing to split |
| chunking the edge scan below 1 M samples | 1.9x **slower** at 64 k (157 ms vs 78 ms) | per-block Python overhead beats the cache win |
| decimating the record before folding | not attempted, and should not be | the crossings and the period fit are measured on every sample; the trace cap is already the honest decimation, and it decimates *traces*, not samples |

## Result

| | before | after (1 thread) | after (8 threads) |
|---|---:|---:|---:|
| 3 M UI, 400 traces | 555 ms | 302 ms (1.84x) | **152 ms (3.65x)** |
| 3 M UI, every UI folded | 3454 ms | 2496 ms (1.38x) | **706 ms (4.89x)** |
| peak RSS | 2988 MB | 925 MB | 1203 MB |

The density is **bit-identical**, not equivalent-within-tolerance: max &#124;d&#124; = 0 over
51 200 counts at the interactive cap and over 383 786 368 counts with every UI folded,
on one thread and on eight.

## The constraint that was not negotiable

`verify.py` re-measures the sibling unit's evidence through the new path:

* recovered period vs the constructed one: **−3.8e-08** (the established figure)
* grid told 1.7x and 0.5x the true rate: density **byte-identical**, max &#124;d&#124; = 0
* 0.20 UI of sinusoidal jitter at **0.02x** the loop bandwidth → **0.0015 UI** rms
  residual, 0.000 UI of crossing smear
* the same amplitude at **3.33x** → **0.1406 UI** rms against 0.1414 built, smearing
  the crossing by **0.394 UI**

and, on five other records plus the 48 Mpt one, every stage against the shipping
stage: crossing positions and directions, all five period fits, fifteen report fields,
the decision instants, and the density at three trace caps — 184 checks, all equal.

The CDR is untouched by construction: `wfmsynth.eye` calls
`wfmsynth.cdr.recover_clock` and never reimplements it. The only change in that
neighbourhood is that the loop's two filters run at the same time instead of one
after the other, and that the tracked-out fraction is computed from the residual the
loop already produced instead of running the loop a second time to get it back.
