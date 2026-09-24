# What is calibrated, what is closed form, and what is a fitted shape

The honest question about any synthetic-data library is *which of these numbers came from
hardware*. This page answers it for wfmsynth's own claims, using the evidence vocabulary
[`DATASET-METHODOLOGY.md`](DATASET-METHODOLOGY.md) §2 already defines for sourced parameters:

| marker | means |
|---|---|
| `MEASURED` | compared against real captured hardware records, and the row says which |
| `DERIVED` | a closed form, with the arithmetic checkable |
| `JUDGEMENT` | a phenomenological shape chosen to look right; fitted, not derived |

Nothing here is new work. Every figure below is already in the source or in
`python -m wfmsynth.validate`; this page exists because it was spread across docstrings, which
is not where a skeptic looks.

**Start here instead of with this page:** `python -m wfmsynth.validate` runs 310 property
assertions against rendered samples and prints each with its measured margin. That is the
primary artifact. This page says what the assertions are *worth*.

---

## The calibration set

Three real exports from a real-time sampling oscilloscope, 40 GSa/s, 8 M samples (200 µs) each,
at three different symbol rates (`instrument.py`, the comment above `interleave_adc` and the
`store_record` docstring). Vendor-neutral by policy — `tests/test_public_hygiene.py` gates that —
so they are referred to as captures A, B and C.

They are used as *targets*, not as data: nothing in the library ships their samples.

---

## `MEASURED` — checked against those captures

| Claim | Evidence |
|---|---|
| **A stored record sits on a code lattice, fully occupied** | A/B/C hold 1,851 / 1,880 / 2,035 distinct values, 1.0000 of them on the lattice, occupancy 0.997–1.000. A render that stops at the converter holds ~7 M distinct values — 5–11× smoother than any real capture. |
| **The record's noise floor is its own lattice's `q²/12`** | The stop-band PSD equals `q²/12/(fs/2)` to under 0.1 dB in 3 of 3 captures. `validate.py` reproduces this to +0.02 dB, and to +3.02 dB with dither. |
| **A link's symbol clock and the record's timebase differ by ~2 ppm** | Fitted from 163k–272k threshold crossings per capture: −1.987 / −1.822 / −1.983 ppm, ≈16 samples of slip over 200 µs. Fit residuals 0.343–0.561 samples. On capture A the offset is stable across ten segments (−1.751 to −2.435 ppm), so it is an offset and not drift. This is why `sample_clock` exists as an independent clock. |
| **A scope's selected-bandwidth filter is not gentle** | The noise floor drops 38 dB across 2 GHz at the corner, flat both sides — which is why `scope(kind="brickwall")` is a digital, zero-phase stage rather than an analog roll-off. |

**What the captures do not establish**, stated in the source and repeated here: which of the two
oscillators is off. The 2 ppm figure is a *ratio*.

---

## `DERIVED` — closed form, arithmetic checkable

These are not calibrated against hardware and do not need to be; they are mathematics, and
`validate.py` checks the implementation reproduces the closed form.

- **Insertion loss** `IL(f) = (a√f + b·f)·length`, with `b = 2.3·√eps_r·tand` dB/in/GHz.
- **Minimum-phase (causal) channel** — the cepstral/Hilbert relation, so loss and phase are
  linked (Kramers-Kronig) rather than independently chosen.
- **Bandwidth ↔ rise time** — `BW·tr = 0.3497` (10–90 %), holding within 0.2 % on the default
  Bessel-4 front end. The 30–70 % constant is 0.1348 in theory and 0.149 there; the library says
  so rather than quoting one number for both.
- **Transmission-line stubs** — `Z_in = −j·z0/tan(βL)` (open), `j·z0·tan(βL)` (short); lossless
  and reciprocal by construction.
- **Reflection lattices, S-parameter cascades, mixed-mode conversion** — `se2mm`, the
  bounce-series `1/(1 − A₂₂B₁₁)`, and renormalization `S' = (S − ΓI)(I − ΓS)⁻¹`.
- **Quantisation floor** `q²/12`, and the interleave spur positions `k·fs/M`.
- **4D-PAM5 / 8B1Q4, 8b/10b, 64b/66b, PAM4 Gray** — coding, verifiable against the arithmetic.

---

## `JUDGEMENT` — fitted shapes, chosen to look right

Use these knowing what they are. They are phenomenological: the *shape* is defensible, the
particular coefficients are an engineering choice.

- **`resonant_reflection`** — a resonant discontinuity as a fitted band-pass Γ(f). A physical
  stub is `sparam.stub()` instead; this is the shape you use when you have a measured resonance
  and want something that matches it.
- **`nominal_nonlinearity`** — soft odd compression, level-dependent noise, rise/fall asymmetry.
  Chosen so a "nominal" record is not suspiciously perfect, which is itself a giveaway.
- **Jitter decomposition** — the *mechanisms* (RJ, PJ, DCD, ISI, SSC profile) are standard; the
  default magnitudes are not calibrated to any particular device.
- **`crosstalk` FEXT/NEXT kernels** — the coupling shape is a model; `db_to_coupling` sets the
  level you ask for.
- **Probe compensation and ground-lead resonance** — a real zero/pole and a second-order
  resonance, with values that are plausible rather than measured from a specific probe.

---

## Known limitations that affect fidelity

Stated here rather than discovered later. `BACKLOG.md` carries the full list with measurements.

- **`probe_loading(causal=True)` applies its pole circularly**, so the response to the record's
  tail lands on its head — confined to about five time constants (22.5 ps at R=50, C=0.45 pF) at
  the record's *head*, which `Signal.lead_in` renders and discards. Making it linear was
  implemented, measured and reverted: it costs 0.05 dB and 1.4° across the *whole* record,
  because an analog pole does not vanish at Nyquist and its sampled impulse response has a 1/k
  tail. BACKLOG #58.
- **Eight of twelve examples are under-sampled** against the library's own `k ≥ 8` rule, two of
  which print measurements. Listed with measured `k` in
  `tests/test_examples_run.py::UNDERSAMPLED`.
- **A record is not a device.** Nothing here models a specific part, and
  `tests/test_public_hygiene.py` enforces that. Standards knowledge lives in the caller's
  pattern registry, not in the library.

---

## The test a skeptic should actually run

`examples/sim_to_real.py` trains a classifier to distinguish synthetic records from real ones
and prints per-feature AUC, with the top feature naming the physics that most gives the
synthetic set away. It is the adversarial version of this page, and it ships.
