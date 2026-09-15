# Building a defensible synthetic waveform dataset

For a team building one of these datasets with no tools but this library. It covers the mental
model, a procedure for producing parameters that hold up under scrutiny, the traps that cost real
time, the arithmetic that sizes a record, the state of the library, and how a consumer filters what
it must not trust.

Two conventions apply to everything below, including anything you add to this file:

- **Cite documents by standard designation only** — "IEEE 802.3 Clause 120.5.11.2.1",
  "PCI Express Base Specification Rev 4.0 Table 8-7". Never a vendor, an instrument model, a
  customer, or another repository or tool. Where you need to refer to the layer that builds
  recipes and stores records, say *the caller* or *a planning layer*.
- **Mark every number with how you know it.** The vocabulary is in §2 and this document uses it on
  its own claims.

Run the suite before and after any change:

```bash
python -m pytest tests -q          # the whole directory, not the file you touched
python -m wfmsynth.validate        # the physics gate: PASS/FAIL per property, nonzero on any fail
```

---

## 1. The mental model

A record is a pipeline. Each stage has one owner. Naming a stage is what lets you label it, sweep
it, and hold it constant.

| # | stage | what it decides | module | ops / entry points |
|---|---|---|---|---|
| 1 | bits | the transmitted bit stream | `physics` (`prbs`, `PRBS_TAPS`) | inside `carrier` |
| 2 | line coding | run length, disparity, DC content, **symbol statistics** | `coding` (`dc_balanced`, `scramble_64b66b`, `running_disparity`, `max_run`) | **no op — see §5** |
| 3 | symbols | the symbol sequence and its period | `physics` (`carrier_symbols`, `prbs13q`, `prbs31q`, `clock_pattern`) | `carrier(pattern=…)`, `symbols(symbols=[…])` |
| 4 | level mapping | how many levels and where they sit | `physics` (`nrz`, `pam4`, `pam(N)`) | `carrier(kind="nrz"/"pam4"/"pam<N>")` |
| 5 | waveform | edge shaping, rise time, source jitter | `physics` (`_shape_edges`, `resolve_rise_time`, `Jitter`) | `carrier(tr_frac=, jitter=, causal=)` |
| 6 | transmit shaping | pre-emphasis, presets, DCD, SSC, pair skew, bus electrics | `physics`, `bus` | `tx_ffe`, `de_emphasis`, `dcd`, `ssc`, `intra_pair_skew`, `open_drain` |
| 7 | supply | rail coupling and slow drift | `physics` | `supply_coupling`, `drift` |
| 8 | channel | loss, reflections, crosstalk, AC coupling, measured S-parameters, fibre | `physics`, `sparam`, `optical` | `lossy`, `reflect`, `resonant_reflect`, `crosstalk`, `crosstalk_matrix`, `ac_couple`, `sparam`, `cascade`, `eo`/`fiber`/`optical_mpi`/`edfa` |
| 9 | localized defects | one weak symbol, one ringing edge, a sag in a run | `events` | `events` |
| 10 | probe | what the instrument observes *through* | `instrument` | `probe` |
| 11 | receiver | CTLE, FFE, DFE, AGC, front-end noise, an independent sampling clock | `rx`, `cdr`, `compose` | `ctle`, `rx_ffe`, `dfe`, `agc`, `rx_noise`, `sample_clock` |
| 12 | instrument | analog bandwidth, timebase jitter, ADC (ENOB, interleave), resampling | `instrument`, `acquire` | `scope`, `timebase`, `digitize`, `acquire` |
| 13 | stored record | integer codes at a stated full scale — the thing a consumer reads | `instrument` (`store_record`) | `store` |
| 14 | measurement | labels measured from the record, independent of the knobs | `measure`, `eye` | `Signal.ground_truth()`, `eye.eye()` |

Around the pipeline: `Grid` binds the abstract grid to Hz/seconds/volts; `Streams` gives every
stochastic factor its own role-tagged RNG so re-rolling one factor leaves the others bit-identical;
`Scene` composes correlated lanes; `sweep` holds confounders constant; `validate` and `tests/` are
the gates.

### Ops versus helper calls

An **op** is an entry in `compose._EXEC`: a name plus the exact knob values used. It lands in
`Signal.recipe()["ops"]`, therefore in the key-sorted JSON of `to_json()`, therefore in
`sha256()` — the content address of the synthesis program. `from_recipe(r).waveform()` reproduces
the samples bit-for-bit.

A helper call does not have this property. Compute a line code in caller code and feed the result
in as `symbols=[…]`, and the record is still reproducible. The recipe records only the levels; it
does not record which code produced them. Nothing can filter by code, sweep the code, or diff two
datasets on it, and a reviewer cannot see it was applied. **Rule:** if a consumer will ever want to
select on a value, make it an op.

Two rules follow. Both have caused real bugs:

- **Emit only the arguments you were given.** The kernel's defaults are the authority. Restating a
  default in the recipe changes the content address of every record carrying that op on the day
  the default changes. `recipe()` writes `lead_in` only when a lead-in is actually set, for exactly
  this reason.
- **A new op is not finished until both gates pass.** Neither may be allowed to default:

  | gate | where | what it demands |
  |---|---|---|
  | knob-sweep completeness | `tests/test_composition.py::test_the_knob_sweep_covers_every_op_the_composer_can_execute` | every `_EXEC` op has a row proving each knob moves the samples, or is explicitly excused. A row that fails is a knob the recipe records and the physics ignores. |
  | lead-in classification | `compose._lead_check_ops` | every op is in exactly one of `_LEAD_LTI` (has an impulse response, so it sizes the guard), `_LEAD_SKIP` (has none), `_LEAD_ANALYTIC` (memory, no probeable response — state the extent) or `_LEAD_REJECT` (a lead-in changes what it means). An unclassified op is refused rather than falling into the safe-looking pile. |

  Also: give the op a stage kind in `OP_KIND`, draw randomness from `streams.role(f"<op>/{idx}")`,
  and if a knob is a *fraction of the record* list it in `_LEAD_RELATIVE` with the absolute-unit
  fix — a fraction is not a physical quantity, and a lead-in changes the record it is a fraction of.

---

## 2. The method that produced defensible parameters

Repeatable, in order, and slower than guessing. It is what lets someone else check a number
against the standard, instead of taking it on trust.

1. **Obtain the governing document, at the revision the device is graded against.** Record whether
   you actually got it. An extract that says *none obtained* is a result; one that implies
   otherwise is a liability. Where a document is paywalled, read the neighbouring revision and say
   so **on every row taken from it**.
2. **Transcribe one row per parameter**, with these columns and no fewer:

   | column | content |
   |---|---|
   | symbol | the standard's own name for it, e.g. `TTX-UTJ` |
   | value + unit | as printed, with rms vs peak-to-peak stated |
   | document | designation |
   | revision | the revision **you read** |
   | clause | clause / section |
   | table or figure | the locator |
   | column | which data-rate / speed-grade column |
   | plane | the test point the number applies at |
   | marker | below |
   | note | the arithmetic, the measurement method, or what would falsify a judgement |

3. **Mark every value:**

   | marker | means |
   |---|---|
   | `CITED` | a document, revision, clause and table, all read by the author |
   | `DERIVED` | arithmetic on cited values, with the arithmetic shown |
   | `MEASURED` | the author ran a measurement, and the row says how |
   | `JUDGEMENT` | an engineering estimate, with what it rests on and what would falsify it |
   | `NOT_SPECIFIED` / `NOT_FOUND` | looked for, absent, and the row says where it was looked for |
   | `PRELIMINARY` | in use, not yet verified — a placeholder that a consumer must be able to filter out |

4. **Have the citations attacked by someone who did not write them**, and who re-obtains the
   primary document rather than taking the author's word. The failure this exists to catch is a
   `JUDGEMENT` marked as if it were `CITED` — a plausible-looking table reference nobody actually
   read.
5. **Apply the corrections, then re-derive the recipes**, and treat any extract the adversarial
   pass did not reach as one researcher's unchecked reading, however well sourced it looks.

**What a correction pass returns.** In one pass over 17 extracted documents it produced 201
corrections and found no fabricated citation. The distribution is worth knowing because it says
where to spend review effort:

| corrected | count | what it was |
|---|---|---|
| locator | 42 | right value, wrong table or figure number |
| marker | 31 | a `DERIVED` or `JUDGEMENT` marked `CITED` |
| rms / peak | 23 | an rms figure used as peak-to-peak, or the reverse |
| column | 21 | right table, wrong data-rate column |
| plane | 16 | the right number at the wrong test point |
| revision | 15 | the number from a neighbouring revision |

The values were usable and the table-level locators were not. Before a correction pass has run
over an extract, cite a number to the extract that carries it, and leave its table reference
unquoted until it has been checked.

**The revision trap** is why every row names the revision of the source read: one jitter limit at
16 GT/s is 12.5 ps in one revision of the base specification and 11.8 ps in the next, and a device
of that generation is graded against the earlier one. Fifteen of the 201 were this.

**An absence can never be cited to a table.** `SSC: none [CITED] §6.1 Table 11` is wrong even when
the conclusion is right — the table does not mention spread-spectrum clocking, which is precisely
why there is nothing to cite. Those rows are `NOT_SPECIFIED`, with the note saying where you looked.

---

## 3. The traps, named

Each cost real time. Each will recur.

### 3.1 Sizing the grid from the unit interval instead of the edge

`samples across the transition = tr × fs`. A fixed samples-per-UI sets `fs` from the *symbol rate*,
which says nothing about the edge.

| bus | baud | UI | tr | `fs = 16 × baud` | samples across the edge |
|---|---|---|---|---|---|
| fast serial | 16 GBd | 62.5 ps | 30 ps | 256 GS/s (dt 3.9 ps) | 7.7 — fine, which is why the bug hides |
| slow differential bus | 500 kbit/s | 2 µs | 10 ns | 8 MS/s (dt 125 ns) | **0.08** |

A record with 0.08 samples per edge **has no edges**: every rise-time, jitter and slew measurement
made on it measures the sample grid. The same defect gave a modelled 1.6 MHz front end an 1175 %
fall-time error on a slow bus.

**Rule:** choose `fs` from the edge, `fs ≥ k / tr`, and derive samples-per-UI from that.
`k ≥ 8` [MEASURED — `k` counts sample intervals across the transition, so k = 4 spans it with
five sample points, three of them interior, which is the fewest that constrains the edge's shape.
Synthesising one edge and comparing the delivered 10–90 % rise time to the requested one gives
1.2 % at k = 8, 0.8 % at k = 10 and 0.04 % at k = 32, and the error falls as the grid is refined,
which is how you know it is the grid's and not the model's. The hard floor is separate and closer:
under `tr_frac × samples_per_ui = 2` the kernel clamps to two samples and warns, and from there
down every edge measurement is the grid's. `tests/test_edge_rise_time.py` holds these.]
**Detection:** measure `tr` on the delivered record. If it comes back near `dt`, the grid is what
you measured.

### 3.2 Reading a rise time with the wrong threshold pair

```
BW · tr = ln((1 - lo) / (1 - hi)) / (2π)          # single-pole, lo/hi as fractions
```

| thresholds | BW·tr | who quotes it |
|---|---|---|
| 10–90 % | 0.3497 | high-speed serial specifications, most textbook work |
| 20–80 % | 0.2206 | some transmitter specifications |
| 30–70 % | 0.1348 | **open-drain bus specifications** |

10–90 % / 30–70 % = 2.59. Reading a 30–70 % figure with the 10–90 % constant is a factor of 2.6
in bandwidth, and it is silent — the record looks plausible.

An instrument bandwidth is the value the compliance procedure names. It is not derived from what
the signal implies. A 2 % rise-time budget on a 21.9 ps edge implies `tr_scope = 21.9 × √(1.02² − 1)
= 4.40 ps`, i.e. `BW = 0.3497 / 4.40 ps ≈ 80 GHz`. Compliance for that bus runs on 33 GHz:
`tr_scope = 10.6 ps`, `√(21.9² + 10.6²) = 24.3 ps`, an edge rounded by ~11 % (measured ≈ 12 %). This
rounding appears in every real capture. **Rule:** model the named bandwidth. Do not "fix" it to the
value the edge implies, or the synthetic set will be systematically sharper than any real
measurement it is compared against.

### 3.3 Confusing a channel code with a line modulation

A **code** (8b/10b, 64b/66b, 128b/13xb, a scrambler) changes *symbol statistics* — run lengths,
disparity, DC content — on a line whose level count it never touches. A **modulation** (NRZ, PAM4,
PAM-N, a ternary line) changes the *level count*.

Reading a code as a modulation produced a ternary record on a binary bus; the mirror of the same
mistake produced a binary record on a ternary one. The check is one division:

```
bits per UI = bit rate × UI          # 20 Gbps at a 50 ps UI = 1.000  ->  binary, Nyquist 10 GHz
```

A PAM3 record on that bus has two eyes where the bus has one. In this library the level count is the
carrier *kind* and the sequence is its *pattern*; `physics._pattern_error` already refuses a
quaternary pattern name on a binary carrier and says which it is. A code belongs at stage 2 — before
symbols — and today has no op (§5).

### 3.4 Indexing a per-record column by row position

A per-record column is aligned to record id. It has nothing to do with iteration order. Filtering,
sorting, sharding or a partial rebuild reorders rows, and a positional read then attaches every
record's provenance to a different record. It fails silently and it looks like label noise.
**Rule:** join by id. **Detection:** shuffle the record order and re-run; a positional bug changes
the answer.

### 3.5 Measuring with the wrong instrument

Several of the errors described in this document were caused by the measuring instrument rather
than by the physics.

| measurement | why it lies | use instead |
|---|---|---|
| integer run lengths | quantised to one sample, so sub-sample displacement (DCD, pair skew, a fraction-of-a-UI shift) is invisible | interpolated threshold crossings (`eye.crossings`, `measure`) |
| mean step over an exponential | dominated by the settling tail, so it under-reads the edge | the 10–90 % (or the specified) threshold pair on the transition itself |
| a fold that assumes an integer pattern period | samples-per-UI is generally non-integer, so the fold smears | `Grid.pattern_period_samples`, realigned sub-sample |

**Rule:** calibrate on a known answer before trusting a measurement on the real one. Synthesise
with the knob set to an exact value, measure, confirm recovery, and only then point the instrument
at the record you care about. That is what `wfmsynth.validate` does for every primitive.

---

## 4. Folding, and how record length follows from it

```
n_ui    = repeats × pattern_period_ui
n       = n_ui × samples_per_ui          samples_per_ui = fs / baud
fs      ≥ k / tr                         (§3.1 — the edge sets fs, not the UI)
noise   ∝ 1 / √repeats
```

| repeats | noise reduction | note |
|---|---|---|
| 128 | 11.3× | the floor [JUDGEMENT — below this a fold measures one instance of the pattern rather than its statistics. Falsified when the measurement's stated uncertainty at 128 repeats is already smaller than the effect being labelled.] |
| 256 | 16× | the point of diminishing returns |
| 1024 | 32× | 4× the samples for 2× the reduction |

Worked, so any bus can be sized the same way:

| pattern | period (UI) | repeats | n_ui | samples/UI | n samples |
|---|---|---|---|---|---|
| PRBS7 | 127 | 256 | 32,512 | 16 | 520,192 |
| PRBS13Q (quaternary) | 8,191 | 256 | 2,096,896 | 16 | 33,550,336 |
| slow bus, 128-UI frame, `fs` from a 10 ns edge | 128 | 128 | 16,384 | 1,600 | 26,214,400 |
| PRBS31 | 2,147,483,647 | — | — | 16 | **3.4 × 10¹⁰ — impossible** |

Read the last two rows together. A slow bus with a fast edge is expensive in samples because of
the edge. The data rate is not the driver, and 128 repeats is often the affordable choice there.
**PRBS31 cannot be folded at all**: its period never repeats inside any realistic record, which is
why short, pattern-lockable stress patterns exist — they make a folded measurement possible on a
bus whose compliance sequence is unfoldable. A long PRBS is still the right stress, since it
carries the run lengths and low-frequency content that actually close an eye. It is not the right
thing to fold.

Choose the pattern for what it must show, and know what each costs:

| pattern | period | what it gives up |
|---|---|---|
| `clock` (1010…) | 2 UI | no runs, no pattern-dependent ISI — the deliberate contrast case |
| PRBS7 (`legacy`) | 127 UI | almost no low-frequency content, so a lossy channel renders **more open** than the real link |
| PRBS9…PRBS15 | 511…32,767 UI | the usable compromise: foldable and still ISI-bearing |
| PRBS31 / PRBS31Q | 2³¹−1 | unfoldable; no repeat-averaging |

An eye that stays open on `clock` but shuts on a long PRBS through the same channel points to ISI;
loss alone would close both. It is worth including as a diagnostic in any dataset.

**The generator polynomial is not a free parameter.** Several maximal-length polynomials exist at a
given order, and each gives a valid pseudo-random sequence with the right level statistics — that no
protocol analyser will pattern-lock to. A capture meant to be *analysable* has to carry the
standard's polynomial. `physics.PRBS_TAPS` holds only polynomials that are stated somewhere
(order 13 per IEEE 802.3 Clause 120.5.11.2.1; orders 11 and 23 per ITU-T O.152 and O.151), every one
of them asserted primitive by `wfmsynth.validate`, and declines any order whose standard polynomial
is not stated rather than guessing a tap set. Add an order by citing its polynomial. Do not add one
just because it happens to work.

---

## 5. State of the library

What the library carries, what it carries only partly, and what it does not carry yet. A dataset's
own state belongs with that dataset, in its provenance columns and its own documentation; this
section is about the tools.

### Done, in the library

| capability | where |
|---|---|
| Binary PRBS 7/9/11/13/15/23/31 from a tap table of **stated** polynomials, each asserted primitive, with a starting phase | `physics.PRBS_TAPS`, `physics.prbs` |
| Quaternary PRBS13Q / PRBS31Q, Gray-coded pairs, two PRBS repetitions because the period is odd | `physics.prbs13q`, `physics.prbs31q` (IEEE 802.3 Clause 120.5.11.2.1) |
| `clock` 1010 pattern; a pattern-name error that says *which* carrier a name belongs to | `physics.clock_pattern`, `physics._pattern_error` |
| NRZ, PAM4, PAM-N for any N; analog and arbitrary carriers | `physics.nrz`, `pam4`, `pam` |
| Open-drain line: the rise charges through the pull-up, so fall and rise are not mirrors; a second sink resolves the wired-AND as a real resistor divider | `bus`, `open_drain` op |
| Analog (`step`/`pulse`/`exp`/`chirp`/`two_tone`/`noise`) and unipolar `cmos` carrier kinds; a real capture from disk (`Signal.capture`, sha256 over sample values); a probe pack (compensation, ground-lead ring, termination, AC coupling, overload recovery); burst/idle; a pass-FET analog switch; AM/ASK/OOK/FM/FSK/PM modulation; a generic clamped-exponential rail event | `physics`, `instrument.probe`, `capture`, `compose` |
| Linear (non-circular) convolution; a rendered-and-discarded lead-in whose guard is measured from the chain's own impulse response | `compose`, `physics.response_extent` |
| Two-rate acquisition, interleaved ADC, ENOB as a noise level, stored integer codes | `acquire`, `instrument`, `digitize`/`store` |
| Role-tagged RNG streams; recipe round-trip; `sha256()` content address | `streams`, `compose` |

### Preliminary — usable, not yet verified

| item | status |
|---|---|
| **Line coding has no op.** It runs through helper functions (`coding.dc_balanced`, `scramble_64b66b`, `running_disparity`, `max_run`). Symbols are PRBS or uniform over the levels. | Levels and eye structure are correct. Symbol statistics are not modelled, and run lengths are where an engineer will spot it first. The code cannot appear in a recipe until it is an op. |
| Table-level locators in a parameter extract | A locator naming a table is weaker than it looks: an extract can carry a correct number under a wrong table reference. Cite the extract that carries the number, and verify the table reference separately before quoting it. |
| An extract no adversarial pass has reached | One unchecked reading. Mark it as such and keep the count of unreached extracts with the dataset. |
| Per-bus `fs`-from-edge values | The library supplies the sizing arithmetic (§4). The per-bus numbers are the caller's to state and to mark. |

### The pattern and level-coding contract

Pinned by `tests/test_patterns.py`, `tests/test_level_coding.py` and `tests/test_recipe_replay.py`.
This contract is the part worth knowing, because it decides what a dataset built on it can claim:

| decision | consequence |
|---|---|
| **Mechanisms in the library, standards knowledge in the caller.** A PRBS is a polynomial, a seed and a length — physics. *Which* pattern a standard names for a given lane rate changes with a document's revision and is the caller's to state, through a registration call. | The library never carries a fact that a revision can invalidate. |
| **A recipe records the pattern name and its resolved parameters.** Name alone is unreplayable without the registry; polynomial alone is unreadable. | A consumer with no registry entry still renders the record — through the generic mechanism the entry names, or through embedded symbols-as-data, which needs no code at all. |
| **A caller-supplied generator that is missing or has changed produces a named error** (`needs pattern 'x' @ <hash>`). | The failure mode is a refusal. It does not silently produce different samples under the same name. |
| **The general arbitrary-polynomial engine must reproduce `physics.prbs` bit-for-bit** on every order in `PRBS_TAPS`, and the block-repeat generator must reproduce `clock_pattern`. | This is the same known-answer check described in §3.5, applied to the arbitrary-polynomial engine. |
| **Levels are stated exactly, as literal values.** `np.linspace(-1, 1, 4)` puts the inner PAM4 levels 5.6e-17 off the thirds. | The level map is pinned in both directions — the explicit array agrees to 1 ulp, and linspace's disagreement was measured directly. |
| **Gray coding is defined by its effect on bit errors.** It is not implemented as a lookup table; precoding changes the error structure and nothing else. | Tested by counting bit errors through an adjacent-level slice (exactly one) and through a precoded symbol error (exactly two), each against the unbounded alternative as the control. |

### Remains

- [ ] A **`line_code` op** (8b/10b, 64b/66b + its self-synchronous scrambler, 128b/13xb framing,
      PAM4 Gray mapping / precoding) that lands in the recipe, with running-disparity and max-run
      assertions and a knob-sweep row. This closes the symbol-statistics gap described above.
- [ ] Standard **compliance/stress patterns** as named, pattern-lockable sequences alongside the
      PRBS family, so a folded measurement is possible on buses whose compliance sequence is not
      (the registry above is the mechanism; the per-standard entries are the caller's to state).
- [ ] Channels **derived from topology** rather than lumped loss-plus-reflection, for the buses that
      currently inherit another bus's channel shape.
- [ ] Ops for the failure modes that have none — e.g. data-to-strobe skew and mistrain, on-die
      termination switching, bus turnaround, reference-voltage drift. Records for such a bus are
      structurally incomplete. This is more than imprecision, and the record should say so.
- [ ] Unipolar open-drain levels (the fall/rise asymmetry is modelled and measured; the absolute DC
      level is not).

---

## 6. Consuming the dataset: filter by evidence

The dataset carries, per record: an id, the waveform, a record-level fidelity flag, and
**per-field markers**. The markers are the thing to filter on. A record-level flag says only that
the fields are *visible*, never that they are all cited.

```python
# --- SAFE -------------------------------------------------------------------
# Per-record columns keyed by record id (§3.4), never by row position.
#   fidelity[rec_id]   -> "cited" | "provisional"
#   provenance[rec_id] -> "tr_ps=CITED,rj_ps=JUDGEMENT,ssc_ppm=NOT_SPECIFIED,..."

BACKED  = {"CITED", "DERIVED", "MEASURED"}          # a document or a measurement stands behind it
UNBACKED = {"JUDGEMENT", "PRELIMINARY", "NOT_SPECIFIED", "NOT_FOUND"}

def markers(provenance_cell):
    """'a=CITED,b=JUDGEMENT' -> {'a': 'CITED', 'b': 'JUDGEMENT'}"""
    out = {}
    for pair in filter(None, (p.strip() for p in provenance_cell.split(","))):
        field, _, marker = pair.partition("=")
        out[field.strip()] = marker.strip().upper()
    return out

def usable(rec_id, provenance, required_fields):
    """True only if every field this experiment depends on is backed by evidence.

    A missing field is unusable: absence of a marker is not a marker. An unrecognised
    marker is unusable too, so adding a new one fails loudly instead of quietly widening
    the filter.
    """
    m = markers(provenance[rec_id])
    return all(m.get(f) in BACKED for f in required_fields)

fields = ("tr_ps", "rj_ps", "loss_db")               # what your label depends on
train = [r for r in record_ids if usable(r, provenance, fields)]

# Report what you dropped and why, so the filter can be checked.
from collections import Counter
dropped = Counter(markers(provenance[r]).get(f, "MISSING")
                  for r in record_ids for f in fields
                  if markers(provenance[r]).get(f) not in BACKED)
print(len(train), "of", len(record_ids), "records;", "dropped by marker:", dict(dropped))


# --- UNSAFE ------------------------------------------------------------------
train = [r for r in record_ids if fidelity[r] == "cited"]     # (1)
prov  = provenance_column[i]                                  # (2)
if "CITED" in prov: ...                                       # (3)
# (1) record-level: the flag says only that the fields are visible; whether they are cited
#     is separate. A record whose rj_ps is a JUDGEMENT and whose tr_ps is CITED passes this.
# (2) positional: the column is aligned to record id. One filter or reshard upstream and
#     every record now wears another record's provenance.
# (3) substring: "CITED" is inside "NOT_CITED"; a cited neighbour field satisfies it; and a
#     field that is simply absent satisfies nothing and is silently treated as fine.
```

A per-field marker also tells you what a *label* is worth. A label derived from a `JUDGEMENT`
parameter is a judgement, whatever the digest says about the samples. The digest proves the record
matches the recipe. It does not prove the recipe matches the bus.

---

## 7. If you extend this

1. Write the test first. Watch it fail. Then implement.
2. Calibrate the measurement on a known answer before trusting it on the real one.
3. Make it an op if a consumer will ever select on it; emit only the arguments you were given.
4. Satisfy both gates in §1 — the knob sweep and the lead-in classifier — and let neither default.
5. Run the whole of `tests/`, not the file you touched.
6. Mark every new number with how you know it, and let someone else try to refute the citation.
