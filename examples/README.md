# wfmsynth examples

These examples form a learning path. You do not need to understand signal integrity before
starting; each step introduces only the concepts needed for the next one.

Run commands from the repository root after installing the project:

```bash
python -m pip install -e ".[test]"
python examples/quickstart.py
```

## Recommended order

### 1. Start with a signal path

[`quickstart.py`](quickstart.py) builds an NRZ source, passes it through a bandwidth-limited
channel, models a scope acquisition, and shows how the same recipe reproduces exactly.

Learn:

- a waveform is a NumPy array of voltage samples;
- `Grid` binds those samples to real time and symbol-rate units;
- a `Signal` chain follows the physical order of the system; and
- `recipe()` is the exact provenance for an ML example.

### 2. See what a realistic system contains

[`realistic_scenario.py`](realistic_scenario.py) composes:

- an electrical PAM4 link;
- an optical PAM4 lane; and
- a two-lane scene with shared power noise and crosstalk.

This is the end-to-end reference, not the first API tutorial. Read the comments stage by
stage and remove one operation at a time to see what changes.

### 3. Build trustworthy ML datasets

Run these in order:

1. [`provenance.py`](provenance.py) — store exact generation recipes and make controlled
   contrastive pairs.
2. [`ground_truth.py`](ground_truth.py) — measure labels from the generated output instead
   of assuming requested knobs equal realized behavior.
3. [`confounder_sweep.py`](confounder_sweep.py) — prevent an ML model from learning an easy
   correlated shortcut.
4. [`sim_to_real.py`](sim_to_real.py) — identify which simple feature distinguishes a
   synthetic set from measured captures.
5. [`events.py`](events.py) — plant rare localized defects in a long record and join
   requested plus measured labels onto ±1 UI windows (your own clock recovery can
   supply the instants).

The distinction that matters:

- **Recipe values** say what was requested.
- **Measured labels** say what the complete pipeline produced.
- **Controlled pairs** change one factor while preserving the rest.
- **Confounder control** prevents a target factor from being predictable through an
  unintended correlated feature.

### 4. Model the acquisition explicitly

[`two_rate_acquisition.py`](two_rate_acquisition.py) separates:

1. fine-grid physical simulation;
2. analog input bandwidth;
3. sampling onto an instrument-rate grid;
4. ADC noise and quantization; and
5. stored-record decimation.

Use this pattern when training data should resemble what a scope or digitizer stores rather
than an ideal simulated node.

### 5. Hand a dataset over, and let the receiver rebuild it

[`replay.py`](replay.py) is the one example that is also a TOOL. It reads a recipe document (a
mapping of record id to `{"ops": [...], "grid": {...}}`), rebuilds every record by executing
those ops on the declared grid, and checks each one against the digest the document carries:

```bash
python examples/replay.py RECIPE.json OUT_DIR        # render + verify
python examples/replay.py RECIPE.json --dry-run      # verify only, write nothing
python examples/replay.py RECIPE.json OUT_DIR --only link/tp2/r0001
```

It imports this library and the standard library and nothing else, which is what makes a
handed-over archive self-sufficient: the recipient needs the document, this file, and this
library at the commit the document pins — no build script, no authoring layer. It exits
non-zero on any mismatch **and on having verified nothing**, so a document that carries no
digests cannot pass by accident. Its docstring is also the shortest tour of the chain
(bits → coding → symbols → levels → waveform → instrument) and of which op sits where.

### 6. Build a protocol layer on top

[`i2c_bus.py`](i2c_bus.py) is the one example that is mostly *not* this library. It shows the
shape of the layer you write above it:

```
bytes --> [ your protocol encoder ] --> holds --> [ scenario ] --> events --> [ net ] --> volts
                 ~40 lines, yours                    library            library
```

An I2C host writes a byte to a 1.8 V target through a MOSFET level translator. The encoder
knows start conditions and acknowledge slots and nothing about electricity; the agents know
about output stages and nothing about I2C. Neither knows about the other, so the same encoder
drives a different board by editing a dict, and the same board runs a different bus by swapping
the encoder.

It is a worked example rather than a module on purpose. Which bits a standard puts on a wire
has a revision date and a document behind it, so it belongs in your code — the same boundary
the pattern registry draws. What the library provides is the hooks: a shared net, a scenario
compiler, and holds that are simultaneously the stimulus and the labels.

[`multi_driver_bus.py`](multi_driver_bus.py) is the other half of that: not who talks, but
*what each talker is*. Three transceivers of different supplies, output impedances and edge
rates take turns on one terminated backplane bus, each driving through its **own** chain —

```python
drivers=[{"id": "ctrl", "node": "bus", "ohm": 20.0, "state": "hiz",
          "source": {"ops": Signal(grid=g).carrier("cmos", v_hi=3.3, tr_s=2e-9)
                                          .timing(rj_ps=12.0),
                     "volts": True}}, ...]
```

— so the net is the first op in the chain and nothing has to arrive at it. Every measured level
and edge lands on its closed form to ~0.2 %, the weakest part misses the receiver's V_IH by
346 mV because a terminated bus penalises it twice, and the bus returns to its bias *exactly*
at every hand-over, against the −1.765 V of drift a by-hand sum of the three contributions
would have accumulated. That last number is the reason the sources belong inside the solve.

[`inductive_handover.py`](inductive_handover.py) is the third: what the same hand-over does once
the **package inductance** is in the net. An R-C net can only settle; a real bus overshoots and
rings, and `{"kind": "l", "henry": 8e-9, "ohm": 0.3}` is the reason —

```python
elements=[{"id": "lp", "kind": "l", "a": "pad", "b": "bus", "henry": 8e-9, "ohm": 0.3},
          {"id": "cp", "kind": "c", "a": "pad", "b": "gnd", "farad": 2e-12}, ...]
```

One 3.3 V transceiver with a 7 Ω output stage drives a 20 pF bus to **+4.396 V** and
**−1.096 V**, 796 mV past both of the receiver's absolute maxima, while the R-C model of the
identical board sits 412 mV *inside* both: the two records differ by 1.758 V, 57 % of the swing.
Letting go while carrying 15.4 mA kicks the pad 0.980 V in 195 ps — a current is a state and
cannot stop — and 12 Ω of series damping is the whole fix, for 131 ps of fall time. The
inductance is also what makes a line reachable as a lumped ladder (`net.line_ladder`), sized by
a measured rule rather than a rule of thumb.

### 7. Compose chains that meet

A chain is a line with one source; a capture is not. Take these in order, and stop when you
have what you need. The cookbook chapter
[Composing a realistic capture](../docs/COOKBOOK.md#composing-a-realistic-capture) is the text
that goes with them.

| Example | Use it when |
|---|---|
| [`sub_chains.py`](sub_chains.py) | An op's second signal has a chain of its own: an aggressor after its own channel, a filtered message. Start here |
| [`joins.py`](joins.py) | A second signal is added, multiplies, gates or delays the record: hum on a sensor, gain on a supply ripple, an enable, a Doppler delay |
| [`sequence_and_switch.py`](sequence_and_switch.py) | The transmission has parts in time (idle, preamble, payload) or a signal selects between sources, and you need per-sample labels of which |
| [`clock_and_data.py`](clock_and_data.py) | Several channels must be correlated the way one PLL, one supply or one bit stream makes them, from one recipe |
| [`law_and_filter.py`](law_and_filter.py) | The shape has no named op: a comparator with hysteresis, an anti-alias filter, with the delay accounted for |
| [`triggered_capture.py`](triggered_capture.py) | The record is what a scope's trigger kept: an edge or pulse-width trigger, segmented memory, holdoff, averaging |
| [`unit_variation.py`](unit_variation.py) | Parts have tolerances and a model must generalise to units it has not seen; shows the leakage a random split hides |
| [`composed_capture.py`](composed_capture.py) | You want to see all of the above on one two-channel triggered capture, and each cause of its jitter isolated by changing one named thing |
| [`state_dwell.py`](state_dwell.py) | A level dwells and jumps at random times: burst noise, an idle/active schedule for a gate or a switch, 1/f noise as a sum of such processes |
| [`clutter.py`](clutter.py) | A clean record is too clean: mains hum, converter ripple, pickup and 1/f from `ws.recipes`, and what each does to a detector's false alarms |

### 8. I/Q and RF

| Example | Use it when |
|---|---|
| [`iq_passband.py`](iq_passband.py) | The signal is a modulated carrier: a complex-baseband record, up to a passband voltage, through a scope, back to symbols. Start here |
| [`iq_impairments.py`](iq_impairments.py) | A transmitter and its oscillator are not ideal: an EVM budget predicted from the chain's own report against the measured EVM |
| [`rf_measures.py`](rf_measures.py) | You need ACLR, PAPR and EVM against the transmitted symbols, with an amplifier compressing the passband voltage |
| [`iq_recordings.py`](iq_recordings.py) | I/Q has to come from or go to a file: SigMF with annotations, raw interleaved samples with a marker channel, an embedded recording |
| [`ofdm_frame.py`](ofdm_frame.py) | The signal is multicarrier: an OFDM frame (preamble, payload, gap) through an echo and noise, with EVM per subcarrier and per symbol against the resource grid |
| [`fading_multipath.py`](fading_multipath.py) | The channel is stated at RF and rendered on the envelope: an amplifier's AM/PM, a cable, multipath and Doppler fading at 2.4 GHz on a 4 MSa/s grid |

### 9. Photonics

| Example | Use it when |
|---|---|
| [`optical_field.py`](optical_field.py) | The record is light between `eo` and `photodetect`: a coupler, an attenuator, an optical filter, two polarisations, and what is refused on a field. Start here |
| [`swept_ring.py`](swept_ring.py) | A tunable laser sweeps a ring with a reference interferometer beside it: Q and extinction across dies, and what a stray reflection, a fast sweep and a mode hop do to the reading |

### 10. Specialist topics

| Example | Use it when |
|---|---|
| [`non_integer_sps.py`](non_integer_sps.py) | Sample and symbol clocks are unrelated, as they normally are in hardware |
| [`clock_recovery.py`](clock_recovery.py) | The receiver or scope CDR changes which jitter remains visible |
| [`touchstone_channel.py`](touchstone_channel.py) | You have measured `.sNp` channel data or need resonances absent from an analytic model |
| [`events.py`](events.py) | Rare localized defects (runt, glitch, ring, droop) in a long record, labelled per UI window |
| [`replay.py`](replay.py) | Rebuilding and verifying a set of records from recipes alone, as a consumer of a handed-over archive |
| [`i2c_bus.py`](i2c_bus.py) | Several parties share one net — a bus, a backplane, any hand-over — and you need a protocol layer above it |
| [`multi_driver_bus.py`](multi_driver_bus.py) | The parties on a shared net are different parts — own supply, own output impedance, own edge rate, own jitter — and must not inherit each other's levels |
| [`inductive_handover.py`](inductive_handover.py) | A hand-over has to ring: package inductance, overshoot past a receiver's absolute maxima, and what a series damping resistor costs |
| [`analog_instrument.py`](analog_instrument.py) | The source is not a serial link: a bench signal (`carrier("sine"/"cmos"/...)`) or a real capture from disk (`Signal.capture`), taking the same later ops |

## Choosing a dataset generator

| Need | Recommended entry |
|---|---|
| Reproducible realistic links with detailed labels | `Signal` + `Grid` + `dataset()` |
| Existing array plus one named labelled fault | `apply_impairment()` |
| Label-preserving variation across capture conditions | `domain_randomize()` |
| Broad, protocol-agnostic morphology pretraining | `generate()` |
| A ready-made segmented PAM4 defect benchmark | `deep_capture()` |
| Rare needles in a long record, labelled per window | `Signal.events()` + `label_windows()` |
| Several parties sharing one net, labelled by who drove | `scenario.Hold`/`Agent` + `Signal.net()` |

`deep_capture()` and `generate()` are specialized dataset presets. For a new system model,
prefer a `Signal` recipe so the physical stages and their ordering are explicit.

## Practical realism checklist

Before calling synthetic data “realistic,” check:

- Are sample rate, symbol rate, and record length plausible for the target instrument?
- Is the source causal, and are timing effects applied at the source?
- Does the signal include ordinary nominal imperfections, not only labelled faults?
- Are channel, receiver, scope/probe, and ADC stages in physical order?
- Are nuisance ranges smaller than the labelled fault ranges?
- Are labels measured from the final output where possible?
- Can `separability()` distinguish synthetic from measured data using an obvious shortcut?

For definitions of UI, ISI, jitter, equalizers, ENOB, and other terms, see the glossary in
the root [`README.md`](../README.md).
