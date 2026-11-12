# wfmsynth cookbook

One recipe per capability, in the order the [README](../README.md)'s capability map introduces
them. Each section is self-contained and runnable; none of them is required reading before the
first waveform.

Two chapters at the end are read in order rather than looked up:
[Composing a realistic capture](#composing-a-realistic-capture), for chains that meet (a chain
as a parameter, several channels, a trigger, unit-to-unit variation), and
[I/Q: complex baseband and RF](#iq-complex-baseband-and-rf).

Start with the [README](../README.md) for orientation — install, the first waveform, the two
numbers that decide whether a record means anything, and the design principles. Come here to
look something up.

See also: [FIDELITY.md](FIDELITY.md) for which numbers are calibrated against hardware,
[COMPATIBILITY.md](COMPATIBILITY.md) for what will not break, and
[DATASET-METHODOLOGY.md](DATASET-METHODOLOGY.md) for sourcing parameters defensibly.

---

## Instrument / ADC model
The ADC stages compose in one place, in the physically correct order — getting it wrong
is silent (quantising before the noise gives the wrong noise floor).

```python
from wfmsynth import digitize_adc, quantize_adc
y, info = digitize_adc(x, noise_floor={"rms":1e-3,"shape":"pink"},
                       interleave={"m_cores":4,"offset_v":1e-3},   # offset in absolute volts
                       clip_full_scale=0.7, bits=10)               # -> noise->interleave->clip->quantise
q = quantize_adc(x, bits=10, full_scale=0.423)   # the converter's real lattice
```

### Quantisation and ENOB are two mechanisms — model them apart

**Quantisation** is the converter's bit depth: a uniform lattice, `q = 2·FS/2**bits`, discrete.
**ENOB** is a SINAD figure: noise *and* distortion, continuous. Collapsing the second into the
first — rounding to a `2**enob` lattice — gets the noise power roughly right and the record's
structure entirely wrong.

```python
from wfmsynth import converter_noise_rms
sigma = converter_noise_rms(enob=5.0, full_scale=0.423,     # published at ...
                            bandwidth_hz=110e9,             # ... this selected bandwidth
                            nyquist_hz=128e9, bits=10)      # -> 8.23 mV rms, the converter's own
```

One fixed converter floor, shaped only by the selected-bandwidth filter, renders **all thirteen**
published bandwidth/ENOB settings (10→110 GHz, 7.0→5.0 bits) to **−0.29…+0.09 bits**.
That is the evidence the converter is fixed and ENOB is emergent. Two consequences:

* the 10-bit lattice sits **9.96 LSB below** the noise that sets ENOB — 0.084 % of the power, so
  it is fully dithered and invisible in a histogram. `quantize_adc(enob=5.9)` instead builds a
  lattice **17× coarser** than the real converter's, and after the DSP filter a real DSO runs
  *after* its converter it measures **7.81** bits, not the 5.9 it was handed.
* the record's visible lattice is not the converter's at all — it is the **terminal store**
  (`.store()`), and the real geometry is measurable: the three real oscilloscope exports carry stride-8
  int16 codes with the screen at ±8192, i.e. **2^11 = 2048 steps across the vertical**, of which
  the records span 1851–2038.

The honest chain, in order — analog front end, converter (its own noise, then its own lattice),
the DSP bandwidth filter, then the export:

```python
(sig.scope(bw_hz=110e9)                                     # analog front end (Bessel)
    .digitize(noise_rms=sigma, bits=10, full_scale=0.423)   # converter: two mechanisms, apart
    .scope(bw_hz=32e9, kind="brickwall")                    # the DSP filter, AFTER the converter
    .store(bits=11, dither_lsb=1/12**0.5))                  # int16 codes: the record's real floor
```

## Impairment mixing at constant power
Separate impairment *magnitude* from *character* — "same SNR, different noise character":

```python
from wfmsynth import mix_at_constant_power
y = mix_at_constant_power([white, pink], weights=[1-a, a], total_rms=0.05)
# total RMS is exactly 0.05 for any a; only the spectral character changes with a
```

## Intermittent impairments (with defect masks)
Intermittency is where stationary analysis fails — apply a defect on a duty cycle and
emit the per-sample mask as ground truth:

```python
from wfmsynth import apply_gated
y, mask = apply_gated(x, glitch_fn, intervals=[(2000,180),(6000,140)])
# y is bit-identical to x outside the gate; mask marks where the defect is active
```

## Localized events (needles in a long record)
Systemic stages (loss, reflections, ADC) apply to every sample. A runt bit, an
edge-excited ring, a PDN sag on a long run, or an asynchronous glitch has **finite
time support**. Placement (when) is independent of mechanism (what):

```python
sig = (ws.Signal(seed=1, grid=g).carrier("nrz", n_ui=n_ui, causal=True)
       .events("runt", on="symbols", count=3, severity=0.6)      # data-locked
       .events("ring", on="edges", which="rising", count=2,      # transition-locked
               f0_hz=220e6, tau_s=12e-9)
       .events("droop", on="pattern", min_run=8, severity=0.5)   # run-locked
       .events("glitch", on="poisson", rate_hz=1e4, severity=0.4))  # not bus-locked
x, events = sig.realize()   # samples + [{sample, ui, t_s, kind, severity}, ...]
```

`on` is the targeting policy: `symbols`, `edges`, `pattern`, `aggressor` (another
lane's edges), `intervals`, `poisson`, `times`. `kind` is the mechanism: `runt`,
`glitch`, `ring`, `overshoot`, `undershoot`, `nonmonotonic`, `droop`, `slow_edge`.

Eye-mask failures and height at a sampling instant are **measured labels**, not
mechanisms — many causes produce them:

```python
windows = ws.nominal_ui_windows(len(x), g, half_ui=1.0)   # ±1 UI on the known baud
rows = ws.label_windows(windows, events, x=x, eye_low=0.12)
# each row: requested event kinds + measured height / slope reversals / eye_violation
```

This module does not recover a clock or cut segments. Pass recovered instants to
`windows_from_centers` and the same join still works. Systemic effects that were
never placed as events (a resonant stub that rings only on some patterns) still
appear in `measured` when they trip a flag. See [`examples/events.py`](../examples/events.py).

## Tx FFE (pre-emphasis)
Real transmitters run multi-tap FFE — a deliberate pre-cursor de-emphasizing ISI:

```python
sig = (ws.Signal(seed=1, grid=g).carrier("pam4", n_ui=n_ui, pattern="prbs13q", causal=True)
       .tx_ffe(taps=[-0.15, 1.0, -0.25], pre=1)      # place before the channel
       .lossy(loss_db=8.0, loss_at_ghz=25.0, causal=True))
```

## Full RX equalization (CTLE / RX FFE / DFE)
The receiver side is composable end-to-end — equalize before slicing, just like real hardware:

```python
sig = (ws.Signal(seed=1, grid=g).carrier("pam4", n_ui=n_ui, pattern="prbs13q", causal=True)
       .tx_ffe([1.0, -0.2]).lossy(loss_db=10, loss_at_ghz=25, causal=True)
       .ctle(fz_ghz=2, fp1_ghz=5, fp2_ghz=10)   # analog HF peaking
       .rx_ffe([1.0, -0.3, 0.1], spacing_ui=0.5) # fractionally-spaced FIR
       .dfe([0.25, 0.1])                          # cancel post-cursors (taps = channel post-cursors)
       .digitize(snr_db=40))
```

## Optical E/O/E link (electrical → optical → electrical)
A true optical link on a **complex field** — so laser chirp and fibre dispersion interact correctly,
and square-law detection closes the loop back to electrical:

```python
det = (ws.Signal(seed=1, grid=g).carrier("pam4", n_ui=n_ui, pattern="prbs13q", causal=True)
       .eo(kind="dml", alpha=3.0, er_db=8)        # E→O: laser + chirp → complex optical field
       .fiber(length_km=4, D_ps_nm_km=17)         # physical β2·L dispersion + attenuation
       .edfa(gain_db=12, nf_db=5)                 # optical amp + ASE (optional)
       .photodetect(responsivity=1.0)             # O→E: i = R·|E|²  (square-law, closes the loop)
       .tia(gain=1.0, bw_hz=50e9)                 # transimpedance front end
       .digitize(snr_db=40))
```
`eo(kind="mzm", ...)` gives a Mach-Zehnder cos² modulator; `optical`/`dispersion` remain for the
simpler intensity-only model. Because the field carries phase, DML chirp through fibre dispersion
distorts the detected eye — the interaction an intensity-only model can't show.

## Causality hazard (build causal chains end-to-end)
Each stage can be causal in isolation while the *pipeline* is not: the default zero-phase
edge shaping reintroduces pre-cursor **after** a causal channel. For an end-to-end causal
chain, set `causal=True` on the carrier (and any lossy stage):

```python
ws.Signal(seed=1, grid=g).carrier("pam4", n_ui=n_ui, pattern="prbs13q", causal=True) \
  .lossy(loss_db=8.0, loss_at_ghz=25.0, causal=True)   # causal shaping AND causal channel
```

## Clock recovery (what the scope records)
A scope folds the record by a recovered clock; the CDR tracks out jitter below its loop
bandwidth. The recorded eye depends on the recovery:

```python
clock, residual = ws.recover_clock(jitter_phase, baud=50e9, loop_bw=1e6, order=2)
# residual is the timing jitter the eye actually shows (low-freq wander tracked out)
ws.tracked_out_fraction(jitter_phase, baud=50e9, loop_bw=1e6)   # 0..1
```

## The recorded eye (folded on the clock recovered from the record)
`recover_clock` is the loop; `eye_density` is the picture that loop makes. It finds the
threshold crossings, fits the symbol period and phase to them, runs the CDR over the
per-symbol timing phase, and folds the record at the decision instants that come out —
so the period comes from the **edges**, not from the rate the grid states, and a timing
impairment shows up instead of being assumed away.

```python
img, meta = ws.eye_density(x, grid, w=128, h=96, ui_span=2.0, max_traces=400)
img.shape                       # (96, 128) uint32 counts, row 0 = highest volts
meta["clock"]["samplesPerUi"]   # the period the EDGES determined
meta["clock"]["residualRmsUi"]  # the jitter this picture therefore shows
meta["clock"]["source"]         # 'recovered', or 'nominal' + a stated reason
```

Tell the grid a symbol rate 1.7x or 0.5x the truth and the density is byte-identical:
the picture is folded on what was measured. Built for long records — 3 M UI (48 Mpt) is
152 ms and 1.2 GB peak, against 555 ms and 3.0 GB for the straightforward transcription
of the same arithmetic, with every count identical. `spikes/eye3m/` has the measurements,
including which techniques did nothing. Threads are used where the work partitions
(`threads=1` disables them; Pyodide gets one automatically) and never change a count.

## Measured S-parameter channels (Touchstone)
Drive synthesis through a real `.sNp` channel — resonances the analytic model can't make:

```python
f, S = ws.read_touchstone("thru.s2p")                     # (nf, n, n) complex
sig = ws.Signal(seed=1, grid=g).carrier("pam4", n_ui=n_ui, pattern="prbs13q", causal=True) \
        .sparam(path="thru.s2p")                          # apply S21 as the channel
```

## Resonant reflections
Real discontinuities resonate — frequency-dependent Γ, not a flat mirror:

```python
ws.Signal(seed=1, grid=g).carrier("pam4", n_ui=n_ui, pattern="prbs13q", causal=True) \
  .resonant_reflect(td_ps=40.0, f0_ghz=30.0, q=10.0, gamma0=0.4)   # |Γ| peaks at f0
```

## Nominal imperfections (no class is perfect)
Real transmitters are imperfect always — a too-perfect nominal class is a giveaway:

```python
ws.Signal(seed=1, grid=g).carrier("pam4", n_ui=n_ui, pattern="prbs13q", causal=True) \
  .nonlinearity(compression=0.04, level_noise=0.005, rise_fall_ratio=1.3)
```

## Multiple aggressors (asynchronous by default)
Crosstalk from a coupling vector; aggressors run at offset bauds so they are not clock-
locked to the victim (a synchronous default makes crosstalk artificially easy to detect):

```python
ws.Signal(seed=1, grid=g).carrier("pam4", n_ui=n_ui, pattern="prbs13q", causal=True) \
  .crosstalk_matrix(couplings=[0.12, 0.08, 0.05])   # async by default; synchronous=True to lock
```

By default `crosstalk_matrix` **manufactures** its aggressors — one `nrz` per coupling, at a
seed and a baud offset. That is right when the neighbours are unknown traffic. When they are the
other lanes of a *known* link, pass them as chains, one per coupling:
`crosstalk_matrix(couplings=[0.08, 0.03], aggressors=[lane_a, lane_b])` — see
[A chain as a parameter](#a-chain-as-a-parameter).

## One observed channel of a four-pair link (4D-PAM5 / 8B1Q4)
A probe on one pair of a live four-pair link does not see the wanted signal. It sees the **sum**:
the far-end transmitter of its own pair, its own transmitter leaking back through the hybrid's
finite isolation, near-end crosstalk from the three local lanes and far-end crosstalk from the
three remote ones. Storing that as one channel is not an approximation of the measurement — it is
what the measurement is. What a single stored channel loses is the ability to *correlate* an
aggressor with the victim, and a model trained on a single-pair capture is in the probe's
position anyway.

```python
g = ws.Grid(fs=2e9, baud=125e6, n=1 << 16)            # IEEE 802.3 Clause 40: 125 MBd per pair
ws.Signal(seed=1, grid=g).carrier("pam5", n_ui=4096, pattern="8b1q4", causal=True) \
  .multipair(**ws.CLAUSE_40_BUDGET, echo_td_ps=200.0)  # echo + NEXT + FEXT, one channel
```

The carrier is the real line code, not a uniform PAM5 draw: 8 bits become one **quartet** of
quinary symbols (one per pair), the level marginal is `{21, 28, 30, 28, 21}/128` rather than 1/5
each, and the transmit partial-response filter `3/4 + 1/4 z⁻¹` turns five transmit levels into the
seventeen a probe at the MDI sees. `pattern="test_mode_1"` … `"test_mode_4"` are the transmitter
test-mode symbol sequences of Clause 40.6.1.1.2. `wfmsynth.quinary`'s module docstring states
exactly which of that is cited and which is PRELIMINARY.

Omit or `inf` any of the three budgets and that mechanism is gone — with all three off the op
returns the wanted samples bit-for-bit, so an ablation over the budget has a real zero.

## Noise beyond white Gaussian
Heavy tails and 1/f structure — real noise is not one flat Gaussian:

```python
n = ws.realistic_noise(1<<16, rms=1e-3, df=5, pink_frac=0.3)   # heavy tails + some 1/f
```

## Streaming (deep-memory records)
Apply a channel to a multi-megapoint record in bounded memory (overlap-save):

```python
h = ws.channel_fir(lambda a: ws.physics.lossy_channel(a, length_in=8, causal=True), n_taps=512)
for block in ws.stream_blocks(x, h, chunk=1<<16):   # never a full-length FFT/output
    write(block)
```

## Sim-to-real separability harness
Ask whether a trivial classifier can tell synthetic from real — and which feature it uses
(that names the missing physics):

```python
rep = ws.separability(synthetic_set, real_set, grid)
rep["best_feature"], rep["best_auc"]   # e.g. ("hf_fraction", 0.98) -> HF content is wrong
rep["auc"]                             # AUC per feature = the full diagnosis
```

## Critical sampling (a non-obvious hazard)
If the modelled front-end bandwidth sits at or near the **Nyquist frequency** of the sample
rate, the capture is *critically sampled* and any analysis needing sub-sample interpolation
becomes lossy. The case that bites: pattern-averaging must realign repetitions onto a common
grid, and with non-integer samples-per-UI the period is fractional. Sub-sample realignment is
exact only well below Nyquist, so near critical sampling the deterministic part fails to
cancel and the residual is dominated by leftover **signal** rather than by the impairment you
were trying to isolate — which reads as a mysteriously large noise floor rather than as an
aliasing problem. Keep the front-end bandwidth comfortably below Nyquist (oversample), or
treat near-Nyquist residuals with suspicion.

## Receiver equalization (CTLE + DFE)
Real receivers equalize before slicing — CTLE (analog peaking) and DFE (decision feedback):

```python
sig = ws.Signal(seed=1, grid=g).carrier("pam4", n_ui=nui, pattern="prbs13q", causal=True) \
        .lossy(loss_db=9, loss_at_ghz=25, causal=True).ctle(fz_ghz=6, fp1_ghz=22, fp2_ghz=45)
eq, decisions = ws.dfe(per_symbol_samples, taps=[0.35])   # cancel a post-cursor
```

## Spread-spectrum clocking (SSC)
Triangular clock-frequency modulation (ubiquitous for EMI) — spreads the spectrum and adds
low-frequency wander a CDR must track:

```python
ws.Signal(seed=1, grid=g).carrier("pam4", n_ui=nui, pattern="prbs13q", causal=True) \
  .ssc(f_ssc=32e3, spread=0.005, profile="down")
```

## Differential-pair non-idealities
Split to P/N with intra-pair skew, common-mode and mode conversion (real links are differential):

```python
p, n = ws.differential_pair(x, grid=g, skew_ps=6.0, gain_imbalance=0.05)
diff, cm = ws.differential_mode(p, n), ws.common_mode(p, n)   # skew closes diff eye, makes cm
```

## Power-supply / PDN coupling
A supply rail coupling onto the signal as correlated amplitude modulation + supply-induced
jitter (a rich root-cause target that downstream tools compose):

```python
ws.Signal(seed=1, grid=g).carrier("pam4", n_ui=nui, pattern="prbs13q", causal=True) \
  .supply_coupling(f_ripple_hz=1e6, am_depth=0.03, psij_ps=2.0)   # AM + PSIJ from one rail
```

## Timing-modulation source (composable jitter)
Compose any mix of clock effects into one phase and inject it into a carrier:

```python
ws.Signal(seed=1, grid=g).carrier("pam4", n_ui=nui, pattern="prbs13q", causal=True) \
  .timing(ssc=dict(f_ssc=32e3, spread=0.005), pj=dict(amp_ps=2, f_hz=5e9), rj_ps=0.3)
```

## Multi-signal scenes
Compose several correlated lanes — a shared supply rail, lane-to-lane crosstalk, diff pairs.
`Scene` works on arrays you already hold; when the lanes are chains that should replay from one
recipe, [`ws.Channels`](#several-channels-one-record) is the canonical form.

```python
sc = (ws.Scene(g).add("lane0", w0).add("lane1", w1)
      .shared_supply(f_ripple_hz=1e6, am_depth=0.03)      # one rail, correlated across lanes
      .couple(into="lane1", frm="lane0", coupling=0.08))  # arbitrary lane-to-lane crosstalk
```

## Optical-link primitives
Optical intensity with extinction ratio, RIN, shot noise, and chromatic dispersion:

```python
ws.Signal(seed=1, grid=g).carrier("pam4", n_ui=nui, pattern="prbs13q", causal=True) \
  .optical(er_db=6.0, rin_db_per_hz=-135, shot=True).dispersion(strength=10.0)
```

## Acquisition chain (scope / probe)
Model the instrument beyond the ADC — band-limited front end and timebase jitter:

```python
ws.Signal(seed=1, grid=g).carrier("pam4", n_ui=nui, pattern="prbs13q", causal=True) \
  .scope(bw_hz=33e9).timebase(rms_ps=0.5)   # scope bandwidth + sample-clock jitter
```

## Analog, CMOS, and captured sources
A chain does not have to start with a digital carrier. Analog kinds (`sine`, `square`,
`triangle`, `sawtooth`, `dc`, `step`, `pulse`, `exp`, `chirp`, `two_tone`, `noise`) and the
unipolar `cmos` kind take the same later ops a serial carrier does, and `capture()` starts a
chain from a real file instead of a synthesized one:

```python
g = ws.Grid(fs=10e9, n=4096, v_full=3.3)   # k = tr_s * fs = 20, the k >= 8 rule again
clock = ws.Signal(seed=1, grid=g) \
  .carrier("cmos", v_lo=0.0, v_hi=3.3, duty=0.5, f_hz=10e6, tr_s=2e-9) \
  .probe(r_source=200.0, c_load_f=5e-12).waveform()   # real volts; a loaded-output RC pole

fs_cap = 5e9
captured = ws.Signal(seed=1, grid=ws.Grid(fs=fs_cap, n=4096)) \
  .capture(path="scope.npy", fs_hz=fs_cap) \
  .lossy(loss_db=3.0, loss_at_ghz=1.0).waveform()     # recipe carries a sha256 over the values
```

`cmos` stays in real volts (0 V is 0, not stretched onto -1); `capture`'s digest is over the
sample values, not the file's bytes, so a missing or changed file is a named error rather than
zeros or a silently different waveform. Pass a `grid=Grid(fs=...)` matching the file's own rate
— `fs_hz=` alone is provenance and does not make a downstream Hz-denominated knob like
`loss_at_ghz` mean anything real. `modulate()` (AM/ASK/OOK/FM/FSK/PM, message a nested
carrier spec) and `events("clamped_exp", ...)` (a generic rail transient) round out the set —
see `.claude/skills/wfmsynth/REFERENCE.md` for the full worked example, including the wired-AND
`open_drain(second=...)` sink and the `pass_fet()` analog switch.

## Low-speed buses
Embedded-bus signaling (open-drain wired-AND, UART framing):

```python
bus = ws.open_drain([driver_a, driver_b])     # low if any driver pulls; else pull-up high
wave = ws.uart_frame([0x55, 0xA3], samples_per_bit=16)   # idle-high start/stop framing
```

A second sink on the SAME physical line is resolved as a real resistor divider, not a boolean
OR — `Signal(...).open_drain(r_pullup_ohm=..., c_bus_f=..., second=dict(kind="nrz", ...), r_sink_b_ohm=...)`.

## Level coding (PAM4 Gray + precoding, PAM3)
The bit-to-level layer, between a bit stream and a PAM carrier. It changes **symbol and error
statistics, not the eye** — which is exactly why it is easy to leave out and hard to notice
missing.

```python
bits = ws.dc_balanced(P.prbs(13, 1 << 14, seed=3))       # or any bit source

sym  = ws.pam4_gray_levels(ws.precode(bits))             # PAM4: precode, then Gray-map
tern = ws.pam3_encode(bits[: 11 * (len(bits) // 11)])    # PAM3: 11 bits -> 7 ternary symbols

wave = ws.Signal(seed=1, grid=g).symbols(symbols=sym.tolist(), causal=True).waveform()
```

- **Gray coding** (IEEE 802.3 Clause 120.5.11.2.1: `00b`, `01b`, `11b`, `10b` from the bottom
  level up) makes adjacent levels differ in one bit, so a PAM4 slicer's dominant error — one
  level — costs exactly one bit. Natural binary costs 4/3 of that for the identical waveform.
- **Precoding** (`precode` / `precode_inverse`, the GF(2) `1/(1+D)` recursion that PCI Express
  Base Specification Rev 6.0 makes mandatory at 64 GT/s) leaves the level histogram and the
  0.75 transition density untouched and changes ~75% of the symbols. What it buys is the error
  structure: the receiver's inverse is feedforward, so one received-symbol error decodes to
  **exactly 2** bit errors instead of a burst that runs to the end of the record. A record
  generated without it has a plausible eye and the wrong error statistics.
- **PAM3** (`pam3_encode` / `pam3_decode`) is the 11-bit-to-7-ternary-symbol block code behind
  1.57 bits/symbol — 7 is the shortest ternary block that holds 11 bits, and 11/7 reaches
  99.15% of the `log2(3)` ceiling. The block size and rate are established; the codeword
  **table** is the canonical base-3 mapping and is marked PRELIMINARY in the source, with the
  measured consequence (a skewed leading symbol, 139 spare codewords) stated there.

Levels come from `physics.pam_levels()` and are never restated: the four PAM4 levels are
equally spaced to one ulp of 2/3, and `PAM4_LEVELS_MV` is that same map at a nominal 800 mV
pk-pk swing (-400 / -133.33 / +133.33 / +400 mV, 800/3 = 266.67 mV apart).

## Two-rate acquisition (simulation grid -> acquisition -> stored record)
Simulate physics on a fine grid, then model the acquisition system (front end + digitizer +
optional record decimation) onto the sample rate / record length that actually gets stored:

```python
from wfmsynth import AcquisitionProfile
prof = AcquisitionProfile(sample_rate_hz=2.5e9, record_length=16384, input_bandwidth_hz=800e6,
                          enob=7, decimation=dict(mode="peak_hold", depth=1024))
sig = ws.Signal(seed=1, grid=ws.Grid(fs=200e9, baud=25e9, n=1<<14)).carrier("nrz", n_ui=256, causal=True)
stored = sig.acquire(prof).waveform()             # what an acquisition system would store
taps   = sig.acquire_taps(prof)                   # simulated / conditioned / digitized / stored
```
Carrier-agnostic (NRZ/PAM4/clocks/buses/optical). `peak_hold` decimation keeps a 2-channel
(min,max) record so narrow transients survive time compression.


## Shared nets (several drivers, switched terminations)

A `cascade` is a series 2-port chain: one source, one probe. A shared net is not that. Several
agents drive it at different times through hi-Z, terminations switch in and out, and a weaker
driver can hold a level against a stronger one.

```python
bus = dict(
    nodes=["bus"], rails={"vdd": 3.3},
    elements=[{"id": "rp", "kind": "r", "a": "bus", "b": "vdd", "ohm": 885.0},
              {"id": "cb", "kind": "c", "a": "bus", "b": "gnd", "farad": 400e-12}],
    drivers=[{"id": "d0", "node": "bus", "volts": 0.0, "ohm": 20.0, "state": "hiz"},
             {"id": "d1", "node": "bus", "volts": 0.0, "ohm": 47.0, "state": "hiz"}],
)

x = (ws.Signal(seed=1, grid=g)
     .carrier("nrz", n_ui=n_ui, pattern="prbs13", tr_frac=0.5, causal=True)
     .net(**bus, follow="d0",                       # the carrier drives d0's state
          schedule=[{"t_s": 2e-6, "id": "d1", "state": "driving"},   # a second agent joins
                    {"t_s": 5e-6, "id": "d1", "state": "hiz"}])
     ).waveform()
```

`follow` names the driver whose **state** the incoming waveform sets, thresholded at the
midpoint of its own excursion — the rule `open_drain_line` uses, which that function's one-node
case reproduces to 2.2e-15.

A `schedule` is a plain list of `{"t_s", "id", "state"}` in **absolute seconds** against stable
ids, so a protocol, traffic or fault generator can emit one without this op knowing anything
about it. Times are absolute deliberately: a fraction would silently rescale if the record
changed length, and for the same reason `net` refuses a lead-in.

For several probes off one solve — related channels that must share one truth — call the
function directly:

```python
v = ws.net.solve(spec, fs=g.fs, n=g.n, schedule=sched, probes=["bus", "stub"])
```

**It is exact, not integrated.** Between events the network is constant and is solved in closed
form, so there is no step size to choose and no accuracy that decays over a long record. What
the grid still decides is *visibility*: a transient faster than a sample completes inside one,
and the solver says so rather than letting you measure the grid.

**This tier is lumped R, L and C** — the inductance is what makes a hand-over ring rather than
settle, and it has its own section below. A distributed transmission line, S-parameter blocks
inside a net, sub-nets, voltage-dependent laws and event emission are deferred — see
`BACKLOG.md` for the order.

## Scenario layer: who held the net when (holds, agents, labels)

A schedule of `{"t_s", "id", "state"}` is the right interface for a solver and the wrong one to
write by hand. `wfmsynth.scenario` is the layer between: **holds** say who was on the net and
when, **agents** say what being on it does, and `schedule()` compiles the two into the net's
event list. It adds no physics — it emits the events a careful person would have written.

```python
from wfmsynth.scenario import Hold, Agent, merge_holds, hold_windows

holds = [Hold("host", 1.0e-6, 2.5e-6, field="addr", bit=0),   # tags are free-form
         Hold("host", 2.5e-6, 4.0e-6, field="addr", bit=1),   # abuts: ONE hold after merging
         Hold("target", 3.9e-6, 5.0e-6, field="ack")]         # overlaps -> a hand-over

agents = {
    "host":   Agent(during={"pad_h": "driving"}, after={"pad_h": "hiz"},
                    lead=20e-9, lag=60e-9),       # the output stage leads and lags the logic
    "target": Agent(during={"pad_t": "driving"}, after={"pad_t": "hiz"},
                    lead=20e-9, lag=60e-9),
}

x = sig.net(**bus, holds=[h.to_dict() for h in holds],
            agents={k: a.to_dict() for k, a in agents.items()}, probe="bus").waveform()
```

- **`lead` / `lag`** are device properties, not protocol: an output stage conducts before the
  logic edge that asked for it and after the one that released it. Keeping them on the agent is
  what lets the same encoder drive a different board.
- **Merging is required, not an optimisation.** Consecutive identical bits abut, and without
  `merge_holds` the bus would release and re-acquire at every bit boundary — seven spurious
  edges inside a byte. It tolerates the float-noise case, where `t0 + i*dt + dt` and
  `t0 + (i+1)*dt` are the same instant and differ in the last bit.
- **A hand-over needs no special case.** Two agents holding at once are two parallel drivers,
  which the nodal solve resolves. Events landing on one sample are applied as *one*
  configuration change, so a hand-over never passes through a moment with nobody driving.
- **`then` is a release reaction** — `(dt, id, state)` rows after the agent lets go, for a
  switch that keeps conducting for a while or a stage that settles. **`unless`** names agents
  whose holds suppress it: the net changed hands rather than being released. The whole reaction
  is suppressed together, so `after` must be a complete idle row on its own.
- **The holds are also the labels.** `hold_windows(holds, fs, n)` produces the window dicts
  `events.label_windows` consumes, carrying `who` and every tag. One list made the waveform and
  describes it, so they cannot disagree — and a fault injected by editing the holds labels
  itself.

An event may also set `g` (conductance in siemens, where zero *is* an open) or `src` (a driver's
Thevenin level in volts), so a weaker driver taking over from a stronger one is two drivers at
two strengths. A quantity that varies *during* an interval (`gate`, a turn-on slew; `relax`,
settling) is refused by name — that is a time-varying element, which is what the closed form
rules out — with the staircase of `g` events to use instead.

**[`examples/i2c_bus.py`](../examples/i2c_bus.py) is the worked example**: an I2C write across a
3.3 V / 1.8 V MOSFET level translator, with a ~40-line protocol encoder that knows nothing about
volts and agents that know nothing about I2C. It finds a real result — the loaded 1.8 V side
rises in 1209 ns against a 300 ns Fast-mode limit, and the translator's R<sub>ds(on)</sub> eats
90 % of the 3.3 V side's V<sub>OL</sub> budget.

**The protocol layer is deliberately not in this library.** Encoders, traffic and gap
generators, fault injection and parameter priors are caller code — plain classes returning a
list of holds. Which bits a standard puts on a wire has a revision date and a document behind
it; "an agent held the net and its output stage turned on early" does not. Same boundary as the
pattern registry: mechanisms here, standards knowledge yours.

### A driver with its own chain

A driver's Thevenin level may be a **waveform** rather than a number — its own supply, its own
edge rate, its own timing impairments — injected through its own scheduled output conductance:

```python
sig.net(nodes=["bus"], rails={"vbias": 1.5},
        elements=[{"id": "rt", "kind": "r", "a": "bus", "b": "vbias", "ohm": 56.0},
                  {"id": "cb", "kind": "c", "a": "bus", "b": "gnd", "farad": 60e-12}],
        drivers=[
            {"id": "ctrl", "node": "bus", "ohm": 20.0, "state": "hiz",
             "source": {"ops": Signal(grid=g).carrier("cmos", v_lo=0.0, v_hi=3.3,
                                                      f_hz=20e6, tr_s=2e-9)
                                             .timing(rj_ps=12.0),      # its own clock
                        "volts": True}},
            {"id": "n2", "node": "bus", "ohm": 55.0, "state": "hiz",
             "source": {"kind": "nrz", "n_ui": 328, "seed": 37,        # the compact form
                        "volts": [0.0, 1.8]}}],                        # ±1 → these volts
        holds=..., agents=..., probe="bus")
```

- **A source body is exactly one of three.** `from="chain"` (the waveform arriving at this op,
  so this driver *is* the chain), `ops=` (its own op list, or a `Signal` to take it from), or a
  carrier spec by `kind=` — the idiom `crosstalk(aggressor=)` already uses. With every driver
  carrying its own source the net can be the **first op**: nothing has to arrive at it, and its
  length comes from the grid.
- **`volts` is required**, and says the level domain: `True` for a chain already in volts (a
  `cmos` carrier, or anything ending in one), or `[lo, hi]` to map the ±1 carrier convention
  onto those two volts. A net is in volts; a normalised carrier passed through unconverted is a
  driver a thousand times too weak, and the solve would be perfectly correct about it. The map
  is affine, so a `tx_ffe`'s amplitude or an overshoot survives it rather than being
  renormalised away.
- **A released driver contributes nothing**, whatever its own chain is then doing — it is out
  of the stamp, so the level it was holding is *absent* rather than small. Adding each party's
  contribution by hand outside the solve is where that level gets carried into the next owner's
  baseline, and the bus drifts by one held offset at every change of hands.
- **Still exact.** Only `b` moves between events, so the eigenbasis is computed once per
  interval as before and the sampled source is carried through it by the zero-order-hold
  recurrence `physics.open_drain_line` uses on its target. An interval across which nothing
  moves is routed back to the closed form, so a constant source equals a constant `volts`
  bit-for-bit. A time-varying **conductance** is still refused (`gate`, `relax`).
- Each driver's randomness is namespaced by its id, so two drivers written from the same spec
  are two devices rather than one written twice, and one `Signal(seed=)` still moves the whole
  record.

**[`examples/multi_driver_bus.py`](../examples/multi_driver_bus.py) is the worked example**:
three transceivers of different supplies, output impedances and edge rates taking turns on one
terminated backplane bus. Every measured level and edge lands on its closed form to ~0.2 %, the
weakest part misses the receiver's V<sub>IH</sub> by 346 mV because a terminated bus penalises
it twice, and the bus returns to its bias **exactly** at every hand-over against the −1.765 V a
by-hand sum would have accumulated.

### An inductance, and a hand-over that rings

An R-C net can only *settle*. What a scope shows when a fast driver takes a bus, changes it or
lets go of it is overshoot and a ring — at a frequency set by the package and trace inductance
against the node capacitance, decaying at a rate set by the termination. `{"kind": "l"}` is that
inductance (GitHub #74):

```python
elements=[{"id": "lp", "kind": "l", "a": "pad", "b": "bus", "henry": 8e-9, "ohm": 0.3},
          {"id": "cp", "kind": "c", "a": "pad", "b": "gnd", "farad": 2e-12},
          {"id": "cb", "kind": "c", "a": "bus", "b": "gnd", "farad": 20e-12},
          {"id": "rt", "kind": "r", "a": "bus", "b": "vb", "ohm": 100.0}]
```

- **`ohm` on an inductor is the series resistance of the same piece of wire**, not a parallel
  resistor, and absent means ideal. It is one element with two numbers because the node *between*
  a series R and a series L has no capacitance, and this tier solves states — so writing them as
  two elements is refused, with that explanation. A bond wire is one piece of metal with both
  properties anyway.
- **An inductor is not switchable**, and the schedule refuses it by name. Its current is a state
  variable carried across every switch exactly as capacitor charge is, so taking it out of the
  stamp would drive that current to zero within one sample, which is the one thing an inductor
  cannot do. Switch what is in *series* with it — the driver, or an `r`. The current then flows
  on into the node capacitance, which is what rings on a real board, and is why "every node has
  some capacitance" is physics here rather than a solver's convenience. (A capacitor is refused
  the same way, for the same reason; both used to render as though the event had not been
  written.)
- **`probes=["bus", "i(lp)"]`** reads an inductor's current in **amps** — the state the solve now
  carries, and the only way to see it. `v0={"bus": 0.0, "i(lp)": 0.0}` sets it; a current `v0`
  does not name starts at zero, which is at rest.
- **Still closed-form, by a different decomposition.** An inductor costs the symmetry the R-C
  eigensolver needs, and eigenvectors do not survive *exactly* critical damping
  (`R = 2·√(L/C)` — a damping network somebody designs on purpose), where the matrix is defective
  and a modal solve returns 368 mV of nonsense on a 1 V step. So the inductive path takes the
  exact zero-order-hold pair for one sample and back-substitutes it through a **Schur** form,
  whose basis is unitary at every damping. Measured against the textbook series-RLC step
  response: **3.5e-14 V** underdamped, **8.2e-14** critically damped, **4.3e-14** overdamped, on
  a 1 V step. A net with **no** inductors does not take this path and renders bit for bit as
  before.
- **A ring the grid cannot show aliases**, and that is worse than a decay it cannot show: a fast
  decay is simply missing from the record, but an aliased ring comes back at a frequency that is
  not the circuit's and looks entirely plausible. Below 8 samples per ring period the solver says
  so, and past Nyquist it says the record's ring is an alias.

**[`examples/inductive_handover.py`](../examples/inductive_handover.py) is the worked example**:
one 3.3 V transceiver, 7 Ω output stage, 8 nH of package, 20 pF of bus. The bus overshoots to
**+4.396 V** and undershoots to **−1.096 V** — 796 mV past both of the receiver's absolute
maxima — while the R-C model of the identical board sits 412 mV *inside* both and is wrong about
every edge; the two records differ by up to **1.758 V, 57 % of the swing**. Letting go while
carrying 15.4 mA kicks the pad 0.980 V in 195 ps, against the 1.020 V that current into the
loop's own 66 Ω accounts for, ringing at a measured 1339 MHz against 1320 MHz closed form. And
12 Ω of series damping is the whole fix, for 131 ps of fall time.

### A line between two nodes, as a lumped ladder

A transmission line inside a net is still deferred — `lossy`, `sparam`, `cascade` and `reflect`
represent one *as* a line, exactly and at any length, and the only thing they cannot do is sit
inside a switched net. What the inductor makes reachable is a lumped **ladder**, which is a real
approximation with a stated error:

```python
nodes, elements = ws.net.line_ladder("near", "far", z0_ohm=50.0, delay_s=1e-9, tr_s=500e-12)
```

The edge is an argument rather than an afterthought, because the approximation *is* a cutoff: a
ladder carries an edge that spans several sections and mangles one that does not — as its own
overshoot, which is the one artefact indistinguishable from the ringing the net is there to show.
The section count comes from a **measured** rule:

    sections = ceil(6 × (delay_s / tr_s) ** 1.5)

| line length (delay / rise time) | sections | ladder's own error |
|---|---|---|
| 0.5 | 3 | 1.14 % of the swing |
| 1 | 6 | 1.94 % |
| 2 | 17 | 1.73 % |
| 3 | 32 | 1.60 % |
| 5 | 68 | 1.67 % |
| 10 | 190 | 1.82 % |

measured against a matched source into a matched load, as the worst deviation from the same edge
delayed and halved. Below the rule it degrades quickly — about **5 %** of the swing at half the
count and **10–16 %** at a quarter — so a smaller `sections=` warns rather than rendering
quietly. The cost of the accuracy is states: the count grows as the 1.5 power of the line's
length in rise times and the per-interval solve as roughly its cube (0.07 s at five rise times,
0.45 s at ten, *each time the schedule switches*), so past 256 sections it is refused and names
the frequency-domain ops instead. `loss_ohm` spreads a total series DC resistance along the line
— a divider, and nothing about frequency: skin effect and dielectric loss belong to `lossy`.

---

# Composing a realistic capture

A chain is a line with one source. A capture is not: the aggressor has its own channel, the clock
and the data share a PLL, the transmitter idles before it sends, and the scope keeps only the
windows its trigger chose. This chapter is the handful of primitives that let chains meet. Take
the first entry and stop there until you need the next; the last entry uses all of them on one
record. The rule for what is an op and what is a composition is in
[ARCHITECTURE.md](ARCHITECTURE.md#what-becomes-an-op).

Every block in this chapter runs top to bottom as one script.

## A chain as a parameter
Any parameter that takes a signal takes a whole chain. Write it as a `Signal()` with no seed and
no grid; it joins this record.

```python
import json, numpy as np, wfmsynth as ws
from wfmsynth import Signal

g = ws.Grid(fs=64e9, baud=4e9, n=8192)                 # 16 samples/UI, 512 UI
lane = Signal().carrier("nrz", n_ui=512, seed=9).lossy(loss_db=6.0, loss_at_ghz=2.0)
sig = (Signal(seed=1, grid=g)
       .carrier("nrz", n_ui=512, seed=3)
       .crosstalk(aggressor=lane, coupling=0.1)        # the neighbour AFTER its own channel
       .rx_noise(rms=0.01))
x = sig.waveform()
print(sig.summary())
```

```
Signal  seed=1  fs=6.4e+10 Hz, baud=4e+09, n=8192
  0  carrier    kind='nrz', n_ui=512, seed=3
  1  crosstalk  coupling=0.1
       aggressor  [crosstalk/1/aggressor]
         0  carrier  kind='nrz', n_ui=512, seed=9
         1  lossy    loss_db=6.0, loss_at_ghz=2.0
  2  rx_noise   rms=0.01
```

The name in brackets is the sub-chain's namespace. It is how you reach inside:

```python
aggr = sig.subchain("crosstalk/1/aggressor")           # the aggressor exactly as it was coupled
sig.roles()                                            # ['rx_noise/2'], plus any role inside it
assert Signal.from_recipe(json.loads(sig.to_json())).sha256() == sig.sha256()
```

The recipe holds the sub-chain as `{"ops": [...]}`, so it replays with no other file. The
parameters that take one: `crosstalk(aggressor=)`, `crosstalk_matrix(aggressors=[...])`,
`modulate(message=)`, `open_drain(second=)`, `hybrid_echo(own=)`, `pass_fet(gate=)`,
`trigger(on=)`, a `net` driver's `source`, and everything below.

**Another record, pinned.** A `Signal()` sub-chain draws from the parent's seed, so it moves
when the parent's seed does. `{"from": recipe}` is the other thing: a record with its own seed
and its own baud, embedded as rendered. Same sample rate and length, or it is refused; nothing is
resampled. Pinned includes its unit: `unit()` draws the parent's toleranced parts again and
leaves the embedded record's as they were, and `roles()` lists its noise but not its parts.

```python
other = (Signal(seed=5, grid=ws.Grid(fs=64e9, baud=2.5e9, n=8192))
         .carrier("pam4", n_ui=320).timing(rj_ps=1.0))
pinned = (Signal(seed=1, grid=g).carrier("nrz", n_ui=512)
          .crosstalk(aggressor={"from": other.recipe()}, coupling=0.1))
assert np.array_equal(pinned.subchain("crosstalk/1/aggressor"), other.waveform())
```

## Adding, scaling, gating, delaying
`combine` puts a second signal on the record. `gain`, `offset`, `gate` and `delay` each take
`by`: a number, or a chain.

```python
gs = ws.Grid(fs=1e6, n=20000)                          # 20 ms of a 1 kHz sensor

def sensor():                                          # a builder mutates; make one per variant
    return Signal(seed=1, grid=gs).carrier("sine", f_hz=1e3)

def tone(**k):
    return Signal().carrier("sine", **k)

def line(sig, k):                                      # amplitude of FFT bin k (50 Hz per bin)
    return abs(np.fft.rfft(sig.waveform())[k]) / (gs.n / 2)

line(sensor().combine(tone(f_hz=50.0, amp=0.02)), 1)                 # -> 0.0200: the hum, added
line(sensor().gain(by=tone(f_hz=100.0, amp=0.03, offset=1.0)), 18)   # -> 0.0150: a 3 % ripple's sideband
sensor().offset(by=0.25).waveform().mean()                           # -> 0.25

gated, events = sensor().gate(by=tone(f_hz=100.0, amp=1.0), threshold=0.5).realize()
len([e for e in events if e.kind == "transition"])                   # -> 4, each with its crossing time

moving = sensor().delay(by=tone(f_hz=30.0, amp=2e-6, offset=4e-6))   # seconds, never negative
moving.delay_errors()[0]["error"]                                    # -> 2.3e-05 of a tone below 0.4 fs
```

`gain(by=)` is the factor itself, so a 3 % ripple is a sine of amplitude 0.03 about 1.0. A
`delay` that would go negative is refused and names `timing`, which may advance. `combine` also
takes `how="product"`, `"min"` or `"max"`, and `gains=[...]`, one per other signal.

A sum is superposition, which holds when neither source loads the other: a coupled ripple, an
interferer, a tone on a supply. Two drivers on one node load each other, and that is
[`net`](#shared-nets-several-drivers-switched-terminations). `combine` cannot tell from samples
and does not try.

## In time: sequence and switch
`sequence` is a source made of chains end to end, with stated durations. `switch` chooses among
chains by a signal, and emits the instants it chose.

```python
g2 = ws.Grid(fs=1e9, baud=50e6, n=8000)                # 8 us, 400 UI
seq = (Signal(seed=1, grid=g2)
       .sequence([{"chain": Signal().carrier("dc", level=-1.0), "duration_s": 1e-6},
                  {"chain": Signal().carrier("nrz", n_ui=400, pattern="clock"), "duration_s": 2e-6},
                  {"chain": Signal().carrier("nrz", n_ui=400, pattern="prbs7")}])
       .lossy(loss_db=3.0, loss_at_ghz=0.05))          # one channel, after the join
x, events = seq.realize()
[(e.kind, e.sample) for e in events]                   # -> [('segment', 1000), ('segment', 3000)]

label = np.zeros(g2.n, int)                            # a per-sample label, from the recipe
for part in ws.sequence.layout(seq.ops[0]["segments"], g2):
    label[part["start"]:part["stop"]] = part["segment"]
np.bincount(label)                                     # -> [1000 2000 5000]

mux = Signal(seed=1, grid=g2).switch(
    select=Signal().carrier("square", f_hz=500e3),
    inputs=[Signal().carrier("dc", level=0.0), Signal().carrier("nrz", n_ui=400)],
    threshold=0.0, t_switch_s=20e-9)
len(mux.realize()[1])                                  # -> 7 changes, each a "switch" event
```

Durations are whole samples and fill the record; the last segment may omit `duration_s`. A segment
restarts its chain at its own t = 0 unless it says `"state": "continue"`, which takes the chain
as if it had been running since the record began. `join="crossfade"` or `"hold"` with `t_join_s`
shapes the boundary; the default is a cut. Each chain is rendered for the whole record, so the
cost is one render per segment.

## Several channels, one record
A clock and its data are one record. `ws.Channels` holds named chains under one seed and one
grid, and `use(name)` refers to another chain: applied to a waveform when the named chain is a
stage, standing for its waveform when it starts with a source.

```python
gc = ws.Grid(fs=64e9, baud=1e9, n=1 << 16)             # 1024 UI
rec = ws.Channels(
    {"clk":  Signal().carrier("nrz", n_ui=1024, pattern="clock").use("pll"),
     "data": Signal().carrier("nrz", n_ui=1024, pattern="prbs7").use("pll").rx_noise(rms=0.01)},
    shared={"pll": Signal().timing(phase_noise=dict(rms_ps=2.0))}, seed=1, grid=gc)
w = rec.waveforms()                                    # {"clk": array, "data": array}
rec.roles()                                            # -> ['data/rx_noise/2', 'pll/timing/0']

again = rec.contrast("pll/timing/0")                   # another PLL realisation: both channels move
other = rec.contrast("data/rx_noise/2")                # another noise record: only data moves
float(np.abs(other["clk"] - w["clk"]).max())           # -> 0.0

pair = ws.Channels({"p": Signal().carrier("nrz", n_ui=1024),
                    "n": Signal().use("p").law("poly", coeffs=[0.0, -1.0])},
                   seed=1, grid=gc).waveforms()
float(np.abs(pair["p"] + pair["n"]).max())             # -> 0.0: one bit stream, two legs
```

A chain draws its randomness under its own name wherever it is used, which is why the two
channels share one PLL realisation. The trap is the same fact: noise inside a shared stage is
the *same* noise on every channel using it, so put `rx_noise` on each channel. Two lanes that are
each other's crosstalk cannot name each other; declare what each *launched* under `shared` and
couple those (`aggressor=Signal().use("tx1")`). `rec.write_hdf5(path)` writes every channel and
the recipe to one file. [`examples/clock_and_data.py`](../examples/clock_and_data.py) measures
the shared jitter edge by edge.

## A shape with no op of its own: law and filter
`law` is what comes out as a function of what goes in; `filter` is a named linear design with
its corner in hertz. Both are kinds with parameters, never callables, so the recipe stays data.

```python
gl = ws.Grid(fs=1e6, n=50000)                          # 50 ms: ten cycles of 200 Hz

def noisy():
    return Signal(seed=1, grid=gl).carrier("sine", f_hz=200.0).rx_noise(rms=0.05)

def edges(sig):
    return len([e for e in sig.realize()[1] if e.kind == "transition"])

edges(noisy().law("comparator", v_lo=0.0, v_hi=3.3))                  # -> 861: it chatters
edges(noisy().law("comparator", hysteresis=0.3, v_lo=0.0, v_hi=3.3))  # -> 19
slow = noisy().filter("butter", f_hz=5e3, order=2).law("comparator", v_lo=0.0, v_hi=3.3)
edges(slow)                                                           # -> 19
slow.filter_delays()[0]["group_delay_s"]                              # -> 4.5e-05: it also moved them
# (group_delay_s is for a feature slow against the filter, as here; a FAST edge's midpoint
#  arrives step_delay_s later, a different number that sits beside it)
```

A comparator's output edge sits on a sample; the true crossing time is the event's `t_s`. Laws:
`tanh`, `poly`, `limiter`, `deadzone`, `rectify`, `quantize`, `comparator`, `diode`. Filters:
`butter`, `bessel`, `cheby1`, `elliptic`, `rrc`, `fir`, `sos`, `zpk`, `integrate`,
`differentiate` (`ws.LAW_KINDS`, `ws.FILTER_KINDS`). A filter is causal and starts settled;
`zero_phase=True` removes the delay without squaring the response.

These are not a second way to spell the named ops. `filter("gaussian")` is refused and names
`scope`; a converter is `digitize`, an equaliser is `ctle`. See
[which name is canonical](#which-name-is-canonical).

## What the scope keeps: trigger
The grid is the LONG render. `trigger` finds the condition on the realised waveform, noise and
channel included, and keeps a window about each firing.

```python
long_grid = ws.Grid(fs=64e9, baud=1e9, n=1 << 16)      # 1024 UI to trigger within

def running():
    return (Signal(seed=1, grid=long_grid)
            .carrier("nrz", n_ui=1024, pattern="prbs7")
            .lossy(loss_db=3.0, loss_at_ghz=0.5).rx_noise(rms=0.01))

shot = running().trigger(pre_s=1e-9, post_s=3e-9, level=0.0, slope="rising",
                         n_segments=8, holdoff_s=20e-9)
x, events = shot.realize()
rows = x.reshape(8, -1)                                # -> 8 windows of 256 samples, end to end
shot.stored_grid(x).n                                  # -> 2048; sig.grid still says 65536
fired = [e for e in events if e.kind == "trigger"][0]
fired.t_s                                              # -> 9.98e-10: the crossing, not exactly pre_s
fired.params["timestamp_s"]                            # -> 6.0e-09: when, in the long render
```

A window is a slice, never resampled, so the crossing sits within half a sample of `pre_s` and
the event says exactly where. Asking for more windows than the render holds is refused, and the
message states the length that would do. `condition="width"` and `"runt"` trigger on a pulse;
`jitter_s` is trigger jitter (role `trigger/<i>`), which moves a whole window and is not
`timebase`; `average=` averages consecutive triggers. `trigger(on=<chain>)` finds the condition
on another signal, which is how one clock cuts every channel of a `Channels` record at the same
instants. [`examples/triggered_capture.py`](../examples/triggered_capture.py) has each.

## Unit to unit
Put a tolerance where a number goes. It is drawn once, on the way into the recipe, and the
recipe keeps both the value and what it was drawn from.

```python
gt = ws.Grid(fs=100e9, n=8000)

def board(seed):
    return (Signal(seed=seed, grid=gt)
            .carrier("cmos", v_lo=0.0, v_hi=3.3, f_hz=25e6, tr_s=1e-9)
            .probe(r_source=ws.tol(1e3, 5),                     # 1 kohm +/- 5 %, uniform
                   c_load_f=ws.tol(2e-12, 10, dist="normal"))   # 10 % is 3 sigma, truncated
            .rx_noise(rms=0.005))

s = board(3)
ws.op_params(s.recipe(), "probe")["r_source"]          # -> 1019.1: the label, a plain number
s.roles()                                              # -> ['rx_noise/2', 'unit/1/r_source', 'unit/1/c_load_f']

another = s.unit(8)                                    # another unit of the design, the same noise
ws.op_params(another.recipe(), "probe")["r_source"]    # -> 963.9
assert np.array_equal(s.contrast("unit/*", seed=8), another.waveform())

X, recipes = ws.unit_dataset(lambda rng: board(int(rng.integers(1 << 30))), 6, 4, seed=1)
units = np.array([ws.unit_of(r) for r in recipes])     # 24 records, 6 units: split on THIS
```

`ws.unit_dataset` takes a builder that returns a `ws.Channels` as well: `X` is then
`{name: (n, L)}`, one stack per channel, and `ws.op_params(recipe, "lossy", chain="clk")` reads a
label from one chain of the record's recipe. A part inside a sub-chain is named like everything
else inside it: `crosstalk/1/aggressor/unit/1/loss_db`, beside `crosstalk/1/aggressor/rx_noise/2`.

Split train and test on `ws.unit_of(recipe)`, never at random: captures of one unit on both
sides is leakage ([`examples/unit_variation.py`](../examples/unit_variation.py) measures it,
1.7 ohm error on a random split against 31.3 ohm on held-out units). A whole-number parameter
takes `ws.corners({"few": 2, "many": 6})`, which keeps its type. Parameters that share
`group="name"` take one draw, across the chains of a `Channels` record too: the same board, the
same process corner.

## All of it at once
[`examples/composed_capture.py`](../examples/composed_capture.py) is a forwarded-clock link on
two scope channels, 1 GBd at 64 GSa/s, the scope triggered from the clock. The transmitter is a
`sequence` (idle, preamble, payload); clock and data `use` one PLL, one supply (`gain(by=)`) and
one scope (`trigger(on=)`); the data trace is longer (`delay`), a 133 MHz neighbour leaks onto
it (`combine`), and the board's loss is one toleranced part shared by both lines
(`ws.tol(..., group="board")`). The whole record is 1888 bytes of JSON and replays bit for bit.

The PLL is 8 ps rms. Triggered on the clock, the data edges are 6.3 ps wide in the preamble and
14.1 ps in the payload. Every candidate cause is one named thing in the recipe, so each is
tested by changing exactly that:

| Changed | Preamble, ps rms | Payload, ps rms |
|---|---|---|
| nothing (the capture) | 6.29 | 14.13 |
| PLL jitter removed | 5.71 | 14.17 |
| a PLL each, not one shared | 7.13 | 18.86 |
| trace mismatch 0.3 ns to 30 ns | 6.52 | 14.90 |
| supply ripple removed | 6.45 | 14.10 |
| scope noise removed | 5.98 | 13.90 |
| 133 MHz neighbour removed | 3.82 | 13.39 |
| channel loss 6 dB to 0.5 dB | 2.03 | 2.03 |

87 preamble and 330 payload edges, so an rms is known to about 8 % and 4 %. It is not the PLL: a
shared PLL moves clock and data together and a clock-triggered scope does not see it. The
preamble is spread by the neighbour, the payload by the data's own history through the channel.
Five units of the board span 5.36 to 6.81 dB on the data line and 14.1 to 21.2 ps on the
payload. Nothing was fitted; the example prints `rec.summary()` first, and that tree is the
fastest way to see how the pieces sit.

## Which name is canonical
The older named ops stay, unchanged in call and output. Each is the name of a physical effect;
the general stage is for the shape that has no name.

| Named op | The general form it is a shape of | Reach for the named op when | Reach for the general one when |
|---|---|---|---|
| `crosstalk` | `combine(how="sum")` after a coupling response | the second signal is coupled | it is added as it stands: hum, an interferer |
| `modulate` | `combine(how="product")` for AM/ASK/OOK | it is a modulation (FM and PM have no other form) | any other product |
| `burst` | `gate` with a periodic cadence and ramped edges | the on/off pattern is periodic | the enable is a signal from somewhere else |
| `drift` | `gain` or `offset` by a ramp or half-sine | the drift is one of its named shapes | it follows another signal |
| `timing` | `delay` by a named jitter source | jitter about zero, which may advance | a path length that moves; causal, with a stated error |
| `scope`, `probe`, `ctle`, `tx_ffe`, `rx_ffe`, `dfe`, `ac_couple` | `filter` | it is that instrument or equaliser | a Butterworth, Chebyshev or elliptic design, an integrator, your own taps |
| `nonlinearity`, `digitize`, `store`, `pass_fet` | `law` | it is a transmitter, a converter, a switch | a comparator, rectifier, dead zone, limiter, diode |
| `agc` | `gain(by=)` | the gain is a loop that sets itself from the signal's level | the gain is a stated signal: a supply, a fade, an envelope's level control |
| `supply_coupling` | `gain(by=)` and `timing` from one ripple | one rail gives both the amplitude ripple and the supply-induced jitter | the ripple is a chain of its own (`gain(by=)`), or simply adds (`combine`) |
| `Scene` | `Channels` | you hold arrays already | the channels are chains and must replay from one recipe |

`gain(by=chain)` and a product `combine` are the same samples, as are `offset(by=chain)` and a
sum: say `gain`/`offset` when the other signal is a property of this one (its supply, its
temperature) and `combine` when it is a signal in its own right.

---

# I/Q: complex baseband and RF

A record is in one of two domains. `real` is a voltage, and is everything above. `iq` is complex
baseband: each sample is I + jQ, the envelope of a signal centred on `fc_hz`, and `waveform()`
returns complex128. Two ops move between them and every other op is real-only unless it has
opted in, so nothing silently drops an imaginary part.

Every block in this chapter runs top to bottom as one script.

## A record in the I/Q domain
The domain is the grid's. The `iq` source maps bits onto a constellation and shapes them with a
root-raised-cosine pulse; `transmitted()` is the ground truth.

```python
import os, tempfile, numpy as np, wfmsynth as ws
from wfmsynth import Signal, rf

g = ws.Grid(fs=6.4e9, baud=25e6, n=1 << 18, domain="iq", fc_hz=1e9)

def source():                                          # a builder mutates; make one per variant
    return Signal(seed=1, grid=g).iq("qam16", prbs=15, prbs_phase=100)

x = source().waveform()                                # complex128, 1024 symbols
sent, t_s = source().transmitted()                     # the symbols, and when each is centred
rf.evm(x, g, sent, skip=8)["evm_pct"]                  # -> 0.34 %: the pulse's own truncation
```

Constellations are `bpsk`, `qpsk`, `qam16`, `qam64`, Gray-coded at unit average power, or
`iq(symbols=[...])` for any alphabet. The envelope *peaks* near 2 for QAM-16, so size a
converter's full scale from the record.

## Up to a passband voltage and back
`upconvert` makes the real passband record on the same grid, and from there every real op
applies. `downconvert` needs its low-pass stated, because that decides what the I/Q contains.

```python
loop = source().upconvert().downconvert(lpf=20e6, fs_out=400e6)     # about the grid's fc_hz
y = loop.waveform()
gy = loop.stored_grid(y)                               # 400 MSa/s, 16384 samples, domain "iq"
rf.evm(y, gy, sent, skip=8)["evm_pct"]                 # -> 0.32 %
```

Neither mixer resamples. The grid has to carry the carrier, and a carrier too close to DC or
Nyquist for the envelope's bandwidth is refused with the folded power stated. After `downconvert(fs_out=)` or `acquire`, measure with `stored_grid(x)`.

An op written for a voltage is refused on an envelope, by name, with what to do:

```python
try:
    source().lossy(loss_db=6.0, loss_at_ghz=1.0)
except ValueError as e:
    print(e)        # lossy (op 1): the record is a complex baseband record (domain 'iq') ...
```

## Measures against the symbols that were sent
`ws.rf` holds the RF measures. The amplifier here is not an op: it is `law("tanh")` on the
passband voltage, where an amplifier acts.

```python
LPF = dict(pass_hz=40e6, stop_hz=120e6, atten_db=80)

def through(sat=None):
    s = source().upconvert()
    if sat:
        s = s.law("tanh", sat=sat)
    return s.probe(bw_hz=3e9).downconvert(fc_hz=1e9, lpf=LPF, fs_out=400e6)

z = through(sat=2.5).waveform()
rf.evm(z, gy, sent, skip=8)["evm_pct"]                           # -> 26.7 %: mostly a rotation
fit = rf.evm(z, gy, sent, skip=8, correct=("gain", "phase"))
fit["evm_pct"], fit["removed"]                                   # -> 1.81 %, -0.62 dB and -0.267 rad
rf.aclr(z, gy, channel_bw_hz=33.75e6, offsets_hz=[40e6])["aclr_db"][0]   # -> 42.4 dB (60.9 linear)
rf.ccdf(z[1024:-1024])["papr_db"]                                # -> 4.96 dB (5.66 linear)
spec = rf.spectrum(z, gy, rbw_hz=1e6)                            # Welch; offsets from the centre
rf.band_power(spec, -16.875e6, 16.875e6)                         # -> 0.92
```

`rf.evm` fits nothing unless asked, and `removed` says how much each named correction took out:
`"timing"`, `"frequency"`, `"offset"`, `"gain"`, `"phase"`, `"equalizer"`. A front end's group
delay turns the constellation by -2&pi;·fc·delay, so raw EVM after one is a rotation, not
distortion. `reference="decided"` scores against decisions instead of the truth and reads low
once decisions fail; pass `sent` as well and it counts them.
[`examples/rf_measures.py`](../examples/rf_measures.py) has the table.

## Impairments, and the report of what they came to
Seven ops act on an envelope: `dac`, `iq_imbalance`, `iq_offset`, `phase_noise`, `freq_offset`,
`awgn`, `spur`. Their order is physical (imbalance before a frequency offset is the
transmitter's, after it the receiver's).

```python
gi = ws.Grid(fs=400e6, baud=25e6, n=1 << 16, domain="iq", fc_hz=1e9)   # 4096 symbols

def burst():
    return Signal(seed=1, grid=gi).iq("qam16", prbs=15, prbs_phase=100)

tx = (burst()
      .dac(bits=8, headroom_pct=5)                         # the converter
      .iq_imbalance(gain_db=0.3, phase_deg=2.0)            # Q rail against I, at the modulator
      .iq_offset(i=0.02, q=-0.01)                          # carrier leakage
      .phase_noise(profile=[(1e4, -75), (1e5, -95), (1e6, -115)])   # (offset Hz, dBc/Hz)
      .freq_offset(f_hz=2e3)
      .awgn(es_n0_db=25, ref_power=1.0))
xi = tx.waveform()
si, _ = tx.transmitted()
{r["name"]: r for r in tx.rf_report()}["iq_imbalance"]     # -> image_rejection_db 32.2
tx.roles()                                                 # -> ['phase_noise/4', 'awgn/6']

rf.evm(xi, gi, si, skip=8)["evm_pct"]                      # -> 108 %: 2 kHz turns it all the way round
fit = rf.evm(xi, gi, si, skip=8, correct=("frequency", "offset", "gain", "phase"))
fit["evm_pct"], fit["removed"]["frequency_hz"]             # -> 6.4 %, 1983 Hz
derotated = xi * np.exp(-1j * tx.lo_phase())               # the oscillator's phase is a label
rf.evm(derotated, gi, si, skip=8)["evm_pct"]               # -> 7.0 %, nothing estimated
```

`rf_report()` is what each request came to on this record (the DAC's SQNR and clipped fraction,
the leak in dBc, the noise density), and `lo_phase()` is the phase every oscillator op applied,
per sample. `awgn` takes exactly one of `es_n0_db`, `ebn0_db`, or `snr_db` with `bw_hz`; an
SNR with no bandwidth is refused. Its reference power is the record's own mean power as it
reaches the op unless `ref_power=` states one. `phase_noise` draws nothing outside the offsets
its profile states. There is no `image` op: `iq_imbalance(**rf.imbalance_for(30))` is an image
30 dB down. [`examples/iq_impairments.py`](../examples/iq_impairments.py) predicts each
impairment's EVM from `rf_report()` alone and measures it.

## Recordings in and out
`ws.read_iq` starts a chain from a file, on a grid taken from the file. `to_sigmf` and
`to_interleaved` write one, events included.

```python
work = tempfile.mkdtemp()
xb, events = burst().realize()

ws.to_sigmf(os.path.join(work, "burst"), xb, gi, events=events)      # .sigmf-meta + .sigmf-data
back = ws.read_iq(os.path.join(work, "burst"), format="sigmf")       # fs, fc, n from the metadata
assert np.array_equal(back.waveform(), xb.astype(np.complex64))      # written as float32

path = os.path.join(work, "burst.iq")
scale = float(max(abs(xb.real).max(), abs(xb.imag).max())) / 32767   # the value of ONE COUNT
ws.to_interleaved(path, xb, dtype="int16", scale=scale)
raw = ws.read_iq(path, format="iq_interleaved", dtype="int16", scale=scale,
                 fs_hz=gi.fs, fc_hz=gi.fc_hz, baud=gi.baud, embed=True)
float(abs((raw.waveform() - xb).real).max() / scale)                 # -> 0.5 LSB
recipe = raw.recipe()
os.remove(path)
assert np.array_equal(Signal.from_recipe(recipe).waveform(), raw.waveform())   # embedded: no file
```

An interleaved file states nothing, so the reader must be told `dtype`, `scale`, `fs_hz` and
`fc_hz`. `scale` is the value of one count, not a full-scale voltage; integer export at the
default (full scale ±1) refuses to clip and names the scale that fits. A byte-swapped file is
detected and refused (from 4096 values up); a wrong `scale` cannot be, because every scale gives
a self-consistent record. `format="csv"` and `"mat"` are read too. SigMF annotations come back
as events and events go out as annotations; `to_interleaved(events=)` writes a marker channel
beside the samples. [`examples/iq_recordings.py`](../examples/iq_recordings.py).

## The composition primitives on an envelope
Each general stage either means something on an envelope or is refused naming the alternative.

| Stage | On I/Q | Meaning, or what to use |
|---|---|---|
| `filter` | yes | stated on the grid, about the centre frequency; `filter("rrc", beta=0.35)` is the source's matched filter (state the source's `beta`) |
| `gain(by=)`, `gate(by=)` | yes | a REAL `by` acts on both rails: a supply, an AGC, a T/R switch |
| `combine` (sum only) | yes | envelopes about the same centre add: an interferer, a blocker |
| `sequence`, `switch` | yes | I/Q chains in time (preamble, then payload); `select` is real |
| `use`, `Channels` | yes | a shared `phase_noise` stage is one oscillator for several channels |
| `offset` | no | `iq_offset(i=, q=)` says which rail |
| `delay` | no | a passband delay also turns the carrier, which an envelope shift omits |
| `law` | no | `dac` quantises an envelope; an amplifier acts on the passband record |
| `trigger` | no | a scope triggers on a voltage |

```python
frame = (Signal(seed=1, grid=gi)
         .sequence([{"chain": Signal().iq("qpsk", prbs=9), "duration_s": 256 / 25e6},    # preamble
                    {"chain": Signal().iq("qam16", prbs=15)}])                    # payload
         .gain(by=Signal().carrier("sine", f_hz=40e3, amp=0.05, offset=1.0))      # a real supply
         .combine(Signal().iq("qpsk", prbs=11).freq_offset(f_hz=40e6), gains=[0.1]))
xf, ev = frame.realize()
[(e.kind, e.sample) for e in ev.on_record()]            # -> [('segment', 4096)]
spec = rf.spectrum(xf, gi, rbw_hz=1e6)
10 * np.log10(rf.band_power(spec, 23e6, 57e6) / rf.band_power(spec, -17e6, 17e6))   # -> -20.1 dB
sent, t_s = frame.transmitted()                         # preamble then payload, on the record's clock
rf.evm(xf, gi, sent, skip=8)["evm_pct"]                 # -> 3.53 %: the 5 % supply ripple, as an rms

two = ws.Channels(
    {"a": Signal().iq("qpsk", prbs=9).use("lo"), "b": Signal().iq("qam16", prbs=15).use("lo")},
    shared={"lo": Signal().phase_noise(profile=[(1e4, -75), (1e5, -95), (1e6, -115)])},
    seed=2, grid=gi)
two.roles()                                             # -> ['lo/phase_noise/0']
a = two.signal("a")                                     # channel a, exactly as it is in the record
a.transmitted()                                         # the symbols a sent
a.lo_phase()                                            # the shared oscillator's phase: one label for a and b
a.rf_report()                                           # -> [{'op': 0, 'name': 'phase_noise', 'chain': 'lo', ...}]
```

`transmitted()` follows a `sequence`: each segment's symbols at the times they sit in the record.
`rec.signal(name)` is one channel as a `Signal` that renders as it does in the record, so
everything a Signal reports about itself (`transmitted`, `rf_report`, `lo_phase`, `subchain`,
`realize`, `filter_delays`) is available per channel; its recipe is the record's.

A sub-chain written as `Signal()` with no grid starts in the domain its own ops need, so a real
supply or enable sits inside an I/Q record without ceremony.
