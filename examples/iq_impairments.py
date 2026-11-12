"""A QAM-16 transmitter that is not ideal, and an error-vector budget that adds up.

Run from the repository root after ``pip install -e .``:

    python examples/iq_impairments.py

WHAT THIS IS AN EXAMPLE OF
--------------------------
The things a real radio does to an I/Q record, ONE OP PER EFFECT, each stated in the unit an
engineer would state it in:

    s = (ws.Signal(seed=1, grid=g).iq("qam16", prbs=15)
         .dac(bits=8, headroom_pct=5)                      # the converter
         .iq_imbalance(gain_db=0.3, phase_deg=2.0)         # the modulator's two rails
         .iq_offset(i=0.02, q=-0.01)                       # carrier leakage
         .phase_noise(profile=[(1e4, -75), (1e5, -95), (1e6, -115)])   # the oscillator
         .freq_offset(f_hz=2e3)                            # the receiver is 2 kHz off
         .awgn(es_n0_db=25))                               # the channel

    s.rf_report()     # what each request came to on this record
    s.lo_phase()      # the oscillator's phase at every sample -- the label, not an estimate

THE THREE THINGS WORTH READING OFF THE OUTPUT
---------------------------------------------
1. Every impairment's contribution to EVM can be PREDICTED from `rf_report()` alone, and the
   record agrees: the image, the leak, the rms phase, the quantisation noise that falls in
   the symbol band, and 1/sqrt(Es/N0). That is what "ground truth in the recipe" buys.
2. A frequency offset is not an error-vector magnitude at all -- it is a rotation that grows
   -- so it is read back as the slope of the symbols' phase instead.
3. The contributions add in power. The whole chain's EVM is the root-sum-square of its own
   report once the offset's known rotation is taken out -- the receiver's carrier recovery,
   done with the answer in hand -- and taking out all of `lo_phase()` removes the phase
   noise's share as well.

WHAT THE LIBRARY REFUSES HERE
-----------------------------
An SNR with no bandwidth (the last lines provoke it), a phase-noise profile the record cannot
resolve, a DAC driven past full scale without `clip="hard"`, and all of these ops on a real
voltage record. There is no `image` op: an image is what imbalance produces, and
`rf.imbalance_for(rejection_db)` gives the imbalance that makes one.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import wfmsynth as ws
from wfmsynth import rf

BAUD = 25e6
G = ws.Grid(fs=16 * BAUD, baud=BAUD, n=1 << 16, domain="iq", fc_hz=2.4e9)
EDGE = 16                                  # symbols at each end whose neighbours are missing
PROFILE = [(1e4, -75), (1e5, -95), (1e6, -115)]

# each impairment, as the one line that adds it
STEPS = {
    "dac": lambda s: s.dac(bits=8, headroom_pct=5),
    "iq_imbalance": lambda s: s.iq_imbalance(gain_db=0.3, phase_deg=2.0),
    "iq_offset": lambda s: s.iq_offset(i=0.02, q=-0.01),
    "phase_noise": lambda s: s.phase_noise(profile=PROFILE),
    "awgn": lambda s: s.awgn(es_n0_db=25, ref_power=1.0),
}


def source():
    return ws.Signal(seed=1, grid=G).iq("qam16", prbs=15, prbs_phase=100)


def evm_of(sig, derotate=None):
    """EVM of the record's matched-filtered symbols against the ones sent, one complex gain
    fitted (what a receiver's carrier recovery does), the edge symbols left out."""
    x = sig.waveform()
    if derotate is not None:
        x = x * np.exp(-1j * derotate)
    sent, _ = sig.transmitted()
    return rf.evm(x, sig.grid, sent, skip=EDGE, correct=("gain", "phase"))["evm"]


def predicted(name, r):
    """One impairment's EVM, from its `rf_report()` entry and nothing else."""
    if name == "dac":            # white over fs; the matched filter keeps baud/fs of it
        return 10 ** (-r["sqnr_db"] / 20) * np.sqrt(BAUD / G.fs)
    if name == "iq_imbalance":   # the image lands on the symbols as uncorrelated error
        return 10 ** (-r["image_rejection_db"] / 20)
    if name == "iq_offset":      # a constant, through a filter of unit DC gain
        return 10 ** (r["leak_dbc"] / 20)
    if name == "phase_noise":    # a small rotation theta is an error of theta
        return r["realised_rms_rad"]
    return 10 ** (-r["es_n0_db"] / 20)


def main():
    floor = evm_of(source())
    print(f"the source alone: EVM {100 * floor:.2f} %  (the pulse is truncated to 16 symbols)\n")
    print(f"  {'impairment':<13}{'rf_report() says':<34}{'predicted':>10}{'measured':>10}")
    for name, add in STEPS.items():
        s = add(source())
        r = s.rf_report()[0]
        key = {"dac": "sqnr_db", "iq_imbalance": "image_rejection_db", "iq_offset": "leak_dbc",
               "phase_noise": "realised_rms_rad", "awgn": "es_n0_db"}[name]
        want = float(np.hypot(predicted(name, r), floor))
        print(f"  {name:<13}{key + ' = ' + format(r[key], '.4g'):<34}"
              f"{100 * want:>9.2f}%{100 * evm_of(s):>9.2f}%")

    # ---- the frequency offset: a slope, not an EVM
    cfo = source().freq_offset(f_hz=2e3)
    sent, _ = cfo.transmitted()
    took = rf.evm(cfo.waveform(), G, sent, skip=EDGE, correct=("frequency",))["removed"]
    print(f"\n  freq_offset  f_hz = 2000       read back from the symbols' phase: "
          f"{took['frequency_hz']:.1f} Hz")

    # ---- everything at once
    s = source()
    for add in STEPS.values():
        s = add(s)
        if s.ops[-1]["op"] == "phase_noise":
            s = s.freq_offset(f_hz=2e3)
    # the budget of THIS chain, from its own report: its oscillator and its noise are their
    # own draws (roles phase_noise/4 and awgn/6), not the ones in the table above
    total = np.sqrt(floor ** 2 + sum(predicted(r["name"], r) ** 2 for r in s.rf_report()
                                     if r["name"] != "freq_offset"))
    print(f"\nall of it, {len(s.ops) - 1} impairments, roles {s.roles()}:")
    print(f"  raw EVM (one gain fitted)                {100 * evm_of(s):7.2f} %   "
          f"<- the offset turned the constellation {2e3 * G.n / G.fs * 360:.0f} degrees")
    # the oscillator's phase is ground truth; take out the offset's part and what is left is
    # the budget above (the phase noise is already in it)
    ramp = rf.lo_ramp(G.n, G.fs, 2e3)
    print(f"  with the known offset removed            {100 * evm_of(s, ramp):7.2f} %")
    print(f"  root-sum-square from this chain's report {100 * total:7.2f} %")
    print(f"  with all of lo_phase() removed           {100 * evm_of(s, s.lo_phase()):7.2f} %   "
          f"<- the phase noise gone too")

    # ---- a refusal, on purpose
    try:
        source().awgn(snr_db=25).waveform()
    except ValueError as e:
        print(f"\nrefused, as it should be:\n  {str(e)[:150]}...")


if __name__ == "__main__":
    main()
