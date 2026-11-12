"""A QAM-16 radio signal, captured by a scope at RF, and what the capture did to it.

Run from the repository root after ``pip install -e .``:

    python examples/iq_passband.py

WHAT THIS IS AN EXAMPLE OF
--------------------------
A record that is COMPLEX BASEBAND -- I + jQ about a centre frequency -- and the two ops that
move between it and an ordinary real voltage:

    g = ws.Grid(fs=16e9, baud=25e6, n=1 << 18, domain="iq", fc_hz=1e9)
    s = (ws.Signal(seed=1, grid=g)
         .iq("qam16", prbs=15)                     # complex symbols, root-raised-cosine
         .upconvert()                              # -> a real 1 GHz passband voltage
         .acquire(scope)                           # ...which a scope can look at
         .downconvert(fc_hz=1e9, lpf=dict(pass_hz=20e6, stop_hz=60e6), fs_out=250e6))

Between `upconvert` and `downconvert` the record is a voltage like any other, so the probe,
front end, sampler and converter this library already has model an instrument looking at RF.
Nothing in that part of the chain knows it is carrying a radio signal.

THE THREE THINGS WORTH READING OFF THE OUTPUT
---------------------------------------------
1. The source alone demodulates to its own symbols. `transmitted()` is the ground truth, and
   the small residual is the pulse being truncated to `span_ui` symbols, nothing else.
2. A RAW comparison after the scope is terrible, and it is not noise. The front end has group
   delay, a delay at 1 GHz is a rotation of the constellation, and `rf.evm` fits nothing
   unless asked. `correct=("gain", "phase")` takes it out, which is what a receiver's carrier
   recovery does, and `["removed"]` says how far it had turned.
3. What is left is the instrument: 8 bits and a 3 GHz front end, in numbers.

WHAT THE LIBRARY REFUSES HERE
-----------------------------
A real-only op on the I/Q record (the last lines provoke one, to show the message), a grid too
slow for the carrier, and a low-pass that would pass the mixing image. It never resamples
behind your back and never takes the real part of anything.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import wfmsynth as ws
from wfmsynth import iq

FC, BAUD = 1e9, 25e6                       # a 1 GHz carrier, 25 MBd
FS = 16e9                                  # 16 samples per carrier cycle on the synthesis grid
G = ws.Grid(fs=FS, baud=BAUD, n=1 << 18, domain="iq", fc_hz=FC)
LPF = dict(pass_hz=20e6, stop_hz=60e6)     # the analysis filter: flat to 20 MHz, 80 dB by 60
EDGE = 16                                  # symbols at each end whose neighbours are missing


def source():
    return ws.Signal(seed=1, grid=G).iq("qam16", prbs=15, prbs_phase=100)


def main():
    # 1. the source, and its ground truth
    src = source()
    x = src.waveform()
    sent, t_s = src.transmitted()
    print(f"source: {x.dtype}, {len(x)} samples, {len(sent)} QAM-16 symbols, "
          f"first centred at {t_s[0] * 1e9:.0f} ns")
    print(f"  matched filter against transmitted(): EVM "
          f"{ws.rf.evm(x, G, sent, skip=EDGE)['evm_pct']:.3f} %  (pulse truncation)")

    # 2. the same signal through an instrument at RF
    scope = ws.AcquisitionProfile(sample_rate_hz=4e9, record_length=1 << 16,
                                  input_bandwidth_hz=3e9, bits=8)
    rf = source().upconvert().acquire(scope).downconvert(fc_hz=FC, lpf=LPF, fs_out=250e6)
    y = rf.waveform()
    stored = rf.stored_grid(y)             # NOT G: the scope stored at its own rate
    print(f"\nchain: {rf.stage_kinds(domains=True)}")
    print(f"stored: {y.dtype}, {len(y)} samples at {stored.fs / 1e6:.0f} MSa/s, "
          f"domain {stored.domain!r} about {stored.fc_hz / 1e9:.1f} GHz")
    raw = ws.rf.evm(y, stored, sent, skip=EDGE)                    # nothing fitted
    fit = ws.rf.evm(y, stored, sent, skip=EDGE, correct=("gain", "phase"))
    print(f"  raw EVM {raw['evm_pct']:.1f} %  -- the front end turned the "
          f"constellation {np.degrees(fit['removed']['phase_rad']):+.1f} degrees")
    print(f"  EVM with that one gain and rotation taken out: "
          f"{fit['evm_pct']:.2f} %  (8 bits, 3 GHz front end)")

    # 3. what the stated filter costs, measured on its own taps
    pass_err, stop_db = iq.lowpass_error(LPF, 4e9)
    print(f"\nlow-pass at the scope's 4 GSa/s: {len(iq.lowpass(LPF, 4e9))} taps, "
          f"passband within {pass_err:.1e}, stopband {stop_db:.1f} dB")

    # 4. the refusal
    try:
        source().lossy(loss_db=6.0, loss_at_ghz=1.0)
    except ValueError as e:
        print(f"\nrefused, as it should be:\n  {e}")


if __name__ == "__main__":
    main()
