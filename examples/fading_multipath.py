"""A radio link on an I/Q record: amplifier, cable, multipath, fading -- at 2.4 GHz, on a
4 MSa/s grid.

Run from the repository root after ``pip install -e .``:

    python examples/fading_multipath.py

WHAT THIS IS AN EXAMPLE OF
--------------------------
Everything here is stated at RF and rendered on the ENVELOPE, on a grid 600 times too slow to
hold the carrier:

    law("saleh")          the amplifier: AM/AM and AM/PM on the envelope's magnitude
    lossy                 a cable, 3 dB at 2.4 GHz: its response is read at fc + f
    delay                 a path's length: the envelope is shifted and the carrier turned
    gain(by=complex)      a path's amplitude and phase; by a chain, a fade
    recipes.fading        complex noise -> filter("doppler"): a Rayleigh or Rician gain
    recipes.multipath     combine of delayed, scaled copies of one transmitter

The last two are plain functions that return a chain of the ops above them. Nothing is hidden:
section 2 prints the chain, and section 4 changes one op of it.

WHAT THE LIBRARY DOES NOT DO HERE
---------------------------------
It has no channel models by name. Which delays and powers a standard calls "pedestrian" or
"urban" is a table with a document behind it; the three paths below are this example's own.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import wfmsynth as ws
from wfmsynth import Signal

FS, BAUD, FC = 4e6, 1e6, 2.4e9
GRID = ws.Grid(fs=FS, baud=BAUD, n=1 << 16, fc_hz=FC)        # 16.4 ms, 16384 symbols
SALEH = dict(alpha_a=2.1587, beta_a=1.1517, alpha_phi_rad=4.0033, beta_phi=9.1040)
DELAYS_S = [0.0, 1.5e-6, 4.0e-6]
GAINS = [1.0, 0.5, 0.25]
DOPPLER_HZ = 600.0                                           # 270 km/h at 2.4 GHz


def transmitter(backoff=0.25):
    """QAM-16 into a travelling-wave-tube amplifier, `backoff` below its saturation drive."""
    return Signal(seed=1, grid=GRID).iq("qam16", prbs=15, amp=backoff).law("saleh", **SALEH)


def evm(sig, **kw):
    x = sig.waveform()
    sent, _ = sig.transmitted()
    return ws.rf.evm(x, sig.stored_grid(x), sent, **kw)["evm_pct"]


print("1. The amplifier: AM/PM is what a constellation shows first")
for backoff in (0.05, 0.25, 0.5):
    r = np.array([backoff])
    amp, phi = ws.stages.am_am_pm("saleh", r, **SALEH)
    e = evm(transmitter(backoff), correct=("gain", "phase"))
    print(f"   drive {backoff:4.2f} of saturation: gain {amp[0] / backoff:.3f}, phase "
          f"{np.degrees(phi[0]):5.2f} deg at the rms level; EVM {e:5.2f} % with one gain and "
          f"phase fitted")

print("\n2. A cable and three paths, each fading: the whole chain")
link = ws.recipes.multipath(transmitter().lossy(loss_db=3.0, loss_at_ghz=2.4), DELAYS_S, GAINS,
                            doppler_hz=DOPPLER_HZ)
print("   " + link.summary().replace("\n", "\n   "))
print(f"   random roles: {link.roles()}")

print("\n3. What the record holds")
y = link.waveform()
tx = transmitter().lossy(loss_db=3.0, loss_at_ghz=2.4).waveform()
k = np.fft.fftfreq(len(tx))
X = np.fft.fft(tx)
A = np.stack([np.fft.ifft(X * np.exp(-2j * np.pi * k * d * FS)) for d in DELAYS_S], axis=1)
block = 256                                                  # 64 us: 4 % of a Doppler period
m = (len(y) - 512) // block
Ab = A[256:256 + m * block].reshape(m, block, 3)
yb = y[256:256 + m * block].reshape(m, block, 1)
Ah = np.conj(np.swapaxes(Ab, 1, 2))
taps = np.linalg.solve(Ah @ Ab, Ah @ yb)[..., 0]             # the channel, block by block
pdp = np.mean(np.abs(taps) ** 2, axis=0)
print(f"   power delay profile over {m} blocks: "
      + ", ".join(f"{d * 1e6:g} us {p:.3f}" for d, p in zip(DELAYS_S, pdp))
      + f"   (stated {', '.join(f'{g * g:.3f}' for g in GAINS)}; {DOPPLER_HZ * GRID.duration:.0f} "
        f"Doppler periods is a short average)")
deep = 20 * np.log10(np.abs(taps[:, 0]).min() / np.sqrt(np.mean(np.abs(taps[:, 0]) ** 2)))
print(f"   the first path's deepest fade in this record: {deep:.1f} dB below its rms")

print("\n4. One op changed: the first path given a line of sight")
los = ws.recipes.multipath(transmitter().lossy(loss_db=3.0, loss_at_ghz=2.4), DELAYS_S, GAINS,
                           doppler_hz=DOPPLER_HZ)
los.ops[3]["by"] ={"ops": ws.recipes.fading(DOPPLER_HZ, k_db=10.0).ops}     # path 0's fade
for name, sig in (("Rayleigh", link), ("Rician, K = 10 dB", los)):
    env = np.abs(sig.subchain("gain/3/by"))
    print(f"   {name:18s} first path's gain: {20 * np.log10(env.min()):6.1f} dB at its deepest, "
          f"{100 * np.mean(env < 0.316):4.1f} % of the record more than 10 dB down")

print("\n5. A delay is also a phase, and a moving delay a Doppler shift")
for d in (0.0, 0.1e-9, 0.2083e-9):
    g = np.vdot(tx[64:-64], Signal(seed=1, grid=GRID).iq("qam16", prbs=15, amp=0.25)
                .law("saleh", **SALEH).lossy(loss_db=3.0, loss_at_ghz=2.4)
                .delay(by=d).waveform()[64:-64])
    print(f"   {d * 1e9:6.4f} ns of path: the envelope barely moves, the carrier turns "
          f"{np.degrees(np.angle(g)):7.2f} deg")
ramp = Signal().carrier("dc", level=2.5e-7).filter("integrate", tau_s=1.0)   # 75 m/s away
moving = Signal(seed=1, grid=GRID).capture(values=np.ones(GRID.n, complex)).delay(by=ramp)
ph = np.unwrap(np.angle(moving.waveform()[64:-64]))
print(f"   a path lengthening at 75 m/s: carrier moved by "
      f"{np.polyfit(np.arange(len(ph)) / FS, ph, 1)[0] / (2 * np.pi):.1f} Hz "
      f"(-fc * v/c = {-FC * 2.5e-7:.1f})")
