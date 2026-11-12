"""A QAM-16 transmitter with a compressing amplifier, measured the way a signal analyser would.

Run from the repository root after ``pip install -e .``:

    python examples/rf_measures.py

WHAT THIS IS AN EXAMPLE OF
--------------------------
`wfmsynth.rf`: spectrum, adjacent-channel leakage, the envelope's power statistics and EVM,
on a record whose transmitted symbols are known.

    x = sig.waveform()                      # complex baseband
    g = sig.stored_grid(x)                  # the grid that describes it
    sent, _ = sig.transmitted()             # the ground truth
    rf.evm(x, g, sent, skip=8)              # nothing fitted
    rf.evm(x, g, sent, skip=8, correct=("gain", "phase"))   # and ["removed"] says what went

The amplifier is not a new op. It is `law("tanh")` on the real passband record between
`upconvert` and `downconvert`: a memoryless voltage law at RF, which is what compression is.

THE THREE THINGS WORTH READING OFF THE OUTPUT
---------------------------------------------
1. Compression shows up in every measure at once: the neighbouring channel fills in (ACLR
   falls), the peaks are clipped (PAPR falls) and the symbols move (EVM rises).
2. EVM with nothing fitted is dominated by things that are not distortion: the probe's delay
   turned the constellation and the amplifier lost gain. Naming `gain` and `phase` takes those
   out, and the result says how many dB and how many degrees they were. The figure left over
   is the distortion.
3. A receiver without the data has to decide what was sent. At a low signal-to-noise ratio it
   decides wrongly, and its EVM reads LOWER than the truth. Only the ground truth shows it.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import wfmsynth as ws
from wfmsynth import rf

FC, BAUD, BETA = 1e9, 25e6, 0.35
G = ws.Grid(fs=6.4e9, baud=BAUD, n=1 << 19, domain="iq", fc_hz=FC)
LPF = dict(pass_hz=80e6, stop_hz=160e6)        # wide enough to keep the neighbouring channel
CHANNEL = (1 + BETA) * BAUD                    # the occupied bandwidth of the signal
NEXT = 40e6                                    # where the neighbouring channel sits
SKIP = 8                                       # end symbols whose neighbours are missing


def transmitter(sat=None):
    s = ws.Signal(seed=1, grid=G).iq("qam16", prbs=15, prbs_phase=100, beta=BETA).upconvert()
    if sat is not None:
        s = s.law("tanh", sat=sat)             # the amplifier: soft compression of the voltage
    return s.probe(bw_hz=3e9).downconvert(fc_hz=FC, lpf=LPF, fs_out=400e6)


def main():
    print(f"{'amplifier':<22}{'ACLR dB':>9}{'PAPR dB':>9}{'EVM raw %':>11}{'EVM fitted %':>14}"
          f"   removed")
    for name, sat in (("linear", None), ("saturates at 4.0", 4.0), ("saturates at 2.5", 2.5)):
        sig = transmitter(sat)
        x = sig.waveform()
        g = sig.stored_grid(x)
        sent, _ = sig.transmitted()
        leak = rf.aclr(x, g, channel_bw_hz=CHANNEL, offsets_hz=[NEXT])
        inner = x[len(x) // 16: -len(x) // 16]            # leave the burst's turn-on out
        raw = rf.evm(x, g, sent, skip=SKIP)
        fit = rf.evm(x, g, sent, skip=SKIP, correct=("gain", "phase"))
        rem = fit["removed"]
        print(f"{name:<22}{leak['aclr_db'][0]:>9.1f}{rf.ccdf(inner)['papr_db']:>9.2f}"
              f"{raw['evm_pct']:>11.2f}{fit['evm_pct']:>14.2f}"
              f"   {rem['gain_db']:+.2f} dB, {np.degrees(rem['phase_rad']):+.1f} deg")

    # the spectrum behind the ACLR figure, in the analyser's terms
    spec = rf.spectrum(x, g, rbw_hz=1e6)
    print(f"\nspectrum: {len(spec['f_hz'])} points, {spec['window']} window, resolution "
          f"bandwidth {spec['rbw_hz'] / 1e6:.2f} MHz, {spec['segments']} averages, "
          f"offsets from {spec['fc_hz'] / 1e9:.1f} GHz")
    print(f"  channel power {rf.band_power(spec, -CHANNEL / 2, CHANNEL / 2):.3f}, of "
          f"{np.mean(np.abs(x) ** 2):.3f} in the whole record")

    # a decided reference flatters; the truth says by how much
    clean = ws.Signal(seed=1, grid=ws.Grid(fs=400e6, baud=BAUD, n=1 << 16, domain="iq"))
    y = clean.iq("qam16", prbs=15, prbs_phase=100, beta=BETA).waveform()
    sent, _ = clean.transmitted()
    print("\nnoise added by hand at a stated Es/N0 (EVM against truth is 10**(-EsN0/20)):")
    rng = np.random.default_rng(0)
    for esn0_db in (25, 15, 10):
        var = clean.grid.fs / (BAUD * 10 ** (esn0_db / 10))       # per complex sample
        noisy = y + np.sqrt(var / 2) * (rng.standard_normal(len(y))
                                        + 1j * rng.standard_normal(len(y)))
        truth = rf.evm(noisy, clean.grid, sent, skip=SKIP)
        blind = rf.evm(noisy, clean.grid, sent, skip=SKIP, reference="decided",
                       constellation="qam16")
        print(f"  Es/N0 {esn0_db} dB: analytic {100 * 10 ** (-esn0_db / 20):5.2f} %, against "
              f"truth {truth['evm_pct']:5.2f} %, decided {blind['evm_pct']:5.2f} % "
              f"({blind['decision_errors']} of {blind['n_symbols']} decisions wrong)")

    cells = rf.constellation_density(truth["received"], bins=48)
    print(f"\nconstellation_density: a {cells['density'].shape} grid of counts over "
          f"+/-{cells['i_edges'][-1]:.2f}, {int(cells['density'].sum())} symbols")

    try:
        passband = ws.Signal(seed=1, grid=G).iq("qam16", prbs=15).upconvert().waveform()
        rf.evm(passband, G, sent)                         # a real record has no symbols yet
    except ValueError as e:
        print(f"\nrefused, as it should be:\n  {e}")


if __name__ == "__main__":
    main()
