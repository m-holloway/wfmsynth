"""An OFDM frame -- preamble, payload, gap -- through an echo and noise, measured against truth.

Run from the repository root after ``pip install -e .``:

    python examples/ofdm_frame.py

WHAT THIS IS AN EXAMPLE OF
--------------------------
`Signal.ofdm(...)`, the multicarrier source, and what is deliberately NOT part of it:

    ofdm.register("demo_20mhz", n_fft=64, scs_hz=312.5e3, cp=16, active=52, ...)   # your names
    payload = Signal().ofdm("demo_20mhz", constellation="qam16", prbs=15, n_symbols=20)
    frame = Signal(seed=1, grid=g).sequence([{"chain": preamble, "duration_s": ...},
                                             {"chain": payload, "duration_s": ...}, ...])
    sent, t_s = frame.transmitted()                    # the resource grid, row per symbol
    rf.evm(x, g, sent, ofdm=payload.ops[0], correct=("equalizer",))["removed"]

The op makes symbols. The FRAME is a `sequence` of chains; the GAP is a burst shorter than
its segment; the PREAMBLE is an `ofdm` chain whose every element is a pilot; the CHANNEL is a
`filter`; the NOISE is `awgn`. The parameter set's name is the caller's and travels in the
recipe beside its numbers -- nothing here is, or claims to be, a standard's waveform.

THE THREE THINGS WORTH READING OFF THE OUTPUT
---------------------------------------------
1. The envelope is near-Gaussian: its peaks stand several dB higher over its average than a
   single carrier's with the same constellation. That, not the constellation, is what an
   amplifier downstream has to survive.
2. An echo inside the cyclic prefix costs nothing in orthogonality: it scales each
   subcarrier by one complex number H(k). Raw EVM counts that in full; naming `equalizer`
   takes it out and RETURNS it. What is left is the noise -- divided by H(k) with the
   signal, so larger on the subcarriers the echo faded -- at the value that predicts.
3. The grid has to carry the FFT rate (or a whole multiple). The PRBS's start matters too:
   left at its default state the first symbols are nearly all one constellation point, and
   an inverse FFT of a constant is an impulse. State `prbs_seed=`.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import wfmsynth as ws
from wfmsynth import ofdm, rf

N_FFT, SCS, CP = 64, 312.5e3, 16
RATE = N_FFT * SCS                               # 20 MSa/s: what the grid must carry
SYMBOL_S = (N_FFT + CP) / RATE                   # 4 us, prefix included
K = list(range(-26, 0)) + list(range(1, 27))     # 52 subcarriers, the centre left out
PRE, PAY, GAP, FRAMES = 2, 20, 8, 3              # symbols per frame, and frames
SNR_DB = 25.0
ECHO = [1.0, 0.0, 0.0, 0.0, 0.0, 0.35]           # a second path 5 samples late, 9 dB down


def frame_record():
    ofdm.register("demo_20mhz", n_fft=N_FFT, scs_hz=SCS, cp=CP, active=52,
                  pilots=[-21, -7, 7, 21], source="examples/ofdm_frame.py; not a standard")
    try:
        payload = ws.Signal().ofdm("demo_20mhz", constellation="qam16", prbs=15, prbs_seed=0x2F6E15,
                                   n_symbols=PAY)
    finally:
        ofdm.unregister("demo_20mhz")            # the recipe holds the numbers from here on
    known = [1.0 if (i * 7) % 5 < 3 else -1.0 for i in range(52)]
    preamble = ws.Signal().ofdm(n_fft=N_FFT, scs_hz=SCS, cp=CP, active=52,
                                pilots={"subcarriers": K, "values": known}, n_symbols=PRE)
    n_sym = FRAMES * (PRE + PAY + GAP)
    g = ws.Grid(fs=RATE, n=(N_FFT + CP) * n_sym)
    segments = []
    for _ in range(FRAMES):
        segments += [{"chain": preamble, "duration_s": PRE * SYMBOL_S},
                     {"chain": payload, "duration_s": (PAY + GAP) * SYMBOL_S}]   # burst, then quiet
    del segments[-1]["duration_s"]               # the last segment runs to the record's end
    return ws.Signal(seed=1, grid=g).sequence(segments), payload, g


def main():
    frame, payload, g = frame_record()
    bw = ofdm.plan(payload.ops[0], g)["noise_bw_hz"]
    rx = (frame.copy().filter("fir", taps=ECHO)
          .awgn(snr_db=SNR_DB, bw_hz=bw, ref_power=1.0))     # against the burst, not the gap
    x, events = rx.realize()
    sent, t_s = rx.transmitted()
    op = rx.recipe()["ops"][0]["segments"][1]["chain"]["ops"][0]
    print(f"record: {len(x)} samples at {g.fs / 1e6:g} MSa/s, {sent.shape[0]} symbols of "
          f"{sent.shape[1]} subcarriers sent; "
          f"{sum(e.kind == 'ofdm_symbol' for e in events)} symbol-boundary events, "
          f"{sum(e.kind == 'segment' for e in events)} segment boundaries")
    print(f"recipe: the payload is '{op['name']}' ({op['source']}): n_fft={op['n_fft']}, "
          f"scs_hz={op['scs_hz']:g}, cp={op['cp']}, active={op['active']}, pilots={op['pilots']}")

    # 1. the envelope's statistics, on the first payload burst of the CLEAN record
    a = PRE * (N_FFT + CP)
    burst = frame.waveform()[a:a + PAY * (N_FFT + CP)]
    single = ws.Signal(seed=1, grid=ws.Grid(fs=8 * 1e6, baud=1e6, n=1 << 14)).iq(
        "qam16", prbs=15, prbs_seed=0x2F6E15).waveform()[256:-256]
    for name, rec in (("ofdm, 52 x qam16", burst), ("single carrier qam16", single)):
        c = rf.ccdf(rec, levels_db=[6.0])
        print(f"  {name:<22} peak/average {c['papr_db']:5.2f} dB, "
              f"{100 * c['probability'][0]:.3f} % of samples more than 6 dB over the average "
              f"(a Gaussian envelope: {100 * np.exp(-10 ** 0.6):.3f} %)")

    # 2. EVM of the first payload, per resource element, against the truth
    truth = sent[PRE:PRE + PAY]
    params = dict(payload.ops[0])
    raw = rf.evm(x, g, truth, ofdm=params, delay_s=PRE * SYMBOL_S)
    fit = rf.evm(x, g, truth, ofdm=params, delay_s=PRE * SYMBOL_S, correct=("equalizer",))
    h = fit["removed"]["equalizer"]
    want = 1 + ECHO[5] * np.exp(-2j * np.pi * 5 * fit["subcarriers"] / N_FFT)
    n0 = [r for r in rx.rf_report() if r["name"] == "awgn"][0]["n0"]
    floor = np.sqrt(bw * n0 / np.mean(np.abs(truth) ** 2))
    print(f"EVM, first payload ({raw['n_elements']} elements): {raw['evm_pct']:.2f} % as it "
          f"stands, {fit['evm_pct']:.2f} % with the per-subcarrier equaliser named")
    print(f"  what the equaliser removed: |H| from {abs(h).min():.3f} to {abs(h).max():.3f} "
          f"across the subcarriers (the echo: {abs(want).min():.3f} to {abs(want).max():.3f}; "
          f"largest difference {abs(h - want).max():.3f}, which is noise)")
    faded = np.sqrt(np.mean(1.0 / np.abs(want) ** 2))        # the noise is divided by H too
    print(f"  predicted: noise {100 * floor:.2f} % on a flat channel, x {faded:.3f} for the "
          f"fades, x {np.sqrt(1 - 1 / PAY):.3f} the fit absorbed over {PAY} symbols = "
          f"{100 * floor * faded * np.sqrt(1 - 1 / PAY):.2f} %")

    # 3. where the power is
    spec = rf.spectrum(burst, g, rbw_hz=SCS / 4)
    total = rf.band_power(spec, spec["f_hz"][0], spec["f_hz"][-1])
    inside = rf.band_power(spec, -26.5 * SCS, 26.5 * SCS)
    print(f"occupied band: {100 * inside / total:.2f} % of the burst's power within "
          f"+/-{26.5 * SCS / 1e6:.2f} MHz, the 52 active subcarriers")


if __name__ == "__main__":
    main()
