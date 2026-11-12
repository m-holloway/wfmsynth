"""An I/Q recording out to SigMF and interleaved binary, and back in as the start of a chain.

Run from the repository root after ``pip install -e .``:

    python examples/iq_recordings.py

WHAT THIS IS AN EXAMPLE OF
--------------------------
Handing a record to another tool, and starting a chain from one another tool made:

    ws.to_sigmf("burst", x, sig.stored_grid(x), events=ev)        # out, with its ground truth
    sig = ws.read_iq("burst", format="sigmf")                     # in: grid taken from the file
    sig = ws.read_iq("burst.iq", format="iq_interleaved", dtype="int16", scale=s, fs_hz=250e6)

`read_iq` is `Signal(grid=<the file's>).capture(path=..., format=...)`, so everything after it
is an ordinary chain and the recipe replays it.

THE THREE THINGS WORTH READING OFF THE OUTPUT
---------------------------------------------
1. SigMF states its own rate, centre frequency and annotations, and the annotations come back
   as EVENTS -- the same events that were written, field for field.
2. An interleaved file states nothing. You say `dtype`, `byteorder` and `scale`, and the scale
   is the value of ONE COUNT. int16 holds the record to half a count.
3. A file read in the wrong byte order is refused, with the reason, instead of becoming a
   record of plausible noise. A wrong `scale` cannot be detected, so `describe` shows the level.
"""
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import wfmsynth as ws                                                        # noqa: E402

work = tempfile.mkdtemp()
g = ws.Grid(fs=250e6, baud=25e6, n=1 << 13, domain="iq", fc_hz=1e9)
src = ws.Signal(seed=1, grid=g).iq("qam16", prbs=15)
x = src.waveform()

# Ground truth to send along: where the first few symbols sit, as events.
sent, t_s = src.transmitted()
ev = ws.EventList([ws.Event(sample=int(round(t * g.fs)), kind="symbol", width=1, t_s=float(t),
                            placement="iq/0", params={"i": float(v.real), "q": float(v.imag)})
                   for v, t in zip(sent[:4], t_s[:4])], n=len(x))

# --- 1. SigMF -----------------------------------------------------------------------------
out = ws.to_sigmf(os.path.join(work, "burst"), x, src.stored_grid(x), events=ev,
                  description="QAM-16, 25 MBd")
sig = ws.read_iq(os.path.join(work, "burst"), format="sigmf")
y, got = sig.realize()
print(f"SigMF        {out['datatype']}  ->  fs {sig.grid.fs / 1e6:g} MHz, "
      f"fc {sig.grid.fc_hz / 1e9:g} GHz, {sig.grid.n} samples, all read from the metadata")
print(f"             samples equal the float32 that was written: "
      f"{np.array_equal(y, x.astype(np.complex64))}")
print(f"             {len(got)} annotations back as events, identical to the ones written: "
      f"{got.to_dicts() == ev.to_dicts()}")
print(f"             first: {got.events[0].kind!r} at sample {got.events[0].sample}, "
      f"params {got.events[0].params}")

# --- 2. interleaved int16 ------------------------------------------------------------------
# Full scale is +/-1 by default and this record peaks near 2, so the default is refused rather
# than clipped. The message names the scale that fits.
path = os.path.join(work, "burst.iq")
try:
    ws.to_interleaved(path, x, dtype="int16")
except ValueError as e:
    print(f"\nint16        refused at the default scale: ...{str(e)[14:118]}...")
scale = float(max(np.abs(x.real).max(), np.abs(x.imag).max())) / 32767
wrote = ws.to_interleaved(path, x, dtype="int16", scale=scale, events=ev)
back = ws.read_iq(path, format="iq_interleaved", dtype="int16", scale=scale,
                  fs_hz=g.fs, fc_hz=g.fc_hz)
err = max(np.abs(back.waveform().real - x.real).max(), np.abs(back.waveform().imag - x.imag).max())
d = ws.iqfile.describe(path, "iq_interleaved", dtype="int16", scale=scale)
print(f"             scale {scale:.3e} per count: worst error {err / scale:.3f} LSB; "
      f"peak {d['peak_counts']} counts = {100 * d['full_scale_fraction']:.1f} % of full scale")
print(f"             marker channel {os.path.basename(wrote['markers'])}: "
      f"{int(np.fromfile(wrote['markers'], np.uint8).sum())} samples marked")

# --- 3. the wrong byte order ---------------------------------------------------------------
try:
    ws.read_iq(path, format="iq_interleaved", dtype="int16", byteorder="big", scale=scale,
               fs_hz=g.fs)
except ValueError as e:
    print(f"\nwrong order  refused: ...{str(e).split(': ', 2)[2]}")

# --- and it is a chain: the recipe replays it with the files gone ---------------------------
# No file states a symbol rate, so `baud=` goes on the grid for the matched filter to read.
keep = ws.read_iq(path, format="iq_interleaved", dtype="int16", scale=scale, fs_hz=g.fs,
                  fc_hz=g.fc_hz, baud=g.baud, embed=True).filter("rrc", beta=0.35, zero_phase=True)
recipe = keep.recipe()
for name in os.listdir(work):
    os.remove(os.path.join(work, name))
os.rmdir(work)
# the record is ALREADY matched-filtered, so its symbols are its samples at the centres and
# `iq.evm` is the arithmetic on symbols in hand (a fraction); `rf.evm` is the measure of a record
k = np.rint(t_s * g.fs).astype(int)[10:-10]
evm = ws.iq.evm(ws.Signal.from_recipe(recipe).waveform()[k], sent[10:-10])
print(f"\nembedded     replayed from the recipe alone, matched-filtered: EVM {100 * evm:.2f} % "
      f"against the symbols that were sent")
