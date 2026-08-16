"""
wfmsynth.bus — low-speed bus signaling primitives.

A scope measures far more than multi-gigabit serial: embedded buses (I2C, SPI, UART, CAN,
RS-485) at kHz–MHz rates, with their own signaling. Two things distinguish them from the
push-pull NRZ/PAM4 carriers:

  * open-drain / wired-AND (I2C, 1-Wire, CAN's dominant bit) — the line is pulled HIGH and any
    driver can only pull it LOW, so the bus is the logical AND of all drivers (a low anywhere
    wins). Bus contention and arbitration fall out of this.
  * framed slow signaling (UART) — an idle-high line with per-byte start/stop framing.

These are the primitives; a full protocol capture composes them (framing + open-drain + the
channel/probe effects the fast carriers already use).
"""
from __future__ import annotations

import numpy as np


def open_drain(drivers, high=1.0, low=0.0):
    """Open-drain / wired-AND bus. ``drivers`` is a list of 0/1 waveforms (1 = released/
    high-Z, 0 = actively pulling the line low). The line is ``high`` only where EVERY driver
    is released and ``low`` wherever any driver pulls — so a dominant (low) bit always wins,
    which is exactly bus contention / CAN arbitration."""
    d = np.stack([np.asarray(a) for a in drivers])
    released = np.all(d == 1, axis=0)
    return np.where(released, high, low)


def uart_frame(data_bytes, samples_per_bit=16, idle=1.0):
    """UART framing: an idle-high line with, per byte, a start bit (0), 8 LSB-first data bits,
    and a stop bit (1). Returns the bit-level waveform (``samples_per_bit`` samples per bit)."""
    bits = []
    for b in data_bytes:
        bits.append(0)                                  # start
        bits.extend((int(b) >> i) & 1 for i in range(8))   # 8 data bits, LSB first
        bits.append(1)                                  # stop
    return np.repeat(np.array(bits, float), int(samples_per_bit))


def uart_decode(wave, samples_per_bit=16):
    """Recover bytes from a UART waveform (inverse of `uart_frame`) — sample the middle of
    each bit of each 10-bit frame. Assumes the waveform starts at a frame."""
    spb = int(samples_per_bit)
    nbytes = len(wave) // (10 * spb)
    out = []
    for j in range(nbytes):
        base = j * 10 * spb
        b = 0
        for i in range(8):
            if wave[base + (1 + i) * spb + spb // 2] > 0.5:   # skip start bit, LSB first
                b |= (1 << i)
        out.append(b)
    return out
