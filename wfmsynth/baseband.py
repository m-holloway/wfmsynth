"""wfmsynth.baseband -- a channel stated at RF, applied to the ENVELOPE of a signal about fc_hz.

A channel op is stated at real frequencies: `lossy(loss_db=3, loss_at_ghz=2.4)` loses 3 dB at
2.4 GHz, and a Touchstone file holds what was measured from 2 to 3 GHz. On a real passband
record the op meets the signal where it is. An I/Q record holds only the envelope, and the
same op applied to those samples as they stand would filter about DC -- a different channel.
The BASEBAND EQUIVALENT is the same response read where the signal actually is:

    H_bb(f) = H(fc_hz + f)          f the offset from the centre, -fs/2 .. +fs/2

so the envelope comes out as it would from `upconvert` -> the op -> `downconvert`, without a
grid fast enough to hold the carrier (2.4 GHz on a 20 MSa/s record). The response is NOT
conjugate-symmetric about the centre, so the result is complex whatever the input: a slope
across the band turns I into Q, which is the point.

    g = Grid(fs=40e6, baud=10e6, n=1 << 14, fc_hz=2.4e9)
    Signal(seed=1, grid=g).iq("qam16", prbs=15).lossy(loss_db=20.0, loss_at_ghz=2.4, causal=True)

WHICH OPS, AND WHY ONLY THESE. `lossy` and `sparam` are each ONE stated frequency response, so
reading it at fc + f is the whole translation. The ops built on a time delay (`reflect`,
`cascade`, `resonant_reflect`) also turn the carrier by -2*pi*fc*delay per bounce, and
`crosstalk` takes a second signal whose own centre would have to be stated; none of those is
opted in. A reflection on an envelope is `combine` of `delay`ed copies (`wfmsynth.recipes`).

WHAT IS EXACT AND WHAT IS A CHOICE.

  * the MAGNITUDE is the op's own law at fc + f, exactly, on any grid.
  * `sparam` reads the measured complex points at fc + f: nothing is assumed.
  * `lossy(causal=True)` needs a PHASE, and the minimum phase at one frequency depends on the
    loss at every frequency. A real record takes it over its own band, 0 to its Nyquist. An
    envelope has no such band, so one is stated: the phase is the minimum phase of the loss
    law from 0 to `passband_top(fs, fc_hz)`, the first whole multiple of fs/2 that holds
    fc + fs/2. That is what the passband route computes on a real grid whose Nyquist is that
    frequency, and the two agree to the bound tests/test_baseband_channel.py states. Another
    top frequency is, to first order, another bulk delay.
  * a frequency the envelope's band reaches below 0 Hz (fc < fs/2) is read at |f| and
    conjugated, as a real channel's response is there.

ONE THING THE ENVELOPE'S RESPONSE IS NOT. Over the whole of -fs/2 .. +fs/2 it is a slice of
the channel's response, cut at the band's two edges, and a slice is not the response of a
causal filter even when the channel is causal: the cut rings, before t = 0 as well as after.
That ringing lives at the band's edges, where an envelope that fits its grid holds nothing
(`upconvert` refuses one that does not), so it does not reach the record; it is why the
agreement above is stated for a band-limited envelope.

The convolution is LINEAR, as for the real ops (`physics.apply_transfer`): the record is
zero-padded past the response's own length. `linear=False` is the circular form; `guard=`
states the length. `method="overlap"` is refused: the block path is real-only.
"""
from __future__ import annotations

import numpy as np

from . import physics as P

AUX_MAX = 1 << 22          # points of the passband axis the minimum phase is built on, at most
AUX_PER_BAND = 4096        # ... and at least this many of them across the envelope's own band


def passband_top(fs, fc_hz):
    """The top of the passband axis a causal `lossy` builds its minimum phase on, in Hz: the
    first whole multiple of fs/2 at or above fc_hz + fs/2 (module docstring)."""
    return 0.5 * fs * int(np.ceil((fc_hz + 0.5 * fs) / (0.5 * fs) - 1e-9))


def _centre(fc_hz, who):
    if fc_hz is None:
        raise ValueError(
            f"{who}: this I/Q record's centre frequency was never stated, and a channel stated "
            f"at RF is read at fc + f. State it on the grid (Grid(..., fc_hz=...)); after a "
            f"`downconvert` it is that op's fc_hz.")
    return float(fc_hz)


def _extent(make_H, n, rel=P.RESPONSE_REL):
    """`physics.response_extent` for a two-sided response, capped as `apply_transfer` caps it."""
    cap = int(min(P.PROBE_MAX, max(4 * n, 4 * P.PROBE_N0)))
    m = int(min(P.PROBE_N0, max(8, n)))
    while True:
        h = np.abs(np.fft.ifft(make_H(m)))
        peak = float(h.max())
        if peak == 0.0:
            return 1
        extent = P._circular_support(h >= rel * peak)
        if extent <= m // 2 or m >= cap:
            return min(extent, 2 * n)
        m *= 2


def apply(x, make_H, linear=True, guard=None, method="fft", who="baseband"):
    """Apply a two-sided response to a complex envelope as a LINEAR convolution.

    `make_H(nfft)` returns the response on the ``np.fft.fftfreq(nfft)`` axis (offsets from the
    centre, in cycles per sample). The complex counterpart of `physics.apply_transfer`."""
    if method != "fft":
        raise ValueError(
            f"{who}: method={method!r} on an I/Q record. The block (overlap-save) path is "
            f"written for a real record; on an envelope the channel is applied in one "
            f"transform. Drop method=.")
    x = np.asarray(x, complex)
    n = len(x)
    if not linear:
        return np.fft.ifft(np.fft.fft(x) * make_H(n))
    guard = max(1, int(_extent(make_H, n) if guard is None else guard))
    nfft = P.linear_fft_length(n, guard)
    X = np.fft.fft(x, nfft)
    X *= make_H(nfft)
    return np.fft.ifft(X)[:n].copy()


def _signed(f_hz, of_abs):
    """A real channel's response at signed frequencies: `of_abs(|f|)`, conjugated below 0."""
    H = np.asarray(of_abs(np.abs(f_hz)), complex)
    return np.where(f_hz < 0, np.conj(H), H)


def lossy_response(f_hz, fs, fc_hz, causal=False, **law):
    """The baseband-equivalent response of `lossy` at offsets `f_hz` from `fc_hz`, for an
    envelope sampled at `fs`. `law` is `physics.insertion_loss_db`'s (length_in, tand, eps_r,
    skin_k, loss_db, loss_at_ghz, trend, trend_floor_db)."""
    f_hz = np.asarray(f_hz, float)
    at = fc_hz + f_hz
    if not causal:
        return (10.0 ** (-P.insertion_loss_db(np.abs(at) / 1e9, **law) / 20.0)).astype(complex)
    top = passband_top(fs, fc_hz)
    m = int(round(top / (0.5 * fs)))                      # the passband rate is m * fs
    n_aux = 2 * P.next_smooth_length(min(AUX_MAX, m * max(len(f_hz), AUX_PER_BAND)) // 2)
    f_aux = np.fft.rfftfreq(n_aux) * 2.0 * top
    H = P._min_phase_H(10.0 ** (-P.insertion_loss_db(f_aux / 1e9, **law) / 20.0), n_aux, half=True)
    # log-magnitude and unwrapped phase are both smooth in f, so they interpolate where the
    # complex value (which turns) would not
    lg, ph = np.log(np.abs(H) + 1e-300), np.unwrap(np.angle(H))
    return _signed(at, lambda f: np.exp(np.interp(f, f_aux, lg) + 1j * np.interp(f, f_aux, ph)))


def lossy(x, fs, fc_hz, causal=False, linear=True, guard=None, method="fft", **law):
    """`lossy` on a complex envelope about `fc_hz`: the loss law read at fc + f."""
    fc_hz = _centre(fc_hz, "lossy")

    def make_H(nfft):
        return lossy_response(np.fft.fftfreq(nfft, 1.0 / fs), fs, fc_hz, causal=causal, **law)

    return apply(x, make_H, linear=linear, guard=guard, method=method, who="lossy")


def sparam(x, fs, fc_hz, freqs, s21, linear=True, guard=None, band="refuse", dc="extend",
           band_tol=1e-3, method="fft"):
    """`sparam` on a complex envelope about `fc_hz`: the measured transfer read at fc + f.

    `band` and `dc` decide the edges exactly as for a real record (`sparam.sparam_channel`),
    with the envelope's power outside the file's band measured where it sits at RF."""
    from . import sparam as SP
    fc_hz = _centre(fc_hz, "sparam")
    x = np.asarray(x, complex)
    freqs, s21 = np.asarray(freqs, float), np.asarray(s21, complex)
    f_lo, f_hi = float(freqs.min()), float(freqs.max())
    at = np.abs(fc_hz + np.fft.fftfreq(len(x), 1.0 / fs))
    X = np.abs(np.fft.fft(x)) ** 2
    tot = float(X.sum())
    lost_lo, lost_hi = ((float(X[at < f_lo].sum()) / tot, float(X[at > f_hi].sum()) / tot)
                        if tot > 0 else (0.0, 0.0))
    SP.band_gate(lost_lo, lost_hi, f_lo, f_hi, fc_hz + 0.5 * fs, band, dc, band_tol,
                 fix=f"use a file that covers {max(fc_hz - 0.5 * fs, 0.0) / 1e9:.4g} to "
                     f"{(fc_hz + 0.5 * fs) / 1e9:.4g} GHz, band-limit the envelope with a "
                     f"`filter` first")

    def make_H(nfft):
        f = fc_hz + np.fft.fftfreq(nfft, 1.0 / fs)
        return _signed(f, lambda a: SP._interp_response(a, freqs, s21, band=band, dc=dc))

    return apply(x, make_H, linear=linear, guard=guard, method=method, who="sparam")
