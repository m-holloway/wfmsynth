"""wfmsynth.recipes -- useful ARRANGEMENTS of ops, as plain functions that return a chain.

THE RULE (docs/ARCHITECTURE.md, "What becomes an op"). An op is one mechanism nothing else
expresses. Anything that can be written as a chain of existing ops is not an op: it lives here,
as a function that returns a `Signal()` sub-chain. Nothing is hidden by that -- `summary()`,
the recipe, `roles()` and `contrast()` show the ops it is made of, and a caller can take the
chain and change one of them. A function here adds no mechanism and no parameter to any op.

    import wfmsynth as ws
    tx = ws.Signal(seed=1, grid=g).iq("qpsk", prbs=11)
    rx = ws.recipes.multipath(tx, delays_s=[0.0, 2e-6, 5e-6], gains=[1.0, 0.5, 0.25],
                              doppler_hz=200.0).awgn(es_n0_db=20)
    print(rx.summary())                                    # every op of every path

    fade = ws.recipes.fading(doppler_hz=200.0)             # a unit-power complex gain
    tx.gain(by=fade)
"""
from __future__ import annotations

import numpy as np

from .compose import Signal


def fading(doppler_hz, k_db=None, los_hz=0.0, shape="jakes"):
    """A FADING GAIN: the complex factor a moving, scattering path puts on an envelope, as a
    chain to hand to `gain(by=)`. Mean power 1.

        sig.gain(by=ws.recipes.fading(doppler_hz=100.0))                # Rayleigh
        sig.gain(by=ws.recipes.fading(doppler_hz=100.0, k_db=6.0))      # Rician, K = 6 dB

    It is three ops, each doing what it says:

        awgn(snr_db=0, bw_hz=2*doppler_hz, ref_power=P)    complex Gaussian scatter, with
                                                           power P in the Doppler band
        filter("doppler", f_hz=doppler_hz, shape=shape)    shaped to the Doppler spectrum;
                                                           the power in that band is kept
        spur(f_hz=los_hz, dbc=0, ref_power=1 - P)          Rician only: the line-of-sight
                                                           path, a steady tone at its own
                                                           Doppler shift `los_hz`

    `doppler_hz` is the largest Doppler shift, speed / wavelength. `k_db` is the Rician K
    factor, line-of-sight power over scattered power: P = 1 / (1 + K). Left out, there is no
    line of sight, P = 1 and |gain| is Rayleigh. The one random role is the scatter's
    (`.../awgn/0`), so `contrast()` re-rolls the fade and nothing else. The record has to hold
    at least 4 Doppler periods (`filter("doppler")` says why)."""
    if not doppler_hz > 0:
        raise ValueError(f"fading: doppler_hz is the largest Doppler shift (speed / "
                         f"wavelength) and must be positive, got {doppler_hz!r}")
    scatter = 1.0 if k_db is None else 1.0 / (1.0 + 10.0 ** (float(k_db) / 10.0))
    if k_db is None and los_hz:
        raise ValueError(f"fading: los_hz={los_hz!r} is the Doppler shift of the "
                         f"line-of-sight path, and without k_db= there is none. State the "
                         f"Rician K factor, k_db=.")
    s = (Signal().awgn(snr_db=0.0, bw_hz=2.0 * float(doppler_hz), ref_power=scatter)
         .filter("doppler", f_hz=float(doppler_hz), shape=shape))
    if k_db is not None:
        s = s.spur(f_hz=float(los_hz), dbc=0.0, ref_power=1.0 - scatter)
    return s


def multipath(tx, delays_s, gains, doppler_hz=None, shape="jakes"):
    """A MULTIPATH channel: the sum of delayed, scaled copies of one transmitted envelope.

        tx = ws.Signal().iq("qpsk", prbs=11)
        ws.recipes.multipath(tx, delays_s=[0.0, 2e-6], gains=[1.0, 0.5])

    Path k is `tx.delay(by=delays_s[k]).gain(by=gains[k])`, and the paths are joined by
    `combine`. On an I/Q record `delay` turns the carrier by -2*pi*fc*delay as well as
    shifting the envelope, so each path's PHASE is its delay's, as on air; a `gains` entry may
    be complex to state a phase on top (a reflection's sign). With `doppler_hz=` every path
    also gets its own `fading(doppler_hz, shape=shape)`, independent of the others, so the
    power delay profile is `abs(gains)**2` on average and each tap is Rayleigh.

    `tx` is the transmitter's chain. Written with the record's seed and grid, the result is
    the record; written as `Signal()`, a sub-chain for another op. It is rendered once per
    path, so it must be DETERMINISTIC -- a copy that drew its own noise would be a different
    signal, not a delayed one -- and a `tx` with random roles is refused. For a transmitter
    with randomness, name it in a `Channels` record and pass `ws.Signal().use(name)`: a named
    chain is the same samples wherever it is used.

    Returns the chain: path 0, then one `combine` of the others. Add the receiver after it."""
    delays_s, gains = list(delays_s), list(gains)
    if not delays_s or len(delays_s) != len(gains):
        raise ValueError(f"multipath: one delay and one gain per path; got {len(delays_s)} "
                         f"delays and {len(gains)} gains.")
    if not isinstance(tx, Signal) or not tx.ops:
        raise ValueError("multipath: tx is the transmitter's chain, a Signal that already "
                         "has its source: ws.Signal(seed=1, grid=g).iq('qpsk', prbs=11).")
    drawn = tx.roles()
    if drawn and tx.ops[0]["op"] != "use":
        raise ValueError(
            f"multipath: tx draws random roles {drawn}, and it is rendered once per path "
            f"under that path's own names, so each path would carry different noise rather "
            f"than a delayed copy. Put the transmitter in a Channels record as a named chain "
            f"and pass ws.Signal().use(<name>), which is the same samples wherever it is "
            f"used; or move the randomness after the multipath.")

    def path(d, g):
        s = tx.copy()
        if d:
            s = s.delay(by=float(d))
        if doppler_hz is not None:
            s = s.gain(by=fading(doppler_hz, shape=shape))
        if g != 1:
            s = s.gain(by=complex(g) if np.iscomplexobj(g) else float(g))
        return s

    paths = [path(d, g) for d, g in zip(delays_s, gains)]
    return paths[0].combine(paths[1:]) if len(paths) > 1 else paths[0]
