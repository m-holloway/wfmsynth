"""wfmsynth.swept -- a swept laser, the frame its field is held in, and a resonator solved there.

THE FRAME. A wavelength sweep moves the optical carrier by terahertz while the record is
sampled at kilohertz to megahertz, so the field cannot be an envelope about a FIXED carrier:
the laser would leave the grid in the first microsecond. A `laser` record is therefore held in
the frame that moves WITH the laser,

    E(t) = A(t) * exp(j*2*pi * integral of nu(t) dt)          the samples are A(t)

and the carrier nu(t) is a stated function of time: the ground-truth map from time to
wavelength (`wavelength_map`). A laser alone is A = sqrt(power), a constant.

WHAT A DEVICE DOES IN THAT FRAME. A path of delay tau gives

    A_out(t) = A(t - tau) * exp(-j*2*pi * integral of nu from t - tau to t)

so against the direct path it beats at (sweep rate) x tau: the fringes of an interferometer,
of a stray reflection, of a wavelength reference. That is the `delay` op on a field whose
carrier is stated -- it shifts the envelope and turns it by exactly that integral
(`Carrier.turns`) -- and `combine` adds the paths, so those devices are chains
(`wfmsynth.recipes`: `mzi`, `stray_reflection`). A resonator is FEEDBACK, not a finite sum of
copies, and is the one mechanism here (`resonator`).

THE LASER. Stated in wavelength, as a tunable laser is:

    lambda(t) = start_nm + speed * u + bow_nm * 4*(u/T)*(1 - u/T)        u = t clipped to 0..T
                + ripple_nm * sin(2*pi*ripple_hz*t) + (every hop at or before t)

`T` is the sweep time: the record, or (stop_nm - start_nm)/speed_nm_per_s when both are
stated, after which the laser parks at `stop_nm`. Before the record it sits at `start_nm`.
`bow_nm` and `ripple_nm` are the sweep's nonlinearity: a slow bow that peaks at mid-sweep and
a periodic wobble. `hops` are MODE HOPS, `[{"t_s": ..., "by_nm": ...}]`: the wavelength steps
and stays. A hop is a parameter of the source and not a second source, because it is ONE
laser whose field stays continuous while nu(t) does not; each is also an event
(`kind="mode_hop"`). Everything is data, so nu(t) is a pure function of the recipe, and what
varies from unit to unit is said with `ws.tol` on these numbers. nu = c / lambda.

NOT MODELLED: linewidth (its phase noise lies far outside a kHz-MHz grid; it would matter for
a delay comparable to the coherence time), power variation across the sweep (`gain(by=)`
after the laser), and the power dip at a hop.

THE RESONATOR, by coupled-mode theory. One mode of amplitude a(t), loss rates kappa_i
(intrinsic) and kappa_e (to the bus), driven by the field on the bus:

    da/dt = (j*w0 - kappa/2) a + sqrt(kappa_e) E_in         E_out = E_in - sqrt(kappa_e) a

In the swept frame, with y = a / sqrt(kappa_e) and D(t) = 2*pi*(nu(t) - nu0(t)):

    dy/dt = -(kappa/2 + j*D(t)) y + A_in(t)                 A_out = A_in - kappa_e * y

Standing still it is the Lorentzian 1 - kappa_e/(kappa/2 + j*D): a dip of full width
kappa/(2*pi) Hz reaching ((kappa_i - kappa_e)/(kappa_i + kappa_e))**2 on resonance. Swept, it
is whatever the equation says: a fast sweep leaves light in the ring that beats with the
laser as it moves on (ring-down), and the dip is shallower, later and wider. The equation is
integrated EXACTLY for a detuning constant across each sub-step and an input linear across
each sample; the sub-step is chosen so the detuning's change inside it turns the phase by
under `PHASE_TOL` radians, and a sample that holds a mode hop is cut at the hop's instant, so
the detuning steps where the laser does and not at the next sample. `fsr_hz` repeats the resonance every free spectral range, one
independent mode each. EVERY order of the ring is counted: those the sweep comes within a
free spectral range of are solved, and the rest, which sit settled far from the laser, are
summed in closed form (the sum of 1/(z - m) over all m is pi*cot(pi*z)). Standing still that
is the ring formula (r - a*exp(j*phi))/(1 - r*a*exp(j*phi)) to first order in the coupling
per round trip, pi*coupling_hz/fsr_hz, and to every order in the loss; the difference,
about that number squared, is measured in tests/test_swept.py (2.5e-4 at a finesse of 67,
1.6e-2 at 12.5). `split_hz` couples the two directions of travel (backscatter): the two
supermodes sit `split_hz` apart, each taking half the coupling. `by` moves the resonance in
nm while the sweep runs -- a number, or a chain (a ring heating up). Only the through port
is modelled: no drop port, no reflected wave.

REFUSED BY NAME: a sweep that crosses the linewidth in under `MIN_SAMPLES` samples (the
ring-down's beat would still be above 1e-3 of its start when it passes Nyquist, and alias);
a mode hop that leaves the laser half the sample rate or more from a resonance still holding
over `RING_TOL` of the field (the same alias, from a step instead of a sweep); a delay whose beat leaves the grid; a mode hop inside a delay of a sample or more; a
`combine` of fields about two different carriers (`same_carrier`); and `optical_mpi` on a
laser's field, which copies the envelope without the carrier's turn (`wfmsynth.domain`).

PRECISION. nu * tau is formed in the lag, never as a difference of two times, and is good to
a double's last digits of nu * tau turns: 1e-12 of a turn for picoseconds, 1e-7 for a delay
of microseconds.
"""
from __future__ import annotations

import numpy as np

C_M_S = 299792458.0
LASER_KEYS = ("start_nm", "stop_nm", "speed_nm_per_s", "bow_nm", "ripple_nm", "ripple_hz",
              "hops", "power")
RESONATOR_KEYS = ("wavelength_nm", "intrinsic_hz", "coupling_hz", "fsr_hz", "split_hz")

PHASE_TOL = 1e-5          # rad: phase error allowed from the detuning moving inside a sub-step
MEMORY = 60.0             # the ring's memory is cut where exp(-kappa*t/2) has fallen to e**-30
MIN_SAMPLES = 4.4         # samples across a linewidth: 4*ln(1000)/(2*pi)
RING_TOL = 1e-3           # ringing past Nyquist that a record may carry, as a share of the field
_FACT = np.cumprod(np.concatenate(([1.0], np.arange(1.0, 14.0))))      # _FACT[k] = k!
_GL = ((-np.sqrt(0.6), 5.0 / 9.0), (0.0, 8.0 / 9.0), (np.sqrt(0.6), 5.0 / 9.0))


def hz(wavelength_nm):
    """Optical frequency of a vacuum wavelength: c / lambda."""
    return C_M_S * 1e9 / np.asarray(wavelength_nm, float)


def _number(op, key, default=None, positive=False):
    v = op.get(key, default)
    if v is None:
        return None
    if isinstance(v, (bool, str)) or not np.isfinite(float(v)) or (positive and not v > 0):
        raise ValueError(f"laser: {key}={v!r} must be a finite "
                         f"{'positive ' if positive else ''}number.")
    return float(v)


class Carrier:
    """nu(t) of a `laser` op on a record of `n` samples at `fs`: the frame its field is in."""

    def __init__(self, op, fs, n):
        self.fs, self.n = float(fs), int(n)
        dur = self.n / self.fs
        if op.get("start_nm") is None:
            raise ValueError("laser needs start_nm=: the wavelength at the record's start. "
                             "Add stop_nm= (reached at the record's end) or speed_nm_per_s= "
                             "for a sweep; alone it is a laser standing still.")
        self.start = _number(op, "start_nm", positive=True)
        stop, speed = _number(op, "stop_nm", positive=True), _number(op, "speed_nm_per_s")
        self.t_sweep = dur
        if speed is None:
            speed = 0.0 if stop is None else (stop - self.start) / dur
        elif stop is not None:
            if speed == 0.0 or (stop - self.start) / speed <= 0:
                raise ValueError(
                    f"laser: speed_nm_per_s={speed:g} does not lead from start_nm="
                    f"{self.start:g} to stop_nm={stop:g}. State stop_nm alone (the sweep then "
                    f"takes the record), or a speed of the sweep's own sign.")
            self.t_sweep = (stop - self.start) / speed
            if self.t_sweep > dur * (1.0 + 1e-9):
                raise ValueError(
                    f"laser: {self.start:g} to {stop:g} nm at {speed:g} nm/s takes "
                    f"{self.t_sweep:.6g} s and the record is {dur:.6g} s, so stop_nm would "
                    f"not be reached. Drop stop_nm (the record ends mid-sweep) or "
                    f"speed_nm_per_s (the sweep takes the record), or lengthen the record.")
        self.speed = speed
        self.bow = _number(op, "bow_nm", 0.0)
        self.ripple = _number(op, "ripple_nm", 0.0)
        self.ripple_hz = _number(op, "ripple_hz", positive=True)
        if (self.ripple != 0.0) != (self.ripple_hz is not None):
            raise ValueError("laser: a ripple is ripple_nm= (its peak) AND ripple_hz= (how "
                             "often); one was given without the other.")
        self.power = _number(op, "power", 1.0, positive=True)
        self.hops = []
        for h in op.get("hops") or ():
            if not isinstance(h, dict) or set(h) != {"t_s", "by_nm"}:
                raise ValueError(f"laser: hops is a list of {{'t_s': <seconds>, 'by_nm': "
                                 f"<step>}}, one per mode hop; got {h!r}.")
            t, by = float(h["t_s"]), float(h["by_nm"])
            if not (0.0 < t < dur and np.isfinite(by) and by != 0.0):
                raise ValueError(f"laser: a hop at t_s={t!r} by {by!r} nm. Its time lies "
                                 f"inside the record (0 to {dur:.6g} s) and its step is not "
                                 f"zero.")
            self.hops.append((t, by))
        self.hops.sort()
        if np.min(self.wavelength_nm(np.arange(self.n) / self.fs)) <= 0:
            raise ValueError("laser: the sweep reaches a wavelength of zero or less.")

    def wavelength_nm(self, t, hops=True):
        """The laser's wavelength at times `t` (seconds from the record's start)."""
        t = np.asarray(t, float)
        u = np.clip(t, 0.0, self.t_sweep)
        lam = self.start + self.speed * u
        if self.bow:
            lam = lam + self.bow * 4.0 * (u / self.t_sweep) * (1.0 - u / self.t_sweep)
        if self.ripple:
            lam = lam + self.ripple * np.sin(2.0 * np.pi * self.ripple_hz * t)
        if hops:
            for at, by in self.hops:
                lam = lam + by * (t >= at)
        return lam

    def hz(self, t, hops=True):
        """nu(t), the optical frequency at times `t`."""
        return hz(self.wavelength_nm(t, hops))

    def turns(self, t, tau):
        """The integral of nu from ``t - tau`` to ``t``, in cycles: how far a path of delay
        `tau` lags the direct one. Three-point Gauss-Legendre on each stretch between mode
        hops, so a hop inside the delay is split at its instant; exact while nu is a quintic
        across `tau`, which a sweep is to better than 1e-9 of a turn."""
        t = np.asarray(t, float)
        # in the LAG u = t - s, from tau down to 0: a picosecond is far below the spacing of
        # doubles near t, so the stretch is never formed as a difference of two times
        edges = [tau] + [np.clip(t - at, 0.0, tau) for at, _ in self.hops] + [0.0]
        total = 0.0
        for a, b in zip(edges[:-1], edges[1:]):
            half, mid = 0.5 * (a - b), 0.5 * (a + b)
            total = total + half * sum(w * self.hz(t - (mid + half * x)) for x, w in _GL)
        return total

    def rate_hz_per_s(self):
        """The largest |d nu / dt| of the sweep itself over the record, hops left out."""
        if self.n < 2:
            return 0.0
        nu = self.hz(np.arange(self.n) / self.fs, hops=False)
        return float(np.abs(np.diff(nu)).max() * self.fs)


def laser(op, grid, idx, sink=None):
    """The `laser` op: a constant field of `power` in the frame of its own carrier, and one
    `mode_hop` event per hop."""
    if grid is None:
        raise ValueError("laser: a sweep is stated in seconds, which needs the record's rate "
                         "and length -- Signal(grid=Grid(fs=..., n=...)).")
    car = Carrier(op, grid.fs, grid.n)
    if sink is not None:
        from .events import Event
        for at, by in car.hops:
            sink.append(Event(
                sample=min(int(np.ceil(at * car.fs)), car.n - 1), kind="mode_hop",
                severity=0.0, t_s=at,
                width=1, direction=float(np.sign(by)), placement=f"laser/{idx}",
                params={"by_nm": by, "wavelength_nm": float(car.wavelength_nm(at))}))
    return np.full(car.n, np.sqrt(car.power), complex)


def turn(y, d, fs, carrier):
    """The carrier's half of a field `delay`: `y`, the envelope already shifted by `d`
    seconds (a number or an array), turned by the phase the carrier ran through meanwhile."""
    car = Carrier(carrier, fs, len(y))
    t = np.arange(len(y)) / fs
    beat = float(np.abs(car.hz(t, hops=False) - car.hz(t - d, hops=False)).max())
    if beat >= 0.5 * fs:
        raise ValueError(
            f"delay: against the undelayed field this path beats at up to {beat:.4g} Hz "
            f"(sweep rate x delay), past this grid's Nyquist frequency {0.5 * fs:g} Hz. "
            f"Shorten the delay, slow the sweep, or raise Grid(fs=).")
    if car.hops and float(np.max(d)) * fs >= 1.0:
        raise ValueError(
            f"delay: the laser has a mode hop and this delay is {float(np.max(d)) * fs:.3g} "
            f"samples. For that long the two paths carry different modes and beat at the "
            f"hop's size, far past the grid. A delay under one sample (the hop is then one "
            f"instant), or a laser without hops.")
    return y * np.exp(-2j * np.pi * np.mod(car.turns(t, d), 1.0))


def check_resonator(op):
    """The resonator's numbers, refused by name where they mean nothing:
    ``(wavelength_nm, kappa_i, kappa_e, fsr_hz, split_hz)``, the rates in rad/s."""
    for k in ("wavelength_nm", "intrinsic_hz", "coupling_hz"):
        if op.get(k) is None:
            raise ValueError(
                f"resonator needs {k}=. It is stated by its resonance wavelength_nm and its "
                f"two loss rates as linewidths in Hz: intrinsic_hz (absorption, scattering) "
                f"and coupling_hz (to the bus). The dip is intrinsic_hz + coupling_hz wide. "
                f"From a Q and an extinction: ws.recipes.ring(...).")
    lam, ki, ke = (float(op[k]) for k in ("wavelength_nm", "intrinsic_hz", "coupling_hz"))
    fsr, split = op.get("fsr_hz"), op.get("split_hz")
    if not (lam > 0 and ki >= 0 and ke >= 0 and ki + ke > 0 and np.isfinite(lam + ki + ke)):
        raise ValueError(f"resonator: wavelength_nm={lam!r} is positive, and intrinsic_hz="
                         f"{ki!r} and coupling_hz={ke!r} are linewidths: not negative, and "
                         f"not both zero (a ring with no loss never settles).")
    if fsr is not None and not float(fsr) > 4.0 * (ki + ke):
        raise ValueError(
            f"resonator: fsr_hz={fsr!r} against a linewidth of {ki + ke:g} Hz. The "
            f"resonances are solved one mode each, which holds while they do not overlap: "
            f"the free spectral range must be over 4 linewidths (a finesse above 4).")
    if split is not None and not float(split) > 0:
        raise ValueError(f"resonator: split_hz={split!r} is the spacing of the two "
                         f"backscatter-coupled modes, a positive frequency.")
    return (lam, 2 * np.pi * ki, 2 * np.pi * ke, None if fsr is None else float(fsr),
            None if split is None else float(split))


def _modes(car, nu0, fsr, split):
    """``[(offset_hz, weight), ...]``: every mode's resonance relative to `nu0`, and the
    share of the coupling it takes. With a free spectral range, every order inside the sweep
    and one beyond each end."""
    orders = [0]
    if fsr is not None:
        nu = car.hz(np.arange(car.n) / car.fs)
        orders = range(int(np.floor((nu.min() - nu0) / fsr)) - 1,
                       int(np.ceil((nu.max() - nu0) / fsr)) + 2)
    halves = [(0.0, 1.0)] if split is None else [(-0.5 * split, 0.5), (0.5 * split, 0.5)]
    return [(m * (fsr or 0.0) + off, w) for m in orders for off, w in halves]


def _settled(d, kappa, fsr, split):
    """The sum over EVERY mode of ``weight / (kappa/2 + j*2*pi*(d - offset))`` at a detuning
    `d` (Hz) from the stated resonance: what the ring holds per unit of input once settled.
    With a free spectral range the sum over all orders is ``cot(pi*z) / (2j*fsr)``."""
    halves = [(0.0, 1.0)] if split is None else [(-0.5 * split, 0.5), (0.5 * split, 0.5)]
    if fsr is None:
        return sum(w / (0.5 * kappa + 2j * np.pi * (d - off)) for off, w in halves)
    return sum(w / (2j * fsr * np.tan(np.pi * ((d - off) / fsr - 1j * kappa / (4 * np.pi * fsr))))
               for off, w in halves)


def _recur(g, q, y0):
    """``y[k+1] = exp(-g[k])*y[k] + q[k]`` from ``y[0] = y0``, in stretches short enough for
    the inverse of the running decay to stay finite."""
    n, out = len(q), np.empty(len(q) + 1, complex)
    out[0] = y0
    step = max(int(600.0 / max(float(g.real.max()), 1e-300)), 1) if n else 1
    for a in range(0, n, step):
        b = min(a + step, n)
        G = np.cumsum(g[a:b])
        out[a + 1:b + 1] = np.exp(-G) * (out[a] + np.cumsum(q[a:b] * np.exp(G)))
    return out


def _phi(z):
    """``(phi1, phi2)`` of the exact step of ``y' = -z*y + a`` for an input `a` linear across
    it: ``phi1 = (1 - exp(-z))/z`` and ``phi2 = (z - 1 + exp(-z))/z**2``, the second by its
    series where the closed form cancels."""
    p1 = -np.expm1(-z) / z
    p2 = (z + np.expm1(-z)) / (z * z)
    small = np.abs(z) < 0.1
    if small.any():
        zs, acc = z[small], 0.0
        for k in range(11, 1, -1):                   # sum of (-z)**j / (j + 2)!, by Horner
            acc = 1.0 / _FACT[k] - zs * acc
        p2[small] = acc
    return p1, p2


def _hop_on_grid(at, fs, kappa, ringing, det, x):
    """Refuse a mode hop that leaves light in the ring beating with the laser past Nyquist.
    `ringing` is each sample's distance from the settled response (as a share of the output),
    `det` the detuning in Hz: after the hop the first decays at kappa/2 and turns at the
    second."""
    k = int(np.ceil(at * fs))                        # the first sample at or after the hop
    if k >= len(x):
        return
    beat, left = abs(float(det[k])), abs(ringing[k]) / max(float(np.abs(x).max()), 1e-300)
    if beat >= 0.5 * fs and left > RING_TOL:
        raise ValueError(
            f"resonator: the mode hop at t_s={at:g} leaves the laser {beat:.4g} Hz from a "
            f"resonance that still holds light ({left:.2g} of the field at the next sample, "
            f"decaying over {2.0 * fs / kappa:.3g} samples). That light beats with the "
            f"laser at the detuning, past this grid's Nyquist frequency {0.5 * fs:g} Hz, "
            f"and the record would alias it. Raise Grid(fs=) above {2.0 * beat:.4g} Sa/s, "
            f"or hop from further off the resonance.")


def resonator(x, fs, carrier, op, shift_nm=0.0):
    """The `resonator` op on the field `x` of a laser whose `carrier` op is given:
    ``A_out = A_in - kappa_e * sum over modes of y``, each `y` solved as the module docstring
    says. `shift_nm` moves the resonance, a number or one value per sample."""
    lam0, ki, ke, fsr, split = check_resonator(op)
    x = np.asarray(x, complex)
    n = len(x)
    car = Carrier(carrier, fs, n)
    kappa, dt = ki + ke, 1.0 / fs
    shift = np.broadcast_to(np.asarray(shift_nm, float), (n,))
    if not np.all(np.isfinite(shift)) or np.min(lam0 + shift) <= 0:
        raise ValueError("resonator: `by` moves the resonance in nm and must stay finite, "
                         "with the resonance above zero.")
    nu0 = hz(lam0 + shift)
    rate = car.rate_hz_per_s() + (float(np.abs(np.diff(nu0)).max()) * fs if n > 1 else 0.0)
    width = kappa / (2 * np.pi)
    if rate * dt * MIN_SAMPLES > width:
        raise ValueError(
            f"resonator: the sweep moves {rate * dt:.4g} Hz per sample against a linewidth "
            f"of {width:.4g} Hz, so it crosses the resonance in "
            f"{width / (rate * dt):.2g} samples. Below {MIN_SAMPLES} the light left in the "
            f"ring still beats with the laser when that beat passes Nyquist, and the record "
            f"would alias it. Raise Grid(fs=) to at least {MIN_SAMPLES * rate / width:.4g} "
            f"Sa/s, or slow the sweep.")
    # each sample: the last `span` seconds before it (all of it, or as far back as the ring
    # remembers), in `m` sub-steps short enough for the moving detuning
    span = min(dt, MEMORY / kappa)
    m = max(int(np.ceil(span * np.sqrt(2 * np.pi * rate / (8.0 * PHASE_TOL)))), 1)
    edges = (np.arange(m + 1) - m) * (span / m)      # sub-step edges, back from the sample
    block = max((1 << 18) // m, 1)
    # a mode hop between two samples: that sample's sub-steps are cut at the hop's instant
    hopped = {}
    for at, _ in car.hops:
        k = int(np.ceil(at * fs)) - 1                # the hop lies in (k, k + 1] samples
        if 0 <= k < n - 1 and -span < at - (k + 1) * dt < 0:
            hopped.setdefault(k, []).append(at - (k + 1) * dt)
    det = car.hz(np.arange(n) * dt) - nu0            # at each sample, Hz
    out = x.copy()
    solved = np.zeros(n, complex)                    # the same modes, had they settled

    def advance(a, b, e, off):
        """Samples a..b-1 to the next: (what the input adds, the decay), over edges `e`."""
        a_in = x[a:b, None] + (x[a + 1:b + 1] - x[a:b])[:, None] * (1.0 + e / dt)
        h, mid = np.diff(e), 0.5 * (e[1:] + e[:-1])
        res = nu0[a:b, None] + (nu0[a + 1:b + 1] - nu0[a:b])[:, None] * (1.0 + mid / dt) + off
        at = np.arange(a + 1, b + 1)[:, None] * dt + mid
        zh = (0.5 * kappa + 2j * np.pi * (car.hz(at) - res)) * h
        after = np.cumsum(zh[:, ::-1], axis=1)[:, ::-1] - zh     # from a sub-step's end on
        p1, p2 = _phi(zh)
        add = (a_in[:, :-1] * (p1 - p2) + a_in[:, 1:] * p2) * h
        return (add * np.exp(-after)).sum(axis=1), zh.sum(axis=1)

    for off, w in _modes(car, hz(lam0), fsr, split):
        q, g = np.empty(n - 1, complex), np.empty(n - 1, complex)
        for a in range(0, n - 1, block):
            b = min(a + block, n - 1)
            q[a:b], g[a:b] = advance(a, b, edges, off)
        for k, cuts in hopped.items():
            q[k:k + 1], g[k:k + 1] = advance(k, k + 1, np.union1d(edges, cuts), off)
        settled = 1.0 / (0.5 * kappa + 2j * np.pi * (det - off))
        y0 = x[0] * settled[0]                           # as the laser left it before t = 0
        # a ring that forgets within one sample carries nothing over from the one before
        y = np.concatenate(([y0], q)) if span < dt else _recur(g, q, y0)
        for at, _ in car.hops:
            _hop_on_grid(at, fs, kappa, w * ke * (y - x * settled), det - off, x)
        out -= w * ke * y
        solved += w * settled
    if fsr is not None:        # every other order: at least a free spectral range away
        out -= ke * x * (_settled(det, kappa, fsr, split) - solved)
    return out


def steady(wavelength_nm, op):
    """The transmission a `resonator` op has STANDING STILL at `wavelength_nm`: the complex
    field ratio ``1 - kappa_e * sum over modes of 1/(kappa/2 + j*D)``, every order of the
    ring included. Its squared magnitude is what a slow sweep measures: the stated response
    a record is checked against."""
    lam0, ki, ke, fsr, split = check_resonator(op)
    return 1.0 - ke * _settled(hz(wavelength_nm) - hz(lam0), ki + ke, fsr, split)


def _said(c):
    return ("a laser's: " + ", ".join(f"{k}={c[k]!r}" for k in LASER_KEYS if k in c)
            if isinstance(c, dict) else "none stated")


def same_carrier(others, carrier, grid, record=None):
    """Refuse a `combine` of fields that are not about ONE carrier. Samples of two fields add
    coherently only in the same frame: two lasers, or a laser and a field that states no
    carrier, beat at their difference -- terahertz -- which no sample here holds."""
    from . import domain as DOMAIN
    expand = None if record is None else record.rec.expand
    mine = {k: carrier[k] for k in LASER_KEYS if k in carrier} if isinstance(carrier, dict) \
        else None
    for k, spec in enumerate(others):
        ops = spec.get("ops") if isinstance(spec, dict) else None
        if not isinstance(ops, (list, tuple)) or not ops:
            continue                                   # a pinned record: its own business
        own = DOMAIN.trace(ops, DOMAIN.grid_for(ops, grid), expand=expand)[-1][1]
        if own == DOMAIN.CARRIER_ELSEWHERE:
            continue
        if (None if own is None else {k: own[k] for k in LASER_KEYS if k in own}) != mine:
            raise ValueError(
                f"combine: others[{k}] is a field about another carrier ({_said(own)}) than "
                f"the field it joins ({_said(carrier)}). Fields add coherently only about "
                f"the SAME carrier; two sources beat at their difference, which is far past "
                f"this grid. Derive both paths from one laser (the same chain, or "
                f"Signal().use(<its name>) in a Channels record).")


def carrier_of(sig):
    """The `laser` op whose carrier the chain `sig` ends its field on, or None: for a Signal,
    or a channel of a `Channels` record handed out by ``rec.signal(name)``."""
    from . import domain as DOMAIN
    rec = sig.within[0] if getattr(sig, "within", None) else None
    doms = DOMAIN.trace(sig.ops, sig.grid, expand=None if rec is None else rec.expand)
    held = [fc for dom, fc in doms if dom == DOMAIN.OPTICAL and isinstance(fc, dict)]
    return held[-1] if held else None


def wavelength_map(sig):
    """THE GROUND-TRUTH MAP of a swept record: ``{"t_s", "wavelength_nm", "f_hz"}``, one
    value per sample of the laser's grid, read from the recipe (nothing is rendered). `sig`
    is a Signal with a `laser` in it, or ``rec.signal(name)`` of a `Channels` record."""
    op = carrier_of(sig)
    if op is None:
        raise ValueError("wavelength_map: this chain has no `laser`, so there is no map from time to "
                         "wavelength. (A field from `eo` states no carrier.)")
    car = Carrier(op, sig.grid.fs, sig.grid.n)
    t = np.arange(car.n) / car.fs
    lam = car.wavelength_nm(t)
    return {"t_s": t, "wavelength_nm": lam, "f_hz": hz(lam)}
