"""
wfmsynth.grammar — a COMPOSITIONAL GRAMMAR of waveforms.

The manifold-coverage thesis: instead of enumerating named protocols, define a small
set of physics-grounded PRIMITIVES and compose them:

    waveform  =  Envelope( Carrier [+ Carrier] )  ->  Channel  ->  Impairments  ->  Capture

Sampling this grammar generates a combinatorially vast space of signals — most
corresponding to NO named protocol but densely covering the "shape manifold." A real
signal a user brings is then almost always INTERIOR to the training distribution.

The carriers deliberately SPAN the morphologies of the named families (gauss_bumps ≈
ECG; damped_osc ≈ PDN/ringing; qam ≈ QAM) WITHOUT being them — so a model trained only
on the grammar can be tested for zero-shot generalization to the named protocols.

numpy/scipy only. Impairments reuse `wfmsynth.impairments`.
"""
from __future__ import annotations
import numpy as np
from scipy import signal as _sig

from . import physics as P

N = P.N
T = P.T

# ---------------------------------------------------------------- carriers (general primitives)
def _prbs_levels(rng, n_ui, levels):
    spb = N / n_ui
    idx = np.clip((np.arange(N) / spb).astype(int), 0, n_ui - 1)
    seq = rng.choice(levels, size=n_ui)
    x = seq[idx].astype(float)
    tr = max(rng.uniform(0.05, 0.3) * spb, 2)
    return _sig.sosfiltfilt(_sig.bessel(4, min(0.7 / tr, 0.98), output="sos"), x)


CARRIERS = [
    "dc", "sine", "multitone", "square", "pam", "chirp", "am", "fm", "psk", "qam",
    "pulse_train", "gauss_bumps", "sawtooth", "damped_osc", "exp_settle", "noise_band",
]


def carrier(rng, kind=None):
    kind = kind or CARRIERS[rng.integers(len(CARRIERS))]
    if kind == "dc":
        return np.zeros(N) + rng.uniform(-0.5, 0.5) + rng.uniform(-0.2, 0.2) * T, kind
    if kind == "sine":
        return np.sin(2 * np.pi * rng.uniform(2, 40) * T + rng.uniform(0, 6.28)), kind
    if kind == "multitone":
        y = sum(rng.uniform(0.3, 1) * np.sin(2 * np.pi * rng.uniform(2, 45) * T + rng.uniform(0, 6.28))
                for _ in range(rng.integers(2, 4)))
        return y, kind
    if kind == "square":
        return _prbs_levels(rng, rng.integers(8, 48), np.array([-1.0, 1.0])), kind
    if kind == "pam":
        m = rng.choice([3, 4, 5]); lv = np.linspace(-1, 1, m)
        return _prbs_levels(rng, rng.integers(8, 40), lv), kind
    if kind == "chirp":
        return _sig.chirp(T, f0=rng.uniform(1, 5), f1=rng.uniform(20, 55), t1=1.0, method="linear"), kind
    if kind == "am":
        fc = rng.uniform(25, 55)
        return (1 + rng.uniform(0.3, 0.9) * np.sin(2 * np.pi * rng.uniform(2, 5) * T)) * np.cos(2 * np.pi * fc * T), kind
    if kind == "fm":
        msg = np.sin(2 * np.pi * rng.uniform(2, 5) * T)
        return np.cos(2 * np.pi * rng.uniform(25, 55) * T + rng.uniform(2, 8) * np.cumsum(msg) / N * 2 * np.pi), kind
    if kind == "psk":
        n = rng.integers(8, 24); spb = N / n
        idx = np.clip((np.arange(N) / spb).astype(int), 0, n - 1)
        ph = rng.integers(0, rng.choice([2, 4, 8]), n)[idx] * (2 * np.pi / 4)
        return np.cos(2 * np.pi * rng.uniform(25, 55) * T + ph), kind
    if kind == "qam":
        n = rng.integers(8, 24); spb = N / n
        idx = np.clip((np.arange(N) / spb).astype(int), 0, n - 1)
        I = rng.choice([-1, -1 / 3, 1 / 3, 1], n)[idx]; Q = rng.choice([-1, -1 / 3, 1 / 3, 1], n)[idx]
        fc = rng.uniform(25, 55)
        return I * np.cos(2 * np.pi * fc * T) - Q * np.sin(2 * np.pi * fc * T), kind
    if kind == "pulse_train":
        f = rng.uniform(4, 25); duty = rng.uniform(0.1, 0.6)
        return _sig.sosfiltfilt(_sig.bessel(4, 0.2, output="sos"),
                                (_sig.square(2 * np.pi * f * T, duty=duty) + 1) / 2), kind
    if kind == "gauss_bumps":        # spans ECG / spectroscopy morphology
        x = np.zeros(N); k = rng.integers(3, 12)
        for _ in range(k):
            c = rng.uniform(0, 1); w = rng.uniform(0.003, 0.03); a = rng.uniform(-1, 1)
            x += a * np.exp(-((T - c) ** 2) / (2 * w ** 2))
        return x, kind
    if kind == "sawtooth":
        return _sig.sawtooth(2 * np.pi * rng.uniform(3, 30) * T, width=rng.uniform(0, 1)), kind
    if kind == "damped_osc":         # spans PDN / ringing morphology
        t0 = rng.uniform(0.1, 0.4); f = rng.uniform(5, 40); tau = rng.uniform(0.03, 0.2)
        env = np.where(T > t0, np.exp(-(T - t0) / tau), 0.0)
        return env * np.sin(2 * np.pi * f * (T - t0)), kind
    if kind == "exp_settle":         # RC steps
        x = np.zeros(N); n = rng.integers(2, 6); pts = np.sort(rng.uniform(0, 1, n))
        cur = rng.uniform(-1, 1)
        for p in pts:
            tgt = rng.uniform(-1, 1); tau = rng.uniform(0.01, 0.08)
            m = T >= p; x[m] = tgt + (cur - tgt) * np.exp(-(T[m] - p) / tau); cur = tgt
        return x, kind
    if kind == "noise_band":         # bandlimited noise (textures)
        w = rng.standard_normal(N)
        lo, hi = sorted(rng.uniform(0.02, 0.5, 2))
        return _sig.sosfiltfilt(_sig.butter(4, [max(lo, 1e-3), hi], btype="band", output="sos"), w), kind
    return np.zeros(N), kind


# ---------------------------------------------------------------- envelopes
ENVELOPES = ["const", "gate", "burst", "ramp", "expdecay", "amslow"]


def envelope(rng, kind=None):
    kind = kind or ENVELOPES[rng.integers(len(ENVELOPES))]
    if kind == "const":
        return np.ones(N)
    if kind == "gate":
        a, b = sorted(rng.uniform(0.1, 0.9, 2))
        e = np.zeros(N); e[(T >= a) & (T <= b)] = 1.0
        return _sig.sosfiltfilt(_sig.bessel(4, 0.1, output="sos"), e).clip(0, 1)
    if kind == "burst":
        f = rng.uniform(2, 6)
        return (0.5 * (1 + _sig.square(2 * np.pi * f * T, duty=rng.uniform(0.3, 0.7)))).clip(0, 1)
    if kind == "ramp":
        return np.linspace(rng.uniform(0, 0.4), rng.uniform(0.6, 1.0), N)
    if kind == "expdecay":
        return np.exp(-T / rng.uniform(0.2, 0.8))
    if kind == "amslow":
        return 0.6 + 0.4 * np.sin(2 * np.pi * rng.uniform(1, 3) * T)
    return np.ones(N)


# ---------------------------------------------------------------- compose one grammar waveform
def sample(rng, apply_impairments=None):
    """Compose one grammar waveform. Returns (waveform on the P.N grid, carrier_kind,
    [impairment names applied]). `apply_impairments` optionally restricts the impairment
    pool (defaults to the full `impairments.IMPAIRMENTS` vocabulary)."""
    from . import impairments as IMP               # reuse impairment library + domain randomization
    x, ck = carrier(rng)
    # optional second, lighter carrier (mixtures broaden the manifold)
    if rng.uniform() < 0.35:
        x2, _ = carrier(rng)
        x = x + rng.uniform(0.15, 0.5) * x2
    x = envelope(rng) * x
    # optional physical channel
    if rng.uniform() < 0.4:
        x = P.lossy_channel(x, length_in=rng.uniform(2, 12), tand=rng.uniform(0.005, 0.025))
    if rng.uniform() < 0.25:
        x = P.multi_reflection(x, td_frac=rng.uniform(0.06, 0.2),
                               gamma_s=rng.uniform(0.15, 0.4), gamma_l=rng.uniform(0.15, 0.45))
    # impairments + capture-condition domain randomization
    pool = apply_impairments if apply_impairments is not None else IMP.IMPAIRMENTS
    k = int(rng.choice([0, 1, 1, 2]))
    imps = list(rng.choice(pool, size=min(k, len(pool)), replace=False)) if k else []
    for im in imps:
        x = IMP.apply_impairment(im, x, rng)
    x = IMP.domain_randomize(x, rng)
    return x, ck, imps


def generate(n, seed=0):
    """Generate `n` grammar waveforms. Returns a dict of raw arrays (no normalization):
        X       (n, P.N) float32 waveforms
        carrier (n,)     carrier-kind string per waveform
        imps    (n,)     list of impairment names applied per waveform (object array)
    Pure synthesis output — normalize / resample / label as your downstream needs."""
    rng = np.random.default_rng(seed)
    X = np.empty((n, N), np.float32)
    carriers = np.empty(n, dtype=object)
    imps = np.empty(n, dtype=object)
    for i in range(n):
        x, ck, im = sample(rng)
        X[i] = x.astype(np.float32); carriers[i] = ck; imps[i] = im
    return dict(X=X, carrier=carriers, imps=imps)
