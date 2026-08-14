"""
wfmsynth.physics — physics-informed waveform synthesis primitives.

Frequency-domain lossy channels (skin + dielectric, causal minimum-phase option),
transmission-line multi-reflection, crosstalk (NEXT/FEXT), AC-coupling, physically-
decomposed jitter (Rj/Pj/DCD), digital signaling (NRZ/PAM4 from PRBS), RF modulation
(AM/FM/PSK/QAM), chirp, PDN transients, and an ECG-like generator.

Every primitive is paired with an assertion in `wfmsynth.validate` that checks the
physical property actually holds — nothing is trusted without that check. Works on a
normalized, unitless time grid (`N` samples over [0,1)); rate-parameterizable to any
symbol/sample rate downstream. numpy/scipy only.
"""
from __future__ import annotations
import numpy as np
from scipy import signal

N = 4096                       # default working grid
T = np.linspace(0.0, 1.0, N, endpoint=False)

# Every primitive below infers its grid length from the array it is given, and the
# generators take an optional `n`. N/T remain the DEFAULTS, so any call that worked
# against the fixed 4096-point grid behaves bit-identically. This is what makes the
# "rate-parameterizable" claim actually true: a deep-memory capture is tens of
# millions of points, and nothing here should care how long the record is.


# ---------------------------------------------------------------- channel physics
def _min_phase_H(Hmag, n=None):
    """Causal minimum-phase complex response from a real magnitude |H| (rfft bins),
    via the cepstral / Hilbert relation so loss and phase are physically LINKED
    (Kramers-Kronig; the Djordjevic-Sarkar-style causal channel). A real magnitude-
    only response is zero-phase -> non-causal (symmetric pre/post-ringing); the
    minimum-phase version concentrates the response AFTER t=0 (asymmetric post-cursor
    ISI), which is what a real dispersive interconnect does.

    `n` is the full (two-sided) transform length; inferred from Hmag when omitted."""
    if n is None:
        n = 2 * (len(Hmag) - 1)
    mag_full = np.concatenate([Hmag, Hmag[-2:0:-1]])      # symmetric, length n
    logmag = np.log(mag_full + 1e-12)
    c = np.fft.ifft(logmag).real                          # real cepstrum
    w = np.zeros(n); w[0] = 1.0; w[1:n // 2] = 2.0; w[n // 2] = 1.0   # causal folding
    return np.exp(np.fft.fft(c * w))                      # complex min-phase, length n


def lossy_channel(x, length_in=6.0, tand=0.02, eps_r=4.3, f_nyq_ghz=8.0,
                  skin_k=0.0, causal=False, grid=None, loss_db=None, loss_at_ghz=None):
    """Apply a frequency-dependent SI channel: insertion loss
        IL(f)[dB] = (a_skin*sqrt(f_GHz) + b_diel*f_GHz) * length_in
    with dielectric-loss coefficient b_diel = 2.3*sqrt(eps_r)*tand (dB/in/GHz)
    [KB: All-About-Circuits/Bogatin]. f axis is scaled so the Nyquist bin maps to
    f_nyq_ghz, whatever the record length.
    causal=False: zero-phase magnitude response (legacy). causal=True: physically
    correct minimum-phase response -> asymmetric post-cursor ISI (real dispersion).

    Absolute units (wfmsynth.grid.Grid): pass grid=Grid(...) to take the real
    frequency axis (f_nyq_ghz) from the grid's Nyquist. Pass loss_db + loss_at_ghz to
    request a channel with a stated insertion loss (dB) at a stated frequency (GHz):
    the skin+dielectric SHAPE is kept and scaled so IL(loss_at_ghz) == loss_db exactly."""
    x = np.asarray(x, float)
    n = len(x)
    if grid is not None:
        f_nyq_ghz = grid.f_nyquist / 1e9                  # real frequency axis from the grid
    f_ghz = np.fft.rfftfreq(n) * 2.0 * f_nyq_ghz          # 0..f_nyq_ghz at Nyquist
    b_diel = 2.3 * np.sqrt(eps_r) * tand
    a_skin = skin_k if skin_k > 0 else 0.35               # ~dB/in/sqrt(GHz) typ
    il_db = (a_skin * np.sqrt(f_ghz) + b_diel * f_ghz) * length_in
    if loss_db is not None and loss_at_ghz is not None:
        # keep the skin+dielectric SHAPE; scale so IL(loss_at_ghz) == loss_db exactly
        shape_at = (a_skin * np.sqrt(loss_at_ghz) + b_diel * loss_at_ghz) * length_in
        il_db = il_db * (loss_db / (shape_at + 1e-12))
    Hmag = 10.0 ** (-il_db / 20.0)
    if causal:
        Hc = _min_phase_H(Hmag, n)                        # full-spectrum complex H
        return np.fft.ifft(np.fft.fft(x) * Hc).real
    return np.fft.irfft(np.fft.rfft(x) * Hmag, n=n)


def crosstalk(x, aggressor, coupling=0.12, kind="fext", td_frac=0.05):
    """Add coupled noise from a neighboring (aggressor) line. FEXT couples through
    the mutual C/L as the DERIVATIVE of the aggressor (∝ d/dt); NEXT is a delayed,
    broadband coupling. `coupling` scales the aggressor relative to x's span. Real
    high-speed links are frequently crosstalk-limited — a major missing effect."""
    x = np.asarray(x, float)
    n = len(x)
    a = np.asarray(aggressor, float)
    if kind == "fext":
        k = np.gradient(a)
    else:                                                 # next: delayed coupling
        d = int(td_frac * n); k = np.zeros(n); k[d:] = a[:n - d]
    k = k / (np.abs(k).max() + 1e-9)
    return x + coupling * (np.ptp(x) + 1e-9) * k


def ac_couple(x, fc_frac=0.004, fc_hz=None, grid=None):
    """AC-coupling (series cap) as a 1st-order high-pass -> baseline wander/droop
    that grows with run length. fc_frac is the corner as a fraction of Nyquist.
    Ubiquitous on real serial links. Absolute units: pass fc_hz + grid=Grid(...) to
    state the corner in Hz (converted to a fraction of the grid's Nyquist)."""
    if fc_hz is not None:
        if grid is None:
            raise ValueError("fc_hz requires grid=Grid(...)")
        fc_frac = grid.hz_to_frac_nyquist(fc_hz)
    fc = float(np.clip(fc_frac, 1e-4, 0.5))
    sos = signal.butter(1, fc, btype="high", output="sos")
    return signal.sosfiltfilt(sos, np.asarray(x, float))


def multi_reflection(x, td_frac=0.12, gamma_s=0.3, gamma_l=0.4, n_bounce=6,
                     td_samples=None, td_ps=None, grid=None):
    """Transmission-line bounce diagram: received = incident + reflected train.
    Each round trip is delayed by 2*td and scaled by (gamma_s*gamma_l)^n. This is
    the lattice/bounce superposition for a mismatched line.

    `td_frac` is a fraction of the record, which is convenient on a normalized grid
    but scales with record length. `td_samples` overrides it with an absolute delay
    in samples. `td_ps` (with grid=Grid(...)) states the one-way delay in picoseconds
    -- a via stub is a fixed number of picoseconds away regardless of how long you
    acquire for -- and is converted to samples via the grid's sample rate."""
    x = np.asarray(x, float)
    nx = len(x)
    if td_ps is not None:
        if grid is None:
            raise ValueError("td_ps requires grid=Grid(...)")
        td_samples = round(td_ps * 1e-12 * grid.fs)
    d = int(td_samples) if td_samples is not None else int(td_frac * nx)
    y = x.copy()
    g = gamma_s * gamma_l
    for k in range(1, n_bounce + 1):
        shift = 2 * d * k
        if shift >= nx:
            break
        refl = np.zeros_like(x)
        refl[shift:] = x[:nx - shift]
        y = y + (g ** k) * gamma_l * refl
    return y


# ---------------------------------------------------------------- jitter physics
def inject_jitter(x, sigma_rj=0.0, a_pj=0.0, f_pj=5.0, dcd=0.0, rng=None,
                  sigma_rj_s=None, a_pj_s=None, f_pj_hz=None, dcd_s=None, grid=None):
    """Physically decomposed jitter via time-axis warp then resample.
      Rj: BANDLIMITED Gaussian phase noise (smooth, so edges shift COHERENTLY),
          renormalized to RMS = sigma_rj samples. (Per-sample white noise would
          scramble the waveform, not jitter its edges — validated.)
      Pj: sinusoidal A*sin(2*pi*f*t) (samples)
      DCD: polarity-dependent offset (rising +dcd/2, falling -dcd/2)
    sigma_rj/a_pj/dcd are in SAMPLES of timing displacement; f_pj in cycles-per-record.

    Absolute units (grid=Grid(...)): sigma_rj_s/a_pj_s/dcd_s state the displacement in
    SECONDS (converted to samples via fs), and f_pj_hz states the periodic-jitter tone
    in Hz (converted to cycles-per-record). Real jitter is quoted in fs/ps and Hz."""
    rng = rng or np.random.default_rng()
    x = np.asarray(x, float)
    n = len(x)
    if grid is not None:
        if sigma_rj_s is not None:
            sigma_rj = grid.to_samples(sigma_rj_s)
        if a_pj_s is not None:
            a_pj = grid.to_samples(a_pj_s)
        if dcd_s is not None:
            dcd = grid.to_samples(dcd_s)
        if f_pj_hz is not None:
            f_pj = grid.hz_to_cycles_per_record(f_pj_hz)
    t = np.linspace(0.0, 1.0, n, endpoint=False)
    disp = np.zeros(n)
    if sigma_rj > 0:
        w = rng.standard_normal(n)
        sos = signal.bessel(2, 0.03, output="sos")          # smooth to ~phase-noise BW
        w = signal.sosfiltfilt(sos, w)
        w = w / (w.std() + 1e-9) * sigma_rj                 # exact RMS = sigma_rj
        disp += w
    if a_pj > 0:
        disp += a_pj * np.sin(2 * np.pi * f_pj * t)
    if dcd != 0:
        slope = np.sign(np.gradient(x))
        disp += (dcd / 2.0) * slope
    src = np.clip(np.arange(n) + disp, 0, n - 1)
    return np.interp(src, np.arange(n), x)


# ---------------------------------------------------------------- signaling
# Standard PRBS generator polynomials, as (descending) tap exponents.
# Order 13 is the IEEE 802.3 Clause 120.5.11.2.1 polynomial used by PRBS13Q:
#     G(x) = 1 + x + x^2 + x^12 + x^13
# THE POLYNOMIAL IS NOT A FREE PARAMETER. Several other maximal-length degree-13
# polynomials exist (13,12,11,8 among them) and produce a perfectly valid
# pseudo-random sequence with the right level statistics -- that a protocol
# analyser will never pattern-lock to. If a capture is meant to be analysable by
# an instrument, it has to be the standard's polynomial.
PRBS_TAPS = {7: (7, 6), 9: (9, 5), 13: (13, 12, 2, 1), 15: (15, 14), 31: (31, 28)}

# Gray mapping from a bit pair to a PAM4 level, per IEEE 802.3 120.5.11.2.1.
GRAY_PAM4 = {(0, 0): -1.0, (0, 1): -1.0 / 3.0, (1, 1): 1.0 / 3.0, (1, 0): 1.0}


def prbs(order, length, seed=1):
    """Fibonacci LFSR PRBS of the given order. See PRBS_TAPS for polynomials."""
    taps = PRBS_TAPS[order]
    st = seed & ((1 << order) - 1) or 1
    out = np.empty(length, np.int8)
    for i in range(length):
        b = 0
        for t in taps:
            b ^= (st >> (t - 1)) & 1
        out[i] = st & 1
        st = ((st << 1) | b) & ((1 << order) - 1)
    return out


def prbs13q(n_symbols, seed=1):
    """PRBS13Q symbol sequence per IEEE 802.3 Clause 120.5.11.2.1.

    An 8191-symbol repeating sequence formed by Gray coding CONSECUTIVE BIT PAIRS
    taken from TWO repetitions of PRBS13. Two repetitions are required because 8191
    is odd, so bit pairs do not align to the PRBS period -- taking pairs from a
    single repetition silently produces a different (non-conformant) sequence.

    Returns PAM4 levels in {-1, -1/3, +1/3, +1}, tiled to n_symbols.

    Conformance (asserted in wfmsynth.validate against published IEEE figures):
    transition density 0.7501 and level probabilities 0.2499/0.2500/0.2500/0.2500.
    """
    period = 8191
    bits = prbs(13, period * 2, seed)                 # 16382 bits -> 8191 symbols
    pairs = bits.reshape(-1, 2)
    base = np.array([GRAY_PAM4[(int(a), int(b))] for a, b in pairs], dtype=float)
    reps = int(np.ceil(n_symbols / period))
    return np.tile(base, reps)[:n_symbols]


def _shape_edges(x, tr_samples, causal=False):
    """Band-limit a piecewise-constant symbol stream into finite-rise-time edges.

    causal=False keeps the original zero-phase (sosfiltfilt) shaping, which is
    symmetric and therefore adds pre-cursor as well as post-cursor content.
    causal=True uses a forward-only filter, so the edge shaping cannot move energy
    backwards in time. The library's headline principle is causality, and
    zero-phase edge shaping quietly violates it; the flag defaults to the legacy
    behaviour so existing output is unchanged.

    The forward filter is initialised to STEADY STATE at x[0] (sosfilt_zi). With
    zero initial conditions the output starts at 0 regardless of the signal, which
    plants a full-scale settling transient at the head of every record -- caught by
    the validation check for pre-edge disturbance.
    """
    sos = signal.bessel(4, min(0.7 / tr_samples, 0.98), output="sos")
    if not causal:
        return signal.sosfiltfilt(sos, x)
    zi = signal.sosfilt_zi(sos) * x[0]
    y, _ = signal.sosfilt(sos, x, zi=zi)
    return y


def nrz(n_ui=32, tr_frac=0.15, seed=1, n=None, causal=False):
    n = N if n is None else int(n)
    spb = n / n_ui
    bits = prbs(7, n_ui, seed)
    idx = np.clip((np.arange(n) / spb).astype(int), 0, n_ui - 1)
    x = np.where(bits[idx] > 0, 1.0, -1.0)
    return _shape_edges(x, max(tr_frac * spb, 2), causal)


def pam4(n_ui=32, tr_frac=0.15, seed=1, n=None, causal=False, pattern="legacy"):
    """PAM4 carrier.

    pattern="legacy"  the original PRBS7/PRBS9 Gray-ish map. Not a standard
                      sequence -- fine for shape coverage, will NOT pattern-lock.
    pattern="prbs13q" IEEE 802.3 PRBS13Q (8191 symbols). Use this when the capture
                      has to be analysable by an instrument.
    """
    n = N if n is None else int(n)
    spb = n / n_ui
    levels = np.array([-1.0, -1 / 3, 1 / 3, 1.0])
    if pattern == "prbs13q":
        syms = prbs13q(n_ui, seed)
    elif pattern == "legacy":
        b0 = prbs(7, n_ui, seed); b1 = prbs(9, n_ui, seed + 3)
        gray = (b0 * 2 + (b0 ^ b1))                  # simple Gray-ish map -> 0..3
        syms = levels[np.clip(gray, 0, 3)]
    else:
        raise ValueError(f"unknown pattern {pattern!r}; use 'legacy' or 'prbs13q'")
    idx = np.clip((np.arange(n) / spb).astype(int), 0, n_ui - 1)
    return _shape_edges(syms[idx], max(tr_frac * spb, 2), causal)


# ---------------------------------------------------------------- RF / analog
def am(fc=40.0, fm=3.0, depth=0.6, n=None):
    t = T if n is None else np.linspace(0.0, 1.0, int(n), endpoint=False)
    msg = np.sin(2 * np.pi * fm * t)
    return (1 + depth * msg) * np.cos(2 * np.pi * fc * t)


def fm(fc=40.0, fm_rate=3.0, beta=5.0, n=None):
    t = T if n is None else np.linspace(0.0, 1.0, int(n), endpoint=False)
    msg = np.sin(2 * np.pi * fm_rate * t)
    phase = 2 * np.pi * fc * t + beta * np.cumsum(msg) / len(t) * (2 * np.pi)
    return np.cos(phase)


def psk(fc=40.0, n_sym=16, m=4, seed=1, n=None):
    n = N if n is None else int(n)
    t = np.linspace(0.0, 1.0, n, endpoint=False)
    rng = np.random.default_rng(seed)
    syms = rng.integers(0, m, n_sym)
    spb = n / n_sym
    idx = np.clip((np.arange(n) / spb).astype(int), 0, n_sym - 1)
    ph = syms[idx] * (2 * np.pi / m)
    return np.cos(2 * np.pi * fc * t + ph)


def qam(fc=40.0, n_sym=16, seed=1, n=None):
    n = N if n is None else int(n)
    t = np.linspace(0.0, 1.0, n, endpoint=False)
    rng = np.random.default_rng(seed)
    I = rng.choice([-1, -1 / 3, 1 / 3, 1], n_sym); Q = rng.choice([-1, -1 / 3, 1 / 3, 1], n_sym)
    spb = n / n_sym
    idx = np.clip((np.arange(n) / spb).astype(int), 0, n_sym - 1)
    return I[idx] * np.cos(2 * np.pi * fc * t) - Q[idx] * np.sin(2 * np.pi * fc * t)


def chirp(f0=3.0, f1=40.0, n=None):
    t = T if n is None else np.linspace(0.0, 1.0, int(n), endpoint=False)
    return signal.chirp(t, f0=f0, f1=f1, t1=1.0, method="linear")


def pdn_transient(droop=0.08, tau1=0.03, tau2=0.25, t0=0.3, n=None):
    t = T if n is None else np.linspace(0.0, 1.0, int(n), endpoint=False)
    tt = t - t0
    sag = np.where(tt > 0, droop * (0.5 * np.exp(-tt / tau1) + 0.5 * np.exp(-tt / tau2)), 0.0)
    return 1.0 - sag


def ecg_like(hr=8.0, n=None):
    """Synthetic ECG-ish: Gaussian-bump QRS + P/T waves per beat."""
    n = N if n is None else int(n)
    T = np.linspace(0.0, 1.0, n, endpoint=False)
    x = np.zeros(n)
    for k in range(int(hr)):
        c = (k + 0.5) / hr
        x += 1.0 * np.exp(-((T - c) ** 2) / (2 * 0.004 ** 2))          # R
        x -= 0.15 * np.exp(-((T - c + 0.012) ** 2) / (2 * 0.004 ** 2))  # Q
        x -= 0.15 * np.exp(-((T - c - 0.012) ** 2) / (2 * 0.004 ** 2))  # S
        x += 0.2 * np.exp(-((T - c - 0.05) ** 2) / (2 * 0.012 ** 2))    # T
        x += 0.1 * np.exp(-((T - c + 0.06) ** 2) / (2 * 0.010 ** 2))    # P
    return x


# family registry: name -> (generator thunk taking rng) ; broad + physical
def family_bank():
    return {
        "nrz_si": lambda r: lossy_channel(nrz(seed=r.integers(1, 127)),
                                          length_in=r.uniform(2, 14), tand=r.uniform(0.005, 0.025)),
        "pam4": lambda r: lossy_channel(pam4(seed=r.integers(1, 127)), length_in=r.uniform(2, 10)),
        "nrz_reflect": lambda r: multi_reflection(nrz(seed=r.integers(1, 127)),
                                                  td_frac=r.uniform(0.06, 0.2),
                                                  gamma_s=r.uniform(0.1, 0.4), gamma_l=r.uniform(0.1, 0.5)),
        "am": lambda r: am(fc=r.uniform(25, 55), fm=r.uniform(2, 5), depth=r.uniform(0.3, 0.9)),
        "fm": lambda r: fm(fc=r.uniform(25, 55), fm_rate=r.uniform(2, 5), beta=r.uniform(2, 8)),
        "psk": lambda r: psk(fc=r.uniform(25, 55), m=int(r.choice([2, 4, 8])), seed=r.integers(1, 999)),
        "qam": lambda r: qam(fc=r.uniform(25, 55), seed=r.integers(1, 999)),
        "chirp": lambda r: chirp(f0=r.uniform(1, 5), f1=r.uniform(20, 50)),
        "pdn": lambda r: pdn_transient(droop=r.uniform(0.03, 0.15), t0=r.uniform(0.2, 0.5)),
        "ecg": lambda r: ecg_like(hr=r.uniform(5, 12)),
    }
