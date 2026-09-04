"""
wfmsynth.optical — optical-link primitives (400G/800G optical PAM4 and friends).

Optical intensity-modulated / direct-detection links have their own realism dimension that
electrical models don't capture:

  * extinction ratio (ER) — the modulator never fully extinguishes, so the low level carries
    nonzero power; ER = P_high / P_low.
  * relative intensity noise (RIN) — laser noise whose std is PROPORTIONAL to optical power.
  * shot noise — photon-counting (Poisson) noise whose variance is proportional to power
    (this is also the logged electrical shot-noise term, #30).
  * chromatic dispersion — frequency-dependent group delay that SPREADS a pulse over fibre.
  * multipath interference (MPI) — a delayed, attenuated copy (double reflection) beats in.

Signals here are optical INTENSITY (power ≥ 0). `to_optical` maps a bipolar electrical
waveform into intensity with a finite ER. numpy/scipy only.
"""
from __future__ import annotations

import numpy as np


def to_optical(x, er_db=10.0, p_avg=1.0):
    """Map a bipolar electrical waveform to optical INTENSITY (power ≥ 0) with a finite
    extinction ratio ``er_db`` = 10·log10(P_high/P_low). Output is scaled to mean power
    ``p_avg``. Real modulators never fully extinguish, so the low level carries power."""
    x = np.asarray(x, float)
    lo, hi = x.min(), x.max()
    d = (x - lo) / ((hi - lo) + 1e-12)          # 0..1
    r = 10 ** (-er_db / 10.0)                    # P_low / P_high
    power = d * (1.0 - r) + r                    # in [r, 1]
    return power * (p_avg / (power.mean() + 1e-12))


def rin_noise(power, rin_db_per_hz=-140.0, bw_hz=1e10, rng=None):
    """Relative intensity noise: additive noise with std PROPORTIONAL to instantaneous power.
    ``rin_db_per_hz`` integrated over ``bw_hz`` sets the relative noise level."""
    rng = rng or np.random.default_rng()
    power = np.asarray(power, float)
    sigma = np.sqrt(10 ** (rin_db_per_hz / 10.0) * bw_hz) * power
    return power + rng.standard_normal(len(power)) * sigma


def shot_noise(power, photons_per_unit=1e4, rng=None):
    """Photon-counting (Poisson) shot noise: variance PROPORTIONAL to optical power. Higher
    ``photons_per_unit`` (more photons per unit power) lowers the relative noise. Gaussian
    approximation of the Poisson count."""
    rng = rng or np.random.default_rng()
    p = np.clip(np.asarray(power, float), 0.0, None)
    mean = photons_per_unit * p
    return (mean + np.sqrt(mean + 1e-12) * rng.standard_normal(len(p))) / photons_per_unit


def chromatic_dispersion(x, strength=20.0):
    """Chromatic dispersion as a quadratic all-pass phase in frequency — frequency-dependent
    group delay that SPREADS a pulse. ``strength`` scales the accumulated dispersion (∝ D·L)."""
    x = np.asarray(x, float)
    n = len(x)
    f = np.fft.rfftfreq(n)
    H = np.exp(-1j * strength * (f / (f.max() + 1e-12)) ** 2 * 2 * np.pi)
    return np.fft.irfft(np.fft.rfft(x) * H, n)


def laser_chirp(power, alpha=2.0, grid=None):
    """Transient laser chirp: the instantaneous optical FREQUENCY excursion during intensity
    transitions, Δf = (alpha/4π)·d(ln P)/dt. It occurs at the edges (zero on a steady level)
    and scales with the linewidth-enhancement factor ``alpha``; combined with dispersion it
    causes transition-dependent pulse distortion. Returns the frequency deviation (Hz if a
    grid is given, else per-sample)."""
    p = np.clip(np.asarray(power, float), 1e-9, None)
    dt = grid.dt if grid is not None else 1.0
    return (alpha / (4 * np.pi)) * np.gradient(np.log(p)) / dt


def mpi(power, delay_samples, reflectivity=0.05):
    """Multipath interference: add a delayed, attenuated copy (a double-reflection ghost)."""
    p = np.asarray(power, float)
    d = int(delay_samples)
    ghost = np.zeros_like(p)
    if d < len(p):
        ghost[d:] = p[:len(p) - d]
    return p + reflectivity * ghost


# ---------------------------------------------------------------------------------------------------
# Complex-FIELD E/O/E — a true electrical -> optical -> electrical link. The stages above act on
# real INTENSITY; the ones below carry the complex optical field E(t) = sqrt(P)·exp(jφ), so phase
# effects (laser chirp, dispersion) interact correctly and square-law detection closes the loop.
# Chain order: (electrical drive) -> modulate_field -> fiber -> [edfa] -> photodetect -> tia -> ...
# ---------------------------------------------------------------------------------------------------

def modulate_field(drive, kind="mzm", vpi=1.0, bias=0.5, er_db=None, p_avg=1.0,
                   alpha=0.0, adiabatic=0.0, grid=None):
    """Electro-optic modulation: an electrical drive -> a COMPLEX optical field E(t)=sqrt(P)·exp(jφ).

    kind='mzm' (Mach-Zehnder): the true modulator transfer E = cos(π/2·(V/Vπ + bias)) — so the
      INTENSITY is cos² of the drive, not the linear ramp `to_optical` uses. `bias` sets the operating
      point (0.5 = quadrature). A nonzero `alpha` (chirp parameter) adds residual phase (a chirped MZM).
    kind='dml' (directly-modulated laser): intensity follows the drive and the laser CHIRPS —
      dφ/dt = (alpha/2)·(d(lnP)/dt + adiabatic·P) (transient + adiabatic). This is where the
      chirp×dispersion interaction that distorts an optical eye originates.

    `er_db` imposes a finite extinction ratio; the field is scaled to mean power `p_avg`. Returns a
    COMPLEX array — feed it to `fiber`/`edfa`/`photodetect`, not to real-valued electrical ops."""
    d = np.asarray(drive, float)
    dt = grid.dt if grid is not None else 1.0
    if kind == "mzm":
        span = np.max(np.abs(d - d.mean())) + 1e-12
        dn = (d - d.mean()) / span                          # drive normalized to ~[-1, 1]
        arg = (np.pi / 2.0) * (dn / vpi + bias)
        field = np.cos(arg).astype(complex)
        if er_db is not None:
            r = 10 ** (-er_db / 10.0); P = np.abs(field) ** 2
            P = P * (1.0 - r) + r * (P.max() + 1e-12)
            field = np.sqrt(P) * np.exp(1j * np.angle(field + 1e-12))
        if alpha:
            field = field * np.exp(1j * alpha * arg)
    elif kind == "dml":
        P = (d - d.min()) / ((d.max() - d.min()) + 1e-12)   # intensity 0..1 follows the drive
        r = 10 ** (-er_db / 10.0) if er_db is not None else 1e-3
        P = P * (1.0 - r) + r
        phi = np.cumsum((alpha / 2.0) * (np.gradient(np.log(P + 1e-12)) / dt + adiabatic * P)) * dt
        field = np.sqrt(P) * np.exp(1j * phi)
    else:
        raise ValueError("modulate_field kind must be 'mzm' or 'dml'")
    return field * np.sqrt(p_avg / (np.mean(np.abs(field) ** 2) + 1e-12))


def fiber(field, length_km=1.0, D_ps_nm_km=17.0, wavelength_nm=1550.0, atten_db_km=0.2, grid=None):
    """Single-mode fibre on the complex FIELD: chromatic dispersion (PHYSICAL β2·L) + attenuation.

    β2 = -D·λ²/(2πc) from the dispersion parameter `D_ps_nm_km` (ps/(nm·km)) and wavelength; the
    all-pass phase exp(j·β2·L·ω²/2) is applied to the field spectrum, so a chirped field develops
    transition-dependent intensity distortion once square-law-detected. `atten_db_km` is fibre loss
    (0.2 dB/km at 1550 nm). Needs `grid` for real frequency units. Returns the complex field."""
    E = np.asarray(field, complex); n = len(E)
    fs = grid.fs if grid is not None else 1.0
    c = 299792458.0
    D = D_ps_nm_km * 1e-6                                    # ps/(nm·km) -> s/m²
    lam = wavelength_nm * 1e-9
    beta2 = -D * lam ** 2 / (2 * np.pi * c)                  # s²/m  (~ -2.17e-26 at 1550 nm)
    L = length_km * 1e3
    w = 2 * np.pi * np.fft.fftfreq(n, 1.0 / fs)
    E = np.fft.ifft(np.fft.fft(E) * np.exp(1j * beta2 * L * w ** 2 / 2.0))
    return E * 10 ** (-atten_db_km * length_km / 20.0)       # amplitude attenuation


def edfa(field, gain_db=15.0, nf_db=5.0, p_ase_scale=1e-3, rng=None):
    """Optical amplifier (EDFA): amplify the field by √gain and add amplified-spontaneous-emission
    (ASE) noise — complex Gaussian whose power grows with (gain-1)·noise-figure. Enables amplified
    and long-haul links (and the signal-ASE beat noise that shows up after detection)."""
    E = np.asarray(field, complex); g = 10 ** (gain_db / 10.0)
    rng = rng or np.random.default_rng()
    ase = (g - 1.0) * 10 ** (nf_db / 10.0) * p_ase_scale * (np.mean(np.abs(E) ** 2) + 1e-12)
    noise = np.sqrt(ase / 2.0) * (rng.standard_normal(len(E)) + 1j * rng.standard_normal(len(E)))
    return E * np.sqrt(g) + noise


def photodetect(field, responsivity=1.0, shot=True, photons_per_unit=1e4, rng=None):
    """Square-law photodetector (O->E) — the stage that closes the electrical->optical->electrical
    loop. Photocurrent i = responsivity·|E|² (real). With `shot`, adds photon shot noise. Feed the
    output to `tia` and then the usual electrical chain (`digitize`, …)."""
    i = responsivity * np.abs(np.asarray(field, complex)) ** 2
    return shot_noise(i, photons_per_unit, rng=rng) if shot else i


def tia(current, gain=1.0, bw_hz=None, thermal_rms=0.0, grid=None, rng=None):
    """Transimpedance amplifier: the optical-RX front end after the photodiode. Converts photocurrent
    to voltage (`gain`), band-limits to `bw_hz`, and adds input-referred `thermal_rms` noise."""
    v = np.asarray(current, float) * gain
    if bw_hz and grid is not None:
        from .instrument import scope_bandwidth
        v = scope_bandwidth(v, grid, bw_hz)
    if thermal_rms:
        rng = rng or np.random.default_rng()
        v = v + rng.standard_normal(len(v)) * thermal_rms
    return v
