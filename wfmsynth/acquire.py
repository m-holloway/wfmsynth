"""
wfmsynth.acquire — generalized two-rate synthesis: simulation grid -> acquisition -> record.

Physics is simulated on a fine **simulation grid**; a vendor-neutral **AcquisitionProfile** then
models the analog front end + digitizer and samples the result onto an **acquisition grid** (the
sample rate / record length a real acquisition system would store), with optional **record
decimation** to a shallower storage/display depth. This is carrier-agnostic — the same acquisition
story for NRZ, PAM4, clocks, buses, optical, analog — not a PAM4-special path.

Pipeline order (fixed):

    source -> channel/impairments        @ simulation grid   (the Signal chain up to .acquire)
           -> analog input stage         (input-bandwidth limit)
           -> sample clock               (sampled at the acquisition rate, with clock jitter)
           -> digitizer                  (noise floor -> interleave -> clip -> quantize)
           -> record decimation          (optional: sample | peak_hold | average, to a depth)

Tap points expose the record at each stage (`simulated`, `conditioned`, `digitized`, `stored`),
and provenance records the realized simulation vs acquisition rates.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Optional

import numpy as np

from . import instrument as INST
from .grid import Grid


@dataclass(frozen=True)
class AcquisitionProfile:
    """A single, immutable acquisition config (vendor-neutral). Supplies the acquisition-stage
    defaults so callers don't hand-wire the front-end / digitizer / decimation each time."""
    sample_rate_hz: float
    record_length: int
    input_bandwidth_hz: Optional[float] = None          # analog front-end -3 dB
    sample_clock_jitter_rms_s: float = 0.0              # timebase jitter
    enob: Optional[float] = None                        # finite-ENOB quantization
    clip_full_scale: Optional[float] = None             # ADC saturation level
    interleave: Optional[dict] = None                   # dict(m_cores, offset_v, ...)
    noise_floor: Optional[dict] = None                  # dict(rms, shape)
    decimation: Optional[dict] = None                   # dict(mode, depth)

    @property
    def grid(self) -> Grid:
        return Grid(fs=self.sample_rate_hz, n=int(self.record_length))


def record_decimation(x, mode="sample", depth=None):
    """Compress a record to ``depth`` samples for storage/display.

      sample     — keep every k-th sample (naive; aliases fast structure)
      peak_hold  — per interval keep (min, max) -> a 2-channel record; narrow transients and the
                   noise envelope survive time compression (a scope's peak-detect acquisition)
      average    — mean per interval
    """
    x = np.asarray(x, float)
    n = len(x)
    if depth is None or depth >= n:
        return np.stack([x, x]) if mode == "peak_hold" else x
    if mode == "sample":
        return x[:: n // depth][:depth]
    b = n // depth
    xr = x[:b * depth].reshape(depth, b)
    if mode == "peak_hold":
        return np.stack([xr.min(1), xr.max(1)])         # (2, depth)
    if mode == "average":
        return xr.mean(1)
    raise ValueError(f"unknown decimation mode {mode!r} (use 'sample', 'peak_hold' or 'average')")


def acquire_record(x_sim, grid_sim, profile, rng=None):
    """Run the acquisition chain on a simulated waveform and return a dict of taps:
    ``{'simulated', 'conditioned', 'digitized', 'stored', 'info'}``. ``grid_sim`` is the fine
    simulation grid; ``profile`` an AcquisitionProfile."""
    rng = rng or np.random.default_rng()
    sim = np.asarray(x_sim, float)

    # analog front end (at the simulation rate)
    conditioned = INST.scope_bandwidth(sim, grid_sim, profile.input_bandwidth_hz) \
        if profile.input_bandwidth_hz else sim

    # sample onto the acquisition grid at the (optionally jittered) sample-clock times
    t_acq = np.arange(int(profile.record_length)) / profile.sample_rate_hz
    if profile.sample_clock_jitter_rms_s:
        t_acq = t_acq + rng.normal(0.0, profile.sample_clock_jitter_rms_s, len(t_acq))
    t_sim = np.arange(len(conditioned)) / grid_sim.fs
    y = np.interp(t_acq, t_sim, conditioned, left=conditioned[0], right=conditioned[-1])

    g_acq = profile.grid
    if profile.noise_floor:
        y = y + INST.shaped_noise_floor(len(y), rng=rng, **profile.noise_floor)
    if profile.interleave:
        y = INST.interleave_adc(y, rng=rng, **profile.interleave)
    if profile.clip_full_scale is not None:
        y, _ = INST.clip_adc(y, profile.clip_full_scale)
    if profile.enob is not None:
        y = INST.quantize_adc(y, enob=profile.enob, full_scale=profile.clip_full_scale)
    digitized = y

    stored = record_decimation(y, **profile.decimation) if profile.decimation else y
    info = {"sim_fs": grid_sim.fs, "acq_fs": profile.sample_rate_hz,
            "record_length": int(profile.record_length),
            "oversample": grid_sim.fs / profile.sample_rate_hz,
            "decimation": dict(profile.decimation) if profile.decimation else None}
    return {"simulated": sim, "conditioned": conditioned, "digitized": digitized,
            "stored": stored, "info": info}
