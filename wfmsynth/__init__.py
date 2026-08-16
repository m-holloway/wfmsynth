"""
wfmsynth — physics-informed synthetic waveform generation.

Generate realistic voltage-vs-time signals grounded in real signal-integrity physics:
frequency-dependent (causal) channels, transmission-line reflections, crosstalk,
decomposed jitter, AC-coupling, plus digital/RF carriers and full "deep-memory" scope
captures. Everything is validated (`wfmsynth.validate`) and depends only on numpy/scipy.

Modules / public API:
  physics       low-level primitives — lossy_channel, multi_reflection, crosstalk,
                ac_couple, inject_jitter, nrz, pam4, am, fm, psk, qam, chirp, ...
  impairments   apply_impairment(name, x, rng), domain_randomize(x, rng), IMPAIRMENTS
  grammar       carrier(), envelope(), sample(), generate() — compositional signals
  pam4          deep_capture() — realistic segmented PAM4 scope captures with defects
  validate      run as `python -m wfmsynth.validate` — hard physics-property assertions
"""
__version__ = "0.27.0"

from . import (physics, impairments, grammar, pam4, grid, instrument, streams, compose,
               measure, sweep, cdr, sparam, stream, simreal, rx, scene)
from .physics import (N, T, Jitter, tx_ffe, carrier_symbols, resonant_reflection,
                      nominal_nonlinearity, crosstalk_matrix, differential_pair,
                      differential_mode, common_mode, supply_coupling)
from .grid import Grid
from .streams import Streams
from .impairments import (IMPAIRMENTS, apply_impairment, domain_randomize,
                          mix_at_constant_power, burst_gate, apply_gated, realistic_noise)
from .grammar import sample, generate, CARRIERS, ENVELOPES
from .pam4 import deep_capture, PATHOLOGIES
from .instrument import (interleave_adc, shaped_noise_floor, clip_adc, quantize_adc,
                         digitize as digitize_adc)
from .compose import Signal, dataset
from .measure import (eye_height, best_phase, attributes, align_symbols, ground_truth,
                      pattern_period)
from .sweep import hold_constant, realized_table, solve_monotonic
from .cdr import (recover_clock, jitter_transfer, tracked_out_fraction, ssc_phase,
                  apply_ssc, timing_source, apply_phase)
from .sparam import read_touchstone, write_touchstone, sparam_channel, touchstone_channel
from .stream import stream_convolve, stream_blocks, channel_fir
from .simreal import separability, feature_vector
from .rx import ctle, dfe
from .scene import Scene

__all__ = [
    "physics", "impairments", "grammar", "pam4", "grid", "instrument", "streams", "compose",
    "measure", "sweep", "cdr", "sparam", "stream", "simreal", "rx", "scene", "ctle", "dfe", "Scene",
    "stream_convolve", "stream_blocks", "channel_fir", "separability", "feature_vector",
    "recover_clock", "jitter_transfer", "tracked_out_fraction", "ssc_phase", "apply_ssc",
    "timing_source", "apply_phase",
    "read_touchstone", "write_touchstone", "sparam_channel", "touchstone_channel",
    "N", "T", "Grid", "Jitter", "Streams", "tx_ffe", "carrier_symbols", "resonant_reflection",
    "nominal_nonlinearity", "crosstalk_matrix",
    "differential_pair", "differential_mode", "common_mode", "supply_coupling",
    "IMPAIRMENTS", "apply_impairment", "domain_randomize",
    "mix_at_constant_power", "burst_gate", "apply_gated", "realistic_noise",
    "sample", "generate", "CARRIERS", "ENVELOPES", "deep_capture", "PATHOLOGIES",
    "interleave_adc", "shaped_noise_floor", "clip_adc", "quantize_adc", "digitize_adc",
    "Signal", "dataset",
    "eye_height", "best_phase", "attributes", "align_symbols", "ground_truth", "pattern_period",
    "hold_constant", "realized_table", "solve_monotonic", "__version__",
]
