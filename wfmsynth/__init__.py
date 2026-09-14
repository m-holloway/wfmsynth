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
  events        place_events / apply_events — localized needles + per-window labels
  eye           eye_density() — a record folded on the clock a CDR recovers from it
  grammar       carrier(), envelope(), sample(), generate() — compositional signals
  quinary       4D-PAM5 / 8B1Q4 — the four-pair quinary line code, its non-uniform symbol
                statistics, the transmit partial-response filter and the standard's
                transmitter test patterns
  pam4          deep_capture() — realistic segmented PAM4 scope captures with defects
  patterns      a NAME -> symbol-generator registry: register/resolve/describe/replay, a
                general arbitrary-polynomial LFSR and a block-repeat generator. Mechanisms live
                here; WHICH pattern a standard names is the caller's to register.
  validate      run as `python -m wfmsynth.validate` — hard physics-property assertions
"""
__version__ = "0.39.1"

from . import (physics, impairments, events, grammar, pam4, grid, instrument, streams, compose,
               measure, sweep, cdr, eye, sparam, stream, simreal, rx, scene, optical, coding, bus,
               acquire, patterns, quinary, hdf5)
from .physics import (N, T, Jitter, tx_ffe, carrier_symbols, from_symbols, resonant_reflection, de_emphasis_taps,
                      nominal_nonlinearity, crosstalk_matrix, crosstalk_sum, hybrid_echo,
                      single_pair_observation, db_to_coupling, differential_pair,
                      differential_mode, common_mode, supply_coupling)
from .grid import Grid
from .streams import Streams
from .impairments import (IMPAIRMENTS, apply_impairment, domain_randomize,
                          mix_at_constant_power, burst_gate, apply_gated, realistic_noise, drift,
                          electrical_idle)
from .events import (Event, EventList, place_events, apply_events, defect_symbols,
                     label_windows, windows_from_centers, nominal_ui_windows,
                     measure_window, ui_heights, eye_mask, slope_reversals,
                     step_overshoot_fraction, second_order_step, damped_sinusoid,
                     MECHANISMS, PLACEMENTS)
from .grammar import sample, generate, CARRIERS, ENVELOPES
from .pam4 import deep_capture, PATHOLOGIES
from .instrument import (interleave_adc, shaped_noise_floor, clip_adc, quantize_adc,
                         digitize as digitize_adc, scope_bandwidth, probe_loading, timebase_jitter,
                         store_record, quantisation_floor_db_per_hz,
                         sinad_noise_rms, converter_noise_rms)
from .compose import Signal, dataset, rederive_anchor, canonicalize, op_kind
from .measure import (eye_height, best_phase, attributes, align_symbols, ground_truth,
                      pattern_period)
from .sweep import hold_constant, realized_table, solve_monotonic
from .cdr import (recover_clock, jitter_transfer, tracked_out_fraction, ssc_phase,
                  apply_ssc, timing_source, apply_phase, phase_noise, recover_and_fold)
from .eye import eye_density, eye_recover, eye_crossings
from .sparam import read_touchstone, write_touchstone, sparam_channel, touchstone_channel
from .stream import stream_convolve, stream_blocks, channel_fir
from .simreal import separability, feature_vector
from .rx import ctle, dfe, ffe
from .scene import Scene
from .optical import (to_optical, rin_noise, shot_noise, chromatic_dispersion, mpi, laser_chirp,
                      modulate_field, fiber, field_mpi, edfa, photodetect, tia)
from .coding import (dc_balanced, scramble_64b66b, running_disparity, max_run,
                     line_code, line_decode, coded_symbols, SCHEMES,
                     lfsr_keystream, scramble_additive, scramble_self_sync,
                     descramble_self_sync, scramble_64b66b_framed, descramble_64b66b,
                     scramble_128b130b, descramble_128b130b,
                     scramble_128b132b, descramble_128b132b,
                     encode_8b10b_words, decode_8b10b, code_8b10b,
                     pam4_gray_encode, pam4_gray_decode, pam4_gray_levels,
                     precode, precode_inverse, pam3_encode, pam3_decode,
                     GRAY_PAM4_BITS, PAM4_LEVELS_MV, PAM4_OUTER_MV,
                     PAM3_BLOCK_BITS, PAM3_BLOCK_SYMBOLS, PAM3_BITS_PER_SYMBOL)
from .patterns import (register_pattern, resolve_pattern, describe_pattern, replay_pattern,
                       PATTERNS)
from .quinary import (encode_8b1q4, quartet_subsets, quartet_constellation, quinary_volts,
                      partial_response, pam5_symbols, side_stream_bits, subset_indices,
                      test_mode_symbols, QUINARY_SYMBOLS, PARTIAL_RESPONSE_TAPS,
                      CLAUSE_40_BUDGET, SYMBOL_RATE_BD)
from .bus import open_drain, combine_drivers, uart_frame, uart_decode
from .acquire import AcquisitionProfile, acquire_record, record_decimation

__all__ = [
    "physics", "impairments", "events", "grammar", "pam4", "grid", "instrument", "streams", "compose",
    "patterns", "register_pattern", "resolve_pattern", "describe_pattern", "replay_pattern",
    "PATTERNS",
    "measure", "sweep", "cdr", "eye", "sparam", "stream", "simreal", "rx", "scene", "optical", "coding", "bus", "acquire", "ctle", "dfe", "ffe", "Scene",
    "AcquisitionProfile", "acquire_record", "record_decimation",
    "open_drain", "combine_drivers", "uart_frame", "uart_decode",
    "dc_balanced", "scramble_64b66b", "running_disparity", "max_run",
    "quinary", "encode_8b1q4", "quartet_subsets", "quartet_constellation", "quinary_volts",
    "partial_response", "pam5_symbols", "side_stream_bits", "subset_indices",
    "test_mode_symbols", "QUINARY_SYMBOLS", "PARTIAL_RESPONSE_TAPS", "CLAUSE_40_BUDGET",
    "SYMBOL_RATE_BD",
    "line_code", "line_decode", "coded_symbols", "SCHEMES",
    "lfsr_keystream", "scramble_additive", "scramble_self_sync", "descramble_self_sync",
    "scramble_64b66b_framed", "descramble_64b66b",
    "scramble_128b130b", "descramble_128b130b", "scramble_128b132b", "descramble_128b132b",
    "encode_8b10b_words", "decode_8b10b", "code_8b10b",
    "pam4_gray_encode", "pam4_gray_decode", "pam4_gray_levels",
    "precode", "precode_inverse", "pam3_encode", "pam3_decode",
    "GRAY_PAM4_BITS", "PAM4_LEVELS_MV", "PAM4_OUTER_MV",
    "PAM3_BLOCK_BITS", "PAM3_BLOCK_SYMBOLS", "PAM3_BITS_PER_SYMBOL",
    "to_optical", "rin_noise", "shot_noise", "chromatic_dispersion", "mpi", "laser_chirp",
    "modulate_field", "fiber", "field_mpi", "edfa", "photodetect", "tia",
    "stream_convolve", "stream_blocks", "channel_fir", "separability", "feature_vector",
    "eye_density", "eye_recover", "eye_crossings",
    "recover_clock", "jitter_transfer", "tracked_out_fraction", "ssc_phase", "apply_ssc",
    "timing_source", "apply_phase", "phase_noise", "recover_and_fold",
    "read_touchstone", "write_touchstone", "sparam_channel", "touchstone_channel",
    "N", "T", "Grid", "Jitter", "Streams", "tx_ffe", "carrier_symbols", "from_symbols", "resonant_reflection", "de_emphasis_taps",
    "nominal_nonlinearity", "crosstalk_matrix", "crosstalk_sum", "hybrid_echo",
    "single_pair_observation", "db_to_coupling",
    "differential_pair", "differential_mode", "common_mode", "supply_coupling",
    "IMPAIRMENTS", "apply_impairment", "domain_randomize",
    "mix_at_constant_power", "burst_gate", "apply_gated", "realistic_noise", "drift", "electrical_idle",
    "Event", "EventList", "place_events", "apply_events", "defect_symbols",
    "label_windows", "windows_from_centers", "nominal_ui_windows",
    "measure_window", "ui_heights", "eye_mask", "slope_reversals",
    "step_overshoot_fraction", "second_order_step", "damped_sinusoid",
    "MECHANISMS", "PLACEMENTS",
    "sample", "generate", "CARRIERS", "ENVELOPES", "deep_capture", "PATHOLOGIES",
    "interleave_adc", "shaped_noise_floor", "clip_adc", "quantize_adc", "digitize_adc",
    "scope_bandwidth", "probe_loading", "timebase_jitter", "store_record",
    "sinad_noise_rms", "converter_noise_rms",
    "quantisation_floor_db_per_hz",
    "Signal", "dataset", "rederive_anchor", "canonicalize", "op_kind",
    "eye_height", "best_phase", "attributes", "align_symbols", "ground_truth", "pattern_period",
    "hold_constant", "realized_table", "solve_monotonic", "__version__",
]
