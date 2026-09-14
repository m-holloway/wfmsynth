"""The parameter names each op accepts. GENERATED -- see tools/derive_op_keys.py.

Regenerate with:  python tools/derive_op_keys.py > wfmsynth/opkeys.py
"""
from __future__ import annotations

import difflib
import os

# Keys that are structure rather than parameters. Anything else beginning with an
# underscore is the annotation namespace and is also allowed through.
STRUCTURAL = frozenset({"op"})

OP_KEYS = {
    "ac_couple": frozenset({"fc_frac", "fc_hz"}),
    "acquire": frozenset({"op", "profile", "tap"}),
    "agc": frozenset({"gain_limits", "metric", "on_limit", "q", "target", "tau_s"}),
    "carrier": frozenset({"amp", "band_limit_tr", "causal", "cycles", "duty", "f_hz", "fn", "jitter", "kind", "level", "n", "n_ui", "offset", "pattern", "phase_rad", "seed", "symmetry", "tr_frac"}),
    "cascade": frozenset({"node", "path"}),
    "coded": frozenset({"bits", "block", "causal", "header", "jitter", "levels", "lsb_first", "n", "n_bits", "op", "pattern", "poly", "prbs", "prbs_phase", "prbs_seed", "rd", "scheme", "seed", "symbols", "sync", "tr_frac"}),
    "crosstalk": frozenset({"aggressor", "amp", "band_limit_tr", "causal", "coupling", "cycles", "duty", "f_hz", "fn", "jitter", "kind", "level", "n", "n_ui", "offset", "pattern", "phase_rad", "seed", "symmetry", "td_frac", "tr_frac"}),
    "crosstalk_matrix": frozenset({"baud_offsets", "couplings", "echo_db", "echo_f_hp_hz", "echo_td_ps", "fext_db", "kind", "next_db", "next_td_ps", "seeds", "synchronous"}),
    "ctle": frozenset({"dc_gain", "fp1_ghz", "fp2_ghz", "fz_ghz"}),
    "dcd": frozenset({"frac_ui", "half_width", "plateau_ui", "ps", "threshold"}),
    "de_emphasis": frozenset({"db", "spb"}),
    "dfe": frozenset({"causal", "cdr", "levels", "output", "scale", "taps"}),
    "digitize": frozenset({"bits", "enob", "full_scale", "interleave", "n_out", "noise_rms", "snr_db"}),
    "dispersion": frozenset({"strength"}),
    "drift": frozenset({"amount", "kind", "shape"}),
    "edfa": frozenset({"gain_db", "nf_db", "p_ase_scale"}),
    "eo": frozenset({"adiabatic", "alpha", "bias", "er_db", "kind", "linewidth_hz", "p_avg", "vpi"}),
    "events": frozenset({"aggressor", "amp", "band_limit_tr", "causal", "count", "cycles", "depth", "duty", "every", "f0_hz", "f_hz", "floor", "fn", "fraction", "hold_frac", "indices", "intervals", "jitter", "kind", "level", "min_run", "motif", "n", "n_ui", "offset", "on", "op", "pattern", "phase_rad", "polarity", "q", "rate_hz", "samples", "seed", "severity", "symbols", "symmetry", "tau_s", "times", "tr_factor", "tr_frac", "transitions_only", "which", "width", "zeta"}),
    "fiber": frozenset({"D_ps_nm_km", "atten_db_km", "gamma_per_w_km", "length_km", "wavelength_nm"}),
    "hybrid_echo": frozenset({"amp", "band_limit_tr", "causal", "cycles", "duty", "f_hp_frac", "f_hp_hz", "f_hz", "fn", "isolation_db", "jitter", "kind", "level", "n", "n_ui", "offset", "own", "pattern", "phase_rad", "seed", "symmetry", "td_ps", "td_samples", "tr_frac"}),
    "intra_pair_skew": frozenset({"gain_imbalance", "skew_ps"}),
    "lossy": frozenset({"causal", "eps_r", "length_in", "loss_at_ghz", "loss_db", "skin_k", "tand", "trend", "trend_floor_db"}),
    "multipair": frozenset({"echo_db", "echo_f_hp_hz", "echo_td_ps", "fext_db", "levels", "local_seed", "n_ui", "next_db", "next_td_ps", "pattern", "remote_seed", "tr_frac"}),
    "nonlinearity": frozenset({"a_base", "compression", "level_noise", "rise_fall_ratio"}),
    "open_drain": frozenset({"c_bus_f", "center", "r_pullup_ohm", "r_sink_ohm", "v0", "v_dd"}),
    "optical": frozenset({"bw_hz", "er_db", "p_avg", "photons_per_unit", "rin_db_per_hz", "shot"}),
    "optical_mpi": frozenset({"delay_samples", "reflectivity"}),
    "photodetect": frozenset({"photons_per_unit", "responsivity", "shot"}),
    "probe": frozenset({"atten", "bw_hz", "c_load_f", "causal", "kind", "noise_rms", "order", "r_source"}),
    "reflect": frozenset({"gamma_l", "gamma_s", "n_bounce", "node", "td_frac", "td_ps", "td_samples"}),
    "resonant_reflect": frozenset({"f0_frac", "f0_ghz", "gamma0", "q", "td_frac", "td_ps"}),
    "rx_ffe": frozenset({"pre", "spacing_ui", "spb", "tap_spacing", "taps"}),
    "rx_noise": frozenset({"bw_hz", "density", "exact_rms", "rms"}),
    "sample_clock": frozenset({"band_tol", "drift_ppm_per_s", "half_width", "n_out", "phase0_s", "ppm", "span"}),
    "scope": frozenset({"bw_hz", "causal", "kind", "order"}),
    "sparam": frozenset({"freqs", "path", "ports", "s21"}),
    "ssc": frozenset({"f_ssc", "profile", "spread"}),
    "store": frozenset({"bits", "clip", "dither_lsb", "full_scale", "headroom"}),
    "supply_coupling": frozenset({"am_depth", "f_ripple_hz", "psij_ps", "supply"}),
    "symbols": frozenset({"bits", "block", "causal", "header", "jitter", "levels", "lsb_first", "n", "n_bits", "op", "pattern", "poly", "prbs", "prbs_phase", "prbs_seed", "rd", "scheme", "seed", "symbols", "sync", "tr_frac"}),
    "tia": frozenset({"bw_hz", "gain", "thermal_rms"}),
    "timebase": frozenset({"rms_ps"}),
    "timing": frozenset({"phase_noise", "pj", "rj_ps", "ssc", "wander"}),
    "tx_ffe": frozenset({"pre", "spb", "taps"}),
}


def check(op, params):
    """Raise on a parameter this op does not read.

    Ops take their parameters as a dict, so without this an invented or misspelled name is
    accepted and silently does nothing: the record renders, the digest is stable, and it is not
    the record that was asked for. `scope(causal=False)` was once exactly this -- the flag was
    missing from the op's forwarding list, so it rendered the causal path and hashed identically
    to the default.

    Unknown ops are left alone; `compose` reports those with the list of what it can execute.
    """
    if op not in OP_KEYS:
        return
    allowed = OP_KEYS[op]
    unknown = [k for k in params
               if k not in allowed and k not in STRUCTURAL and not k.startswith("_")]
    if not unknown:
        return
    audit = os.environ.get("WFMSYNTH_OPKEY_AUDIT")
    if audit:
        # Maintenance hook. Set the variable to a path and run the suite to collect every key the
        # real code passes that this table lacks, instead of failing on the first one. That is how
        # the table is repaired after an op learns a parameter the derivation cannot see.
        with open(audit, "a") as fh:
            for k in sorted(unknown):
                fh.write(f"{op}" + chr(9) + f"{k}" + chr(10))
        return
    lines = []
    for k in sorted(unknown):
        near = difflib.get_close_matches(k, sorted(allowed), n=2, cutoff=0.6)
        lines.append(f"  {k!r}" + (f" -- did you mean {' or '.join(repr(n) for n in near)}?"
                                   if near else ""))
    raise TypeError(
        f"op {op!r} does not take " + ", ".join(repr(k) for k in sorted(unknown)) + ".\n"
        + "\n".join(lines)
        + f"\n  {op!r} takes: " + ", ".join(sorted(allowed))
        + "\nA parameter an op does not read would be recorded in the recipe and change nothing.")

