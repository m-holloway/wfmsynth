"""A unit suffix means one thing, and a new parameter cannot quietly make it mean another.

This library carries units in parameter SUFFIXES rather than in a type: `tr_s` is seconds,
`td_ps` picoseconds, `bw_hz` hertz, `tr_frac` a fraction, `loss_db` decibels. That is the
proportionate choice here -- 257 public parameter names exist and 171 of them are genuinely
dimensionless, so a `Quantity` layer would tax the majority to police the minority, and `Grid`
already owns the one conversion that matters (fs, baud, and every `_frac` defined against them).

What the convention lacks without this file is a way to stay true. It is already very nearly
true: `_s` has twelve members and exactly ONE is not seconds --

    reflect(gamma_s=) is the SOURCE-end reflection coefficient. It is dimensionless, it sits
    beside a `td_ps` that really is a time, and it is in the README's first example.

-- so the risk is not that name, which is documented. The risk is the SECOND one.

The gate is the same enumerate-and-excuse shape as the knob sweep and the LTI list: the set of
unit-suffixed parameter names is frozen below, so adding one fails until it is written down.
That costs a line when a new unit-carrying parameter appears -- rare -- and it forces the author
to look at the suffix at the one moment when looking is worth anything.
"""
from __future__ import annotations

from wfmsynth.opkeys import OP_KEYS

# The suffixes that carry a dimension, matched longest-first so `_ghz` is not read as `_hz`.
UNIT_SUFFIXES = {
    "_ghz": "gigahertz", "_frac": "a fraction (of a UI, or of the record -- the op says which)",
    "_ohm": "ohms", "_rad": "radians", "_ui": "unit intervals", "_km": "kilometres",
    "_nm": "nanometres", "_bd": "baud", "_hz": "hertz", "_ps": "picoseconds",
    "_db": "decibels", "_in": "inches", "_s": "seconds", "_f": "farads", "_v": "volts",
}

# Names whose suffix does NOT mean what the table above says, each with the reason. Keep this as
# close to empty as the library allows: an entry here is a name that reads as a lie.
EXCEPTIONS = {
    "gamma_s": "SOURCE-end reflection coefficient, not seconds. Dimensionless. Documented in "
               "Signal.reflect and annotated in the README example. See also gamma_l (LOAD).",
}

# Every unit-suffixed parameter the public API currently has. Frozen deliberately: a new one
# must be added here, which is the moment to check that its suffix is honest.
KNOWN_UNIT_PARAMS = frozenset({
    'D_ps_nm_km', 'ac_fc_hz', 'atten_db_km', 'band_hi_hz', 'band_lo_hz', 'bw_hz', 'c_bus_f',
    'c_load_f', 'dev_hz', 'dev_rad', 'diode_on_ohm', 'drift_ppm_per_s', 'echo_db',
    'echo_f_hp_hz', 'echo_td_ps', 'edge_frac', 'er_db', 'f0_frac', 'f0_ghz', 'f0_hz', 'f1_hz',
    'f2_hz', 'f_hp_frac', 'f_hp_hz', 'f_hz', 'f_ripple_hz', 'fc_frac', 'fc_hz', 'fext_db',
    'fp1_ghz', 'fp2_ghz', 'frac_ui', 'fs_hz', 'fz_ghz', 'gain_db', 'gamma_per_w_km',
    'gamma_s', 'hold_frac', 'isolation_db', 'length_in', 'length_km', 'linewidth_hz',
    'loss_at_ghz', 'loss_db', 'n_ui', 'next_db', 'next_td_ps', 'nf_db', 'overload_tau_s',
    'phase0_s', 'phase1_rad', 'phase2_rad', 'phase_rad', 'phase_s', 'pink_frac', 'plateau_ui',
    'psij_ps', 'r_load_ohm', 'r_pullup_ohm', 'r_sink_b_ohm', 'r_sink_ohm', 'r_term_ohm',
    'rate_hz', 'rds_on_ohm', 'rin_db_per_hz', 'rj_ps', 'rms_ps', 'skew_ps', 'snr_db',
    'spacing_ui', 't_off_s', 't_on_s', 't_start_frac', 't_start_s', 't_step_frac', 't_step_s',
    'tau_frac', 'tau_s', 'td_frac', 'td_ps', 'tr_frac', 'tr_s', 'trend_floor_db',
    'wavelength_nm', 'width_frac', 'width_s'
})


def _suffixed(name):
    for suf in sorted(UNIT_SUFFIXES, key=len, reverse=True):
        if name.endswith(suf):
            return suf
    return None


def _all_params():
    return {k for keys in OP_KEYS.values() for k in keys}


def test_a_new_unit_suffixed_parameter_must_be_declared():
    """The gate with teeth. A parameter added with a unit suffix fails here until someone lists
    it -- and listing it is when they notice whether `_s` really means seconds."""
    undeclared = sorted({n for n in _all_params() if _suffixed(n)} - KNOWN_UNIT_PARAMS)
    assert not undeclared, (
        "these carry a unit suffix and are not declared in KNOWN_UNIT_PARAMS:\n  "
        + "\n  ".join(f"{n!r} (suffix {_suffixed(n)!r} = {UNIT_SUFFIXES[_suffixed(n)]})"
                       for n in undeclared)
        + "\n\nAdd each after checking the suffix is honest. If it is not -- if it means "
          "something other than that dimension -- put it in EXCEPTIONS with the reason, the way "
          "`gamma_s` is, and consider whether a different name would serve better.")


def test_the_declared_list_does_not_outlive_the_parameters():
    """Honest in the other direction too, like the UNDERSAMPLED list: a name removed from the
    API must leave here, or the list decays into folklore nobody trusts."""
    stale = sorted(KNOWN_UNIT_PARAMS - _all_params())
    assert not stale, f"declared but no longer a parameter of any op: {stale}"


def test_every_exception_is_a_real_parameter_and_says_why():
    for name, why in EXCEPTIONS.items():
        assert name in KNOWN_UNIT_PARAMS, f"{name!r} is excused but not declared"
        assert name in _all_params(), f"{name!r} is excused but is not a parameter"
        assert len(why) > 30, f"{name!r}: say what it actually means, not just that it differs"


def test_the_one_known_liar_is_still_the_only_one():
    """A tripwire on the project's own claim rather than a correctness assertion. If a second
    name is ever excused, the suffix convention has stopped being a convention, and the decision
    to carry units in names rather than in types deserves revisiting."""
    assert set(EXCEPTIONS) == {"gamma_s"}, (
        f"EXCEPTIONS has grown to {sorted(EXCEPTIONS)}. One exception is a quirk; two is a "
        f"pattern, and the naming convention is no longer carrying its weight.")
