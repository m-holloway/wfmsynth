"""
wfmsynth.compose — provenance-first composable synthesis.

Build a signal as an ordered graph of components, each recording its type and the exact
knob values used, so every waveform carries a complete, serializable **recipe**. The
recipe round-trips: `Signal.from_recipe(r).waveform()` reproduces the samples bit-for-bit
(a property asserted in `wfmsynth.validate`). That turns synthetic data into
ground-truth-to-arbitrary-depth training data — you know precisely what produced each
waveform — and makes datasets reproducible, diffable and auditable.

    g = Grid(fs=256e9, baud=112e9, n=1 << 14)
    sig = (Signal(seed=42, grid=g)
           .carrier("pam4", n_ui=g.n // 8, pattern="prbs13q", jitter=dict(rj=0.4))
           .lossy(loss_db=15.0, loss_at_ghz=26.0, causal=True)
           .reflect(td_ps=55.0, gamma_s=0.4, gamma_l=0.4)
           .digitize(snr_db=32.0, enob=5.5, interleave=dict(m_cores=4, offset_mm=0.01)))
    x, recipe = sig.waveform(), sig.recipe()        # samples + full provenance (JSON-able)
    assert (Signal.from_recipe(recipe).waveform() == x).all()   # exact round-trip

`lead_in=True` renders a LEAD-IN and discards it, so the delivered record never contains the
chain's turn-on: the samples before index 0 are a quiescent line, and a record that begins
mid-pattern otherwise begins on a step no running link has. The guard length is measured from the
chain's own impulse response. Default off, and off is bit-identical (see `Signal.with_lead_in`).

Ops compose over the validated primitives (`physics`, `instrument`), so every knob is a
real, documented parameter — nothing is hidden or randomized-but-unrecorded. Randomness
(jitter, ADC noise) is driven by the Signal's single seed, threaded through the ops in
order, which is what makes the round-trip exact.
"""
from __future__ import annotations
from dataclasses import asdict as _asdict, dataclass, field
from typing import Optional
import warnings

import re

import math

import numpy as np
from scipy.signal import resample_poly

from . import physics as P
from . import opkeys as OPKEYS
from . import instrument as INST
from .grid import Grid
from .streams import Streams


# --------------------------------------------------------------- op executors
# Each executor takes (x, params, streams, grid, idx). Randomness is drawn from a named
# ROLE stream (`streams.role(...)`) keyed by factor + op index, never from a single
# shared rng — so re-rolling one factor leaves every other factor bit-identical.
def _grid_n(grid, p):
    return p.get("n", grid.n if grid is not None else None)


def _carrier(p, streams, grid, idx):
    j = p.get("jitter")
    jitter = P.Jitter(**j) if j else None
    common = dict(n_ui=p.get("n_ui", 32), tr_frac=p.get("tr_frac", 0.15),
                  seed=p.get("seed", 1), n=_grid_n(grid, p), causal=p.get("causal", False),
                  jitter=jitter, rng=streams.role(f"jitter/{idx}"))
    if p["kind"] == "pam4":
        return P.pam4(pattern=p.get("pattern", "legacy"), **common)
    if p["kind"] == "nrz":
        return P.nrz(pattern=p.get("pattern", "legacy"), **common)
    # pam<N> for any other N. 'pam4' is matched above and keeps its own map, so no PAM4
    # stream can reach the generic path.
    m = re.fullmatch(r"pam(\d+)", str(p["kind"]))
    if m:
        return P.pam(int(m.group(1)), pattern=p.get("pattern", "uniform"), **common)
    # a unipolar, VOLTS-NATIVE analog source (see physics.cmos): not on the +/-1 amp/offset
    # convention every other analog kind shares, so it gets its own small dispatch rather than
    # being squeezed through the generic kwarg map below.
    if p["kind"] in P.VOLTS_ANALOG_KINDS:
        kw = dict(n=_grid_n(grid, p), fs=getattr(grid, "fs", None))
        for k in ("v_lo", "v_hi", "duty", "f_hz", "cycles", "tr_s", "phase_rad", "causal"):
            if k in p:
                kw[k] = p[k]
        return {"cmos": P.cmos}[p["kind"]](**kw)
    # analog and arbitrary carriers. These are not data, so the symbol-oriented arguments in
    # `common` do not apply to them: they take a frequency and the grid.
    if p["kind"] in P.ANALOG_KINDS or p["kind"] == "arbitrary":
        kw = dict(n=_grid_n(grid, p), fs=getattr(grid, "fs", None))
        for k in ("f_hz", "cycles", "amp", "offset", "phase_rad", "duty", "symmetry",
                  "tr_frac", "level", "causal", "band_limit_tr",
                  "t_step_s", "t_step_frac", "tr_s", "t_start_s", "t_start_frac",
                  "width_s", "width_frac", "tau_s", "tau_frac", "decay",
                  "f0_hz", "f1_hz", "method", "amp1", "amp2", "phase1_rad", "phase2_rad",
                  "f2_hz", "rms", "df", "pink_frac", "band_lo_hz", "band_hi_hz"):
            if k in p:
                kw[k] = p[k]
        if p["kind"] == "arbitrary":
            fn = p.get("fn")
            if not callable(fn):
                raise ValueError("carrier kind 'arbitrary' needs a callable `fn(t)`; it is "
                                 "not serialisable, so a recipe carrying it cannot be "
                                 "replayed from JSON alone")
            return P.arbitrary(fn, **{k: v for k, v in kw.items()
                                      if k in ("n", "fs", "band_limit_tr", "causal")})
        fn = {"sine": P.sine, "square": P.square, "triangle": P.triangle,
              "sawtooth": P.sawtooth, "dc": P.dc, "step": P.step, "pulse": P.pulse,
              "exp": P.exp, "chirp": P.chirp_sweep, "two_tone": P.two_tone,
              "noise": P.analog_noise}[p["kind"]]
        import inspect
        ok = set(inspect.signature(fn).parameters)
        kw2 = {k: v for k, v in kw.items() if k in ok}
        if p["kind"] == "noise" and "rng" in ok:
            kw2["rng"] = streams.role(f"carrier_noise/{idx}")
        return fn(**kw2)
    raise ValueError(f"unknown carrier kind {p['kind']!r} (use 'nrz', 'pam4', 'pam<N>', "
                     f"one of {P.ANALOG_KINDS}, one of {P.VOLTS_ANALOG_KINDS}, or 'arbitrary')")


def _op_carrier(x, p, streams, grid, idx):
    return _carrier(p, streams, grid, idx)


# The `symbols` op carries its symbols one of two ways, and the pair IS the reproducibility
# contract (see wfmsynth.patterns): a literal list -- data, replayable with no code at all -- or a
# `pattern` BLOCK naming a registered generator and recording its resolved parameters, which is
# readable and diffable against the document the sequence came from. When both are present the
# literal list is authoritative and the block is provenance, so a recipe can be self-describing AND
# need nothing installed. Order matters here: preferring the block would make an embedded recipe
# fail on the consumer who has no registry entry, which is the whole case embedding exists for.
def _source_symbols(p):
    if p.get("symbols") is not None:
        return np.asarray(p["symbols"], float)
    if p["op"] == "coded":
        # DERIVED, not stored: the coded symbols are a function of the recorded bit source and the
        # recorded code, so the recipe carries the code rather than its 10-bits-per-byte output.
        # (A literal list still wins above -- that is how the lead-in's cyclic extension is handed
        # back through this same door.)
        return _coded_symbols(p)
    block = p.get("pattern")
    if block is None:
        raise ValueError("a 'symbols' op needs either symbols=[...] or a pattern block "
                         "(Signal.pattern(name, length=...))")
    from . import patterns as PAT
    return np.asarray(PAT.replay(block), float)


def _source_n_ui(p):
    """The symbol count of a source op, WITHOUT rendering it -- the lead-in sizer needs the
    record's samples-per-UI before anything is generated. A pattern block always records `length`
    because every registered generator takes it, so the count is readable from the JSON."""
    if p["op"] == "carrier":
        return int(p.get("n_ui", 32))
    if p.get("symbols") is not None:
        return len(p["symbols"])
    if p["op"] == "coded":
        # The only source whose symbol count is NOT readable from the JSON: the code's framing
        # overhead and its truncation to whole blocks decide it (64 payload bits become 66), so the
        # count comes from coding it. Cheap -- bits, not samples -- and exact, which the arithmetic
        # would not be for a payload that is not a whole number of blocks.
        return len(_coded_symbols(p))
    return int((p.get("pattern") or {}).get("params", {})["length"])


# --------------------------------------------------------------- the coded source
# `bits -> [code] -> symbols`, recorded as ONE op. A line code transforms BITS and every other op
# in the chain transforms SAMPLES, so there is no mid-chain stage that could express one; folding
# the code into the source is what puts the scheme, the polynomial and the seed into the recipe,
# and therefore into the content digest, instead of leaving them in the caller's scratch code.
def _coded_bits(p):
    """The PAYLOAD bits, from a literal list or from a PRBS order + length. A literal list is data
    (replayable with nothing installed); `prbs=` is the compact route for the long payloads a
    scrambler is normally measured on, and both are fully recorded."""
    if p.get("bits") is not None:
        return np.asarray(p["bits"]).astype(int)
    if p.get("prbs") is None:
        raise ValueError("a 'coded' op needs the payload bits: bits=[...] or prbs=<order> with "
                         "n_bits=<count>")
    # `prbs_seed` / `prbs_phase`, not `seed` / `phase`: on this op `seed` is the SCRAMBLER's seed,
    # and the two are different knobs with the same word for a name.
    kw = {k[5:]: p[k] for k in ("prbs_seed", "prbs_phase") if k in p}
    return P.prbs(int(p["prbs"]), int(p["n_bits"]), **kw).astype(int)


def _coded_symbols(p):
    from . import coding as CODING
    kw = {k: p[k] for k in ("poly", "seed", "sync", "header", "rd", "lsb_first", "block")
          if k in p}
    if "poly" in kw:
        kw["poly"] = tuple(int(t) for t in kw["poly"])       # JSON round-trips a tuple as a list
    return CODING.coded_symbols(p["scheme"], _coded_bits(p),
                                **({"levels": tuple(p["levels"])} if "levels" in p else {}), **kw)


def _op_coded(x, p, streams, grid, idx):
    j = p.get("jitter")
    jitter = P.Jitter(**j) if j else None
    return P.from_symbols(_source_symbols(p), n=_grid_n(grid, p),
                          tr_frac=p.get("tr_frac", 0.15), causal=p.get("causal", False),
                          jitter=jitter, rng=streams.role(f"jitter/{idx}"))


def _op_symbols(x, p, streams, grid, idx):
    j = p.get("jitter")
    jitter = P.Jitter(**j) if j else None
    return P.from_symbols(_source_symbols(p), n=_grid_n(grid, p),
                          tr_frac=p.get("tr_frac", 0.15), causal=p.get("causal", False),
                          jitter=jitter, rng=streams.role(f"jitter/{idx}"))


def _op_capture(x, p, streams, grid, idx):
    from . import capture as CAP
    if p.get("values") is not None:
        y = np.asarray(p["values"], float)
    else:
        y = CAP.load_values(p["path"])
        got = CAP.digest(y)
        want = p.get("sha256")
        if want is not None and got != want:
            raise ValueError(
                f"capture: {p['path']!r} does not match the recipe -- recorded sha256 "
                f"{want} over the sample values, file now digests to {got}. The file changed "
                f"since this recipe was made; re-embed it (capture(..., embed=True)) or point "
                f"the recipe at the file that produced it.")
    fs_hz = p.get("fs_hz")
    grid_fs = getattr(grid, "fs", None) if grid is not None else None
    if fs_hz is not None and grid_fs is None:
        # fs_hz is provenance and what resample=True resamples FROM; with no Grid.fs to
        # reconcile it against, a downstream Hz-denominated knob (lossy(loss_at_ghz=...),
        # probe(bw_hz=...)) has no real rate to mean anything against, silently.
        warnings.warn(
            "capture: fs_hz was given but this chain has no Grid(fs=...), so nothing "
            "downstream has a real sample rate to interpret Hz-denominated knobs against. "
            "Pass grid=Grid(fs=<fs_hz>, n=...) to Signal(...) if this chain uses any.",
            RuntimeWarning, stacklevel=2)
    if fs_hz is not None and grid_fs is not None and float(fs_hz) != float(grid_fs):
        if not p.get("resample"):
            raise ValueError(
                f"capture: the file's fs_hz={float(fs_hz):g} does not match the record's "
                f"fs={float(grid_fs):g}; pass resample=True to resample onto the record's rate")
        target = int(round(len(y) * float(grid_fs) / float(fs_hz)))
        y = resample_poly(y, target, len(y))
    n_target = _grid_n(grid, p)
    if n_target is not None and len(y) != n_target:
        if not p.get("resample"):
            raise ValueError(
                f"capture: {len(y)} samples do not match the record length {n_target} "
                f"(Grid.n or carrier n=); pass resample=True to resample onto it, or match "
                f"the Grid's n to the file's own length")
        y = resample_poly(y, n_target, len(y))
    return y


def _op_burst(x, p, streams, grid, idx):
    """Burst/idle on any carrier -- see `Signal.burst`. Promotes `impairments.burst_gate`
    (already used for localized-defect gating) to a first-class op so it lands in the recipe
    and can be composed after an ANALOG or a UNIPOLAR carrier just as easily as a digital one:
    the gate reads only the sample array, never the carrier's kind."""
    from . import impairments as IMP
    x = np.asarray(x, float)
    n = len(x)
    fs = grid.fs if grid is not None else p.get("fs")
    if fs is None:
        raise ValueError("burst needs a grid (period/phase are given in seconds) or fs=")
    t_on = float(p["t_on_s"]) * fs
    t_off = float(p["t_off_s"]) * fs
    if t_off <= 0:
        return x                    # no off time at all: nothing to gate, not "gate every UI"
    period = t_on + t_off
    phase = float(p.get("phase_s", 0.0)) * fs
    intervals = []
    if t_on > 0:
        start = -(phase % period) if period > 0 else -phase
        while start < n:
            s, w = int(round(start)), int(round(t_on))
            if w > 0 and s + w > 0:
                intervals.append((s, w))
            if period <= 0:
                break
            start += period
    mask = (np.ones(n) if not intervals
            else IMP.burst_gate(n, intervals, edge_frac=float(p.get("edge_frac", 0.1))))
    off = p.get("off", "zero")
    if off == "zero":
        return x * mask
    if off == "hold":
        on = mask >= 0.999
        idx_arr = np.arange(n)
        on_idx = np.where(on, idx_arr, -1)
        ffill = np.maximum.accumulate(on_idx)
        ffill[ffill < 0] = 0                        # nothing on yet: hold the first sample
        held = x[ffill]
        return mask * x + (1.0 - mask) * held
    raise ValueError(f"burst: off must be 'zero' or 'hold', got {off!r}")


def _op_pass_fet(x, p, streams, grid, idx):
    """A pass-FET / analog switch -- see `physics.pass_fet`. `gate` is a nested carrier spec
    for the gate DRIVE (the same nested-spec shape `open_drain`'s `second` and `crosstalk`'s
    `aggressor` already use), thresholded at `vth` to decide the boolean `gate_on`."""
    from .physics import pass_fet
    spec = {"kind": "square", **p.get("gate", {})}
    g = np.asarray(_carrier(spec, streams, grid, f"pass_fet_gate{idx}"), float)
    gate_on = g > float(p.get("vth", 0.0))
    return pass_fet(x, gate_on, rds_on_ohm=p.get("rds_on_ohm", 5.0),
                    r_load_ohm=p.get("r_load_ohm", 1e6), v_rail_hi=p.get("v_rail_hi"),
                    v_rail_lo=p.get("v_rail_lo"), diode_drop=p.get("diode_drop", 0.6),
                    diode_on_ohm=p.get("diode_on_ohm", 1.0))


def _op_modulate(x, p, streams, grid, idx):
    """AM/ASK/OOK, FM/FSK, PM as one op -- `message` is a nested carrier spec (the same shape
    `open_drain`'s `second` and `crosstalk`'s `aggressor` already take), so a caller reaches
    every one of these by choosing the message's `kind`, not a new function per modulation.

    AM/ASK/OOK COMBINE with the upstream `x` (the already-rendered RF carrier -- build it with
    `carrier("sine", f_hz=fc, ...)` first). FM/FSK and PM GENERATE their own carrier at
    `fc_hz` and do not read `x`: frequency/phase modulation is not a per-sample transform of an
    already-rendered fixed-frequency carrier the way AM is (see `physics.fm_modulate`)."""
    from . import physics as P_
    kind = p.get("kind", "am")
    if "message" not in p:
        raise ValueError("modulate needs message=<a carrier spec dict>")
    spec = {"kind": "sine", **p["message"]}
    msg = np.asarray(_carrier(spec, streams, grid, f"modulate_msg{idx}"), float)
    if kind in ("am", "ask", "ook"):
        return P_.am_modulate(x, msg, depth=p.get("depth", 1.0),
                              suppressed=p.get("suppressed", False))
    if "fc_hz" not in p:
        raise ValueError(f"modulate(kind={kind!r}) needs fc_hz=")
    common = dict(n=_grid_n(grid, p), fs=getattr(grid, "fs", None),
                  amp=p.get("amp", 1.0), phase_rad=p.get("phase_rad", 0.0))
    if kind in ("fm", "fsk"):
        if "dev_hz" not in p:
            raise ValueError("modulate(kind='fm'/'fsk') needs dev_hz=")
        return P_.fm_modulate(msg, fc_hz=p["fc_hz"], dev_hz=p["dev_hz"], **common)
    if kind == "pm":
        return P_.pm_modulate(msg, fc_hz=p["fc_hz"], dev_rad=p.get("dev_rad", 1.0), **common)
    raise ValueError(f"modulate: unknown kind {kind!r} (use 'am'/'ask'/'ook'/'fm'/'fsk'/'pm')")


def _op_lossy(x, p, streams, grid, idx):
    kw = {k: p[k] for k in ("length_in", "tand", "eps_r", "skin_k", "causal",
                            "loss_db", "loss_at_ghz", "trend", "trend_floor_db") if k in p}
    if "trend" in kw and kw["trend"] is not None:
        kw["trend"] = tuple(float(t) for t in kw["trend"])   # JSON round-trips it as a list
    return P.lossy_channel(x, grid=grid, **kw)


def _op_reflect(x, p, streams, grid, idx):
    kw = {k: p[k] for k in ("td_frac", "td_samples", "td_ps", "gamma_s", "gamma_l",
                            "n_bounce", "node") if k in p}
    return P.multi_reflection(x, grid=grid, **kw)


def _op_crosstalk(x, p, streams, grid, idx):
    aggr = _carrier({"kind": "nrz", **p.get("aggressor", {"n_ui": 32, "seed": 7})},
                    streams, grid, f"xtalk{idx}")
    return P.crosstalk(x, aggr, coupling=p.get("coupling", 0.1),
                       kind=p.get("kind", "fext"), td_frac=p.get("td_frac", 0.05))


def _op_ac_couple(x, p, streams, grid, idx):
    kw = {k: p[k] for k in ("fc_frac", "fc_hz") if k in p}
    return P.ac_couple(x, grid=grid, **kw)


def _op_tx_ffe(x, p, streams, grid, idx):
    spb = grid.samples_per_ui if grid is not None else p.get("spb")
    if spb is None:
        raise ValueError("tx_ffe needs a grid (for samples/UI) or an explicit spb=")
    return P.tx_ffe(x, p["taps"], spb, pre=p.get("pre", 1))


def _op_crosstalk_matrix(x, p, streams, grid, idx):
    kw = {k: p[k] for k in ("baud_offsets", "seeds", "kind", "synchronous") if k in p}
    return P.crosstalk_matrix(x, grid, p["couplings"], **kw)


_MULTIPAIR_KEYS = ("echo_db", "echo_td_ps", "echo_f_hp_hz", "next_db", "next_td_ps", "fext_db")


def _multipair_lanes(p, grid, n, which, seed, role):
    """The three aggressor lanes (`which`) plus lane 0 of one transmitter, on this grid.

    All four come from the SAME octet stream at the same seed, because on a real four-pair PHY
    they do: the near-end aggressors are the other three coordinates of the victim's own
    transmitter's code word, and the far-end ones are the other three coordinates of the remote
    transmitter's. Generating them as independent random streams would throw away the one
    structural fact a four-pair link has.

    `role` is which END this transmitter is. The two ends of a real link run DIFFERENT scrambler
    polynomials, so master-vs-slave -- not a second seed -- is what makes the far-end lanes
    independent of the near-end ones: seeds of one 33-bit LFSR are PHASES of one sequence, and
    two lanes a few steps apart correlate at 0.58."""
    n_ui = int(p.get("n_ui") or round(n / grid.samples_per_ui))
    pat = p.get("pattern", "8b1q4")
    kw = dict(n=n, tr_frac=p.get("tr_frac", 0.15), causal=True)
    return [P.from_symbols(P.carrier_symbols(f"pam{p.get('levels', 5)}", n_ui, seed=seed,
                                             pattern=pat, pair=k, role=role), **kw)
            for k in which]


def _op_hybrid_echo(x, p, streams, grid, idx):
    """Own transmit leaking into own receive. The `own` spec is a carrier spec, same shape as
    `crosstalk`'s `aggressor`, because the echo's source is a stream the recipe has to name."""
    own = _carrier({"kind": "nrz", **p.get("own", {"n_ui": 32, "seed": 7})},
                   streams, grid, f"echo{idx}")
    kw = {k: p[k] for k in ("isolation_db", "td_ps", "td_samples", "f_hp_hz", "f_hp_frac")
          if k in p}
    return P.hybrid_echo(x, own, grid=grid, **kw)


def _op_multipair(x, p, streams, grid, idx):
    """The single-channel observation of a four-pair link: the chain so far is the WANTED signal
    (the far-end transmitter of this pair through the channel) and this op sums in the other seven
    streams a probe on that pair also sees."""
    if grid is None or grid.baud is None:
        raise ValueError("multipair needs grid=Grid(fs, baud, ...) -- the aggressor lanes are "
                         "generated at the link's own symbol rate, not at a fraction of the record")
    n = len(x)
    other = (1, 2, 3)
    local = _multipair_lanes(p, grid, n, (0,) + other, p.get("local_seed", 1), "master")
    remote = _multipair_lanes(p, grid, n, other, p.get("remote_seed", 1), "slave")
    kw = {k: p[k] for k in _MULTIPAIR_KEYS if k in p}
    return P.single_pair_observation(x, local[0], local[1:], remote, grid=grid, **kw)


def _op_nonlinearity(x, p, streams, grid, idx):
    kw = {k: p[k] for k in ("compression", "level_noise", "rise_fall_ratio", "a_base") if k in p}
    return P.nominal_nonlinearity(x, rng=streams.role(f"nl_noise/{idx}"), **kw)


def _op_resonant_reflect(x, p, streams, grid, idx):
    kw = {k: p[k] for k in ("td_ps", "td_frac", "f0_ghz", "f0_frac", "q", "gamma0") if k in p}
    return P.resonant_reflection(x, grid=grid, **kw)


def _op_supply_coupling(x, p, streams, grid, idx):
    kw = {k: p[k] for k in ("f_ripple_hz", "am_depth", "psij_ps", "supply") if k in p}
    return P.supply_coupling(x, grid, **kw)


def _op_open_drain(x, p, streams, grid, idx):
    """Re-derive the line voltage of an open-drain bus from a two-level carrier.

    The incoming carrier says WHEN a device is sinking; this says what the line then does. Low means
    sinking, so the carrier is thresholded at the midpoint of its own excursion rather than at zero,
    which keeps it correct for a carrier that has already been offset or scaled upstream.

    **This op changes the level domain, deliberately.** An open-drain bus is single-ended between
    ground and its supply, so the output is absolute volts in [V_OL, v_dd] -- unipolar, with a
    settled low that is a resistive DIVIDER above ground, not zero. A centred +/-1 carrier is the
    wrong shape for this bus and a model trained on one learns a driver that does not exist. Pass
    ``center=True`` to re-centre and rescale to the incoming excursion where a downstream stage
    genuinely needs that (it discards V_OL, which is the measurement that reveals a pull-up too
    strong for the sink)."""
    from .physics import open_drain_line
    xa = np.asarray(x, float)
    mid = 0.5 * (np.percentile(xa, 1) + np.percentile(xa, 99))
    sink = xa < mid
    v_dd = float(p.get("v_dd", 3.3))
    sink_b = None
    if p.get("second") is not None:
        # a SECOND driver on the same net, same nested-spec shape `crosstalk`'s `aggressor`
        # and `hybrid_echo`'s `own` already use -- a wired-AND second sink is just another
        # composition of an existing mechanism, not a new one.
        spec = {"kind": "nrz", **p["second"]}
        xb = np.asarray(_carrier(spec, streams, grid, f"open_drain_second{idx}"), float)
        mid_b = 0.5 * (np.percentile(xb, 1) + np.percentile(xb, 99))
        sink_b = xb < mid_b
    if not sink.any() and (sink_b is None or not sink_b.any()):
        # the sink decision thresholds the incoming carrier at the MIDPOINT of its OWN
        # excursion, so a carrier that never varies (a `dc` level, any sign) straddles
        # nothing and never sinks -- the line then does exactly nothing for the whole
        # record, which is the same silent-impairment failure mode
        # docs/ARCHITECTURE.md's "impairments can raise" rule exists to catch.
        warnings.warn(
            "open_drain: the incoming carrier(s) never sink (thresholded at their own "
            "midpoint) -- the line reads as released (v_dd) for the whole record. A "
            "constant carrier can never sink; use one that actually toggles.",
            RuntimeWarning, stacklevel=2)
    y = open_drain_line(sink, grid.fs,
                        r_pullup_ohm=float(p["r_pullup_ohm"]), c_bus_f=float(p["c_bus_f"]),
                        v_dd=v_dd, r_sink_ohm=float(p.get("r_sink_ohm", 20.0)),
                        v0=p.get("v0"), sink_b_on=sink_b,
                        r_sink_b_ohm=(float(p["r_sink_b_ohm"]) if "r_sink_b_ohm" in p else None))
    if p.get("center"):
        lo, hi = float(np.percentile(y, 1)), float(np.percentile(y, 99))
        span = max(hi - lo, 1e-30)
        keep = max(float(np.percentile(xa, 99)) - float(np.percentile(xa, 1)), 1e-30)
        y = (y - 0.5 * (lo + hi)) * (keep / span)
    return y


def _op_intra_pair_skew(x, p, streams, grid, idx):
    # the differential-mode signal a receiver sees after intra-pair skew / gain imbalance
    _p, _n = P.differential_pair(x, grid=grid, skew_ps=p.get("skew_ps", 0.0),
                                 gain_imbalance=p.get("gain_imbalance", 0.0))
    return P.differential_mode(_p, _n)


def _op_ssc(x, p, streams, grid, idx):
    from . import cdr as CDR
    return CDR.apply_ssc(x, grid.fs, f_ssc=p.get("f_ssc", 32e3),
                         spread=p.get("spread", 0.005), profile=p.get("profile", "down"))


def _op_drift(x, p, streams, grid, idx):
    from . import impairments as IMP
    kw = {k: p[k] for k in ("kind", "amount", "shape") if k in p}
    return IMP.drift(x, grid=grid, **kw)


def _op_timing(x, p, streams, grid, idx):
    from . import cdr as CDR
    kw = {k: p[k] for k in ("ssc", "pj", "wander", "rj_ps", "phase_noise") if k in p}
    ph = CDR.timing_source(len(x), grid, rng=streams.role(f"timing/{idx}"), **kw)
    return CDR.apply_phase(x, ph)


def _op_optical(x, p, streams, grid, idx):
    from . import optical as OPT
    y = OPT.to_optical(x, er_db=p.get("er_db", 10.0), p_avg=p.get("p_avg", 1.0))
    if "rin_db_per_hz" in p:
        y = OPT.rin_noise(y, p["rin_db_per_hz"], p.get("bw_hz", 1e10), rng=streams.role(f"rin/{idx}"))
    if p.get("shot"):
        y = OPT.shot_noise(y, p.get("photons_per_unit", 1e4), rng=streams.role(f"shot/{idx}"))
    return y


def _op_dispersion(x, p, streams, grid, idx):
    from . import optical as OPT
    return OPT.chromatic_dispersion(x, strength=p.get("strength", 20.0))


def _op_eo(x, p, streams, grid, idx):                        # E->O: electrical drive -> complex field
    from . import optical as OPT
    return OPT.modulate_field(x, kind=p.get("kind", "mzm"), vpi=p.get("vpi", 1.0),
                              bias=p.get("bias", 0.5), er_db=p.get("er_db"), p_avg=p.get("p_avg", 1.0),
                              alpha=p.get("alpha", 0.0), adiabatic=p.get("adiabatic", 0.0),
                              linewidth_hz=p.get("linewidth_hz", 0.0), grid=grid,
                              rng=streams.role(f"linewidth/{idx}"))


def _op_fiber(x, p, streams, grid, idx):                     # fibre CD (physical β2·L) + attenuation
    from . import optical as OPT
    return OPT.fiber(x, length_km=p.get("length_km", 1.0), D_ps_nm_km=p.get("D_ps_nm_km", 17.0),
                     wavelength_nm=p.get("wavelength_nm", 1550.0), atten_db_km=p.get("atten_db_km", 0.2),
                     gamma_per_w_km=p.get("gamma_per_w_km", 0.0), grid=grid)


def _op_optical_mpi(x, p, streams, grid, idx):               # coherent multipath on the optical field
    from . import optical as OPT
    return OPT.field_mpi(x, delay_samples=p["delay_samples"], reflectivity=p.get("reflectivity", 0.01))


def _op_edfa(x, p, streams, grid, idx):                      # optical amplifier + ASE
    from . import optical as OPT
    return OPT.edfa(x, gain_db=p.get("gain_db", 15.0), nf_db=p.get("nf_db", 5.0),
                    p_ase_scale=p.get("p_ase_scale", 1e-3), rng=streams.role(f"ase/{idx}"))


def _op_photodetect(x, p, streams, grid, idx):               # O->E: square-law detection (closes loop)
    from . import optical as OPT
    return OPT.photodetect(x, responsivity=p.get("responsivity", 1.0), shot=p.get("shot", True),
                           photons_per_unit=p.get("photons_per_unit", 1e4), rng=streams.role(f"shot/{idx}"))


def _op_tia(x, p, streams, grid, idx):                       # optical RX front end
    from . import optical as OPT
    return OPT.tia(x, gain=p.get("gain", 1.0), bw_hz=p.get("bw_hz"), thermal_rms=p.get("thermal_rms", 0.0),
                   grid=grid, rng=streams.role(f"tia/{idx}"))


def _op_de_emphasis(x, p, streams, grid, idx):
    spb = grid.samples_per_ui if grid is not None else p["spb"]
    return P.tx_ffe(x, P.de_emphasis_taps(p["db"]), spb, pre=0)


def _op_scope(x, p, streams, grid, idx):
    # `causal` is forwarded, and it has to be: it is the opt-out for the pre-fix zero-phase
    # front end, and an opt-out a recipe cannot carry is not an opt-out. It was omitted from
    # this dict when the flag was added, and the symptom was that `scope(causal=False)`
    # rendered the causal path and hashed identically to the default.
    kw = {k: p[k] for k in ("kind", "order", "causal") if k in p}
    return INST.scope_bandwidth(x, grid, p["bw_hz"], **kw)


def _op_probe(x, p, streams, grid, idx):
    return INST.probe(x, grid, c_load_f=p.get("c_load_f", 0.5e-12),
                      r_source=p.get("r_source", 50.0), bw_hz=p.get("bw_hz"),
                      kind=p.get("kind", "bessel"), order=p.get("order", 4),
                      noise_rms=p.get("noise_rms", 0.0), atten=p.get("atten", 1.0),
                      causal=p.get("causal"), rng=streams.role(f"probe/{idx}"),
                      r_term_ohm=p.get("r_term_ohm"), compensate=p.get("compensate", 1.0),
                      l_gnd_h=p.get("l_gnd_h", 0.0), coupling=p.get("coupling", "dc"),
                      ac_fc_hz=p.get("ac_fc_hz"), overload_range=p.get("overload_range"),
                      overload_tau_s=p.get("overload_tau_s", 1e-6))


def _op_store(x, p, streams, grid, idx, win=None):
    # `win`: range the EXPORT to the delivered window of a lead-in render, not to the guard.
    # `instrument.store_record(full_scale=None)` uses `headroom * max|x|`; this restates that
    # one expression on the window and passes it explicitly. win=None is bit-identical.
    full_scale = p.get("full_scale")
    if full_scale is None and win is not None:
        peak = float(np.max(np.abs(x[win])))
        if peak > 0.0:
            full_scale = float(p.get("headroom", 1.05)) * peak
    return INST.store_record(x, bits=p.get("bits", 11), full_scale=full_scale,
                             headroom=p.get("headroom", 1.05), clip=p.get("clip", True),
                             dither_lsb=p.get("dither_lsb", 0.0), rng=streams.role(f"store/{idx}"))


def _op_timebase(x, p, streams, grid, idx):
    return INST.timebase_jitter(x, grid, rms_ps=p.get("rms_ps", 0.5), rng=streams.role(f"timebase/{idx}"))


def _op_acquire(x, p, streams, grid, idx):
    from . import acquire as ACQ
    prof = ACQ.AcquisitionProfile(**p["profile"])
    taps = ACQ.acquire_record(x, grid, prof, rng=streams.role(f"acquire/{idx}"))
    return taps[p.get("tap", "stored")]


_EVENT_SKIP = {"op"}


def _op_events(x, p, streams, grid, idx, sink=None):
    """Place + apply a localized mechanism. Optional ``sink`` collects realized events
    for ``Signal.realize()`` without changing the waveform path."""
    from .events import apply_events, place_events
    kw = {k: v for k, v in p.items() if k not in _EVENT_SKIP}
    if kw.get("on") == "aggressor" and isinstance(kw.get("aggressor"), dict):
        spec = {"kind": "nrz", **kw["aggressor"]}
        kw = dict(kw)
        kw["aggressor"] = _carrier(spec, streams, grid, f"events_aggr{idx}")
    ev = place_events(len(x), grid=grid, rng=streams.role(f"events/{idx}"), x=x, **kw)
    y, _mask, ev = apply_events(x, ev, grid=grid)
    if sink is not None:
        sink.extend(ev.events)
    return y


def _op_ctle(x, p, streams, grid, idx):
    from . import rx as RX
    return RX.ctle(x, grid, p["fz_ghz"], p["fp1_ghz"], p["fp2_ghz"], dc_gain=p.get("dc_gain", 1.0))


def _op_rx_ffe(x, p, streams, grid, idx):
    from . import rx as RX
    spb = grid.samples_per_ui if grid is not None else p.get("spb")
    spacing = p.get("tap_spacing") or max(1, int(round(spb * p.get("spacing_ui", 0.5))))
    return RX.ffe(x, p["taps"], spacing, pre=p.get("pre", 0))


# A DFE's decision clock, as a fraction of the symbol rate. See `dfe_instants`.
DFE_LOOP_BW_UI = 1e-3


def dfe_instants(x, p, grid):
    """Where the DFE's decisions are taken, as fractional sample indices — the one thing that
    decides whether a decision-feedback equaliser works or diverges, exposed so it can be read
    and checked rather than inferred from the waveform that comes out.

    **A recovered clock is the DEFAULT**, because a receiver has no other kind. Clock recovery
    and equalisation are one loop in real hardware, and a decision-feedback equaliser is the
    op that cannot survive their being separated: it subtracts its OWN decision's echo from
    the next symbol, so a decision instant that has walked off the symbol centre feeds that
    error forward through the tap history and the equaliser diverges rather than degrading.

    Pass ``cdr={...}`` to set the loop's parameters, or ``cdr=False`` for the historical fixed
    stride — ``int(round(samples_per_ui))`` samples apart, phase chosen once for the whole
    record. That stride is only right while samples-per-UI is a whole number AND constant, so
    it is an opt-out for reproducing an old result, not a modelling choice.

    Measured on one chain, symbols known by construction, everything else held:

    ====================  ==============  ================
    condition             fixed stride    recovered clock
    ====================  ==============  ================
    unmodulated rate      0.0009          0.0018
    rate modulated 0.5%   **0.4279**      0.0017
    ====================  ==============  ================

    So the loop costs about 0.001 of symbol-error rate where there is nothing to track, and
    saves 0.43 where there is. A modulated symbol rate is the common case in serial links.

    ``DFE_LOOP_BW_UI`` is the default loop bandwidth as a fraction of the symbol rate — one
    thousandth, mid-range for serial-link clock recovery. Symbol-error rate is flat within
    0.001 from 3e-4 to 2e-3 and degrades above that, so the value is not delicate.
    """
    levels = np.asarray(p.get("levels", [-1.0, -1 / 3, 1 / 3, 1.0]), float)
    if p.get("cdr", True) is not False:
        from . import cdr as CDR
        c = dict(p["cdr"]) if isinstance(p.get("cdr"), dict) else {}
        if "loop_bw_hz" not in c and "loop_bw_ui" not in c:
            c["loop_bw_ui"] = DFE_LOOP_BW_UI
        spb_f = float(grid.samples_per_ui) if grid is not None else float(p["spb"])
        return CDR.recover_symbol_instants(x, grid=grid, spb=c.pop("spb", spb_f), **c)
    spb = int(round(grid.samples_per_ui)) if grid is not None else int(p["spb"])
    if "phase" in p:
        phase = int(p["phase"])
    else:                                                      # find the eye centre (clearest levels)
        phase, best = spb // 2, 1e9
        for off in range(spb):
            s = x[off::spb]; s = s / (np.percentile(np.abs(s), 99) + 1e-9)
            sep = float(np.mean(np.min(np.abs(s[:, None] - levels[None, :] / np.abs(levels).max()), axis=1)))
            if sep < best:
                best, phase = sep, off
    return np.arange(phase, len(x), spb, dtype=float)


def _dfe_core(x, p, grid):
    """``(instants, samples, scale, equalized, decisions)`` — everything the slicer saw."""
    from . import rx as RX
    levels = np.asarray(p.get("levels", [-1.0, -1 / 3, 1 / 3, 1.0]), float)
    inst = dfe_instants(x, p, grid)
    samples = (x[inst.astype(int)] if p.get("cdr", True) is False
               else np.interp(inst, np.arange(len(x), dtype=float), np.asarray(x, float)))
    if "scale" in p:                                           # a known full scale, e.g. an AGC's
        scale = float(p["scale"])
    else:                                                      # DFE taps/levels are in normalized units
        scale = np.percentile(np.abs(samples), 99) + 1e-9
    eq, dec = RX.dfe(samples / scale, np.asarray(p["taps"], float), levels)
    return inst, samples, scale, eq, dec


def dfe_decisions(x, p, grid):
    """``(instants, equalized, decisions)`` for a ``dfe`` op's parameters — the DFE's per-symbol
    decision path, which is what a symbol-error rate has to be computed from."""
    inst, _samples, _scale, eq, dec = _dfe_core(x, p, grid)
    return inst, eq, dec


def _op_dfe(x, p, streams, grid, idx):
    """The DFE's SUMMING NODE as a waveform — the node a real receiver's slicer looks at.

    A decision-feedback equaliser subtracts a few weighted past decisions from the incoming
    signal. In silicon that subtraction happens at an analog node, continuously, so the node
    carries the FULL bandwidth and the FULL noise of what arrived; what the feedback changes is
    the value AT the sampling instants, which is where the inter-symbol interference is
    cancelled. So this op returns ``x - feedback``, with the feedback held between decisions
    the way a feedback DAC holds it.

    THIS REPLACES A RE-RENDER, and the difference is not cosmetic. The op used to evaluate the
    equalised value once per symbol and rebuild a waveform from those values alone. Measured on
    one chain: the input's 100-120 GHz stop band sits at 23.3 dB, the summing node keeps it at
    23.3 dB, and the re-render dropped it to -7.5 dB. The old output DELETED 30.8 dB OF NOISE
    and every trace of the 15 samples in 16 that fall between decision instants, so a reader
    measuring noise, jitter or edge rate after a DFE was reading the model's pulse shaper
    rather than the signal. A DFE does not clean a waveform up: it cancels ISI and ENHANCES
    random noise through its own tap feedback — measured on the same chain, ISI spread
    0.6214 -> 0.5209 while random-noise spread went 0.0576 -> 0.2198.

    ``output="symbols"`` restores the old re-render for reproducing an earlier result.

    The feedback is carried between decisions as a first-order hold rather than an ideal step.
    An ideal step is not more faithful: a real feedback DAC is band-limited, and an infinitely
    fast step ADDS high-frequency energy the node does not have — measured, it lifted the
    100-120 GHz stop band 1.9 dB above the input's. The DAC's actual shaping is not modelled.
    """
    from . import physics as P
    inst, samples, scale, eq, _dec = _dfe_core(x, p, grid)
    if p.get("output", "node") == "symbols":
        return P.from_symbols(eq, n=len(x), causal=p.get("causal", False))
    x = np.asarray(x, float)
    fb_sym = samples / scale - np.asarray(eq, float)     # exactly what the slicer subtracted
    xi = np.arange(len(x), dtype=float)
    fb = np.interp(xi, inst, fb_sym, left=fb_sym[0], right=fb_sym[-1])
    return x / scale - fb


# ------------------------------------------------------------ DUTY-CYCLE DISTORTION (U-13)
# DCD IS A FIRST-CLASS TRANSMITTER IMPAIRMENT AND HAD NO DIRECT KNOB.
#
# The only way to reach it was `carrier(..., jitter=dict(dcd=<samples>))`, which routes through
# `physics._edge_disp`'s `(dcd/2)*sign(diff(levels))` and then through `_place_symbols`'s
# `searchsorted` on the INTEGER sample grid. MEASURED on a 1024 UI clock pattern at
# fs=256e9/baud=16e9 (mean high pulse width minus mean low pulse width, which is the definition
# of DCD):
#
#     dcd= (samples)   stated      realised mean(high)-mean(low)   ratio to stated
#     0.256            1.000 ps    -7.672703 ps                    -7.67
#     1.024            4.000 ps    -7.672703 ps                    -1.92
#     2.000            7.813 ps   -15.338243 ps                    -1.96
#     4.000           15.625 ps   -30.656340 ps                    -1.96
#
# Two separate faults. The SIGN and SCALE are wrong -- displacing rising edges by +d/2 and
# falling by -d/2 makes the high pulse SHORTER by 2*d, so the realised DCD is -2x the number
# asked for. And the request is QUANTISED to whole samples: 0.256 and 1.024 samples produce the
# BIT-IDENTICAL record above, so any sub-sample DCD -- which is all of them, 1 ps is 0.256
# samples here -- is unreachable. That defect lives in `physics.py`, which this unit does not
# own; it is logged in BACKLOG.md with these numbers.
#
# `_op_dcd` is the direct knob, in ps, on the waveform, with the standard definition:
#
#     dcd_ps == mean(high pulse width) - mean(low pulse width)
#
# and it is realised by displacing each threshold crossing: rising EARLIER by dcd/4, falling
# LATER by dcd/4, which makes each high pulse dcd/2 longer and each low pulse dcd/2 shorter.
# The displacement is applied as a time warp through `instrument.resample_at`, so it is
# sub-sample exact instead of quantised.
#
# THE ONE SUBTLETY, because getting it wrong costs 3 %: the displacement field cannot simply be
# interpolated between crossings. A field ramping from +d at a rising crossing to -d at the next
# falling one has slope s = 2d/T_ui THROUGH the crossing, and warping by a field with slope s
# moves the crossing by d/(1+s), not d -- 3.2 % short at 1 ps DCD on a 62.5 ps UI. So the field
# is held FLAT (a plateau) across each crossing and does its ramping in the middle of the bit,
# where the waveform is settled and warping it does nothing.


def _dcd_displacement(x, spb, dcd_samples, threshold=None, plateau_ui=0.25):
    """The time-warp displacement field for a stated DCD, in samples, one value per sample.
    Positive at rising crossings (which pulls them earlier) and negative at falling ones."""
    x = np.asarray(x, float)
    n = len(x)
    d = float(dcd_samples) / 4.0
    thr = 0.5 * (float(np.percentile(x, 1)) + float(np.percentile(x, 99))) \
        if threshold is None else float(threshold)
    s = x - thr
    up = (s[:-1] <= 0) & (s[1:] > 0)
    dn = (s[:-1] >= 0) & (s[1:] < 0)
    k = np.nonzero(up | dn)[0]
    k = k[(k >= 2) & (k < n - 2)]
    if len(k) == 0:
        return np.zeros(n)
    sign = np.where(up[k], 1.0, -1.0)
    # the crossing position, from a local CUBIC through four samples (t as a cubic in level).
    # A straight-line read is 12x worse HERE, not just in the measurement: it mis-centres the
    # plateau, so the displacement seen at the true crossing is no longer exactly the plateau's.
    c = np.array([np.polyval(np.polyfit(s[i - 1:i + 3], np.arange(i - 1, i + 3, dtype=float), 3),
                             0.0) for i in k])
    gap = np.diff(c, prepend=c[0] - spb, append=c[-1] + spb)
    w = np.minimum(float(plateau_ui) * spb, 0.4 * np.minimum(gap[:-1], gap[1:]))
    knots = np.empty(2 * len(c)); vals = np.empty(2 * len(c))
    knots[0::2] = c - w; knots[1::2] = c + w              # a FLAT plateau across each crossing,
    vals[0::2] = sign * d; vals[1::2] = sign * d          # ramping only where the signal is flat
    return np.interp(np.arange(n, dtype=float), knots, vals, left=vals[0], right=vals[-1])


def _op_dcd(x, p, streams, grid, idx):
    """Duty-cycle distortion as a direct knob. ``ps`` (or ``frac_ui``) is the DCD by its
    standard definition: mean high pulse width minus mean low pulse width. Positive = high
    pulses longer. See the block comment above for the construction and for what the old
    indirect route realised instead.

    MEASURED, 1024 UI clock pattern, tr_frac=0.35, fs=256e9, baud=16e9:

        asked      realised mean(high)-mean(low)     error
        +0.125 ps  +0.12506918 ps                    +0.055 %
        +1.000 ps  +1.00048322 ps                    +0.048 %
        +4.000 ps  +4.00121205 ps                    +0.030 %
        -2.500 ps  -2.50102544 ps                    +0.041 %

    ``threshold`` defaults to the midpoint of the record's 1st and 99th percentile. On a record
    with heavy ISI the crossings are no longer one-per-edge at a fixed level and the realised
    DCD moves off the request; DCD is a TRANSMITTER impairment, so place it on the source."""
    from . import instrument as INST
    spb = grid.samples_per_ui
    if spb is None:
        raise ValueError("dcd needs baud set on the Grid (it is defined per unit interval)")
    if ("ps" in p) == ("frac_ui" in p):
        raise ValueError("dcd takes exactly one of ps= or frac_ui=")
    dcd_s = p["ps"] * 1e-12 if "ps" in p else float(p["frac_ui"]) / grid.baud
    if dcd_s == 0.0:
        return np.asarray(x, float)          # no impairment asked for == no resampling done
    disp = _dcd_displacement(x, spb, dcd_s * grid.fs, threshold=p.get("threshold"),
                             plateau_ui=p.get("plateau_ui", 0.25))
    n = len(np.asarray(x, float))
    return INST.resample_at(x, np.arange(n, dtype=float) + disp,
                            half_width=p.get("half_width", INST.RESAMPLE_HALF_WIDTH))


# ------------------------------------------------------------ receiver front end (U-12), clock (U-09)
def _op_agc(x, p, streams, grid, idx, win=None):
    """Receiver AGC: normalise to a stated level so everything downstream has a known full
    scale. Pair it with ``dfe(scale=<the same target>)``. See `rx.agc_gain`.

    ``win`` is the delivered window of a lead-in render: a BLOCK AGC's gain is measured on the
    samples the caller receives, not on the guard, because the guard is where the turn-on
    ringing lives and a gain set from it scales the whole record wrong. A TRACKING AGC
    (``tau_s=``) already produces a per-sample causal gain, which the window slices correctly
    on its own, so the window does not enter."""
    from . import rx as RX
    kw = dict(target=p.get("target", 1.0), metric=p.get("metric", "rms"), q=p.get("q", 99.0),
              tau_s=p.get("tau_s"), grid=grid, gain_limits=p.get("gain_limits"),
              on_limit=p.get("on_limit", "raise"))
    x = np.asarray(x, float)
    if win is not None and kw["tau_s"] is None:
        return x * RX.agc_gain(x[win], **kw)
    return RX.agc(x, **kw)


def _op_rx_noise(x, p, streams, grid, idx):
    """The RECEIVER's own input-referred noise, at the receiver input -- not the instrument's
    noise floor (`digitize(noise_rms=)`), and not in the same place in the chain. See
    `rx.input_noise`."""
    from . import rx as RX
    return RX.input_noise(x, grid=grid, rms=p.get("rms"), density=p.get("density"),
                          bw_hz=p.get("bw_hz"), rng=streams.role(f"rx_noise/{idx}"),
                          exact_rms=p.get("exact_rms", True))


def _op_sample_clock(x, p, streams, grid, idx):
    """An INDEPENDENT sampling clock: a ppm frequency offset from the link, an optional drift,
    an arbitrary starting phase. See `instrument.sample_clock` for why `resample_poly` cannot."""
    return INST.sample_clock(x, grid, ppm=p.get("ppm", 0.0),
                             drift_ppm_per_s=p.get("drift_ppm_per_s", 0.0),
                             phase0_s=p.get("phase0_s", 0.0), n_out=p.get("n_out"),
                             half_width=p.get("half_width", INST.RESAMPLE_HALF_WIDTH),
                             band_tol=p.get("band_tol", 0.01), span=p.get("span", "fit"))


def _op_sparam(x, p, streams, grid, idx):
    from . import sparam as SP
    common = {k: p[k] for k in ("band", "dc", "band_tol", "linear", "guard") if k in p}
    if "path" in p:
        # `ports` is passed THROUGH, not coerced: a mixed-mode pairing can be a STRING
        # ("13_24") or a tuple of pairs, and `tuple(...)`-wrapping a string shreds it into
        # its individual characters (GitHub #57) instead of forwarding the pairing
        # `touchstone_channel` itself already documents and accepts.
        kw = {k: p[k] for k in ("n_ports", "mode", "term", "check") if k in p}
        if "ports" in p:
            kw["ports"] = p["ports"]
        return SP.touchstone_channel(x, p["path"], grid=grid, **kw, **common)
    return SP.sparam_channel(x, np.asarray(p["freqs"]), np.asarray(p["s21"], complex),
                             grid=grid, **common)


def _op_cascade(x, p, streams, grid, idx):
    from . import sparam as SP
    return SP.cascade_channel(x, p["path"], grid=grid, node=p.get("node", "load"))


def _op_digitize(x, p, streams, grid, idx, win=None):
    # `win` is the delivered window of a lead-in render (see `Signal.lead_in`): the vertical
    # this stage sets must be ranged to the samples the CALLER receives, not to the discarded
    # guard -- the guard is where the turn-on and the still-circular stages' ringing live, and
    # letting it set the range would move the defect instead of removing it. win=None is the
    # default path and every reference below is then the whole record, bit-identically.
    n_out = p.get("n_out")
    if n_out and n_out != len(x):
        x = resample_poly(x, n_out, len(x))
    span = float(np.ptp(x if win is None else x[win])) + 1e-9
    if "noise_rms" in p:                      # absolute noise floor (signal-independent)
        x = x + streams.role(f"noise/{idx}").normal(0.0, p["noise_rms"], len(x))
    elif "snr_db" in p:                       # noise relative to the signal span
        x = x + streams.role(f"noise/{idx}").normal(0.0, span / 10 ** (p["snr_db"] / 20), len(x))
    if p.get("interleave"):
        x = INST.interleave_adc(x, rng=streams.role(f"interleave/{idx}"), **p["interleave"])
    if "bits" in p and "enob" in p:
        raise ValueError("digitize takes bits= (the converter's real depth) or enob= (the legacy "
                         "effective-bits lattice), not both")
    if "bits" in p:                           # the CONVERTER's real lattice, on its full scale
        fs = p.get("full_scale")
        fs = (float(np.max(np.abs(x if win is None else x[win]))) + 1e-12
              if fs is None else float(fs))
        lsb = 2.0 * fs / 2 ** int(p["bits"])
        x = lsb * np.round(x / lsb)
    elif "enob" in p:
        lsb = span / 2 ** p["enob"]
        x = lsb * np.round(x / lsb)
    return x


_EXEC = {"carrier": _op_carrier, "symbols": _op_symbols, "coded": _op_coded,
         "capture": _op_capture, "lossy": _op_lossy, "reflect": _op_reflect,
         "crosstalk": _op_crosstalk, "ac_couple": _op_ac_couple, "digitize": _op_digitize,
         "tx_ffe": _op_tx_ffe, "sparam": _op_sparam, "cascade": _op_cascade,
         "resonant_reflect": _op_resonant_reflect, "nonlinearity": _op_nonlinearity,
         "crosstalk_matrix": _op_crosstalk_matrix,
         "hybrid_echo": _op_hybrid_echo, "multipair": _op_multipair, "ctle": _op_ctle, "rx_ffe": _op_rx_ffe, "dfe": _op_dfe,
         "ssc": _op_ssc,
         "intra_pair_skew": _op_intra_pair_skew, "supply_coupling": _op_supply_coupling,
         "open_drain": _op_open_drain,
         "timing": _op_timing, "optical": _op_optical, "dispersion": _op_dispersion,
         "eo": _op_eo, "fiber": _op_fiber, "optical_mpi": _op_optical_mpi, "edfa": _op_edfa,
         "photodetect": _op_photodetect, "tia": _op_tia,
         "drift": _op_drift, "scope": _op_scope, "timebase": _op_timebase, "store": _op_store,
         "de_emphasis": _op_de_emphasis, "acquire": _op_acquire, "probe": _op_probe,
         "events": _op_events,
         "dcd": _op_dcd, "agc": _op_agc, "rx_noise": _op_rx_noise,
         "sample_clock": _op_sample_clock, "burst": _op_burst, "pass_fet": _op_pass_fet,
         "modulate": _op_modulate}


# --------------------------------------------------------------- fabric: stage-kind homing
# Every op belongs to a stage KIND; the physical chain runs source -> shape -> supply -> channel
# -> instrument. `canonicalize` stable-sorts an op list into that order, so cross-kind authoring
# order commutes by construction (jitter-of-reflection vs reflection-of-jitter is resolved the same
# way regardless of the order they were added). This is OPT-IN via `Signal.canonical()`: the default
# `waveform()` still executes in insertion order, so no existing recipe changes. Order WITHIN a kind
# is preserved (a stable sort), because same-kind ops do not generally commute.
# A probe sits BETWEEN the channel and the instrument -- it is what the instrument observes
# through -- so it gets its own rank rather than sharing the instrument's.
KIND_RANK = {"source": 0, "shape": 1, "supply": 2, "channel": 3, "probe": 4, "instrument": 5}
OP_KIND = {
    "carrier": "source", "symbols": "source", "coded": "source", "capture": "source",
    "tx_ffe": "shape", "de_emphasis": "shape", "events": "shape", "nonlinearity": "shape",
    "timing": "shape", "ssc": "shape", "intra_pair_skew": "shape", "eo": "shape",
    "dcd": "shape", "open_drain": "shape", "burst": "shape", "pass_fet": "shape",
    "modulate": "shape",
    "supply_coupling": "supply", "drift": "supply",
    "lossy": "channel", "reflect": "channel", "resonant_reflect": "channel", "crosstalk": "channel",
    "cascade": "channel",
    "crosstalk_matrix": "channel", "hybrid_echo": "channel", "multipair": "channel",
    "sparam": "channel", "dispersion": "channel", "ac_couple": "channel",
    "optical": "channel", "fiber": "channel", "optical_mpi": "channel", "edfa": "channel",
    "probe": "probe",
    "rx_noise": "instrument", "agc": "instrument",
    "ctle": "instrument", "dfe": "instrument", "rx_ffe": "instrument", "tia": "instrument",
    "sample_clock": "instrument",
    "photodetect": "instrument", "scope": "instrument", "digitize": "instrument",
    "timebase": "instrument", "acquire": "instrument", "store": "instrument",
}


def op_kind(op_name):
    return OP_KIND.get(op_name, "channel")


def canonicalize(ops):
    """Stable-sort an op list into canonical stage-kind order (source->shape->supply->channel->
    instrument). Order within a kind is preserved. Returns a new list."""
    return sorted(ops, key=lambda o: KIND_RANK[op_kind(o["op"])])


# --------------------------------------------------------------- the rendered-and-discarded lead-in
# WHY A RECORD NEEDS A HEAD IT NEVER SHOWS YOU.
#
# Every frequency-domain stage applies a LINEAR convolution (U-16): the response to the record's
# last samples no longer wraps onto its first. That is the correct convolution, and it means the
# samples before index 0 are a QUIESCENT LINE -- so a record that begins mid-pattern begins with a
# turn-on edge, and every stage with memory answers that edge instead of answering the link. A real
# deep-memory capture is a window on a link that was already running and has no such edge.
#
# The fix is not to filter the edge, it is to not deliver it: render `lead` extra samples of the
# same pattern BEFORE the record and `tail` extra AFTER it, run the whole chain on that longer
# record, and hand back only the middle. Whatever the head does -- the turn-on, a zero-phase
# stage's pre-cursor reaching for samples that are not there, a still-circular stage's wrap --
# happens in samples the caller never receives.
#
# HOW LONG. Long enough that the chain's impulse response, started at the extended record's edge,
# is over before the window begins. That length is MEASURED, not assumed: `physics.response_extent`
# is handed the chain's own combined impulse response (the LTI ops applied to a unit impulse) and
# returns the width of the shortest arc holding everything at or above `rel` (1e-5, -100 dB) of its
# peak. The measurement is the same instrument the linear-convolution guard uses, on the same
# threshold, so the lead-in and the guard agree by construction.
#
# WHERE IT LIVES. On the Signal, not on the Grid and not on the render call. `Grid` states what the
# record IS (rate, length, full scale); the lead-in is a property of the CHAIN, because its length
# is a property of the chain's memory -- two Signals on the same Grid need different lead-ins. And
# it has to be in the recipe, or `from_recipe(r).waveform()` would not reproduce the samples.
#
# WHAT IT COSTS, AND WHAT IT CHANGES. The render is (n + lead + tail) / n longer. The delivered
# record is a DIFFERENT record from the same recipe without a lead-in -- necessarily so, since the
# head now has history -- in three ways worth stating plainly:
#   * `carrier` extends the pattern FORWARD, so the window holds symbols[lead_ui : lead_ui+n_ui] of
#     the same PRBS instead of symbols[0 : n_ui]. That is genuine history: those symbols really did
#     precede these. When `lead_ui` is a whole multiple of the pattern's period the window's symbols
#     are unchanged and ONLY the history is added -- which is what the constructed check below uses.
#     (The alternative -- prepending the record's own tail as a cyclic prefix -- would keep the
#     symbols identical for any length, but it would be inventing a history the link never had.)
#   * `symbols` is an explicit, repeating stream, so its own tail IS its history: the lead-in is the
#     cyclic prefix and the window holds exactly the stream the caller passed.
#   * anything whose phase is measured from the record's start (SSC, periodic jitter, a supply tone)
#     now starts mid-cycle, because the link was already running. Random draws are longer, so a
#     jittered or noisy chain draws different numbers.
# Default is OFF (`lead_in=None`) and the recipe carries no lead-in key, so every existing recipe
# renders and hashes exactly as before. Turning it on by default is a versioned change.
LEAD_IN_REL = P.RESPONSE_REL              # -100 dB below the peak: the same threshold as the guard

# THE SOURCE HAS MEMORY TOO, and the extent probe cannot see it: a carrier is not a filter of the
# record, it IS the record, so it contributes no impulse response to measure -- and its edge shaping
# (`physics._shape_edges`, a 4th-order Bessel run forwards AND backwards by `sosfiltfilt`) invents
# the samples past both ends. When a symbol transition falls at the record boundary the record's
# final edge is simply MISSING. MEASURED, as max |record - the same record's steady state|, on a
# constructed periodic pattern with a transition at the wrap, at tr_frac=0.15:
#
#   samples/UI   tr [samples]     no lead-in   1 UI      2 UI      3 UI      4 UI
#        8         2.0 (clamped)     0.76      5.1e-4    8.5e-6    3.5e-8    4.0e-12
#       16         2.4               0.80      3.4e-5    2.7e-9    2.7e-13   0
#       64         9.6               0.95      8.8e-6    4.9e-10   6.0e-14   0
#      256        38.4               0.99      8.0e-6    7.5e-10   8.2e-14   3.8e-14
#
# So the reach is ~13*tr for 1e-9 and ~27*tr to float64 exactness, i.e. it scales with the RISE
# TIME, not with the UI -- a caller who asks for tr_frac=0.5 needs three times the guard a
# tr_frac=0.15 caller does. `lead_in='auto'` therefore takes the LARGER of the measured chain extent
# and `LEAD_SOURCE_TR * tr`, and this floor is why `lead_in='auto'` is never zero on a chain whose
# ops are all memoryless.
LEAD_SOURCE_TR = 32.0                     # multiples of the source's rise time (measured: 27 is exact)

# The ops the extent probe runs: LTI, deterministic, length-preserving, real-valued. These are the
# ops that HAVE an impulse response, and they are exactly the ops that ring on a turn-on.
_LEAD_LTI = {"lossy", "reflect", "resonant_reflect", "ac_couple", "sparam", "cascade", "tx_ffe",
             "de_emphasis", "ctle", "rx_ffe", "scope", "probe", "intra_pair_skew"}
# Ops that are skipped by the probe rather than rejected: memoryless (or a fraction of a UI of
# interpolation), so they do not lengthen the chain's memory, or nonlinear/stochastic, so they have
# no impulse response to measure. An OPTICAL chain is skipped because the field is complex and this
# probe is real -- size an optical chain's lead-in explicitly.
# `dcd` joins `timebase` here for the same reason: it is a sub-UI time WARP through a 64-tap
# interpolator, nonlinear (its displacement field comes from the input's own crossings), so it has
# no impulse response to probe and it does not lengthen the chain's memory. `rx_noise` is
# stochastic. `agc` is nonlinear and its block gain is a functional of the samples it sees, which
# is why it is also in `_LEAD_WINDOW_RANGED` below.
_LEAD_SKIP = {"carrier", "symbols", "coded", "nonlinearity", "crosstalk", "crosstalk_matrix", "digitize",
              "store", "timebase", "timing", "ssc", "supply_coupling", "dfe",
              "optical", "dispersion", "eo", "fiber", "optical_mpi", "edfa", "photodetect", "tia",
              "dcd", "rx_noise", "agc",
              # `burst`'s on/off cadence is periodic, measured from the record's own start --
              # same as `ssc`/`timing` above, it has no impulse response to size a guard from,
              # and the caveat is the same one they already carry: a lead-in changes WHERE in
              # the on/off cycle the delivered window begins, not what the cycle is.
              "burst",
              # `pass_fet` is a nonlinear per-sample decision (gate threshold, then either a
              # divider or a diode clamp) with no impulse response of its own -- the same
              # bucket `dcd`/`agc` are already in.
              "pass_fet"}
# Ops with REAL memory but no probeable impulse response, whose extent is known in closed form.
#
# This category exists because `open_drain` fits none of the other three and putting it in the wrong
# one is a silent error in both directions. It is not LTI -- both the time constant and the target
# switch with the sink state, so superposition fails and the extent probe cannot measure it. But it
# has the LONGEST memory of any stage here: an RC charge through a pull-up is hundreds of
# nanoseconds where every filter in `_LEAD_LTI` is tens of picoseconds. Filing it under `_LEAD_SKIP`
# ("has none") would be exactly the default into the safe-looking pile that `_lead_check_ops` was
# built to prevent, and every record would open on a turn-on transient nobody asked for.
#
# So: state the extent instead of probing for it. Each entry maps an op to a function of its own
# params and the grid, returning the memory extent in SAMPLES. 5 tau is 99.3 % settled, which is the
# same convergence the probe's 1e-5 relative threshold buys on an exponential.
def _echo_reach(td_ps, f_hp_hz, f_hp_frac, grid):
    """The memory of a hybrid echo, in samples: the round trip to the reflecting discontinuity,
    plus the settling of the return-loss high-pass that shapes it.

    5 tau is 99.3 % settled -- the same convergence the LTI probe's 1e-5 threshold buys on a
    first-order response. A first-order high-pass whose corner is `f` of Nyquist has a time
    constant of 1/(pi*f) samples, so the default 0.02 costs ~80 samples and a corner stated in Hz
    costs whatever it costs."""
    f = float(f_hp_frac)
    if f_hp_hz is not None:
        if grid is None:
            raise ValueError("hybrid_echo: f_hp_hz needs a grid")
        f = grid.hz_to_frac_nyquist(float(f_hp_hz))
    td = float(td_ps or 0.0) * 1e-12 * float(grid.fs) if grid is not None else 0.0
    return int(math.ceil(td + 5.0 / (math.pi * max(f, 1e-9))))


# Ops with REAL memory and no probeable impulse response OF THEIR OWN INPUT, whose extent is
# known in closed form.
#
# `hybrid_echo` and `multipair` are here for a reason the LTI probe cannot see: their memory is a
# response to a DIFFERENT stream. Drive the chain with an impulse and the echo term does not
# respond to it at all -- it is a fixed additive function of the op's own transmit -- so
# `response_extent` measures zero and would size the guard to nothing while the echo's delay and
# its return-loss filter tail are both real and both inside the record. `_LEAD_SKIP` ("has no
# memory") would be the same silent error in the same direction. So: state the extent.
_LEAD_ANALYTIC = {
    "open_drain": lambda o, grid: int(math.ceil(
        5.0 * float(o["r_pullup_ohm"]) * float(o["c_bus_f"]) * float(grid.fs))),
    "hybrid_echo": lambda o, grid: _echo_reach(
        o.get("td_ps"), o.get("f_hp_hz"), o.get("f_hp_frac", P.ECHO_HP_FRAC_NYQUIST), grid),
    # the summed observation carries an echo AND a near-end crosstalk delay; the guard has to
    # cover whichever reaches further back
    "multipair": lambda o, grid: max(
        _echo_reach(o.get("echo_td_ps"), o.get("echo_f_hp_hz"),
                    P.ECHO_HP_FRAC_NYQUIST, grid) if o.get("echo_db") is not None else 0,
        int(math.ceil(float(o.get("next_td_ps") or 0.0) * 1e-12 * float(grid.fs)))
        if grid is not None else 0),
}


def _lead_analytic_reach(ops, grid):
    """The largest closed-form memory extent among the ops that have one, in samples."""
    return max((_LEAD_ANALYTIC[o["op"]](o, grid) for o in ops if o["op"] in _LEAD_ANALYTIC),
               default=0)


# Ops a lead-in cannot host, each with the reason. Refusing is the honest answer: the alternative is
# to silently move where these land, and a fault placed at sample 5000 of the record is not the same
# fault at sample 5000 of the record-plus-guard.
_LEAD_REJECT = {
    "modulate": "AM is memoryless but FM/PM integrate the message across the WHOLE record "
               "(a cumulative phase), and the classification is per OP NAME, not per call -- "
               "extending the message backward for a lead-in would change that integral, so "
               "the op is refused uniformly rather than being silently right for one kind and "
               "wrong for another",
    "capture": "a captured record has no synthesized pattern to extend backward in time -- "
               "unlike `carrier`, which renders more of the SAME periodic sequence for the "
               "guard, a lead-in here would have to invent history the capture never had",
    "acquire": "it resamples the record onto a second rate, so the guard's sample count is not the "
               "record's and the window cannot be sliced back out",
    "events": "it places a localized mechanism at an absolute position in the record, and a lead-in "
              "moves the record's origin (BACKLOG: shift event anchors by the lead-in)",
    "drift": "its profile is defined ACROSS the record ('0 to 1 over the capture'), so a longer "
             "render is a different drift",
    "sample_clock": "an independent clock covers a DIFFERENT SPAN of link time than the link's "
                    "own grid, so the guard's sample count is not the record's and the slip "
                    "moves the window's own boundaries; sample the clock after slicing the "
                    "window out, or render the whole thing on the clock's grid",
}
# (op, key) pairs whose value is a FRACTION OF THE RECORD -- a fraction is not a physical quantity,
# and a lead-in changes the record it is a fraction of. State these in absolute units instead.
_LEAD_RELATIVE = {
    ("reflect", "td_frac"): "state the round-trip delay as td_ps= or td_samples=",
    ("resonant_reflect", "td_frac"): "state the delay as td_ps=",
    ("resonant_reflect", "f0_frac"): "state the resonance as f0_ghz=",
    ("ac_couple", "fc_frac"): "state the corner as fc_hz=",
    ("digitize", "n_out"): "a resample changes the record length, so the window cannot be sliced "
                           "back out; resample before the chain instead",
    ("dfe", "phase"): "a fixed decision phase is an absolute sample offset into the record",
}
# The ops that SET THE RECORD'S VERTICAL. They must range to the delivered window: the guard is
# where the ringing lives, and a vertical ranged to the guard would spend the record's codes on
# samples the caller never sees -- which is the very defect the lead-in exists to remove. A BLOCK
# `agc` belongs here for the same reason and it is the sharper case: it does not merely allocate
# codes, it MULTIPLIES the record, so a gain set from the guard's turn-on ringing puts the whole
# record at the wrong level and every stated full scale downstream is then wrong.
_LEAD_WINDOW_RANGED = {"digitize", "store", "agc"}


def _lead_source_geometry(ops, grid):
    """``(n, n_ui, p, q)`` for the source op: the record's sample count, its symbol count, and
    ``n/n_ui`` in lowest terms.

    A lead-in must not change samples-per-UI -- that would re-time the whole record, not add
    history to its head -- and the source's samples-per-UI is exactly ``n / n_ui``. So the lead-in
    is quantized to a whole number of UI: with ``n/n_ui == p/q`` in lowest terms, adding ``k*q``
    symbols and ``k*p`` samples leaves the ratio EXACTLY equal (``(n+kp)/(n_ui+kq) == p/q``), which
    is rational arithmetic, not a tolerance. ``p`` is therefore the lead-in's quantum in samples."""
    from fractions import Fraction
    if not ops:
        raise ValueError("a lead-in needs a source op (add a carrier first)")
    src = ops[0]
    if src["op"] not in ("carrier", "symbols", "coded"):
        raise ValueError(f"a lead-in needs the first op to be the source; ops[0] is "
                         f"{src['op']!r}. The lead-in is rendered BY the source.")
    n = _grid_n(grid, src)
    if n is None:
        raise ValueError("a lead-in needs the record length: pass grid=Grid(n=...) or n= on the "
                         "source op")
    n_ui = _source_n_ui(src)
    if n_ui < 1:
        raise ValueError("a lead-in needs at least one symbol in the source")
    fr = Fraction(int(n), int(n_ui))
    return int(n), int(n_ui), fr.numerator, fr.denominator


def _lead_check_ops(ops):
    """Refuse, with the reason, any op whose meaning a lead-in would silently change -- and refuse
    an op that has not been CLASSIFIED at all. Every op in `_EXEC` is in exactly one of
    `_LEAD_LTI` (it has an impulse response, so it sizes the guard), `_LEAD_SKIP` (memoryless,
    nonlinear or stochastic, so it does not) or `_LEAD_REJECT` (a lead-in would change what it
    means). A new op therefore has to be reasoned about before it can be rendered with a lead-in,
    rather than defaulting into the safe-looking pile."""
    for o in ops:
        if (o["op"] not in _LEAD_LTI and o["op"] not in _LEAD_SKIP
                and o["op"] not in _LEAD_REJECT and o["op"] not in _LEAD_ANALYTIC):
            raise ValueError(f"lead_in: the {o['op']!r} op has no lead-in classification -- add it "
                             f"to _LEAD_LTI (it has an impulse response), _LEAD_SKIP (it has none), "
                             f"_LEAD_ANALYTIC (it has memory but no probeable impulse response, so "
                             f"state the extent) or _LEAD_REJECT (a lead-in changes what it means) "
                             f"in compose.py")
        why = _LEAD_REJECT.get(o["op"])
        if why is not None:
            raise ValueError(f"lead_in: the {o['op']!r} op cannot be rendered with a lead-in "
                             f"because {why}")
        for key, fix in _LEAD_RELATIVE.items():
            if o["op"] == key[0] and o.get(key[1]) is not None:
                raise ValueError(f"lead_in: {o['op']}({key[1]}=...) is relative to the record, "
                                 f"which a lead-in lengthens -- {fix}")
        if o["op"] == "crosstalk" and o.get("kind", "fext") == "next":
            raise ValueError("lead_in: crosstalk(kind='next') delays the aggressor by td_frac, a "
                             "fraction of the record, which a lead-in lengthens")


def _lead_extent(ops, grid, seed, n, rel=LEAD_IN_REL):
    """MEASURE how many samples of history this chain needs, in samples.

    The chain's LTI ops are applied to a unit impulse at index 0 on a probe grid; the transform of
    that is the chain's combined response on the probe's rfft grid, which is exactly what
    `physics.response_extent` consumes -- so the doubling-until-it-fits logic, the shortest-arc
    support rule and the -100 dB threshold are the audited ones, not a second estimator.

    The impulse goes at index 0 deliberately. Placed anywhere else, a causal response fills the
    buffer from the impulse to the end, the longest quiet run is the part BEFORE it, and
    `response_extent`'s "does it fit in half the probe" test passes at every probe length while
    reporting half of it -- measured 2048 / 4096 / 8192 / 21025 for one lossy channel at probe
    lengths 4096 / 8192 / 16384 / 65536, against 21025 from every probe length with the impulse at 0."""
    import dataclasses
    todo = [o for o in ops if o["op"] in _LEAD_LTI]
    if not todo:
        return 0
    st = Streams(int(seed))

    def make_H(nfft):
        nfft = int(nfft)
        d = np.zeros(nfft); d[0] = 1.0
        g = None if grid is None else dataclasses.replace(grid, n=nfft, segments=None)
        for i, o in enumerate(todo):
            if o["op"] == "probe" and (o.get("noise_rms") or o.get("overload_range") is not None):
                # noise is not a response, and overload recovery is nonlinear/stateful (a unit
                # impulse never approaches the overload range, so it would probe as a no-op
                # rather than the op's real behaviour) -- strip both before measuring the
                # impulse response, the same reason `noise_rms` is already stripped here.
                o = {k: v for k, v in o.items() if k not in ("noise_rms", "overload_range")}
            d = _EXEC[o["op"]](d, o, st, g, i)
        return np.fft.rfft(d)

    cap = int(min(P.PROBE_MAX, max(4 * n, 4 * P.PROBE_N0)))
    return int(P.response_extent(make_H, rel=rel, n0=min(P.PROBE_N0, max(8, n)),
                                 probe_max=cap, warn=False))


def _lead_source_reach(src, n, n_ui):
    """The source's OWN settling reach in samples: `LEAD_SOURCE_TR` times its rise time, with the
    same clamp `physics.resolve_rise_time` applies (warn=False -- the render itself warns about a
    clamped rise time; the sizer must not warn a second time for the same recipe)."""
    tr, _ = P.resolve_rise_time(float(src.get("tr_frac", 0.15)), n / n_ui, warn=False)
    return int(np.ceil(LEAD_SOURCE_TR * float(tr)))


def _lead_quantum(req, p_quantum):
    """Round a requested guard UP to a whole number of UI (a multiple of `p_quantum` samples)."""
    req = int(max(0, req))
    if req == 0:
        return 0, 0
    k = -(-req // int(p_quantum))                       # ceil
    return k * int(p_quantum), k


def _lead_size(spec, ops, grid, seed, n, p_quantum, measured=None, floor=0):
    """Resolve a lead-in spec to ``(samples, k_ui_multiples, measured_or_None)``.

    ``None``/``False``/``0`` -> off. ``True``/``'auto'`` -> the measured chain extent, CAPPED at
    one record: a response still above 1e-5 of its peak a whole record later is one whose level
    THERE bounds what the cap leaves behind, and past that cap each further doubling of the render
    buys less than the last. An explicit integer is honoured as given (the caller has said so),
    rounded up to a whole UI."""
    if spec is None or spec is False or spec == 0:
        return 0, 0, measured
    if spec is True or spec == "auto":
        first = measured is None
        if first:
            measured = _lead_extent(ops, grid, seed, n)
        req = max(measured, int(floor))          # the source's own settling, which no probe sees
        if req > n:
            if not first:
                return _lead_quantum(n, p_quantum) + (measured,)   # already warned for the lead-in
            warnings.warn(
                f"lead_in='auto': the chain's measured impulse-response extent is {req} samples, "
                f"longer than the {n}-sample record; capping the lead-in at one record. The "
                f"response is at or below {LEAD_IN_REL:g} of its peak past that point, which "
                f"bounds the residual; pass lead_in={req} to render it in full.",
                RuntimeWarning, stacklevel=3)
            req = n
    else:
        req = int(spec)
        if req < 0:
            raise ValueError("lead_in must be a non-negative number of samples, True/'auto', or None")
    L, k = _lead_quantum(req, p_quantum)
    return L, k, measured


@dataclass
class LeadPlan:
    """What a lead-in render will do, resolved and reportable BEFORE it is rendered: the guard
    lengths in samples and UI, the measured extent they came from, the extended geometry, and the
    op list and grid the extended render uses. `Signal.lead_plan()` returns it."""
    lead: int
    tail: int
    lead_ui: int
    tail_ui: int
    measured: object
    floor: int
    n: int
    n_ui: int
    n_ext: int
    quantum: int
    ops: list
    grid: object

    @property
    def window(self):
        return slice(self.lead, self.lead + self.n)

    def summary(self):
        return (f"lead {self.lead} + record {self.n} + tail {self.tail} = {self.n_ext} samples "
                f"({self.n_ext / self.n:.3f}x the render), guard measured "
                f"{'n/a' if self.measured is None else self.measured} samples against the "
                f"source's own {self.floor}-sample settling, quantised to {self.quantum} (1 UI)")


# --------------------------------------------------------------- the Signal builder
@dataclass
class Signal:
    seed: int = 0
    grid: Optional[Grid] = None
    ops: list = field(default_factory=list)
    # A RENDERED-AND-DISCARDED LEAD-IN (see the block above `KIND_RANK`). None = off, and off is
    # byte-identical to every recipe ever rendered. True/"auto" = measure the chain's own
    # impulse-response extent. An integer = that many samples. `lead_out` mirrors `lead_in` unless
    # it is given: a zero-phase stage reaches FORWARD as well as back, and the frequency-domain
    # stages that are still circular wrap the record's tail onto its head, so both ends need guard.
    lead_in: object = None
    lead_out: object = None

    def _add(self, op, **params):
        """Append one op, rejecting a parameter this op does not read.

        An op takes its parameters as a dict, so an invented or misspelled name would otherwise
        be accepted and do nothing: the record renders, the digest is stable, and it is not the
        record that was asked for.

        A recipe op is FLAT -- `{"op": ..., <parameters>, "_prov": {...}}` -- so the namespace is
        shared and the rule is: `op` is reserved, anything beginning with an underscore is
        annotation, and everything else is a parameter this op has to read.
        """
        OPKEYS.check(op, params)
        self.ops.append({"op": op, **params}); return self

    def annotate(self, **prov):
        """Attach provenance metadata (e.g. ``stage="channel", node="TP2"``) to the most
        recently added op. Stored under a reserved ``_prov`` key that executors ignore, so it
        round-trips in the recipe but leaves the samples bit-identical. This is the hook the
        higher-level authoring layer uses to tag where/what an op is without touching physics."""
        if not self.ops:
            raise ValueError("annotate() called before any op was added")
        self.ops[-1].setdefault("_prov", {}).update(prov)
        return self

    def symbols(self, symbols, **params):
        """First op: a carrier built from an ARBITRARY per-UI symbol sequence (e.g. a coded /
        scrambled stream from wfmsynth.coding). params: n, tr_frac, causal, jitter.

        SYMBOLS-AS-DATA: the sequence is embedded in the recipe, so the record is reproducible
        with no generator code at all. That is the fallback that always works, and the cost is one
        number per symbol in the JSON. `pattern()` is the same op with a NAME attached."""
        # `list(...)` and not a float cast: every recipe this op has ever produced recorded the
        # caller's own numbers, and coercing them would move the content address of all of them.
        return self._add("symbols", symbols=list(symbols), **params)

    def pattern(self, name, length, tr_frac=None, causal=None, jitter=None, n=None,
                embed=False, **pattern_kw):
        """First op: a carrier built from a NAMED pattern out of the registry
        (`wfmsynth.patterns`) -- the reproducible route for a standard test sequence.

            Signal(...).pattern("prbs13", length=8191, seed=9)
            Signal(...).pattern("lfsr", length=1023, taps=[10, 7])      # any polynomial

        The recipe records the NAME **and** the RESOLVED PARAMETERS (the polynomial, the block, the
        seed, the length), so the JSON is readable by a person and replayable by a consumer who has
        this library but not the registry entry. `embed=True` additionally writes the resolved
        symbols into the recipe, which makes it replayable with no generator code at all -- use it
        when handing a record to someone who will never have the entry.

        `length` is the symbol count. `tr_frac`/`causal`/`jitter`/`n` are the carrier's, and every
        other keyword goes to the PATTERN's generator (`seed=`, `taps=`, `phase=`, ...). The split
        is by name and it is deliberate: the two sets end up in different places in the JSON, one
        describing the sequence and one describing the waveform built from it.

        This is the `symbols` op, not a new one -- which is why it is already a `source` for
        stage-kind homing and already classified for the lead-in. A named pattern and a literal
        symbol list are the same physics (`physics.from_symbols`); they differ only in how the
        recipe says where the symbols came from.
        """
        from . import patterns as PAT
        block = PAT.describe(name, length=length, **pattern_kw)
        params = {}
        if embed:
            params["symbols"] = [float(v) for v in PAT.replay(block)]
        # Only what the caller passed: a restated default would move this recipe's content address
        # the day that default changed.
        for key, val in (("tr_frac", tr_frac), ("causal", causal), ("jitter", jitter), ("n", n)):
            if val is not None:
                params[key] = val
        return self._add("symbols", pattern=block, **params)

    def coded(self, scheme, **params):
        """First op: a carrier built from a LINE-CODED / SCRAMBLED bit stream
        (`wfmsynth.coding`) — the composable form of `bits -> [code] -> symbols`.

            Signal(...).coded("64b66b", prbs=13, n_bits=1 << 15)      # IEEE 802.3 Clause 49
            Signal(...).coded("128b130b", prbs=31, n_bits=1 << 15)    # PCI Express Rev 4.0
            Signal(...).coded("8b10b", bits=[...])

        `scheme` is a key of `coding.SCHEMES`. The payload is `bits=[...]` (a literal list, so the
        record replays with nothing installed) or `prbs=<order>` with `n_bits=<count>` (the compact
        route for the long payloads a scrambler is measured on, with `prbs_seed=` / `prbs_phase=`
        for the starting position). The code's own knobs go straight through: a scrambler's
        `poly=`/`seed=`, 128b/130b's `sync=`, 128b/132b's `header=`, 8b/10b's `rd=`/`lsb_first=`,
        `dc_balanced`'s `block=`, plus `levels=(lo, hi)` for the symbol mapping.
        `tr_frac`/`causal`/`jitter`/`n` are the carrier's, as on `symbols`.

        WHY A SOURCE AND NOT A STAGE: every other op transforms samples and a line code transforms
        bits, so nothing mid-chain can express one. Recording it here is what makes the scheme, the
        polynomial and the seed part of the recipe — and therefore part of the content digest, so
        two records scrambled from different seeds no longer share an address.

        The standard's defaults live in `wfmsynth.coding`, never here: only what the caller passed
        is recorded, so the day a default is corrected every record moves with it."""
        return self._add("coded", scheme=scheme, **params)

    def capture(self, path=None, values=None, fs_hz=None, n=None, resample=False, embed=False):
        """First op: import a REAL capture from disk (or literal samples) as the start of a
        chain, so it can take the same later ops a synthesized carrier takes --
        `lossy`/`probe`/`events`/`scope`, anything downstream of a source.

            Signal(seed=1).capture(path="scope.npy")                       # read at render time
            Signal(seed=1).capture(values=[...], fs_hz=1e9)                # embedded, no file
            Signal(seed=1).capture(path="scope.npy", embed=True)           # read now, then embed

        `path=` (``.npy`` / ``.npz`` with a ``'values'`` array / a two-column ``.csv``) is read
        each time the chain renders, and the recipe records the path plus a sha256 DIGEST OVER
        THE SAMPLE VALUES (not the file's bytes -- the same numbers re-saved through a
        different codec or dtype still match). A file that no longer matches that digest is a
        named error at replay time; a missing file is a named error, not zeros.

        `values=[...]` is the always-replayable route -- literal samples embedded in the
        recipe, the same trade `symbols()` makes for a synthesized source. `embed=True` reads
        a `path=` capture NOW and stores its values inline instead of the path, for a record
        that must not depend on the file still being there later.

        `fs_hz` is provenance (and what `resample=True` resamples FROM); it does not change
        the array length on its own, and by itself it does NOT make a downstream Hz-denominated
        knob (`lossy(loss_at_ghz=...)`, `probe(bw_hz=...)`) mean anything real -- those read
        `grid.fs`. Pass a `grid=Grid(fs=..., n=...)` matching the file's own rate; `fs_hz=` with
        no such `Grid` warns, since nothing downstream then has a real rate to interpret against.
        If the loaded length does not match the Grid's `n` (or an explicit `n=`), that is a
        named error unless `resample=True` is passed -- silently truncating or padding a
        captured record is exactly the failure mode `sample_clock`'s `n_out` warning exists to
        prevent, and a capture gets the same refusal instead."""
        if values is not None:
            params = {"values": [float(v) for v in values]}
            if fs_hz is not None:
                params["fs_hz"] = fs_hz
            if n is not None:
                params["n"] = n
            if resample:
                params["resample"] = True
            return self._add("capture", **params)
        if path is None:
            raise ValueError("capture needs path=... or values=[...]")
        if embed:
            from . import capture as CAP
            y = CAP.load_values(path)
            params = {"values": [float(v) for v in y], "sha256": CAP.digest(y)}
        else:
            from . import capture as CAP
            params = {"path": str(path), "sha256": CAP.digest(CAP.load_values(path))}
        if fs_hz is not None:
            params["fs_hz"] = fs_hz
        if n is not None:
            params["n"] = n
        if resample:
            params["resample"] = True
        return self._add("capture", **params)

    def carrier(self, kind, **params):
        """First op: a carrier ('nrz'|'pam4'). params: n_ui, n, seed, tr_frac, causal,
        pattern, jitter=dict(rj,pj,f_pj,dcd) for source jitter.

        pattern is carrier-specific and defaults to 'legacy' for both: NRZ takes
        'legacy' (== 'prbs7'), 'prbs7'/'prbs9'/'prbs11'/'prbs13'/'prbs15'/'prbs23'/
        'prbs31' and 'clock'; PAM4 takes 'legacy', 'prbs13q', 'prbs31q'. Crossing them
        raises rather than coercing -- see physics.carrier_symbols for why the order
        matters to channel ISI.

        Analog sources are the same op with a different `kind`, not a second builder:
        'sine'/'square'/'triangle'/'sawtooth'/'dc' (params: f_hz or cycles, amp, offset,
        phase_rad, duty/symmetry, tr_frac), plus 'step'/'pulse'/'exp' (a band-limited step, a
        one-shot pulse, a single-pole RC step response -- t_step_s/t_start_s/tau_s in seconds,
        or their _frac equivalents as a fraction of the record), 'chirp' (f0_hz/f1_hz, a REAL
        Hz sweep -- see physics.chirp_sweep; physics.chirp is the legacy normalised one and is
        untouched), 'two_tone' (f1_hz/f2_hz), and 'noise' (rms, df, pink_frac, band_lo_hz/
        band_hi_hz). 'cmos' is the one UNIPOLAR kind: v_lo/v_hi/duty/f_hz/tr_s, in real volts,
        not the +/-1 amp/offset the rest of these share (tr_s is an absolute edge time in
        seconds, unlike every tr_frac above -- a real gate's edge does not shrink with
        frequency). Compose `probe(r_source=..., c_load_f=...)` after it for a loaded-output
        RC pole rather than reaching for a knob here."""
        return self._add("carrier", kind=kind, **params)

    def nonlinearity(self, **params):
        """Nominal (always-on) transmitter imperfections so the unfaulted class isn't
        suspiciously perfect. params: compression, level_noise, rise_fall_ratio, a_base."""
        return self._add("nonlinearity", **params)

    def tx_ffe(self, taps, pre=1, **params):
        """Transmitter FFE pre-emphasis (place after carrier, before the channel). `taps`
        are per-UI weights, `pre` the number of pre-cursor taps. Puts a deliberate
        pre-cursor in the pulse response and de-emphasizes post-cursor ISI."""
        return self._add("tx_ffe", taps=list(taps), pre=pre, **params)

    def optical(self, **params):
        """Map to optical intensity with finite extinction ratio (+ optional RIN / shot noise).
        params: er_db, p_avg, rin_db_per_hz, bw_hz, shot, photons_per_unit."""
        return self._add("optical", **params)

    def dispersion(self, **params):
        """Chromatic dispersion (pulse spreading). params: strength (~ D*L)."""
        return self._add("dispersion", **params)

    def eo(self, **params):
        """Electro-optic modulation: electrical drive -> COMPLEX optical field (start of an E/O/E link).
        params: kind('mzm'|'dml'), vpi, bias, er_db, p_avg, alpha (chirp), adiabatic. Follow with
        `fiber`/`edfa`/`photodetect` — NOT the real-valued electrical ops."""
        return self._add("eo", **params)

    def fiber(self, **params):
        """Single-mode fibre on the optical field: physical chromatic dispersion (β2·L) + attenuation.
        params: length_km, D_ps_nm_km (17 @1550nm), wavelength_nm, atten_db_km (0.2)."""
        return self._add("fiber", **params)

    def optical_mpi(self, **params):
        """Coherent optical multipath interference on the field: a delayed, attenuated copy (double
        reflection) that beats with the signal — the field-domain source of the MPI penalty. params:
        delay_samples (required), reflectivity. Place on the field before `photodetect`."""
        return self._add("optical_mpi", **params)

    def edfa(self, **params):
        """Optical amplifier (EDFA): field gain + ASE noise. params: gain_db, nf_db, p_ase_scale."""
        return self._add("edfa", **params)

    def photodetect(self, **params):
        """Square-law photodetector (O->E) — closes the electrical->optical->electrical loop.
        params: responsivity, shot, photons_per_unit. Output is real photocurrent."""
        return self._add("photodetect", **params)

    def tia(self, **params):
        """Transimpedance amplifier (optical RX front end): photocurrent -> voltage, band-limit, noise.
        params: gain, bw_hz, thermal_rms."""
        return self._add("tia", **params)

    def de_emphasis(self, **params):
        """Tx de-emphasis preset. params: db -- NEGATIVE is de-emphasis, quoted the way every
        spec quotes it (PCIe Gen1/Gen2: "-3.5 dB"); see `physics.de_emphasis_taps`."""
        return self._add("de_emphasis", **params)

    def scope(self, **params):
        """Scope acquisition bandwidth. params: bw_hz, kind, order, causal.

        ``kind`` picks WHERE in the instrument the band limit sits, and the two places are not
        the same operator: ``'bessel'``/``'gaussian'`` are the ANALOG front end, causal and
        single-pass by default, realising the corner they are given and delaying by it;
        ``'brickwall'`` is the DIGITAL selected-bandwidth filter AFTER the converter, which is
        legitimately zero-phase. ``causal=False`` is the opt-out for the pre-fix zero-phase
        analog path, which realised HALF the stated bandwidth -- see
        `instrument.scope_bandwidth`."""
        return self._add("scope", **params)

    def input_bandwidth(self, **params):
        """Analog input-stage bandwidth limit (general-purpose alias of `scope`). params: bw_hz."""
        return self._add("scope", **params)

    analog_front_end = input_bandwidth

    def timebase(self, **params):
        """Timebase / sample-clock jitter (smears the eye horizontally). params: rms_ps."""
        return self._add("timebase", **params)

    def sample_clock_jitter(self, **params):
        """Sample-clock jitter (general-purpose alias of `timebase`). params: rms_ps."""
        return self._add("timebase", **params)

    def acquire(self, profile, tap="stored"):
        """Generalized two-rate acquisition: run the analog front end + digitizer + optional
        record decimation described by an ``AcquisitionProfile`` (or its dict), sampling the
        simulated waveform onto the acquisition grid. ``tap`` selects which stage to emit
        ('simulated' | 'conditioned' | 'digitized' | 'stored'). Carrier-agnostic."""
        prof = profile if isinstance(profile, dict) else _asdict(profile)
        return self._add("acquire", profile=prof, tap=tap)

    def acquire_taps(self, profile):
        """Build the simulated waveform from the current chain (no acquire op), then run the
        acquisition and return ALL taps as a dict: simulated / conditioned / digitized / stored /
        info (realized simulation vs acquisition rates)."""
        from .acquire import AcquisitionProfile, acquire_record
        prof = profile if isinstance(profile, AcquisitionProfile) else AcquisitionProfile(**profile)
        return acquire_record(self.waveform(), self.grid, prof, rng=np.random.default_rng(self.seed))

    def ctle(self, **params):
        """Receiver CTLE (high-frequency-peaking analog EQ), placed after the channel.
        params: fz_ghz, fp1_ghz, fp2_ghz, dc_gain."""
        return self._add("ctle", **params)

    def rx_ffe(self, taps, spacing_ui=0.5, pre=0, **params):
        """Receiver feed-forward equalizer — a trainable FIR on the waveform, post-channel. `spacing_ui`
        = 0.5 is fractionally (T/2) spaced (the usual, sampling-phase-robust form); 1.0 is T-spaced.
        `pre` = number of pre-cursor taps. The RX-side companion to `tx_ffe`."""
        return self._add("rx_ffe", taps=list(taps), spacing_ui=spacing_ui, pre=pre, **params)

    def dfe(self, taps, levels=None, **params):
        """Receiver decision-feedback equalizer: samples the symbol centres, cancels post-cursor ISI
        by subtracting `taps · [past decisions]` before slicing, and returns the equalized waveform
        (reconstructed from the per-symbol equalized values). `taps[j]` = post-cursor weight at lag j+1
        (set to the channel's post-cursors to cancel them); `levels` = constellation (default PAM4; pass
        [-1, 1] for NRZ).

        WHERE it decides is the parameter that matters, because a DFE feeds its own decisions back:
        one wrong decision enters the tap history and corrupts the next, so a decision instant that
        drifts makes it DIVERGE rather than degrade.

          cdr=dict(loop_bw_ui=..., order=2, damping=0.707, phase0=..., spb=...)
              take the decisions on a RECOVERED clock (`cdr.recover_symbol_instants`) — a closed
              timing loop whose instants follow a symbol rate that moves. `loop_bw_hz=` may be used
              instead of `loop_bw_ui=` when the Grid carries `baud`. Required for any record whose
              symbol rate is modulated, and for any record whose samples-per-UI is not a whole number.
          scale=  the amplitude the taps and levels are normalized against (e.g. 1.0 for a unit-
              amplitude signal, an AGC's full scale otherwise). Default: the 99th percentile of the
              sampled magnitudes, which an ISI-corrupted record inflates.
          phase=  a fixed starting sample offset for the fixed-stride path.

        DEFAULT (no `cdr=`) is the historical fixed stride: `int(round(samples_per_ui))` samples
        apart, phase chosen once. That stride is right only while samples-per-UI is a whole number
        AND constant; otherwise it walks off the symbol centres for the whole record.
        `compose.dfe_instants` / `compose.dfe_decisions` expose where it decided and what it decided."""
        p = {"taps": list(taps)}
        if levels is not None:
            p["levels"] = list(levels)
        p.update(params)
        return self._add("dfe", **p)

    def probe(self, **params):
        """The probe the measurement is made THROUGH — placed between the channel and the front
        end. Without it a chain models a perfect tap, which does not exist.

        params: c_load_f (input capacitance, F — an RC pole at 1/(2*pi*R*C) against r_source),
        r_source (ohms), bw_hz + kind/order/causal (the probe's OWN bandwidth — the "bandwidth
        limiter" switch on a real instrument is exactly this at a stated corner, e.g.
        bw_hz=20e6; there is no separate knob for it), noise_rms (its own input-referred noise,
        added at the tip), atten (divider ratio as a gain, 0.1 = 10:1).

        The probe pack: r_term_ohm (the probe's own input resistance -- 1e6 for a classic
        high-Z passive probe, 50.0 for a 50-Ohm-terminated input; default None is the original
        implicitly-infinite termination), compensate (a compensation trimmer's adjustment,
        1.0 = correctly compensated = no change, <1 rounds off, >1 peaks -- see
        `instrument._compensation_and_ground_lead_H`), l_gnd_h (ground-lead inductance forming
        a real second-order ring with c_load_f), coupling ('dc' default or 'ac', the latter
        applying `ac_couple` at ac_fc_hz), overload_range/overload_tau_s (a stated linear range
        past which the probe hard-clips and slowly recovers -- default None is off; this stage
        is nonlinear/stateful, unlike everything else here).

        Defaults are the bare loading pole: no bandwidth limit, no noise, no division, and every
        probe-pack knob above at its identity, so existing recipes render bit-identically."""
        return self._add("probe", **params)

    def modulate(self, **params):
        """AM/ASK/OOK, FM/FSK, or PM -- one op, `kind` and the message's own carrier `kind`
        picking which. `message` is a nested carrier spec, the same shape `open_drain`'s
        `second` and `crosstalk`'s `aggressor` already take (e.g.
        ``message=dict(kind="sine", f_hz=1e3)`` for analog, or a `cmos`/digital carrier for
        ASK/OOK/FSK's two-level message).

        params: kind ('am' default, or 'ask'/'ook' -- the same physics, a two-level message;
        'fm'/'fsk'; 'pm'), message (required), depth/suppressed (AM), fc_hz/dev_hz (FM/FSK,
        both required), fc_hz/dev_rad (PM, fc_hz required).

        OOK with a UNIPOLAR message (a `cmos` carrier, 0..1) needs ``suppressed=True``:
        ``(1 + depth*message)`` never reaches 0 for a message that only goes 0..1, while
        ``depth*message`` does -- full suppression at message=0, full amplitude at message=1.

        AM/ASK/OOK COMBINE with the upstream carrier (build it with `carrier("sine", f_hz=fc,
        ...)` first). FM/FSK/PM GENERATE their own carrier at `fc_hz` and do not read the
        upstream signal -- frequency/phase modulation is not a per-sample transform of an
        already-rendered fixed-frequency carrier the way AM is."""
        return self._add("modulate", **params)

    def pass_fet(self, **params):
        """A gate-controlled series element -- a pass-FET or analog switch. `gate` is a nested
        carrier spec for the gate DRIVE (the same shape `open_drain`'s `second` and
        `crosstalk`'s `aggressor` take, e.g. ``gate=dict(kind="square", f_hz=1e6, duty=0.5)``),
        compared against `vth` (default 0.0) to decide on/off.

        params: gate, vth, rds_on_ohm (the ON channel resistance, a divider against
        r_load_ohm), r_load_ohm, v_rail_hi/v_rail_lo (default None each = that side's body
        diode never conducts), diode_drop, diode_on_ohm.

        OFF is HIGH-Z (reads as 0) unless the signal has pushed past a stated rail, in which
        case the body diode clamps it there -- see `physics.pass_fet`."""
        return self._add("pass_fet", **params)

    def burst(self, **params):
        """Burst/idle on ANY carrier -- periodic on/off gating, applied to whatever samples
        arrive so it composes after an analog carrier, a `cmos` clock, or a digital one alike.

        params: t_on_s, t_off_s (the cadence, in seconds -- needs a Grid), off ('zero', the
        default, or 'hold': sample-and-hold the last on-value through the gap, e.g. a squelched
        idle level rather than a return to 0), edge_frac (the on/off transition as a fraction of
        the shorter of t_on_s/t_off_s, so the gate's own edges do not read as signal edges --
        see `impairments.burst_gate`), phase_s (shifts where the first on/off boundary falls).

        The cadence is periodic and its phase is measured from the record's own start, the same
        as `ssc`/`timing`: a lead-in changes WHERE in the on/off cycle the delivered window
        begins, not what the cycle is."""
        return self._add("burst", **params)

    def drift(self, **params):
        """Slow sub-record drift (thermal/VGA/DC). params: kind('gain'|'amplitude'|'dc'),
        amount, shape('linear'|'sine')."""
        return self._add("drift", **params)

    def dcd(self, **params):
        """Duty-cycle distortion as a DIRECT knob, at the transmitter. ``ps=`` (or
        ``frac_ui=``) is DCD by its standard definition -- mean high pulse width minus mean low
        pulse width -- so ``dcd(ps=4.0)`` makes the high pulses 4 ps wider than the low ones and
        nothing else. Realised to 0.05 % (see `_op_dcd`), sub-sample.

        The old route, ``carrier(..., jitter=dict(dcd=<samples>))``, is off by -2x AND quantised
        to whole samples; the numbers are in the block comment above `_dcd_displacement`.
        params: ps | frac_ui, threshold, plateau_ui."""
        return self._add("dcd", **params)

    def agc(self, **params):
        """Receiver AGC -- normalise to a stated level BEFORE equalising, which is what a
        receiver does and which is what makes an absolute level downstream mean anything.
        params: target, metric('rms'|'peak'|'amplitude'|'p99'), tau_s (a tracking loop instead
        of a block gain), gain_limits=(lo,hi), on_limit('raise'|'clip'). Pair with
        ``dfe(..., scale=<the same target>)``."""
        return self._add("agc", **params)

    def rx_noise(self, **params):
        """The RECEIVER's own input-referred noise, added at the receiver input -- i.e. BEFORE
        the CTLE/DFE, where an equaliser's peaking amplifies it. Distinct from the INSTRUMENT's
        noise floor (`digitize(noise_rms=)`), in mechanism and in position.
        params: rms, or density (units/sqrt(Hz)) + bw_hz (the receiver's noise bandwidth)."""
        return self._add("rx_noise", **params)

    def sample_clock(self, **params):
        """Sample the record on an INDEPENDENT timebase: the instrument's clock is a different
        oscillator from the link's. params: ppm (frequency offset), drift_ppm_per_s, phase0_s,
        n_out, span('fit'|'strict'), band_tol.

        `timebase(rms_ps=)` is the sample clock's random JITTER about its nominal times; this is
        its systematic OFFSET from them, which accumulates. 300 ppm over 3 M UI at 16 samples/UI
        is 14,400 samples of slip; the `digitize(n_out=)` path realises 0."""
        return self._add("sample_clock", **params)

    def timing(self, **params):
        """Compose arbitrary clock timing into the carrier (the timing-modulation enabler).
        params: ssc, pj, wander, rj_ps, phase_noise (see cdr.timing_source)."""
        return self._add("timing", **params)

    def ssc(self, **params):
        """Spread-spectrum clocking — triangular clock-frequency modulation (EMI reduction).
        params: f_ssc (Hz), spread (fraction), profile('down'|'up'|'center')."""
        return self._add("ssc", **params)

    def supply_coupling(self, **params):
        """Power-supply / PDN coupling — correlated AM + PSIJ from a supply rail. params:
        f_ripple_hz, am_depth, psij_ps, supply(optional array)."""
        return self._add("supply_coupling", **params)

    def intra_pair_skew(self, **params):
        """Differential-mode signal after intra-pair (P/N) skew / gain imbalance — closes the
        differential eye. params: skew_ps, gain_imbalance. (For the full P/N pair and
        common-mode, use physics.differential_pair.)"""
        return self._add("intra_pair_skew", **params)

    def open_drain(self, **params):
        """The line voltage of an open-drain bus: a driven fall to a resistive divider, and an RC
        rise through the pull-up. Place it directly after the carrier -- it is what the bus does to
        the driver's intent, before anything else acts on the line.

        params: r_pullup_ohm, c_bus_f (both required), v_dd, r_sink_ohm, v0, center.

        The rise is the edge such a bus specifies, and it follows from R and C rather than from a
        rise-time knob: tr(30-70 %) = 0.8473 * Rp * Cb. `physics.rc_tau_for_rise_time` inverts that
        if you have the specified rise time and want the resistor. Raise Cb far enough and the line
        stops reaching the input-high threshold inside a bit, which is how these buses actually
        fail and which a symmetric slow-edge model cannot produce.

        `second=` adds a SECOND driver sharing this net -- a nested carrier spec, the same shape
        `crosstalk`'s `aggressor` and `hybrid_echo`'s `own` already take (e.g.
        ``second=dict(kind="nrz", n_ui=32, seed=9)``), with its own on-resistance `r_sink_b_ohm`
        (defaults to `r_sink_ohm`). This resolves the wired-AND at the ANALOG level: when both
        devices sink together the target is the divider against their PARALLEL on-resistance, an
        analog mid-level dominated by whichever sink is stronger -- not the logic-level OR
        `bus.open_drain`/`combine_drivers` give a boolean carrier.

        The sink decision thresholds each incoming carrier at the MIDPOINT of its OWN
        excursion, so a `dc` carrier -- any level, any sign -- straddles nothing and never
        sinks; only a carrier that actually toggles drives the bus. If NEITHER device ever
        sinks the op warns, since the line then does exactly nothing for the whole record."""
        return self._add("open_drain", **params)

    def lossy(self, **params):
        """Lossy channel. params: length_in, tand, causal, loss_db+loss_at_ghz (real units), or
        trend=(a,b,c) -- the whole fitted |S21| curve rather than one anchor point, which is what
        a real board's loss SHAPE needs (see `physics.lossy_channel`)."""
        return self._add("lossy", **params)

    def sparam(self, **params):
        """Measured S-parameter channel. params: path=<.sNp file> (+ ports=(2,1), or a
        mixed-mode pairing -- ports="13_24"/"12_34" or ports=((1,3),(2,4)), passed through
        exactly as `sparam.touchstone_channel` documents; plus n_ports, mode, term, check), or
        freqs=[Hz] + s21=[complex]. band/dc/band_tol/linear/guard forward to either form.
        Reproduces resonances/structure the analytic model can't."""
        return self._add("sparam", **params)

    def cascade(self, path, **params):
        """A CASCADED channel: `path` is a list of `{"line": {...}}` / `{"disc": {"gamma": g}}` /
        `{"file": {"path": "x.s2p"}}` sections in physical order from driver to receiver, so each
        reflection is generated where it sits and its echo is attenuated by the segment it
        actually traverses. Replaces `lossy` + `reflect` for a path with structure — do not stack
        it on them. params: node ('load' far-end, 'source' driver-plane reverse wave).
        See `wfmsynth.sparam.cascade_channel`."""
        return self._add("cascade", path=path, **params)

    def reflect(self, **params):
        """Multi-reflection, LUMPED: one round-trip delay, one Gamma, no positional structure —
        every echo pays whatever loss was applied before this op, once. Correct for a line
        mismatched at both ends and nowhere else; for a path with discontinuities at different
        distances use `cascade`. params: td_frac | td_samples | td_ps, gamma_s, gamma_l,
        n_bounce, node."""
        return self._add("reflect", **params)

    def resonant_reflect(self, **params):
        """A resonant discontinuity (frequency-dependent Γ, a stub/open that resonates).
        params: td_ps + f0_ghz (with grid) or td_frac + f0_frac, q, gamma0."""
        return self._add("resonant_reflect", **params)

    def crosstalk(self, **params):
        """Crosstalk. params: coupling, kind('fext'|'next'), td_frac, aggressor=dict(carrier spec)."""
        return self._add("crosstalk", **params)

    def crosstalk_matrix(self, **params):
        """Multiple aggressors from a coupling vector, ASYNCHRONOUS by default. params:
        couplings=[...], kind, baud_offsets, seeds, synchronous.

        It MANUFACTURES its aggressors -- one `nrz` per coupling, at a seed and a baud offset --
        and has NO parameter that takes a waveform. Right when the neighbours are unknown
        traffic; wrong when they are the other lanes of a known link, which is what `multipair`
        is for."""
        return self._add("crosstalk_matrix", **params)

    def hybrid_echo(self, **params):
        """Own transmit leaking into own receive on a bidirectional pair: a scaled, delayed,
        filtered copy of a DIFFERENT stream. NOT a reflection of the received signal, which is
        why `reflect` cannot express it -- `reflect` of silence is silence.
        params: isolation_db, td_ps, f_hp_hz | f_hp_frac, own=dict(carrier spec)."""
        return self._add("hybrid_echo", **params)

    def multipair(self, **params):
        """ONE observed channel of a four-pair link: the chain so far is the wanted signal and
        this sums in the seven other streams a probe on that pair also sees -- own transmit
        through the hybrid, near-end crosstalk from the three local lanes, far-end crosstalk
        from the three remote ones. That is not an approximation of the measurement; it is what
        a differential probe on one pair of a live link measures.

        params: echo_db, echo_td_ps, echo_f_hp_hz, next_db, next_td_ps, fext_db (each a LOSS in
        dB, amplitude convention; omitted or inf switches that mechanism off and the op is then
        the identity), plus pattern/levels/n_ui/tr_frac/local_seed/remote_seed for the aggressor
        lanes. `wfmsynth.quinary.CLAUSE_40_BUDGET` is a worked budget with its provenance."""
        return self._add("multipair", **params)

    def ac_couple(self, **params):
        """AC-coupling. params: fc_frac | fc_hz."""
        return self._add("ac_couple", **params)

    def digitize(self, **params):
        """Scope digitization -- the CONVERTER. params: n_out, snr_db (noise vs signal span) |
        noise_rms (absolute noise floor), bits + full_scale (the converter's real depth and
        range), interleave=dict(m_cores, gain_mm, ...), enob (legacy).

        `bits` and `noise_rms` are the two mechanisms, and they are not the same thing: `bits`
        is the converter's discrete lattice, `noise_rms` the continuous noise+distortion that
        sets its SINAD. A published ENOB is the second, not the first -- turn one into the
        other with `instrument.converter_noise_rms`. `enob` is the old single knob that does
        both jobs at once and gets the record's structure wrong; it still works."""
        return self._add("digitize", **params)

    def store(self, **params):
        """The instrument's EXPORT step: write the record as integer codes. params: bits,
        full_scale (omit to range the vertical to this acquisition), headroom, clip.

        This is not `digitize` twice. `digitize` is the converter, in the middle of the chain;
        every DSP stage after it (the selected-bandwidth filter above all) smears its lattice
        back into a continuum. `store` is the lattice the FILE is on, and it is what gives a
        real capture its noise floor -- see `instrument.store_record`."""
        return self._add("store", **params)

    def events(self, kind, on="symbols", **params):
        """Place a localized mechanism on the current waveform (a needle, not a
        systemic LTI stage). ``kind`` is the mechanism, ``on`` is the targeting
        policy — they are independent. See ``wfmsynth.events``.

        kind: runt | glitch | ring | overshoot | undershoot | nonmonotonic |
              droop | slow_edge
        on:   symbols | edges | pattern | aggressor | intervals | poisson | times

        Placement knobs: indices, count, fraction, every, which ('rising'|
        'falling'|'both'), min_run, motif, intervals, rate_hz, times, samples,
        aggressor (array or carrier dict), n_ui, symbols, transitions_only.
        Mechanism knobs: severity, amp, floor, hold_frac, f0_hz, tau_s, q, zeta,
        cycles, depth, polarity, tr_factor, width.

        Drawn placements use role stream ``events/{i}`` so they are reproducible
        and contrast-re-rollable. Realized times come from ``realize()`` /
        ``event_list()``, not from this spec."""
        if "n_ui" not in params:
            car = next((o for o in self.ops if o["op"] in ("carrier", "symbols", "coded")), None)
            if car is not None:
                # the source's symbol count, whether it is a carrier's `n_ui`, a literal symbol
                # list, or a named pattern's recorded `length`
                params = dict(params, n_ui=_source_n_ui(car))
        return self._add("events", kind=kind, on=on, **params)

    def with_lead_in(self, lead_in=True, lead_out=None):
        """Render a LEAD-IN and discard it, so the record the caller receives never contains the
        chain's turn-on. Chainable; returns self.

        A linear convolution (U-16) means the samples before index 0 are a quiescent line, so a
        record that begins mid-pattern begins with a step no running link has, and every stage with
        memory answers that step. This renders `lead_in` extra samples of the same pattern before
        the record and `lead_out` after it, runs the whole chain on the longer record, and hands
        back only the middle -- so the turn-on, a zero-phase stage's missing pre-cursor and any
        still-circular stage's wrap all land in samples that are thrown away.

          lead_in=True / "auto"  MEASURE it: the larger of `physics.response_extent` on the
                                 chain's own combined impulse response (-100 dB below its peak) and
                                 the SOURCE's own settling, `LEAD_SOURCE_TR * tr`, which no impulse
                                 probe can see because a carrier is not a filter of the record.
                                 Capped at one record, with a warning naming the measurement when
                                 the cap bites.
          lead_in=<int>          that many samples, honoured as given.
          lead_in=None / 0       off, and off is bit-identical to no lead-in at all.
          lead_out=None          the same length as the lead-in. 0 for none.

        Either guard is rounded UP to a whole number of UI, because samples-per-UI is ``n/n_ui``
        and a lead-in that changed it would re-time the record instead of adding history to its
        head. `lead_plan()` reports the resolved lengths and the measurement they came from without
        rendering. Ops whose knobs are a fraction of the record, or that change the record's
        length, or that place something at an absolute position in it, are REFUSED with the reason
        rather than silently moved -- see `_LEAD_REJECT` / `_LEAD_RELATIVE`.

        THE RECORD IS A DIFFERENT RECORD, and that is the point: its head now has history. With a
        `carrier` the pattern is extended FORWARD, so the window holds symbols[lead_ui:] of the
        same PRBS -- genuine history, and identical symbols when `lead_ui` is a multiple of the
        pattern period. With `symbols` (an explicit repeating stream) the lead-in is that stream's
        own cyclic prefix, so the window holds exactly the stream that was passed."""
        self.lead_in = lead_in
        self.lead_out = lead_out
        return self

    def lead_plan(self):
        """The resolved `LeadPlan` for this Signal's lead-in, or None when it is off. Reports the
        guard lengths, the measured extent behind them and the extended geometry WITHOUT rendering,
        so a caller can see (and a dataset can record) what a render is about to pay for."""
        if not self.lead_in and not self.lead_out:
            return None
        n, n_ui, quantum, q_ui = _lead_source_geometry(self.ops, self.grid)
        if self.grid is not None and getattr(self.grid, "segments", None):
            raise ValueError("lead_in: a segmented grid's anchors are resolved from the record's "
                             "start, which a lead-in moves; not supported")
        _lead_check_ops(self.ops)
        # The floor is what no probe can see: the source's own settling, and any op whose memory is
        # known in closed form rather than measurable.
        floor = max(_lead_source_reach(self.ops[0], n, n_ui),
                    _lead_analytic_reach(self.ops, self.grid))
        L, kl, meas = _lead_size(self.lead_in, self.ops, self.grid, self.seed, n, quantum,
                                 floor=floor)
        spec_out = self.lead_in if self.lead_out is None else self.lead_out
        T, kt, meas = _lead_size(spec_out, self.ops, self.grid, self.seed, n, quantum,
                                 measured=meas, floor=floor)
        if L == 0 and T == 0:
            return None
        lead_ui, tail_ui = kl * q_ui, kt * q_ui
        n_ext = L + n + T
        src = dict(self.ops[0])
        src["n"] = n_ext
        if src["op"] == "carrier":
            # forward extension: the same PRBS, `lead_ui` symbols earlier in the record
            src["n_ui"] = n_ui + lead_ui + tail_ui
        else:
            # an explicit stream repeats, so its own tail is its history: a cyclic prefix/suffix.
            # A NAMED PATTERN is resolved to its symbols first and then gets exactly this
            # treatment -- the pattern route inherits the assumption rather than inventing a
            # second one. The assumption is that the sequence repeats inside the record; when it
            # does not (a record shorter than the period), the cyclic prefix is not the true
            # history. For a shift-register pattern the true history is available instead: pass
            # `phase=` to start the generator earlier.
            sym = _source_symbols(src)
            j = np.arange(-lead_ui, n_ui + tail_ui)
            src["symbols"] = list(sym[j % n_ui])
        ops = [src] + [dict(o) for o in self.ops[1:]]
        import dataclasses
        grid = None if self.grid is None else dataclasses.replace(self.grid, n=n_ext)
        return LeadPlan(lead=L, tail=T, lead_ui=lead_ui, tail_ui=tail_ui, measured=meas,
                        floor=floor, n=n, n_ui=n_ui, n_ext=n_ext, quantum=quantum, ops=ops,
                        grid=grid)

    def rendered_lead(self, streams=None):
        """``(x_extended, plan)`` — the WHOLE lead-in render, guard included, and the plan that
        sized it; ``(waveform(), None)`` when no lead-in is set.

        This is the audit hook: `waveform()` hands back ``x_extended[plan.window]`` and throws the
        rest away, and the rest is where the turn-on, a zero-phase stage's missing pre-cursor and
        any still-circular stage's wrap live. Look at it rather than taking the mechanism's word
        for it -- and range nothing to it."""
        plan = self.lead_plan()
        if plan is None:
            return self._run(streams), None
        return self._run(streams, _extended=True, _plan=plan), plan

    def _run(self, streams=None, collect_events=False, _extended=False, _plan=None):
        from .events import EventList
        st = streams if streams is not None else Streams(self.seed)
        plan = _plan if _plan is not None else self.lead_plan()
        ops = self.ops if plan is None else plan.ops
        grid = self.grid if plan is None else plan.grid
        win = None if plan is None else plan.window
        x = None
        sink = [] if collect_events else None
        for i, op in enumerate(ops):
            if collect_events and op["op"] == "events":
                x = _op_events(x, op, st, grid, i, sink=sink)
            elif win is not None and op["op"] in _LEAD_WINDOW_RANGED:
                x = _EXEC[op["op"]](x, op, st, grid, i, win=win)   # range the vertical to the window
            else:
                x = _EXEC[op["op"]](x, op, st, grid, i)
        if x is None:
            raise ValueError("empty Signal: add a carrier first")
        if win is not None:
            if len(x) != plan.n_ext:
                raise ValueError(f"lead_in: the chain returned {len(x)} samples for a "
                                 f"{plan.n_ext}-sample render, so the window cannot be sliced out")
            if not _extended:
                x = x[win].copy()
        if collect_events:
            return x, EventList(sink, n=len(x), grid=self.grid)
        return x

    def waveform(self, streams=None):
        """Execute the recipe -> samples. Deterministic given (seed, ops, grid). Pass a
        `Streams` (e.g. from `Streams(seed).reroll(...)`) to re-roll selected factors
        while holding all others bit-identical — `contrast()` wraps the common case."""
        return self._run(streams)

    def realize(self, streams=None):
        """One-pass ``(waveform, EventList)`` — use this when an external segmenter
        needs both the samples and the realized event times."""
        return self._run(streams, collect_events=True)

    def event_list(self, streams=None):
        """Realized events (sample / UI / seconds) for every ``.events()`` op."""
        return self.realize(streams)[1]

    def event_mask(self, streams=None):
        """Per-sample support of every realized event (1 = modified)."""
        return self.realize(streams)[1].mask()

    def roles(self):
        """The re-rollable random factors in this signal, as role names — the valid
        arguments to `contrast()`. Deterministic channel/reflection ops draw no
        randomness and so contribute none."""
        out = []
        for i, op in enumerate(self.ops):
            if op["op"] == "carrier" and op.get("jitter"):
                out.append(f"jitter/{i}")
            elif op["op"] == "crosstalk":
                out.append(f"jitter/xtalk{i}")
            elif op["op"] == "digitize":
                if "snr_db" in op or "noise_rms" in op:
                    out.append(f"noise/{i}")
                if op.get("interleave"):
                    out.append(f"interleave/{i}")
            elif op["op"] == "events":
                out.append(f"events/{i}")
            elif op["op"] == "probe" and op.get("noise_rms"):
                out.append(f"probe/{i}")
        return out

    def contrast(self, *factors, seed=None):
        """A sibling waveform with ONLY the named factors re-rolled and every other
        factor bit-identical — a valid contrastive pair / clean ablation. `factors` are
        role names from `roles()`; `seed` makes the re-roll reproducible."""
        return self.waveform(streams=Streams(self.seed).reroll(*factors, seed=seed))

    def ground_truth(self, levels=None):
        """Ground-truth labels MEASURED from this signal's waveform (never read off the
        knobs): eye height under both definitions, the realized sampling phase, and the
        realized integer-symbol alignment — reconstructing the transmitted stream from the
        recipe so per-symbol statistics compare the right pairs across a channel's group
        delay. `levels` defaults from the carrier: 2 for NRZ, N for pam<N>."""
        from .measure import ground_truth as _gt
        car = next((o for o in self.ops if o["op"] == "carrier"), None)
        tx = None
        if car is not None:
            tx = P.carrier_symbols(car["kind"], car.get("n_ui", 32),
                                   car.get("seed", 1), car.get("pattern", "legacy"))
            if levels is None:
                mk = re.fullmatch(r"pam(\d+)", str(car["kind"]))
                levels = int(mk.group(1)) if mk else 2
        return _gt(self.waveform(), self.grid, tx=tx, levels=levels or 4)

    def recipe(self):
        """A JSON-serializable dict: engine version, seed, grid, and the ordered ops
        with their exact knob values. This IS the ground truth for the waveform."""
        from wfmsynth import __version__
        r = {"wfmsynth_version": __version__, "seed": int(self.seed),
             "ops": [dict(o) for o in self.ops]}
        # only when a lead-in is actually set, so an existing recipe's JSON -- and therefore its
        # sha256 content address -- is byte-identical to what it was before the lead-in existed
        if self.lead_in:
            r["lead_in"] = self.lead_in
        if self.lead_out is not None:
            r["lead_out"] = self.lead_out
        if self.grid is not None:
            g = self.grid
            r["grid"] = {"fs": g.fs, "baud": g.baud, "n": g.n, "v_full": g.v_full}
        return r

    def stage_kinds(self):
        """The stage kind of each op, in order (source/shape/supply/channel/instrument)."""
        return [op_kind(o["op"]) for o in self.ops]

    def canonical(self):
        """A new Signal with the ops homed into canonical stage-kind order (a stable sort). The
        default `waveform()` runs in insertion order and is unchanged; this is the opt-in Fabric
        view where cross-kind authoring order commutes by construction."""
        s = Signal(seed=self.seed, grid=self.grid, lead_in=self.lead_in, lead_out=self.lead_out)
        s.ops = canonicalize(self.ops)
        return s

    def to_json(self):
        """Canonical, key-sorted JSON string of the recipe — a stable serialization for
        hashing, diffing and content-addressing. Two Signals with the same ops/grid/seed
        (regardless of construction order) produce byte-identical JSON."""
        import json
        return json.dumps(self.recipe(), sort_keys=True)

    def sha256(self):
        """Stable content hash of the recipe: the identity of this synthesis program. Because
        `(recipe, seed) -> waveform` is deterministic, equal hashes guarantee equal samples."""
        import hashlib
        return hashlib.sha256(self.to_json().encode("utf-8")).hexdigest()

    @classmethod
    def from_recipe(cls, r):
        """Reconstruct a Signal from a recipe; `.waveform()` reproduces bit-for-bit."""
        grid = Grid(**r["grid"]) if r.get("grid") else None
        s = cls(seed=r["seed"], grid=grid, lead_in=r.get("lead_in"), lead_out=r.get("lead_out"))
        ops = [dict(o) for o in r["ops"]]
        for o in ops:                       # the same gate the builders go through: a recipe
            OPKEYS.check(o.get("op"), o)    # carrying a dead parameter says what it is not
        s.ops = ops
        return s


def rederive_anchor(anchor, grid):
    """Resolve a symbolic time-anchor to an absolute sample index on a (single-segment) grid —
    the one function that lowering and rendering both call to turn a fault's *when* into a
    sample. `anchor` is a dict with exactly one of: ``sample`` (index), ``t`` (seconds),
    ``frac`` (fraction of the record), or ``ui`` (unit intervals; needs baud). The
    piecewise/aperiodic generalization (named segments, landmarks) extends this later without
    changing the single-segment result."""
    if grid is None:
        raise ValueError("rederive_anchor needs a grid")
    if "sample" in anchor:
        return int(round(anchor["sample"]))
    if "t" in anchor:
        return int(round(grid.to_samples(anchor["t"])))
    if "frac" in anchor:
        return int(round(anchor["frac"] * grid.n))
    if "ui" in anchor:
        spu = grid.samples_per_ui
        if spu is None:
            raise ValueError("a 'ui' anchor needs baud set on the grid")
        return int(round(anchor["ui"] * spu))
    raise ValueError(f"unknown anchor spec {anchor!r} (use sample/t/frac/ui)")


def dataset(build, n, seed=0):
    """Ground-truth dataset builder. `build(rng)` returns a Signal (sample its knobs from
    `rng` however you like — the SAMPLED values are baked into the Signal's ops, hence
    recorded). Returns (X, recipes): X is (n, L) stacked waveforms, recipes is a list of n
    per-sample recipes. Each waveform is exactly reproducible from its recipe."""
    rng = np.random.default_rng(seed)
    sigs = [build(rng) for _ in range(n)]
    waves = [s.waveform() for s in sigs]
    L = len(waves[0])
    X = np.empty((n, L), np.float32)
    for i, w in enumerate(waves):
        X[i] = w[:L] if len(w) >= L else np.pad(w, (0, L - len(w)))
    return X, [s.recipe() for s in sigs]
