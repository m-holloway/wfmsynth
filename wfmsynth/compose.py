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

Ops compose over the validated primitives (`physics`, `instrument`), so every knob is a
real, documented parameter — nothing is hidden or randomized-but-unrecorded. Randomness
(jitter, ADC noise) is driven by the Signal's single seed, threaded through the ops in
order, which is what makes the round-trip exact.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.signal import resample_poly

from . import physics as P
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
        return P.nrz(**common)
    raise ValueError(f"unknown carrier kind {p['kind']!r} (use 'nrz' or 'pam4')")


def _op_carrier(x, p, streams, grid, idx):
    return _carrier(p, streams, grid, idx)


def _op_symbols(x, p, streams, grid, idx):
    j = p.get("jitter")
    jitter = P.Jitter(**j) if j else None
    return P.from_symbols(np.asarray(p["symbols"], float), n=_grid_n(grid, p),
                          tr_frac=p.get("tr_frac", 0.15), causal=p.get("causal", False),
                          jitter=jitter, rng=streams.role(f"jitter/{idx}"))


def _op_lossy(x, p, streams, grid, idx):
    kw = {k: p[k] for k in ("length_in", "tand", "eps_r", "skin_k", "causal",
                            "loss_db", "loss_at_ghz") if k in p}
    return P.lossy_channel(x, grid=grid, **kw)


def _op_reflect(x, p, streams, grid, idx):
    kw = {k: p[k] for k in ("td_frac", "td_samples", "td_ps", "gamma_s", "gamma_l",
                            "n_bounce") if k in p}
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


def _op_nonlinearity(x, p, streams, grid, idx):
    kw = {k: p[k] for k in ("compression", "level_noise", "rise_fall_ratio", "a_base") if k in p}
    return P.nominal_nonlinearity(x, rng=streams.role(f"nl_noise/{idx}"), **kw)


def _op_resonant_reflect(x, p, streams, grid, idx):
    kw = {k: p[k] for k in ("td_ps", "td_frac", "f0_ghz", "f0_frac", "q", "gamma0") if k in p}
    return P.resonant_reflection(x, grid=grid, **kw)


def _op_supply_coupling(x, p, streams, grid, idx):
    kw = {k: p[k] for k in ("f_ripple_hz", "am_depth", "psij_ps", "supply") if k in p}
    return P.supply_coupling(x, grid, **kw)


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


def _op_de_emphasis(x, p, streams, grid, idx):
    spb = grid.samples_per_ui if grid is not None else p["spb"]
    return P.tx_ffe(x, P.de_emphasis_taps(p["db"]), spb, pre=0)


def _op_scope(x, p, streams, grid, idx):
    kw = {k: p[k] for k in ("kind", "order") if k in p}
    return INST.scope_bandwidth(x, grid, p["bw_hz"], **kw)


def _op_timebase(x, p, streams, grid, idx):
    return INST.timebase_jitter(x, grid, rms_ps=p.get("rms_ps", 0.5), rng=streams.role(f"timebase/{idx}"))


def _op_ctle(x, p, streams, grid, idx):
    from . import rx as RX
    return RX.ctle(x, grid, p["fz_ghz"], p["fp1_ghz"], p["fp2_ghz"], dc_gain=p.get("dc_gain", 1.0))


def _op_sparam(x, p, streams, grid, idx):
    from . import sparam as SP
    if "path" in p:
        return SP.touchstone_channel(x, p["path"], grid=grid, ports=tuple(p.get("ports", (2, 1))))
    return SP.sparam_channel(x, np.asarray(p["freqs"]), np.asarray(p["s21"], complex), grid=grid)


def _op_digitize(x, p, streams, grid, idx):
    n_out = p.get("n_out")
    if n_out and n_out != len(x):
        x = resample_poly(x, n_out, len(x))
    span = float(np.ptp(x)) + 1e-9
    if "noise_rms" in p:                      # absolute noise floor (signal-independent)
        x = x + streams.role(f"noise/{idx}").normal(0.0, p["noise_rms"], len(x))
    elif "snr_db" in p:                       # noise relative to the signal span
        x = x + streams.role(f"noise/{idx}").normal(0.0, span / 10 ** (p["snr_db"] / 20), len(x))
    if p.get("interleave"):
        x = INST.interleave_adc(x, rng=streams.role(f"interleave/{idx}"), **p["interleave"])
    if "enob" in p:
        lsb = span / 2 ** p["enob"]
        x = lsb * np.round(x / lsb)
    return x


_EXEC = {"carrier": _op_carrier, "symbols": _op_symbols, "lossy": _op_lossy, "reflect": _op_reflect,
         "crosstalk": _op_crosstalk, "ac_couple": _op_ac_couple, "digitize": _op_digitize,
         "tx_ffe": _op_tx_ffe, "sparam": _op_sparam,
         "resonant_reflect": _op_resonant_reflect, "nonlinearity": _op_nonlinearity,
         "crosstalk_matrix": _op_crosstalk_matrix, "ctle": _op_ctle, "ssc": _op_ssc,
         "intra_pair_skew": _op_intra_pair_skew, "supply_coupling": _op_supply_coupling,
         "timing": _op_timing, "optical": _op_optical, "dispersion": _op_dispersion,
         "drift": _op_drift, "scope": _op_scope, "timebase": _op_timebase,
         "de_emphasis": _op_de_emphasis}


# --------------------------------------------------------------- the Signal builder
@dataclass
class Signal:
    seed: int = 0
    grid: Optional[Grid] = None
    ops: list = field(default_factory=list)

    def _add(self, op, **params):
        self.ops.append({"op": op, **params}); return self

    def symbols(self, symbols, **params):
        """First op: a carrier built from an ARBITRARY per-UI symbol sequence (e.g. a coded /
        scrambled stream from wfmsynth.coding). params: n, tr_frac, causal, jitter."""
        return self._add("symbols", symbols=list(symbols), **params)

    def carrier(self, kind, **params):
        """First op: a carrier ('nrz'|'pam4'). params: n_ui, n, seed, tr_frac, causal,
        pattern (pam4), jitter=dict(rj,pj,f_pj,dcd) for source jitter."""
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

    def de_emphasis(self, **params):
        """Tx de-emphasis preset (dB). params: db."""
        return self._add("de_emphasis", **params)

    def scope(self, **params):
        """Scope acquisition bandwidth (band-limited front end). params: bw_hz, kind, order."""
        return self._add("scope", **params)

    def timebase(self, **params):
        """Timebase / sample-clock jitter (smears the eye horizontally). params: rms_ps."""
        return self._add("timebase", **params)

    def ctle(self, **params):
        """Receiver CTLE (high-frequency-peaking analog EQ), placed after the channel.
        params: fz_ghz, fp1_ghz, fp2_ghz, dc_gain."""
        return self._add("ctle", **params)

    def drift(self, **params):
        """Slow sub-record drift (thermal/VGA/DC). params: kind('gain'|'amplitude'|'dc'),
        amount, shape('linear'|'sine')."""
        return self._add("drift", **params)

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

    def lossy(self, **params):
        """Lossy channel. params: length_in, tand, causal, or loss_db+loss_at_ghz (real units)."""
        return self._add("lossy", **params)

    def sparam(self, **params):
        """Measured S-parameter channel. params: path=<.sNp file> (+ ports=(2,1)), or
        freqs=[Hz] + s21=[complex]. Reproduces resonances/structure the analytic model can't."""
        return self._add("sparam", **params)

    def reflect(self, **params):
        """Multi-reflection. params: td_frac | td_samples | td_ps, gamma_s, gamma_l, n_bounce."""
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
        couplings=[...], kind, baud_offsets, seeds, synchronous."""
        return self._add("crosstalk_matrix", **params)

    def ac_couple(self, **params):
        """AC-coupling. params: fc_frac | fc_hz."""
        return self._add("ac_couple", **params)

    def digitize(self, **params):
        """Scope digitization. params: n_out, snr_db (noise vs signal span) | noise_rms
        (absolute noise floor), enob, interleave=dict(m_cores, gain_mm, ...)."""
        return self._add("digitize", **params)

    def waveform(self, streams=None):
        """Execute the recipe -> samples. Deterministic given (seed, ops, grid). Pass a
        `Streams` (e.g. from `Streams(seed).reroll(...)`) to re-roll selected factors
        while holding all others bit-identical — `contrast()` wraps the common case."""
        st = streams if streams is not None else Streams(self.seed)
        x = None
        for i, op in enumerate(self.ops):
            x = _EXEC[op["op"]](x, op, st, self.grid, i)
        if x is None:
            raise ValueError("empty Signal: add a carrier first")
        return x

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
        delay. `levels` defaults from the carrier (2 for NRZ, 4 for PAM4)."""
        from .measure import ground_truth as _gt
        car = next((o for o in self.ops if o["op"] == "carrier"), None)
        tx = None
        if car is not None:
            tx = P.carrier_symbols(car["kind"], car.get("n_ui", 32),
                                   car.get("seed", 1), car.get("pattern", "legacy"))
            if levels is None:
                levels = 4 if car["kind"] == "pam4" else 2
        return _gt(self.waveform(), self.grid, tx=tx, levels=levels or 4)

    def recipe(self):
        """A JSON-serializable dict: engine version, seed, grid, and the ordered ops
        with their exact knob values. This IS the ground truth for the waveform."""
        from wfmsynth import __version__
        r = {"wfmsynth_version": __version__, "seed": int(self.seed),
             "ops": [dict(o) for o in self.ops]}
        if self.grid is not None:
            g = self.grid
            r["grid"] = {"fs": g.fs, "baud": g.baud, "n": g.n, "v_full": g.v_full}
        return r

    @classmethod
    def from_recipe(cls, r):
        """Reconstruct a Signal from a recipe; `.waveform()` reproduces bit-for-bit."""
        grid = Grid(**r["grid"]) if r.get("grid") else None
        s = cls(seed=r["seed"], grid=grid)
        s.ops = [dict(o) for o in r["ops"]]
        return s


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
