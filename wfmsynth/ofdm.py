"""wfmsynth.ofdm -- a multicarrier source: every symbol is an inverse FFT of a row of the
resource grid, with a cyclic prefix in front of it; and the receiver that undoes it.

    g = Grid(fs=20e6, n=1 << 16)                           # fs = n_fft * scs_hz, or a whole multiple
    tx = Signal(seed=1, grid=g).ofdm(n_fft=64, scs_hz=312.5e3, cp=16, active=52,
                                     constellation="qam16", prbs=15)
    x = tx.waveform()                                      # complex128
    sent, t_s = tx.transmitted()                           # the resource grid, [symbol, subcarrier]
    got = ofdm.demodulate(x, g, tx.ops[0])                 # the same parameters, the same shape
    rf.evm(x, g, sent, ofdm=tx.ops[0])                     # per resource element, against truth

WHY THIS IS AN OP. One `iq()` chain per subcarrier, each `freq_offset` to its place and summed
with `combine`, would need a rectangular pulse `iq` does not have, a cyclic prefix nothing
expresses, and one chain per subcarrier. The inverse FFT per symbol is one mechanism.

THE CONVENTIONS, none of which is guessable:

  * A SUBCARRIER is a signed whole number k: its frequency is ``k * scs_hz`` from the centre
    frequency, k = 0 being the centre itself (the carrier, at RF). ``-n_fft/2 <= k < n_fft/2``.
  * `active=` is which subcarriers carry anything. A COUNT c means the c nearest the centre
    with the centre left out, ``-c/2 .. -1, 1 .. c/2``; a LIST is those k exactly (list 0 to
    use the centre). Every other subcarrier is zero on every symbol.
  * One symbol is ``x(t) = amp / sqrt(n_active) * sum_k X[l, k] * exp(j*2*pi*k*scs_hz*t)``,
    t counted from the start of its useful part, which lasts ``1 / scs_hz``. In front of it
    sits the CYCLIC PREFIX: the last `cp` samples of that useful part, repeated. Symbols
    follow each other with no gap. With unit-power values on every active subcarrier the
    useful part's mean ``|x|**2`` is exactly ``amp**2`` (Parseval).
  * `cp=` and `window=` are COUNTS OF SAMPLES AT THE FFT RATE, ``n_fft * scs_hz`` -- the
    unit every published parameter set states them in -- whatever the grid's own rate is.
    `cp=[a, b, ...]` is a different prefix per symbol, cycled.
  * THE GRID carries the FFT rate or a whole multiple L of it. On a multiple the same finite
    sum is evaluated on the finer grid -- an inverse FFT of length ``L * n_fft`` with the
    extra bins zero -- which is the same signal exactly, not an interpolation of it. Any other
    rate is REFUSED with the rate needed; nothing is resampled.
  * DATA fills the elements that are not pilots, symbol by symbol and within a symbol in
    ascending k, from bits mapped by `iq.map_bits` (`constellation=`, one, named as on `iq`).
    The bits come from `prbs=<order>` (`prbs_seed=`, `prbs_phase=`) or `bits=[...]`, as on `iq`.
  * A PILOT is a known value. `pilots=[k, ...]` puts +1 on those subcarriers of every symbol;
    `pilots=dict(subcarriers=[...], values=[...], every=m, first=f)` states the values and
    which symbols carry them (`first`, then every m-th); a list of such dicts is several
    sets (a staggered pattern). An element that is a pilot on one symbol is data on the rest.
  * `window=w` is the raised-cosine EDGE of the symbol's own construction: the first w
    samples of the prefix ramp up, and w samples of cyclic SUFFIX ramp down on top of the
    next symbol's rising prefix, the two ramps summing to one. The symbol spacing and the
    useful part are untouched, so the receiver's FFT still returns the grid exactly; w of the
    prefix is spent on it. Any other filtering is a following `filter`.

TWO THINGS THAT ARE TRUE OF THE SIGNAL AND SURPRISE PEOPLE:

  * `prbs_seed` defaults to 1, as on `iq`, and an LFSR started in state 1 emits almost
    nothing but zeros for its first few hundred bits. Every subcarrier of the first symbols
    then carries the same point, and the inverse FFT of a constant is an impulse: 15.6 dB
    peak-to-average on 52 QPSK subcarriers with prbs=31, against 9.7 dB from a mixed state.
    State `prbs_seed=` (`physics.random_phase(order, rng)` draws one).
  * A rectangular symbol's skirts fall only as 1/f**2, so an unwindowed record has power far
    from its active subcarriers (-33 dB of it 4 to 8 spacings out, on 52 of 64). `upconvert`
    refuses the part that would fold; `window=` or a following `filter` is the remedy, as it
    is in a transmitter.

NOISE is `awgn(snr_db=s, bw_hz=plan(...)["noise_bw_hz"])`: in that bandwidth, ``n_active *
scs_hz``, the SNR is the one each demodulated element sees. An OFDM grid states no `baud`,
so `es_n0_db=` is refused there.

WHAT IS NOT HERE, because it composes. A frame, a burst and a gap are `sequence` and `gate`
around this op (`n_symbols=` makes a burst of that many). A preamble is another `ofdm` chain
whose elements are all pilots, sequenced in front. Different constellations on different
subcarriers are two `ofdm` chains with disjoint `active=`, added with `combine` -- the
subcarriers are orthogonal, so each chain's own `demodulate` still returns its own grid.

NAMED PARAMETER SETS. Which numbers a radio standard calls by which name is a fact with a
document and a revision behind it, so it is the caller's to state, exactly as a test pattern
is (`wfmsynth.patterns`):

    ofdm.register("my_20mhz", n_fft=64, scs_hz=312.5e3, cp=16, active=52,
                  pilots=[-21, -7, 7, 21], source="<document, clause, table>")
    Signal(grid=g).ofdm("my_20mhz", constellation="qpsk", prbs=15)

The recipe records the NAME AND THE RESOLVED NUMBERS, so it replays with no registry and can
still be read. A name says where the numbers came from; nothing here claims conformance to
anything -- the claim is "OFDM with these parameters".

THE RECEIVER (`demodulate`) is the transmitter run backwards: an FFT over each useful part,
opened where the transmitter's began (`delay_s` later), the active bins scaled back. It fits
nothing. `evm` (reached as `rf.evm(..., ofdm=params)`) compares what it returns with the
truth per resource element, and takes out only what `correct=` names.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar

from . import iq as IQ
from .events import Event

# the numbers a named set may state (`register`), and everything the op reads
SET_KEYS = ("n_fft", "scs_hz", "cp", "active", "pilots", "window")
KEYS = ("name", "source") + SET_KEYS + ("constellation", "n_symbols", "amp", "prbs", "prbs_seed",
                                        "prbs_phase", "bits")
_PILOT_KEYS = ("subcarriers", "values", "every", "first")
CORRECTIONS = ("timing", "gain", "common_phase", "equalizer")
RATE_TOL = 1e-9           # how far fs may be from a whole multiple of the FFT rate, as a fraction
TIMING_SEARCH = 8         # the timing search's coarse step is 1 / (this * n_fft * scs_hz)

# what `rf.evm`'s single-carrier corrections are called here, or why they do not exist
_NOT_HERE = {
    "phase": "'common_phase' is the rotation of each OFDM symbol, which a constant one is a "
             "case of",
    "frequency": "a frequency error moves every subcarrier off its bin, which is leakage "
                 "between subcarriers and not a rotation; take it out of the record first "
                 "(x * exp(-2j*pi*f*t)). What is left turns each symbol by a constant, which "
                 "'common_phase' removes and returns per symbol",
    "offset": "a constant added to the record lands on subcarrier 0 alone; leave 0 out of "
              "active= (a count already does)",
}

REGISTRY = {}


# ------------------------------------------------------------------ named parameter sets
def register(name, source=None, **numbers):
    """Name a parameter set: any of `n_fft`, `scs_hz`, `cp`, `active`, `pilots`, `window`.
    `source` says where the numbers come from (a document, a clause) and travels into the
    recipe beside the name. `Signal.ofdm(name, ...)` then states the rest."""
    extra = sorted(set(numbers) - set(SET_KEYS))
    if extra or not numbers:
        raise ValueError(f"ofdm.register: a named set states some of {list(SET_KEYS)}"
                         + (f"; it does not take {extra} (constellation, bits and length belong to "
                            f"the call)" if extra else ", and this one states none"))
    if name in REGISTRY:
        raise ValueError(f"ofdm.register: {name!r} is already registered; unregister it first "
                         f"(a name that changed meaning would label two different records)")
    entry = jsonable(dict(numbers))
    if source is not None:
        entry["source"] = str(source)
    _numerology(entry, partial=True)                  # refused here, where it is written
    REGISTRY[str(name)] = entry


def unregister(name):
    REGISTRY.pop(name, None)


def named(name, params):
    """The op parameters of `Signal.ofdm(name, **params)`: the set's numbers, its name and
    source, and what the call adds. A number the set states cannot be restated in the call."""
    if name not in REGISTRY:
        raise ValueError(f"ofdm: no parameter set named {name!r} is registered "
                         f"(registered: {sorted(REGISTRY) or 'none'}). Names are the caller's: "
                         f"ofdm.register({name!r}, n_fft=..., scs_hz=..., cp=..., active=..., "
                         f"source='<where the numbers come from>'), or pass the numbers.")
    entry = REGISTRY[name]
    both = sorted(k for k in params if k in entry)
    if both:
        raise ValueError(f"ofdm: {name!r} already states {both}, and the call states "
                         f"{'it' if len(both) == 1 else 'them'} again. The name would then "
                         f"label numbers it does not describe: drop {both}, or pass every "
                         f"number without the name.")
    return dict({"name": str(name)}, **entry, **params)


def jsonable(params):
    """`params` as a recipe holds them: lists for arrays, ``[re, im]`` pairs for complex
    pilot values."""
    out = dict(params)
    for k in ("cp", "active"):
        if k in out and not isinstance(out[k], (int, np.integer)):
            out[k] = [int(v) for v in np.asarray(out[k]).ravel()]
    for k in ("n_fft", "cp", "active", "window", "n_symbols"):
        if isinstance(out.get(k), np.integer):
            out[k] = int(out[k])
    if out.get("bits") is not None:
        out["bits"] = [int(b) for b in np.asarray(out["bits"]).ravel()]
    if out.get("pilots") is not None:
        sets = out["pilots"]
        one = isinstance(sets, dict)
        fixed = []
        for s in ([sets] if one else list(sets)):
            if isinstance(s, dict):
                s = dict(s)
                if "subcarriers" in s:
                    s["subcarriers"] = [int(v) for v in np.asarray(s["subcarriers"]).ravel()]
                if s.get("values") is not None:
                    v = np.asarray(s["values"])
                    s["values"] = (IQ.as_pairs(v) if np.iscomplexobj(v)
                                   else np.asarray(v, float).tolist())
            else:
                s = int(s)
            fixed.append(s)
        out["pilots"] = fixed[0] if one else fixed
    return out


# ------------------------------------------------------------------ the parameters, resolved
def _count(v, name, least):
    if isinstance(v, bool) or not isinstance(v, (int, np.integer)) or v < least:
        raise ValueError(f"ofdm: {name} is a whole number >= {least}, got {v!r}")
    return int(v)


def _subcarriers(v, name, n_fft):
    k = np.asarray(v)
    if k.ndim != 1 or k.size == 0 or not np.issubdtype(k.dtype, np.integer):
        raise ValueError(f"ofdm: {name} is a list of whole subcarrier numbers (signed, 0 the "
                         f"centre frequency), got {v!r}")
    lo, hi = -(n_fft // 2), (n_fft - 1) // 2
    if k.min() < lo or k.max() > hi or len(set(k.tolist())) != k.size:
        raise ValueError(f"ofdm: {name} must be distinct subcarriers in {lo} .. {hi} for "
                         f"n_fft={n_fft} (subcarrier k is k * scs_hz from the centre), got "
                         f"{k.min()} .. {k.max()}, {k.size} of them")
    return k.astype(np.int64)


def _pilot_sets(pilots, sc, n_fft):
    """`pilots=` -> ``[(columns into sc, values, every, first), ...]``."""
    if pilots is None:
        return []
    if isinstance(pilots, dict):
        pilots = [pilots]
    elif not isinstance(pilots, (list, tuple)) or not len(pilots):
        raise ValueError(f"ofdm: pilots is a list of subcarriers, a dict of subcarriers= / "
                         f"values= / every= / first=, or a list of such dicts; got {pilots!r}")
    elif not isinstance(pilots[0], dict):
        pilots = [{"subcarriers": list(pilots)}]
    col = {int(k): i for i, k in enumerate(sc)}
    out = []
    for s in pilots:
        extra = sorted(set(s) - set(_PILOT_KEYS)) if isinstance(s, dict) else None
        if extra is None or extra or "subcarriers" not in s:
            raise ValueError(f"ofdm: a pilot set is a dict of subcarriers (required), values, "
                             f"every and first"
                             + (f"; it does not take {extra}" if extra else f", got {s!r}"))
        k = _subcarriers(s["subcarriers"], "a pilot set's subcarriers", n_fft)
        off = sorted(int(v) for v in k if int(v) not in col)
        if off:
            raise ValueError(f"ofdm: pilot subcarriers {off} are not in active=; a pilot is "
                             f"carried on an active subcarrier")
        v = (np.ones(k.size, complex) if s.get("values") is None
             else IQ.as_complex(s["values"]))
        if v.size != k.size:
            raise ValueError(f"ofdm: a pilot set has {k.size} subcarriers and {v.size} values; "
                             f"values= is one per subcarrier, real or [re, im]")
        out.append((np.array([col[int(q)] for q in k]), v,
                    _count(s.get("every", 1), "a pilot set's every", 1),
                    _count(s.get("first", 0), "a pilot set's first", 0)))
    return out


def _numerology(p, partial=False):
    """Everything about the symbol that does not depend on the grid or the record length.
    `partial` is a named set, which may state only some of it."""
    missing = [k for k in ("n_fft", "scs_hz", "cp", "active") if p.get(k) is None]
    if missing and not partial:
        raise ValueError(f"ofdm needs {', '.join(k + '=' for k in missing)}: the FFT length, "
                         f"the subcarrier spacing in Hz, the cyclic prefix in samples at the "
                         f"FFT rate (0 for none) and the active subcarriers (a count, or a "
                         f"list of signed k). e.g. ofdm(n_fft=64, scs_hz=312.5e3, cp=16, "
                         f"active=52, constellation='qpsk', prbs=15)")
    out = {}
    if p.get("n_fft") is not None:
        out["n_fft"] = _count(p["n_fft"], "n_fft", 2)
    if p.get("scs_hz") is not None:
        out["scs_hz"] = float(p["scs_hz"])
        if not out["scs_hz"] > 0:
            raise ValueError(f"ofdm: scs_hz, the subcarrier spacing, must be positive; got "
                             f"{p['scs_hz']!r}")
    if p.get("cp") is not None:
        cp = p["cp"]
        cp = [cp] if isinstance(cp, (int, np.integer)) else list(cp)
        if not cp:
            raise ValueError("ofdm: cp is a count of samples at the FFT rate, or a list of "
                             "them cycled over the symbols; got an empty list")
        out["cp"] = [_count(c, "cp (samples at the FFT rate)", 0) for c in cp]
        if "n_fft" in out and max(out["cp"]) > out["n_fft"]:
            raise ValueError(f"ofdm: cp={max(out['cp'])} is longer than the useful part it "
                             f"repeats (n_fft={out['n_fft']} samples at the FFT rate)")
    out["window"] = _count(p.get("window", 0), "window (samples at the FFT rate)", 0)
    if "cp" in out and out["window"] > min(out["cp"]):
        raise ValueError(f"ofdm: window={out['window']} is longer than the shortest cyclic "
                         f"prefix ({min(out['cp'])}): the edge ramps inside the prefix, so the "
                         f"useful part stays whole. Use window <= {min(out['cp'])}, or a "
                         f"following filter.")
    if p.get("active") is not None and "n_fft" in out:
        n_fft, a = out["n_fft"], p["active"]
        if isinstance(a, (int, np.integer)) and not isinstance(a, bool):
            if a < 2 or a % 2 or a > 2 * ((n_fft - 1) // 2):
                raise ValueError(
                    f"ofdm: active={a} as a COUNT means that many subcarriers nearest the "
                    f"centre with the centre left out, half each side, so it is even and at "
                    f"most {2 * ((n_fft - 1) // 2)} for n_fft={n_fft}. Pass a list of signed "
                    f"subcarrier numbers for anything else.")
            sc = np.concatenate([np.arange(-(a // 2), 0), np.arange(1, a // 2 + 1)])
        else:
            sc = np.sort(_subcarriers(a, "active", n_fft))
        out["subcarriers"] = sc
        out["pilot_sets"] = _pilot_sets(p.get("pilots"), sc, n_fft)
    amp = p.get("amp", 1.0)
    if isinstance(amp, bool) or not isinstance(amp, (int, float, np.integer, np.floating)):
        raise ValueError(f"ofdm: amp is one real scale factor, got {amp!r}")
    out["amp"] = float(amp)
    return out


def check(params):
    """Refuse what can be refused without a grid, where the call is written."""
    _numerology(params)
    _constellation(params.get("constellation"))


def _constellation(con):
    if con is not None and con not in IQ.CONSTELLATIONS:
        raise ValueError(f"ofdm: unknown constellation {con!r}: the data elements are on "
                         f"one of {sorted(IQ.CONSTELLATIONS)}")
    return con


def _bit_source(p, n_data, k):
    """The data bits, or None when no element carries data. One source, as on `iq`."""
    given = [q for q in ("bits", "prbs") if p.get(q) is not None]
    if n_data == 0:
        stray = given + [q for q in ("constellation", "prbs_seed", "prbs_phase") if q in p]
        if stray:
            raise ValueError(f"ofdm: every active element is a pilot, so there is no data, "
                             f"and {stray} would be recorded and change nothing. Drop "
                             f"{'it' if len(stray) == 1 else 'them'}.")
        return None
    if len(given) != 1:
        raise ValueError(
            f"ofdm needs its data bits from exactly one source: prbs=<order> (a PRBS, filling "
            f"the data elements) or bits=[...]; got "
            f"{' and '.join(q + '=' for q in given) or 'neither'}.")
    need = n_data * k
    if given[0] == "bits":
        for q in ("prbs_seed", "prbs_phase"):
            if q in p:
                raise ValueError(f"ofdm: {q}= sets where a PRBS starts, and this source is "
                                 f"bits=, which states every bit itself. Drop {q}=.")
        if np.asarray(p["bits"]).size != need:
            raise ValueError(
                f"ofdm: {np.asarray(p['bits']).size} bits for {n_data} data elements of "
                f"{p['constellation']} ({k} bits each): pass exactly {need}. Nothing is padded or "
                f"truncated; n_symbols= sets how many symbols there are.")
    from .compose import _coded_bits                 # the one bit source `coded` and `iq` use
    return _coded_bits(dict(p, n_bits=need))


def plan(params, grid, n=None):
    """Where everything is, for these parameters on this grid: a dict of

      ``n_fft``, ``scs_hz``, ``fft_rate_hz``, ``oversample`` (the grid's rate over the FFT
      rate, a whole number), ``n_symbols``,
      ``subcarriers``      the active k, ascending -- the columns of the resource grid
      ``f_hz``             their offsets from the centre frequency
      ``pilot``            bool ``[symbol, subcarrier]``: True where the element is a pilot
      ``cp``               each symbol's cyclic prefix, in samples at the FFT rate
      ``start``            each symbol's first sample in the record (its prefix's)
      ``useful``           each symbol's first USEFUL sample, where the receiver's FFT opens
      ``samples``          each symbol's length in record samples, prefix included
      ``t_s``              ``useful / fs``
      ``window``           the edge ramp, in record samples
      ``noise_bw_hz``      ``n_active * scs_hz``: the bandwidth in which white noise reaches
                           the demodulated elements, so `awgn(snr_db=s, bw_hz=noise_bw_hz)`
                           is an SNR of s dB per resource element

    `n` is the record length where it is not the grid's. `params` is the op's own dict
    (``sig.ops[0]``) or the same keywords."""
    return {k: v for k, v in _plan(params, grid, n).items() if not k.startswith("_")}


def _plan(p, grid, n):
    num = _numerology(p)
    fs = getattr(grid, "fs", None) if grid is not None else None
    if fs is None:
        raise ValueError("ofdm: needs the sample rate -- Grid(fs=..., n=...)")
    fs, n_fft, scs = float(fs), num["n_fft"], num["scs_hz"]
    rate = n_fft * scs
    over = int(round(fs / rate))
    if over < 1 or abs(over * rate - fs) > RATE_TOL * fs:
        up = max(int(np.ceil(fs / rate - RATE_TOL)), 1)
        raise ValueError(
            f"ofdm: n_fft={n_fft} subcarriers {scs:.10g} Hz apart are an FFT rate of "
            f"{rate:.10g} Sa/s, and this grid is fs={fs:.10g}, {fs / rate:.6g} times that. "
            f"The symbol is an inverse FFT, so the grid must be the FFT rate or a WHOLE "
            f"multiple of it (the same subcarriers evaluated on a finer grid, exactly): "
            f"Grid(fs={rate:.10g}), or Grid(fs={up * rate:.10g}) for {up}x. Nothing is "
            f"resampled.")
    n = int(grid.n if n is None else n)
    cps = num["cp"]
    cycle = sum(n_fft + c for c in cps) * over
    whole, left = divmod(n, cycle)
    fit = whole * len(cps)
    for c in cps:                                    # the part of a cycle that still fits
        if left < (n_fft + c) * over:
            break
        left -= (n_fft + c) * over
        fit += 1
    if p.get("n_symbols") is None:
        n_sym = fit
        if not n_sym:
            raise ValueError(f"ofdm: a record of {n} samples is shorter than one symbol "
                             f"({(n_fft + cps[0]) * over} samples at {over}x the FFT rate)")
    else:
        n_sym = _count(p["n_symbols"], "n_symbols", 1)
        if n_sym > fit:
            raise ValueError(f"ofdm: n_symbols={n_sym} do not fit in a record of {n} samples "
                             f"({fit} whole symbols do). Lengthen the record or ask for "
                             f"fewer; nothing is truncated.")
    cp = np.array([cps[i % len(cps)] for i in range(n_sym)], np.int64)
    samples = (n_fft + cp) * over
    start = np.concatenate([[0], np.cumsum(samples)[:-1]]).astype(np.int64)
    sc = num["subcarriers"]
    pilot = np.zeros((n_sym, sc.size), bool)
    value = np.zeros((n_sym, sc.size), complex)
    for cols, v, every, first in num["pilot_sets"]:
        rows = np.arange(first, n_sym, every)
        if pilot[np.ix_(rows, cols)].any():
            raise ValueError("ofdm: two pilot sets claim the same element (the same "
                             "subcarrier on the same symbol); one value per element")
        pilot[np.ix_(rows, cols)] = True
        value[np.ix_(rows, cols)] = v
    useful = start + cp * over
    return {"n_fft": n_fft, "scs_hz": scs, "fft_rate_hz": rate, "oversample": over,
            "n_symbols": n_sym, "subcarriers": sc, "f_hz": sc * scs, "pilot": pilot,
            "cp": cp, "start": start, "useful": useful, "samples": samples,
            "t_s": useful / fs, "window": num["window"] * over,
            "noise_bw_hz": sc.size * scs, "_value": value, "_amp": num["amp"], "_n": n}


def resource_grid(params, grid, n=None):
    """THE GROUND TRUTH: `plan(...)` plus

      ``grid``    complex ``[symbol, subcarrier]`` -- the value on every active element
      ``bits``    the data bits in the order they were mapped (symbol by symbol, ascending
                  subcarrier within a symbol), or None when every element is a pilot

    ``grid[~pilot]`` are the data symbols in that same order. Resolved by the function that
    renders, so the two cannot disagree. `Signal.transmitted()` returns ``(grid, t_s)``."""
    return {k: v for k, v in _resolved(params, grid, n).items() if not k.startswith("_")}


def _resolved(params, grid, n):
    pl = _plan(params, grid, n)
    X = pl.pop("_value")
    n_data = int((~pl["pilot"]).sum())
    con = _constellation(params.get("constellation"))
    if n_data and con is None:
        raise ValueError(f"ofdm needs constellation=, the constellation of its data elements "
                         f"(one of {sorted(IQ.CONSTELLATIONS)})")
    bits = _bit_source(params, n_data, IQ.bits_per_symbol(con) if n_data else 0)
    if bits is not None:
        X[~pl["pilot"]] = IQ.map_bits(con, bits)
    return dict(pl, grid=X, bits=bits)


# ------------------------------------------------------------------ the two transforms
def _bins(pl):
    return pl["subcarriers"] % (pl["n_fft"] * pl["oversample"])


def _edge(w):
    """The rising half of the raised-cosine edge over `w` samples; the falling half is
    ``1 - _edge(w)``, so two overlapped symbols' ramps sum to one."""
    return np.sin(0.5 * np.pi * (np.arange(w) + 0.5) / w) ** 2


def modulate(params, grid, n=None):
    """The record: every symbol's inverse FFT, its prefix in front, the edges windowed."""
    return _render(_resolved(params, grid, n))


def _render(rg):
    n, m, w = rg["_n"], rg["n_fft"] * rg["oversample"], rg["window"]
    full = np.zeros((rg["n_symbols"], m), complex)
    full[:, _bins(rg)] = rg["grid"]
    u = np.fft.ifft(full, axis=1) * (m * rg["_amp"] / np.sqrt(rg["subcarriers"].size))
    up = _edge(w) if w else None
    x = np.zeros(n, complex)
    for i in range(rg["n_symbols"]):
        c = int(rg["useful"][i] - rg["start"][i])
        s = np.concatenate([u[i, m - c:], u[i], u[i, :w]])       # prefix, useful, suffix
        if w:
            s[:w] *= up
            s[-w:] *= 1.0 - up
        a = int(rg["start"][i])
        keep = min(len(s), n - a)        # the last symbol's suffix may pass the record's end
        x[a:a + keep] += s[:keep]
    return x


def run(params, grid, idx, n=None, sink=None):
    """The `ofdm` op: the record, and one event per symbol boundary."""
    pl = _resolved(params, grid, n)
    x, fs = _render(pl), float(grid.fs)
    if sink is not None:
        for i in range(pl["n_symbols"]):
            # an observation of the record's structure, not something done to it: a label
            # mask does not cover it (`on_record=False`)
            sink.append(Event(
                sample=int(pl["start"][i]), kind="ofdm_symbol", severity=0.0,
                t_s=float(pl["start"][i] / fs), width=int(pl["samples"][i]),
                placement=f"ofdm/{idx}", on_record=False,
                params={"symbol": i, "cp_s": float(pl["cp"][i] / pl["fft_rate_hz"]),
                        "t_useful_s": float(pl["t_s"][i])}))
    return x


def demodulate(x, grid, params, delay_s=0.0):
    """The record back to its resource grid: complex ``[symbol, subcarrier]``, the shape and
    order of `Signal.transmitted()[0]`.

    An FFT over each symbol's useful part, the active bins scaled by
    ``sqrt(n_active) / (n_fft * oversample)``. `grid` must describe `x`
    (`Signal.stored_grid(x)` after anything that changes the rate); its rate must still be a
    whole multiple of the FFT rate. `delay_s` is how late the record is against the
    transmitter's clock -- a channel's or a filter's group delay; negative opens the window
    early, inside the prefix. The window moves by the nearest whole sample and the rest is
    turned back as a phase slope across the subcarriers, which is exact while the window
    holds one symbol only. Nothing is estimated and nothing is fitted: a level error or a
    rotation comes back in the values. Without noise, a record straight from the op returns
    its grid to floating-point rounding (bound 1e-12; measured 9e-16, tests/test_ofdm.py)."""
    a = np.asarray(x)
    if a.ndim != 1 or not np.iscomplexobj(a):
        raise ValueError("ofdm.demodulate: expected one complex baseband record. "
                         "downconvert(fc_hz=..., lpf=...) a passband record first.")
    pl = plan(params, grid, a.size)
    fs, m = float(grid.fs), pl["n_fft"] * pl["oversample"]
    d = int(np.round(delay_s * fs))
    lo, hi = int(pl["useful"][0]) + d, int(pl["useful"][-1]) + d + m
    if lo < 0 or hi > a.size:
        raise ValueError(f"ofdm.demodulate: delay_s={delay_s:g} puts an FFT window outside "
                         f"the record (samples {lo} .. {hi} of {a.size}). Demodulate fewer "
                         f"symbols (n_symbols=), or a longer record.")
    rows = (pl["useful"][:, None] + d + np.arange(m)[None, :])
    got = np.fft.fft(a[rows], axis=1)[:, _bins(pl)] * (np.sqrt(pl["subcarriers"].size) / m)
    rest = delay_s - d / fs
    return got * np.exp(2j * np.pi * pl["f_hz"] * rest) if rest else got


# ------------------------------------------------------------------ EVM per resource element
def _timing(ref, got, f_hz, pl):
    """The delay whose phase slope across the subcarriers best explains `got`: the peak of
    ``|sum conj(ref) * got * exp(j*2*pi*f*t)|``, searched over one useful symbol on a grid
    `TIMING_SEARCH` times finer than the FFT rate, then refined."""
    c = np.sum(np.conj(ref) * got, axis=0)
    m = TIMING_SEARCH * pl["n_fft"]
    buf = np.zeros(m, complex)
    buf[pl["subcarriers"] % m] = c
    k = int(np.argmax(np.abs(np.fft.ifft(buf))))
    step = 1.0 / (m * pl["scs_hz"])
    t0 = (k if k < m // 2 else k - m) * step
    best = minimize_scalar(lambda t: -abs(np.sum(c * np.exp(2j * np.pi * f_hz * t))),
                           bounds=(t0 - step, t0 + step), method="bounded",
                           options={"xatol": 1e-9 * step})
    return float(best.x)


def evm(x, grid, sent, params, correct=(), skip=0, delay_s=0.0):
    """What `rf.evm(x, grid, sent, ofdm=params)` returns: the RMS distance between the
    demodulated resource grid and `sent` (`Signal.transmitted()[0]`), over the RMS of `sent`,
    counted over every active element -- data and pilots.

    `skip` leaves that many OFDM symbols out at each end. `delay_s` is the receiver's
    (`demodulate`). `correct` names what to take out first; nothing is taken out otherwise,
    each is a least-squares fit against `sent` in this order, and what it removed comes back
    in ``["removed"]``:

      ``"timing"``        one delay, as the phase slope across the subcarriers it causes;
                          ``timing_s``, found within +/- half a useful symbol. The window is
                          not moved, so a record late by more than its prefix covers keeps
                          the neighbouring symbol's leakage: whole delays go in `delay_s`
      ``"common_phase"``  the rotation of each symbol (an oscillator's phase, a residual
                          frequency error); ``common_phase_rad``, one per symbol
      ``"gain"``          one real scale factor; ``gain_db``
      ``"equalizer"``     one complex coefficient per subcarrier, the channel as the pilots
                          and data together show it; ``equalizer``, one per subcarrier

    A FIT ABSORBS NOISE. The equaliser spends one complex number per subcarrier, so over N
    counted symbols it takes 1/N of white noise's power with it and the figure reads low by
    ``sqrt(1 - 1/N)``; with one symbol it would read zero, and is refused. The common phase
    takes ``1 / (2 * n_active)`` the same way. And the equaliser DIVIDES by the channel, noise
    included: on a subcarrier the channel faded to ``|H|`` the noise left is ``1 / |H|``
    larger, which is the channel's cost and stays in the figure.

    Returns ``evm``, ``evm_pct``, ``evm_db``, ``evm_peak``; ``evm_per_subcarrier`` and
    ``evm_per_symbol`` (each over the RMS of the whole reference); ``subcarriers``;
    ``n_symbols``, ``n_elements``, ``skip``, ``reference``, ``receiver``, ``delay_s``,
    ``removed``; and ``received`` / ``reference_symbols``, ``[symbol, subcarrier]`` after
    `skip`, for the error on any one element."""
    correct = (correct,) if isinstance(correct, str) else tuple(correct)
    for c in correct:
        if c in _NOT_HERE:
            raise ValueError(f"evm: correct={c!r} is a single-carrier correction. On an OFDM "
                             f"record {_NOT_HERE[c]}. OFDM takes {list(CORRECTIONS)}.")
    if [c for c in correct if c not in CORRECTIONS] or len(set(correct)) != len(correct):
        raise ValueError(f"evm: with ofdm=, correct= takes each of {list(CORRECTIONS)} at "
                         f"most once, got {list(correct)}")
    if sent is None:
        raise ValueError("evm: an OFDM record is measured against the truth: "
                         "sent=sig.transmitted()[0], the resource grid.")
    ref = np.asarray(sent, complex)
    pl = plan(params, grid, np.asarray(x).size)
    want = (pl["n_symbols"], pl["subcarriers"].size)
    if ref.shape != want:
        raise ValueError(f"evm: sent has shape {ref.shape} and these ofdm parameters on this "
                         f"record are {want[0]} symbols of {want[1]} active subcarriers. Pass "
                         f"sig.transmitted()[0], the op's own parameters (sig.ops[0]) and the "
                         f"grid that describes the record (sig.stored_grid(x)).")
    skip = int(skip)
    if skip < 0 or ref.shape[0] - 2 * skip < 1:
        raise ValueError(f"evm: skip={skip} at each end leaves {ref.shape[0] - 2 * skip} of "
                         f"{ref.shape[0]} OFDM symbols; nothing to measure")
    keep = slice(skip, ref.shape[0] - skip)
    ref, got = ref[keep], demodulate(x, grid, params, delay_s=delay_s)[keep]
    power = float(np.mean(np.abs(ref) ** 2))
    if not power > 0:
        raise ValueError("evm: the reference grid is all zero; there is nothing to normalise "
                         "the error by")
    removed = {}
    if "timing" in correct:
        removed["timing_s"] = _timing(ref, got, pl["f_hz"], pl)
        got = got * np.exp(2j * np.pi * pl["f_hz"] * removed["timing_s"])
    if "common_phase" in correct:
        removed["common_phase_rad"] = np.angle(np.sum(np.conj(ref) * got, axis=1))
        got = got * np.exp(-1j * removed["common_phase_rad"])[:, None]
    if "gain" in correct:
        gain = abs(np.vdot(ref, got)) / (power * ref.size)
        removed["gain_db"] = float(20 * np.log10(gain))
        got = got / gain
    if "equalizer" in correct:
        if ref.shape[0] < 2:
            raise ValueError("evm: correct='equalizer' fits one coefficient per subcarrier, "
                             "and over a single OFDM symbol that fit is exact whatever the "
                             "record holds -- the figure would be zero. Count 2 symbols or "
                             "more.")
        energy = np.sum(np.abs(ref) ** 2, axis=0)
        if not energy.all():
            raise ValueError("evm: correct='equalizer' needs every active subcarrier to carry "
                             "something on a counted symbol; one is zero throughout")
        removed["equalizer"] = np.sum(np.conj(ref) * got, axis=0) / energy
        got = got / removed["equalizer"]
    err = np.abs(got - ref) ** 2
    val = float(np.sqrt(err.mean() / power))
    return {"evm": val, "evm_pct": 100.0 * val,
            "evm_db": float(20 * np.log10(val)) if val > 0 else float("-inf"),
            "evm_peak": float(np.sqrt(err.max() / power)),
            "evm_per_subcarrier": np.sqrt(err.mean(axis=0) / power),
            "evm_per_symbol": np.sqrt(err.mean(axis=1) / power),
            "subcarriers": pl["subcarriers"], "n_symbols": int(ref.shape[0]),
            "n_elements": int(ref.size), "skip": skip, "reference": "truth",
            "receiver": f"fft of {pl['n_fft'] * pl['oversample']} samples per symbol, "
                        f"{pl['subcarriers'].size} active subcarriers",
            "delay_s": float(delay_s), "removed": removed,
            "received": got, "reference_symbols": ref}
