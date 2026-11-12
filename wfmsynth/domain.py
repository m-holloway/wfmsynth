"""wfmsynth.domain -- which kind of sample each op accepts, and the refusal when it is handed
the other.

A record is in one of two domains:

    "real"   a voltage. Every record before complex baseband existed, and still the default.
    "iq"     COMPLEX BASEBAND: each sample is I + jQ, the envelope of a signal centred on
             `fc_hz`. `waveform()` returns complex128.

The domain a chain STARTS in is its SOURCE's when the grid does not state one: `iq(...)`, an
I/Q capture, or a `sequence` / `switch` of I/Q chains is an envelope, `carrier` / `symbols` /
`coded` / a real capture a voltage (`follow`). A grid may state it -- `Grid(domain="iq")`,
`Grid(domain="real")`, or `fc_hz=`, which only an envelope has -- and a source that
contradicts a stated domain is refused by name. Two ops move between the domains --
`upconvert` (iq -> real passband) and `downconvert` (real -> iq) -- and every other op leaves
the domain alone.

A chain that was NOT handed its grid by its author -- a sub-chain, a named chain of a
`Channels` record, a `Signal()` not yet given one -- inherits a grid it did not state, so it
starts in the domain its own ops need (`needs`): `Signal().carrier("sine", ...)` is a voltage
and `Signal().iq("qam16", ...)` an envelope whatever record they sit in. That is what lets an
envelope be gated by a real enable or scaled by a real supply. A Signal that states a grid is
still held to it: a voltage source on an I/Q grid is a contradiction and is refused.

WHAT THE GENERAL STAGES MEAN ON AN ENVELOPE, each tested both ways in tests/test_iq_seams.py:

    filter                 yes  its response is stated on the grid, about the centre frequency
    gain(by=)              yes  a real factor scales I and Q alike (a supply, an AGC, a fade)
    gate(by=)              yes  a real enable switches the envelope (a burst, a T/R switch)
    combine(how="sum")     yes  envelopes about the SAME centre add: an interferer, a blocker
    sequence, switch       yes  I/Q chains in time order (preamble then payload); `select` real
    use                    yes  a named chain is whatever domain its own ops make it
    combine product/min/max no  a product of envelopes is not a mixer and an order of complex
                                numbers does not exist
    offset                  no  a real number added to an envelope is a leak on the I rail
                                only; `iq_offset(i=, q=)` says which rail
    delay                   no  delaying a passband signal also turns its carrier by
                                -2*pi*fc*delay, which an envelope-only shift leaves out
    law                     no  the laws act on a voltage; `dac` is the quantiser of an
                                envelope, and an amplifier acts on the passband record
    trigger                 no  a scope triggers on a voltage

AN OP IS REAL-ONLY UNLESS IT SAYS OTHERWISE. That is the whole safety of the design: an op
written for a voltage that is handed a complex array does something -- numpy's `float` cast
drops the imaginary part with a warning nobody reads, an `rfft` raises three frames down, a
threshold compares complex numbers -- and none of those is the op's physics. So the default is
a refusal by name, and I/Q support is opted into one op at a time, by the person who has
checked that the op's physics means something on an envelope.

TO OPT AN OP IN, add ONE line to `ACCEPTS`:

    ACCEPTS = {
        ...
        "filter": BOTH,            # stated on the grid, so it means the same on an envelope
    }

and make its executor honest for a complex `x` (the composer checks one thing for you: an op
that returns a real array for an iq record is stopped, because that is the imaginary part
being dropped). An op that CHANGES the domain also gets a line in `EMITS`.

"A linear filter is linear, so it works on complex too" is NOT sufficient reason to opt in. The
general `filter` op is in because its response is stated relative to the grid. A channel stated
at RF (`lossy(loss_at_ghz=26)`) applied to an envelope filters around DC, not
around the carrier, which is a different channel -- its baseband equivalent is the response
shifted down by `fc_hz`, and that is new physics to write and check, not a flag to flip.
"""
from __future__ import annotations

REAL, IQ = "real", "iq"
BOTH = (REAL, IQ)

# op -> the domains it accepts. ABSENT MEANS REAL-ONLY.
ACCEPTS = {
    "capture": BOTH,           # a source: emits whatever domain the grid declares
    "iq": (IQ,),               # a source of complex symbols
    "upconvert": (IQ,),
    "downconvert": (REAL,),
    # `filter` states its response on the GRID (a corner in Hz from DC, taps, a symbol rate),
    # not at RF, so on an envelope it is exactly what it says: a filter about the centre
    # frequency -- the matched filter, a channel-select filter. tests/test_iq_domain.py.
    "filter": BOTH,
    # the I/Q impairments (`wfmsynth/rf.py`) are DEFINED on an envelope -- two rails, a phase,
    # a centre frequency -- and have no meaning on a real voltage. tests/test_iq_impairments.py.
    "iq_imbalance": (IQ,), "iq_offset": (IQ,), "freq_offset": (IQ,), "phase_noise": (IQ,),
    "awgn": (IQ,), "spur": (IQ,), "dac": (IQ,),
    # the general stages that mean the same thing on an envelope (table in the docstring).
    # `combine` is a sum only there, and what each takes as its OTHER signal -- a real factor,
    # a real enable, I/Q peers -- is checked where the op runs.
    "gain": BOTH, "gate": BOTH, "combine": BOTH, "sequence": BOTH, "switch": BOTH,
    # a named chain of a several-channel record: whatever its own ops make it (`trace`)
    "use": BOTH,
}

# an op refused in one domain that has a counterpart in the other, or a reason: said in its
# refusal. Keyed by the domain the record is in.
_INSTEAD = {
    IQ: {
        "offset": "On an envelope a real number is a leak on the I rail alone: "
                  "iq_offset(i=, q=) states both rails.",
        "delay": "A delay of a passband signal also turns its carrier by -2*pi*fc*delay, "
                 "which shifting the envelope alone would leave out; delay the real passband "
                 "record.",
        "law": "The laws act on a voltage. dac(bits=) is the quantiser of an envelope, and an "
               "amplifier's law acts on the passband record.",
        "trigger": "A scope triggers on a voltage.",
        "rx_noise": "Noise on an envelope is awgn(snr_db= | es_n0_db= | ebn0_db=): circular, "
                    "and stated against the signal's power.",
        "digitize": "The converter of an envelope is dac(bits=), which quantises both rails.",
        "store": "The converter of an envelope is dac(bits=), which quantises both rails.",
        "agc": "A gain that follows a level is gain(by=<a real chain>), which scales both "
               "rails alike.",
        "timing": "An oscillator's jitter on an envelope is phase_noise(profile=), and a "
                  "frequency error is freq_offset(f_hz=).",
    },
    REAL: {
        "awgn": "Noise on a voltage is rx_noise(rms=).",
        "iq_offset": "A DC term on a voltage is offset(by=).",
        "dac": "The converter of a voltage is digitize(bits=).",
        "phase_noise": "Clock jitter on a voltage is timing(rj_ps=, phase_noise=).",
    },
}
# the ops that START a voltage chain: on an I/Q grid there is no record yet to convert
_REAL_SOURCES = ("carrier", "symbols", "coded")

# op -> the domain it converts the record TO. Absent means the op leaves the domain alone.
EMITS = {"upconvert": REAL, "downconvert": IQ}

_WHAT = {REAL: "a real voltage record (domain 'real')",
         IQ: "a complex baseband record (domain 'iq')"}
_FIX = {
    IQ: "upconvert(fc_hz=...) turns it into a real passband record first; a chain meant to be "
        "real throughout needs a Grid without domain='iq'.",
    REAL: "downconvert(fc_hz=..., lpf=...) turns a real passband record into I/Q; a chain that "
          "starts in I/Q starts with .iq(...) or an I/Q capture.",
}


def of(grid):
    """The domain a chain on this grid starts in. No grid is a real record."""
    return getattr(grid, "domain", REAL) if grid is not None else REAL


def accepts(op):
    return ACCEPTS.get(op, (REAL,))


def refuse(op, dom, i=None):
    """The one refusal, so every op says it the same way: what the record is, what the op
    takes, and which op converts."""
    where = "" if i is None else f" (op {i})"
    want = " or ".join(repr(d) for d in accepts(op))
    if dom == IQ and op in _REAL_SOURCES and not i:
        return ValueError(
            f"{op}{where}: this chain is on an I/Q grid (Grid(domain='iq')), and {op} is the "
            f"source of a real voltage. An I/Q record starts with "
            f".iq(<constellation>, prbs=<order>); a real one needs a Grid without "
            f"domain='iq'.")
    if dom == REAL and op == "iq" and not i:
        return ValueError(
            f"iq: this chain's grid STATES domain='real', and iq is the source of a "
            f"complex baseband record. Leave domain= off the Grid -- the record's domain then "
            f"follows its source -- or state Grid(..., domain='iq', fc_hz=...).")
    instead = _INSTEAD[dom].get(op)
    return ValueError(
        f"{op}{where}: the record is {_WHAT[dom]} at this point in the chain, and {op} takes "
        f"domain {want} only. {instead + ' ' if instead else ''}"
        f"{_FIX[dom]} Nothing is converted implicitly: dropping or "
        f"inventing an imaginary part would be a different record under the same recipe.")


def needs(ops):
    """The domain these ops need the chain to START in, or None when none of them cares: that
    of the first op that takes one domain only, looked for up to the first mixer."""
    for o in ops:
        name = o.get("op")
        if len(accepts(name)) == 1:
            return accepts(name)[0]
        if name in EMITS:
            break
    return None


def _paired(values):
    """True for a recipe's complex `values`: ``[[re, im], ...]``."""
    return bool(values) and isinstance(values[0], (list, tuple)) and len(values[0]) == 2


def _spec_needs(spec):
    """The domain a sub-chain spec is in at its START, or None when it does not say."""
    if not isinstance(spec, dict):
        return None
    if isinstance(spec.get("ops"), (list, tuple)):
        return needs(spec["ops"]) or (source_needs(spec["ops"][0]) if spec["ops"] else None)
    if isinstance(spec.get("from"), dict):
        return (spec["from"].get("grid") or {}).get("domain", REAL)
    return None


def source_needs(op):
    """The domain a chain whose FIRST op is `op` is in, or None when that op does not say (a
    stage; a capture whose file is real, missing, or a .csv, which cannot say). A `sequence`
    or `switch` is what its chains are, when they agree."""
    name = op.get("op")
    if name == "capture":
        if op.get("format") is not None:
            return IQ
        if op.get("values") is not None:
            return IQ if _paired(op["values"]) else REAL
        if op.get("path") is not None:
            from . import capture as CAP
            return IQ if CAP.holds_complex(op["path"]) else None
        return None
    if name in ("sequence", "switch"):
        specs = ([s.get("chain") for s in op.get("segments") or () if isinstance(s, dict)]
                 if name == "sequence" else list(op.get("inputs") or ()))
        doms = {_spec_needs(s) for s in specs} - {None}
        return doms.pop() if len(doms) == 1 else None
    return needs([op])


def follow(grid, op):
    """The grid a chain STARTING with `op` is on: `grid`, in the source's domain when the grid
    did not state one. A stated domain is left alone, for `trace` to refuse a contradiction."""
    if grid is None or getattr(grid, "domain_stated", True):
        return grid
    want = source_needs(op)
    if want is None or want == of(grid):
        return grid
    import dataclasses
    return dataclasses.replace(grid, domain=want)


def grid_for(ops, grid):
    """The grid an INHERITED chain is rendered on: `grid` (the record's), in the domain the
    chain's own ops need. For a sub-chain and for a named chain of a `Channels` record, which
    did not state the grid they are given; never for a Signal that did."""
    want = needs(ops)
    if grid is None or want is None or want == of(grid):
        return grid
    import dataclasses
    return dataclasses.replace(grid, domain=want, fc_hz=None)


def trace(ops, grid, start=None, expand=None):
    """``[(domain, fc_hz), ...]`` -- the record's domain and centre frequency AFTER each op,
    raising the named refusal at the first op handed a domain it does not accept.

    `fc_hz` is None for a real record and for an I/Q record whose centre was never stated.

    `start` is ``(domain, fc_hz)`` where the chain does not begin on the grid's: a shared stage
    begins on whatever the chain using it has reached. With no grid at all the chain starts in
    the domain its ops need. `expand` resolves a `use` op -- ``expand(name)`` gives
    ``(is_signal, ops)`` of the named chain with its own `use`s already replaced -- and without
    it a `use` is taken to leave the domain alone (a lone chain cannot know)."""
    if start is not None:
        dom, fc = start
    elif grid is None:
        dom, fc = needs(ops) or REAL, None
    else:
        dom, fc = of(grid), getattr(grid, "fc_hz", None)
    out = []
    for i, o in enumerate(ops):
        name = o["op"]
        if dom not in accepts(name):
            raise refuse(name, dom, i)
        if name == "use" and expand is not None:
            is_signal, inner = expand(o["name"])
            if is_signal:                    # it IS the record from here: its own domain
                g = grid_for(inner, grid)
                dom, fc = trace(inner, g)[-1]
            else:
                dom, fc = trace(inner, grid, start=(dom, fc))[-1]
            out.append((dom, fc))
            continue
        to = EMITS.get(name)
        if to == REAL:
            fc = None
        elif to == IQ:
            fc = o.get("fc_hz")
        if to is not None:
            dom = to
        out.append((dom, fc))
    return out
