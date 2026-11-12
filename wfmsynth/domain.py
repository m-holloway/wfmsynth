"""wfmsynth.domain -- which kind of sample each op accepts, and the refusal when it is handed
the other.

A record is in one of two domains:

    "real"   a voltage. Every record before complex baseband existed, and still the default.
    "iq"     COMPLEX BASEBAND: each sample is I + jQ, the envelope of a signal centred on
             `fc_hz`. `waveform()` returns complex128.

The domain a chain STARTS in is the grid's (`Grid(domain="iq", fc_hz=...)`). Two ops move
between them -- `upconvert` (iq -> real passband) and `downconvert` (real -> iq) -- and every
other op leaves the domain alone.

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
}

# op -> the domain it converts the record TO. Absent means the op leaves the domain alone.
EMITS = {"upconvert": REAL, "downconvert": IQ}

_WHAT = {REAL: "a real voltage record (domain 'real')",
         IQ: "a complex baseband record (domain 'iq')"}
_FIX = {
    IQ: "upconvert(fc_hz=...) turns it into a real passband record first; a chain meant to be "
        "real throughout needs a Grid without domain='iq'.",
    REAL: "downconvert(fc_hz=..., lpf=...) turns a real passband record into I/Q; a chain that "
          "starts in I/Q needs Signal(grid=Grid(..., domain='iq', fc_hz=...)).",
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
    return ValueError(
        f"{op}{where}: the record is {_WHAT[dom]} at this point in the chain, and {op} takes "
        f"domain {want} only. {_FIX[dom]} Nothing is converted implicitly: dropping or "
        f"inventing an imaginary part would be a different record under the same recipe.")


def trace(ops, grid):
    """``[(domain, fc_hz), ...]`` -- the record's domain and centre frequency AFTER each op,
    raising the named refusal at the first op handed a domain it does not accept.

    `fc_hz` is None for a real record and for an I/Q record whose centre was never stated."""
    dom = of(grid)
    fc = getattr(grid, "fc_hz", None) if grid is not None else None
    out = []
    for i, o in enumerate(ops):
        name = o["op"]
        if dom not in accepts(name):
            raise refuse(name, dom, i)
        to = EMITS.get(name)
        if to == REAL:
            fc = None
        elif to == IQ:
            fc = o.get("fc_hz")
        if to is not None:
            dom = to
        out.append((dom, fc))
    return out
