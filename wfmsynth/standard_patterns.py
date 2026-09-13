"""
wfmsynth.standard_patterns — the sequences documents designate, each with its PERIOD and the
clause it was read from.

`wfmsynth.patterns` holds the mechanisms and deliberately holds no standards knowledge: a PRBS is
a polynomial, a seed and a length, while "which sequence does clause X designate" is a fact that
moves with the revision of a document. This module is the other half of that split — the knowledge
layer — kept in one file so that a revision touches one file and the physics kernel never depends
on a document. Importing it registers the names below; nothing else in the library imports it, so
the kernel stays document-free.

THE LOAD-BEARING FIELD IS `period`. A caller sizes a record from it (an integer number of pattern
repetitions is what makes a capture foldable, and what an analyser pattern-locks to), so a wrong
period does not raise — it silently produces a record that cannot be folded. Every period here is
therefore the MINIMAL period of the emitted symbol sequence, and `tests/test_standard_patterns.py`
asserts, for every entry, that the sequence repeats at that period and at no proper divisor of it.

EVERY FACT IS MARKED. `CITED` means it was read in the document named on the entry.
`PRELIMINARY` means it came from somewhere else (a working-group contribution, a comment
resolution, a conformance-test document quoting a clause) and says what would confirm it. A guess
is not a third category: where the construction could not be established, the name is NOT
registered and the gap is written down at the bottom of this file with what it needs.

    import wfmsynth.standard_patterns          # registers the names
    import wfmsynth.patterns as PAT

    PAT.resolve("ssprq", length=65535)         # one full period of PAM4 symbols
    PAT.get("ssprq").period                    # 65535 -- what to size the record from
    Signal(...).pattern("ssprq", length=65535)
"""
from __future__ import annotations

import numpy as np

from . import patterns as PAT
from . import physics as P

# --------------------------------------------------------------- SSPRQ (quaternary)
#
# SSPRQ -- Short Stress Pattern Random Quaternary, IEEE Std 802.3 Clause 120.5.11.2.3. The point of
# it: PRBS31Q carries the stressors a compliance measurement wants and has a period of 2^31-1
# symbols, so no instrument can PATTERN LOCK to it inside a capture. SSPRQ takes the stressful
# sub-sequences out of PRBS31 and closes at 2^16-1 = 65535 symbols, which folds.
#
# CONSTRUCTION, and what each fact rests on:
#
#   CITED  Three sections of PRBS31, seeded from Table 120-2, of 10924 + 10922 + 10922 bits
#          (32768 bits total). The PRBS31 generator is the one of Clause 49 Figure 49-9:
#          G(x) = 1 + x^28 + x^31 with an INVERTER at the output, and the hex value is the SEED
#          that presets S30..S0 (S30 = MSB), not the first bits emitted. That last distinction is
#          the whole reason the sequence was re-issued: the same table read as "the first 31 bits
#          of each section" gives a different, non-conformant pattern.
#   CITED  The bits are Gray coded to PAM4 symbols per Clause 120.5.11.2.1 (the mapping already in
#          `physics.GRAY_PAM4`), and the symbols of every second repetition of the 32768-bit
#          sequence are INVERTED (out = 3 - in).
#   CITED  The pattern is that 32768-bit sequence sent twice (65536 bits), then sent twice again
#          with the first and last bit removed (65534 bits): 131070 bits = 65535 symbols, which is
#          how an odd symbol count comes out of an even-length building block.
#
# VERIFIED, and this is the fact that matters more than any of the above: the 65535 symbols this
# code produces are byte-identical to the machine-readable SSPRQ symbol sequence IEEE publishes as
# an extract of IEEE Std 802.3 (file `SSPRQ_sequence.csv` on the IEEE 802.3 machine-readable
# extracts page). The test pins the sha256 of that file's symbols, so any drift in the constants
# below is caught against the normative artefact rather than against this comment.
#
# The seeds are quoted from Table 120-2 as resolved by the D2.1 ballot comment that changed the
# table heading from "Start" to "Seed" (#152, accepted); the section lengths were read from the
# same table. PRELIMINARY on the provenance of those two lines only -- the ballot record and a
# secondary quotation of the table, not the published clause text. What would confirm them: the
# text of 120.5.11.2.3 and Table 120-2 in a published revision of IEEE Std 802.3. The SEQUENCE
# needs no such confirmation; it is checked against IEEE's own file.
SSPRQ_SEEDS = (0x00000002, 0x34013FF7, 0x0CCCCCCC)
SSPRQ_SECTION_BITS = (10924, 10922, 10922)
SSPRQ_PERIOD = (1 << 16) - 1                      # 65535 symbols -- the reason the pattern exists

# The two symbol ranges whose symbols are inverted: repetitions 2 and 4 of the 32768-bit sequence.
# Expressed in SYMBOLS rather than bits because "out = 3 - in" is defined on symbols, and because
# the second pair of repetitions is offset by one bit -- inverting the wrong parity of bits there
# is the error this spelling makes impossible.
_SSPRQ_INVERTED = ((16384, 32768), (49151, 65535))


def _prbs31_section(seed, n_bits):
    """One SSPRQ section: `n_bits` from the Clause 49 Figure 49-9 PRBS31 generator preset with
    `seed`.

    The seed presets S30..S0 (MSB into S30) and the output is inverted, which together mean the
    seed is the 31 bits IMMEDIATELY BEFORE the first emitted bit rather than the first 31 bits
    emitted. So the register window starts as the seed and the first output is the next bit of the
    sequence, complemented:

        p[i] = p[i-31] XOR p[i-28]        (G(x) = 1 + x^28 + x^31)
        out[i] = NOT p[i]                 (the output inverter of Figure 49-9)

    A plain Fibonacci LFSR that emits its last stage instead would emit the seed as its first 31
    bits and produce a sequence 31 symbols out of phase with the standard's -- a pattern with the
    right statistics that no analyser locks to.
    """
    w = [(int(seed) >> (30 - j)) & 1 for j in range(31)]        # S30..S0, MSB first
    out = np.empty(int(n_bits), np.int8)
    for i in range(int(n_bits)):
        b = w[0] ^ w[3]                        # p[i] from the window p[i-31 .. i-1]
        out[i] = 1 - b
        w = w[1:] + [b]
    return out


def _ssprq_period():
    """One full period: 65535 PAM4 levels in {-1, -1/3, +1/3, +1}. Cached -- the sections cost a
    32768-step shift-register loop and the sequence is a constant."""
    cached = getattr(_ssprq_period, "_cache", None)
    if cached is not None:
        return cached
    a = np.concatenate([_prbs31_section(s, n)
                        for s, n in zip(SSPRQ_SEEDS, SSPRQ_SECTION_BITS)])     # 32768 bits
    # Sent twice, then sent twice again with the first and last bit removed. 131070 bits, and the
    # trim is what makes the symbol count odd (65535) instead of a power of two.
    bits = np.concatenate([a, a, a[1:], a[:-1]])
    pairs = bits.reshape(-1, 2)
    sym = np.array([P.GRAY_PAM4[(int(x), int(y))] for x, y in pairs], float)
    for lo, hi in _SSPRQ_INVERTED:
        sym[lo:hi] = -sym[lo:hi]               # out = 3 - in, on levels symmetric about zero
    _ssprq_period._cache = sym
    return sym


def ssprq(length, phase=0):
    """SSPRQ PAM4 symbols, IEEE Std 802.3 Clause 120.5.11.2.3 — a repeating 65535-symbol sequence
    built from four repetitions of three PRBS31 sections (see the construction notes above).

    Returns `length` levels in {-1, -1/3, +1/3, +1}, tiled from the period. `phase` is a starting
    position in the pattern, in symbols: a real capture starts wherever the link happened to be,
    and every record starting at symbol 0 lets a model learn the position instead of the signal.

    QUATERNARY, and it does not degrade to a binary link: the stressors it carries are level
    transitions of a four-level eye. The NRZ counterpart is not this pattern at a lower level
    count -- see `kr_training` and `kr_square16`.
    """
    n = int(length)
    base = _ssprq_period()
    if phase:
        base = np.roll(base, -(int(phase) % SSPRQ_PERIOD))
    return np.tile(base, int(np.ceil(n / SSPRQ_PERIOD)))[:n] if n else base[:0]


PAT.register("ssprq", ssprq, levels=4, period=SSPRQ_PERIOD,
             source="IEEE Std 802.3 Clause 120.5.11.2.3, Table 120-2 (PRBS31 generator of "
                    "Figure 49-9); Gray coding per Clause 120.5.11.2.1")


# --------------------------------------------------------------- the other quaternary sequences
#
# Already in the kernel; registered here because the CLAUSE is the part that belongs in this file,
# and because a recipe cannot name a kernel function -- it names a registry entry. The adapters
# exist only to put `length` first, which is the registry's calling contract.
def prbs13q(length, seed=1):
    """PRBS13Q PAM4 symbols, IEEE Std 802.3 Clause 120.5.11.2.1. Period 8191 SYMBOLS: the bit
    sequence is 8191 bits, odd, so the pattern closes only after two repetitions of it (16382
    bits) -- which is why the symbol period equals the bit period rather than half of it."""
    return P.prbs13q(int(length), seed)


def prbs31q(length, seed=1):
    """PRBS31Q PAM4 symbols. Period 2147483647 symbols (2 x (2^31-1) bits), far beyond a record:
    `length` symbols from the LFSR state `seed` is a capture-length SEGMENT of it, which is what a
    compliance capture of this pattern is. The contrast case for SSPRQ -- same stressors, no fold."""
    return P.prbs31q(int(length), seed)


PAT.register("prbs13q", prbs13q, levels=4, period=8191,
             source="IEEE Std 802.3 Clause 120.5.11.2.1")
# PRELIMINARY on the clause number only: the sub-subclause that defines PRBS31Q was not read (the
# numbering under 120.5.11.2 moved between drafts -- SSPRQ itself was 120.5.11.2.5 in D2.1 and is
# 120.5.11.2.3 in the published revision). The SEQUENCE is the kernel's, whose Gray mapping and
# level statistics are checked in wfmsynth.validate. What would confirm it: the clause list of
# 120.5.11.2 in a published revision.
PAT.register("prbs31q", prbs31q, levels=4, period=(1 << 31) - 1,
             source="IEEE Std 802.3 Clause 120.5.11.2 (PRBS31Q; sub-subclause not verified)")


# --------------------------------------------------------------- NRZ backplane Ethernet
#
# THE ANSWER TO "WHAT IS THE NRZ SSPRQ": there is not one, and that is a finding rather than a gap.
# For 10GBASE-KR (IEEE Std 802.3 Clause 72) the named sequences are, CITED from Clause 72:
#
#   * the TRAINING PATTERN of 72.6.10.2.6 -- 4094 bits of PRBS11 followed by two zeros, 512 octets
#     per training frame, on the PRBS11 polynomial G(x) = 1 + x^9 + x^11 of Equation (72-1). This
#     is the short, lockable stress sequence on the link: order 11, period 2047 bits, and the frame
#     it sits in is 4096 bits.
#   * a SQUARE WAVE with a period of 16 unit intervals (72.6.4, and the transmitter output waveform
#     of 72.7.1.11 which calls for the square wave of 52.9.1.2 with a run of at least eight
#     consecutive ones). Eight ones and eight zeros: one transition per eight UI, no pattern-
#     dependent ISI at all, which makes it the ISI-free contrast case rather than a stress pattern.
#   * test patterns 2 or 3 of 52.9.1.1 for jitter and output waveform (72.7.1.11, 72.7.1.12) and
#     for interference tolerance (72.7.2). NOT REGISTERED -- see the gaps at the bottom.
#
# Annex 69A, which the task suggested might hold an SSPRQ analogue, does not define a pattern at
# all: it is the interference-tolerance METHOD (test channel, interference generator, amplitude
# limits) and it refers the pattern out to 52.9.1.1. CITED from Annex 69A.2.1.

KR_TRAINING_BITS = 4096                   # 512 octets: 4094 PRBS11 bits + two zeros
KR_PRBS11_TAPS = (11, 9)                  # G(x) = 1 + x^9 + x^11, Equation (72-1)


def kr_training(length, seed=1):
    """The 10GBASE-KR training pattern, IEEE Std 802.3 Clause 72.6.10.2.6: 4094 bits from a PRBS11
    generator (G(x) = 1 + x^9 + x^11) followed by two zeros, 4096 bits in all, as ±1 per UI.

    PERIOD 4096 BITS ONLY IF THE SEED IS HELD. The clause requires "a random seed at the start of
    the training pattern", so a real link re-seeds every frame and a capture of it does NOT fold at
    4096 -- which is why compliance measurements on this port type use the sequences of 52.9.1.1
    rather than the training pattern. Holding `seed` is a deliberate choice a dataset makes to get
    a foldable record, and the recipe records the seed it held.

    The two zeros are the reason this is not simply "PRBS11": they break the shift register's
    recursion once per frame, so a receiver locking to the polynomial alone mispredicts two bits in
    4096 -- a detail worth having in training data about lock failures.
    """
    n = int(length)
    frame = np.empty(KR_TRAINING_BITS, float)
    frame[:4094] = np.where(P.prbs(11, 4094, seed) > 0, 1.0, -1.0)
    frame[4094:] = -1.0                                       # two zeros
    return np.tile(frame, int(np.ceil(n / KR_TRAINING_BITS)))[:n] if n else frame[:0]


PAT.register("kr_training", kr_training, levels=2, period=KR_TRAINING_BITS,
             source="IEEE Std 802.3 Clause 72.6.10.2.6, Equation (72-1) and Figure 72-3")

# Eight ones then eight zeros. Registered as a block rather than as a special case, because that is
# what it is, and the block is the whole documentation of the period.
PAT.register("kr_square16", PAT.block_repeat, levels=2, period=16, generator="block_repeat",
             params={"block": [1.0] * 8 + [-1.0] * 8},
             source="IEEE Std 802.3 Clause 72.6.4 (square wave, period 16 UI; the pattern of "
                    "Clause 52.9.1.2 as referenced by Clause 72.7.1.11)")


# --------------------------------------------------------------- PCI Express
#
# The compliance pattern of Polling.Compliance: the 8b/10b symbols K28.5, D21.5, K28.5, D10.2
# repeating, with the running disparity the table fixes (negative, positive, positive, negative).
# CITED, and the ten-bit code groups are quoted verbatim from the same table, which is why no
# 8b/10b encoder is needed here:
#
#     K28.5 (RD-)  0011111010        D21.5 (RD+)  1010101010
#     K28.5 (RD+)  1100000101        D10.2 (RD-)  0101010101
#
# Period 40 bits: four code groups, and the disparity returns to negative at the end of the fourth,
# so the 40-bit sequence closes rather than alternating into an 80-bit one.
#
# NOT the per-lane variant: the specification delays every eighth lane by four symbols (two K28.5
# delay symbols at each end). That is a LANE-INDEX-dependent sequence, so it is a property of a
# multi-lane scene rather than of a pattern name, and registering one name for it would put a lane
# number inside a pattern. A caller building a multi-lane record applies the delay itself.
_PCIE_COMPLIANCE_BITS = ("0011111010" "1010101010" "1100000101" "0101010101")

PAT.register("pcie_compliance", PAT.block_repeat, levels=2, period=40, generator="block_repeat",
             params={"block": [1.0 if c == "1" else -1.0 for c in _PCIE_COMPLIANCE_BITS]},
             source="PCI Express Base Specification Revision 4.0 Version 1.0, Section 4.2.8")


# --------------------------------------------------------------- DisplayPort
#
# CITED from the VESA DisplayPort Standard Version 2.0:
#
#   * PRBS7, G(x) = x^7 + x^6 + 1 (Appendix N.1), period 127 bits. The standard prints the
#     sequence itself in the DPCD LINK_QUAL_PATTERN_SET description ("11b = PRBS7 transmitted",
#     upper left transmitted first) -- 127 bits, so the PHASE is specified and not just the
#     polynomial. `seed=96` is the state of `physics.prbs` that reproduces those 127 bits exactly;
#     the test asserts it against the printed sequence, so the phase cannot drift silently.
#   * PRBS31, G(x) = x^31 + x^28 + 1 (Appendix N.6), period 2147483647 bits, unscrambled. Same
#     polynomial as the Ethernet PRBS31, and the entry exists under its own name because the
#     document that designates it is the part that is worth citing.
#
# TPS4 is NOT registered: the document defines it by reference ("CP2520 Pattern 3", the HBR2
# compliance eye pattern, a repetition of scrambled 00h) into the DisplayPort PHY compliance test
# standard, which was not read. See the gaps.
DP_PRBS7_SEED = 96          # the phase the standard prints; verified bit-for-bit in the tests

PAT.register("dp_prbs7", PAT.lfsr, levels=2, period=127, generator="lfsr",
             params={"taps": [7, 6], "seed": DP_PRBS7_SEED},
             source="VESA DisplayPort Standard Version 2.0, Appendix N.1 and the DPCD "
                    "LINK_QUAL_PATTERN_SET = 11b bit sequence")
PAT.register("dp_prbs31", PAT.lfsr, levels=2, period=(1 << 31) - 1, generator="lfsr",
             params={"taps": [31, 28]},
             source="VESA DisplayPort Standard Version 2.0, Appendix N.6")


# --------------------------------------------------------------- USB
#
# CITED from the Universal Serial Bus 3.2 Specification, Revision 1.1, Table 6-14 (compliance
# pattern sequences) and 6.4.4.1. The patterns are cycled by ping LFPS, so each is its own
# sequence rather than a stage of one long one.
#
# Registered: the ones whose emitted bit sequence the documents fix.
#
#   CP1  D10.2, "Nyquist frequency". The 8b/10b code group is 0101010101 (disparity neutral, so it
#        does not alternate with running disparity) -- alternating bits, minimal period 2 UI. The
#        code group's bits are CITED from the PCI Express table above rather than from an 8b/10b
#        table, since both standards use the same code and one of them prints the bits.
#   CP3  K28.5, the COM pattern. K28.5 has disparity +-2, so consecutive COMs alternate between
#        the two code groups and the period is 20 UI, not 10.
#   CP7  "Repeating 50-250 1's and then 50-250 0's" (with de-emphasis; CP8 is the same sequence
#        without it, which is a transmitter setting and not a different sequence). The run length
#        is a RANGE in the document, so the period is not a property of the name until the run is
#        fixed: this entry fixes it at the 50 the range starts from, and its period says 100. A
#        caller wanting another run registers their own name for it -- the alternative, a run
#        parameter on an entry whose `period` cannot follow it, is exactly the silent folding break
#        that the period field exists to prevent.
#   CP12 "Uncoded LFSR15 ... The polynomial is x^15+x^14+1", not 128b/132b encoded, so the LFSR
#        bits ARE the line sequence. Period 32767 UI.
_USB_CP1_CODE_GROUP = "0101010101"         # D10.2, RD-neutral
_USB_CP3_CODE_GROUPS = "0011111010" "1100000101"        # K28.5 RD- then RD+

PAT.register("usb_cp1_nyquist", PAT.block_repeat, levels=2, period=2, generator="block_repeat",
             params={"block": [1.0 if c == "1" else -1.0 for c in _USB_CP1_CODE_GROUP]},
             source="Universal Serial Bus 3.2 Specification Revision 1.1, Table 6-14 (CP1, "
                    "D10.2); code group bits per PCI Express Base Specification Revision 4.0 "
                    "Section 4.2.8")
PAT.register("usb_cp3_com", PAT.block_repeat, levels=2, period=20, generator="block_repeat",
             params={"block": [1.0 if c == "1" else -1.0 for c in _USB_CP3_CODE_GROUPS]},
             source="Universal Serial Bus 3.2 Specification Revision 1.1, Table 6-14 (CP3, "
                    "K28.5); code group bits per PCI Express Base Specification Revision 4.0 "
                    "Section 4.2.8")
PAT.register("usb_cp7_lowfreq50", PAT.block_repeat, levels=2, period=100,
             generator="block_repeat", params={"block": [1.0] * 50 + [-1.0] * 50},
             source="Universal Serial Bus 3.2 Specification Revision 1.1, Table 6-14 (CP7/CP8, "
                    "50-250 ones then zeros; this entry fixes the run at 50)")
PAT.register("usb_cp12_lfsr15", PAT.lfsr, levels=2, period=(1 << 15) - 1, generator="lfsr",
             params={"taps": [15, 14]},
             source="Universal Serial Bus 3.2 Specification Revision 1.1, Table 6-14 (CP12, "
                    "uncoded LFSR15, x^15 + x^14 + 1)")


# --------------------------------------------------------------- what is NOT registered, and why
#
# Each of these is a name a caller may reasonably expect. Declining is the honest answer: a period
# is a number a caller sizes a record from, and a guessed one breaks folding silently. What each
# needs is written down so the next pass is short.
#
#   USB CP0 and CP9. Both are SCRAMBLED sequences: CP0 is "D0.0 scrambled ... exactly the same as
#       logical idle", CP9 is a SYNC ordered set followed by scrambled 00h symbols and the document
#       states its period directly -- 65536 symbols (6.4.4.1). Generating either needs the USB
#       scrambler, its per-lane seeds (6.13.5) and, for CP9, 128b/132b framing, none of which were
#       read. The declared period is not enough: the entry must produce the symbols to be worth
#       having. What it needs: 6.13.5 and the 128b/132b block layout.
#   USB CP2, CP5, CP6 (D24.3, K28.7). Their ten-bit code groups were not read in any document here
#       -- the specification names the code group and not its bits -- and inferring them from the
#       8b/10b rules would be a guess about running disparity. What it needs: the 8b/10b tables of
#       IEEE Std 802.3 Clause 36 (Table 36-1, Table 36-2) or a document printing the bits.
#   USB CP4 (LFPS) is a low-frequency periodic SIGNALLING burst, not a data pattern: it is a burst
#       of a 10-50 MHz square wave gated by tBurst/tRepeat. It belongs to `impairments.burst_gate`
#       and a scene, not to a pattern name whose period is one number.
#   Test patterns 1, 2 and 3 of IEEE Std 802.3 Clause 52.9.1.1, which Clause 72 calls for on
#       10GBASE-KR jitter, output waveform and interference tolerance. Clause 52 was not read (it
#       is in a different section of the standard from Clause 72). What it needs: 52.9.1.1 and
#       52.9.1.2. Until then `kr_training` and `kr_square16` are the Clause 72 sequences this file
#       can defend.
#   1000BASE-T test modes 1-4 (IEEE Std 802.3 Clause 40.6.1.1.2, Figures 40-19 to 40-21). The
#       conformance documents to hand cite the clause and reproduce none of it, and the symbol
#       sequences are quinary (`wfmsynth.quinary` has the levels). Test modes 2 and 3 are widely
#       described as the same short sequence sent with MASTER and with SLAVE timing, which would
#       make them one pattern and two clock sources rather than two patterns -- but "widely
#       described" is not a citation. What it needs: 40.6.1.1.2 and its three figures.
#   The PCI Express MODIFIED compliance pattern (Section 4.2.9) appends two error-status symbols
#       whose value is the receiver's error count. The sequence is therefore not a constant, and a
#       period of 80 would be true only while no errors are counted.

__all__ = ["ssprq", "prbs13q", "prbs31q", "kr_training", "SSPRQ_SEEDS", "SSPRQ_SECTION_BITS",
           "SSPRQ_PERIOD", "KR_TRAINING_BITS", "KR_PRBS11_TAPS", "DP_PRBS7_SEED"]
