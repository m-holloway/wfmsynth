"""
wfmsynth.grid — absolute-unit binding for the synthesis primitives.

The primitives work on a normalized, unitless grid: time is samples (or a fraction of
the record), and frequency is cycles-per-record (or a fraction of Nyquist). A `Grid`
binds that abstract grid to real units — sample rate, symbol rate, record length,
full-scale voltage — so a caller can express a *real* link: delays in seconds, jitter
in seconds, periodic-jitter and corner frequencies in Hz, channel loss in dB at a
stated frequency.

Pass `grid=Grid(...)` (plus the absolute-unit keyword) to a primitive to specify in
real units; omit it and the legacy fraction/sample/cycles-per-record forms are
unchanged (bit-identical). See `wfmsynth.physics` for which primitives accept it.

    g = Grid(fs=256e9, baud=112e9, n=1 << 20)   # 256 GSa/s, 112 GBd, ~1M points
    g.dt, g.duration, g.f_nyquist                # 3.9 ps, 4.1 us, 128 GHz
    g.samples_per_ui                             # 2.2857...  (non-integer, as in reality)
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Grid:
    fs: float                          # sample rate [Hz]
    baud: Optional[float] = None       # symbol rate [Bd] (optional)
    n: int = 4096                      # record length [samples]
    v_full: float = 1.0                # full-scale voltage: amplitude +/-1 maps to +/-v_full/2
    # optional piecewise timebase: a tuple of (name, baud, n_symbols) segments back-to-back. None =
    # a single uniform segment (unchanged, bit-identical). Enables dual-rate records (CAN-FD BRS,
    # DDR bursts) whose anchors resolve WITHIN their segment.
    segments: Optional[tuple] = None
    # What a sample IS. "real" (the default, and every record before complex baseband existed) is
    # a voltage. "iq" is COMPLEX BASEBAND: each sample is I + jQ, the envelope of a signal centred
    # on `fc_hz`, and `waveform()` returns complex128. This is the domain the chain STARTS in;
    # `upconvert` / `downconvert` move between the two mid-chain, and every op states which it
    # accepts (`wfmsynth.domain`) and refuses the other by name.
    domain: str = "real"
    fc_hz: Optional[float] = None      # the centre frequency an "iq" record is the envelope of

    def __post_init__(self):
        if self.domain not in ("real", "iq"):
            raise ValueError(f"Grid: unknown domain {self.domain!r} (use 'real' or 'iq')")
        if self.domain == "real" and self.fc_hz is not None:
            raise ValueError(
                "Grid: fc_hz is the centre frequency of a complex baseband record and means "
                "nothing on a real one -- pass domain='iq' with it, or drop it. (A real "
                "passband record carries its carrier in its samples.)")
        if self.fc_hz is not None and not self.fc_hz > 0:
            raise ValueError(f"Grid: fc_hz must be positive, got {self.fc_hz!r}")

    def resolve(self, anchor):
        """Resolve a symbolic time-anchor to an absolute sample index. `anchor` is a dict with one
        of: ``sample`` | ``t`` (seconds) | ``frac`` (of the record) | ``ui`` (single-segment; needs
        baud) | ``{"segment": name, "bit": k}`` (piecewise; resolves within that segment). The one
        resolver both lowering and rendering call."""
        if "sample" in anchor:
            return int(round(anchor["sample"]))
        if "t" in anchor:
            return int(round(anchor["t"] * self.fs))
        if "frac" in anchor:
            return int(round(anchor["frac"] * self.n))
        if "segment" in anchor:
            if not self.segments:
                raise ValueError("a 'segment' anchor needs a segmented grid")
            off = 0.0
            for name, baud, n_sym in self.segments:
                if name == anchor["segment"]:
                    return int(round(off + anchor.get("bit", 0) * (self.fs / baud)))
                off += n_sym * (self.fs / baud)
            raise ValueError(f"unknown segment {anchor['segment']!r}")
        if "ui" in anchor:
            if self.samples_per_ui is None:
                raise ValueError("a 'ui' anchor needs baud set on the grid")
            return int(round(anchor["ui"] * self.samples_per_ui))
        raise ValueError(f"unknown anchor spec {anchor!r} (use sample/t/frac/ui/segment)")

    def segment_bounds(self):
        """(name, start_sample, stop_sample) for each segment; empty if single-segment."""
        if not self.segments:
            return []
        out, off = [], 0.0
        for name, baud, n_sym in self.segments:
            width = n_sym * (self.fs / baud)
            out.append((name, int(round(off)), int(round(off + width))))
            off += width
        return out

    # ---- derived quantities ----
    @property
    def dt(self) -> float:             # sample period [s]
        return 1.0 / self.fs

    @property
    def duration(self) -> float:       # record length [s]
        return self.n / self.fs

    @property
    def f_nyquist(self) -> float:      # [Hz]
        return 0.5 * self.fs

    @property
    def ui_seconds(self) -> Optional[float]:      # unit interval [s]
        return None if self.baud is None else 1.0 / self.baud

    @property
    def samples_per_ui(self) -> Optional[float]:  # samples per symbol (usually non-integer)
        return None if self.baud is None else self.fs / self.baud

    # ---- conversions into the abstract-grid units the primitives use ----
    def to_samples(self, seconds: float) -> float:
        """seconds -> samples (may be fractional)."""
        return seconds * self.fs

    def to_frac(self, seconds: float) -> float:
        """seconds -> fraction of the record."""
        return seconds / self.duration

    def hz_to_frac_nyquist(self, hz: float) -> float:
        """Hz -> fraction of Nyquist (0..1), e.g. for a corner frequency."""
        return hz / self.f_nyquist

    def hz_to_cycles_per_record(self, hz: float) -> float:
        """Hz -> cycles per record, e.g. for a periodic-jitter tone."""
        return hz * self.duration

    def pattern_period_samples(self, n_symbols: int) -> float:
        """Exact (fractional) sample period of a repeating symbol pattern of length
        `n_symbols` UI. Because samples-per-UI is generally non-integer, this is NOT an
        integer — anything that folds/averages repetitions must realign to this
        fractional period sub-sample, or it drifts. Requires baud to be set."""
        if self.baud is None:
            raise ValueError("pattern_period_samples requires baud to be set")
        return n_symbols * self.samples_per_ui

    def volts(self, x):
        """Scale a normalized (+/-1) waveform to volts at this grid's full scale."""
        return x * (0.5 * self.v_full)
