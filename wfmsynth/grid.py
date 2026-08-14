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

    def volts(self, x):
        """Scale a normalized (+/-1) waveform to volts at this grid's full scale."""
        return x * (0.5 * self.v_full)
