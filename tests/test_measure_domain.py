"""A measure handed the wrong kind of record refuses by name, and says which measure is right.

`measure` and `eye` fold, threshold and cluster a real voltage. Handed a complex baseband
record they cast it to float -- numpy drops Q under a ComplexWarning -- and returned a
plausible figure for the I rail alone, or (for `eye_height`) raised from inside numpy three
frames down. The envelope measures in `rf` had the mirror-image guard already; both directions
are pinned here, one row per public function.
"""
import warnings

import numpy as np
import pytest

import wfmsynth as ws
from wfmsynth import Grid, Signal, eye, measure, rf

G = Grid(fs=400e6, baud=25e6, n=1 << 14, fc_hz=100e6)
SRC = Signal(seed=1, grid=G).iq("qpsk", prbs=9)
Z = SRC.waveform()                                   # complex baseband
V = Signal(seed=1, grid=Grid(fs=400e6, baud=25e6, n=1 << 14)).carrier(
    "nrz", n_ui=1024).waveform()                     # a real voltage
RG = Grid(fs=400e6, baud=25e6, n=1 << 14)
RF_NAMES = ("rf.evm", "rf.constellation_density", "upconvert(fc_hz=", "x.real")

REAL_MEASURES = {
    "sample_at_phase": lambda x: measure.sample_at_phase(x, 16.0, 3.0),
    "eye_height": lambda x: measure.eye_height(x, RG, levels=2),
    "best_phase": lambda x: measure.best_phase(x, RG, levels=2),
    "align_symbols": lambda x: measure.align_symbols(x, RG, np.ones(64), levels=2),
    "ground_truth": lambda x: measure.ground_truth(x, RG, levels=2),
    "attributes": lambda x: measure.attributes(x, RG, levels=2),
    "eye.levels": lambda x: eye.levels(x),
    "eye.crossings": lambda x: eye.crossings(x, 0.0, 0.1),
    "eye.recover": lambda x: eye.recover(x, 400e6, 16.0),
    "eye.density": lambda x: eye.density(x, np.arange(8, 16000, 16.0), 16.0),
    "eye_density": lambda x: ws.eye_density(x, RG),
}


@pytest.mark.parametrize("name", sorted(REAL_MEASURES))
def test_a_voltage_measure_refuses_an_iq_record_and_names_what_measures_one(name):
    with warnings.catch_warnings():
        warnings.simplefilter("error")               # the refusal comes before any cast warns
        with pytest.raises(ValueError) as e:
            REAL_MEASURES[name](Z)
    msg = str(e.value)
    assert msg.startswith(f"{name}: the record is complex baseband (I/Q)")
    assert all(word in msg for word in RF_NAMES)


@pytest.mark.parametrize("name", sorted(REAL_MEASURES))
def test_the_same_measure_takes_a_voltage_and_one_rail_stated(name):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        REAL_MEASURES[name](V)                       # a real record: measured
        REAL_MEASURES[name](Z.real)                  # one rail, said on purpose: measured


def test_a_complex_array_with_no_imaginary_part_is_still_an_envelope():
    """The dtype says what the record is; an envelope whose Q happens to be zero is not a
    voltage, and guessing would make the refusal depend on the data."""
    with pytest.raises(ValueError, match="complex baseband"):
        measure.eye_height(V.astype(complex), RG, levels=2)


def test_pattern_period_refuses_complex_symbols():
    with pytest.raises(ValueError, match=r"pattern_period: the symbols are complex.*"
                                         r"symbols\.real"):
        measure.pattern_period(SRC.transmitted()[0])
    assert measure.pattern_period(np.tile([1.0, -1.0, 1.0, 1.0], 32))[0] % 4 == 0


def test_signal_ground_truth_on_an_iq_chain_is_refused_the_same_way():
    with pytest.raises(ValueError, match=r"ground_truth: the record is complex baseband"):
        SRC.ground_truth(levels=2)


ENVELOPE_MEASURES = {
    "evm": lambda x: rf.evm(x, RG, sent=np.ones(8, complex)),
    "ccdf": lambda x: rf.ccdf(x),
    "constellation_density": lambda x: rf.constellation_density(x),
    "demodulate": lambda x: ws.iq.demodulate(x, RG),
}


@pytest.mark.parametrize("name", sorted(ENVELOPE_MEASURES))
def test_an_envelope_measure_refuses_a_voltage_and_names_the_converter(name):
    with pytest.raises(ValueError) as e:
        ENVELOPE_MEASURES[name](V)
    msg = str(e.value)
    assert msg.startswith(name) and ("downconvert(" in msg or "complex" in msg)


def test_the_measures_defined_on_both_take_both():
    """A spectrum and a channel power mean the same on a voltage and on an envelope (one-sided
    from 0, or two-sided about the centre), so they refuse neither."""
    for x, g in ((V, RG), (Z, G)):
        sp = rf.spectrum(x, g, rbw_hz=1e6)
        assert np.isclose(np.sum(sp["psd_per_hz"]) * sp["bin_hz"], np.mean(np.abs(x) ** 2),
                          rtol=0.05)
    assert rf.aclr(V, RG, 25e6, [50e6], center_hz=12.5e6)["channel_power"] > 0
    assert rf.aclr(Z, G, 25e6, [50e6])["channel_power"] > 0


def test_the_route_the_refusal_names_gives_the_figure():
    """`upconvert` and then the voltage measure; and `rf.evm` on the envelope itself."""
    y = Signal(seed=1, grid=G).iq("qpsk", prbs=9).upconvert().waveform()
    assert y.dtype == np.float64 and np.isfinite(measure.attributes(y, RG, levels=2)["rms"])
    assert rf.evm(Z, G, SRC.transmitted()[0], skip=8)["evm"] < 0.01
