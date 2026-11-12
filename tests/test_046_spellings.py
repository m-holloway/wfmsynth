"""0.46.0's own parameter names follow the unit-suffix convention before it is released.

Three were out of line: `downconvert(fs_out=)` was a rate with no unit in its name,
`awgn(ebn0_db=)` was spelled unlike its sibling `es_n0_db`, and `downconvert(lpf=20e6)` was a
frequency in a parameter that also takes a spec. None was ever in a release, so there is no
deprecation path: the old spelling is refused naming the new one, and the new one is checked
to do what the old one did.
"""
import warnings

import numpy as np
import pytest

from wfmsynth import Grid, Signal

FS, BAUD, FC, N = 8e9, 25e6, 1e9, 1 << 16
G = Grid(fs=FS, baud=BAUD, n=N, fc_hz=FC)


def _src():
    return Signal(seed=1, grid=G).iq("qam16", prbs=15, prbs_phase=100)


def _rf():
    return _src().upconvert()


@pytest.mark.parametrize("call,names", [
    (lambda: _rf().downconvert(lpf=dict(pass_hz=20e6), fs_out=250e6), "fs_out_hz="),
    (lambda: _rf().downconvert(lpf=dict(pass_hz=20e6), fs_hz=250e6), "fs_out_hz="),
    (lambda: _rf().downconvert(lpf_hz=20e6), "lpf=dict(pass_hz="),
    (lambda: _src().awgn(ebn0_db=8.0, bits_per_symbol=4), "eb_n0_db="),
    (lambda: _src().awgn(esn0_db=8.0), "es_n0_db="),
])
def test_the_old_spelling_is_refused_naming_the_new_one(call, names):
    with pytest.raises((TypeError, ValueError)) as e:
        call()
    assert names in str(e.value)


def test_a_bare_number_for_the_low_pass_is_refused_with_the_spec_that_means_it():
    with pytest.raises(ValueError, match=r"lpf=2e\+07 is a bare number.*"
                                         r"lpf=dict\(pass_hz=2e\+07\)"):
        _rf().downconvert(lpf=20e6)
    # in a hand-written recipe too
    r = _rf().downconvert(lpf=dict(pass_hz=20e6)).recipe()
    r["ops"][-1]["lpf"] = 20e6
    with pytest.raises(ValueError, match="bare number"):
        Signal.from_recipe(r).waveform()


def test_pass_hz_alone_is_the_filter_the_number_used_to_mean():
    """Stopband from twice the passband edge, 80 dB down: the same record as stating both."""
    a = _rf().downconvert(lpf=dict(pass_hz=20e6)).waveform()
    b = _rf().downconvert(lpf=dict(pass_hz=20e6, stop_hz=40e6, atten_db=80.0)).waveform()
    assert np.array_equal(a, b)


def test_fs_out_hz_is_the_stored_rate():
    s = _rf().downconvert(lpf=dict(pass_hz=20e6, stop_hz=60e6), fs_out_hz=250e6)
    x = s.waveform()
    g = s.stored_grid(x)
    assert (g.fs, len(x), g.domain) == (250e6, N // 32, "iq")
    assert s.recipe()["ops"][-1]["fs_out_hz"] == 250e6


def test_eb_n0_is_es_n0_less_the_bits_per_symbol():
    """Eb/N0 = Es/N0 - 10 log10(bits per symbol): QAM-16 at Eb/N0 8 dB is Es/N0 14.0206 dB,
    the same noise draw scaled the same way."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        per_bit = _src().awgn(eb_n0_db=8.0, ref_power=1.0)
        per_sym = _src().awgn(es_n0_db=8.0 + 10 * np.log10(4), ref_power=1.0)
        assert per_bit.ops[-1]["bits_per_symbol"] == 4            # taken from the iq source
        assert np.allclose(per_bit.waveform(), per_sym.waveform(), rtol=0, atol=1e-12)
        rep = per_bit.rf_report()[0]
    assert rep["es_n0_db"] == pytest.approx(14.0206, abs=1e-4)      # what the request came to
