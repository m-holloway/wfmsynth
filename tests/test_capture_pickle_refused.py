"""A capture file is DATA, never code.

`.npy` and `.npz` can carry a pickled object array, and unpickling one runs arbitrary code. The
whole point of `Signal.capture()` is reading a file the user did not necessarily produce -- a
capture handed over by a colleague, pulled off an instrument, or fetched from a dataset -- so
this is the one place in the library where a hostile input is a realistic scenario rather than
a hypothetical.

numpy has defaulted `allow_pickle=False` since 1.16.3, so this was never open. These tests
exist so it cannot QUIETLY become open again: a future numpy default, a refactor that reaches
for `np.load` without the flag, or a well-meaning "allow_pickle=True fixes the error" would all
be caught here rather than shipped.
"""
from __future__ import annotations

import os

import numpy as np
import pytest

from wfmsynth import capture


class _Payload:
    """Reduces to a shell command. If this ever runs, the load path executed code."""

    def __reduce__(self):
        return (os.system, ("echo wfmsynth-pickle-executed",))


def _write_pickled_npy(path):
    np.save(path, np.array([_Payload()], dtype=object), allow_pickle=True)
    return str(path)


def _write_pickled_npz(path):
    np.savez(path, values=np.array([_Payload()], dtype=object))
    return str(path)


def test_a_pickled_npy_is_refused(tmp_path):
    p = _write_pickled_npy(tmp_path / "evil.npy")
    with pytest.raises(ValueError, match="PICKLED"):
        capture.load_values(p)


def test_a_pickled_npz_is_refused(tmp_path):
    p = _write_pickled_npz(tmp_path / "evil.npz")
    with pytest.raises(ValueError, match="PICKLED"):
        capture.load_values(p)


def test_the_refusal_names_the_file_and_says_what_to_do(tmp_path):
    """A raw numpy error ("Object arrays cannot be loaded when allow_pickle=False") is accurate
    and useless to a caller of this library: it names neither the file nor the way out."""
    p = _write_pickled_npy(tmp_path / "evil.npy")
    with pytest.raises(ValueError) as exc:
        capture.load_values(p)
    msg = str(exc.value)
    assert "evil.npy" in msg
    assert "np.save" in msg                       # the way to produce a loadable file
    assert "executes code" in msg                 # why it is refused, not just that it is


def test_an_ordinary_capture_still_loads(tmp_path):
    """The bar this must not cross: refusing pickles cannot cost the normal path."""
    want = np.linspace(-1.0, 1.0, 128)
    p = tmp_path / "ok.npy"
    np.save(p, want)
    assert np.array_equal(capture.load_values(str(p)), want)

    pz = tmp_path / "ok.npz"
    np.savez(pz, values=want)
    assert np.array_equal(capture.load_values(str(pz)), want)


def test_a_non_pickle_value_error_is_not_swallowed(tmp_path):
    """The handler keys on the pickle error specifically and must re-raise anything else --
    otherwise a corrupt file would be reported as a pickle it is not."""
    p = tmp_path / "corrupt.npy"
    p.write_bytes(b"\x93NUMPY\x01\x00not-a-real-header")
    with pytest.raises(Exception) as exc:
        capture.load_values(str(p))
    assert "PICKLED" not in str(exc.value)
