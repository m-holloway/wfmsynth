"""The dependency floor is numpy 2.0 (0.46.1): a complex record replays to the byte only there.

pip enforces the floor on an install; a source tree on PYTHONPATH is checked at import.
"""
import re
import pathlib

import numpy as np
import pytest

import wfmsynth

ROOT = pathlib.Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("version", ["1.21.6", "1.26.4", "1.99.0rc1"])
def test_a_numpy_below_the_floor_is_named_with_what_breaks_and_the_fix(version):
    msg = wfmsynth._numpy_floor(version)
    assert version in msg and "complex record" in msg and "numpy>=2.0" in msg


@pytest.mark.parametrize("version", ["2.0.0", "2.0.2", "2.4.2", "3.0.0.dev0", "not-a-version"])
def test_a_numpy_at_or_above_the_floor_is_silent(version):
    assert wfmsynth._numpy_floor(version) is None


def test_the_running_numpy_meets_the_floor_the_package_declares():
    deps = re.search(r'^dependencies = \[(.*)\]', (ROOT / "pyproject.toml").read_text(), re.M).group(1)
    floor = re.search(r'numpy>=(\d+)\.(\d+)', deps)
    assert floor and (int(floor.group(1)), int(floor.group(2))) == (2, 0)
    assert tuple(int(p) for p in np.__version__.split(".")[:2]) >= (2, 0)
    assert wfmsynth._numpy_floor(np.__version__) is None
