import pytest

from wfmsynth import physics


@pytest.fixture
def zero_edges():
    """Pad a linear stage with zeros past the record, as every stage did before the ends were
    held (#100). For a test that reads an impulse response off an impulse at index 0, or checks a
    stage against `np.convolve`, which is zero-padded by definition."""
    with physics.zero_edges():
        yield
