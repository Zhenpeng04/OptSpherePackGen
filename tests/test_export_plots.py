import numpy as np
import pytest

from spherepackgen.exporters.files import _gaussian_smooth


def test_gaussian_smooth_preserves_length_and_constant_values():
    values = np.ones(21, dtype=float) * 2.5

    smoothed = _gaussian_smooth(values, sigma=3.0)

    assert smoothed.shape == values.shape
    assert smoothed == pytest.approx(values)


def test_gaussian_smooth_reduces_isolated_spike_without_shifting_peak():
    values = np.zeros(21, dtype=float)
    values[10] = 1.0

    smoothed = _gaussian_smooth(values, sigma=2.0)

    assert smoothed[10] < values[10]
    assert int(np.argmax(smoothed)) == 10
    assert np.sum(smoothed) == pytest.approx(np.sum(values), rel=1.0e-3)


def test_gaussian_smooth_keeps_short_series_unchanged():
    values = np.array([1.0, 2.0])

    smoothed = _gaussian_smooth(values, sigma=3.0)

    assert smoothed == pytest.approx(values)


def test_gaussian_smooth_fills_isolated_nonfinite_values_for_plotting():
    values = np.array([0.0, np.nan, 2.0, 2.0, 2.0])

    smoothed = _gaussian_smooth(values, sigma=1.0)

    assert np.all(np.isfinite(smoothed))
