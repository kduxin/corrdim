import numpy as np
import pytest

from corrdim.dimension import auto_linear_region_bounds, estimate_dimension_from_curve, estimate_dimension_from_curves
from corrdim.types import CurveResult


def _make_power_curve(sequence_length: int, epsilons: np.ndarray, dim: float) -> CurveResult:
    # corrints = eps^dim in correlation-integral space; ensures a nearly perfect linear fit in log10-log10.
    corrints = epsilons**dim
    return CurveResult(sequence_length=sequence_length, epsilons=epsilons, corrints=corrints)


def test_auto_linear_region_bounds_selects_expected_range():
    n = 150
    eps = np.array([1e-6, 1e-5, 1e-4], dtype=np.float64)
    # Should fall into the second (low, high*10) range for n=150.
    corrints = np.array([1e-2, 2e-2, 3e-2], dtype=np.float64)

    low, high = auto_linear_region_bounds(sequence_length=n, epsilons=eps, corrints=corrints)

    expected_low = 20.0 / n / (n - 1)
    expected_high = 1.0 / n * 10
    assert low == pytest.approx(expected_low)
    assert high == pytest.approx(expected_high)


def test_estimate_dimension_defaults_enable_auto_bounds():
    n = 200
    eps = np.logspace(-3, -2, 5, dtype=np.float64)
    dim = 1.75
    curve = _make_power_curve(sequence_length=n, epsilons=eps, dim=dim)

    res = estimate_dimension_from_curve(curve)
    assert res.linear_region_bounds != (None, None)
    assert res.epsilons_linear_region.shape[0] >= 2
    assert res.corrints_linear_region.shape[0] >= 2
    assert np.isfinite(res.corrdim)


def test_estimate_dimension_explicit_epsilon_range_disables_auto_bounds():
    n = 200
    eps = np.logspace(-3, -2, 5, dtype=np.float64)
    dim = 1.75
    curve = _make_power_curve(sequence_length=n, epsilons=eps, dim=dim)

    res = estimate_dimension_from_curve(curve, correlation_integral_range=None, epsilon_range=(1e-3, 1e-2))
    assert res.linear_region_bounds == (None, None)
    assert res.epsilons_linear_region.tolist() == curve.epsilons[1:4].tolist()
    assert res.corrints_linear_region.tolist() == curve.corrints[1:4].tolist()
    assert np.isfinite(res.corrdim)


def test_estimate_dimension_epsilon_range_filters_points():
    n = 200
    eps = np.array([1e-4, 2e-4, 3e-4, 4e-4], dtype=np.float64)
    dim = 2.0
    curve = _make_power_curve(sequence_length=n, epsilons=eps, dim=dim)

    res = estimate_dimension_from_curve(
        curve,
        correlation_integral_range=None,
        # strict bounds: (2e-4, 4e-4) would include only 3e-4,
        # but the implementation requires at least 2 points and falls back.
        # Use a range that keeps at least two points under strict filtering.
        epsilon_range=(1.5e-4, 4e-4),  # strict: includes 2e-4 and 3e-4
    )
    assert res.epsilons_linear_region.tolist() == [2e-4, 3e-4]
    assert res.corrints_linear_region.tolist() == [curve.corrints[1], curve.corrints[2]]


def test_estimate_dimension_fallback_to_full_curve_on_too_narrow_range():
    n = 200
    eps = np.array([1e-4, 2e-4, 3e-4, 4e-4], dtype=np.float64)
    dim = 1.25
    curve = _make_power_curve(sequence_length=n, epsilons=eps, dim=dim)

    # Pick an interval that excludes most points; should fallback to the full curve
    # (as long as at least 2 positive finite points exist).
    res = estimate_dimension_from_curve(curve, correlation_integral_range=(0.9, 0.95))
    assert res.linear_region_bounds == (0.9, 0.95)
    assert res.epsilons_linear_region.shape[0] >= 2


def test_estimate_dimension_raises_when_not_enough_positive_finite_points():
    n = 200
    eps = np.logspace(-3, -2, 4, dtype=np.float64)
    # Only one positive finite corrint; others are <=0 or non-finite.
    corrints = np.array([1e-3, 0.0, -1.0, np.nan], dtype=np.float64)
    curve = CurveResult(sequence_length=n, epsilons=eps, corrints=corrints)

    with pytest.raises(ValueError, match="Not enough points with positive finite correlation integral"):
        _ = estimate_dimension_from_curve(curve, correlation_integral_range=(0.0, 0.01))


def test_estimate_dimension_from_curves_maps_over_list():
    n = 200
    eps = np.logspace(-3, -2, 4, dtype=np.float64)
    curve1 = _make_power_curve(n, eps, dim=1.0)
    curve2 = _make_power_curve(n, eps, dim=2.0)

    out = estimate_dimension_from_curves([curve1, curve2], correlation_integral_range=None)
    assert len(out) == 2
    assert out[0].sequence_length == n
    assert out[1].sequence_length == n
    assert out[0].linear_region_bounds != (None, None)
    assert out[1].linear_region_bounds != (None, None)
    assert out[0].corrints_linear_region.shape[0] >= 2
    assert out[1].corrints_linear_region.shape[0] >= 2

