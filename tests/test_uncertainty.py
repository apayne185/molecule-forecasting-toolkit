import numpy as np
import pytest

from src.uncertainty import coverage, mean_interval_width, prediction_interval


class _ConstantModel:
    def __init__(self, value: float):
        self.value = value

    def predict(self, X):
        return np.full(len(X), self.value)


_N = 10
_X_DUMMY = np.zeros((_N, 1))


class TestCoverage:
    def test_perfect_coverage(self):
        y = np.array([5.0] * _N)
        lower = np.array([4.0] * _N)
        upper = np.array([6.0] * _N)
        assert coverage(y, lower, upper) == pytest.approx(1.0)

    def test_zero_coverage(self):
        y = np.array([0.0] * _N)
        lower = np.array([1.0] * _N)
        upper = np.array([2.0] * _N)
        assert coverage(y, lower, upper) == pytest.approx(0.0)

    def test_partial_coverage(self):
        y = np.array([0.0, 1.5, 3.0, 1.5, 0.0])
        lower = np.array([1.0] * 5)
        upper = np.array([2.0] * 5)
        # only indices 1 and 3 are within [1, 2]
        assert coverage(y, lower, upper) == pytest.approx(2 / 5)

    def test_boundary_values_included(self):
        y = np.array([1.0, 2.0])
        lower = np.array([1.0, 2.0])
        upper = np.array([1.0, 2.0])
        assert coverage(y, lower, upper) == pytest.approx(1.0)

    def test_returns_float(self):
        y = np.array([1.0])
        assert isinstance(coverage(y, np.array([0.0]), np.array([2.0])), float)

    def test_accepts_pandas_series(self):
        import pandas as pd
        y = pd.Series([1.0, 1.5, 2.0])
        lower = np.array([1.0, 1.0, 1.0])
        upper = np.array([2.0, 2.0, 2.0])
        assert coverage(y, lower, upper) == pytest.approx(1.0)


class TestMeanIntervalWidth:
    def test_uniform_width(self):
        lower = np.zeros(_N)
        upper = np.full(_N, 10.0)
        assert mean_interval_width(lower, upper) == pytest.approx(10.0)

    def test_zero_width(self):
        arr = np.ones(_N)
        assert mean_interval_width(arr, arr) == pytest.approx(0.0)

    def test_mixed_widths(self):
        lower = np.array([0.0, 0.0])
        upper = np.array([2.0, 4.0])
        assert mean_interval_width(lower, upper) == pytest.approx(3.0)

    def test_returns_float(self):
        result = mean_interval_width(np.zeros(3), np.ones(3))
        assert isinstance(result, float)


class TestPredictionInterval:
    def test_output_shapes(self):
        model_lower = _ConstantModel(1.0)
        model_upper = _ConstantModel(9.0)
        lb, ub = prediction_interval(model_lower, model_upper, _X_DUMMY)
        assert lb.shape == (_N,)
        assert ub.shape == (_N,)

    def test_lower_values(self):
        lb, _ = prediction_interval(_ConstantModel(1.0), _ConstantModel(9.0), _X_DUMMY)
        assert (lb == 1.0).all()

    def test_upper_values(self):
        _, ub = prediction_interval(_ConstantModel(1.0), _ConstantModel(9.0), _X_DUMMY)
        assert (ub == 9.0).all()
