import numpy as np
import pandas as pd
import pytest

from src.features import FEATURES
from src.uncertainty import (
    coverage,
    mean_interval_width,
    prediction_interval,
    train_quantile_bounds,
)


def _make_feature_df(n: int = 20) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    months = [i % 12 + 1 for i in range(n)]
    df = pd.DataFrame({
        'year':                [2020] * n,
        'month':               months,
        'quarter':             [(m - 1) // 3 + 1 for m in months],
        'week_of_year':        [1] * n,
        'Value_Lag1':          rng.uniform(1000, 5000, n),
        'Packs_Lag1':          rng.uniform(100, 500, n),
        'Value_RollingMean_3': rng.uniform(1000, 5000, n),
        'Packs_RollingMean_3': rng.uniform(100, 500, n),
        'Value_RollingMean_6': rng.uniform(1000, 5000, n),
        'Packs_RollingMean_6': rng.uniform(100, 500, n),
        'MoleculeName':        pd.Categorical(['MolA', 'MolB'] * (n // 2)),
        'TradeName':           pd.Categorical(['TradeX', 'TradeY'] * (n // 2)),
        'ProductName':         pd.Categorical(['Prod1', 'Prod2'] * (n // 2)),
    })
    return df[FEATURES]


_LGB_PARAMS = {'n_estimators': 5, 'num_leaves': 7}
_Y_QR = np.random.default_rng(99).uniform(1000, 5000, 20)


class TestTrainQuantileBounds:
    def test_returns_two_models_lgb(self):
        X = _make_feature_df()
        lower, upper = train_quantile_bounds(X, _Y_QR, _LGB_PARAMS, model_type='lgb')
        assert lower is not None and upper is not None

    def test_predict_shape_lgb(self):
        X = _make_feature_df()
        lower, upper = train_quantile_bounds(X, _Y_QR, _LGB_PARAMS, model_type='lgb')
        lb, ub = lower.predict(X), upper.predict(X)
        assert lb.shape == (20,) and ub.shape == (20,)

    def test_lower_q_below_upper_q_on_average(self):
        X = _make_feature_df()
        lower, upper = train_quantile_bounds(
            X, _Y_QR, _LGB_PARAMS, lower_q=0.1, upper_q=0.9, model_type='lgb'
        )
        lb, ub = lower.predict(X), upper.predict(X)
        assert np.mean(lb) < np.mean(ub)

    def test_invalid_model_type_raises(self):
        X = _make_feature_df()
        with pytest.raises(ValueError, match='model_type'):
            train_quantile_bounds(X, _Y_QR, _LGB_PARAMS, model_type='invalid')


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
