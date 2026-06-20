import numpy as np
import pandas as pd
import pytest

from src.train import evaluate


class _ConstantModel:
    """Stub that always predicts the same scalar value."""
    def __init__(self, value: float):
        self.value = value

    def predict(self, X):
        return np.full(len(X), self.value)


class _EchoModel:
    """Stub that echoes a pre-set array of predictions."""
    def __init__(self, preds):
        self.preds = np.asarray(preds)

    def predict(self, X):
        return self.preds


_DUMMY_X = pd.DataFrame({'a': [0, 0, 0]})


class TestEvaluate:
    def test_perfect_predictions_give_zero_rmse(self):
        y = pd.Series([10.0, 20.0, 30.0])
        model = _EchoModel(y.values)
        rmse, preds = evaluate(model, _DUMMY_X, y)
        assert rmse == pytest.approx(0.0)

    def test_constant_offset_gives_correct_rmse(self):
        y = pd.Series([0.0, 0.0, 0.0])
        model = _ConstantModel(5.0)
        rmse, preds = evaluate(model, _DUMMY_X, y)
        assert rmse == pytest.approx(5.0)

    def test_returns_predictions_array(self):
        y = pd.Series([1.0, 2.0, 3.0])
        model = _ConstantModel(1.0)
        _, preds = evaluate(model, _DUMMY_X, y)
        assert len(preds) == len(y)

    def test_rmse_is_float(self):
        y = pd.Series([1.0, 2.0])
        model = _ConstantModel(0.0)
        rmse, _ = evaluate(model, pd.DataFrame({'a': [0, 0]}), y)
        assert isinstance(rmse, float)

    def test_known_rmse_value(self):
        # RMSE of [0,0,0] vs [3,4,0] = sqrt((9+16+0)/3) = sqrt(25/3)
        y = pd.Series([0.0, 0.0, 0.0])
        model = _EchoModel([3.0, 4.0, 0.0])
        rmse, _ = evaluate(model, _DUMMY_X, y)
        assert rmse == pytest.approx(np.sqrt(25.0 / 3.0))
