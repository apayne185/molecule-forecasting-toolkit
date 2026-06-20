import numpy as np
import pytest

from src.preprocessing import engineer_features, get_future_features
from src.predict import generate_forecasts


class _ConstantModel:
    def __init__(self, value: float):
        self.value = value

    def predict(self, X):
        return np.full(len(X), self.value)


@pytest.fixture(scope='module')
def future_proc(historical, future):
    hist_eng = engineer_features(historical)
    return get_future_features(hist_eng, future)


class TestGenerateForecasts:
    def test_output_has_predicted_columns(self, future_proc):
        result = generate_forecasts(_ConstantModel(100.0), _ConstantModel(50.0), future_proc)
        assert 'Predicted_Value' in result.columns
        assert 'Predicted_Packs' in result.columns

    def test_output_length_matches_input(self, future_proc):
        result = generate_forecasts(_ConstantModel(100.0), _ConstantModel(50.0), future_proc)
        assert len(result) == len(future_proc)

    def test_predicted_values_match_model_output(self, future_proc):
        result = generate_forecasts(_ConstantModel(999.0), _ConstantModel(888.0), future_proc)
        assert (result['Predicted_Value'] == 999.0).all()
        assert (result['Predicted_Packs'] == 888.0).all()

    def test_original_columns_preserved(self, future_proc):
        result = generate_forecasts(_ConstantModel(1.0), _ConstantModel(1.0), future_proc)
        for col in future_proc.columns:
            assert col in result.columns

    def test_predictions_are_non_negative(self, future_proc):
        result = generate_forecasts(_ConstantModel(0.0), _ConstantModel(0.0), future_proc)
        assert (result['Predicted_Value'] >= 0).all()
        assert (result['Predicted_Packs'] >= 0).all()
