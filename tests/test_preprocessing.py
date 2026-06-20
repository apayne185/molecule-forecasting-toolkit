from unittest.mock import patch

import pandas as pd
import pytest

from src.features import CAT_FEATURES, FEATURES, GROUP_COLS
from src.preprocessing import engineer_features, get_future_features, load_data


class TestLoadData:
    def test_raises_on_missing_required_columns(self):
        incomplete = pd.DataFrame({'year': [2020], 'month': [1]})
        with patch('src.preprocessing.pd.read_excel', return_value=incomplete):
            with pytest.raises(ValueError, match='missing required columns'):
                load_data('dummy.xlsx')

    def test_error_names_the_missing_column(self):
        incomplete = pd.DataFrame({'year': [2020], 'month': [1]})
        with patch('src.preprocessing.pd.read_excel', return_value=incomplete):
            with pytest.raises(ValueError, match='Packs'):
                load_data('dummy.xlsx')


class TestEngineerFeatures:
    def test_adds_temporal_columns(self, historical):
        result = engineer_features(historical)
        for col in ('quarter', 'week_of_year', 'date'):
            assert col in result.columns

    def test_adds_lag_and_rolling_columns(self, historical):
        result = engineer_features(historical)
        for col in ('Value_Lag1', 'Packs_Lag1',
                    'Value_RollingMean_3', 'Packs_RollingMean_3',
                    'Value_RollingMean_6', 'Packs_RollingMean_6'):
            assert col in result.columns

    def test_categorical_dtypes_set(self, historical):
        result = engineer_features(historical)
        for col in CAT_FEATURES:
            assert result[col].dtype.name == 'category', f'{col} should be category dtype'

    def test_preserves_row_count(self, historical):
        result = engineer_features(historical)
        assert len(result) == len(historical)

    def test_first_row_per_product_has_nan_lag(self, historical):
        result = engineer_features(historical)
        first_rows = result.groupby(GROUP_COLS, observed=True).head(1)
        assert first_rows['Value_Lag1'].isna().all()
        assert first_rows['Packs_Lag1'].isna().all()

    def test_rolling_mean_non_negative(self, historical):
        result = engineer_features(historical)
        assert (result['Value_RollingMean_3'] >= 0).all()
        assert (result['Packs_RollingMean_3'] >= 0).all()

    def test_does_not_mutate_input(self, historical):
        original_cols = list(historical.columns)
        engineer_features(historical)
        assert list(historical.columns) == original_cols


@pytest.fixture(scope='module')
def future_proc(historical, future):
    hist_eng = engineer_features(historical)
    return get_future_features(hist_eng, future)


class TestGetFutureFeatures:
    def test_has_all_feature_columns(self, future_proc):
        for col in FEATURES:
            assert col in future_proc.columns, f'Missing feature column: {col}'

    def test_no_nan_lag_for_known_products(self, future_proc):
        lag_cols = [
            'Value_Lag1', 'Packs_Lag1',
            'Value_RollingMean_3', 'Packs_RollingMean_3',
            'Value_RollingMean_6', 'Packs_RollingMean_6',
        ]
        for col in lag_cols:
            assert not future_proc[col].isna().any(), f'NaN in {col}'

    def test_row_count_matches_future(self, future, future_proc):
        assert len(future_proc) == len(future)

    def test_year_and_month_correct(self, future_proc):
        assert (future_proc['year'] == 2021).all()
        assert (future_proc['month'] == 1).all()

    def test_categorical_dtypes_set(self, future_proc):
        for col in CAT_FEATURES:
            assert future_proc[col].dtype.name == 'category', f'{col} should be category dtype'
