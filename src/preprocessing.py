from pathlib import Path

import pandas as pd

from .features import GROUP_COLS, DATA_SPLIT_ROW


def load_data(path: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load Excel workbook and split into historical and future DataFrames."""
    df = pd.read_excel(path)
    historical = df.iloc[:DATA_SPLIT_ROW].dropna().copy()
    future     = df.iloc[DATA_SPLIT_ROW:].copy()
    return historical, future


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add temporal, lag, and rolling mean features. Sets categorical dtypes."""
    df = df.copy()
    df['date']         = pd.to_datetime(df[['year', 'month']].assign(day=1))
    df['quarter']      = df['date'].dt.quarter
    df['week_of_year'] = df['date'].dt.isocalendar().week.astype(int)

    for col in GROUP_COLS:
        df[col] = df[col].astype('category')

    df['Value_Lag1'] = df.groupby(GROUP_COLS, observed=True)['Value'].shift(1)
    df['Packs_Lag1'] = df.groupby(GROUP_COLS, observed=True)['Packs'].shift(1)

    for window in (3, 6):
        df[f'Value_RollingMean_{window}'] = (
            df.groupby(GROUP_COLS, observed=True)['Value']
            .transform(lambda x: x.rolling(window, min_periods=1).mean())
        )
        df[f'Packs_RollingMean_{window}'] = (
            df.groupby(GROUP_COLS, observed=True)['Packs']
            .transform(lambda x: x.rolling(window, min_periods=1).mean())
        )

    return df


def get_future_features(
    historical: pd.DataFrame,
    future: pd.DataFrame,
) -> pd.DataFrame:
    """Build feature matrix for the forecast period.

    Lag and rolling values are derived from the tail of each product's
    historical series — not global means — so every product starts from
    its own last-known state.
    """
    lag_cols = [
        'Value_Lag1', 'Packs_Lag1',
        'Value_RollingMean_3', 'Packs_RollingMean_3',
        'Value_RollingMean_6', 'Packs_RollingMean_6',
    ]

    hist_sorted = historical.sort_values(['year', 'month'])

    product_stats = (
        hist_sorted
        .groupby(GROUP_COLS, observed=True)
        .apply(lambda g: pd.Series({
            'Value_Lag1':          g['Value'].iloc[-1],
            'Packs_Lag1':          g['Packs'].iloc[-1],
            'Value_RollingMean_3': g['Value'].tail(3).mean(),
            'Packs_RollingMean_3': g['Packs'].tail(3).mean(),
            'Value_RollingMean_6': g['Value'].tail(6).mean(),
            'Packs_RollingMean_6': g['Packs'].tail(6).mean(),
        }), include_groups=False)
        .reset_index()
    )

    future_proc = future.copy()
    future_proc['date']         = pd.to_datetime(future_proc[['year', 'month']].assign(day=1))
    future_proc['quarter']      = future_proc['date'].dt.quarter
    future_proc['week_of_year'] = future_proc['date'].dt.isocalendar().week.astype(int)

    for col in GROUP_COLS:
        future_proc[col] = future_proc[col].astype('category')

    future_proc = future_proc.merge(product_stats, on=GROUP_COLS, how='left')

    for col in lag_cols:
        if future_proc[col].isna().any():
            future_proc[col] = future_proc[col].fillna(historical[col].mean())

    return future_proc
