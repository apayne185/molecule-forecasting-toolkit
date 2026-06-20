FEATURES = [
    'year', 'month', 'quarter', 'week_of_year',
    'Value_Lag1', 'Packs_Lag1',
    'Value_RollingMean_3', 'Packs_RollingMean_3',
    'Value_RollingMean_6', 'Packs_RollingMean_6',
    'MoleculeName', 'TradeName', 'ProductName',
]
CAT_FEATURES = ['MoleculeName', 'TradeName', 'ProductName']
GROUP_COLS   = ['MoleculeName', 'TradeName', 'ProductName']

TARGET_VALUE = 'Value'
TARGET_PACKS = 'Packs'

DATA_SPLIT_ROW    = 4260  # rows 0–4259 are historical (Dec 2017 – Dec 2020)
START_ROW_EXCEL   = 4261  # 1-indexed Excel row where forecast range starts
MAX_FORECAST_ROWS = 1339
