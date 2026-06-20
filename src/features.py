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

DATA_SPLIT_ROW    = 4260
FORECAST_YEAR     = 2021
START_ROW_EXCEL   = 4261
MAX_FORECAST_ROWS = 1339
