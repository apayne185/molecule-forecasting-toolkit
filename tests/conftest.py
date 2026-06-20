import numpy as np
import pandas as pd
import pytest

_PRODUCTS = [
    ('MoleculeA', 'TradeA', 'ProductA'),
    ('MoleculeB', 'TradeB', 'ProductB'),
    ('MoleculeC', 'TradeC', 'ProductC'),
]
_RNG = np.random.default_rng(42)


@pytest.fixture(scope='session')
def historical():
    """36 months × 3 products of synthetic sales data (108 rows)."""
    records = []
    for mol, trade, prod in _PRODUCTS:
        for year in [2018, 2019, 2020]:
            for month in range(1, 13):
                records.append({
                    'year': year,
                    'month': month,
                    'Value': float(_RNG.uniform(1_000, 50_000)),
                    'Packs': int(_RNG.integers(10, 500)),
                    'MoleculeName': mol,
                    'TradeName': trade,
                    'ProductName': prod,
                })
    return pd.DataFrame(records)


@pytest.fixture(scope='session')
def future():
    """One Jan-2021 forecast row per product (3 rows)."""
    records = []
    for mol, trade, prod in _PRODUCTS:
        records.append({
            'year': 2021,
            'month': 1,
            'Value': np.nan,
            'Packs': np.nan,
            'MoleculeName': mol,
            'TradeName': trade,
            'ProductName': prod,
        })
    return pd.DataFrame(records)
