# Pharmaceutical Sales Forecasting

A time-series forecasting pipeline for predicting monthly pharmaceutical product sales (revenue and unit volume) across 143 SKUs, 66 brands, and 27 molecules.

## Problem

Given 36 months of historical sales data (December 2017 – December 2020) for a portfolio of pharmaceutical products, forecast `Value` (revenue) and `Packs` (unit volume) for January 2021 across all active SKU–brand–molecule combinations.

## Data

| Attribute | Detail |
|---|---|
| Historical records | 4,260 (Dec 2017 – Dec 2020) |
| Forecast horizon | 1,339 rows (Jan 2021) |
| Products | 143 SKUs × 66 brands × 27 molecules |
| Targets | `Value` (revenue, €), `Packs` (unit volume) |
| Value range | €0.04 – €3.45M per record |
| Packs range | 3 – 452,364 units per record |

## Approach

### Feature Engineering

- **Temporal:** month, quarter, ISO week-of-year
- **Rolling statistics:** 3-month and 6-month rolling means per product (molecule × brand × SKU)
- **Lag features:** 1-month lag values per product series
- **Categorical encoding:** one-hot encoding of molecule, brand, and SKU identity (243 binary features)
- **Validation strategy:** time-series-aware cross-validation (`TimeSeriesSplit`, 5 folds) — no future data leaks into training splits

### Model Comparison

| Model | Value RMSE | Packs RMSE | Notes |
|---|---|---|---|
| **XGBoost** (Optuna-tuned) | **79,782** | **10,325** | Best overall; 50-trial Bayesian HPO |
| Random Forest | 99,582 | 12,276 | GridSearchCV; slower training |
| SARIMA(1,1,1)(1,1,1,12) | 345,487 | 63,852 | Per-aggregate series; loses product structure |
| Prophet | 345,488 | — | Bayesian structural; same structural limitation |
| ARIMA(1,1,1) | 305,582 | — | Baseline; univariate only |
| Linear Regression | 309,038 | — | Baseline; no temporal patterns |

XGBoost outperformed all alternatives by a significant margin. Classical time-series models (SARIMA, Prophet, ARIMA) struggled because this is a **panel dataset** — 143 concurrent product series — not a single aggregate series. A global gradient boosting model with product-identity encodings captures cross-series patterns and product-specific momentum simultaneously.

### XGBoost Final Hyperparameters (from Optuna, 50 trials)

| Parameter | Value target | Packs target |
|---|---|---|
| `n_estimators` | 384 | 200 |
| `learning_rate` | 0.0105 | 0.1 |
| `max_depth` | 9 | 5 |
| `subsample` | 0.75 | 0.8 |
| `colsample_bytree` | 0.945 | 0.8 |

## Notebooks

| Notebook | Contents |
|---|---|
| [01_eda.ipynb](notebooks/01_eda.ipynb) | EDA: distributions, seasonality, ACF/PACF, product coverage |
| [02_model_comparison.ipynb](notebooks/02_model_comparison.ipynb) | Baseline comparison of 6 models; justifies XGBoost selection |
| [03_xgboost.ipynb](notebooks/03_xgboost.ipynb) | Optuna hyperparameter tuning, final training, prediction, Excel export |

Run in order: `01` → `02` → `03` → `04` → `05`

## Setup

```bash
pip install -r requirements.txt
```

`03_xgboost.ipynb` and `04_lightgbm.ipynb` each read `test_data_working_students.xlsx` and writes predictions to `forecasted_results.xlsx`.

## Key Findings

- **Product-level rolling means** (3-month window per SKU) are the dominant predictive features — product momentum outweighs calendar seasonality for this dataset
- **SKU identity** (`ProductName` one-hot features) is the strongest categorical predictor, confirming that product-level historical behaviour is more informative than molecule or brand groupings alone
- **Classical time-series models underperform** because the task is inherently multi-series (panel data): SARIMA and Prophet treat the aggregated dataset as a single series and lose the per-product structure that tree models exploit naturally
- **Optuna vs. GridSearchCV:** Bayesian optimisation (Optuna, 50 trials) achieved lower MSE than GridSearchCV (100+ combinations) in roughly the same wall time, validating the use of TPE sampling over exhaustive search for this parameter space
