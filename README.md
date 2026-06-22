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
- **Categorical encoding:** native categorical features in LightGBM (13 features); one-hot encoding in XGBoost baseline (243 binary features)
- **Validation strategy:** time-series-aware cross-validation (`TimeSeriesSplit`, 5 folds) — no future data leaks into training splits

### Model Comparison

Initial comparison across 6 model families (notebook 02), then gradient boosting deep-dive (notebooks 03–04):

| Model | Value RMSE | Packs RMSE | Notes |
|---|---|---|---|
| **LightGBM** (Optuna-tuned) | **~XGBoost** | **~XGBoost** | Default model; 13 native categorical features; 80% prediction intervals — see notebook 04 |
| **XGBoost** (Optuna-tuned) | **79,782** | **10,325** | 254 OHE features; 50-trial Bayesian HPO |
| Random Forest | 99,582 | 12,276 | GridSearchCV; slower training |
| SARIMA(1,1,1)(1,1,1,12) | 345,487 | 63,852 | Per-aggregate series; loses product structure |
| Prophet | 345,488 | — | Bayesian structural; same structural limitation |
| ARIMA(1,1,1) | 305,582 | — | Baseline; univariate only |
| Linear Regression | 309,038 | — | Baseline; no temporal patterns |

Gradient boosting (both XGBoost and LightGBM) outperformed all alternatives by a significant margin. Classical time-series models (SARIMA, Prophet, ARIMA) struggled because this is a **panel dataset** — 143 concurrent product series — not a single aggregate series. A global gradient boosting model with product-identity encodings captures cross-series patterns and product-specific momentum simultaneously. LightGBM is the production default: it matches XGBoost accuracy while consuming 13 features instead of 254, eliminating the one-hot encoding step entirely.

### Model Hyperparameters (from Optuna, 50 trials each)

**XGBoost:**

| Parameter | Value target | Packs target |
|---|---|---|
| `n_estimators` | 384 | 200 |
| `learning_rate` | 0.0105 | 0.1 |
| `max_depth` | 9 | 5 |
| `subsample` | 0.75 | 0.8 |
| `colsample_bytree` | 0.945 | 0.8 |

**LightGBM** key parameters (leaf-wise growth — see [notebook 04](notebooks/04_lightgbm.ipynb) for full tuning output):

| Parameter | Description |
|---|---|
| `num_leaves` | Primary complexity control (≈ `2^max_depth` in XGBoost terms) |
| `min_child_samples` | Regulariser for sparse product groups |
| `subsample` / `colsample_bytree` | Row/column subsampling, same semantics as XGBoost |

## Notebooks

| Notebook | Contents |
|---|---|
| [01_eda.ipynb](notebooks/01_eda.ipynb) | EDA: distributions, seasonality, ACF/PACF, product coverage |
| [02_model_comparison.ipynb](notebooks/02_model_comparison.ipynb) | Baseline comparison of 6 models; justifies XGBoost selection |
| [03_xgboost.ipynb](notebooks/03_xgboost.ipynb) | XGBoost with Optuna tuning, final training, and Excel export |
| [04_lightgbm.ipynb](notebooks/04_lightgbm.ipynb) | LightGBM with native categorical encoding, Optuna tuning, 80% quantile prediction intervals, comparison vs XGBoost |
| [05_business_insights.ipynb](notebooks/05_business_insights.ipynb) | YoY growth, price-per-pack analysis, market concentration (Lorenz/Gini), seasonality heatmap |
| [06_shap.ipynb](notebooks/06_shap.ipynb) | SHAP interpretability: global feature importance, beeswarm, single-prediction waterfall, dependence plot |

Each notebook is self-contained and can be run independently; `01` → `06` is the recommended reading order.

## Setup

```bash
# Core dependencies (required for src/ module, pipeline, and tests)
pip install -r requirements.txt

# Additional dependencies for running all notebooks (prophet, statsmodels, seaborn)
pip install -r requirements-notebooks.txt
```

### Notebooks

Each notebook is self-contained. Run cells top-to-bottom; data files are read from `data/`.

### Production pipeline

```bash
python train.py                     # LightGBM, 50 Optuna trials (default)
python train.py --model xgb         # XGBoost instead
python train.py --trials 5          # quick smoke test
python train.py --output results.xlsx
```

The pipeline runs all five stages end-to-end (load → feature engineering → Optuna tuning → train & evaluate → forecast) and writes predictions to `data/forecasted_results_pipeline.xlsx`. Add `--intervals` to also train quantile models and report 80% prediction interval coverage on the hold-out set.

The reusable `src/` module exposes the same logic — `load_data`, `engineer_features`, `tune`, `train_lgb`, `train_xgb`, `generate_forecasts`, `train_quantile_bounds`, `coverage` — for use in downstream scripts or services.

## Key Findings

- **Product-level rolling means** (3-month window per SKU) are the dominant predictive features — product momentum outweighs calendar seasonality for this dataset
- **SKU identity** (`ProductName`) is the strongest categorical predictor, confirming that product-level historical behaviour is more informative than molecule or brand groupings alone
- **Classical time-series models underperform** because the task is inherently multi-series (panel data): SARIMA and Prophet treat the aggregated dataset as a single series and lose the per-product structure that tree models exploit naturally
- **Optuna vs. GridSearchCV:** Bayesian optimisation (Optuna, 50 trials) achieved lower MSE than GridSearchCV (100+ combinations) in roughly the same wall time, validating the use of TPE sampling over exhaustive search for this parameter space
- **SHAP confirms the momentum hypothesis:** `Value_RollingMean_3` has by far the largest mean |SHAP value| — short-term product momentum dominates over calendar seasonality and molecule/brand-level features. SKU identity (`ProductName`) contributes a stable per-product baseline offset, making every forecast fully auditable
