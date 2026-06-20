import numpy as np
import optuna
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error

from .features import CAT_FEATURES

optuna.logging.set_verbosity(optuna.logging.WARNING)


def _lgb_objective(trial, X, y) -> float:
    params = {
        'objective':         'regression',
        'metric':            'rmse',
        'verbosity':         -1,
        'num_leaves':        trial.suggest_int('num_leaves', 20, 300),
        'learning_rate':     trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
        'n_estimators':      trial.suggest_int('n_estimators', 100, 1000),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'subsample':         trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree':  trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_alpha':         trial.suggest_float('reg_alpha', 0.0, 1.0),
        'reg_lambda':        trial.suggest_float('reg_lambda', 0.0, 1.0),
    }
    tscv = TimeSeriesSplit(n_splits=5)
    scores = []
    for train_idx, val_idx in tscv.split(X):
        model = lgb.LGBMRegressor(**params)
        model.fit(X.iloc[train_idx], y.iloc[train_idx], categorical_feature=CAT_FEATURES)
        scores.append(mean_squared_error(y.iloc[val_idx], model.predict(X.iloc[val_idx])))
    return float(np.mean(scores))


def _xgb_objective(trial, X, y) -> float:
    params = {
        'objective':        'reg:squarederror',
        'n_estimators':     trial.suggest_int('n_estimators', 100, 500),
        'learning_rate':    trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
        'max_depth':        trial.suggest_int('max_depth', 3, 10),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'subsample':        trial.suggest_float('subsample', 0.7, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 1.0),
    }
    tscv = TimeSeriesSplit(n_splits=5)
    scores = []
    for train_idx, val_idx in tscv.split(X):
        model = xgb.XGBRegressor(**params)
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        scores.append(mean_squared_error(y.iloc[val_idx], model.predict(X.iloc[val_idx])))
    return float(np.mean(scores))


_OBJECTIVES = {'lgb': _lgb_objective, 'xgb': _xgb_objective}


def tune(
    model_type: str,
    X,
    y,
    n_trials: int = 50,
    seed: int = 42,
) -> dict:
    """Run Optuna TPE search and return best hyperparameters.

    Args:
        model_type: 'lgb' or 'xgb'
        X: feature DataFrame
        y: target Series
        n_trials: number of Optuna trials
        seed: random seed for reproducibility
    """
    if model_type not in _OBJECTIVES:
        raise ValueError(f"model_type must be one of {list(_OBJECTIVES)}, got '{model_type}'")

    study = optuna.create_study(
        direction='minimize',
        sampler=optuna.samplers.TPESampler(seed=seed),
    )
    study.optimize(
        lambda trial: _OBJECTIVES[model_type](trial, X, y),
        n_trials=n_trials,
        show_progress_bar=True,
    )
    return study.best_params


def train_lgb(X, y, params: dict) -> lgb.LGBMRegressor:
    """Train LightGBM on full dataset with the given hyperparameters."""
    model = lgb.LGBMRegressor(**params, objective='regression', verbosity=-1)
    model.fit(X, y, categorical_feature=CAT_FEATURES)
    return model


def train_xgb(X, y, params: dict) -> xgb.XGBRegressor:
    """Train XGBoost on full dataset with the given hyperparameters."""
    model = xgb.XGBRegressor(**params, objective='reg:squarederror')
    model.fit(X, y)
    return model


def evaluate(model, X_test, y_test) -> tuple[float, np.ndarray]:
    """Return (RMSE, predictions) on a held-out test set."""
    preds = model.predict(X_test)
    rmse  = float(np.sqrt(mean_squared_error(y_test, preds)))
    return rmse, preds
