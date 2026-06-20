import numpy as np
import lightgbm as lgb
import xgboost as xgb

from .features import CAT_FEATURES


def train_quantile_bounds(
    X,
    y,
    params: dict,
    lower_q: float = 0.1,
    upper_q: float = 0.9,
    model_type: str = 'lgb',
) -> tuple:
    """Train lower and upper quantile models for prediction intervals.

    Returns:
        (model_lower, model_upper) — predict lower/upper bounds of the interval.
    """
    if model_type == 'lgb':
        base = {k: v for k, v in params.items() if k not in ('objective', 'metric')}
        model_lower = lgb.LGBMRegressor(**base, objective='quantile', alpha=lower_q, verbosity=-1)
        model_lower.fit(X, y, categorical_feature=CAT_FEATURES)
        model_upper = lgb.LGBMRegressor(**base, objective='quantile', alpha=upper_q, verbosity=-1)
        model_upper.fit(X, y, categorical_feature=CAT_FEATURES)
    elif model_type == 'xgb':
        base = {k: v for k, v in params.items()
                if k not in ('objective', 'enable_categorical')}
        model_lower = xgb.XGBRegressor(
            **base, objective='reg:quantileerror', quantile_alpha=lower_q, enable_categorical=True
        )
        model_lower.fit(X, y)
        model_upper = xgb.XGBRegressor(
            **base, objective='reg:quantileerror', quantile_alpha=upper_q, enable_categorical=True
        )
        model_upper.fit(X, y)
    else:
        raise ValueError(f"model_type must be 'lgb' or 'xgb', got '{model_type}'")

    return model_lower, model_upper


def prediction_interval(model_lower, model_upper, X) -> tuple[np.ndarray, np.ndarray]:
    """Return (lower_bound, upper_bound) prediction arrays."""
    return model_lower.predict(X), model_upper.predict(X)


def coverage(y_true, lower: np.ndarray, upper: np.ndarray) -> float:
    """Fraction of true values falling within [lower, upper]."""
    y = np.asarray(y_true)
    return float(np.mean((y >= lower) & (y <= upper)))


def mean_interval_width(lower: np.ndarray, upper: np.ndarray) -> float:
    """Mean absolute width of prediction intervals."""
    return float(np.mean(upper - lower))
