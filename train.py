#!/usr/bin/env python3
"""
End-to-end training and forecasting pipeline.

Usage:
    python train.py                         # default paths, LightGBM, 50 trials
    python train.py --model xgb             # use XGBoost instead
    python train.py --trials 20             # fewer Optuna trials (faster)
    python train.py --data data/test_data_working_students.xlsx
"""
import argparse
import logging
import time
from pathlib import Path

import joblib

from src import (
    load_data,
    engineer_features,
    get_future_features,
    FEATURES,
    TARGET_VALUE,
    TARGET_PACKS,
    tune,
    train_lgb,
    train_xgb,
    evaluate,
    generate_forecasts,
    save_to_excel,
    train_quantile_bounds,
    prediction_interval,
    coverage,
    mean_interval_width,
)

DATA_PATH     = Path('data/test_data_working_students.xlsx')
OUTPUT_PATH   = Path('data/forecasted_results_pipeline.xlsx')
TEST_FRACTION = 0.2

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)-8s  %(message)s',
    datefmt='%H:%M:%S',
)
log = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Molecule sales forecasting pipeline')
    p.add_argument('--data',      default=str(DATA_PATH),   help='Path to input Excel file')
    p.add_argument('--output',    default=str(OUTPUT_PATH), help='Path for output Excel file')
    p.add_argument('--model',     default='lgb', choices=['lgb', 'xgb'], help='Model type')
    p.add_argument('--trials',    default=50, type=int, help='Optuna trials per target')
    p.add_argument('--seed',      default=42,  type=int, help='Random seed')
    p.add_argument('--model-dir', default='models', help='Directory for saved models (default: models/)')
    p.add_argument('--intervals', action='store_true',
                   help='Also train q10/q90 quantile models and report hold-out coverage')
    return p.parse_args()


def main() -> None:
    args = parse_args()
    t0 = time.time()

    # ── 1. Load ───────────────────────────────────────────────────────────────
    log.info('[1/5] Loading data from %s', args.data)
    historical, future = load_data(args.data)
    log.info('      Historical: %s rows  |  Future: %s rows',
             f'{len(historical):,}', f'{len(future):,}')

    # ── 2. Feature engineering ────────────────────────────────────────────────
    log.info('[2/5] Engineering features')
    historical = engineer_features(historical)
    future_proc = get_future_features(historical, future)

    split_idx = int(len(historical) * (1 - TEST_FRACTION))
    X       = historical[FEATURES]
    y_value = historical[TARGET_VALUE]
    y_packs = historical[TARGET_PACKS]

    X_train, X_test             = X.iloc[:split_idx],         X.iloc[split_idx:]
    y_train_value, y_test_value = y_value.iloc[:split_idx],   y_value.iloc[split_idx:]
    y_train_packs, y_test_packs = y_packs.iloc[:split_idx],   y_packs.iloc[split_idx:]
    log.info('      Train: %s  |  Test: %s  |  Features: %d',
             f'{len(X_train):,}', f'{len(X_test):,}', len(FEATURES))

    # ── 3. Hyperparameter tuning ──────────────────────────────────────────────
    log.info('[3/5] Tuning %s — %d trials per target', args.model.upper(), args.trials)

    log.info('      → Value target')
    best_params_value = tune(args.model, X, y_value, n_trials=args.trials, seed=args.seed)

    log.info('      → Packs target')
    best_params_packs = tune(args.model, X, y_packs, n_trials=args.trials, seed=args.seed)

    # ── 4. Train & evaluate ───────────────────────────────────────────────────
    log.info('[4/5] Training final models on full historical dataset')

    train_fn = train_lgb if args.model == 'lgb' else train_xgb
    model_value = train_fn(X, y_value, best_params_value)
    model_packs = train_fn(X, y_packs, best_params_packs)

    rmse_value, _ = evaluate(model_value, X_test, y_test_value)
    rmse_packs, _ = evaluate(model_packs, X_test, y_test_packs)
    log.info('      Hold-out RMSE — Value: %s', f'{rmse_value:>12,.0f}')
    log.info('      Hold-out RMSE — Packs: %s', f'{rmse_packs:>12,.0f}')

    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model_value, model_dir / f'{args.model}_value.pkl')
    joblib.dump(model_packs, model_dir / f'{args.model}_packs.pkl')
    log.info('      Saved  models → %s/%s_value.pkl, %s/%s_packs.pkl',
             model_dir, args.model, model_dir, args.model)

    if args.intervals:
        log.info('      Training 10th/90th percentile quantile models')
        lower_v, upper_v = train_quantile_bounds(X, y_value, best_params_value, model_type=args.model)
        lower_p, upper_p = train_quantile_bounds(X, y_packs,  best_params_packs,  model_type=args.model)

        lb_v, ub_v = prediction_interval(lower_v, upper_v, X_test)
        lb_p, ub_p = prediction_interval(lower_p, upper_p, X_test)

        cov_v = coverage(y_test_value, lb_v, ub_v)
        cov_p = coverage(y_test_packs, lb_p, ub_p)
        width_v = mean_interval_width(lb_v, ub_v)
        width_p = mean_interval_width(lb_p, ub_p)
        log.info('      Interval coverage — Value: %.1f%%  Packs: %.1f%%', cov_v * 100, cov_p * 100)
        log.info('      Mean width       — Value: %s  Packs: %s',
                 f'{width_v:,.0f}', f'{width_p:,.0f}')

        joblib.dump(lower_v, model_dir / f'{args.model}_value_q10.pkl')
        joblib.dump(upper_v, model_dir / f'{args.model}_value_q90.pkl')
        joblib.dump(lower_p, model_dir / f'{args.model}_packs_q10.pkl')
        joblib.dump(upper_p, model_dir / f'{args.model}_packs_q90.pkl')
        log.info('      Saved  quantile models → %s/', model_dir)

    # ── 5. Forecast & save ────────────────────────────────────────────────────
    log.info('[5/5] Generating forecasts → %s', args.output)
    predictions = generate_forecasts(model_value, model_packs, future_proc)
    save_to_excel(predictions, template_path=args.data, output_path=args.output)

    elapsed = time.time() - t0
    log.info('Done in %.1fs  |  Model: %s  |  Trials: %d  |  Seed: %d',
             elapsed, args.model.upper(), args.trials, args.seed)


if __name__ == '__main__':
    main()
