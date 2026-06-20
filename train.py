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
)

DATA_PATH     = Path('data/test_data_working_students.xlsx')
OUTPUT_PATH   = Path('data/forecasted_results_pipeline.xlsx')
TEST_FRACTION = 0.2


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Molecule sales forecasting pipeline')
    p.add_argument('--data',   default=str(DATA_PATH),   help='Path to input Excel file')
    p.add_argument('--output', default=str(OUTPUT_PATH), help='Path for output Excel file')
    p.add_argument('--model',  default='lgb', choices=['lgb', 'xgb'], help='Model type')
    p.add_argument('--trials', default=50, type=int, help='Optuna trials per target')
    p.add_argument('--seed',      default=42,      type=int, help='Random seed')
    p.add_argument('--model-dir', default='models', help='Directory for saved models (default: models/)')
    return p.parse_args()


def main() -> None:
    args = parse_args()
    t0 = time.time()

    # ── 1. Load ───────────────────────────────────────────────────────────────
    print(f"\n[1/5] Loading data from {args.data}")
    historical, future = load_data(args.data)
    print(f"      Historical: {len(historical):,} rows  |  Future: {len(future):,} rows")

    # ── 2. Feature engineering ────────────────────────────────────────────────
    print("\n[2/5] Engineering features")
    historical = engineer_features(historical)
    future_proc = get_future_features(historical, future)

    split_idx = int(len(historical) * (1 - TEST_FRACTION))
    X       = historical[FEATURES]
    y_value = historical[TARGET_VALUE]
    y_packs = historical[TARGET_PACKS]

    X_train, X_test           = X.iloc[:split_idx],         X.iloc[split_idx:]
    y_train_value, y_test_value = y_value.iloc[:split_idx], y_value.iloc[split_idx:]
    y_train_packs, y_test_packs = y_packs.iloc[:split_idx], y_packs.iloc[split_idx:]
    print(f"      Train: {len(X_train):,}  |  Test: {len(X_test):,}  |  Features: {len(FEATURES)}")

    # ── 3. Hyperparameter tuning ──────────────────────────────────────────────
    print(f"\n[3/5] Tuning {args.model.upper()} — {args.trials} trials per target")

    print("      → Value target")
    best_params_value = tune(args.model, X, y_value, n_trials=args.trials, seed=args.seed)

    print("      → Packs target")
    best_params_packs = tune(args.model, X, y_packs, n_trials=args.trials, seed=args.seed)

    # ── 4. Train & evaluate ───────────────────────────────────────────────────
    print("\n[4/5] Training final models on full historical dataset")

    train_fn = train_lgb if args.model == 'lgb' else train_xgb
    model_value = train_fn(X, y_value, best_params_value)
    model_packs = train_fn(X, y_packs, best_params_packs)

    rmse_value, _ = evaluate(model_value, X_test, y_test_value)
    rmse_packs, _ = evaluate(model_packs, X_test, y_test_packs)
    print(f"      Hold-out RMSE — Value: {rmse_value:>12,.0f}")
    print(f"      Hold-out RMSE — Packs: {rmse_packs:>12,.0f}")

    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model_value, model_dir / f'{args.model}_value.pkl')
    joblib.dump(model_packs, model_dir / f'{args.model}_packs.pkl')
    print(f"      Saved  models → {model_dir}/{args.model}_value.pkl, {model_dir}/{args.model}_packs.pkl")

    # ── 5. Forecast & save ────────────────────────────────────────────────────
    print(f"\n[5/5] Generating forecasts and saving to {args.output}")
    predictions = generate_forecasts(model_value, model_packs, future_proc)
    save_to_excel(predictions, template_path=args.data, output_path=args.output)

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s")
    print(f"  Model : {args.model.upper()} | Trials: {args.trials} | Seed: {args.seed}")
    print(f"  Output: {args.output}")


if __name__ == '__main__':
    main()
