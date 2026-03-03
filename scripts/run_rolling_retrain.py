#!/usr/bin/env python3
"""Unified Rolling Retrain: expanding window, from scratch.

Per-series models (expanding window, 1-step ahead):
  seasonal_naive, ets_damped, theta
LightGBM at checkpoints [12, 18, 23]

Usage:
  python scripts/run_rolling_retrain.py --direction inbound
  python scripts/run_rolling_retrain.py --direction outbound
"""
from __future__ import annotations

import argparse
import logging
import os
import time
import warnings
from pathlib import Path

logging.disable(logging.CRITICAL)
os.environ["CMDSTAN_VERBOSE"] = "false"

import numpy as np
import pandas as pd

from config import (
    CALL_TYPES,
    grain_cols_for, target_for,
)
from models import (
    forecast_seasonal_naive_1step, forecast_ets_damped_1step,
    forecast_theta_1step, wape_single,
    build_lgbm_features, LGBM_BASE_FEATURES, LGBM_TREND_FEATURES,
)

ROOT = Path(__file__).resolve().parent.parent
LGBM_CHECKPOINTS = [12, 18, 23]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Unified rolling retrain evaluation.")
    p.add_argument("--direction", required=True, choices=["inbound", "outbound"])
    return p.parse_args()


def run_per_series_rolling(df: pd.DataFrame, grain_cols: list[str],
                           direction: str) -> list[dict]:
    """Run expanding-window 1-step-ahead evaluation for per-series models."""
    rows = []

    grouped = {k: g.sort_values("CALL_YEAR_MONTH") for k, g in df.groupby(grain_cols)}
    series_keys = sorted(grouped.keys())
    n = len(series_keys)
    t0 = time.time()

    models = ["ets_damped", "theta"]

    for idx, key in enumerate(series_keys):
        series = grouped[key]
        call_type = key[grain_cols.index("CALL_TYPE")]
        target = target_for(call_type, direction)

        values = series[target].values.astype(float)
        months = series["CALL_YEAR_MONTH"].values

        if len(values) < 13:
            continue

        base = {c: key[i] for i, c in enumerate(grain_cols)}

        # Expanding window: train on [0..t-1], predict t
        for t in range(12, len(values)):
            train_vals = values[:t]
            actual = values[t]
            forecast_month = int(months[t])
            train_size = t

            for model_name in models:
                if model_name == "ets_damped":
                    pred = forecast_ets_damped_1step(train_vals)
                elif model_name == "theta":
                    pred = forecast_theta_1step(train_vals)
                else:
                    continue

                used = model_name
                if pred is None:
                    pred = forecast_seasonal_naive_1step(train_vals)
                    used = f"{model_name}_fb"

                rows.append({
                    **base,
                    "target": target, "model": used,
                    "forecast_month": forecast_month,
                    "train_size": train_size,
                    "month_num": forecast_month % 100 if forecast_month > 9999 else (t % 12) + 1,
                    "actual": actual,
                    "predicted": pred,
                    "ape": wape_single(actual, pred),
                })

        if (idx + 1) % 200 == 0:
            el = time.time() - t0
            r = (idx + 1) / el
            print(f"  [per-series] {idx+1}/{n} ({el:.0f}s, {r:.1f}/s, ETA {(n-idx-1)/r:.0f}s)", flush=True)

    el = time.time() - t0
    print(f"  [per-series] done: {n} series, {len(rows)} rows in {el:.0f}s", flush=True)
    return rows


def run_lgbm_checkpoints(df: pd.DataFrame, grain_cols: list[str],
                         direction: str) -> list[dict]:
    """Run LightGBM at training-size checkpoints."""
    try:
        import lightgbm as lgb
    except ImportError:
        print("  LightGBM not available, skipping", flush=True)
        return []

    rows = []
    all_months = sorted(df["CALL_YEAR_MONTH"].unique())
    feature_cols = list(LGBM_BASE_FEATURES) + LGBM_TREND_FEATURES

    for ct in CALL_TYPES:
        ct_mask = df["CALL_TYPE"] == ct
        ct_df = df[ct_mask].sort_values(grain_cols + ["CALL_YEAR_MONTH"]).reset_index(drop=True)
        target = target_for(ct, direction)

        feat_df = build_lgbm_features(ct_df, target, grain_cols=grain_cols, add_trend=True)

        for cp in LGBM_CHECKPOINTS:
            if cp >= len(all_months):
                continue
            train_months_set = set(all_months[:cp])
            test_month = all_months[cp]

            train_mask = feat_df["CALL_YEAR_MONTH"].isin(train_months_set)
            test_mask = feat_df["CALL_YEAR_MONTH"] == test_month

            train_feat = feat_df[train_mask].dropna(subset=feature_cols)
            test_feat = feat_df[test_mask].copy()

            if len(train_feat) < 100 or len(test_feat) == 0:
                continue

            for col in feature_cols:
                test_feat[col] = test_feat[col].fillna(train_feat[col].median())

            try:
                dtrain = lgb.Dataset(train_feat[feature_cols], train_feat["y"], free_raw_data=False)
                params = {
                    "objective": "regression", "metric": "mae", "num_leaves": 31,
                    "learning_rate": 0.05, "feature_fraction": 0.8,
                    "bagging_fraction": 0.8, "bagging_freq": 5, "verbose": -1, "n_jobs": -1,
                }
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model = lgb.train(params, dtrain, num_boost_round=300)

                preds = np.clip(model.predict(test_feat[feature_cols]), 0, None)
                test_feat["pred"] = preds

                for _, row in test_feat.iterrows():
                    actual = float(row["y"])
                    predicted = float(row["pred"])
                    base = {c: row[c] for c in grain_cols}
                    rows.append({
                        **base,
                        "target": target, "model": "lgbm",
                        "forecast_month": int(test_month),
                        "train_size": cp,
                        "month_num": int(test_month) % 100 if int(test_month) > 9999 else (cp % 12) + 1,
                        "actual": actual,
                        "predicted": predicted,
                        "ape": wape_single(actual, predicted),
                    })
            except Exception as e:
                print(f"  lgbm checkpoint {cp} error for {ct}/{target}: {e}", flush=True)

        print(f"  lgbm done for {ct}/{target}", flush=True)

    return rows


def main():
    args = parse_args()
    direction = args.direction
    grain_cols = grain_cols_for(direction)
    reports = ROOT / "reports"
    reports.mkdir(parents=True, exist_ok=True)

    input_path = ROOT / "data" / f"{direction}_set.parquet"
    print(f"Direction: {direction}", flush=True)
    print(f"Grain: {grain_cols}", flush=True)
    print(f"Input: {input_path}", flush=True)

    df = pd.read_parquet(input_path)
    df = df.sort_values(grain_cols + ["CALL_YEAR_MONTH"]).reset_index(drop=True)

    all_months = sorted(df["CALL_YEAR_MONTH"].unique())
    series_count = df.groupby(grain_cols).ngroups
    print(f"Series: {series_count}, Months: {len(all_months)}", flush=True)
    for ct in CALL_TYPES:
        ct_n = df[df["CALL_TYPE"] == ct].groupby(grain_cols).ngroups
        print(f"  {ct}: {ct_n} series -> target: {target_for(ct, direction)}", flush=True)
    print(f"Per-series models: ets_damped, theta", flush=True)
    print(f"LightGBM checkpoints: {LGBM_CHECKPOINTS}", flush=True)

    # Per-series rolling
    print(f"\n=== Per-series expanding window ===", flush=True)
    t0 = time.time()
    per_series_rows = run_per_series_rolling(df, grain_cols, direction)

    # LightGBM checkpoints
    print(f"\n=== LightGBM checkpoints ===", flush=True)
    lgbm_rows = run_lgbm_checkpoints(df, grain_cols, direction)

    all_rows = per_series_rows + lgbm_rows
    result_df = pd.DataFrame(all_rows)

    output_path = reports / f"{direction}_rolling_accuracy.csv"
    result_df.to_csv(output_path, index=False)
    print(f"\nOutput: {output_path} ({len(result_df)} rows)", flush=True)
    print(f"Total time: {time.time()-t0:.0f}s", flush=True)

    # Summary
    print("\nAverage APE by model:", flush=True)
    print(result_df.groupby("model")["ape"].mean().sort_values().to_string(), flush=True)

    if "train_size" in result_df.columns:
        print("\nAverage APE by model and training size:", flush=True)
        summary = result_df.groupby(["model", "train_size"])["ape"].mean().unstack(level=0)
        print(summary.to_string(), flush=True)


if __name__ == "__main__":
    main()
