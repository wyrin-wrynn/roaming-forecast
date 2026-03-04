"""Generate 2026 forecasts (202601-202612) using best models from horserace.

Reads each direction's horserace_winners.csv to determine the best model per
series, then produces 12-month-ahead forecasts. Per-series statistical models
run directly; LightGBM uses an iterative 1-step-ahead approach so that each
month's prediction feeds into the next month's lag features.

Output: reports/{direction}_forecast_2026.csv
"""
from __future__ import annotations

import argparse
import logging
import os
import warnings
from pathlib import Path

# Suppress noisy logging from dependencies before any imports trigger it
logging.disable(logging.CRITICAL)
os.environ["CMDSTAN_VERBOSE"] = "false"

import numpy as np
import pandas as pd

from config import DASHBOARD_CALL_TYPES, grain_cols_for, target_for
from models import (
    build_lgbm_features,
    forecast_ets,
    forecast_sarima,
    forecast_seasonal_naive,
    forecast_theta,
    LGBM_BASE_FEATURES,
    LGBM_TREND_FEATURES,
)

ROOT = Path(__file__).resolve().parent.parent

FORECAST_MONTHS = list(range(202601, 202613))
# Statistical models forecast from end-of-training (Nov 2025), so position 0 = Dec 2025.
# We use horizon=13 and skip fc[0] to get Jan–Dec 2026.
STAT_HORIZON = 13

# Mapping from model name (stripping _fb suffix) to forecast function
MODEL_FN = {
    "seasonal_naive": forecast_seasonal_naive,
    "ets_damped": lambda train, h: forecast_ets(train, h, damped=True, trend="add"),
    "sarima": forecast_sarima,
    "theta": forecast_theta,
}

PER_SERIES_MODELS = set(MODEL_FN.keys())
# Also recognise _fb variants as per-series
PER_SERIES_MODELS_ALL = set()
for m in list(PER_SERIES_MODELS):
    PER_SERIES_MODELS_ALL.add(m)
    PER_SERIES_MODELS_ALL.add(f"{m}_fb")


def _forecast_per_series(
    winners: pd.DataFrame,
    df: pd.DataFrame,
    grain_cols: list[str],
    target: str,
    call_type: str,
) -> list[dict]:
    """Run per-series statistical forecasts for a single call_type.

    Returns a list of row dicts ready for DataFrame construction.
    """
    rows: list[dict] = []
    ct_winners = winners[winners["CALL_TYPE"] == call_type].copy()
    ct_winners = ct_winners[ct_winners["best_model"].isin(PER_SERIES_MODELS_ALL)]
    if ct_winners.empty:
        return rows

    ct_data = df[df["CALL_TYPE"] == call_type]
    n_series = len(ct_winners)

    for idx, (_, win_row) in enumerate(ct_winners.iterrows()):
        if (idx + 1) % 200 == 0:
            print(f"    per-series {call_type}: {idx + 1}/{n_series}", flush=True)

        grain_key = {c: win_row[c] for c in grain_cols}
        best = win_row["best_model"]

        # Filter to this series
        mask = pd.Series(True, index=ct_data.index)
        for c in grain_cols:
            mask &= ct_data[c] == grain_key[c]
        series_df = ct_data[mask].sort_values("CALL_YEAR_MONTH")
        train = series_df[target].values.astype(float)

        if len(train) < 12:
            # Not enough data; use seasonal_naive on whatever we have
            fc_raw = forecast_seasonal_naive(train, STAT_HORIZON) if len(train) > 0 else np.zeros(STAT_HORIZON)
            model_tag = "seasonal_naive_fb"
        else:
            # Determine base model (strip _fb)
            base_model = best.replace("_fb", "")
            is_fb = best.endswith("_fb")
            fn = MODEL_FN.get(base_model, forecast_seasonal_naive)

            fc_raw = fn(train, STAT_HORIZON)

            if fc_raw is not None:
                model_tag = best
            else:
                # Model failed — fallback to seasonal_naive
                fc_raw = forecast_seasonal_naive(train, STAT_HORIZON)
                if is_fb:
                    model_tag = "seasonal_naive_fb"
                else:
                    model_tag = f"{base_model}_fb"

        # Skip fc_raw[0] (Dec 2025) — take positions 1..12 for Jan–Dec 2026
        preds = fc_raw[1:STAT_HORIZON]

        for i, ym in enumerate(FORECAST_MONTHS):
            row = dict(grain_key)
            row["CALL_YEAR_MONTH"] = ym
            row["target"] = target
            row["model"] = model_tag
            row["predicted"] = float(preds[i])
            rows.append(row)

    print(f"    per-series {call_type}: {n_series}/{n_series} done", flush=True)
    return rows


def _forecast_lgbm(
    winners: pd.DataFrame,
    df: pd.DataFrame,
    grain_cols: list[str],
    target: str,
    call_type: str,
) -> list[dict]:
    """Iterative 1-step-ahead LightGBM forecast for a single call_type.

    Returns a list of row dicts ready for DataFrame construction.
    """
    import lightgbm as lgb

    rows: list[dict] = []
    ct_winners = winners[winners["CALL_TYPE"] == call_type].copy()
    lgbm_models = {"lgbm", "lgbm_fb"}
    ct_winners = ct_winners[ct_winners["best_model"].isin(lgbm_models)]
    if ct_winners.empty:
        return rows

    ct_data = df[df["CALL_TYPE"] == call_type].copy()

    # Build set of winning grain keys
    win_keys = ct_winners[grain_cols].drop_duplicates()

    # Filter data to only lgbm-winning series
    ext_df = ct_data.merge(win_keys, on=grain_cols, how="inner").copy()

    # Append 12 future rows per series
    unique_series = ext_df[grain_cols].drop_duplicates()
    future_rows = []
    for _, srow in unique_series.iterrows():
        for ym in FORECAST_MONTHS:
            frow = {c: srow[c] for c in grain_cols}
            frow["CALL_YEAR_MONTH"] = ym
            frow["YEAR"] = ym // 100
            frow["MONTH"] = ym % 100
            frow[target] = np.nan
            future_rows.append(frow)

    future_df = pd.DataFrame(future_rows)
    # Align columns: add any missing columns as NaN
    for col in ext_df.columns:
        if col not in future_df.columns:
            future_df[col] = np.nan

    ext_df = pd.concat([ext_df, future_df[ext_df.columns]], ignore_index=True)
    ext_df = ext_df.sort_values(grain_cols + ["CALL_YEAR_MONTH"]).reset_index(drop=True)

    feature_cols = list(LGBM_BASE_FEATURES) + list(LGBM_TREND_FEATURES)

    lgb_params = {
        "objective": "regression",
        "metric": "mae",
        "num_leaves": 31,
        "learning_rate": 0.05,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbose": -1,
        "n_jobs": -1,
    }
    num_boost_round = 300

    # Iterative 1-step-ahead
    for step_i, ym in enumerate(FORECAST_MONTHS):
        print(f"    lgbm {call_type}: step {step_i + 1}/12 (month {ym})", flush=True)

        # Rebuild features each step (so new predictions feed into lags)
        feat_df = build_lgbm_features(ext_df, target, grain_cols=grain_cols, add_trend=True)

        train_mask = feat_df["CALL_YEAR_MONTH"] < ym
        pred_mask = feat_df["CALL_YEAR_MONTH"] == ym

        train_feat = feat_df[train_mask].dropna(subset=["y"])
        pred_feat = feat_df[pred_mask].copy()

        if len(train_feat) < 100 or pred_feat.empty:
            # Not enough training data; skip this step
            continue

        # Fill NaN features in prediction rows with training medians
        for col in feature_cols:
            if col in pred_feat.columns:
                median_val = train_feat[col].median()
                pred_feat[col] = pred_feat[col].fillna(median_val)
            if col in train_feat.columns:
                median_val = train_feat[col].median()
                train_feat[col] = train_feat[col].fillna(median_val)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            dtrain = lgb.Dataset(
                train_feat[feature_cols], train_feat["y"], free_raw_data=False
            )
            model = lgb.train(lgb_params, dtrain, num_boost_round=num_boost_round)

        preds = np.clip(model.predict(pred_feat[feature_cols]), 0, None)
        pred_feat["_pred"] = preds

        # Write predictions back into ext_df so next step's lags pick them up
        for _, prow in pred_feat.iterrows():
            key_mask = ext_df["CALL_YEAR_MONTH"] == ym
            for c in grain_cols:
                key_mask &= ext_df[c] == prow[c]
            ext_df.loc[key_mask, target] = prow["_pred"]

    # Collect results — also handle fallback for series with missing predictions
    for _, srow in unique_series.iterrows():
        grain_key = {c: srow[c] for c in grain_cols}
        mask = pd.Series(True, index=ext_df.index)
        for c in grain_cols:
            mask &= ext_df[c] == grain_key[c]

        series_ext = ext_df[mask].sort_values("CALL_YEAR_MONTH")
        future_vals = series_ext[series_ext["CALL_YEAR_MONTH"].isin(FORECAST_MONTHS)]

        preds_arr = future_vals[target].values.astype(float)
        has_nan = np.any(np.isnan(preds_arr))

        if has_nan:
            # Fallback: use seasonal_naive on the historical portion
            hist = series_ext[series_ext["CALL_YEAR_MONTH"] < 202601][target].values.astype(float)
            hist = hist[~np.isnan(hist)]
            if len(hist) > 0:
                fb_preds = forecast_seasonal_naive(hist, 12)
            else:
                fb_preds = np.zeros(12)
            # Fill only NaN positions
            for i in range(12):
                if np.isnan(preds_arr[i]):
                    preds_arr[i] = fb_preds[i]
            model_tag = "lgbm_fb"
        else:
            model_tag = "lgbm"

        for i, ym in enumerate(FORECAST_MONTHS):
            row = dict(grain_key)
            row["CALL_YEAR_MONTH"] = ym
            row["target"] = target
            row["model"] = model_tag
            row["predicted"] = float(preds_arr[i])
            rows.append(row)

    return rows


def run_direction(direction: str) -> None:
    """Generate 2026 forecasts for one direction (inbound or outbound)."""
    print(f"\n{'='*60}")
    print(f"  Forecasting 2026 — {direction.upper()}")
    print(f"{'='*60}")

    grain_cols = grain_cols_for(direction)

    # Load winners
    winners_path = ROOT / "reports" / f"{direction}_horserace_winners.csv"
    winners = pd.read_csv(winners_path)
    winners = winners[winners["CALL_TYPE"].isin(DASHBOARD_CALL_TYPES)]
    print(f"  Loaded {len(winners)} winner rows (filtered to {DASHBOARD_CALL_TYPES})")

    # Load data
    data_path = ROOT / "data" / f"{direction}_set.parquet"
    df = pd.read_parquet(data_path)
    print(f"  Loaded data: {df.shape[0]} rows, months {df['CALL_YEAR_MONTH'].min()}-{df['CALL_YEAR_MONTH'].max()}")

    all_rows: list[dict] = []

    for call_type in DASHBOARD_CALL_TYPES:
        target = target_for(call_type, direction)
        print(f"\n  --- {call_type} (target={target}) ---")

        ct_winners = winners[winners["CALL_TYPE"] == call_type]
        model_counts = ct_winners["best_model"].value_counts().to_dict()
        print(f"  Model distribution: {model_counts}")

        # Per-series statistical models
        ps_rows = _forecast_per_series(winners, df, grain_cols, target, call_type)
        all_rows.extend(ps_rows)
        print(f"  Per-series: {len(ps_rows) // 12 if ps_rows else 0} series forecasted")

        # LightGBM iterative
        lgbm_rows = _forecast_lgbm(winners, df, grain_cols, target, call_type)
        all_rows.extend(lgbm_rows)
        print(f"  LightGBM:   {len(lgbm_rows) // 12 if lgbm_rows else 0} series forecasted")

    # Build output
    out_df = pd.DataFrame(all_rows)
    out_cols = grain_cols + ["CALL_YEAR_MONTH", "target", "model", "predicted"]
    out_df = out_df[out_cols]
    out_df = out_df.sort_values(grain_cols + ["CALL_YEAR_MONTH"]).reset_index(drop=True)

    out_path = ROOT / "reports" / f"{direction}_forecast_2026.csv"
    out_df.to_csv(out_path, index=False)
    print(f"\n  Saved {len(out_df)} rows to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate 2026 forecasts")
    parser.add_argument(
        "--direction",
        choices=["inbound", "outbound"],
        default=None,
        help="Run for one direction only. Default: both.",
    )
    args = parser.parse_args()

    directions = [args.direction] if args.direction else ["inbound", "outbound"]
    for d in directions:
        run_direction(d)

    print("\nDone.")


if __name__ == "__main__":
    main()
