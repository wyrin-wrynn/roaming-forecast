#!/usr/bin/env python3
"""Unified Horse Race: 5 models, parameterized by direction.

Models (5):
  Per-series: seasonal_naive, ets_damped, sarima, theta
  Global:     lgbm (with trend features baked in)
  (sarima fallbacks tracked as sarima_fb)

Usage:
  python scripts/run_horserace.py --direction inbound
  python scripts/run_horserace.py --direction outbound
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
    CALL_TYPES, TRAIN_END, TEST_START, TEST_END,
    grain_cols_for, target_for,
)
from models import (
    forecast_seasonal_naive, forecast_ets, forecast_sarima, forecast_theta,
    forecast_lgbm_global, wape, smape_metric,
)

ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Unified model horse race.")
    p.add_argument("--direction", required=True, choices=["inbound", "outbound"])
    return p.parse_args()


def run_per_series_model(model_name: str, train_grouped: dict, test_grouped: dict,
                         series_keys: list, grain_cols: list[str],
                         direction: str) -> list[dict]:
    """Run a single per-series model across all series."""
    pred_rows = []
    n = len(series_keys)
    t0 = time.time()
    fallbacks = 0

    for idx, key in enumerate(series_keys):
        tr = train_grouped[key]
        te = test_grouped.get(key)
        if te is None or len(te) == 0:
            continue

        call_type = key[grain_cols.index("CALL_TYPE")]
        target = target_for(call_type, direction)
        test_months = te["CALL_YEAR_MONTH"].values
        horizon = len(te)

        train_vals = tr[target].values.astype(float)
        actuals = te[target].values.astype(float)

        if model_name == "seasonal_naive":
            fc = forecast_seasonal_naive(train_vals, horizon)
        elif model_name == "ets_damped":
            fc = forecast_ets(train_vals, horizon, damped=True, trend="add")
        elif model_name == "sarima":
            fc = forecast_sarima(train_vals, horizon)
        elif model_name == "theta":
            fc = forecast_theta(train_vals, horizon)
        else:
            continue

        used = model_name
        if fc is None:
            fc = forecast_seasonal_naive(train_vals, horizon)
            used = f"{model_name}_fb"
            fallbacks += 1

        base = {c: key[i] for i, c in enumerate(grain_cols)}
        for i in range(horizon):
            pred_rows.append({
                **base,
                "CALL_YEAR_MONTH": int(test_months[i]),
                "target": target, "model": used,
                "actual": actuals[i], "predicted": fc[i],
            })

        if (idx + 1) % 200 == 0:
            el = time.time() - t0
            r = (idx + 1) / el
            print(f"  [{model_name}] {idx+1}/{n} ({el:.0f}s, {r:.1f}/s, ETA {(n-idx-1)/r:.0f}s)", flush=True)

    el = time.time() - t0
    print(f"  [{model_name}] done in {el:.0f}s ({fallbacks} fallbacks)", flush=True)
    return pred_rows


def run_lgbm(df: pd.DataFrame, grain_cols: list[str], direction: str,
             series_keys: list, test: pd.DataFrame) -> list[dict]:
    """Run single LightGBM (with trend) per call_type."""
    print(f"\n--- lgbm ---", flush=True)
    t0 = time.time()
    rows = []
    test_lookup = test.set_index(grain_cols + ["CALL_YEAR_MONTH"])

    for ct in CALL_TYPES:
        ct_mask = df["CALL_TYPE"] == ct
        ct_df = df[ct_mask].reset_index(drop=True)
        ct_train_mask = ct_df["CALL_YEAR_MONTH"] <= TRAIN_END
        ct_test_mask = (ct_df["CALL_YEAR_MONTH"] >= TEST_START) & (ct_df["CALL_YEAR_MONTH"] <= TEST_END)

        target = target_for(ct, direction)
        res = forecast_lgbm_global(ct_df, target, ct_train_mask, ct_test_mask,
                                   grain_cols=grain_cols, add_trend=True)
        if res is not None:
            for _, row in res.iterrows():
                key = tuple(row[c] for c in grain_cols) + (int(row["CALL_YEAR_MONTH"]),)
                if key in test_lookup.index:
                    base = {c: row[c] for c in grain_cols}
                    rows.append({
                        **base,
                        "CALL_YEAR_MONTH": int(row["CALL_YEAR_MONTH"]),
                        "target": target, "model": "lgbm",
                        "actual": float(test_lookup.loc[key, target]),
                        "predicted": float(row["pred"]),
                    })
            print(f"  lgbm done for {ct}/{target}", flush=True)

    print(f"  lgbm complete in {time.time()-t0:.0f}s", flush=True)
    return rows


def compute_and_save_metrics(preds_df: pd.DataFrame, grain_cols: list[str],
                             direction: str):
    """Compute per-series, overall, and winner metrics."""
    reports = ROOT / "reports"

    # Per-series metrics
    metrics_rows = []
    group_cols = grain_cols + ["target", "model"]
    for key, grp in preds_df.groupby(group_cols):
        yt, yp = grp["actual"].values, grp["predicted"].values
        row = dict(zip(group_cols, key))
        row.update({
            "wape": wape(yt, yp), "smape": smape_metric(yt, yp),
            "mae": float(np.mean(np.abs(yt - yp))),
            "rmse": float(np.sqrt(np.mean((yt - yp) ** 2))),
        })
        metrics_rows.append(row)
    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(reports / f"{direction}_horserace_metrics.csv", index=False)

    # Overall
    overall_rows = []
    for (ct, target, model), grp in preds_df.groupby(["CALL_TYPE", "target", "model"]):
        yt, yp = grp["actual"].values, grp["predicted"].values
        overall_rows.append({
            "CALL_TYPE": ct, "target": target, "model": model,
            "n_series": grp[grain_cols].drop_duplicates().shape[0],
            "n_obs": len(grp), "wape": wape(yt, yp), "smape": smape_metric(yt, yp),
            "mae": float(np.mean(np.abs(yt - yp))),
            "rmse": float(np.sqrt(np.mean((yt - yp) ** 2))),
        })
    overall_df = pd.DataFrame(overall_rows)
    overall_df.to_csv(reports / f"{direction}_horserace_overall.csv", index=False)

    print(f"\nOverall WAPE:", flush=True)
    pivot = overall_df.pivot_table(index="model", columns="CALL_TYPE", values="wape")
    pivot["avg_wape"] = pivot.mean(axis=1)
    print(pivot.sort_values("avg_wape").to_string(), flush=True)

    # Winners
    winner_rows = []
    winner_group = grain_cols + ["target"]
    for key, grp in metrics_df.groupby(winner_group):
        valid = grp.dropna(subset=["wape"])
        if valid.empty:
            continue
        best = valid.loc[valid["wape"].idxmin()]
        row = dict(zip(winner_group, key))
        row.update({"best_model": best["model"], "best_wape": best["wape"]})
        winner_rows.append(row)
    winners_df = pd.DataFrame(winner_rows)
    winners_df.to_csv(reports / f"{direction}_horserace_winners.csv", index=False)
    print(f"\nWinner distribution:", flush=True)
    print(winners_df["best_model"].value_counts().to_string(), flush=True)


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

    train_mask = df["CALL_YEAR_MONTH"] <= TRAIN_END
    test_mask = (df["CALL_YEAR_MONTH"] >= TEST_START) & (df["CALL_YEAR_MONTH"] <= TEST_END)
    train = df[train_mask]
    test = df[test_mask]

    train_grouped = {k: g.sort_values("CALL_YEAR_MONTH") for k, g in train.groupby(grain_cols)}
    test_grouped = {k: g.sort_values("CALL_YEAR_MONTH") for k, g in test.groupby(grain_cols)}
    series_keys = sorted(train_grouped.keys())

    # Summary
    total_tasks = len(series_keys)  # 1 target per series now
    print(f"Series: {len(series_keys)}, Total forecast tasks: {total_tasks}", flush=True)
    for ct in CALL_TYPES:
        ct_keys = [k for k in series_keys if k[grain_cols.index("CALL_TYPE")] == ct]
        print(f"  {ct}: {len(ct_keys)} series → target: {target_for(ct, direction)}", flush=True)
    print(f"Models: seasonal_naive, ets_damped, sarima, theta, lgbm", flush=True)

    all_rows: list[dict] = []

    # Per-series models
    for model_name in ["seasonal_naive", "ets_damped", "sarima", "theta"]:
        print(f"\n--- {model_name} ---", flush=True)
        rows = run_per_series_model(model_name, train_grouped, test_grouped,
                                    series_keys, grain_cols, direction)
        all_rows.extend(rows)

    # LightGBM (single, with trend)
    lgbm_rows = run_lgbm(df, grain_cols, direction, series_keys, test)
    all_rows.extend(lgbm_rows)

    # Save predictions
    preds_df = pd.DataFrame(all_rows)
    preds_path = reports / f"{direction}_horserace_predictions.csv"
    preds_df.to_csv(preds_path, index=False)
    print(f"\nPredictions: {preds_path} ({len(preds_df)} rows)", flush=True)

    # Metrics
    compute_and_save_metrics(preds_df, grain_cols, direction)


if __name__ == "__main__":
    main()
