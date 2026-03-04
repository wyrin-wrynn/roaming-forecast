#!/usr/bin/env python3
"""Produce Dec-2025 forecasts using the best rolling model per series.

Reads rolling accuracy results to pick the best model (lowest median APE)
for each series, then trains that model on all available data and generates
a 1-step-ahead forecast for Dec 2025 (CALL_YEAR_MONTH = 202512).

Output (to reports/):
  {direction}_forecast_dec2025.csv

Usage:
  python scripts/run_forecast_dec2025.py                      # both directions
  python scripts/run_forecast_dec2025.py --direction inbound   # one direction
"""
from __future__ import annotations

import argparse
import logging
import os
import time
from pathlib import Path

logging.disable(logging.CRITICAL)
os.environ["CMDSTAN_VERBOSE"] = "false"

import numpy as np
import pandas as pd

from config import (
    DASHBOARD_CALL_TYPES,
    grain_cols_for,
    target_for,
)
from models import (
    forecast_ets_damped_1step,
    forecast_theta_1step,
    forecast_seasonal_naive_1step,
)

ROOT = Path(__file__).resolve().parent.parent

MODEL_FUNCS = {
    "ets_damped": forecast_ets_damped_1step,
    "theta": forecast_theta_1step,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Dec-2025 forecast using best rolling model.")
    p.add_argument("--direction", choices=["inbound", "outbound"], default=None,
                   help="Run for one direction only; omit for both.")
    return p.parse_args()


def best_model_per_series(
    acc_df: pd.DataFrame,
    grain_cols: list[str],
) -> pd.DataFrame:
    """Return a DataFrame with one row per series: grain_cols + ['best_model'].

    Filters to DASHBOARD_CALL_TYPES and non-fallback, non-lgbm models
    (ets_damped, theta), then picks the model with lowest median APE.
    """
    kept_models = list(MODEL_FUNCS.keys())

    mask = (
        acc_df["CALL_TYPE"].isin(DASHBOARD_CALL_TYPES)
        & acc_df["model"].isin(kept_models)
    )
    subset = acc_df.loc[mask].copy()

    median_ape = (
        subset.groupby(grain_cols + ["model"])["ape"]
        .median()
        .reset_index()
        .rename(columns={"ape": "median_ape"})
    )
    # Drop rows where median_ape is NaN (no valid APE data)
    median_ape = median_ape.dropna(subset=["median_ape"])

    # For each series pick the model with lowest median APE
    idx = median_ape.groupby(grain_cols)["median_ape"].idxmin()
    best = median_ape.loc[idx, grain_cols + ["model"]].rename(columns={"model": "best_model"})
    return best.reset_index(drop=True)


def run_direction(direction: str) -> None:
    """Generate Dec-2025 forecasts for a single direction."""
    grain_cols = grain_cols_for(direction)
    reports = ROOT / "reports"
    reports.mkdir(parents=True, exist_ok=True)

    # --- Load rolling accuracy and pick best model per series ---
    acc_path = reports / f"{direction}_rolling_accuracy.csv"
    print(f"\n{'='*60}", flush=True)
    print(f"Direction : {direction}", flush=True)
    print(f"Accuracy  : {acc_path}", flush=True)

    acc_df = pd.read_csv(acc_path)
    best_df = best_model_per_series(acc_df, grain_cols)
    print(f"Series with best-model selection: {len(best_df)}", flush=True)

    # --- Load parquet data ---
    data_path = ROOT / "data" / f"{direction}_set.parquet"
    print(f"Data      : {data_path}", flush=True)
    df = pd.read_parquet(data_path)
    df = df.sort_values(grain_cols + ["CALL_YEAR_MONTH"]).reset_index(drop=True)

    # Filter to dashboard call types
    df = df[df["CALL_TYPE"].isin(DASHBOARD_CALL_TYPES)].copy()

    grouped = {k: g.sort_values("CALL_YEAR_MONTH") for k, g in df.groupby(grain_cols)}

    # Build a lookup: grain tuple -> best_model name
    best_lookup: dict[tuple, str] = {}
    for _, row in best_df.iterrows():
        key = tuple(row[c] for c in grain_cols)
        best_lookup[key] = row["best_model"]

    # --- Forecast loop ---
    rows: list[dict] = []
    fallbacks = 0
    skipped = 0
    t0 = time.time()

    series_keys = sorted(grouped.keys())
    n = len(series_keys)

    for idx, key in enumerate(series_keys):
        if key not in best_lookup:
            skipped += 1
            continue

        series = grouped[key]
        call_type = key[grain_cols.index("CALL_TYPE")]
        target = target_for(call_type, direction)
        train = series[target].values.astype(float)

        best_model = best_lookup[key]
        model_func = MODEL_FUNCS[best_model]
        pred = model_func(train)

        used_model = best_model
        if pred is None:
            pred = forecast_seasonal_naive_1step(train)
            used_model = f"{best_model}_fb"
            fallbacks += 1

        base = {c: key[i] for i, c in enumerate(grain_cols)}
        rows.append({
            **base,
            "CALL_YEAR_MONTH": 202512,
            "target": target,
            "model": used_model,
            "predicted": pred,
        })

        if (idx + 1) % 500 == 0:
            el = time.time() - t0
            rate = (idx + 1) / el
            print(f"  {idx+1}/{n} series ({el:.0f}s, {rate:.1f}/s)", flush=True)

    el = time.time() - t0
    result = pd.DataFrame(rows)

    output_path = reports / f"{direction}_forecast_dec2025.csv"
    result.to_csv(output_path, index=False)

    print(f"\nDone: {len(result)} forecasts in {el:.1f}s", flush=True)
    print(f"  Fallbacks : {fallbacks}", flush=True)
    print(f"  Skipped   : {skipped} (no rolling accuracy)", flush=True)
    print(f"  Output    : {output_path}", flush=True)


def main():
    args = parse_args()

    if args.direction is not None:
        directions = [args.direction]
    else:
        directions = ["inbound", "outbound"]

    for d in directions:
        run_direction(d)


if __name__ == "__main__":
    main()
