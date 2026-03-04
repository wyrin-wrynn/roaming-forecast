# 2026 Forecast — Design Document

## Goal

Two new scripts:
1. `scripts/run_forecast_dec2025.py` — rolling 1-step forecast for Dec 2025 (best rolling model per market)
2. `scripts/run_forecast_2026.py` — retrain best horserace model per market on all data, forecast Jan–Dec 2026

Update the dashboard explorer to display both as new series.

## Scope

- **Call types:** GPRS and MOC only (matching dashboard)
- **Directions:** Inbound and Outbound
- **Dec 2025:** Best rolling model (ets_damped or theta) trained on all 35 months, 1-step forecast
- **2026:** Best horserace model retrained on all 35 months, 12-month horizon forecast

## Script 1: `run_forecast_dec2025.py`

### What it does

For each market, finds the best rolling model (lowest median APE from rolling_accuracy.csv), trains it on all data through Nov 2025, produces 1-step forecast for Dec 2025.

### Input files

- `data/{direction}_set.parquet` — time series data
- `reports/{direction}_rolling_accuracy.csv` — to determine best rolling model per series

### Output files

- `reports/inbound_forecast_dec2025.csv`
- `reports/outbound_forecast_dec2025.csv`

### Output format

| Column | Description |
|--------|-------------|
| grain_cols | Series identifiers |
| CALL_YEAR_MONTH | 202512 |
| target | Target variable |
| model | Best rolling model name |
| predicted | Forecast value |

## Script 2: `run_forecast_2026.py`

### What it does

For each market, retrains the best horserace model on all data through Nov 2025, forecasts 12 months (Jan–Dec 2026).

### Input files

- `data/{direction}_set.parquet` — time series data
- `reports/{direction}_horserace_winners.csv` — best model per market

### Processing

```
for direction in [inbound, outbound]:
    load parquet data + winners CSV
    filter winners to GPRS + MOC call types
    group winners by best_model

    for each model type (seasonal_naive, ets_damped, sarima, theta):
        for each series where this model won:
            extract full training array (all months ≤ 202511)
            call forecast function with horizon=12
            if fails: fallback to seasonal_naive, tag as {model}_fb
            store predictions for months 202601–202612

    for lgbm winners (batch):
        collect all lgbm-winning series per call type
        build features for full dataset + 12 future months
        train global LightGBM model
        extract per-series 12-month predictions
        if fails per series: fallback to seasonal_naive

    concat all predictions → write CSV
```

### Output files

- `reports/inbound_forecast_2026.csv`
- `reports/outbound_forecast_2026.csv`

### Output format

| Column | Description |
|--------|-------------|
| grain_cols | Series identifiers |
| CALL_YEAR_MONTH | 202601–202612 |
| target | Target variable |
| model | Winning model name (or {model}_fb on fallback) |
| predicted | Forecast value |

### Fallback logic

Same as horserace: if best model fails, fall back to `seasonal_naive`, tag with `_fb` suffix.

### CLI

```
python scripts/run_forecast_dec2025.py                    # both directions
python scripts/run_forecast_dec2025.py --direction inbound
python scripts/run_forecast_2026.py                       # both directions
python scripts/run_forecast_2026.py --direction inbound
```

## Dashboard Changes: `dashboard_v2.py`

### Bug fix (already done)

Rolling predictions filtered to `forecast_month >= 202501` to prevent Dec 2024 data leaking into the YoY month-12 slot.

### Data loading

- Load `reports/{direction}_forecast_dec2025.csv`
- Load `reports/{direction}_forecast_2026.csv`

### Explorer integration

- New pill: `"Dec 2025 Forecast"` — shows single Dec point from rolling model
- New pill: `"2026 Forecast"` — shows Jan–Dec 2026 line from best horserace model
- **Chart:** Two new lines on YoY chart
- **Table:** Two new rows with month values. No WAPE/Accuracy (no actuals yet).
- Colors: Distinct from existing series
