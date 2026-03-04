# Architecture

## Dual-Grain Design

Roaming traffic has different granularity requirements per direction:

- **Inbound**: forecast at `SRC_TADIG x DST_TADIG x CALL_TYPE` (operator-to-operator routes).
  Each route is a specific foreign operator sending traffic to a specific home-network operator.
  ~7,700 series across 4 call types.

- **Outbound**: forecast at `SRC_TADIG x DST_COUNTRY x CALL_TYPE` (country-level aggregation).
  Multiple destination operators within a country are summed because outbound pricing is
  typically set at country level. ~1,900 series across 4 call types.

The 20 portfolio countries define the scope. Only routes involving these countries are
extracted and forecast.

## Data Model

### SQLite (`data/forecasting.db`)

The `traffic` table has 28 columns:

- **Identifiers (9)**: SRC_TADIG, DST_TADIG, DST_COUNTRY, DST_NAME, GROUPNAME, NEGOTIATOR, CALL_TYPE, CALL_YEAR_MONTH (integer YYYYMM), plus derived fields
- **Inbound metrics (9)**: INBOUND_CALLS, INBOUND_CHARGED_CALLS, INBOUND_VOL_MB, INBOUND_CHARGED_VOLUME_MB, INBOUND_DURATION, INBOUND_CHARGED_DURATION, INBOUND_TAP_CHARGES, INBOUND_AA14_CHARGES, INBOUND_TAX
- **Outbound metrics (9)**: OUTBOUND_CALLS, OUTBOUND_CHARGED_CALLS, OUTBOUND_VOL_MB, OUTBOUND_CHARGED_VOLUME_MB, OUTBOUND_DURATION, OUTBOUND_CHARGED_DURATION, OUTBOUND_TAP_CHARGES, OUTBOUND_AA14_CHARGES, OUTBOUND_TAX

A `traffic_features` view adds derived YEAR and MONTH columns.

### Parquet (`data/`)

- `inbound_set.parquet` — filtered to portfolio countries, inbound grain columns + all metrics + CALL_YEAR_MONTH
- `outbound_set.parquet` — filtered to portfolio countries, aggregated to country-level grain

### Report CSVs (`reports/`)

- **Predictions**: grain columns + CALL_YEAR_MONTH + target + model + predicted value
- **Metrics**: grain columns + target + model + WAPE + SMAPE + MAE + RMSE
- **Winners**: grain columns + target + best_model + best_wape
- **Rolling accuracy**: grain columns + target + model + forecast_month + actual + predicted + APE
- **Dec 2025 forecast**: grain columns + CALL_YEAR_MONTH (202512) + target + model + predicted
- **2026 forecast**: grain columns + CALL_YEAR_MONTH (202601–202612) + target + model + predicted

## Model Cascade

Five models compete per series in the horserace:

| Model | Type | Description |
|-------|------|-------------|
| `seasonal_naive` | Baseline | Repeats last 12 months of training data |
| `ets_damped` | Statistical | Exponential smoothing with damped additive trend + additive seasonal (period=12) |
| `sarima` | Statistical | Seasonal ARIMA via `pmdarima.auto_arima` (max orders p,q=2; P,Q=1; d,D=1) |
| `theta` | Statistical | Theta decomposition with seasonal period=12 |
| `lgbm` | ML (global) | LightGBM trained globally per call-type across all series |

### Fallback Logic

When a per-series statistical model fails to fit (insufficient data, convergence issues,
NaN/Inf output), the caller falls back to `seasonal_naive` and tags the prediction with
a `_fb` suffix (e.g. `sarima_fb`, `ets_damped_fb`). This ensures every series gets a
prediction from every model slot.

### LightGBM Feature Engineering

`build_lgbm_features()` constructs:

- **Calendar**: month (1-12), quarter (1-4)
- **Series encoding**: integer code per unique grain combination
- **Lags**: lag_1, lag_2, lag_3, lag_6, lag_12
- **Rolling stats**: roll_mean_{3,6,12}, roll_std_{3,6,12} (shifted by 1 to avoid leakage)
- **Trend features** (v2): yoy_ratio, trend_slope_6 (normalized 6-month slope), growth_momentum (6-month mean ratio), time_idx (cumulative count)

LightGBM params: `num_leaves=31, lr=0.05, feature_fraction=0.8, bagging_fraction=0.8, 300 rounds, MAE objective`.
Minimum 100 training rows required; otherwise returns None.

## Production Forecasting

After evaluation, two scripts generate forward-looking forecasts using the best models:

### Dec 2025 Forecast (`run_forecast_dec2025.py`)

- **Model selection**: best rolling retrain model per series (lowest median APE, ets_damped or theta only)
- **Method**: trains on all data through Nov 2025, produces 1-step-ahead forecast for Dec 2025
- **Fallback**: seasonal naive if the selected model fails to fit
- **Output**: `reports/{direction}_forecast_dec2025.csv`

### 2026 Forecast (`run_forecast_2026.py`)

- **Model selection**: best horserace model per series (from `horserace_winners.csv`)
- **Scope**: GPRS + MOC call types only (`DASHBOARD_CALL_TYPES`)
- **Method**: retrains the winning model on all data through Nov 2025, forecasts Jan–Dec 2026
- **Statistical models**: forecast with horizon=13, skip position 0 (Dec 2025) to get aligned Jan–Dec 2026 predictions
- **LightGBM**: iterative 1-step-ahead — each month's prediction feeds back into lag features for the next month
- **Fallback**: seasonal naive with `_fb` suffix if the selected model fails
- **Output**: `reports/{direction}_forecast_2026.csv`

## Evaluation Framework

### Horserace (Static Train/Test)

- **Split**: train through Dec 2024, test Jan-Nov 2025 (11 months)
- **Method**: fit each model once on training data, forecast full test horizon
- **Metrics**: WAPE, SMAPE, MAE, RMSE per series per model
- **Winner selection**: lowest WAPE per series

### Rolling Retrain (Expanding Window)

- **Method**: walk-forward from month 13 onward, expanding the training window by 1 month each step
- **Models**: ets_damped, theta (1-step variants); LightGBM at checkpoints [12, 18, 23] months
- **Metric**: APE per observation, median APE for summary accuracy

The rolling retrain validates whether a model's accuracy holds as new data arrives,
complementing the horserace's static snapshot.

## Dashboard Categorization

Every route is assigned to one of 5 confidence tiers:

| Category | Rule | Color |
|----------|------|-------|
| Trustworthy | Static accuracy >= 85% AND rolling accuracy >= 80% | Green |
| Promising | Static accuracy >= 85%, rolling < 80% or unavailable | Blue |
| Review Needed | Static accuracy 60-85%, no outliers | Yellow |
| Volatile | Static accuracy 60-85%, has outliers (>3x IQR) | Orange |
| Unreliable | Static accuracy < 60% | Red |

**Static accuracy** = `(1 - WAPE) * 100` from the horserace best model.

**Rolling accuracy** = `(1 - median APE) * 100` from the best rolling retrain model
(ets_damped or theta, excluding fallback models).

### Outlier Detection

A series has outliers if any non-zero value exceeds `Q3 + 3 * IQR`, where Q1 and Q3
are the 25th and 75th percentiles of non-zero values. At least 4 non-zero observations
are required.

## Target Mapping

Each call type maps to exactly one forecast target per direction:

| Call Type | Inbound Target | Outbound Target |
|-----------|---------------|-----------------|
| GPRS | INBOUND_VOL_MB | OUTBOUND_VOL_MB |
| MOC | INBOUND_DURATION | OUTBOUND_DURATION |
| MTC | INBOUND_DURATION | OUTBOUND_DURATION |
| SMS-MT | INBOUND_CALLS | OUTBOUND_CALLS |

Note: SMS-MO was dropped in v2. MOC and MTC both map to DURATION.

## Dashboard Pages (`dashboard_v2.py`)

### Portfolio Overview

Summary view showing all routes grouped by confidence tier. Each tier displays a
sortable table with route details and accuracy. Clicking a row navigates to the Explorer
for that route.

### Explorer

Per-route deep dive with cascading filters:
- **Inbound**: Call Type → SRC_TADIG → DST_COUNTRY → DST_TADIG
- **Outbound**: Call Type → SRC_TADIG → DST_COUNTRY

Displays a YoY line chart (Jan–Dec x-axis) with toggleable series via `st.pills`:
- **Actuals**: 2023, 2024, 2025 (each toggleable independently)
- **Horserace models**: top 3 by WAPE + all others available
- **Rolling models**: ets_damped, theta (test period only, filtered to 2025+)
- **Dec 2025 Forecast**: single-point star marker from best rolling model
- **2026 Forecast**: dashdot line from best horserace model (Jan–Dec 2026)

Below the chart, a transposed data table shows models as rows and months as columns,
with color-coded cells (green ≤10% APE, yellow 10–25%, red >25%) and WAPE/Accuracy
summary columns. Forecast rows have no color-coding (no actuals to compare).

### Forecast Table

Tabular view of all routes with forecast values, sortable and filterable.
