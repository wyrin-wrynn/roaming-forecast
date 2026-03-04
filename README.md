# Roaming Traffic Forecasting

Monthly traffic forecasting for 20 portfolio countries across inbound and outbound roaming.
Uses a dual-grain architecture: inbound routes at TADIG-to-TADIG operator level,
outbound routes aggregated at country level. Five candidate models compete per series
in a horserace evaluation, with rolling retrain for walk-forward validation.

## Quick Start

```bash
# 1. Load Excel workbooks into SQLite
python scripts/load_forecasting_data.py --replace

# 2. Extract portfolio datasets to Parquet
python scripts/extract_portfolio_data.py

# 3. Run the full model pipeline (horserace + rolling retrain, ~10-14h)
python scripts/run_all_v2.py

# 4. Launch the dashboard
streamlit run scripts/dashboard_v2.py --server.port 8505 --server.headless true
```

Steps 1-2 take minutes. Step 3 is long-running — use `tmux` or `nohup`.
You can also run individual directions:

```bash
python scripts/run_horserace.py --direction inbound
python scripts/run_rolling_retrain.py --direction outbound
```

## Pipeline

```
 ┌──────────────┐     ┌───────────────┐     ┌──────────────────┐
 │ Excel files  │────▶│ SQLite DB     │────▶│ Parquet datasets │
 │ *Actual*.xlsx│     │ forecasting.db│     │ inbound_set      │
 └──────────────┘     └───────────────┘     │ outbound_set     │
   load_forecasting     extract_portfolio   └──────────────────┘
   _data.py             _data.py                     │
                                                     ▼
                                          ┌─────────────────────┐
                                          │ Model pipeline      │
                                          │ run_horserace.py    │
                                          │ run_rolling_retrain │
                                          └─────────┬───────────┘
                                                    │
                                                    ▼
                                          ┌─────────────────────┐
                                          │ reports/ CSVs       │
                                          │ predictions, metrics│
                                          │ winners, rolling    │
                                          └─────────┬───────────┘
                                                    │
                                                    ▼
                                          ┌─────────────────────┐
                                          │ Streamlit Dashboard │
                                          │ dashboard_v2.py     │
                                          └─────────────────────┘
```

## File Inventory

| Script | Purpose | Inputs | Outputs |
|--------|---------|--------|---------|
| `config.py` | Shared constants: grains, targets, portfolio, train/test split | — | Imported by all scripts |
| `inspect_excels.py` | One-time Excel schema discovery and profiling | `*Actual*.xlsx` | `reports/excel_inventory.json` |
| `load_forecasting_data.py` | Load Excel workbooks into SQLite | `*Actual*.xlsx` | `data/forecasting.db` |
| `extract_portfolio_data.py` | Extract portfolio subsets to Parquet | `data/forecasting.db` | `data/{in,out}bound_set.parquet`, `data/*_series.csv` |
| `models.py` | Model library: 5 forecasting models + metrics + LightGBM features | — | Imported by horserace/rolling |
| `run_horserace.py` | Static train/test evaluation of all 5 models | Parquet datasets | `reports/{dir}_horserace_*.csv` |
| `run_rolling_retrain.py` | Expanding-window 1-step-ahead evaluation | Parquet datasets | `reports/{dir}_rolling_accuracy.csv` |
| `run_all_v2.py` | Orchestrates full pipeline (4 subprocess steps) | Parquet datasets | All report CSVs |
| `run_forecast_dec2025.py` | Dec 2025 forecast using best rolling model | `reports/{dir}_rolling_accuracy.csv`, Parquet | `reports/{dir}_forecast_dec2025.csv` |
| `run_forecast_2026.py` | Jan–Dec 2026 forecast using best horserace model | `reports/{dir}_horserace_winners.csv`, Parquet | `reports/{dir}_forecast_2026.csv` |
| `dashboard_v2.py` | Interactive Streamlit dashboard (3 pages) | `data/forecasting.db`, `reports/*.csv`, Parquet | Browser UI |

## Data Files

### `data/`

| File | Description |
|------|-------------|
| `forecasting.db` | SQLite database with `traffic` table (28 columns: 9 identifiers + 18 metrics + CALL_YEAR_MONTH) |
| `inbound_set.parquet` | Inbound series at TADIG-to-TADIG grain for 20 portfolio countries |
| `outbound_set.parquet` | Outbound series at country-level grain for 20 portfolio countries |
| `inbound_series.csv` | Route listing with DST_COUNTRY for reference |
| `outbound_series.csv` | Route listing |

### `reports/`

Per-direction CSVs (inbound + outbound):

| File pattern | Description |
|-------------|-------------|
| `{dir}_horserace_predictions.csv` | Per-series per-month predictions from all 5 models |
| `{dir}_horserace_metrics.csv` | Per-series accuracy (WAPE, SMAPE, MAE, RMSE) |
| `{dir}_horserace_overall.csv` | Aggregated metrics by model x call-type |
| `{dir}_horserace_winners.csv` | Best model per series (lowest WAPE) |
| `{dir}_rolling_accuracy.csv` | Per-series per-month rolling retrain predictions + APE |
| `{dir}_forecast_dec2025.csv` | Dec 2025 forecast (1-step, best rolling model) |
| `{dir}_forecast_2026.csv` | Jan–Dec 2026 forecast (best horserace model, GPRS+MOC only) |

## Configuration

All pipeline constants live in `scripts/config.py`:

- **Train/test split**: train through Dec 2024, test Jan-Nov 2025
- **Portfolio**: 20 countries (see `PORTFOLIO_COUNTRIES`)
- **Grain**: inbound = `SRC_TADIG x DST_TADIG x CALL_TYPE`, outbound = `SRC_TADIG x DST_COUNTRY x CALL_TYPE`
- **Target mapping**: each call-type maps to one metric per direction (e.g. GPRS -> VOL_MB, MOC -> DURATION)

See [docs/architecture.md](docs/architecture.md) for full design details.

## Dashboard

Launch: `streamlit run scripts/dashboard_v2.py --server.port 8505 --server.headless true`

Three pages:

1. **Portfolio Overview** — category distribution (Trustworthy/Promising/Review Needed/Volatile/Unreliable), direction toggle, click-to-explore
2. **Explorer** — drill into a single route with YoY chart, model overlays, Dec 2025 + 2026 forecast lines, color-coded data table. Cascading filters: Call Type → SRC_TADIG → DST_COUNTRY → DST_TADIG (inbound) or DST_COUNTRY (outbound)
3. **Forecast Table** — filterable/sortable table of all routes with accuracy, category, data quality flags, CSV download

See [docs/architecture.md](docs/architecture.md) for category tier definitions and outlier detection rules.
