# Glossary

**APE** — Absolute Percentage Error. `|actual - predicted| / |actual|`. Used per-observation in rolling retrain; median APE summarizes a model's rolling accuracy.

**CALL_YEAR_MONTH** — Integer `YYYYMM` identifying the traffic month (e.g. `202501` = January 2025). Primary time axis for all series.

**Call types** — The 4 roaming service categories tracked in v2:
- **GPRS** — mobile data (measured in MB)
- **MOC** — Mobile Originated Call, i.e. voice calls made (measured in duration)
- **MTC** — Mobile Terminated Call, i.e. voice calls received (measured in duration)
- **SMS-MT** — SMS Mobile Terminated, i.e. SMS received (measured in call count)

**Category tiers** — 5 confidence levels assigned to each forecasted route based on accuracy and data quality. From best to worst: Trustworthy, Promising, Review Needed, Volatile, Unreliable. See [architecture.md](architecture.md) for exact rules.

**DST_COUNTRY** — Destination country identifier. Used as a grain column for outbound routes where traffic is aggregated across operators within a country.

**DST_TADIG** — Destination TADIG code identifying the receiving operator. Used as a grain column for inbound routes.

**Expanding window** — Training strategy where the model is retrained each month using all data available up to that point. The window grows by one month per step, testing how models adapt to new data.

**Fallback (_fb)** — When a statistical model (e.g. SARIMA, ETS) fails to fit a series, the pipeline substitutes a seasonal naive forecast and tags it with a `_fb` suffix (e.g. `sarima_fb`). This ensures every model slot produces a prediction.

**Grain** — The combination of columns that uniquely identifies a time series. Inbound grain: `SRC_TADIG x DST_TADIG x CALL_TYPE`. Outbound grain: `SRC_TADIG x DST_COUNTRY x CALL_TYPE`.

**Horserace** — Static train/test evaluation where all 5 models are fitted once on training data (through Dec 2024) and forecast the full test period (Jan-Nov 2025). The model with the lowest WAPE wins per series.

**LightGBM** — Gradient boosting ML model trained globally across all series within a call type. Uses lag, rolling, and trend features. Contrasts with per-series statistical models.

**Portfolio countries** — The 20 countries in scope for forecasting and evaluation. Defined in `config.py` as `PORTFOLIO_COUNTRIES`. All data extraction and modeling is filtered to routes involving these countries.

**Rolling retrain** — Walk-forward evaluation using an expanding window. Starting from month 13, each subsequent month is forecast 1-step ahead using all prior data. Complements the horserace by testing model stability over time.

**Roaming traffic** — Telecommunications usage that occurs when a subscriber uses a network other than their home network:
- **Inbound**: foreign subscribers using the home network (traffic arriving)
- **Outbound**: home subscribers using foreign networks (traffic departing)

**SMAPE** — Symmetric Mean Absolute Percentage Error. `mean(2 * |actual - predicted| / (|actual| + |predicted|))`. Bounded metric that handles near-zero values better than MAPE.

**SRC_TADIG** — Source TADIG code identifying the originating operator. Present in both inbound and outbound grains.

**TADIG** — Transferred Account Data Interchange Group code. A unique identifier for a mobile network operator, assigned by the GSMA. Format is typically a 5-character alphanumeric code (e.g. `GBRCN`).

**DASHBOARD_CALL_TYPES** — The subset of call types displayed on the dashboard and used for production forecasting: `["GPRS", "MOC"]`. Defined in `config.py`.

**Iterative 1-step-ahead** — LightGBM forecasting strategy for multi-month horizons. Each month is predicted individually, and its prediction is written back into the feature matrix so the next month's lag features incorporate it. Avoids the error accumulation of direct multi-step forecasting.

**WAPE** — Weighted Absolute Percentage Error. `sum(|actual - predicted|) / sum(|actual|)`. Primary accuracy metric used for model selection in the horserace. Lower is better; `(1 - WAPE) * 100` gives accuracy percentage.
