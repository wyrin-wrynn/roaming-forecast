"""Shared model library for the roaming forecasting pipeline.

Extracted from run_trend_horserace.py so that both horserace and rolling
retrain scripts can import the same implementations.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.forecasting.theta import ThetaModel

from config import INBOUND_GRAIN


# --------------- Per-series models ---------------

def forecast_seasonal_naive(train: np.ndarray, horizon: int) -> np.ndarray:
    """Repeat last year's values."""
    last_12 = train[-12:]
    n = len(last_12)
    reps = (horizon // n) + 1
    return np.clip(np.tile(last_12, reps)[:horizon], 0, None)


def forecast_ets(train: np.ndarray, horizon: int,
                 damped: bool = True, trend: str = "add") -> np.ndarray | None:
    """ETS with configurable trend/damping and additive seasonal."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = ExponentialSmoothing(
                train, trend=trend, damped_trend=damped,
                seasonal="add", seasonal_periods=12,
            ).fit(optimized=True, use_brute=False)
            fc = np.asarray(model.forecast(horizon))
            if not np.any(np.isnan(fc)) and not np.any(np.isinf(fc)):
                return np.clip(fc, 0, None)
    except Exception:
        pass
    return None


def forecast_sarima(train: np.ndarray, horizon: int,
                    order: tuple | None = None,
                    seasonal_order: tuple | None = None) -> np.ndarray | None:
    """SARIMA via pmdarima. Optionally reuse a pre-fitted (order, seasonal_order)."""
    try:
        import pmdarima as pm
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if order is not None and seasonal_order is not None:
                from statsmodels.tsa.statespace.sarimax import SARIMAX
                model = SARIMAX(train, order=order, seasonal_order=seasonal_order,
                                enforce_stationarity=False, enforce_invertibility=False)
                result = model.fit(disp=False, maxiter=50)
                fc = result.forecast(horizon)
            else:
                model = pm.auto_arima(
                    train, seasonal=True, m=12,
                    start_p=0, start_q=0, max_p=2, max_q=2,
                    start_P=0, start_Q=0, max_P=1, max_Q=1,
                    max_d=1, max_D=1,
                    stepwise=True, suppress_warnings=True, error_action="ignore",
                )
                fc = model.predict(n_periods=horizon)
            if not np.any(np.isnan(fc)) and not np.any(np.isinf(fc)):
                return np.clip(np.asarray(fc), 0, None)
    except Exception:
        pass
    return None


def fit_sarima_order(train: np.ndarray) -> tuple[tuple, tuple] | None:
    """Run auto_arima and return (order, seasonal_order) for reuse."""
    try:
        import pmdarima as pm
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = pm.auto_arima(
                train, seasonal=True, m=12,
                start_p=0, start_q=0, max_p=2, max_q=2,
                start_P=0, start_Q=0, max_P=1, max_Q=1,
                max_d=1, max_D=1,
                stepwise=True, suppress_warnings=True, error_action="ignore",
            )
            return model.order, model.seasonal_order
    except Exception:
        return None


def forecast_theta(train: np.ndarray, horizon: int) -> np.ndarray | None:
    """Theta model with seasonal decomposition."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            idx = pd.period_range(start="2023-01", periods=len(train), freq="M")
            series = pd.Series(train, index=idx)
            model = ThetaModel(series, period=12, deseasonalize=True)
            result = model.fit()
            fc = result.forecast(horizon).values
            if not np.any(np.isnan(fc)) and not np.any(np.isinf(fc)):
                return np.clip(fc, 0, None)
    except Exception:
        pass
    return None


# --------------- 1-step variants for rolling retrain ---------------

def forecast_seasonal_naive_1step(train: np.ndarray) -> float:
    """Forecast next month = same month last year."""
    if len(train) >= 12:
        return max(0.0, float(train[-12]))
    return max(0.0, float(train[-1]))


def forecast_ets_damped_1step(train: np.ndarray) -> float | None:
    """1-step ETS with damped trend, additive seasonal."""
    if len(train) < 24:
        return None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = ExponentialSmoothing(
                train, trend="add", damped_trend=True, seasonal="add", seasonal_periods=12,
            ).fit(optimized=True, use_brute=False)
            fc = float(model.forecast(1)[0])
            if np.isnan(fc) or np.isinf(fc):
                return None
            return max(0.0, fc)
    except Exception:
        return None


def forecast_sarima_1step(train: np.ndarray,
                          order: tuple | None = None,
                          seasonal_order: tuple | None = None) -> float | None:
    """1-step SARIMA, optionally reusing order from auto_arima."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if order is not None and seasonal_order is not None:
                from statsmodels.tsa.statespace.sarimax import SARIMAX
                model = SARIMAX(train, order=order, seasonal_order=seasonal_order,
                                enforce_stationarity=False, enforce_invertibility=False)
                result = model.fit(disp=False, maxiter=50)
                fc = float(result.forecast(1)[0])
            else:
                import pmdarima as pm
                model = pm.auto_arima(
                    train, seasonal=True, m=12,
                    start_p=0, start_q=0, max_p=2, max_q=2,
                    start_P=0, start_Q=0, max_P=1, max_Q=1,
                    max_d=1, max_D=1,
                    stepwise=True, suppress_warnings=True, error_action="ignore",
                )
                fc = float(model.predict(n_periods=1)[0])
            if np.isnan(fc) or np.isinf(fc):
                return None
            return max(0.0, fc)
    except Exception:
        return None


def forecast_theta_1step(train: np.ndarray) -> float | None:
    """1-step Theta model."""
    if len(train) < 12:
        return None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            idx = pd.period_range(start="2023-01", periods=len(train), freq="M")
            series = pd.Series(train, index=idx)
            model = ThetaModel(series, period=12, deseasonalize=True)
            result = model.fit()
            fc = float(result.forecast(1).values[0])
            if np.isnan(fc) or np.isinf(fc):
                return None
            return max(0.0, fc)
    except Exception:
        return None


# --------------- Metrics ---------------

def wape(y_true, y_pred):
    """Weighted absolute percentage error."""
    d = np.abs(y_true).sum()
    return float(np.abs(y_true - y_pred).sum() / d) if d > 0 else np.nan


def wape_single(y_true: float, y_pred: float) -> float:
    """Single-observation APE."""
    if abs(y_true) == 0:
        return np.nan
    return abs(y_true - y_pred) / abs(y_true)


def smape_metric(y_true, y_pred):
    """Symmetric mean absolute percentage error."""
    d = np.abs(y_true) + np.abs(y_pred)
    m = d > 0
    return float((2 * np.abs(y_true[m] - y_pred[m]) / d[m]).mean()) if m.any() else np.nan


# --------------- LightGBM helpers ---------------

def build_lgbm_features(df: pd.DataFrame, target: str,
                        grain_cols: list[str] | None = None,
                        add_trend: bool = True) -> pd.DataFrame:
    """Build LightGBM feature matrix for a given target.

    Parameters
    ----------
    grain_cols : list[str], optional
        Columns defining a unique series. Defaults to GRAIN_COLS (inbound grain).
    add_trend : bool
        Include trend features (yoy_ratio, trend_slope_6, etc). Default True in v2.
    """
    gc = grain_cols if grain_cols is not None else INBOUND_GRAIN
    df = df.sort_values(gc + ["CALL_YEAR_MONTH"]).copy()
    df["series_code"] = (
        df[gc].astype(str).agg("_".join, axis=1)
    ).astype("category").cat.codes
    df["month"] = df["MONTH"]
    df["quarter"] = (df["MONTH"] - 1) // 3 + 1

    for lag in [1, 2, 3, 6, 12]:
        df[f"lag_{lag}"] = df.groupby(gc)[target].shift(lag)

    for window in [3, 6, 12]:
        df[f"roll_mean_{window}"] = df.groupby(gc)[target].transform(
            lambda x: x.shift(1).rolling(window, min_periods=1).mean()
        )
        df[f"roll_std_{window}"] = df.groupby(gc)[target].transform(
            lambda x: x.shift(1).rolling(window, min_periods=1).std()
        )

    if add_trend:
        df["yoy_ratio"] = df.groupby(gc)[target].transform(
            lambda x: x.shift(1) / x.shift(13).replace(0, np.nan)
        )
        df["trend_slope_6"] = df.groupby(gc)[target].transform(
            lambda x: x.shift(1).rolling(6, min_periods=3).apply(
                lambda v: np.polyfit(range(len(v)), v, 1)[0] / (v.mean() + 1e-9) if len(v) >= 3 else 0,
                raw=False,
            )
        )
        df["growth_momentum"] = df.groupby(gc)[target].transform(
            lambda x: x.shift(1).rolling(6, min_periods=3).mean() /
                      x.shift(7).rolling(6, min_periods=3).mean().replace(0, np.nan)
        )
        df["time_idx"] = df.groupby(gc).cumcount()

    df["y"] = df[target]
    return df


LGBM_BASE_FEATURES = (
    ["series_code", "month", "quarter"]
    + [f"lag_{l}" for l in [1, 2, 3, 6, 12]]
    + [f"roll_mean_{w}" for w in [3, 6, 12]]
    + [f"roll_std_{w}" for w in [3, 6, 12]]
)
LGBM_TREND_FEATURES = ["yoy_ratio", "trend_slope_6", "growth_momentum", "time_idx"]


def forecast_lgbm_global(df, target, train_mask, test_mask,
                         grain_cols: list[str] | None = None,
                         add_trend: bool = True):
    """Train a global LightGBM model and return predictions.

    Parameters
    ----------
    grain_cols : list[str], optional
        Columns defining a unique series. Defaults to GRAIN_COLS (inbound grain).
    add_trend : bool
        Include trend features. Default True in v2.
    """
    gc = grain_cols if grain_cols is not None else INBOUND_GRAIN
    try:
        import lightgbm as lgb
        feat_df = build_lgbm_features(df, target, grain_cols=gc, add_trend=add_trend)

        feature_cols = list(LGBM_BASE_FEATURES)
        if add_trend:
            feature_cols += LGBM_TREND_FEATURES

        train_feat = feat_df[train_mask].dropna(subset=feature_cols)
        test_feat = feat_df[test_mask].copy()
        if len(train_feat) < 100:
            return None
        for col in feature_cols:
            test_feat[col] = test_feat[col].fillna(train_feat[col].median())

        dtrain = lgb.Dataset(train_feat[feature_cols], train_feat["y"], free_raw_data=False)
        params = {
            "objective": "regression", "metric": "mae", "num_leaves": 31,
            "learning_rate": 0.05, "feature_fraction": 0.8,
            "bagging_fraction": 0.8, "bagging_freq": 5, "verbose": -1, "n_jobs": -1,
        }
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = lgb.train(params, dtrain, num_boost_round=300)

        preds = model.predict(test_feat[feature_cols])
        test_feat["pred"] = np.clip(preds, 0, None)
        return test_feat[gc + ["CALL_YEAR_MONTH", "pred"]]
    except Exception as e:
        print(f"  LightGBM error: {e}", flush=True)
        return None
