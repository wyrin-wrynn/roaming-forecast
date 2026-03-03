#!/usr/bin/env python3
"""Pipeline v2 Dashboard — interactive Streamlit app for exploring forecast results.

Reads horserace and rolling retrain outputs from reports/ and route metadata
from data/forecasting.db. Provides 5 pages:

  1. Portfolio Overview — category distribution (Trustworthy → Unreliable),
     direction toggle, example markets with click-to-explore navigation.
  2. Inbound Explorer — drill into a single inbound route with model overlay,
     data quality insights, and month-by-month comparison table.
  3. Outbound Explorer — same as above for outbound (country-level) routes.
  4. Inbound Forecast Table — filterable/sortable table of all routes with
     accuracy, category, data quality flags, and CSV download.
  5. Outbound Forecast Table — same for outbound routes.

Navigation: clicking a row in Overview or Forecast Table sets session state
and navigates to the corresponding Explorer page via st.rerun().

Category assignment logic (categorize_markets):
  - Trustworthy:   static accuracy ≥ 85% AND rolling ≥ 80%
  - Promising:     static ≥ 85%, rolling < 80% or unavailable
  - Review Needed: static 60–85%, no outliers
  - Volatile:      static 60–85%, has outliers (>3× IQR)
  - Unreliable:    static < 60%

Launch:
  streamlit run scripts/dashboard_v2.py --server.port 8505 --server.headless true
"""
from __future__ import annotations

import calendar
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from config import (
    CALL_TYPES, INBOUND_GRAIN, OUTBOUND_GRAIN,
    INBOUND_TARGET, OUTBOUND_TARGET, PORTFOLIO_COUNTRIES,
    grain_cols_for, target_for,
)

ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "reports"
DATA = ROOT / "data"
DB_PATH = DATA / "forecasting.db"

TARGET_LABELS = {
    "INBOUND_CALLS": "Inbound Calls",
    "OUTBOUND_CALLS": "Outbound Calls",
    "INBOUND_VOL_MB": "Inbound Data (MB)",
    "OUTBOUND_VOL_MB": "Outbound Data (MB)",
    "INBOUND_CHARGED_VOLUME_MB": "Inbound Charged Data (MB)",
    "OUTBOUND_CHARGED_VOLUME_MB": "Outbound Charged Data (MB)",
    "INBOUND_DURATION": "Inbound Duration",
    "OUTBOUND_DURATION": "Outbound Duration",
}

MODEL_DESCRIPTIONS = {
    "lgbm": "LightGBM — Global ML model with trend features",
    "sarima_fb": "SARIMA (fallback) — Fell back to seasonal naive",
    "sarima": "SARIMA — Seasonal ARIMA fitted per series",
    "ets_damped": "ETS Damped — Exponential smoothing with damped trend",
    "theta": "Theta — Decomposition-based statistical method",
    "seasonal_naive": "Seasonal Naive — Repeats last year (baseline)",
}

CATEGORY_META = {
    "Trustworthy": {"color": "#00CC96", "icon": "🟢",
                    "definition": "Static accuracy ≥ 85% and rolling ≥ 80%."},
    "Promising": {"color": "#636EFA", "icon": "🔵",
                  "definition": "Static accuracy ≥ 85%, rolling < 80% or unavailable."},
    "Review Needed": {"color": "#FECB52", "icon": "🟡",
                      "definition": "Static accuracy 60–85%, clean data."},
    "Volatile": {"color": "#FFA15A", "icon": "🟠",
                 "definition": "Static accuracy 60–85% with outlier spikes."},
    "Unreliable": {"color": "#EF553B", "icon": "🔴",
                   "definition": "Static accuracy below 60%."},
}
CATEGORY_COLORS = {k: v["color"] for k, v in CATEGORY_META.items()}


def format_month(ym: int) -> str:
    s = str(ym).zfill(6)
    return f"{calendar.month_abbr[int(s[4:6])]} {s[:4]}"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

@st.cache_data
def load_inbound_metadata() -> pd.DataFrame:
    """Per-route metadata for inbound (TADIG-to-TADIG)."""
    conn = sqlite3.connect(str(DB_PATH))
    meta = pd.read_sql(
        "SELECT SRC_TADIG, DST_TADIG, CALL_TYPE, DST_NAME, DST_COUNTRY, GROUPNAME, NEGOTIATOR "
        "FROM traffic GROUP BY SRC_TADIG, DST_TADIG, CALL_TYPE", conn)
    conn.close()
    return meta.rename(columns={
        "DST_NAME": "Operator", "DST_COUNTRY": "Country",
        "GROUPNAME": "Group", "NEGOTIATOR": "Negotiator",
    })


@st.cache_data
def load_outbound_metadata() -> pd.DataFrame:
    """Per-route metadata for outbound (country-level), with aggregated negotiators/operators."""
    conn = sqlite3.connect(str(DB_PATH))
    meta = pd.read_sql(
        "SELECT SRC_TADIG, DST_COUNTRY, CALL_TYPE, "
        "GROUP_CONCAT(DISTINCT DST_NAME) AS Operators, "
        "GROUP_CONCAT(DISTINCT NEGOTIATOR) AS Negotiators, "
        "GROUP_CONCAT(DISTINCT GROUPNAME) AS Groups, "
        "COUNT(DISTINCT DST_TADIG) AS n_operators "
        "FROM traffic GROUP BY SRC_TADIG, DST_COUNTRY, CALL_TYPE", conn)
    conn.close()
    return meta.rename(columns={"DST_COUNTRY": "Country"})


def _load_direction_data(direction: str):
    """Load actuals, predictions, metrics, winners, rolling for a direction."""
    gc = grain_cols_for(direction)
    actuals = pd.read_parquet(DATA / f"{direction}_set.parquet")
    ym_str = actuals["CALL_YEAR_MONTH"].astype(str).str.zfill(6)
    actuals["date"] = pd.to_datetime(ym_str + "01", format="%Y%m%d")

    preds_path = REPORTS / f"{direction}_horserace_predictions.csv"
    preds = pd.read_csv(preds_path) if preds_path.exists() else pd.DataFrame()

    metrics_path = REPORTS / f"{direction}_horserace_metrics.csv"
    metrics = pd.read_csv(metrics_path) if metrics_path.exists() else pd.DataFrame()

    winners_path = REPORTS / f"{direction}_horserace_winners.csv"
    winners = pd.read_csv(winners_path) if winners_path.exists() else pd.DataFrame()

    rolling_path = REPORTS / f"{direction}_rolling_accuracy.csv"
    rolling = pd.read_csv(rolling_path) if rolling_path.exists() else pd.DataFrame()

    return actuals, preds, metrics, winners, rolling


@st.cache_data
def load_inbound_data():
    return _load_direction_data("inbound")

@st.cache_data
def load_outbound_data():
    return _load_direction_data("outbound")


@st.cache_data
def categorize_markets(winners: pd.DataFrame, rolling: pd.DataFrame,
                       actuals: pd.DataFrame, grain_cols: list[str],
                       direction: str) -> pd.DataFrame:
    """Categorize every route into confidence tiers."""
    if winners.empty:
        return pd.DataFrame()

    ms = winners.copy()
    ms["static_accuracy"] = ((1 - ms["best_wape"]) * 100).round(1)

    # Outlier detection per route
    outlier_rows = []
    for ct, ct_df in actuals.groupby("CALL_TYPE"):
        target = target_for(ct, direction)
        for key, grp in ct_df.groupby(grain_cols):
            vals = grp[target].values.astype(float)
            nonzero = vals[vals > 0]
            has_outlier = False
            n_outliers = 0
            if len(nonzero) >= 4:
                q1, q3 = np.percentile(nonzero, [25, 75])
                iqr = q3 - q1
                upper = q3 + 3 * iqr
                n_outliers = int((nonzero > upper).sum())
                has_outlier = n_outliers > 0
            row = dict(zip(grain_cols, key if isinstance(key, tuple) else (key,)))
            row.update({"target": target, "has_outlier": has_outlier, "n_outliers": n_outliers})
            outlier_rows.append(row)
    outlier_df = pd.DataFrame(outlier_rows)
    ms = ms.merge(outlier_df, on=grain_cols + ["target"], how="left")
    ms["has_outlier"] = ms["has_outlier"].fillna(False)

    # Rolling accuracy
    if not rolling.empty:
        # Use primary models (exclude fallbacks)
        primary_models = ["ets_damped", "theta"]
        proper = rolling[
            rolling["model"].isin(primary_models)
            & (rolling["forecast_month"] >= 202501)
        ]
        if not proper.empty:
            roll_acc = (proper.groupby(grain_cols + ["target", "model"])["ape"]
                        .median().reset_index(name="median_ape"))
            roll_acc["rolling_accuracy"] = ((1 - roll_acc["median_ape"]) * 100).round(2)
            best_idx = roll_acc.groupby(grain_cols + ["target"])["rolling_accuracy"].idxmax().dropna()
            best_roll = roll_acc.loc[best_idx, grain_cols + ["target", "model", "rolling_accuracy"]]
            best_roll = best_roll.rename(columns={"model": "best_rolling_model"})
            best_roll = best_roll.drop_duplicates(subset=grain_cols + ["target"])
            ms = ms.merge(best_roll, on=grain_cols + ["target"], how="left")
        else:
            ms["rolling_accuracy"] = np.nan
            ms["best_rolling_model"] = None
    else:
        ms["rolling_accuracy"] = np.nan
        ms["best_rolling_model"] = None

    def _cat(row):
        static = row["static_accuracy"] if pd.notna(row["static_accuracy"]) else 0
        rolling_acc = row["rolling_accuracy"] if pd.notna(row["rolling_accuracy"]) else None
        has_out = row["has_outlier"]
        if static >= 85 and rolling_acc is not None and rolling_acc >= 80:
            return "Trustworthy"
        elif static >= 85:
            return "Promising"
        elif static >= 60 and not has_out:
            return "Review Needed"
        elif static >= 60 and has_out:
            return "Volatile"
        else:
            return "Unreliable"

    ms["Category"] = ms.apply(_cat, axis=1)
    return ms


# ---------------------------------------------------------------------------
# Navigation helper
# ---------------------------------------------------------------------------

def _navigate_to_explorer(row, direction: str, grain_cols: list[str]):
    """Set session state to navigate to the explorer for a given market row."""
    for c in grain_cols:
        st.session_state[f"ex_{direction}_{c}"] = row[c]
    st.session_state["_nav_to_explorer"] = f"{direction.capitalize()} Explorer"


def _merge_metadata(df: pd.DataFrame, meta: pd.DataFrame,
                    gc: list[str]) -> pd.DataFrame:
    """Merge route metadata onto a dataframe, handling the DST_COUNTRY/Country rename."""
    if "DST_COUNTRY" in gc and "Country" in meta.columns and "DST_COUNTRY" not in meta.columns:
        meta = meta.rename(columns={"Country": "DST_COUNTRY"})
    merge_cols = [c for c in gc if c in meta.columns]
    extra_cols = [c for c in meta.columns if c not in gc]
    result = df.merge(meta[merge_cols + extra_cols].drop_duplicates(),
                      on=merge_cols, how="left")
    if "DST_COUNTRY" in result.columns and "Country" not in result.columns:
        result["Country"] = result["DST_COUNTRY"]
    return result


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------

def page_overview():
    st.markdown("<h2 style='font-size:1.4rem'>Portfolio Overview</h2>", unsafe_allow_html=True)

    in_actuals, in_preds, in_metrics, in_winners, in_rolling = load_inbound_data()
    out_actuals, out_preds, out_metrics, out_winners, out_rolling = load_outbound_data()

    # Top-level stats
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Inbound Routes", f"{len(in_winners):,}" if not in_winners.empty else "N/A")
    c2.metric("Outbound Routes", f"{len(out_winners):,}" if not out_winners.empty else "N/A")
    if not in_winners.empty:
        c3.metric("Inbound Median Accuracy", f"{(1 - in_winners['best_wape'].median()) * 100:.1f}%")
    if not out_winners.empty:
        c4.metric("Outbound Median Accuracy", f"{(1 - out_winners['best_wape'].median()) * 100:.1f}%")

    # Direction toggle for charts
    dir_toggle = st.radio("Direction", ["Inbound", "Outbound"], horizontal=True, key="ov_dir")
    direction = dir_toggle.lower()
    gc = grain_cols_for(direction)

    if direction == "inbound":
        actuals, preds, metrics, winners, rolling = in_actuals, in_preds, in_metrics, in_winners, in_rolling
        meta = load_inbound_metadata()
    else:
        actuals, preds, metrics, winners, rolling = out_actuals, out_preds, out_metrics, out_winners, out_rolling
        meta = load_outbound_metadata()

    if winners.empty:
        st.warning(f"No {direction} horserace results yet. Run: `python scripts/run_horserace.py --direction {direction}`")
        return

    # Categorize
    cats = categorize_markets(winners, rolling, actuals, gc, direction)
    if cats.empty:
        return

    # Add display columns from metadata
    cats = _merge_metadata(cats, meta, gc)

    # Category distribution
    st.subheader("Forecast Confidence Distribution")
    cat_counts = cats["Category"].value_counts().reindex(CATEGORY_META.keys()).fillna(0).astype(int)
    fig_cat = px.bar(x=cat_counts.index, y=cat_counts.values, text=cat_counts.values,
                     color=cat_counts.index, color_discrete_map=CATEGORY_COLORS)
    fig_cat.update_traces(textposition="outside")
    fig_cat.update_layout(height=300, showlegend=False, xaxis_title="", yaxis_title="Routes")
    st.plotly_chart(fig_cat, use_container_width=True)

    # Category definitions & example markets
    st.subheader("Category Definitions & Examples")
    st.caption("Click a row to open it in the explorer.")

    cat_order = list(CATEGORY_META.keys())
    for cat_name in cat_order:
        meta_cat = CATEGORY_META[cat_name]
        cat_df = cats[cats["Category"] == cat_name].copy()
        count = len(cat_df)
        pct = count / len(cats) * 100 if len(cats) > 0 else 0

        st.markdown(
            f"#### {meta_cat['icon']} {cat_name} — {count:,} markets ({pct:.0f}%)\n\n"
            f"{meta_cat['definition']}"
        )

        if cat_df.empty:
            st.caption("No markets in this category.")
            continue

        # Pick 5 representative examples
        if cat_name in ("Trustworthy", "Promising"):
            examples = cat_df.nlargest(5, "static_accuracy")
        elif cat_name == "Volatile":
            examples = cat_df.sort_values("n_outliers", ascending=False).head(5)
        elif cat_name == "Unreliable":
            examples = cat_df.nsmallest(5, "static_accuracy")
        else:
            examples = cat_df.sort_values("static_accuracy", ascending=False).head(5)

        # Build display table
        display_cols = list(gc)
        if "Country" in examples.columns:
            display_cols.append("Country")
        if direction == "inbound" and "Operator" in examples.columns:
            display_cols.append("Operator")
        display_cols += ["target", "static_accuracy", "best_model"]
        if "rolling_accuracy" in examples.columns:
            display_cols.append("rolling_accuracy")
        display_cols = [c for c in display_cols if c in examples.columns]

        ex_table = examples[display_cols].copy().reset_index(drop=True)
        ex_display = ex_table.rename(columns={
            "CALL_TYPE": "Call Type", "static_accuracy": "Accuracy %",
            "best_model": "Best Model", "rolling_accuracy": "Rolling Acc %",
            "target": "Target", "DST_COUNTRY": "Dest Country",
        })
        if "Accuracy %" in ex_display.columns:
            ex_display["Accuracy %"] = ex_display["Accuracy %"].round(1)
        if "Rolling Acc %" in ex_display.columns:
            ex_display["Rolling Acc %"] = ex_display["Rolling Acc %"].round(1)

        selection = st.dataframe(
            ex_display, use_container_width=True, hide_index=True,
            on_select="rerun", selection_mode="single-row",
            key=f"ex_{direction}_{cat_name}",
        )
        if selection and selection.selection and selection.selection.rows:
            sel_idx = selection.selection.rows[0]
            ex_nav_key = f"ex_{direction}_{cat_name}_last"
            if st.session_state.get(ex_nav_key) != sel_idx:
                row = ex_table.iloc[sel_idx]
                st.session_state[ex_nav_key] = sel_idx
                _navigate_to_explorer(row, direction, gc)
                st.rerun()

    # Model winner distribution
    st.subheader("Best Model Distribution")
    model_counts = winners["best_model"].value_counts().reset_index()
    model_counts.columns = ["Model", "Count"]
    fig_m = px.bar(model_counts, x="Model", y="Count", text="Count", color="Model")
    fig_m.update_traces(textposition="outside")
    fig_m.update_layout(height=300, showlegend=False)
    st.plotly_chart(fig_m, use_container_width=True)



def _explorer_page(direction: str):
    """Shared explorer logic for inbound/outbound."""
    label = direction.capitalize()
    gc = grain_cols_for(direction)
    target_map = INBOUND_TARGET if direction == "inbound" else OUTBOUND_TARGET

    st.markdown(f"<h2 style='font-size:1.4rem'>{label} Explorer</h2>", unsafe_allow_html=True)

    if direction == "inbound":
        actuals, preds, metrics, winners, rolling = load_inbound_data()
        meta = load_inbound_metadata()
    else:
        actuals, preds, metrics, winners, rolling = load_outbound_data()
        meta = load_outbound_metadata()

    if winners.empty:
        st.warning(f"No {direction} results yet.")
        return

    # Merge metadata onto winners for filter display
    winners_m = _merge_metadata(winners, meta, gc)

    # Filters
    prefix = f"ex_{direction}"
    filter_cols = st.columns(len(gc))
    working = winners_m.copy()

    for i, col in enumerate(gc):
        with filter_cols[i]:
            opts = sorted(working[col].dropna().unique())
            if not opts:
                st.info(f"No {col}.")
                return
            sel = st.selectbox(col, opts, key=f"{prefix}_{col}")
        working = working[working[col] == sel]

    call_type = working["CALL_TYPE"].iloc[0] if not working.empty else CALL_TYPES[0]
    target = target_for(call_type, direction)
    target_label = TARGET_LABELS.get(target, target)

    # Route info
    route_winner = pd.DataFrame()
    if not working.empty:
        info = working.iloc[0]
        info_cols = st.columns(5)
        if direction == "inbound":
            info_cols[0].metric("Operator", info.get("Operator", "N/A"))
            info_cols[1].metric("Country", info.get("Country", "N/A"))
            info_cols[2].metric("Negotiator", info.get("Negotiator", "N/A"))
        else:
            info_cols[0].metric("Country", info.get("Country", "N/A"))
            info_cols[1].metric("Operators", str(info.get("n_operators", "N/A")))
            info_cols[2].metric("Negotiators", str(info.get("Negotiators", "N/A"))[:30])

        route_winner = working[working["target"] == target]
        if not route_winner.empty:
            rw = route_winner.iloc[0]
            info_cols[3].metric("Horserace Best", f"{rw['best_model']} ({(1 - rw['best_wape']) * 100:.1f}%)")

    # Build filter for this route
    route_filter = {c: working[c].iloc[0] for c in gc}

    # Find best rolling model for info display
    route_rolling_all = rolling.copy() if not rolling.empty else pd.DataFrame()
    for c, v in route_filter.items():
        if c in route_rolling_all.columns:
            route_rolling_all = route_rolling_all[route_rolling_all[c] == v]
    if not route_rolling_all.empty:
        route_rolling_all = route_rolling_all[route_rolling_all["target"] == target]

    if not working.empty and not route_rolling_all.empty:
        roll_primary = route_rolling_all[~route_rolling_all["model"].str.endswith("_fb")]
        if not roll_primary.empty:
            roll_med = roll_primary.groupby("model")["ape"].median()
            best_roll_name = roll_med.idxmin()
            best_roll_acc = (1 - roll_med.min()) * 100
            info_cols[4].metric("Rolling Best", f"{best_roll_name} ({best_roll_acc:.1f}%)")
        else:
            info_cols[4].metric("Rolling Best", "N/A")
    elif not working.empty:
        info_cols[4].metric("Rolling Best", "N/A")

    # Model selection
    route_preds = preds.copy()
    for c, v in route_filter.items():
        if c in route_preds.columns:
            route_preds = route_preds[route_preds[c] == v]
    route_preds = route_preds[route_preds["target"] == target]

    route_rolling = route_rolling_all

    model_options = {}
    for m in sorted(route_preds["model"].unique()) if not route_preds.empty else []:
        model_options[f"{m} (horserace)"] = ("horserace", m)
    for m in sorted(route_rolling["model"].unique()) if not route_rolling.empty else []:
        model_options[f"{m} (rolling)"] = ("rolling", m)

    if model_options:
        # Default to best horserace + best rolling model
        best_hr = route_winner.iloc[0]["best_model"] if not route_winner.empty else None
        best_hr_key = f"{best_hr} (horserace)" if best_hr and f"{best_hr} (horserace)" in model_options else None

        # Find best rolling model (lowest median APE)
        best_roll_key = None
        if not route_rolling.empty:
            roll_primary = route_rolling[~route_rolling["model"].str.endswith("_fb")]
            if not roll_primary.empty:
                roll_median = roll_primary.groupby("model")["ape"].median()
                best_roll_name = roll_median.idxmin()
                candidate = f"{best_roll_name} (rolling)"
                if candidate in model_options:
                    best_roll_key = candidate

        defaults = [k for k in [best_hr_key, best_roll_key] if k is not None]
        if not defaults:
            defaults = list(model_options.keys())[:2]

        selected = st.multiselect("Models", list(model_options.keys()),
                                  default=defaults, key=f"{prefix}_models")
    else:
        selected = []

    # Chart
    route_actuals = actuals.copy()
    for c, v in route_filter.items():
        if c in route_actuals.columns:
            route_actuals = route_actuals[route_actuals[c] == v]
    route_actuals = route_actuals.sort_values("CALL_YEAR_MONTH")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=route_actuals["date"], y=route_actuals[target],
        mode="lines+markers", name="Actual",
        line=dict(color="#636EFA", width=2.5), marker=dict(size=5),
    ))

    palette = ["#00CC96", "#EF553B", "#AB63FA", "#FFA15A", "#19D3F3", "#FF6692", "#B6E880"]
    for i, sel_key in enumerate(selected):
        source, model_name = model_options[sel_key]
        color = palette[i % len(palette)]
        if source == "horserace":
            m_data = route_preds[route_preds["model"] == model_name].sort_values("CALL_YEAR_MONTH")
            if not m_data.empty:
                ym = m_data["CALL_YEAR_MONTH"].astype(str).str.zfill(6)
                dates = pd.to_datetime(ym + "01", format="%Y%m%d")
                fig.add_trace(go.Scatter(
                    x=dates, y=m_data["predicted"], mode="lines+markers", name=sel_key,
                    line=dict(color=color, width=2, dash="dash"), marker=dict(size=5),
                ))
        else:
            m_data = route_rolling[route_rolling["model"] == model_name].sort_values("forecast_month")
            if not m_data.empty:
                ym = m_data["forecast_month"].astype(str).str.zfill(6)
                dates = pd.to_datetime(ym + "01", format="%Y%m%d")
                fig.add_trace(go.Scatter(
                    x=dates, y=m_data["predicted"], mode="lines+markers", name=sel_key,
                    line=dict(color=color, width=2, dash="dot"), marker=dict(size=5, symbol="diamond"),
                ))

    fig.add_vline(x="2025-01-01", line_dash="dash", line_color="gray", opacity=0.5)
    title_parts = [str(route_filter.get(c, "")) for c in gc]
    fig.update_layout(
        title=f"{' | '.join(title_parts)} | {target_label}",
        xaxis_title="", yaxis_title=target_label, height=450,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        hovermode="x unified", margin=dict(t=60, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)

    # --------------- Series insights ---------------
    st.subheader("Series Data & Insights")

    vals = route_actuals[target].values.astype(float)
    insights = []

    # Zero months
    n_zeros = int((vals == 0).sum())
    if n_zeros > 0:
        zero_pct = n_zeros / len(vals) * 100
        insights.append(f"**{n_zeros} zero months** ({zero_pct:.0f}% of history)")

    # Missing / NaN
    n_nan = int(np.isnan(vals).sum())
    if n_nan > 0:
        insights.append(f"**{n_nan} missing values**")

    # Outlier detection (IQR)
    nonzero = vals[(vals > 0) & ~np.isnan(vals)]
    if len(nonzero) >= 4:
        q1, q3 = np.percentile(nonzero, [25, 75])
        iqr = q3 - q1
        upper = q3 + 3 * iqr
        lower = q1 - 3 * iqr
        outlier_high = nonzero[nonzero > upper]
        outlier_low = nonzero[nonzero < lower] if lower > 0 else np.array([])
        n_outliers = len(outlier_high) + len(outlier_low)
        if n_outliers > 0:
            insights.append(f"**{n_outliers} outlier(s)** detected (>3× IQR)")

    # Sudden spikes/drops (>100% month-over-month change)
    nonzero_vals = vals[vals > 0]
    if len(nonzero_vals) >= 2:
        pct_changes = np.abs(np.diff(nonzero_vals) / nonzero_vals[:-1])
        n_spikes = int((pct_changes > 1.0).sum())
        if n_spikes > 0:
            insights.append(f"**{n_spikes} sudden spike(s)/drop(s)** (>100% MoM change)")

    # Low volume warning
    if len(nonzero) > 0 and np.median(nonzero) < 10:
        insights.append("**Very low volume series** (median < 10)")

    # Seasonality strength (coefficient of variation)
    if len(nonzero) >= 12:
        cv = np.std(nonzero) / np.mean(nonzero) if np.mean(nonzero) > 0 else 0
        if cv > 1.0:
            insights.append(f"**High variability** (CV = {cv:.2f})")

    # YoY trend
    if len(vals) >= 24:
        first_12 = vals[:12]
        last_12 = vals[-12:]
        f12_mean = np.nanmean(first_12[first_12 > 0]) if (first_12 > 0).any() else 0
        l12_mean = np.nanmean(last_12[last_12 > 0]) if (last_12 > 0).any() else 0
        if f12_mean > 0:
            yoy_change = (l12_mean - f12_mean) / f12_mean * 100
            if abs(yoy_change) > 30:
                direction_label = "growing" if yoy_change > 0 else "declining"
                insights.append(f"**Strong {direction_label} trend** ({yoy_change:+.0f}% YoY)")

    # Display insights
    if insights:
        st.info("**Insights:** " + " | ".join(insights))
    else:
        st.success("Clean series — no data quality issues detected.")

    # --------------- Full table — all months with actuals + selected model predictions ---------------
    sel_parsed = []
    if selected:
        for sel_key in selected:
            if sel_key in model_options:
                source, model_name = model_options[sel_key]
                sel_parsed.append((sel_key, source, model_name))

    # Build prediction lookup per model
    hr_preds = {}  # {sel_key: {ym: predicted}}
    roll_preds = {}
    for sel_key, source, model_name in sel_parsed:
        if source == "horserace":
            m_data = route_preds[route_preds["model"] == model_name]
            hr_preds[sel_key] = dict(zip(m_data["CALL_YEAR_MONTH"].astype(int), m_data["predicted"]))
        else:
            m_data = route_rolling[route_rolling["model"] == model_name]
            roll_preds[sel_key] = dict(zip(m_data["forecast_month"].astype(int), m_data["predicted"]))

    # Build table from all actual months
    comp_rows = []
    for _, r in route_actuals.iterrows():
        ym = int(r["CALL_YEAR_MONTH"])
        actual = float(r[target])
        row = {"Month": format_month(ym), "Actual": round(actual, 1)}

        for sel_key, source, model_name in sel_parsed:
            lookup = hr_preds.get(sel_key, {}) if source == "horserace" else roll_preds.get(sel_key, {})
            pred = lookup.get(ym, np.nan)
            row[sel_key] = round(float(pred), 1) if pd.notna(pred) else None
            if pd.notna(pred) and actual != 0:
                row[f"{sel_key} Err%"] = round(abs(actual - float(pred)) / abs(actual) * 100, 1)
            else:
                row[f"{sel_key} Err%"] = None

        comp_rows.append(row)

    if comp_rows:
        comp_df = pd.DataFrame(comp_rows)
        st.dataframe(comp_df, use_container_width=True, hide_index=True,
                     height=min(35 * len(comp_df) + 38, 600))

        # Overall accuracy per selected model (test period only)
        if sel_parsed:
            acc_rows = []
            for sel_key, source, model_name in sel_parsed:
                if sel_key in comp_df.columns:
                    valid = comp_df[["Actual", sel_key]].dropna()
                    if not valid.empty and valid["Actual"].abs().sum() > 0:
                        denom = valid["Actual"].abs().sum()
                        wape = (valid["Actual"] - valid[sel_key]).abs().sum() / denom
                        acc_rows.append({"Model": sel_key, "WAPE": f"{wape*100:.1f}%",
                                         "Accuracy": f"{(1-wape)*100:.1f}%"})
            if acc_rows:
                st.dataframe(pd.DataFrame(acc_rows), use_container_width=True, hide_index=True)


def _forecast_table_page(direction: str):
    """Filterable forecast table with CSV download."""
    label = direction.capitalize()
    gc = grain_cols_for(direction)

    st.markdown(f"<h2 style='font-size:1.4rem'>{label} Forecast Table</h2>", unsafe_allow_html=True)

    if direction == "inbound":
        actuals, preds, metrics, winners, rolling = load_inbound_data()
        meta = load_inbound_metadata()
    else:
        actuals, preds, metrics, winners, rolling = load_outbound_data()
        meta = load_outbound_metadata()

    if winners.empty:
        st.warning(f"No {direction} results yet.")
        return

    # Merge metadata
    table = _merge_metadata(winners, meta, gc)

    table["Accuracy %"] = ((1 - table["best_wape"]) * 100).round(1)
    table["Target"] = table["target"].map(TARGET_LABELS)

    # Categorize
    cats = categorize_markets(winners, rolling, actuals, gc, direction)
    if not cats.empty:
        cat_cols = gc + ["target", "Category"]
        cat_merge = cats[[c for c in cat_cols if c in cats.columns]].drop_duplicates()
        table = table.merge(cat_merge, on=gc + ["target"], how="left")

    # --- Compute per-series data quality stats from actuals ---
    from config import TRAIN_END, TEST_START, TEST_END
    quality_rows = []
    for ct, ct_df in actuals.groupby("CALL_TYPE"):
        tgt = target_for(ct, direction)
        for key, grp in ct_df.groupby(gc):
            vals = grp[tgt].values.astype(float)
            months = grp["CALL_YEAR_MONTH"].values
            train_vals = vals[months <= TRAIN_END]
            test_vals = vals[(months >= TEST_START) & (months <= TEST_END)]

            n_zeros = int((vals == 0).sum())
            nonzero = vals[(vals > 0) & ~np.isnan(vals)]
            has_outlier = False
            if len(nonzero) >= 4:
                q1, q3 = np.percentile(nonzero, [25, 75])
                iqr = q3 - q1
                has_outlier = bool((nonzero > q3 + 3 * iqr).any())

            row = dict(zip(gc, key if isinstance(key, tuple) else (key,)))
            row.update({
                "target": tgt,
                "Train Obs": int(len(train_vals)),
                "Test Obs": int(len(test_vals)),
                "Zeros": n_zeros,
                "Has Outlier": has_outlier,
                "Full Data": len(train_vals) >= 24 and len(test_vals) >= 11,
            })
            quality_rows.append(row)

    quality_df = pd.DataFrame(quality_rows)
    table = table.merge(quality_df, on=gc + ["target"], how="left")
    table["Zeros"] = table["Zeros"].fillna(0).astype(int)
    table["Has Outlier"] = table["Has Outlier"].fillna(False)
    table["Full Data"] = table["Full Data"].fillna(False)

    # Add best rolling model info from categorized data
    if not cats.empty and "best_rolling_model" in cats.columns:
        roll_cols = gc + ["target", "best_rolling_model", "rolling_accuracy"]
        roll_merge = cats[[c for c in roll_cols if c in cats.columns]].drop_duplicates()
        roll_merge = roll_merge.rename(columns={
            "best_rolling_model": "Rolling Model",
            "rolling_accuracy": "Rolling Acc %",
        })
        table = table.merge(roll_merge, on=gc + ["target"], how="left")

    # Filters
    prefix = f"ft_{direction}"
    fc1, fc2, fc3 = st.columns(3)
    with fc1:
        ct_opts = ["All"] + sorted(table["CALL_TYPE"].dropna().unique())
        sel_ct = st.selectbox("Call Type", ct_opts, key=f"{prefix}_ct")
    if sel_ct != "All":
        table = table[table["CALL_TYPE"] == sel_ct]

    if "Country" in table.columns:
        with fc2:
            country_opts = ["All"] + sorted(table["Country"].dropna().unique())
            sel_country = st.selectbox("Country", country_opts, key=f"{prefix}_country")
        if sel_country != "All":
            table = table[table["Country"] == sel_country]

    if "Category" in table.columns:
        with fc3:
            cat_opts = ["All"] + sorted(table["Category"].dropna().unique())
            sel_cat = st.selectbox("Category", cat_opts, key=f"{prefix}_cat")
        if sel_cat != "All":
            table = table[table["Category"] == sel_cat]

    # Data quality checkbox filters
    qc1, qc2, qc3 = st.columns(3)
    with qc1:
        only_full = st.checkbox("Full data only (24 train + 11 test)", key=f"{prefix}_full")
    with qc2:
        only_nonzero = st.checkbox("No zeros in history", key=f"{prefix}_nonzero")
    with qc3:
        no_outliers = st.checkbox("No outliers", key=f"{prefix}_nooutlier")

    if only_full:
        table = table[table["Full Data"]]
    if only_nonzero:
        table = table[table["Zeros"] == 0]
    if no_outliers:
        table = table[~table["Has Outlier"]]

    # Display columns
    display_cols = gc.copy()
    if direction == "inbound" and "Operator" in table.columns:
        display_cols.append("Operator")
    if "Country" in table.columns:
        display_cols.append("Country")
    display_cols += ["Target", "best_model", "Accuracy %", "Rolling Model", "Rolling Acc %", "Train Obs", "Test Obs", "Zeros"]
    if "Category" in table.columns:
        display_cols.append("Category")
    display_cols = [c for c in display_cols if c in table.columns]

    sorted_table = table.sort_values("Accuracy %", ascending=True).reset_index(drop=True)

    selection = st.dataframe(
        sorted_table[display_cols],
        use_container_width=True, hide_index=True, height=500,
        on_select="rerun", selection_mode="single-row",
        key=f"{prefix}_table",
    )

    if selection and selection.selection and selection.selection.rows:
        sel_idx = selection.selection.rows[0]
        # Only navigate if this is a new selection (different from last navigated row)
        last_nav = st.session_state.get(f"{prefix}_last_nav")
        current_key = (sel_idx, len(sorted_table))
        if last_nav != current_key:
            row = sorted_table.iloc[sel_idx]
            st.session_state[f"{prefix}_last_nav"] = current_key
            _navigate_to_explorer(row, direction, gc)
            st.rerun()

    st.caption(f"{len(sorted_table)} routes · Click a row to open in explorer")

    # CSV download
    csv = table[display_cols].to_csv(index=False)
    st.download_button(f"Download {label} CSV", csv, f"{direction}_forecast_table.csv",
                       "text/csv", key=f"{prefix}_dl")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Forecast Dashboard v2", layout="wide")

st.markdown("""<style>
div[data-testid="stMetric"] label { font-size: 0.72rem !important; }
div[data-testid="stMetric"] [data-testid="stMetricValue"] { font-size: 1rem !important; }
</style>""", unsafe_allow_html=True)

# Handle pending navigation from example market clicks
if st.session_state.get("_nav_to_explorer"):
    st.session_state["nav_page"] = st.session_state.pop("_nav_to_explorer")

PAGE_LIST = [
    "Portfolio Overview",
    "Inbound Explorer",
    "Outbound Explorer",
    "Inbound Forecast Table",
    "Outbound Forecast Table",
]

if "nav_page" not in st.session_state:
    st.session_state["nav_page"] = PAGE_LIST[0]

page = st.sidebar.radio("Page", PAGE_LIST, key="nav_page")

if page == "Portfolio Overview":
    page_overview()
elif page == "Inbound Explorer":
    _explorer_page("inbound")
elif page == "Outbound Explorer":
    _explorer_page("outbound")
elif page == "Inbound Forecast Table":
    _forecast_table_page("inbound")
elif page == "Outbound Forecast Table":
    _forecast_table_page("outbound")
