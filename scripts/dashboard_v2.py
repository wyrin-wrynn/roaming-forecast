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
    INBOUND_GRAIN, OUTBOUND_GRAIN,
    INBOUND_TARGET, OUTBOUND_TARGET, PORTFOLIO_COUNTRIES, DASHBOARD_CALL_TYPES,
    grain_cols_for, target_for,
)

ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "reports"
DATA = ROOT / "data"
DB_PATH = DATA / "forecasting.db"

# Call types shown in the dashboard (client scope: GPRS + MOC only)

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

MODEL_COLORS = {
    "Actual":          "#636EFA",
    "seasonal_naive":  "#00CC96",
    "ets_damped":      "#EF553B",
    "sarima":          "#AB63FA",
    "theta":           "#FFA15A",
    "lgbm":            "#19D3F3",
    "sarima_fb":       "#FF6692",
    "ets_damped_fb":   "#FF6692",
    "theta_fb":        "#B6E880",
    "Dec 2025 Forecast": "#FF6F61",
    "2026 Forecast":     "#E45756",
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
    ct_list = ",".join(f"'{c}'" for c in DASHBOARD_CALL_TYPES)
    meta = pd.read_sql(
        "SELECT SRC_TADIG, DST_TADIG, CALL_TYPE, DST_NAME, DST_COUNTRY, GROUPNAME, NEGOTIATOR "
        f"FROM traffic WHERE CALL_TYPE IN ({ct_list}) GROUP BY SRC_TADIG, DST_TADIG, CALL_TYPE", conn)
    conn.close()
    return meta.rename(columns={
        "DST_NAME": "Operator", "DST_COUNTRY": "Country",
        "GROUPNAME": "Group", "NEGOTIATOR": "Negotiator",
    })


@st.cache_data
def load_outbound_metadata() -> pd.DataFrame:
    """Per-route metadata for outbound (country-level), with aggregated negotiators/operators."""
    conn = sqlite3.connect(str(DB_PATH))
    ct_list = ",".join(f"'{c}'" for c in DASHBOARD_CALL_TYPES)
    meta = pd.read_sql(
        "SELECT SRC_TADIG, DST_COUNTRY, CALL_TYPE, "
        "GROUP_CONCAT(DISTINCT DST_NAME) AS Operators, "
        "GROUP_CONCAT(DISTINCT NEGOTIATOR) AS Negotiators, "
        "GROUP_CONCAT(DISTINCT GROUPNAME) AS Groups, "
        "COUNT(DISTINCT DST_TADIG) AS n_operators "
        f"FROM traffic WHERE CALL_TYPE IN ({ct_list}) GROUP BY SRC_TADIG, DST_COUNTRY, CALL_TYPE", conn)
    conn.close()
    return meta.rename(columns={"DST_COUNTRY": "Country"})


def _load_direction_data(direction: str):
    """Load actuals, predictions, metrics, winners, rolling for a direction."""
    gc = grain_cols_for(direction)
    actuals = pd.read_parquet(DATA / f"{direction}_set.parquet")
    actuals = actuals[actuals["CALL_TYPE"].isin(DASHBOARD_CALL_TYPES)]
    ym_str = actuals["CALL_YEAR_MONTH"].astype(str).str.zfill(6)
    actuals["date"] = pd.to_datetime(ym_str + "01", format="%Y%m%d")

    preds_path = REPORTS / f"{direction}_horserace_predictions.csv"
    preds = pd.read_csv(preds_path) if preds_path.exists() else pd.DataFrame()
    if not preds.empty:
        preds = preds[preds["CALL_TYPE"].isin(DASHBOARD_CALL_TYPES)]

    metrics_path = REPORTS / f"{direction}_horserace_metrics.csv"
    metrics = pd.read_csv(metrics_path) if metrics_path.exists() else pd.DataFrame()
    if not metrics.empty:
        metrics = metrics[metrics["CALL_TYPE"].isin(DASHBOARD_CALL_TYPES)]

    winners_path = REPORTS / f"{direction}_horserace_winners.csv"
    winners = pd.read_csv(winners_path) if winners_path.exists() else pd.DataFrame()
    if not winners.empty:
        winners = winners[winners["CALL_TYPE"].isin(DASHBOARD_CALL_TYPES)]

    rolling_path = REPORTS / f"{direction}_rolling_accuracy.csv"
    rolling = pd.read_csv(rolling_path) if rolling_path.exists() else pd.DataFrame()
    if not rolling.empty:
        rolling = rolling[rolling["CALL_TYPE"].isin(DASHBOARD_CALL_TYPES)]

    dec25_path = REPORTS / f"{direction}_forecast_dec2025.csv"
    dec25 = pd.read_csv(dec25_path) if dec25_path.exists() else pd.DataFrame()
    if not dec25.empty:
        dec25 = dec25[dec25["CALL_TYPE"].isin(DASHBOARD_CALL_TYPES)]

    fc26_path = REPORTS / f"{direction}_forecast_2026.csv"
    fc26 = pd.read_csv(fc26_path) if fc26_path.exists() else pd.DataFrame()
    if not fc26.empty:
        fc26 = fc26[fc26["CALL_TYPE"].isin(DASHBOARD_CALL_TYPES)]

    return actuals, preds, metrics, winners, rolling, dec25, fc26


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
    """Set session state to navigate to the Explorer page for a given market row."""
    st.session_state["ex_direction"] = direction.capitalize()
    for c in grain_cols:
        st.session_state[f"ex_{direction}_{c}"] = row[c]
    # For inbound, also set DST_COUNTRY so the explorer country filter is correct
    if direction == "inbound" and "DST_COUNTRY" not in grain_cols:
        country = row.get("Country") or row.get("DST_COUNTRY")
        if country:
            st.session_state[f"ex_{direction}_DST_COUNTRY"] = country
    st.session_state["_nav_to_explorer"] = "Explorer"


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

    in_actuals, in_preds, in_metrics, in_winners, in_rolling, _, _ = load_inbound_data()
    out_actuals, out_preds, out_metrics, out_winners, out_rolling, _, _ = load_outbound_data()

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



def _explorer_page():
    """Unified explorer page with Direction as the first filter.

    Layout (top to bottom):
      1. Page title
      2. Filter bar: Direction → Call Type → SRC_TADIG → DST_COUNTRY → DST_TADIG (inbound only)
      3. Route info caption line
      4. Data quality expander (collapsed)
      5. Model toggle pills (Actual + top 3 horserace + best rolling)
      6. Line chart (full 2023-2025 timeline)
      7. Transposed data table (models as rows, months as cols, color-coded)
    """
    st.markdown("<h2 style='font-size:1.4rem'>Explorer</h2>", unsafe_allow_html=True)

    # Direction dropdown
    dir_options = ["Inbound", "Outbound"]
    dir_default = dir_options.index(st.session_state.get("ex_direction", "Inbound"))
    direction = st.selectbox("Direction", dir_options, index=dir_default,
                             key="ex_direction").lower()
    gc = grain_cols_for(direction)

    if direction == "inbound":
        actuals, preds, metrics, winners, rolling, dec25, fc26 = load_inbound_data()
        meta = load_inbound_metadata()
    else:
        actuals, preds, metrics, winners, rolling, dec25, fc26 = load_outbound_data()
        meta = load_outbound_metadata()

    if winners.empty:
        st.warning(f"No {direction} results yet.")
        return

    # Merge metadata onto winners for filter display
    winners_m = _merge_metadata(winners, meta, gc)

    # --- Filter bar: Call Type → SRC_TADIG → DST_COUNTRY → DST_TADIG (inbound) ---
    prefix = f"ex_{direction}"
    working = winners_m.copy()

    # Derive DST_COUNTRY for inbound from metadata if not already present
    if direction == "inbound" and "Country" in working.columns and "DST_COUNTRY" not in working.columns:
        working["DST_COUNTRY"] = working["Country"]

    n_filters = 4 if direction == "inbound" else 3
    filter_cols = st.columns(n_filters)

    # Filter 1: Call Type
    with filter_cols[0]:
        ct_opts = sorted(working["CALL_TYPE"].dropna().unique())
        if not ct_opts:
            st.info("No call types available.")
            return
        sel_ct = st.selectbox("Call Type", ct_opts, key=f"{prefix}_CALL_TYPE")
    working = working[working["CALL_TYPE"] == sel_ct]

    # Filter 2: SRC_TADIG
    with filter_cols[1]:
        src_opts = sorted(working["SRC_TADIG"].dropna().unique())
        if not src_opts:
            st.info("No SRC_TADIG available.")
            return
        sel_src = st.selectbox("SRC_TADIG", src_opts, key=f"{prefix}_SRC_TADIG")
    working = working[working["SRC_TADIG"] == sel_src]

    # Filter 3: DST_COUNTRY
    with filter_cols[2]:
        if direction == "inbound":
            # Derive country from metadata
            country_col = "DST_COUNTRY" if "DST_COUNTRY" in working.columns else "Country"
            country_opts = sorted(working[country_col].dropna().unique())
        else:
            country_opts = sorted(working["DST_COUNTRY"].dropna().unique())
            country_col = "DST_COUNTRY"
        if not country_opts:
            st.info("No countries available.")
            return
        sel_country = st.selectbox("DST_COUNTRY", country_opts, key=f"{prefix}_DST_COUNTRY")
    working = working[working[country_col] == sel_country]

    # Filter 4: DST_TADIG (inbound only)
    if direction == "inbound":
        with filter_cols[3]:
            tadig_opts = sorted(working["DST_TADIG"].dropna().unique())
            if not tadig_opts:
                st.info("No DST_TADIG available.")
                return
            sel_dst = st.selectbox("DST_TADIG", tadig_opts, key=f"{prefix}_DST_TADIG")
        working = working[working["DST_TADIG"] == sel_dst]

    call_type = sel_ct
    target = target_for(call_type, direction)
    target_label = TARGET_LABELS.get(target, target)

    # Build filter dict for this route
    route_filter = {c: working[c].iloc[0] for c in gc}

    # --- Route info caption line ---
    best_model_str = "N/A"
    best_acc_str = ""
    route_winner = working[working["target"] == target]
    if not route_winner.empty:
        rw = route_winner.iloc[0]
        best_model_str = rw["best_model"]
        best_acc_str = f"{(1 - rw['best_wape']) * 100:.1f}%"

    info_parts = []
    if direction == "inbound":
        info_parts.append(f"Operator: **{working.iloc[0].get('Operator', 'N/A')}**")
        info_parts.append(f"Country: **{working.iloc[0].get('Country', 'N/A')}**")
    else:
        info_parts.append(f"Country: **{working.iloc[0].get('Country', sel_country)}**")
    info_parts.append(f"Best Model: **{best_model_str}** ({best_acc_str})")
    st.caption(" · ".join(info_parts))

    # --- Data for this route ---
    route_preds = preds.copy()
    for c, v in route_filter.items():
        if c in route_preds.columns:
            route_preds = route_preds[route_preds[c] == v]
    route_preds = route_preds[route_preds["target"] == target]

    route_rolling_all = rolling.copy() if not rolling.empty else pd.DataFrame()
    for c, v in route_filter.items():
        if c in route_rolling_all.columns:
            route_rolling_all = route_rolling_all[route_rolling_all[c] == v]
    if not route_rolling_all.empty:
        route_rolling_all = route_rolling_all[route_rolling_all["target"] == target]
    route_rolling = route_rolling_all

    # Filter forecast data for this route
    route_dec25 = dec25.copy() if not dec25.empty else pd.DataFrame()
    for c, v in route_filter.items():
        if c in route_dec25.columns:
            route_dec25 = route_dec25[route_dec25[c] == v]
    if not route_dec25.empty:
        route_dec25 = route_dec25[route_dec25["target"] == target]

    route_fc26 = fc26.copy() if not fc26.empty else pd.DataFrame()
    for c, v in route_filter.items():
        if c in route_fc26.columns:
            route_fc26 = route_fc26[route_fc26[c] == v]
    if not route_fc26.empty:
        route_fc26 = route_fc26[route_fc26["target"] == target]

    route_actuals = actuals.copy()
    for c, v in route_filter.items():
        if c in route_actuals.columns:
            route_actuals = route_actuals[route_actuals[c] == v]
    route_actuals = route_actuals.sort_values("CALL_YEAR_MONTH")

    # --- Data quality expander (collapsed) ---
    vals = route_actuals[target].values.astype(float)
    insights = []

    n_zeros = int((vals == 0).sum())
    if n_zeros > 0:
        zero_pct = n_zeros / len(vals) * 100
        insights.append(f"**{n_zeros} zero months** ({zero_pct:.0f}% of history)")

    n_nan = int(np.isnan(vals).sum())
    if n_nan > 0:
        insights.append(f"**{n_nan} missing values**")

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

    nonzero_vals = vals[vals > 0]
    if len(nonzero_vals) >= 2:
        pct_changes = np.abs(np.diff(nonzero_vals) / nonzero_vals[:-1])
        n_spikes = int((pct_changes > 1.0).sum())
        if n_spikes > 0:
            insights.append(f"**{n_spikes} sudden spike(s)/drop(s)** (>100% MoM change)")

    if len(nonzero) > 0 and np.median(nonzero) < 10:
        insights.append("**Very low volume series** (median < 10)")

    if len(nonzero) >= 12:
        cv = np.std(nonzero) / np.mean(nonzero) if np.mean(nonzero) > 0 else 0
        if cv > 1.0:
            insights.append(f"**High variability** (CV = {cv:.2f})")

    if len(vals) >= 24:
        first_12 = vals[:12]
        last_12 = vals[-12:]
        f12_mean = np.nanmean(first_12[first_12 > 0]) if (first_12 > 0).any() else 0
        l12_mean = np.nanmean(last_12[last_12 > 0]) if (last_12 > 0).any() else 0
        if f12_mean > 0:
            yoy_change = (l12_mean - f12_mean) / f12_mean * 100
            if abs(yoy_change) > 30:
                trend_label = "growing" if yoy_change > 0 else "declining"
                insights.append(f"**Strong {trend_label} trend** ({yoy_change:+.0f}% YoY)")

    with st.expander("Data Quality", expanded=False):
        if insights:
            st.info("**Insights:** " + " | ".join(insights))
        else:
            st.success("Clean series — no data quality issues detected.")

    # --- Build model options and compute defaults ---
    # Horserace models: all available, ranked by WAPE
    hr_models = sorted(route_preds["model"].unique()) if not route_preds.empty else []
    # Rolling models: non-fallback only
    roll_models = []
    if not route_rolling.empty:
        # Filter to 2025 test period only (avoid mixing 2024 month numbers)
        route_rolling = route_rolling[route_rolling["forecast_month"] >= 202501]
        roll_primary = route_rolling[~route_rolling["model"].str.endswith("_fb")]
        if not roll_primary.empty:
            roll_models = sorted(roll_primary["model"].unique())

    # Build keyed options: {display_key: (source, model_name)}
    model_options = {}
    for m in hr_models:
        model_options[f"{m} (horserace)"] = ("horserace", m)
    for m in roll_models:
        model_options[f"{m} (rolling)"] = ("rolling", m)

    # --- Build YoY data structures (needed before pills for year labels) ---
    actual_by_ym = dict(zip(
        route_actuals["CALL_YEAR_MONTH"].astype(int),
        route_actuals[target].values.astype(float),
    ))
    # Group actuals by year → {year: {month_num: value}}
    actuals_by_year = {}
    for ym, val in actual_by_ym.items():
        ym_s = str(ym).zfill(6)
        year, month = int(ym_s[:4]), int(ym_s[4:6])
        actuals_by_year.setdefault(year, {})[month] = val
    years_available = sorted(actuals_by_year.keys())

    # Per-year actual labels
    actual_labels = [f"{y} Actual" for y in years_available]

    # Determine defaults: all actuals + best 3 horserace + best 1 rolling
    default_pills = list(actual_labels)
    if not route_winner.empty:
        if not metrics.empty:
            route_metrics = metrics.copy()
            for c, v in route_filter.items():
                if c in route_metrics.columns:
                    route_metrics = route_metrics[route_metrics[c] == v]
            route_metrics = route_metrics[route_metrics["target"] == target]
            if not route_metrics.empty and "wape" in route_metrics.columns:
                top3 = route_metrics.nsmallest(3, "wape")["model"].tolist()
            else:
                top3 = [route_winner.iloc[0]["best_model"]]
        else:
            top3 = [route_winner.iloc[0]["best_model"]]

        for m in top3:
            key = f"{m} (horserace)"
            if key in model_options:
                default_pills.append(key)

    # Best rolling model
    if roll_models and not route_rolling.empty:
        roll_primary = route_rolling[~route_rolling["model"].str.endswith("_fb")]
        if not roll_primary.empty:
            roll_median = roll_primary.groupby("model")["ape"].median()
            best_roll_name = roll_median.idxmin()
            best_roll_key = f"{best_roll_name} (rolling)"
            if best_roll_key in model_options:
                default_pills.append(best_roll_key)

    # Forecast pills
    forecast_pills = []
    if not route_dec25.empty:
        forecast_pills.append("Dec 2025 Forecast")
    if not route_fc26.empty:
        forecast_pills.append("2026 Forecast")
    # Default: include forecasts if available
    default_pills.extend(forecast_pills)

    # All pill options: per-year actuals + model keys + forecasts
    all_pill_options = actual_labels + list(model_options.keys()) + forecast_pills
    default_pills = [p for p in default_pills if p in all_pill_options]
    if not default_pills:
        default_pills = all_pill_options[:3]

    selected_pills = st.pills("Series", all_pill_options, selection_mode="multi",
                              default=default_pills, key=f"{prefix}_pills")
    if selected_pills is None:
        selected_pills = []

    selected_years = [y for y in years_available if f"{y} Actual" in selected_pills]
    selected_models = [k for k in selected_pills if k in model_options]
    show_dec25 = "Dec 2025 Forecast" in selected_pills
    show_fc26 = "2026 Forecast" in selected_pills

    # Build prediction lookups: {sel_key: {month_num: value}} (2025 only)
    hr_preds = {}
    roll_preds = {}
    sel_parsed = []
    for sel_key in selected_models:
        if sel_key not in model_options:
            continue
        source, model_name = model_options[sel_key]
        sel_parsed.append((sel_key, source, model_name))
        if source == "horserace":
            m_data = route_preds[route_preds["model"] == model_name]
            lookup = {}
            for _, r in m_data.iterrows():
                ym_s = str(int(r["CALL_YEAR_MONTH"])).zfill(6)
                lookup[int(ym_s[4:6])] = float(r["predicted"])
            hr_preds[sel_key] = lookup
        else:
            m_data = route_rolling[route_rolling["model"] == model_name]
            lookup = {}
            for _, r in m_data.iterrows():
                ym_s = str(int(r["forecast_month"])).zfill(6)
                lookup[int(ym_s[4:6])] = float(r["predicted"])
            roll_preds[sel_key] = lookup

    month_abbrs = [calendar.month_abbr[m] for m in range(1, 13)]
    year_colors = {2023: "#AAAAAA", 2024: "#888888", 2025: MODEL_COLORS["Actual"]}

    # --- YoY Chart: Jan-Dec x-axis, one line per year/model ---
    fig = go.Figure()

    for year in selected_years:
            monthly = actuals_by_year[year]
            months = sorted(monthly.keys())
            fig.add_trace(go.Scatter(
                x=[calendar.month_abbr[m] for m in months],
                y=[monthly[m] for m in months],
                mode="lines+markers",
                name=f"{year} Actual",
                line=dict(color=year_colors.get(year, "#636EFA"),
                          width=2.5 if year == 2025 else 1.5,
                          dash="solid" if year == 2025 else "dot"),
                marker=dict(size=5),
            ))

    for sel_key in selected_models:
        if sel_key not in model_options:
            continue
        source, model_name = model_options[sel_key]
        color = MODEL_COLORS.get(model_name, "#888888")
        lookup = hr_preds.get(sel_key, {}) if source == "horserace" else roll_preds.get(sel_key, {})
        if lookup:
            months = sorted(lookup.keys())
            fig.add_trace(go.Scatter(
                x=[calendar.month_abbr[m] for m in months],
                y=[lookup[m] for m in months],
                mode="lines+markers",
                name=sel_key,
                line=dict(color=color, width=2,
                          dash="dash" if source == "horserace" else "dot"),
                marker=dict(size=5,
                            symbol="circle" if source == "horserace" else "diamond"),
            ))

    # Dec 2025 forecast point
    if show_dec25 and not route_dec25.empty:
        dec_val = float(route_dec25.iloc[0]["predicted"])
        fig.add_trace(go.Scatter(
            x=["Dec"], y=[dec_val],
            mode="markers", name="Dec 2025 Forecast",
            marker=dict(size=10, color="#FF6F61", symbol="star"),
        ))

    # 2026 forecast line
    if show_fc26 and not route_fc26.empty:
        fc26_lookup = {}
        for _, r in route_fc26.iterrows():
            ym_s = str(int(r["CALL_YEAR_MONTH"])).zfill(6)
            fc26_lookup[int(ym_s[4:6])] = float(r["predicted"])
        if fc26_lookup:
            months_26 = sorted(fc26_lookup.keys())
            fig.add_trace(go.Scatter(
                x=[calendar.month_abbr[m] for m in months_26],
                y=[fc26_lookup[m] for m in months_26],
                mode="lines+markers", name="2026 Forecast",
                line=dict(color="#E45756", width=2.5, dash="dashdot"),
                marker=dict(size=6, symbol="star-triangle-up"),
            ))

    title_parts = [str(route_filter.get(c, "")) for c in gc]
    fig.update_layout(
        title=f"{' | '.join(title_parts)} | {target_label}",
        xaxis_title="", yaxis_title=target_label, height=450,
        xaxis=dict(categoryorder="array",
                   categoryarray=month_abbrs),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        hovermode="x unified", margin=dict(t=60, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)

    # --- YoY Table: rows = year actuals + 2025 forecasts, cols = Jan-Dec + Total ---
    table_rows = []

    # Actual rows per year (only for years selected in pills)
    for year in selected_years:
        monthly = actuals_by_year[year]
        row = {"Series": f"{year} Actual"}
        total = 0.0
        for m in range(1, 13):
            val = monthly.get(m, np.nan)
            row[calendar.month_abbr[m]] = round(float(val), 1) if pd.notna(val) else np.nan
            if pd.notna(val):
                total += val
        row["Total"] = round(total, 1) if total > 0 else np.nan
        table_rows.append(row)

    # 2025 actual lookup for color-coding
    actual_2025 = actuals_by_year.get(2025, {})

    # Forecast model rows (2025 only)
    for sel_key, source, model_name in sel_parsed:
        lookup = hr_preds.get(sel_key, {}) if source == "horserace" else roll_preds.get(sel_key, {})
        row = {"Series": sel_key}
        total = 0.0
        abs_errors = []
        abs_actuals = []
        for m in range(1, 13):
            pred = lookup.get(m, np.nan)
            row[calendar.month_abbr[m]] = round(float(pred), 1) if pd.notna(pred) else np.nan
            if pd.notna(pred):
                total += pred
            actual = actual_2025.get(m, np.nan)
            if pd.notna(pred) and pd.notna(actual) and actual != 0:
                abs_errors.append(abs(pred - actual))
                abs_actuals.append(abs(actual))
        row["Total"] = round(total, 1) if total > 0 else np.nan
        if abs_actuals:
            wape = sum(abs_errors) / sum(abs_actuals)
            row["WAPE"] = f"{wape * 100:.1f}%"
            row["Accuracy %"] = f"{(1 - wape) * 100:.1f}%"
        else:
            row["WAPE"] = ""
            row["Accuracy %"] = ""
        table_rows.append(row)

    # Dec 2025 Forecast row
    if show_dec25 and not route_dec25.empty:
        row = {"Series": "Dec 2025 Forecast"}
        dec_val = float(route_dec25.iloc[0]["predicted"])
        for m in range(1, 13):
            row[calendar.month_abbr[m]] = round(dec_val, 1) if m == 12 else np.nan
        row["Total"] = round(dec_val, 1)
        row["WAPE"] = ""
        row["Accuracy %"] = ""
        table_rows.append(row)

    # 2026 Forecast row
    if show_fc26 and not route_fc26.empty:
        row = {"Series": "2026 Forecast"}
        total = 0.0
        fc26_by_month = {}
        for _, r in route_fc26.iterrows():
            ym_s = str(int(r["CALL_YEAR_MONTH"])).zfill(6)
            fc26_by_month[int(ym_s[4:6])] = float(r["predicted"])
        for m in range(1, 13):
            val = fc26_by_month.get(m, np.nan)
            row[calendar.month_abbr[m]] = round(val, 1) if pd.notna(val) else np.nan
            if pd.notna(val):
                total += val
        row["Total"] = round(total, 1) if total > 0 else np.nan
        row["WAPE"] = ""
        row["Accuracy %"] = ""
        table_rows.append(row)

    if table_rows:
        raw_df = pd.DataFrame(table_rows)
        col_order = ["Series"] + month_abbrs + ["Total"]
        # Add WAPE/Accuracy columns if any forecast rows exist
        if sel_parsed or show_dec25 or show_fc26:
            col_order += ["WAPE", "Accuracy %"]
        col_order = [c for c in col_order if c in raw_df.columns]
        raw_df = raw_df[col_order]

        # Find the 2025 Actual row index for color-coding
        actual_2025_idx = None
        for idx, r in raw_df.iterrows():
            if r["Series"] == "2025 Actual":
                actual_2025_idx = idx
                break

        # Build color styles
        styles_df = pd.DataFrame("", index=raw_df.index, columns=raw_df.columns)
        if actual_2025_idx is not None:
            for row_idx in range(len(raw_df)):
                series = raw_df.iloc[row_idx]["Series"]
                if "Actual" in series or "Forecast" in series:
                    continue  # No coloring for actual or forecast rows
                for ma in month_abbrs:
                    if ma not in raw_df.columns:
                        continue
                    pred_val = raw_df.iloc[row_idx][ma]
                    actual_val = raw_df.loc[actual_2025_idx, ma]
                    if pd.notna(pred_val) and pd.notna(actual_val) and actual_val != 0:
                        ape = abs(float(pred_val) - float(actual_val)) / abs(float(actual_val))
                        if ape <= 0.10:
                            styles_df.iloc[row_idx, styles_df.columns.get_loc(ma)] = "background-color: #d4edda"
                        elif ape <= 0.25:
                            styles_df.iloc[row_idx, styles_df.columns.get_loc(ma)] = "background-color: #fff3cd"
                        else:
                            styles_df.iloc[row_idx, styles_df.columns.get_loc(ma)] = "background-color: #f8d7da"

        # Format for display
        display_df = raw_df.copy()
        fmt_cols = [c for c in display_df.columns if c not in ("Series", "WAPE", "Accuracy %")]
        for col in fmt_cols:
            display_df[col] = display_df[col].apply(
                lambda v: f"{v:,.1f}" if pd.notna(v) else ""
            )

        styled = (display_df.style
                  .apply(lambda _: styles_df, axis=None)
                  .set_properties(subset=["Series"], **{"font-weight": "bold"}))
        st.dataframe(styled, use_container_width=True, hide_index=True,
                     height=min(35 * len(display_df) + 38, 400))


def _forecast_table_page():
    """Filterable forecast table with CSV download and direction selector."""
    st.markdown("<h2 style='font-size:1.4rem'>Forecast Table</h2>", unsafe_allow_html=True)

    dir_options = ["Inbound", "Outbound"]
    dir_default = dir_options.index(st.session_state.get("ft_direction", "Inbound"))
    direction = st.selectbox("Direction", dir_options, index=dir_default,
                             key="ft_direction").lower()
    label = direction.capitalize()
    gc = grain_cols_for(direction)

    if direction == "inbound":
        actuals, preds, metrics, winners, rolling, _, _ = load_inbound_data()
        meta = load_inbound_metadata()
    else:
        actuals, preds, metrics, winners, rolling, _, _ = load_outbound_data()
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
    "Explorer",
    "Forecast Table",
]

if "nav_page" not in st.session_state:
    st.session_state["nav_page"] = PAGE_LIST[0]

page = st.sidebar.radio("Page", PAGE_LIST, key="nav_page")

if page == "Portfolio Overview":
    page_overview()
elif page == "Explorer":
    _explorer_page()
elif page == "Forecast Table":
    _forecast_table_page()
