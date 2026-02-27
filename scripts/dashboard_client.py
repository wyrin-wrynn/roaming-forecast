#!/usr/bin/env python3
"""Client Forecast Dashboard — simplified view for account managers.

Views: Summary | Route Explorer | Route Detail (actual vs static vs rolling)
Filtered to 20 client countries. Sidebar: search + cascading filters.
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

from config import GRAIN_COLS

ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "reports"
DATA = ROOT / "data"
DB_PATH = DATA / "forecasting.db"

MERGE_KEYS = ["SRC_TADIG", "DST_TADIG", "CALL_TYPE"]

CLIENT_COUNTRIES = {
    "Country 17", "Country 43", "Country 71", "Country 77", "Country 82",
    "Country 101", "Country 104", "Country 105", "Country 110", "Country 116",
    "Country 120", "Country 149", "Country 155", "Country 175", "Country 190",
    "Country 202", "Country 209", "Country 225", "Country 231", "Country 233",
}

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

SCENARIO_COLORS = {
    "Rescue": "#EF553B", "Trend Shift": "#FFA15A", "Workhorse": "#00CC96",
    "Polisher": "#636EFA", "Stable & Good": "#AB63FA",
    "Honest Limit": "#7F7F7F", "Unclassified": "#BABBBD",
}

SCENARIO_DESCRIPTIONS = {
    "Rescue": "Static forecast badly wrong, rolling self-corrects",
    "Trend Shift": "Moderate drift caught by rolling retrain",
    "Workhorse": "Both methods excellent — stable pattern",
    "Polisher": "Good static forecast, rolling makes it great",
    "Stable & Good": "Static sufficient, no rolling needed",
    "Honest Limit": "Genuinely hard to predict for both methods",
}

STATIC_MODEL_COLORS = {
    "lgbm_trend": "#EF553B",
    "lgbm": "#FF6692",
    "sarima": "#FFA15A",
    "sarima_fb": "#FECB52",
    "ets_damped": "#AB63FA",
    "ets_undamped": "#B6E880",
    "ets_mul_trend": "#19D3F3",
    "theta": "#FF97FF",
    "seasonal_naive": "#BAB0AC",
    "trended_snaive": "#72B7B2",
}

STATIC_MODEL_STYLES = {
    "lgbm_trend": dict(dash="dash", symbol="diamond"),
    "lgbm": dict(dash="dash", symbol="square"),
    "sarima": dict(dash="dot", symbol="triangle-up"),
    "sarima_fb": dict(dash="dot", symbol="triangle-down"),
    "ets_damped": dict(dash="dashdot", symbol="cross"),
    "ets_undamped": dict(dash="dashdot", symbol="x"),
    "ets_mul_trend": dict(dash="dashdot", symbol="pentagon"),
    "theta": dict(dash="dot", symbol="hexagon"),
    "seasonal_naive": dict(dash="dot", symbol="circle"),
    "trended_snaive": dict(dash="dot", symbol="star-triangle-up"),
}


def _fmt_month(ym: int) -> str:
    s = str(ym).zfill(6)
    return f"{calendar.month_abbr[int(s[4:6])]} {s[:4]}"


# ---------------------------------------------------------------------------
# Data loading (all cached)
# ---------------------------------------------------------------------------

@st.cache_data
def _load_metadata() -> pd.DataFrame:
    conn = sqlite3.connect(str(DB_PATH))
    meta = pd.read_sql(
        "SELECT SRC_TADIG, DST_TADIG, CALL_TYPE, DST_NAME, DST_COUNTRY, GROUPNAME, NEGOTIATOR "
        "FROM traffic GROUP BY SRC_TADIG, DST_TADIG, CALL_TYPE", conn,
    )
    conn.close()
    return meta.rename(columns={
        "DST_NAME": "Operator", "DST_COUNTRY": "Country",
        "GROUPNAME": "Group", "NEGOTIATOR": "Negotiator",
    })


@st.cache_data
def _load_all():
    """Load, merge metadata, filter to client countries, classify scenarios."""
    meta = _load_metadata()
    client_tadigs = set(meta[meta["Country"].isin(CLIENT_COUNTRIES)]["DST_TADIG"].unique())

    def _client(df):
        return df["SRC_TADIG"].isin(client_tadigs) | df["DST_TADIG"].isin(client_tadigs)

    # Actuals
    p90 = pd.read_parquet(DATA / "pareto90_set.parquet")
    p90 = p90[_client(p90)].copy()
    p90 = p90.merge(meta, on=MERGE_KEYS, how="left")
    ym = p90["CALL_YEAR_MONTH"].astype(str).str.zfill(6)
    p90["date"] = pd.to_datetime(ym + "01", format="%Y%m%d")

    # Predictions (horserace + trend, dedup keeping trend)
    hr = pd.read_csv(REPORTS / "pareto90_horserace_predictions.csv")
    tr = pd.read_csv(REPORTS / "pareto90_trend_predictions.csv")
    tr = tr[_client(tr)].copy()
    if "CALL_TYPE" in hr.columns:
        hr = hr[_client(hr)].copy()
    all_preds = pd.concat([hr, tr], ignore_index=True)
    dedup = [c for c in ["SRC_TADIG", "DST_TADIG", "CALL_TYPE", "CALL_YEAR_MONTH", "target", "model"]
             if c in all_preds.columns]
    all_preds = all_preds.drop_duplicates(subset=dedup, keep="last")

    # Rolling
    rolling = pd.read_csv(REPORTS / "pareto90_rolling_accuracy.csv")
    rolling = rolling[_client(rolling)].copy()
    roll_merge = [c for c in MERGE_KEYS if c in rolling.columns]
    rolling = rolling.merge(
        meta[MERGE_KEYS + ["Operator", "Country", "Group", "Negotiator"]].drop_duplicates(),
        on=roll_merge, how="left",
    )

    # Scenario classification
    roll_ets = rolling[rolling["model"] == "ets_damped"].copy()
    roll_ets["ape"] = roll_ets["ape"].clip(upper=2.0)
    roll_agg = (
        roll_ets.groupby(MERGE_KEYS + ["target"])["ape"].mean()
        .reset_index().rename(columns={"ape": "rolling_wape"})
    )
    st_test = tr[(tr["model"] == "lgbm_trend") & (tr["CALL_YEAR_MONTH"] >= 202501)].copy()
    st_test["ape"] = (abs(st_test["predicted"] - st_test["actual"]) / st_test["actual"].clip(lower=1e-9)).clip(upper=2.0)
    stat_agg = (
        st_test.groupby(MERGE_KEYS + ["target"])["ape"].mean()
        .reset_index().rename(columns={"ape": "static_wape"})
    )
    scenarios = roll_agg.merge(stat_agg, on=MERGE_KEYS + ["target"], how="inner")
    scenarios["gap"] = scenarios["static_wape"] - scenarios["rolling_wape"]
    conds = [
        (scenarios["static_wape"] > 0.40) & (scenarios["rolling_wape"] < 0.15),
        (scenarios["static_wape"] > 0.25) & (scenarios["rolling_wape"] < 0.15),
        (scenarios["static_wape"] < 0.10) & (scenarios["rolling_wape"] < 0.10),
        (scenarios["static_wape"] < 0.15) & (scenarios["gap"] > scenarios["static_wape"] * 0.20),
        (scenarios["static_wape"] < 0.15),
        (scenarios["static_wape"] > 0.30) & (scenarios["rolling_wape"] > 0.30),
    ]
    scenarios["scenario"] = np.select(
        conds, ["Rescue", "Trend Shift", "Workhorse", "Polisher", "Stable & Good", "Honest Limit"],
        default="Unclassified",
    )
    scenarios = scenarios.merge(meta, on=MERGE_KEYS, how="left")

    return p90, all_preds, rolling, scenarios


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def _sidebar(scenarios):
    """Persistent sidebar: search + cascading filters + page selector."""
    with st.sidebar:
        st.header("Filters")
        search = st.text_input("Search", key="search", placeholder="Country, operator, or TADIG...")

        w = scenarios.copy()
        if search and search.strip():
            s = search.strip().lower()
            w = w[
                w["Country"].str.lower().str.contains(s, na=False)
                | w["Operator"].str.lower().str.contains(s, na=False)
                | w["SRC_TADIG"].str.lower().str.contains(s, na=False)
                | w["DST_TADIG"].str.lower().str.contains(s, na=False)
            ]

        for col, key in [("Negotiator", "f_neg"), ("Country", "f_co"), ("CALL_TYPE", "f_ct"), ("scenario", "f_sc"), ("target", "f_tg")]:
            label = col if col not in ("CALL_TYPE", "scenario", "target") else {"CALL_TYPE": "Call Type", "scenario": "Scenario", "target": "Target"}[col]
            opts = ["All"] + sorted(w[col].dropna().unique())
            choice = st.selectbox(label, opts, key=key)
            if choice != "All":
                w = w[w[col] == choice]

        st.divider()
        st.caption(f"{len(w):,} routes matching")
        page = st.radio("View", ["Summary", "Routes", "Detail", "Help"], key="page")
    return w, page


# ---------------------------------------------------------------------------
# View 1: Summary
# ---------------------------------------------------------------------------

def _page_summary(filt):
    st.title("Forecast Summary")
    if filt.empty:
        st.warning("No routes match the current filters.")
        return

    n = len(filt)
    acc = (1 - filt["static_wape"].mean()) * 100
    rw = int((filt["gap"] > 0).sum())
    rescue = int((filt["scenario"] == "Rescue").sum())

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Routes", f"{n:,}")
    c2.metric("Avg Static Accuracy", f"{acc:.1f}%")
    c3.metric("Rolling Wins", f"{rw} ({rw * 100 // n}%)")
    c4.metric("Rescue Routes", rescue)

    # Scenario bar
    st.subheader("Scenario Distribution")
    sc = filt["scenario"].value_counts().reset_index()
    sc.columns = ["Scenario", "Count"]
    fig = px.bar(sc, x="Scenario", y="Count", color="Scenario",
                 color_discrete_map=SCENARIO_COLORS, text="Count")
    fig.update_traces(textposition="outside")
    fig.update_layout(height=350, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    # Accuracy by country (horizontal)
    st.subheader("Accuracy by Country")
    ca = filt.groupby("Country")["static_wape"].apply(lambda x: (1 - x.mean()) * 100).round(1)
    ca = ca.reset_index().rename(columns={"static_wape": "Accuracy %"}).sort_values("Accuracy %")
    fig2 = px.bar(ca, y="Country", x="Accuracy %", orientation="h", text="Accuracy %",
                  color="Accuracy %", color_continuous_scale=["#EF553B", "#FECB52", "#00CC96"],
                  range_color=[50, 100])
    fig2.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig2.update_layout(height=max(300, len(ca) * 28), showlegend=False,
                       xaxis_range=[0, 105], xaxis_title="", yaxis_title="")
    st.plotly_chart(fig2, use_container_width=True)

    # Accuracy by negotiator
    st.subheader("Accuracy by Negotiator")
    na_ = filt.groupby("Negotiator")["static_wape"].apply(lambda x: (1 - x.mean()) * 100).round(1)
    na_ = na_.reset_index().rename(columns={"static_wape": "Accuracy %"}).sort_values("Accuracy %", ascending=False)
    fig3 = px.bar(na_, x="Negotiator", y="Accuracy %", text="Accuracy %",
                  color="Accuracy %", color_continuous_scale=["#EF553B", "#FECB52", "#00CC96"],
                  range_color=[50, 100])
    fig3.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig3.update_layout(height=350, showlegend=False, yaxis_range=[0, 105])
    st.plotly_chart(fig3, use_container_width=True)


# ---------------------------------------------------------------------------
# View 2: Route Explorer
# ---------------------------------------------------------------------------

def _page_routes(filt):
    st.title("Route Explorer")
    if filt.empty:
        st.warning("No routes match the current filters.")
        return

    disp = filt.copy()
    disp["Static %"] = (disp["static_wape"] * 100).round(1)
    disp["Rolling %"] = (disp["rolling_wape"] * 100).round(1)
    disp["Gap (pp)"] = (disp["gap"] * 100).round(1)
    disp["Target"] = disp["target"].map(TARGET_LABELS)

    cols = ["Operator", "Country", "CALL_TYPE", "Target", "Static %", "Rolling %", "Gap (pp)", "scenario"]
    st.dataframe(
        disp[cols].sort_values("Gap (pp)", ascending=False).reset_index(drop=True),
        use_container_width=True, hide_index=True, height=400,
        column_config={
            "scenario": st.column_config.TextColumn("Scenario"),
            "Static %": st.column_config.NumberColumn(format="%.1f%%"),
            "Rolling %": st.column_config.NumberColumn(format="%.1f%%"),
            "Gap (pp)": st.column_config.NumberColumn(format="%.1f"),
        },
    )

    # Route selector
    st.divider()
    labels, keys = [], []
    for _, r in disp.sort_values("gap", ascending=False).iterrows():
        labels.append(
            f"{r.get('Operator', r['DST_TADIG'])} ({r['Country']}) | "
            f"{r['CALL_TYPE']} | {TARGET_LABELS.get(r['target'], r['target'])} | "
            f"{r['scenario']}"
        )
        keys.append((r["SRC_TADIG"], r["DST_TADIG"], r["CALL_TYPE"], r["target"]))

    if not labels:
        return
    idx = st.selectbox("Select route for detail", range(len(labels)),
                       format_func=lambda i: labels[i], key="route_sel")
    st.session_state["detail_route"] = keys[idx]
    st.info("Switch to **Detail** in the sidebar to see the comparison chart.")


# ---------------------------------------------------------------------------
# View 3: Route Detail (the gold chart)
# ---------------------------------------------------------------------------

def _page_detail(p90, all_preds, rolling, scenarios, filtered):
    st.title("Route Detail")

    # Build route selector from filtered data
    if filtered.empty:
        st.warning("No routes match the current filters.")
        return

    labels, keys = [], []
    for _, r in filtered.sort_values("gap", ascending=False).iterrows():
        labels.append(
            f"{r.get('Operator', r['DST_TADIG'])} ({r['Country']}) | "
            f"{r['CALL_TYPE']} | {TARGET_LABELS.get(r['target'], r['target'])} | "
            f"{r['scenario']}"
        )
        keys.append((r["SRC_TADIG"], r["DST_TADIG"], r["CALL_TYPE"], r["target"]))

    # Pre-select the route chosen from Routes page if it exists in filtered set
    pre = st.session_state.get("detail_route")
    default_idx = 0
    if pre and pre in keys:
        default_idx = keys.index(pre)

    idx = st.selectbox("Select route", range(len(labels)),
                       format_func=lambda i: labels[i], key="detail_sel",
                       index=default_idx)
    src, dst, ct, target = keys[idx]
    tgt_label = TARGET_LABELS.get(target, target)

    # Route metadata
    rs_df = scenarios[
        (scenarios["SRC_TADIG"] == src) & (scenarios["DST_TADIG"] == dst)
        & (scenarios["CALL_TYPE"] == ct) & (scenarios["target"] == target)
    ]
    if not rs_df.empty:
        rs = rs_df.iloc[0]
        ic1, ic2, ic3, ic4 = st.columns(4)
        ic1.metric("Operator", rs.get("Operator", "N/A"))
        ic2.metric("Country", rs.get("Country", "N/A"))
        ic3.metric("Negotiator", rs.get("Negotiator", "N/A"))
        ic4.metric("Scenario", rs.get("scenario", "N/A"))
        scen = rs.get("scenario", "")
        if scen in SCENARIO_DESCRIPTIONS:
            st.info(f"**{scen}**: {SCENARIO_DESCRIPTIONS[scen]}")
    else:
        rs = None

    # Series data
    series = p90[(p90["SRC_TADIG"] == src) & (p90["DST_TADIG"] == dst) & (p90["CALL_TYPE"] == ct)]
    series = series.sort_values("CALL_YEAR_MONTH")
    if series.empty:
        st.warning("No series data found.")
        return

    def _ct(df):
        return (df["CALL_TYPE"] == ct) if "CALL_TYPE" in df.columns else True

    # Rolling
    roll = rolling[
        (rolling["SRC_TADIG"] == src) & (rolling["DST_TADIG"] == dst) & _ct(rolling)
        & (rolling["target"] == target) & (rolling["model"] == "ets_damped")
        & (rolling["forecast_month"] >= 202501)
    ].sort_values("forecast_month")

    # All static predictions for this route+target
    route_preds = all_preds[
        (all_preds["SRC_TADIG"] == src) & (all_preds["DST_TADIG"] == dst) & _ct(all_preds)
        & (all_preds["target"] == target)
    ]
    available_models = sorted(route_preds["model"].unique())

    # Model selector
    default_models = [m for m in ["lgbm_trend", "seasonal_naive"] if m in available_models]
    if not default_models:
        default_models = available_models[:2]
    selected_models = st.multiselect(
        "Static models", available_models, default=default_models, key="detail_models",
    )

    # --- Chart ---
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=series["date"], y=series[target], mode="lines+markers", name="Actual",
        line=dict(color="#636EFA", width=2.5), marker=dict(size=6),
    ))
    for model in selected_models:
        m_df = route_preds[route_preds["model"] == model].sort_values("CALL_YEAR_MONTH")
        if m_df.empty:
            continue
        d = pd.to_datetime(m_df["CALL_YEAR_MONTH"].astype(str).str.zfill(6) + "01", format="%Y%m%d")
        style = STATIC_MODEL_STYLES.get(model, dict(dash="solid", symbol="circle"))
        fig.add_trace(go.Scatter(
            x=d, y=m_df["predicted"].values, mode="lines+markers", name=f"Static: {model}",
            line=dict(color=STATIC_MODEL_COLORS.get(model, "#999"), width=2, dash=style["dash"]),
            marker=dict(size=5, symbol=style["symbol"]),
        ))
    if not roll.empty:
        d = pd.to_datetime(roll["forecast_month"].astype(str).str.zfill(6) + "01", format="%Y%m%d")
        fig.add_trace(go.Scatter(
            x=d, y=roll["predicted"].values, mode="lines+markers", name="Rolling: ets_damped",
            line=dict(color="#00CC96", width=2.5), marker=dict(size=7, symbol="star"),
        ))

    fig.add_vline(x="2025-01-01", line_dash="dash", line_color="gray", opacity=0.5)
    fig.add_annotation(x="2024-12-15", y=1, yref="paper", text="Train", showarrow=False,
                       font=dict(color="gray"), yshift=10)
    fig.add_annotation(x="2025-01-15", y=1, yref="paper", text="Test", showarrow=False,
                       font=dict(color="gray"), yshift=10)
    title = (f"{rs.get('Operator', dst)} ({rs.get('Country', '')}) | {ct} | {tgt_label}"
             if rs is not None else f"{src} > {dst} | {ct} | {tgt_label}")
    fig.update_layout(title=title, xaxis_title="Month", yaxis_title=tgt_label, height=550,
                      legend=dict(orientation="h", yanchor="bottom", y=1.02), hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

    # --- Month-by-month table ---
    if roll.empty or not selected_models:
        return
    st.subheader("Month-by-Month Comparison")

    # Build per-model prediction lookup
    model_preds = {}
    for model in selected_models:
        m_df = route_preds[route_preds["model"] == model].set_index("CALL_YEAR_MONTH")["predicted"]
        model_preds[model] = m_df

    rows = []
    for _, rr in roll.iterrows():
        fm = int(rr["forecast_month"])
        actual = rr["actual"]
        r_pred = rr["predicted"]
        r_err = abs(actual - r_pred) / abs(actual) * 100 if actual != 0 else np.nan
        row = {
            "Month": _fmt_month(fm),
            "Actual": f"{actual:,.0f}",
            "Rolling": f"{r_pred:,.0f}",
            "Rolling APE%": f"{r_err:.1f}%",
        }
        for model in selected_models:
            s_pred = model_preds[model].get(fm, np.nan)
            s_err = abs(actual - s_pred) / abs(actual) * 100 if (pd.notna(s_pred) and actual != 0) else np.nan
            row[f"{model}"] = f"{s_pred:,.0f}" if pd.notna(s_pred) else "N/A"
            row[f"{model} APE%"] = f"{s_err:.1f}%" if pd.notna(s_err) else "N/A"
        rows.append(row)
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # Summary metrics
    r_apes = [float(r["Rolling APE%"].rstrip("%")) for r in rows]
    cols = st.columns(1 + len(selected_models))
    cols[0].metric("Avg Rolling APE", f"{np.mean(r_apes):.1f}%")
    for i, model in enumerate(selected_models):
        s_apes = [float(r[f"{model} APE%"].rstrip("%")) for r in rows if r[f"{model} APE%"] != "N/A"]
        if s_apes:
            avg = np.mean(s_apes)
            diff = avg - np.mean(r_apes)
            cols[i + 1].metric(f"Avg {model} APE", f"{avg:.1f}%", delta=f"{diff:+.1f}pp vs rolling", delta_color="inverse")


# ---------------------------------------------------------------------------
# View 4: Help
# ---------------------------------------------------------------------------

def _page_help():
    st.title("How to Use This Dashboard")

    st.markdown("""
This dashboard compares two forecasting approaches for roaming traffic across
your 20 client countries:

- **Static forecast** — trained once on historical data (up to Dec 2024),
  then predicts all future months without updating.
- **Rolling forecast** — retrained every month with the latest actuals,
  producing a fresh 1-month-ahead prediction each time.

The key question: *does retraining each month actually improve accuracy?*
""")

    st.subheader("Views")
    st.markdown("""
| View | What it shows |
|------|---------------|
| **Summary** | High-level metrics and charts — overall accuracy, scenario breakdown, accuracy by country and negotiator. |
| **Routes** | Searchable table of all routes with static vs rolling error and scenario tag. Pick a route to inspect. |
| **Detail** | The comparison chart for a selected route — actual traffic vs static and rolling predictions, month by month. |
""")

    st.subheader("Filters")
    st.markdown("""
All filters are in the sidebar and apply across every view:

- **Search** — type any country name, operator name, or TADIG code to narrow results instantly.
- **Negotiator / Country / Call Type** — cascading dropdowns. Picking a negotiator limits the countries shown, picking a country limits call types, etc.
- **Scenario** — filter to a specific scenario category (see below).
- **Target** — the traffic metric being forecast (e.g. Inbound Calls, Outbound Data).
""")

    st.subheader("Scenarios")
    st.markdown("Each route is classified into a scenario based on how static and rolling forecasts perform:\n")

    scenario_table = []
    for name, desc in SCENARIO_DESCRIPTIONS.items():
        color = SCENARIO_COLORS.get(name, "#999")
        scenario_table.append(f"| **{name}** | {desc} |")

    st.markdown(
        "| Scenario | Meaning |\n|----------|--------|\n"
        + "\n".join(scenario_table)
    )

    st.markdown("""
**Scenario definitions (technical):**

| Scenario | Static WAPE | Rolling WAPE | Condition |
|----------|------------|-------------|-----------|
| Rescue | > 40% | < 15% | Static badly wrong, rolling saves it |
| Trend Shift | > 25% | < 15% | Moderate drift caught by rolling |
| Workhorse | < 10% | < 10% | Stable pattern, both excellent |
| Polisher | < 15% | 20%+ better relatively | Already good, rolling improves |
| Stable & Good | < 15% | — | Static is sufficient |
| Honest Limit | > 30% | > 30% | Hard to predict for both |
""")

    st.subheader("Metrics")
    st.markdown("""
- **WAPE** (Weighted Absolute Percentage Error) — how far off the prediction is
  as a percentage of actual traffic. Lower is better.
- **Accuracy** — shown as `1 - WAPE`. Higher is better. Displayed as a percentage.
- **Gap** — difference between static and rolling WAPE. Positive means rolling is
  more accurate.
- **APE** — Absolute Percentage Error per month, shown in the detail comparison table.
- **Rolling Advantage** — how many percentage points rolling beats static on average.
""")

    st.subheader("The Comparison Chart")
    st.markdown("""
The detail chart shows four lines:

1. **Actual** (blue solid) — real traffic volumes
2. **Static: lgbm_trend** (red dashed) — gradient-boosted tree forecast, trained once
3. **Static: seasonal_naive** (light green dotted) — simple last-year baseline
4. **Rolling: ets_damped** (green solid, star markers) — exponential smoothing retrained monthly

The vertical dashed line marks **Jan 2025** — everything to the right is the test period
where forecasts are evaluated.
""")

    st.subheader("Data Coverage")
    st.markdown(f"""
- Routes are filtered to your **20 client countries** ({len(CLIENT_COUNTRIES)} countries, covering the Pareto-90% traffic set).
- 4 countries (Country 17, 101, 110, 120) fall below the 90% traffic threshold and have no forecast data.
- The remaining 16 countries have full model coverage across all test months (Jan–Nov 2025).
""")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Client Forecast Dashboard", layout="wide")

p90, all_preds, rolling, scenarios = _load_all()
filtered, page = _sidebar(scenarios)

if page == "Summary":
    _page_summary(filtered)
elif page == "Routes":
    _page_routes(filtered)
elif page == "Detail":
    _page_detail(p90, all_preds, rolling, scenarios, filtered)
elif page == "Help":
    _page_help()
