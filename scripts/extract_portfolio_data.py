#!/usr/bin/env python3
"""Extract inbound + outbound datasets for the 20 portfolio countries.

Pipeline step 2 (after load_forecasting_data.py populates the SQLite DB).
Reads from the 'traffic' table (inbound, TADIG-level) and 'traffic_model_grain'
view (outbound, country-level aggregation) in data/forecasting.db.

Produces:
  data/inbound_set.parquet   — TADIG-to-TADIG grain (SRC_TADIG + DST_TADIG + CALL_TYPE)
  data/outbound_set.parquet  — country grain (SRC_TADIG + DST_COUNTRY + CALL_TYPE)
  data/inbound_series.csv    — route listing with DST_COUNTRY for reference
  data/outbound_series.csv   — route listing

Usage:
  python scripts/extract_portfolio_data.py
"""
from __future__ import annotations

import sqlite3
import time
from pathlib import Path

import pandas as pd

from config import (
    CALL_TYPES,
    INBOUND_GRAIN,
    OUTBOUND_GRAIN,
    PORTFOLIO_COUNTRIES,
)

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "data" / "forecasting.db"
DATA = ROOT / "data"


def extract_inbound(conn: sqlite3.Connection) -> pd.DataFrame:
    """Extract inbound dataset at TADIG-to-TADIG grain."""
    placeholders = ",".join("?" for _ in PORTFOLIO_COUNTRIES)
    ct_ph = ",".join("?" for _ in CALL_TYPES)

    query = f"""
        SELECT *
        FROM traffic
        WHERE DST_COUNTRY IN ({placeholders})
          AND CALL_TYPE IN ({ct_ph})
    """
    df = pd.read_sql(query, conn, params=PORTFOLIO_COUNTRIES + CALL_TYPES)

    # Derive YEAR / MONTH from CALL_YEAR_MONTH
    ym = df["CALL_YEAR_MONTH"].astype(str).str.zfill(6)
    df["YEAR"] = ym.str[:4].astype(int)
    df["MONTH"] = ym.str[4:6].astype(int)

    routes = df.groupby(INBOUND_GRAIN).size().reset_index(name="n_months")
    print(f"Inbound: {len(df):,} rows, {len(routes):,} routes, "
          f"{df['DST_COUNTRY'].nunique()} countries, {df['CALL_TYPE'].nunique()} call types")

    return df


def extract_outbound(conn: sqlite3.Connection) -> pd.DataFrame:
    """Extract outbound dataset at country grain from traffic_model_grain."""
    placeholders = ",".join("?" for _ in PORTFOLIO_COUNTRIES)
    ct_ph = ",".join("?" for _ in CALL_TYPES)

    query = f"""
        SELECT *
        FROM traffic_model_grain
        WHERE DST_COUNTRY IN ({placeholders})
          AND CALL_TYPE IN ({ct_ph})
    """
    df = pd.read_sql(query, conn, params=PORTFOLIO_COUNTRIES + CALL_TYPES)

    # Derive YEAR / MONTH
    ym = df["CALL_YEAR_MONTH"].astype(str).str.zfill(6)
    df["YEAR"] = ym.str[:4].astype(int)
    df["MONTH"] = ym.str[4:6].astype(int)

    routes = df.groupby(OUTBOUND_GRAIN).size().reset_index(name="n_months")
    print(f"Outbound: {len(df):,} rows, {len(routes):,} routes, "
          f"{df['DST_COUNTRY'].nunique()} countries, {df['CALL_TYPE'].nunique()} call types")

    return df


def main():
    t0 = time.time()
    conn = sqlite3.connect(str(DB_PATH))

    # --- Inbound ---
    inbound = extract_inbound(conn)
    inbound.to_parquet(DATA / "inbound_set.parquet", index=False)
    # Series listing with country for reference
    inbound_series = (
        inbound.groupby(INBOUND_GRAIN + ["DST_COUNTRY", "DST_NAME"])
        .agg(n_months=("CALL_YEAR_MONTH", "nunique"),
             first_month=("CALL_YEAR_MONTH", "min"),
             last_month=("CALL_YEAR_MONTH", "max"))
        .reset_index()
    )
    inbound_series.to_csv(DATA / "inbound_series.csv", index=False)
    print(f"  Saved {DATA / 'inbound_set.parquet'} and inbound_series.csv")

    # --- Outbound ---
    outbound = extract_outbound(conn)
    outbound.to_parquet(DATA / "outbound_set.parquet", index=False)
    outbound_series = (
        outbound.groupby(OUTBOUND_GRAIN)
        .agg(n_months=("CALL_YEAR_MONTH", "nunique"),
             first_month=("CALL_YEAR_MONTH", "min"),
             last_month=("CALL_YEAR_MONTH", "max"))
        .reset_index()
    )
    outbound_series.to_csv(DATA / "outbound_series.csv", index=False)
    print(f"  Saved {DATA / 'outbound_set.parquet'} and outbound_series.csv")

    conn.close()
    print(f"\nDone in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
