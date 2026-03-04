"""Shared constants for the roaming forecasting pipeline.

Central configuration for grain definitions, target mappings, portfolio scope,
and train/test split boundaries. Imported by every other script in the pipeline.

Architecture (v2 — dual-grain):
  - Inbound:  SRC_TADIG × DST_TADIG × CALL_TYPE  (operator-to-operator routes)
  - Outbound: SRC_TADIG × DST_COUNTRY × CALL_TYPE (country-level aggregation)

Each call type maps to exactly one forecast target per direction (e.g. GPRS → VOL_MB).
The 20 portfolio countries define the scope of all forecasting and evaluation.
"""
from __future__ import annotations

# --------------- Grain definitions ---------------

INBOUND_GRAIN = ["SRC_TADIG", "DST_TADIG", "CALL_TYPE"]
OUTBOUND_GRAIN = ["SRC_TADIG", "DST_COUNTRY", "CALL_TYPE"]

CALL_TYPES = ["GPRS", "MOC", "MTC", "SMS-MT"]  # SMS-MO dropped in v2
DASHBOARD_CALL_TYPES = ["GPRS", "MOC"]  # subset shown on dashboard & used for production forecasts

# --------------- Target mappings (1 target per call_type per direction) ---------------

INBOUND_TARGET: dict[str, str] = {
    "GPRS": "INBOUND_VOL_MB",
    "MOC": "INBOUND_DURATION",
    "MTC": "INBOUND_DURATION",
    "SMS-MT": "INBOUND_CALLS",
}

OUTBOUND_TARGET: dict[str, str] = {
    "GPRS": "OUTBOUND_VOL_MB",
    "MOC": "OUTBOUND_DURATION",
    "MTC": "OUTBOUND_DURATION",
    "SMS-MT": "OUTBOUND_CALLS",
}

# --------------- Portfolio ---------------

PORTFOLIO_COUNTRIES = [
    "Country 17", "Country 43", "Country 71", "Country 77", "Country 82",
    "Country 101", "Country 104", "Country 105", "Country 110", "Country 116",
    "Country 120", "Country 149", "Country 155", "Country 175", "Country 190",
    "Country 202", "Country 209", "Country 225", "Country 231", "Country 233",
]

TRAIN_END = 202412
TEST_START = 202501
TEST_END = 202511

# --------------- Helper functions ---------------


def grain_cols_for(direction: str) -> list[str]:
    """Return grain columns for the given direction ('inbound' or 'outbound')."""
    if direction == "inbound":
        return INBOUND_GRAIN
    elif direction == "outbound":
        return OUTBOUND_GRAIN
    raise ValueError(f"Unknown direction: {direction!r}. Use 'inbound' or 'outbound'.")


def target_for(call_type: str, direction: str) -> str:
    """Return the single target column for a call_type + direction."""
    mapping = INBOUND_TARGET if direction == "inbound" else OUTBOUND_TARGET
    return mapping[call_type]
