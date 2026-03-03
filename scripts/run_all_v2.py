#!/usr/bin/env python3
"""Run full v2 pipeline: horserace + rolling retrain for both directions.

Usage:
  tmux new -s pipeline
  python3 scripts/run_all_v2.py
  # Ctrl-B D to detach

Expected total runtime: ~10-14 hours on this machine.
  - Outbound horserace:  ~1.5h   (1,876 series × 5 models)
  - Inbound horserace:   ~4-5h   (7,737 series × 5 models)
  - Outbound rolling:    ~1-2h   (1,876 series × 4 per-series models + lgbm checkpoints)
  - Inbound rolling:     ~3-5h   (7,737 series × 4 per-series models + lgbm checkpoints)
"""
from __future__ import annotations

import subprocess
import sys
import time
from datetime import datetime


def run_step(label: str, cmd: list[str]) -> bool:
    """Run a subprocess, stream output, return True on success."""
    print(f"\n{'='*70}", flush=True)
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] STARTING: {label}", flush=True)
    print(f"  Command: {' '.join(cmd)}", flush=True)
    print(f"{'='*70}\n", flush=True)

    t0 = time.time()
    result = subprocess.run(cmd, cwd="/home/dev/roaming")
    elapsed = time.time() - t0

    status = "SUCCESS" if result.returncode == 0 else f"FAILED (exit {result.returncode})"
    print(f"\n[{datetime.now():%Y-%m-%d %H:%M:%S}] {status}: {label} ({elapsed/60:.1f} min)\n", flush=True)

    return result.returncode == 0


def main():
    print(f"Pipeline v2 — Full Run", flush=True)
    print(f"Started: {datetime.now():%Y-%m-%d %H:%M:%S}", flush=True)
    t_start = time.time()

    steps = [
        ("Outbound Horserace",       [sys.executable, "scripts/run_horserace.py", "--direction", "outbound"]),
        ("Inbound Horserace",        [sys.executable, "scripts/run_horserace.py", "--direction", "inbound"]),
        ("Outbound Rolling Retrain", [sys.executable, "scripts/run_rolling_retrain.py", "--direction", "outbound"]),
        ("Inbound Rolling Retrain",  [sys.executable, "scripts/run_rolling_retrain.py", "--direction", "inbound"]),
    ]

    results = []
    for label, cmd in steps:
        ok = run_step(label, cmd)
        results.append((label, ok))
        if not ok:
            print(f"\n*** {label} failed — continuing with remaining steps ***\n", flush=True)

    # Summary
    total = time.time() - t_start
    print(f"\n{'='*70}", flush=True)
    print(f"PIPELINE COMPLETE — {datetime.now():%Y-%m-%d %H:%M:%S} ({total/3600:.1f} hours)", flush=True)
    print(f"{'='*70}", flush=True)
    for label, ok in results:
        print(f"  {'✓' if ok else '✗'} {label}", flush=True)
    print(flush=True)

    # Check expected outputs
    from pathlib import Path
    reports = Path("/home/dev/roaming/reports")
    expected = [
        "outbound_horserace_predictions.csv",
        "inbound_horserace_predictions.csv",
        "outbound_rolling_accuracy.csv",
        "inbound_rolling_accuracy.csv",
    ]
    print("Output files:", flush=True)
    for f in expected:
        p = reports / f
        if p.exists():
            size = p.stat().st_size / 1024 / 1024
            print(f"  ✓ {f} ({size:.1f} MB)", flush=True)
        else:
            print(f"  ✗ {f} (MISSING)", flush=True)

    print(f"\nTo launch dashboard:", flush=True)
    print(f"  streamlit run scripts/dashboard_v2.py --server.port 8505 --server.headless true", flush=True)


if __name__ == "__main__":
    main()
