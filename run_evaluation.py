#!/usr/bin/env python3
"""
Run reproducible evaluation: price-model backtest metrics + full pipeline latency.

Usage
-----
::

  python run_evaluation.py --ticker AAPL
  python run_evaluation.py --ticker MSFT --output-dir ./eval_runs/msft --plots

"""
from __future__ import annotations

import argparse
import asyncio
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yfinance as yf

from evaluation.pipeline_timing import measure_pipeline_stages
from evaluation.prediction_backtest import run_linear_backtest
from evaluation.reporting import build_evaluation_report, write_json_report, write_optional_plots
from utils.logging import setup_logging


def _fetch_prices(ticker: str, period: str) -> pd.DataFrame:
    """Download OHLCV history for ``ticker`` from Yahoo Finance."""
    return yf.Ticker(ticker).history(period=period)


async def _run_async(args: argparse.Namespace) -> int:
    """Execute one evaluation run and write JSON (and optional plots) under ``output_dir``."""
    setup_logging(verbose=args.verbose)
    ticker = args.ticker.strip().upper()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    df = _fetch_prices(ticker, args.period)
    pred_eval = run_linear_backtest(df, test_fraction=args.test_fraction)
    plot_detail = None
    if args.plots and "y_test" in pred_eval and "y_pred" in pred_eval:
        plot_detail = {"y_test": pred_eval["y_test"], "y_pred": pred_eval["y_pred"]}
    if args.no_raw_series:
        pred_eval = {k: v for k, v in pred_eval.items() if k not in ("y_test", "y_pred")}

    sys_eval = await measure_pipeline_stages(ticker)

    meta = {
        "price_history_period": args.period,
        "yfinance_rows": int(len(df)),
        "python": sys.version.split()[0],
    }

    report = build_evaluation_report(
        ticker=ticker,
        prediction_backtest=pred_eval,
        pipeline_timing=sys_eval,
        metadata=meta,
    )

    json_path = out_dir / "evaluation_report.json"
    write_json_report(report, json_path)
    print(f"Wrote {json_path}")

    if args.plots:
        written = write_optional_plots(report, out_dir, prediction_backtest_detail=plot_detail)
        for w in written:
            print(f"Wrote plot {w}")

    # Compact summary to stdout
    m = pred_eval.get("metrics") or {}
    print("\n--- Prediction (hold-out) ---")
    if m:
        if isinstance(m.get("mae"), (int, float)):
            print(f"  MAE:  {m['mae']:.6f}")
        if isinstance(m.get("rmse"), (int, float)):
            print(f"  RMSE: {m['rmse']:.6f}")
        if isinstance(m.get("r2"), (int, float)):
            print(f"  R2:   {m['r2']:.6f}")
        if isinstance(m.get("mape"), (int, float)):
            print(f"  MAPE: {m['mape']:.4f} %")
        if "directional_accuracy" in m:
            print(f"  Directional accuracy: {m['directional_accuracy']:.4f}")
    else:
        print("  ", pred_eval.get("error", "no metrics"))

    print("\n--- Pipeline timing ---")
    se = sys_eval.get("total_pipeline_seconds")
    print(f"  Total: {se} s" if se is not None else "  n/a")
    for s in sys_eval.get("stages") or []:
        print(f"  {s['name']}: {s['duration_seconds']:.3f}s  ok={s['succeeded']}")

    return 0


def main() -> None:
    """CLI entry: parse args, stamp output directory, and exit with asyncio status code."""
    p = argparse.ArgumentParser(description="Multi-modal stock analysis evaluation runner")
    p.add_argument("--ticker", default="AAPL", help="Stock ticker")
    p.add_argument("--period", default="1y", help="yfinance history period for backtest")
    p.add_argument("--test-fraction", type=float, default=0.2, help="Hold-out fraction (chronological tail)")
    p.add_argument("--output-dir", default="./evaluation_runs", help="Directory for JSON and plots")
    p.add_argument("--plots", action="store_true", help="Write PNG plots (stage latency, pred vs actual)")
    p.add_argument(
        "--no-raw-series",
        action="store_true",
        help="Omit y_test/y_pred arrays from JSON (smaller file)",
    )
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    # Unique subfolder per run unless user points to explicit file parent
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    args.output_dir = str(Path(args.output_dir) / f"{args.ticker.upper()}_{stamp}")

    code = asyncio.run(_run_async(args))
    sys.exit(code)


if __name__ == "__main__":
    main()
