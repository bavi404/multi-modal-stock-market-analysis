"""
Offline evaluation: regression metrics, orchestrator timing, JSON reports, optional plots.

Used by :mod:`run_evaluation` for reproducible benchmarks.
"""

from .metrics import directional_accuracy, mape, prediction_metrics
from .prediction_backtest import run_linear_backtest
from .pipeline_timing import measure_pipeline_stages
from .reporting import build_evaluation_report, write_json_report, write_optional_plots

__all__ = [
    "directional_accuracy",
    "mape",
    "prediction_metrics",
    "run_linear_backtest",
    "measure_pipeline_stages",
    "build_evaluation_report",
    "write_json_report",
    "write_optional_plots",
]
