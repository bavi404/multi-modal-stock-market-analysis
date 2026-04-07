"""
JSON report assembly and optional matplotlib figures.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


def build_evaluation_report(
    *,
    ticker: str,
    prediction_backtest: Dict[str, Any],
    pipeline_timing: Dict[str, Any],
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Merge prediction backtest output and pipeline timing into one versioned document.

    Returns
    -------
    dict
        JSON-serializable report with ``schema_version`` and UTC ``generated_at``.
    """
    return {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "ticker": ticker,
        "prediction_evaluation": prediction_backtest,
        "system_evaluation": pipeline_timing,
        "metadata": metadata or {},
    }


def write_json_report(report: Dict[str, Any], path: Path) -> None:
    """Write ``report`` to ``path`` with UTF-8 encoding and stable key ordering (``indent=2``)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)


def write_optional_plots(
    report: Dict[str, Any],
    out_dir: Path,
    *,
    prediction_backtest_detail: Optional[Dict[str, Any]] = None,
) -> List[str]:
    """
    Write optional PNG figures next to the JSON report.

    Produces a horizontal bar chart of pipeline stage durations. If ``prediction_backtest_detail``
    includes ``y_test`` and ``y_pred``, also writes a scatter of actual vs predicted closes.

    Returns
    -------
    list[str]
        Paths of files written (may be empty if matplotlib is unavailable or no data).
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    written: List[str] = []
    out_dir.mkdir(parents=True, exist_ok=True)

    stages = report.get("system_evaluation", {}).get("stages") or []
    if stages:
        names = [s["name"] for s in stages]
        secs = [s["duration_seconds"] for s in stages]
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.barh(names, secs, color="#2c5282")
        ax.set_xlabel("Seconds")
        ax.set_title("Pipeline stage latency")
        fig.tight_layout()
        p = out_dir / "stage_latency.png"
        fig.savefig(p, dpi=150)
        plt.close(fig)
        written.append(str(p))

    if prediction_backtest_detail and "y_test" in prediction_backtest_detail and "y_pred" in prediction_backtest_detail:
        yt = np.asarray(prediction_backtest_detail["y_test"])
        yp = np.asarray(prediction_backtest_detail["y_pred"])
        fig, ax = plt.subplots(figsize=(6, 6))
        lim = min(yt.min(), yp.min()), max(yt.max(), yp.max())
        ax.scatter(yt, yp, alpha=0.5, s=12)
        ax.plot(lim, lim, "r--", lw=1, label="ideal")
        ax.set_xlabel("Actual close")
        ax.set_ylabel("Predicted close")
        ax.set_title("Test set: actual vs predicted")
        ax.legend()
        ax.set_aspect("equal", adjustable="box")
        fig.tight_layout()
        p = out_dir / "pred_vs_actual.png"
        fig.savefig(p, dpi=150)
        plt.close(fig)
        written.append(str(p))

    return written
