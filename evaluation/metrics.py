"""
Regression and direction metrics for hold-out price prediction evaluation.

Functions operate on NumPy arrays or sequences coerced to float64.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def mape(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-9) -> float:
    """Mean absolute percentage error (%)."""
    y_t = np.asarray(y_true, dtype=float)
    y_p = np.asarray(y_pred, dtype=float)
    denom = np.maximum(np.abs(y_t), epsilon)
    return float(np.mean(np.abs((y_t - y_p) / denom)) * 100.0)


def directional_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    prev_close: np.ndarray,
) -> float:
    """
    Fraction of samples where predicted move vs previous close matches actual move vs previous close.
    """
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    prev = np.asarray(prev_close, dtype=float)
    actual_sign = np.sign(yt - prev)
    pred_sign = np.sign(yp - prev)
    # Treat exact zero change as positive agreement if both zero
    return float(np.mean(actual_sign == pred_sign))


def prediction_metrics(
    y_true: Sequence[float],
    y_pred: Sequence[float],
    prev_close: Optional[Sequence[float]] = None,
) -> Dict[str, Any]:
    """MAE, RMSE, R², MAPE, and optional directional accuracy."""
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    out: Dict[str, Any] = {
        "mae": float(mean_absolute_error(yt, yp)),
        "rmse": float(np.sqrt(mean_squared_error(yt, yp))),
        "r2": float(r2_score(yt, yp)),
        "mape": mape(yt, yp),
        "n_samples": int(len(yt)),
    }
    if prev_close is not None and len(prev_close) == len(yt):
        out["directional_accuracy"] = directional_accuracy(yt, yp, np.asarray(prev_close, dtype=float))
    return out
