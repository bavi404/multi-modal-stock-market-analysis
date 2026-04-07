"""
Time-series backtest for the linear price model (matches PricePredictionAgent features).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

from utils import config
from agents.price_prediction_agent import PricePredictionAgent
from evaluation.metrics import prediction_metrics

logger = logging.getLogger(__name__)


def _prev_closes_for_samples(price_df: pd.DataFrame, n_samples: int) -> np.ndarray:
    """
    Build previous-close series aligned with supervised samples.

    ``prev_close[i]`` is the close at the end of window ``i`` (day before the predicted target).
    """
    pdays = config.PREDICTION_DAYS
    return np.array([float(price_df["Close"].iloc[pdays + k]) for k in range(n_samples)], dtype=float)


def run_linear_backtest(
    price_df: pd.DataFrame,
    test_fraction: float = 0.2,
    sentiment_scores: Optional[Sequence[float]] = None,
) -> Dict[str, Any]:
    """
    Chronological train/test split and linear regression metrics mirroring production features.

    The hold-out set is the trailing ``test_fraction`` of samples. Features match
    ``PricePredictionAgent._create_training_data`` with LSTM disabled.

    Parameters
    ----------
    price_df
        OHLCV history; must include ``Close``.
    test_fraction
        Fraction of chronological samples reserved for testing.
    sentiment_scores
        Optional per-row sentiment override (rarely used; defaults to neutral in the agent).

    Returns
    -------
    dict
        Metrics, sample counts, optional ``y_test`` / ``y_pred`` lists, or an ``error`` key.
    """
    if price_df is None or price_df.empty or "Close" not in price_df.columns:
        return {"error": "empty_or_invalid_price_data", "metrics": {}}

    agent = PricePredictionAgent()
    # Evaluation uses linear regression path for reproducible metrics
    agent.use_lstm = False
    agent.is_trained = False

    X, y = agent._create_training_data(price_df, sentiment_scores)
    if X.size == 0 or y.size == 0:
        return {"error": "insufficient_history_for_backtest", "metrics": {}}

    n = len(X)
    prev_all = _prev_closes_for_samples(price_df, n)

    split = max(1, int(n * (1.0 - test_fraction)))
    if split >= n:
        split = n - 1
    if split < 10:
        split = max(1, n // 2)

    if n - split < 2:
        return {"error": "test_set_too_small_increase_history_or_lower_test_fraction", "metrics": {}}

    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    prev_test = prev_all[split:]

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    model = LinearRegression()
    model.fit(X_train_s, y_train)
    y_pred = model.predict(X_test_s)

    metrics = prediction_metrics(y_test, y_pred, prev_close=prev_test)

    out: Dict[str, Any] = {
        "train_samples": int(split),
        "test_samples": int(len(y_test)),
        "test_fraction_config": float(test_fraction),
        "feature_dim": int(X.shape[1]) if X.ndim == 2 else 0,
        "metrics": metrics,
        "y_test": [float(x) for x in y_test],
        "y_pred": [float(x) for x in y_pred],
    }
    return out
