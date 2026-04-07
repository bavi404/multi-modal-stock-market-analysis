"""
Approximate prediction explainability for the linear price model (coefficient × scaled feature).
LSTM path uses a lightweight heuristic (sentiment, volatility proxy, news).
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

import numpy as np

from models.data_models import (
    DriverImpact,
    FeatureImportanceEntry,
    PredictionDriver,
    PredictionExplainability,
)


FEATURE_LABELS: Dict[str, str] = {
    "sentiment_score": "Aggregate text sentiment (FinBERT-derived)",
    "sma_5": "5-day simple moving average",
    "price_change_pct": "1-day price momentum (%)",
    "volume_ratio": "Volume vs 5-day average",
    "volatility": "Short-term price volatility (5-day std)",
}

for i in range(1, 30):
    FEATURE_LABELS[f"close_price_t-{i}"] = f"Historical close (lag {i})"


def _label_for_feature(name: str) -> str:
    if name in FEATURE_LABELS:
        return FEATURE_LABELS[name]
    m = re.match(r"close_price_t-(\d+)", name)
    if m:
        return f"Historical closing price (lag {m.group(1)} days)"
    return name.replace("_", " ").title()


def _impact_from_share(share: float) -> DriverImpact:
    if share >= 0.22:
        return "high"
    if share >= 0.10:
        return "medium"
    return "low"


def sentiment_contribution_text(sentiment_score: float, sentiment_feature_share: float) -> str:
    """Map sentiment score + relative importance to explanation text."""
    mag = abs(sentiment_score)
    direction = "bullish" if sentiment_score > 0.05 else "bearish" if sentiment_score < -0.05 else "neutral"
    strength = "strong" if mag > 0.35 else "moderate" if mag > 0.15 else "weak"
    rel = (
        "a primary driver of the fitted level"
        if sentiment_feature_share >= 0.18
        else "a modest input relative to price history"
        if sentiment_feature_share >= 0.08
        else "a small input relative to price history"
    )
    return (
        f"Sentiment is {strength} and {direction} (score {sentiment_score:.2f}). "
        f"As a model input, sentiment acts as {rel} in this linear blend."
    )


def emotion_context_text(dominant_emotion: str, emotion_scores: Optional[Dict[str, float]]) -> str:
    """Map market emotion aggregate to narrative (not a direct model feature for linear path)."""
    dom = (dominant_emotion or "neutral").lower()
    if emotion_scores:
        top = sorted(emotion_scores.items(), key=lambda x: x[1], reverse=True)[:3]
        detail = ", ".join(f"{k} ({v:.2f})" for k, v in top)
        return (
            f"Text-derived emotion lean is '{dom}'. Top emotion scores: {detail}. "
            "These inform the sentiment pipeline that feeds the price model; they are not separate features in the linear model."
        )
    return (
        f"Text-derived emotion lean is '{dom}'. "
        "Emotion aggregates inform sentiment; the price model uses the aggregated sentiment score as one feature."
    )


def recent_events_from_news(news_articles: List[Dict[str, str]], max_items: int = 5) -> List[str]:
    out: List[str] = []
    for a in news_articles[: max_items * 2]:
        t = (a.get("title") or "").strip()
        if not t:
            t = (a.get("description") or "").strip()[:160]
        if t and t not in out:
            out.append(t[:300])
        if len(out) >= max_items:
            break
    return out


def build_linear_explainability(
    predicted_price: float,
    model_confidence: float,
    feature_names: List[str],
    coefficients: np.ndarray,
    scaled_features: np.ndarray,
    sentiment_score: float,
    news_articles: List[Dict[str, str]],
    emotion_dominant: str,
    emotion_scores: Optional[Dict[str, float]] = None,
) -> PredictionExplainability:
    """
    |coef_i * x_scaled_i| as contribution magnitude; normalize to importances and drivers.
    """
    coef = np.asarray(coefficients).ravel()
    x = np.asarray(scaled_features).ravel()
    n = min(len(coef), len(x), len(feature_names))
    if n == 0:
        return PredictionExplainability(
            prediction=predicted_price,
            confidence=model_confidence,
            drivers=[
                PredictionDriver(factor="Insufficient feature data for attribution", impact="low")
            ],
            sentiment_contribution=sentiment_contribution_text(sentiment_score, 0.0),
            emotion_context=emotion_context_text(emotion_dominant, emotion_scores),
            recent_events=recent_events_from_news(news_articles),
            attribution_method="linear_coefficient_product",
        )

    contrib = np.abs(coef[:n] * x[:n])
    names = feature_names[:n]
    total = float(np.sum(contrib)) + 1e-12
    importance_raw = contrib / total

    entries: List[FeatureImportanceEntry] = []
    order = np.argsort(-importance_raw)
    for idx in order:
        fi = float(importance_raw[idx])
        name = names[idx]
        entries.append(
            FeatureImportanceEntry(feature=name, label=_label_for_feature(name), importance=round(fi, 4))
        )

    sentiment_idx = next((i for i, nm in enumerate(names) if nm == "sentiment_score"), None)
    if sentiment_idx is not None:
        sentiment_share = float(importance_raw[sentiment_idx])
    else:
        sentiment_share = 0.0

    drivers: List[PredictionDriver] = []
    for idx in order[:5]:
        share = float(importance_raw[idx])
        if share < 0.02:
            continue
        label = _label_for_feature(names[idx])
        drivers.append(PredictionDriver(factor=label, impact=_impact_from_share(share)))

    for ev in recent_events_from_news(news_articles, max_items=2):
        short = ev if len(ev) < 90 else ev[:87] + "…"
        drivers.append(PredictionDriver(factor=f"Recent headline context: {short}", impact="medium"))

    if len(drivers) > 8:
        drivers = drivers[:8]

    return PredictionExplainability(
        prediction=predicted_price,
        confidence=model_confidence,
        drivers=drivers,
        feature_importance=entries[:12],
        sentiment_contribution=sentiment_contribution_text(sentiment_score, sentiment_share),
        emotion_context=emotion_context_text(emotion_dominant, emotion_scores),
        recent_events=recent_events_from_news(news_articles),
        attribution_method="linear_coefficient_product",
    )


def build_heuristic_explainability(
    predicted_price: float,
    model_confidence: float,
    sentiment_score: float,
    price_history_volatility: float,
    news_articles: List[Dict[str, str]],
    emotion_dominant: str,
    emotion_scores: Optional[Dict[str, float]] = None,
) -> PredictionExplainability:
    """Fallback when LSTM or linear attribution is unavailable: drivers from sentiment, vol, headlines."""
    drivers: List[PredictionDriver] = []
    if abs(sentiment_score) > 0.2:
        drivers.append(
            PredictionDriver(
                factor="Sentiment signal (aggregated from social/news text)",
                impact="high" if abs(sentiment_score) > 0.45 else "medium",
            )
        )
    if price_history_volatility > 0:
        vol_pct = min(1.0, price_history_volatility / (abs(predicted_price) + 1e-6) * 100)
        drivers.append(
            PredictionDriver(
                factor="Recent price volatility (5-day std)",
                impact="high" if vol_pct > 3 else "medium" if vol_pct > 1.5 else "low",
            )
        )
    for ev in recent_events_from_news(news_articles, max_items=2):
        short = ev if len(ev) < 80 else ev[:77] + "…"
        drivers.append(PredictionDriver(factor=f"News headline: {short}", impact="medium"))

    if not drivers:
        drivers.append(
            PredictionDriver(
                factor="Model output (LSTM/heuristic — feature attribution not decomposed)",
                impact="medium",
            )
        )

    return PredictionExplainability(
        prediction=predicted_price,
        confidence=model_confidence,
        drivers=drivers[:8],
        feature_importance=[],
        sentiment_contribution=sentiment_contribution_text(sentiment_score, 0.15),
        emotion_context=emotion_context_text(emotion_dominant, emotion_scores),
        recent_events=recent_events_from_news(news_articles),
        attribution_method="heuristic_lstm_fallback",
    )


def volatility_proxy(price_history_close) -> float:
    """Last 5 closes std; returns 0 if unavailable."""
    try:
        import pandas as pd

        if hasattr(price_history_close, "tail"):
            s = price_history_close.tail(5)
            return float(s.std()) if len(s) > 1 else 0.0
    except Exception:
        pass
    return 0.0
