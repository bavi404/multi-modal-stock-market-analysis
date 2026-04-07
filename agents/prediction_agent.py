from typing import Any, Dict, List, Optional

from agents.base_agent import BaseAgent
from agents.price_prediction_agent import PricePredictionAgent
from models.data_models import PredictionResult


class PredictionAgent(BaseAgent):
    """Independent agent responsible for price prediction (LinearRegression or LSTM)."""

    def __init__(self) -> None:
        super().__init__()
        self._predictor = PricePredictionAgent()
        # Keep backward compatibility with the Streamlit toggle.
        self.use_lstm = getattr(self._predictor, "use_lstm", False)

    @property
    def use_lstm(self) -> bool:  # type: ignore[override]
        return getattr(self._predictor, "use_lstm", False)

    @use_lstm.setter
    def use_lstm(self, value: bool) -> None:  # type: ignore[override]
        # Update underlying predictor mode.
        setattr(self._predictor, "use_lstm", bool(value))
        # If switching to LSTM, ensure model placeholder exists.
        if getattr(self._predictor, "use_lstm", False) and hasattr(self._predictor, "_init_lstm"):
            try:
                self._predictor._init_lstm()
            except Exception:
                # If initialization fails, prediction will fall back gracefully.
                self.logger.exception("Failed to init LSTM after toggling USE_LSTM")

    async def predict(
        self,
        price_history: Any,
        sentiment_score: float,
        news_articles: Optional[List[Dict[str, str]]] = None,
        emotion_dominant: Optional[str] = None,
        emotion_scores: Optional[Dict[str, float]] = None,
    ) -> PredictionResult:
        """
        Run prediction asynchronously (wraps the underlying sync predictor).
        """
        self.log_stage("predicting next-day price")
        return await self.run_blocking(
            self._predictor.predict,
            price_history,
            sentiment_score,
            news_articles,
            emotion_dominant,
            emotion_scores,
        )

    def close(self) -> None:
        # PricePredictionAgent has no persistent connections.
        return

