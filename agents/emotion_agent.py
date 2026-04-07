"""
Emotion Analysis Agent for detecting market emotions from text
"""
from transformers import pipeline
import asyncio
import numpy as np
from typing import List, Dict, Any
import logging
from utils import config
from models.data_models import EmotionResult


class EmotionAgent:
    """Agent responsible for detecting emotions like fear, greed, confidence, uncertainty"""
    
    def __init__(self):
        """Initialize the emotion analysis agent with a pre-trained model"""
        self.logger = logging.getLogger(__name__)
        self.emotion_pipeline = None
        self._load_emotion_model()
    
    def _load_emotion_model(self):
        """Load the pre-trained emotion classification model"""
        try:
            self.logger.info(f"Loading emotion model: {config.EMOTION_MODEL}")
            self.emotion_pipeline = pipeline(
                "text-classification",
                model=config.EMOTION_MODEL,
                return_all_scores=True,
                truncation=True
            )
            self.logger.info("Emotion model loaded successfully")
        except Exception as e:
            self.logger.error(f"Error loading emotion model: {e}")
            self.emotion_pipeline = None
    
    def _preprocess_text(self, text: str) -> str:
        if not text or not isinstance(text, str):
            return ""
        text = text.strip()
        text = ' '.join(text.split())
        return text[:512]
    
    def _aggregate_emotions(self, per_text_scores: List[Dict[str, float]]) -> Dict[str, float]:
        if not per_text_scores:
            return {}
        # Average scores across texts
        keys = list(per_text_scores[0].keys())
        agg = {k: 0.0 for k in keys}
        for scores in per_text_scores:
            for k in keys:
                agg[k] += scores.get(k, 0.0)
        n = float(len(per_text_scores))
        return {k: (v / n) for k, v in agg.items()}
    
    def _map_to_market_emotions(self, emotion_scores: Dict[str, float]) -> str:
        if not emotion_scores:
            return "neutral"
        # Map common emotion labels to market meta-emotions
        # Base model labels (e.g., j-hartmann): joy, anger, fear, sadness, surprise, disgust, neutral, etc.
        fear = emotion_scores.get('fear', 0.0) + emotion_scores.get('sadness', 0.0)
        greed = emotion_scores.get('joy', 0.0) + emotion_scores.get('optimism', 0.0)
        confidence = emotion_scores.get('trust', 0.0) + emotion_scores.get('confidence', 0.0) + emotion_scores.get('joy', 0.0)
        uncertainty = emotion_scores.get('anticipation', 0.0) + emotion_scores.get('surprise', 0.0) + emotion_scores.get('confusion', 0.0)
        neutral = emotion_scores.get('neutral', 0.0)
        market_map = {
            'fear': fear,
            'greed': greed,
            'confidence': confidence,
            'uncertainty': uncertainty,
            'neutral': neutral
        }
        return max(market_map.items(), key=lambda x: x[1])[0]

    @staticmethod
    def _scores_list_from_pipeline(raw: Any) -> List[Dict[str, Any]]:
        """HF text-classification may return [[{label, score}, ...]] or [{...}, ...]."""
        if not raw:
            return []
        first = raw[0]
        if isinstance(first, dict):
            return list(raw)
        if isinstance(first, list):
            return list(first)
        return []

    def analyze(self, texts: List[str]) -> EmotionResult:
        """Analyze market emotions across a list of texts"""
        self.logger.info(f"Starting emotion analysis on {len(texts)} texts")
        if not texts or not self.emotion_pipeline:
            label = getattr(config, "FALLBACK_EMOTION_LABEL", "neutral")
            self.logger.warning(
                "EmotionAgent: no texts or pipeline unavailable — using fallback emotion '%s'",
                label,
            )
            return EmotionResult(
                dominant_emotion=label,
                emotion_scores={},
                confidence=0.0,
                summary=(
                    "No texts provided or model unavailable; "
                    f"using fallback emotion label '{label}'."
                ),
            )
        per_text_scores = []
        confidences = []
        for i, text in enumerate(texts):
            try:
                cleaned = self._preprocess_text(text)
                if not cleaned:
                    continue
                raw = self.emotion_pipeline(cleaned, top_k=None)
                result = self._scores_list_from_pipeline(raw)
                if not result:
                    continue
                score_dict = {item['label'].lower(): float(item['score']) for item in result}
                per_text_scores.append(score_dict)
                confidences.append(max(float(item['score']) for item in result))
            except Exception as e:
                self.logger.error(f"Error analyzing emotion for text {i}: {e}")
                continue
        agg_scores = self._aggregate_emotions(per_text_scores)
        dominant = self._map_to_market_emotions(agg_scores)
        overall_conf = float(np.mean(confidences)) if confidences else 0.0
        summary = (f"Detected dominant market emotion: {dominant}. "
                   f"Top signals: {sorted(agg_scores.items(), key=lambda x: x[1], reverse=True)[:3]}")
        return EmotionResult(
            dominant_emotion=dominant,
            emotion_scores=agg_scores,
            confidence=overall_conf,
            summary=summary
        )

    async def analyze_async(self, texts: List[str]) -> EmotionResult:
        """Async wrapper for orchestration pipelines."""
        return await asyncio.to_thread(self.analyze, texts)


