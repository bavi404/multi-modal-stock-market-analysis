"""
Emotion Analysis Agent for detecting market emotions from text
"""
from transformers import pipeline
import numpy as np
from typing import List, Dict
import logging
import config
from utils.data_models import EmotionResult


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
    
    def analyze(self, texts: List[str]) -> EmotionResult:
        """Analyze market emotions across a list of texts"""
        self.logger.info(f"Starting emotion analysis on {len(texts)} texts")
        if not texts or not self.emotion_pipeline:
            return EmotionResult(
                dominant_emotion="neutral",
                emotion_scores={},
                confidence=0.0,
                summary="No texts provided or model unavailable for emotion analysis."
            )
        per_text_scores = []
        confidences = []
        for i, text in enumerate(texts):
            try:
                cleaned = self._preprocess_text(text)
                if not cleaned:
                    continue
                result = self.emotion_pipeline(cleaned, top_k=None)[0]
                # Convert to dict
                score_dict = {item['label'].lower(): float(item['score']) for item in result}
                per_text_scores.append(score_dict)
                confidences.append(max(item['score'] for item in result))
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


