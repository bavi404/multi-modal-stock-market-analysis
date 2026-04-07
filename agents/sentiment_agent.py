"""
Sentiment Analysis Agent for analyzing text sentiment and emotions
"""
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import asyncio
import torch
import numpy as np
from typing import List, Dict, Tuple
import logging
from utils import config
from models.data_models import SentimentResult


class SentimentAgent:
    """Agent responsible for performing sentiment analysis on text data"""
    
    def __init__(self):
        """Initialize the sentiment analysis agent with pre-trained models"""
        self.logger = logging.getLogger(__name__)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize sentiment analysis pipeline
        self.sentiment_pipeline = None
        self._load_sentiment_model()
        
    def _load_sentiment_model(self):
        """Load the pre-trained sentiment analysis model"""
        try:
            self.logger.info(f"Loading sentiment model: {config.SENTIMENT_MODEL}")
            
            # Load FinBERT for financial sentiment analysis
            self.sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model=config.SENTIMENT_MODEL,
                tokenizer=config.SENTIMENT_MODEL,
                device=0 if torch.cuda.is_available() else -1
            )
            
            self.logger.info("Sentiment model loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Error loading sentiment model: {e}")
            # Fallback to a general sentiment model
            try:
                self.logger.info("Falling back to general sentiment model")
                self.sentiment_pipeline = pipeline(
                    "sentiment-analysis",
                    model="distilbert-base-uncased-finetuned-sst-2-english",
                    device=0 if torch.cuda.is_available() else -1
                )
                self.logger.info("Fallback sentiment model loaded successfully")
            except Exception as fallback_error:
                self.logger.error(f"Error loading fallback model: {fallback_error}")
    
    def _preprocess_text(self, text: str) -> str:
        """
        Preprocess text for sentiment analysis
        
        Args:
            text: Raw text to preprocess
            
        Returns:
            Cleaned text
        """
        if not text or not isinstance(text, str):
            return ""
            
        # Basic text cleaning
        text = text.strip()
        text = ' '.join(text.split())  # Remove extra whitespace
        
        # Truncate if too long (BERT models have token limits)
        if len(text) > 512:
            text = text[:512]
            
        return text
    
    def _analyze_single_text(self, text: str) -> Dict[str, float]:
        """
        Analyze sentiment of a single text
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with sentiment scores
        """
        if not self.sentiment_pipeline:
            return {'label': 'NEUTRAL', 'score': 0.0}
            
        try:
            cleaned_text = self._preprocess_text(text)
            if not cleaned_text:
                return {'label': 'NEUTRAL', 'score': 0.0}
                
            result = self.sentiment_pipeline(cleaned_text)[0]
            
            # Normalize the result based on the model type
            if config.SENTIMENT_MODEL == 'ProsusAI/finbert':
                # FinBERT returns positive, negative, neutral
                label_mapping = {
                    'positive': 1.0,
                    'negative': -1.0,
                    'neutral': 0.0
                }
                sentiment_score = label_mapping.get(result['label'].lower(), 0.0) * result['score']
            else:
                # General models return POSITIVE/NEGATIVE
                label_mapping = {
                    'POSITIVE': 1.0,
                    'NEGATIVE': -1.0,
                    'NEUTRAL': 0.0
                }
                sentiment_score = label_mapping.get(result['label'], 0.0) * result['score']
            
            return {
                'label': result['label'],
                'score': sentiment_score,
                'confidence': result['score']
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing text sentiment: {e}")
            return {'label': 'NEUTRAL', 'score': 0.0, 'confidence': 0.0}
    
    def _determine_dominant_emotion(self, individual_scores: List[dict]) -> str:
        """
        Determine the dominant emotion from individual sentiment scores
        
        Args:
            individual_scores: List of individual sentiment analysis results
            
        Returns:
            String representing the dominant emotion
        """
        if not individual_scores:
            return "neutral"
            
        # Calculate average sentiment
        avg_sentiment = np.mean([score.get('score', 0.0) for score in individual_scores])
        
        # Map sentiment to financial emotions
        if avg_sentiment > 0.3:
            return "optimistic"
        elif avg_sentiment > 0.1:
            return "positive"
        elif avg_sentiment < -0.3:
            return "fearful"
        elif avg_sentiment < -0.1:
            return "pessimistic"
        else:
            return "neutral"
    
    def _calculate_aggregate_sentiment(self, individual_scores: List[dict]) -> float:
        """
        Calculate aggregate sentiment score from individual analyses
        
        Args:
            individual_scores: List of individual sentiment analysis results
            
        Returns:
            Aggregate sentiment score between -1.0 and 1.0
        """
        if not individual_scores:
            return 0.0
            
        # Use weighted average based on confidence
        total_weight = 0.0
        weighted_sum = 0.0
        
        for score_dict in individual_scores:
            score = score_dict.get('score', 0.0)
            confidence = score_dict.get('confidence', 0.0)
            
            # Use confidence as weight
            weight = max(confidence, 0.1)  # Minimum weight to avoid zero weights
            weighted_sum += score * weight
            total_weight += weight
        
        if total_weight == 0:
            return 0.0
            
        aggregate_score = weighted_sum / total_weight
        
        # Ensure the score is within bounds
        return max(-1.0, min(1.0, aggregate_score))
    
    def _generate_summary(self, sentiment_score: float, dominant_emotion: str, 
                         text_count: int) -> str:
        """
        Generate a human-readable summary of the sentiment analysis
        
        Args:
            sentiment_score: Aggregate sentiment score
            dominant_emotion: Dominant emotion detected
            text_count: Number of texts analyzed
            
        Returns:
            Summary string
        """
        sentiment_description = "neutral"
        if sentiment_score > 0.2:
            sentiment_description = "positive"
        elif sentiment_score > 0.5:
            sentiment_description = "very positive"
        elif sentiment_score < -0.2:
            sentiment_description = "negative"
        elif sentiment_score < -0.5:
            sentiment_description = "very negative"
        
        return (f"Analyzed {text_count} texts. Overall sentiment is {sentiment_description} "
                f"(score: {sentiment_score:.2f}) with a dominant emotion of {dominant_emotion}.")
    
    def analyze(self, texts: List[str]) -> SentimentResult:
        """
        Analyze sentiment of a list of texts
        
        Args:
            texts: List of text strings to analyze
            
        Returns:
            SentimentResult object containing analysis results
        """
        self.logger.info(f"Starting sentiment analysis on {len(texts)} texts")
        
        if not texts:
            fb = float(getattr(config, "FALLBACK_SENTIMENT_SCORE", 0.0))
            self.logger.warning("SentimentAgent: no texts — using configured neutral fallback (score=%s)", fb)
            return SentimentResult(
                sentiment_score=fb,
                dominant_emotion="neutral",
                confidence=0.0,
                individual_scores=[],
                summary=(
                    f"No texts available (data sources empty or skipped); "
                    f"using fallback sentiment score {fb:.2f}."
                ),
            )
        
        # Analyze each text individually
        individual_scores = []
        for i, text in enumerate(texts):
            if i % 10 == 0:  # Log progress every 10 texts
                self.logger.info(f"Processing text {i+1}/{len(texts)}")
                
            score = self._analyze_single_text(text)
            individual_scores.append(score)
        
        # Calculate aggregate metrics
        aggregate_sentiment = self._calculate_aggregate_sentiment(individual_scores)
        dominant_emotion = self._determine_dominant_emotion(individual_scores)
        
        # Calculate overall confidence
        confidences = [score.get('confidence', 0.0) for score in individual_scores]
        overall_confidence = np.mean(confidences) if confidences else 0.0
        
        # Generate summary
        summary = self._generate_summary(aggregate_sentiment, dominant_emotion, len(texts))
        
        result = SentimentResult(
            sentiment_score=aggregate_sentiment,
            dominant_emotion=dominant_emotion,
            confidence=overall_confidence,
            individual_scores=individual_scores,
            summary=summary
        )
        
        self.logger.info(f"Sentiment analysis completed. Score: {aggregate_sentiment:.2f}, "
                        f"Emotion: {dominant_emotion}")
        
        return result
    
    def analyze_batch(self, text_batches: Dict[str, List[str]]) -> Dict[str, SentimentResult]:
        """
        Analyze sentiment for multiple batches of texts (e.g., tweets vs news vs reddit)
        
        Args:
            text_batches: Dictionary with source names as keys and text lists as values
            
        Returns:
            Dictionary with source names as keys and SentimentResult objects as values
        """
        results = {}
        
        for source, texts in text_batches.items():
            self.logger.info(f"Analyzing sentiment for {source}")
            results[source] = self.analyze(texts)
            
        return results

    async def analyze_async(self, texts: List[str]) -> SentimentResult:
        """Async wrapper for orchestration pipelines."""
        return await asyncio.to_thread(self.analyze, texts)

