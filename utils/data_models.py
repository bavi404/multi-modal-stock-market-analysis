"""
Data models for the Multi-Modal Stock Market Analysis Framework
"""
from typing import List, Dict, Optional, Any
from datetime import datetime
from pydantic import BaseModel
import pandas as pd


class StockData(BaseModel):
    """Model for stock price data"""
    ticker: str
    prices: Dict[str, Any]  # Will contain pandas DataFrame data
    tweets: List[str]
    reddit_posts: List[str] 
    news_articles: List[Dict[str, str]]
    
    class Config:
        arbitrary_types_allowed = True


class SentimentResult(BaseModel):
    """Model for sentiment analysis results"""
    sentiment_score: float  # Range: -1.0 to 1.0
    dominant_emotion: str
    confidence: float
    individual_scores: List[Dict[str, float]]
    summary: str


class PredictionResult(BaseModel):
    """Model for price prediction results"""
    predicted_price: float
    confidence_interval: Dict[str, float]  # {'lower': x, 'upper': y}
    prediction_date: datetime
    model_confidence: float
    features_used: List[str]


class KnowledgeResult(BaseModel):
    """Model for knowledge graph and article recommendation results"""
    recommended_articles: List[Dict[str, str]]
    entities_extracted: List[Dict[str, str]]
    relationships_created: List[Dict[str, str]]
    graph_summary: str


class EmotionResult(BaseModel):
    """Model for emotion analysis results"""
    dominant_emotion: str
    emotion_scores: Dict[str, float]
    confidence: float
    summary: str


class AnalysisReport(BaseModel):
    """Complete analysis report model"""
    ticker: str
    analysis_date: datetime
    stock_data: StockData
    sentiment_analysis: SentimentResult
    emotion_analysis: EmotionResult
    price_prediction: PredictionResult
    knowledge_insights: KnowledgeResult
    executive_summary: str
    
    class Config:
        arbitrary_types_allowed = True

