"""
Pydantic domain models for the multi-modal stock analysis framework.
"""
from typing import List, Dict, Optional, Any, Literal
from datetime import datetime
from pydantic import BaseModel, Field
import pandas as pd


class StockData(BaseModel):
    """Model for stock price data"""
    ticker: str
    prices: Dict[str, Any]
    tweets: List[str]
    reddit_posts: List[str]
    news_articles: List[Dict[str, str]]

    class Config:
        arbitrary_types_allowed = True


DataSourceStatusKind = Literal["ok", "failed", "skipped", "empty", "cached"]


class DataSourceStatus(BaseModel):
    """Per-source outcome for a gather run (used in reports and /health)."""

    source: str
    status: DataSourceStatusKind
    message: Optional[str] = None


class DataResult(BaseModel):
    """Raw gathered data for a given ticker."""
    ticker: str
    stock_prices: Any
    tweets: List[str]
    reddit_posts: List[str]
    news_articles: List[Dict[str, str]]
    data_gathered_at: datetime
    source_details: List[DataSourceStatus] = Field(default_factory=list)

    class Config:
        arbitrary_types_allowed = True


class MultimodalContext(BaseModel):
    """Minimal multimodal context used by the AdvisorAgent."""
    ticker: str
    latest_price: Optional[float] = None
    sentiment_score: float = 0.0
    dominant_emotion: str = "neutral"
    predicted_price: Optional[float] = None
    model_confidence: Optional[float] = None
    tweets_count: int = 0
    reddit_count: int = 0
    news_count: int = 0

    class Config:
        arbitrary_types_allowed = True


class ChatTurn(BaseModel):
    """One message in short-term advisor conversation memory."""

    role: Literal["user", "assistant"]
    content: str


DriverImpact = Literal["high", "medium", "low"]


class PredictionDriver(BaseModel):
    """Human-readable driver for prediction explainability."""

    factor: str
    impact: DriverImpact


class FeatureImportanceEntry(BaseModel):
    """Approximate importance for one model feature."""

    feature: str
    label: str
    importance: float


class PredictionExplainability(BaseModel):
    """Structured output for why the model predicted what it did."""

    prediction: float
    confidence: float
    drivers: List[PredictionDriver] = Field(default_factory=list)
    feature_importance: List[FeatureImportanceEntry] = Field(default_factory=list)
    sentiment_contribution: str = ""
    emotion_context: str = ""
    recent_events: List[str] = Field(default_factory=list)
    attribution_method: str = "linear_coefficient_product"


class PredictionResult(BaseModel):
    """Model for price prediction results"""
    predicted_price: float
    confidence_interval: Dict[str, float]
    prediction_date: datetime
    model_confidence: float
    features_used: List[str]
    explainability: Optional[PredictionExplainability] = None


class PricePredictionSignals(BaseModel):
    """Structured price / model output for the advisor data layer."""

    latest_price: Optional[float] = None
    predicted_price: Optional[float] = None
    model_confidence: Optional[float] = None
    confidence_interval_lower: Optional[float] = None
    confidence_interval_upper: Optional[float] = None
    prediction_horizon_note: str = "Model uses recent prices + sentiment as features; treat as indicative only."
    explainability: Optional[PredictionExplainability] = None


class SentimentSignals(BaseModel):
    score: float
    summary: str


class EmotionSignals(BaseModel):
    dominant: str
    summary: str
    top_scores: Dict[str, float] = Field(default_factory=dict)


class NewsHeadlineItem(BaseModel):
    title: str
    source: Optional[str] = None


class AdvisorDataLayer(BaseModel):
    """Deterministic pipeline output passed to the LLM reasoning layer."""

    layer: Literal["advisor_data"] = "advisor_data"
    ticker: str
    as_of: datetime
    price_prediction: PricePredictionSignals
    sentiment: SentimentSignals
    emotion: EmotionSignals
    key_news_events: List[NewsHeadlineItem] = Field(default_factory=list)
    source_counts: Dict[str, int] = Field(default_factory=dict)


class LiveUpdateResult(BaseModel):
    """Structured payload pushed to the live WebSocket panel."""

    type: str = "live_update"
    timestamp: datetime
    ticker: str
    latest_price: Optional[float] = None
    sentiment_score: float = 0.0
    dominant_emotion: str = "neutral"
    predicted_price: Optional[float] = None
    model_confidence: Optional[float] = None
    tweets_count: int = 0
    reddit_count: int = 0
    news_count: int = 0


class IndividualSentimentScore(BaseModel):
    """One FinBERT (or pipeline) output: string label plus numeric score and model confidence."""

    label: str
    score: float
    confidence: float = 0.0


class SentimentResult(BaseModel):
    """Model for sentiment analysis results"""
    sentiment_score: float
    dominant_emotion: str
    confidence: float
    individual_scores: List[IndividualSentimentScore]
    summary: str


class KnowledgeResult(BaseModel):
    """Model for knowledge graph and article recommendation results"""
    recommended_articles: List[Dict[str, str]]
    # spaCy offsets (start/end) are ints; values are not all strings
    entities_extracted: List[Dict[str, Any]]
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
    performance_summary: Optional["PerformanceSummary"] = None
    data_source_status: List[DataSourceStatus] = Field(default_factory=list)

    class Config:
        arbitrary_types_allowed = True


class PerformanceStage(BaseModel):
    name: str
    duration_seconds: float
    succeeded: bool
    error: Optional[str] = None


class PerformanceSummary(BaseModel):
    """Timing information for the full pipeline."""

    started_at: datetime
    finished_at: datetime
    total_duration_seconds: float
    stages: List[PerformanceStage]
