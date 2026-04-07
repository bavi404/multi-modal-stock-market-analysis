"""
Agents package for the Multi-Modal Stock Market Analysis Framework
"""

from .orchestrator_agent import OrchestratorAgent
from .base_agent import BaseAgent
from .data_agent import DataAgent
from .data_gathering_agent import DataGatheringAgent
from .sentiment_agent import SentimentAgent
from .prediction_agent import PredictionAgent
from .price_prediction_agent import PricePredictionAgent
from .knowledge_agent import KnowledgeAgent
from .emotion_agent import EmotionAgent
from .advisor_agent import AdvisorAgent

__all__ = [
    'OrchestratorAgent',
    'BaseAgent',
    'DataAgent',
    'DataGatheringAgent', 
    'SentimentAgent',
    'PredictionAgent',
    'PricePredictionAgent',
    'KnowledgeAgent',
    'EmotionAgent',
    'AdvisorAgent',
]

