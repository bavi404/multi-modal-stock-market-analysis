"""
Orchestrator Agent - The master agent that coordinates all other agents
"""
import logging
from datetime import datetime
from typing import Dict, List, Optional
import pandas as pd

from .data_gathering_agent import DataGatheringAgent
from .sentiment_agent import SentimentAgent
from .price_prediction_agent import PricePredictionAgent
from .knowledge_agent import KnowledgeAgent
from .emotion_agent import EmotionAgent
from utils.data_models import AnalysisReport, StockData


class OrchestratorAgent:
    """Master agent that coordinates the entire analysis workflow"""
    
    def __init__(self):
        """Initialize the orchestrator with all sub-agents"""
        self.logger = logging.getLogger(__name__)
        
        # Initialize all agents
        self.logger.info("Initializing all agents...")
        
        self.data_agent = DataGatheringAgent()
        self.sentiment_agent = SentimentAgent()
        self.prediction_agent = PricePredictionAgent()
        self.knowledge_agent = KnowledgeAgent()
        self.emotion_agent = EmotionAgent()
        
        self.logger.info("All agents initialized successfully")
    
    def _get_company_name(self, ticker: str) -> str:
        """
        Get company name from ticker (simplified mapping)
        In a production system, this would use a proper API or database
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Company name for news searches
        """
        # Common ticker to company name mappings
        ticker_mapping = {
            'AAPL': 'Apple Inc',
            'GOOGL': 'Google Alphabet',
            'GOOG': 'Google Alphabet',
            'MSFT': 'Microsoft',
            'AMZN': 'Amazon',
            'TSLA': 'Tesla',
            'META': 'Meta Facebook',
            'NVDA': 'NVIDIA',
            'NFLX': 'Netflix',
            'AMD': 'Advanced Micro Devices',
            'INTC': 'Intel',
            'CRM': 'Salesforce',
            'ORCL': 'Oracle',
            'ADBE': 'Adobe',
            'PYPL': 'PayPal',
            'DIS': 'Disney',
            'BA': 'Boeing',
            'JPM': 'JPMorgan Chase',
            'V': 'Visa',
            'MA': 'Mastercard',
            'WMT': 'Walmart',
            'HD': 'Home Depot',
            'PG': 'Procter & Gamble',
            'JNJ': 'Johnson & Johnson',
            'UNH': 'UnitedHealth Group',
            'XOM': 'Exxon Mobil',
            'CVX': 'Chevron'
        }
        
        return ticker_mapping.get(ticker.upper(), ticker)
    
    def _combine_text_data(self, tweets: List[str], reddit_posts: List[str], 
                          news_articles: List[Dict[str, str]]) -> List[str]:
        """
        Combine all text data for sentiment analysis
        
        Args:
            tweets: List of tweet texts
            reddit_posts: List of Reddit post texts
            news_articles: List of news article dictionaries
            
        Returns:
            Combined list of all text data
        """
        all_texts = []
        
        # Add tweets
        all_texts.extend(tweets)
        
        # Add Reddit posts
        all_texts.extend(reddit_posts)
        
        # Add news article titles and descriptions
        for article in news_articles:
            title = article.get('title', '')
            description = article.get('description', '')
            if title:
                all_texts.append(title)
            if description and description != title:
                all_texts.append(description)
        
        # Filter out empty texts
        all_texts = [text for text in all_texts if text and text.strip()]
        
        return all_texts
    
    def _generate_executive_summary(self, ticker: str, stock_data: StockData,
                                   sentiment_result, emotion_result, prediction_result, 
                                   knowledge_result) -> str:
        """
        Generate an executive summary of the analysis
        
        Args:
            ticker: Stock ticker
            stock_data: Stock data object
            sentiment_result: Sentiment analysis result
            prediction_result: Price prediction result
            knowledge_result: Knowledge analysis result
            
        Returns:
            Executive summary string
        """
        try:
            # Get current price
            if isinstance(stock_data.prices, dict) and 'Close' in stock_data.prices:
                current_price = stock_data.prices['Close']
                if hasattr(current_price, 'iloc'):
                    current_price = current_price.iloc[-1]
            else:
                current_price = "N/A"
            
            # Format price
            if isinstance(current_price, (int, float)):
                price_str = f"${current_price:.2f}"
            else:
                price_str = str(current_price)
            
            # Get sentiment info
            sentiment_score = sentiment_result.sentiment_score
            sentiment_desc = "neutral"
            if sentiment_score > 0.2:
                sentiment_desc = "positive"
            elif sentiment_score > 0.5:
                sentiment_desc = "very positive"
            elif sentiment_score < -0.2:
                sentiment_desc = "negative"
            elif sentiment_score < -0.5:
                sentiment_desc = "very negative"
            
            # Get prediction info
            predicted_price = prediction_result.predicted_price
            price_change = predicted_price - (current_price if isinstance(current_price, (int, float)) else 0)
            price_change_pct = (price_change / current_price * 100) if isinstance(current_price, (int, float)) and current_price > 0 else 0
            
            direction = "increase" if price_change > 0 else "decrease"
            
            # Get knowledge info
            num_articles = len(knowledge_result.recommended_articles)
            num_entities = len(knowledge_result.entities_extracted)
            # Emotions
            dominant_market_emotion = emotion_result.dominant_emotion
            
            summary = f"""
EXECUTIVE SUMMARY - {ticker} Analysis

Current Status:
• Current Price: {price_str}
• Market Sentiment: {sentiment_desc.title()} (score: {sentiment_score:.2f})
• Emotion Signal: {dominant_market_emotion.title()} (confidence: {emotion_result.confidence:.2f})
• Predicted Next-Day Price: ${predicted_price:.2f} ({direction} of {abs(price_change_pct):.1f}%)

Key Insights:
• Analyzed {len(stock_data.tweets)} tweets, {len(stock_data.reddit_posts)} Reddit posts, and {len(stock_data.news_articles)} news articles
• Overall market sentiment is {sentiment_desc} with text-level emotion lean: {sentiment_result.dominant_emotion}; market emotion signal: {dominant_market_emotion}
• Price prediction model suggests a {direction} to ${predicted_price:.2f} (confidence: {prediction_result.model_confidence:.1f})
• Knowledge analysis processed {num_articles} relevant articles and identified {num_entities} key entities

Recommendation:
Based on the multi-modal analysis combining price data, social sentiment, and news analysis, 
the outlook for {ticker} appears {sentiment_desc}. The prediction model indicates a 
{direction} in price with moderate confidence.

Analysis completed on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            """.strip()
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Error generating executive summary: {e}")
            return f"Analysis completed for {ticker} on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    
    def run_analysis(self, ticker: str) -> AnalysisReport:
        """
        Run the complete multi-modal stock analysis
        
        Args:
            ticker: Stock ticker symbol (e.g., 'TSLA')
            
        Returns:
            AnalysisReport object with complete analysis results
        """
        self.logger.info(f"Starting complete analysis for {ticker}")
        analysis_start_time = datetime.now()
        
        try:
            # Step 1: Data Gathering
            self.logger.info("Step 1: Gathering all data...")
            company_name = self._get_company_name(ticker)
            raw_data = self.data_agent.gather_all_data(ticker, company_name)
            
            # Create StockData object
            stock_data = StockData(
                ticker=ticker,
                prices=raw_data['stock_prices'].to_dict() if not raw_data['stock_prices'].empty else {},
                tweets=raw_data['tweets'],
                reddit_posts=raw_data['reddit_posts'],
                news_articles=raw_data['news_articles']
            )
            
            # Step 2: Sentiment Analysis
            self.logger.info("Step 2: Analyzing sentiment...")
            all_text_data = self._combine_text_data(
                stock_data.tweets, 
                stock_data.reddit_posts, 
                stock_data.news_articles
            )
            
            sentiment_result = self.sentiment_agent.analyze(all_text_data)

            # Step 2.5: Emotion Analysis
            self.logger.info("Step 2.5: Detecting emotions...")
            emotion_result = self.emotion_agent.analyze(all_text_data)
            
            # Step 3: Price Prediction
            self.logger.info("Step 3: Predicting price...")
            if not raw_data['stock_prices'].empty:
                prediction_result = self.prediction_agent.predict(
                    raw_data['stock_prices'], 
                    sentiment_result.sentiment_score
                )
            else:
                self.logger.warning("No stock price data available for prediction")
                from utils.data_models import PredictionResult
                prediction_result = PredictionResult(
                    predicted_price=0.0,
                    confidence_interval={'lower': 0.0, 'upper': 0.0},
                    prediction_date=datetime.now(),
                    model_confidence=0.0,
                    features_used=[]
                )
            
            # Step 4: Knowledge Graph Analysis
            self.logger.info("Step 4: Analyzing knowledge and articles...")
            knowledge_result = self.knowledge_agent.analyze(stock_data.news_articles, ticker)
            
            # Step 5: Generate Executive Summary
            self.logger.info("Step 5: Generating executive summary...")
            executive_summary = self._generate_executive_summary(
                ticker, stock_data, sentiment_result, emotion_result, prediction_result, knowledge_result
            )
            
            # Create final report
            analysis_report = AnalysisReport(
                ticker=ticker,
                analysis_date=analysis_start_time,
                stock_data=stock_data,
                sentiment_analysis=sentiment_result,
                emotion_analysis=emotion_result,
                price_prediction=prediction_result,
                knowledge_insights=knowledge_result,
                executive_summary=executive_summary
            )
            
            analysis_duration = (datetime.now() - analysis_start_time).total_seconds()
            self.logger.info(f"Analysis completed for {ticker} in {analysis_duration:.2f} seconds")
            
            return analysis_report
            
        except Exception as e:
            self.logger.error(f"Error during analysis of {ticker}: {e}")
            
            # Return a minimal report with error information
            from utils.data_models import (StockData, SentimentResult, 
                                         PredictionResult, KnowledgeResult)
            
            error_stock_data = StockData(
                ticker=ticker,
                prices={},
                tweets=[],
                reddit_posts=[],
                news_articles=[]
            )
            
            error_sentiment = SentimentResult(
                sentiment_score=0.0,
                dominant_emotion="unknown",
                confidence=0.0,
                individual_scores=[],
                summary=f"Error occurred during sentiment analysis: {str(e)}"
            )
            
            error_prediction = PredictionResult(
                predicted_price=0.0,
                confidence_interval={'lower': 0.0, 'upper': 0.0},
                prediction_date=datetime.now(),
                model_confidence=0.0,
                features_used=[]
            )
            
            error_knowledge = KnowledgeResult(
                recommended_articles=[],
                entities_extracted=[],
                relationships_created=[],
                graph_summary=f"Error occurred during knowledge analysis: {str(e)}"
            )
            
            from utils.data_models import EmotionResult
            return AnalysisReport(
                ticker=ticker,
                analysis_date=analysis_start_time,
                stock_data=error_stock_data,
                sentiment_analysis=error_sentiment,
                emotion_analysis=EmotionResult(
                    dominant_emotion="neutral",
                    emotion_scores={},
                    confidence=0.0,
                    summary="Emotion analysis skipped due to error"
                ),
                price_prediction=error_prediction,
                knowledge_insights=error_knowledge,
                executive_summary=f"Analysis failed for {ticker}: {str(e)}"
            )
    
    def get_analysis_status(self) -> Dict[str, str]:
        """
        Get the status of all agents
        
        Returns:
            Dictionary with agent status information
        """
        return {
            'data_agent': 'Ready',
            'sentiment_agent': 'Ready' if self.sentiment_agent.sentiment_pipeline else 'Model not loaded',
            'prediction_agent': 'Ready',
            'knowledge_agent': 'Ready' if self.knowledge_agent.embedding_model else 'Model not loaded',
            'emotion_agent': 'Ready' if self.emotion_agent.emotion_pipeline else 'Model not loaded',
            'neo4j_connection': 'Connected' if self.knowledge_agent.neo4j_driver else 'Not connected'
        }
    
    def close(self):
        """Clean up resources and close connections"""
        self.logger.info("Closing orchestrator and cleaning up resources")
        
        try:
            self.knowledge_agent.close()
        except Exception as e:
            self.logger.error(f"Error closing knowledge agent: {e}")
        
        self.logger.info("Orchestrator closed successfully")

