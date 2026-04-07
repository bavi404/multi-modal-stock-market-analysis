"""
Orchestrator: wires data, NLP, prediction, knowledge, emotion, and advisor agents.

Exposes a full async pipeline (:meth:`run_analysis_async`), a synchronous wrapper
(:meth:`run_analysis`), and lighter paths for live UI and streaming (:meth:`get_live_snapshot_async`).
"""

from __future__ import annotations

import logging
import asyncio
from dataclasses import dataclass
from datetime import datetime
import time
from typing import Dict, List, Optional
from .data_agent import DataAgent
from .sentiment_agent import SentimentAgent
from .prediction_agent import PredictionAgent
from .knowledge_agent import KnowledgeAgent
from .emotion_agent import EmotionAgent
from .advisor_agent import AdvisorAgent
from models.data_models import (
    AnalysisReport,
    StockData,
    LiveUpdateResult,
    PerformanceSummary,
    PerformanceStage,
    DataResult,
    SentimentResult,
    EmotionResult,
    PredictionResult,
    KnowledgeResult,
    AdvisorDataLayer,
    PricePredictionSignals,
    SentimentSignals,
    EmotionSignals,
    NewsHeadlineItem,
)


@dataclass
class _LivePipelineBundle:
    """Single pass of data + NLP + prediction for live UI and advisor."""

    ticker: str
    timestamp: datetime
    data_result: DataResult
    sentiment_result: SentimentResult
    emotion_result: EmotionResult
    prediction_result: Optional[PredictionResult]
    latest_price: Optional[float]


class OrchestratorAgent:
    """Coordinates end-to-end analysis and shared sub-agents for the HTTP/WebSocket stack."""

    def __init__(self) -> None:
        """Construct sub-agents (data, sentiment, prediction, knowledge, emotion, advisor)."""
        self.logger = logging.getLogger(__name__)
        
        # Initialize all agents
        self.logger.info("Initializing all agents...")
        
        self.data_agent = DataAgent()
        self.sentiment_agent = SentimentAgent()
        self.prediction_agent = PredictionAgent()
        self.knowledge_agent = KnowledgeAgent()
        self.emotion_agent = EmotionAgent()
        self.advisor_agent = AdvisorAgent()
        
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
    
    def _generate_executive_summary(
        self,
        ticker: str,
        stock_data: StockData,
        sentiment_result,
        emotion_result,
        prediction_result,
        knowledge_result,
        source_details: Optional[List] = None,
    ) -> str:
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

            expl = getattr(prediction_result, "explainability", None)
            explain_block = ""
            if expl is not None and expl.drivers:
                lines = "\n".join(
                    f"  – {d.factor} (impact: {d.impact})" for d in expl.drivers[:6]
                )
                method = expl.attribution_method
                explain_block = f"""
Prediction explainability ({method}):
{lines}
• Sentiment role: {expl.sentiment_contribution[:280]}{"…" if len(expl.sentiment_contribution) > 280 else ""}
"""
            
            source_block = ""
            if source_details:
                lines = "\n".join(
                    f"  • {getattr(s, 'source', '?')}: {getattr(s, 'status', '?')}"
                    + (f" — {s.message}" if getattr(s, 'message', None) else "")
                    for s in source_details
                )
                source_block = f"\nData sources (used / skipped / cached):\n{lines}\n"

            summary = f"""
EXECUTIVE SUMMARY - {ticker} Analysis

Current Status:
• Current Price: {price_str}
• Market Sentiment: {sentiment_desc.title()} (score: {sentiment_score:.2f})
• Emotion Signal: {dominant_market_emotion.title()} (confidence: {emotion_result.confidence:.2f})
• Predicted Next-Day Price: ${predicted_price:.2f} ({direction} of {abs(price_change_pct):.1f}%)
{explain_block}{source_block}
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
    
    async def run_analysis_async(self, ticker: str) -> AnalysisReport:
        """
        Async pipeline for multi-modal stock analysis.

        - Runs independent tasks in parallel where possible
        - Uses asyncio.to_thread to wrap existing blocking agent logic
        - Degrades gracefully when any source/model is unavailable
        """
        self.logger.info("OrchestratorAgent: starting complete analysis for %s", ticker)
        analysis_start_time = datetime.now()
        t0 = time.perf_counter()
        stages: list[PerformanceStage] = []

        try:
            # Step 1: Data Gathering
            self.logger.info("OrchestratorAgent: Step 1 - Gathering all data")
            company_name = self._get_company_name(ticker)
            s = time.perf_counter()
            data_result = await self.data_agent.gather_all_data(ticker, company_name)
            stages.append(
                PerformanceStage(
                    name="data_gathering",
                    duration_seconds=time.perf_counter() - s,
                    succeeded=True,
                )
            )

            # Normalize stock prices into the StockData contract
            price_df = data_result.stock_prices
            prices_dict = (
                price_df.to_dict()
                if price_df is not None and hasattr(price_df, "empty") and not price_df.empty
                else {}
            )
            stock_data = StockData(
                ticker=ticker,
                prices=prices_dict,
                tweets=data_result.tweets,
                reddit_posts=data_result.reddit_posts,
                news_articles=data_result.news_articles,
            )

            # Step 2: Build combined text corpus
            self.logger.info("OrchestratorAgent: Step 2 - Combining text for NLP")
            all_text_data = self._combine_text_data(
                stock_data.tweets,
                stock_data.reddit_posts,
                stock_data.news_articles,
            )

            # Step 2.5/4: Run independent NLP/knowledge steps in parallel
            self.logger.info("OrchestratorAgent: Step 3 - Running sentiment/emotion/knowledge in parallel")
            s_nlp = time.perf_counter()
            sentiment_task = asyncio.create_task(self.sentiment_agent.analyze_async(all_text_data))
            emotion_task = asyncio.create_task(self.emotion_agent.analyze_async(all_text_data))
            knowledge_task = asyncio.create_task(self.knowledge_agent.analyze_async(stock_data.news_articles, ticker))

            sentiment_result, emotion_result, knowledge_result = await asyncio.gather(
                sentiment_task, emotion_task, knowledge_task
            )
            stages.append(
                PerformanceStage(
                    name="nlp_and_knowledge",
                    duration_seconds=time.perf_counter() - s_nlp,
                    succeeded=True,
                )
            )

            # Step 3: Price Prediction (depends on sentiment + price history)
            self.logger.info("OrchestratorAgent: Step 4 - Predicting price")
            s_pred = time.perf_counter()
            if price_df is not None and hasattr(price_df, "empty") and not price_df.empty:
                prediction_result = await self.prediction_agent.predict(
                    price_df,
                    sentiment_result.sentiment_score,
                    news_articles=data_result.news_articles,
                    emotion_dominant=emotion_result.dominant_emotion,
                    emotion_scores=emotion_result.emotion_scores,
                )
                stages.append(
                    PerformanceStage(
                        name="prediction",
                        duration_seconds=time.perf_counter() - s_pred,
                        succeeded=True,
                    )
                )
            else:
                self.logger.warning("OrchestratorAgent: No price history available - skipping prediction")
                prediction_result = PredictionResult(
                    predicted_price=0.0,
                    confidence_interval={"lower": 0.0, "upper": 0.0},
                    prediction_date=datetime.now(),
                    model_confidence=0.0,
                    features_used=[],
                )
                stages.append(
                    PerformanceStage(
                        name="prediction",
                        duration_seconds=time.perf_counter() - s_pred,
                        succeeded=False,
                        error="No price history available",
                    )
                )

            # Step 5: Executive summary
            self.logger.info("OrchestratorAgent: Step 5 - Generating executive summary")
            s_summary = time.perf_counter()
            executive_summary = self._generate_executive_summary(
                ticker,
                stock_data,
                sentiment_result,
                emotion_result,
                prediction_result,
                knowledge_result,
                source_details=data_result.source_details,
            )
            stages.append(
                PerformanceStage(
                    name="summary_generation",
                    duration_seconds=time.perf_counter() - s_summary,
                    succeeded=True,
                )
            )

            finished_at = datetime.now()
            total_duration = time.perf_counter() - t0
            perf = PerformanceSummary(
                started_at=analysis_start_time,
                finished_at=finished_at,
                total_duration_seconds=total_duration,
                stages=stages,
            )

            analysis_report = AnalysisReport(
                ticker=ticker,
                analysis_date=analysis_start_time,
                stock_data=stock_data,
                sentiment_analysis=sentiment_result,
                emotion_analysis=emotion_result,
                price_prediction=prediction_result,
                knowledge_insights=knowledge_result,
                executive_summary=executive_summary,
                performance_summary=perf,
                data_source_status=data_result.source_details,
            )

            self.logger.info("OrchestratorAgent: analysis complete for %s in %.2f seconds", ticker, total_duration)
            return analysis_report

        except Exception as e:
            self.logger.exception("OrchestratorAgent: Error during analysis of %s: %s", ticker, e)

            # Degraded fallback report
            error_stock_data = StockData(
                ticker=ticker,
                prices={},
                tweets=[],
                reddit_posts=[],
                news_articles=[],
            )

            error_sentiment = SentimentResult(
                sentiment_score=0.0,
                dominant_emotion="unknown",
                confidence=0.0,
                individual_scores=[],
                summary=f"Error occurred during sentiment analysis: {str(e)}",
            )

            error_prediction = PredictionResult(
                predicted_price=0.0,
                confidence_interval={"lower": 0.0, "upper": 0.0},
                prediction_date=datetime.now(),
                model_confidence=0.0,
                features_used=[],
            )

            error_knowledge = KnowledgeResult(
                recommended_articles=[],
                entities_extracted=[],
                relationships_created=[],
                graph_summary=f"Error occurred during knowledge analysis: {str(e)}",
            )

            finished_at = datetime.now()
            total_duration = time.perf_counter() - t0
            perf = PerformanceSummary(
                started_at=analysis_start_time,
                finished_at=finished_at,
                total_duration_seconds=total_duration,
                stages=stages
                or [
                    PerformanceStage(
                        name="pipeline",
                        duration_seconds=total_duration,
                        succeeded=False,
                        error=str(e),
                    )
                ],
            )

            return AnalysisReport(
                ticker=ticker,
                analysis_date=analysis_start_time,
                stock_data=error_stock_data,
                sentiment_analysis=error_sentiment,
                emotion_analysis=EmotionResult(
                    dominant_emotion="neutral",
                    emotion_scores={},
                    confidence=0.0,
                    summary="Emotion analysis skipped due to error",
                ),
                price_prediction=error_prediction,
                knowledge_insights=error_knowledge,
                executive_summary=f"Analysis failed for {ticker}: {str(e)}",
                performance_summary=perf,
                data_source_status=[],
            )

    def run_analysis(self, ticker: str) -> AnalysisReport:
        """
        Sync wrapper for existing CLI/UI usage.
        """
        try:
            # If an event loop is already running in this thread, run in a background thread.
            loop = asyncio.get_running_loop()
            if loop.is_running():
                import threading

                result_holder = {}

                def _worker() -> None:
                    result_holder["result"] = asyncio.run(self.run_analysis_async(ticker))

                t = threading.Thread(target=_worker, daemon=True)
                t.start()
                t.join()
                return result_holder["result"]
        except RuntimeError:
            # No event loop running -> safe to asyncio.run directly
            pass

        return asyncio.run(self.run_analysis_async(ticker))

    async def _run_live_pipeline_async(self, ticker: str) -> _LivePipelineBundle:
        self.logger.info("OrchestratorAgent: running live pipeline for %s", ticker)
        company_name = self._get_company_name(ticker)
        data_result = await self.data_agent.gather_all_data(ticker, company_name)

        price_df = data_result.stock_prices
        latest_price = None
        has_prices = price_df is not None and hasattr(price_df, "empty") and not price_df.empty
        if has_prices:
            latest_price = float(price_df["Close"].iloc[-1])

        texts = self._combine_text_data(data_result.tweets, data_result.reddit_posts, data_result.news_articles)

        sentiment_task = asyncio.create_task(self.sentiment_agent.analyze_async(texts))
        emotion_task = asyncio.create_task(self.emotion_agent.analyze_async(texts))

        sentiment_result, emotion_result = await asyncio.gather(sentiment_task, emotion_task)

        prediction_result: Optional[PredictionResult] = None
        if has_prices:
            prediction_result = await self.prediction_agent.predict(
                price_df,
                sentiment_result.sentiment_score,
                news_articles=data_result.news_articles,
                emotion_dominant=emotion_result.dominant_emotion,
                emotion_scores=emotion_result.emotion_scores,
            )

        return _LivePipelineBundle(
            ticker=ticker,
            timestamp=datetime.utcnow(),
            data_result=data_result,
            sentiment_result=sentiment_result,
            emotion_result=emotion_result,
            prediction_result=prediction_result,
            latest_price=latest_price,
        )

    def _bundle_to_live_update(self, b: _LivePipelineBundle) -> LiveUpdateResult:
        pred = b.prediction_result
        return LiveUpdateResult(
            timestamp=b.timestamp,
            ticker=b.ticker,
            latest_price=b.latest_price,
            sentiment_score=b.sentiment_result.sentiment_score,
            dominant_emotion=b.emotion_result.dominant_emotion,
            predicted_price=pred.predicted_price if pred else None,
            model_confidence=pred.model_confidence if pred else None,
            tweets_count=len(b.data_result.tweets),
            reddit_count=len(b.data_result.reddit_posts),
            news_count=len(b.data_result.news_articles),
        )

    def _top_emotion_scores(self, emotion: EmotionResult, n: int = 5) -> Dict[str, float]:
        if not emotion.emotion_scores:
            return {}
        items = sorted(emotion.emotion_scores.items(), key=lambda x: x[1], reverse=True)[:n]
        return dict(items)

    def _headlines_for_advisor(self, articles: List[Dict[str, str]]) -> List[NewsHeadlineItem]:
        from utils import config as _cfg

        max_n = int(getattr(_cfg, "ADVISOR_MAX_NEWS_HEADLINES", 8))
        out: List[NewsHeadlineItem] = []
        for a in articles[:max_n]:
            title = (a.get("title") or "").strip()
            if not title:
                title = (a.get("description") or "").strip()[:200]
            if not title:
                continue
            src = a.get("source") or a.get("name")
            out.append(NewsHeadlineItem(title=title[:500], source=src))
        return out

    def _bundle_to_advisor_data_layer(self, b: _LivePipelineBundle) -> AdvisorDataLayer:
        pred = b.prediction_result
        ci_lo = ci_hi = None
        if pred and pred.confidence_interval:
            ci_lo = pred.confidence_interval.get("lower")
            ci_hi = pred.confidence_interval.get("upper")
        return AdvisorDataLayer(
            ticker=b.ticker,
            as_of=b.timestamp,
            price_prediction=PricePredictionSignals(
                latest_price=b.latest_price,
                predicted_price=pred.predicted_price if pred else None,
                model_confidence=pred.model_confidence if pred else None,
                confidence_interval_lower=ci_lo,
                confidence_interval_upper=ci_hi,
                explainability=pred.explainability if pred else None,
            ),
            sentiment=SentimentSignals(
                score=b.sentiment_result.sentiment_score,
                summary=b.sentiment_result.summary,
            ),
            emotion=EmotionSignals(
                dominant=b.emotion_result.dominant_emotion,
                summary=b.emotion_result.summary,
                top_scores=self._top_emotion_scores(b.emotion_result),
            ),
            key_news_events=self._headlines_for_advisor(b.data_result.news_articles),
            source_counts={
                "tweets": len(b.data_result.tweets),
                "reddit_posts": len(b.data_result.reddit_posts),
                "news_articles": len(b.data_result.news_articles),
            },
        )

    async def get_live_snapshot_async(self, ticker: str) -> LiveUpdateResult:
        """
        Fast pipeline for the streaming dashboard.
        (Data + combined NLP + prediction; skips full knowledge graph for speed.)
        """
        b = await self._run_live_pipeline_async(ticker)
        return self._bundle_to_live_update(b)

    async def get_advisor_data_layer_async(self, ticker: str) -> AdvisorDataLayer:
        """
        Same pipeline as live snapshot, structured for the reasoning advisor (signals + headlines).
        """
        b = await self._run_live_pipeline_async(ticker)
        return self._bundle_to_advisor_data_layer(b)
    
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

