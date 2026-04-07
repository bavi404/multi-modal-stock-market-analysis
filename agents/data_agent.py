from typing import Any, Callable, Dict, List, Optional, Tuple
import asyncio

import pandas as pd

from utils import config
from agents.data_gathering_agent import DataGatheringAgent
from agents.base_agent import BaseAgent
from models.data_models import DataResult, DataSourceStatus
from utils.response_cache import get_response_cache
from datetime import datetime


def _cache_key(kind: str, *parts: str) -> str:
    return f"data:v1:{kind}:" + ":".join(str(p) for p in parts)


def _sanitize_news_articles_for_model(articles: List[dict]) -> List[Dict[str, str]]:
    """NewsAPI sometimes returns null fields; DataResult expects str values only."""
    keys = ("title", "description", "content", "url", "published_at", "source")
    out: List[Dict[str, str]] = []
    for raw in articles:
        if not isinstance(raw, dict):
            continue
        out.append({k: str(raw.get(k) or "") for k in keys})
    return out


class DataAgent(BaseAgent):
    """Independent agent responsible for live data ingestion with cache + per-source status."""

    def __init__(self) -> None:
        super().__init__()
        self._collector = DataGatheringAgent()

    async def _call_with_retry(
        self,
        fn: Callable[..., Any],
        *args: Any,
        retries: int = 2,
        timeout_seconds: float = 8.0,
    ) -> Any:
        last_exc: Optional[Exception] = None
        for attempt in range(1, retries + 2):
            try:
                self.log_stage("DataAgent: calling %s (attempt %d)", getattr(fn, "__name__", str(fn)), attempt)
                return await asyncio.wait_for(
                    self.run_blocking(fn, *args),
                    timeout=timeout_seconds,
                )
            except asyncio.TimeoutError as exc:
                last_exc = exc
                self.logger.warning("DataAgent: timeout calling %s (attempt %d)", fn.__name__, attempt)
            except Exception as exc:
                last_exc = exc
                self.logger.warning("DataAgent: error calling %s (attempt %d): %s", fn.__name__, attempt, exc)
        self.logger.warning("DataAgent: all retries failed for %s — continuing with degraded data: %s", fn.__name__, last_exc)
        return None

    async def _fetch_stock(self, ticker: str) -> Tuple[Any, DataSourceStatus]:
        cache = get_response_cache()
        key = _cache_key("stock", ticker, getattr(config, "DEFAULT_PERIOD", "1y"))
        ttl = int(getattr(config, "CACHE_TTL_STOCK_SECONDS", 300))

        hit = cache.get(key)
        if hit is not None:
            return hit, DataSourceStatus(source="stock_prices", status="cached", message="cache hit")

        raw = await self._call_with_retry(self._collector.get_stock_prices, ticker)
        if raw is None:
            return None, DataSourceStatus(source="stock_prices", status="failed", message="yfinance fetch failed")

        if isinstance(raw, pd.DataFrame) and raw.empty:
            return raw, DataSourceStatus(source="stock_prices", status="empty", message="no rows returned")

        cache.set(key, raw, ttl)
        return raw, DataSourceStatus(source="stock_prices", status="ok")

    async def _fetch_tweets(self, ticker: str) -> Tuple[List[str], DataSourceStatus]:
        if not getattr(self._collector, "stocktwits_enabled", False):
            return [], DataSourceStatus(source="stocktwits", status="skipped", message="StockTwits disabled")

        cache = get_response_cache()
        key = _cache_key("tweets", ticker)
        ttl = int(getattr(config, "CACHE_TTL_STOCKTWITS_SECONDS", 120))
        hit = cache.get(key)
        if hit is not None:
            return hit, DataSourceStatus(source="stocktwits", status="cached", message="cache hit")

        raw = await self._call_with_retry(self._collector.get_tweets, ticker, None)
        if raw is None:
            return [], DataSourceStatus(source="stocktwits", status="failed", message="stocktwits fetch failed")
        if not raw and getattr(self._collector, "last_stocktwits_error", None):
            return [], DataSourceStatus(
                source="stocktwits",
                status="failed",
                message=f"stocktwits fetch failed: {self._collector.last_stocktwits_error}",
            )
        cache.set(key, raw, ttl)
        st = "empty" if not raw else "ok"
        return raw, DataSourceStatus(source="stocktwits", status=st, message=None if raw else "no messages returned")

    async def _fetch_reddit(self, ticker: str) -> Tuple[List[str], DataSourceStatus]:
        if not getattr(self._collector, "reddit_data_available", lambda: False)():
            return [], DataSourceStatus(
                source="reddit",
                status="skipped",
                message=(
                    "Reddit not configured: set REDDIT_DATASET_PATH to your Kaggle CSV "
                    "or set REDDIT_CLIENT_ID / REDDIT_CLIENT_SECRET for PRAW"
                ),
            )

        cache = get_response_cache()
        rd_path = getattr(config, "REDDIT_DATASET_PATH", None) or ""
        key = _cache_key("reddit", ticker, rd_path or "praw")
        ttl = int(getattr(config, "CACHE_TTL_REDDIT_SECONDS", 180))
        hit = cache.get(key)
        if hit is not None:
            return hit, DataSourceStatus(source="reddit", status="cached", message="cache hit")

        raw = await self._call_with_retry(self._collector.get_reddit_posts, ticker, None)
        if raw is None:
            return [], DataSourceStatus(source="reddit", status="failed", message="reddit fetch failed")
        if not raw and getattr(self._collector, "last_reddit_error", None):
            return [], DataSourceStatus(
                source="reddit",
                status="failed",
                message=f"reddit fetch failed: {self._collector.last_reddit_error}",
            )
        cache.set(key, raw, ttl)
        st = "empty" if not raw else "ok"
        return raw, DataSourceStatus(source="reddit", status=st, message=None if raw else "no posts returned")

    async def _fetch_news(self, company_name: str) -> Tuple[List[dict], DataSourceStatus]:
        if not self._collector.news_client:
            return [], DataSourceStatus(source="news", status="skipped", message="NEWS_API_KEY not set")

        cache = get_response_cache()
        key = _cache_key("news", company_name)
        ttl = int(getattr(config, "CACHE_TTL_NEWS_SECONDS", 600))
        hit = cache.get(key)
        if hit is not None:
            return hit, DataSourceStatus(source="news", status="cached", message="cache hit")

        raw = await self._call_with_retry(self._collector.get_news, company_name, None)
        if raw is None:
            return [], DataSourceStatus(source="news", status="failed", message="news fetch failed")
        cache.set(key, raw, ttl)
        st = "empty" if not raw else "ok"
        return raw, DataSourceStatus(source="news", status=st, message=None if raw else "no articles returned")

    async def gather_all_data(
        self, ticker: str, company_name: Optional[str] = None
    ) -> DataResult:
        """
        Gather stock prices + social/news sources with TTL cache and fault tolerance.
        """
        self.log_stage("gathering data for %s", ticker)
        if company_name is None:
            company_name = ticker

        stock_task = asyncio.create_task(self._fetch_stock(ticker))
        tweets_task = asyncio.create_task(self._fetch_tweets(ticker))
        reddit_task = asyncio.create_task(self._fetch_reddit(ticker))
        news_task = asyncio.create_task(self._fetch_news(company_name))

        stock_out, tweets_out, reddit_out, news_out = await asyncio.gather(
            stock_task, tweets_task, reddit_task, news_task
        )

        stock_prices, st_stock = stock_out
        tweets, st_tw = tweets_out
        reddit_posts, st_rd = reddit_out
        news_articles, st_nw = news_out

        source_details = [st_stock, st_tw, st_rd, st_nw]

        if stock_prices is None:
            stock_prices = None
        if tweets is None:
            tweets = []
        if reddit_posts is None:
            reddit_posts = []
        if news_articles is None:
            news_articles = []
        news_articles = _sanitize_news_articles_for_model(news_articles)

        return DataResult(
            ticker=ticker,
            stock_prices=stock_prices,
            tweets=tweets,
            reddit_posts=reddit_posts,
            news_articles=news_articles,
            data_gathered_at=datetime.now(),
            source_details=source_details,
        )
