"""
Data Gathering Agent for collecting stock prices, social media data, and news
"""
import re
from pathlib import Path

import yfinance as yf
import praw
import requests
from newsapi import NewsApiClient
import pandas as pd
from typing import List, Dict, Optional, Tuple
import logging
from datetime import datetime, timedelta
from utils import config

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


class DataGatheringAgent:
    """Agent responsible for gathering all raw data from various sources"""
    
    def __init__(self):
        """Initialize the data gathering agent with API clients"""
        self.logger = logging.getLogger(__name__)
        self.last_stocktwits_error: Optional[str] = None
        
        # StockTwits doesn't require a client object for basic symbol streams.
        # Keep a simple flag for DataAgent source-status decisions.
        self.stocktwits_enabled = True
        
        # Initialize Reddit client (optional when using Kaggle/local CSV dataset)
        self.reddit_client = None
        if config.REDDIT_CLIENT_ID and config.REDDIT_CLIENT_SECRET:
            try:
                self.reddit_client = praw.Reddit(
                    client_id=config.REDDIT_CLIENT_ID,
                    client_secret=config.REDDIT_CLIENT_SECRET,
                    user_agent=config.REDDIT_USER_AGENT
                )
            except Exception as e:
                self.logger.warning(f"Failed to initialize Reddit client: {e}")

        self.last_reddit_error: Optional[str] = None
        self._reddit_df_cache: Optional[pd.DataFrame] = None
        self._reddit_cache_key: Optional[str] = None
        
        # Initialize News API client
        self.news_client = None
        if config.NEWS_API_KEY:
            try:
                self.news_client = NewsApiClient(api_key=config.NEWS_API_KEY)
            except Exception as e:
                self.logger.warning(f"Failed to initialize News API client: {e}")

    def _resolve_reddit_dataset_path(self) -> Optional[Path]:
        raw = getattr(config, "REDDIT_DATASET_PATH", None) or ""
        raw = raw.strip()
        if not raw:
            return None
        p = Path(raw)
        if not p.is_absolute():
            p = _PROJECT_ROOT / p
        return p if p.is_file() else None

    def reddit_data_available(self) -> bool:
        """True if configured Reddit source (dataset file or PRAW) can run."""
        mode = getattr(config, "REDDIT_SOURCE", "auto") or "auto"
        if mode == "dataset":
            return self._resolve_reddit_dataset_path() is not None
        if mode == "praw":
            return self.reddit_client is not None
        if self._resolve_reddit_dataset_path() is not None:
            return True
        return self.reddit_client is not None

    @staticmethod
    def _reddit_title_body_columns(df: pd.DataFrame) -> Tuple[Optional[str], Optional[str]]:
        lower = {c.lower(): c for c in df.columns}
        title_col = None
        for key in ("title", "post_title"):
            if key in lower:
                title_col = lower[key]
                break
        body_col = None
        for key in ("selftext", "body", "self_text", "content", "text"):
            if key in lower:
                body_col = lower[key]
                break
        return title_col, body_col

    @staticmethod
    def _reddit_score_column(df: pd.DataFrame) -> Optional[str]:
        for c in df.columns:
            cl = c.lower()
            if cl in ("score", "upvotes", "ups"):
                return c
        return None

    def _load_reddit_dataset_dataframe(self) -> Optional[pd.DataFrame]:
        path = self._resolve_reddit_dataset_path()
        if not path:
            return None
        key = str(path.resolve())
        if self._reddit_df_cache is not None and self._reddit_cache_key == key:
            return self._reddit_df_cache

        max_rows = int(getattr(config, "REDDIT_DATASET_MAX_ROWS", 0) or 0)
        read_kw = {"low_memory": False}
        if max_rows > 0:
            read_kw["nrows"] = max_rows

        df = pd.read_csv(path, **read_kw)
        self._reddit_df_cache = df
        self._reddit_cache_key = key
        self.logger.info(
            "Loaded Reddit dataset %s (%d rows%s)",
            path.name,
            len(df),
            f", capped at {max_rows}" if max_rows > 0 else "",
        )
        return df

    def _get_reddit_posts_from_dataset(self, query: str, max_results: int) -> List[str]:
        df = self._load_reddit_dataset_dataframe()
        if df is None or df.empty:
            return []

        title_col, body_col = self._reddit_title_body_columns(df)
        if not title_col and not body_col:
            self.logger.error("Reddit dataset: could not find title/body columns in CSV")
            return []

        t = df[title_col].fillna("").astype(str) if title_col else pd.Series([""] * len(df))
        b = df[body_col].fillna("").astype(str) if body_col else pd.Series([""] * len(df))
        combined = (t + " " + b).str.upper()
        needle = (query or "").strip().upper()
        if not needle:
            return []
        mask = combined.str.contains(re.escape(needle), case=False, na=False, regex=True)
        sub = df.loc[mask]
        score_col = self._reddit_score_column(sub)
        if score_col and score_col in sub.columns:
            try:
                sub = sub.sort_values(by=score_col, ascending=False)
            except Exception:
                pass

        posts: List[str] = []
        for _, row in sub.head(max_results).iterrows():
            title = str(row[title_col]).strip() if title_col else ""
            body = str(row[body_col]).strip() if body_col else ""
            text = f"{title}. {body}".strip(" .")
            if text:
                posts.append(text)
        return posts

    def _get_reddit_posts_praw(self, query: str, max_results: int) -> List[str]:
        posts: List[str] = []
        if not self.reddit_client:
            return posts
        try:
            subreddits = ["stocks", "investing", "SecurityAnalysis", "StockMarket", "wallstreetbets"]
            for subreddit_name in subreddits:
                subreddit = self.reddit_client.subreddit(subreddit_name)
                for submission in subreddit.search(query, limit=max_results // len(subreddits)):
                    post_text = f"{submission.title}. {submission.selftext}"
                    posts.append(post_text.strip())
                    if len(posts) >= max_results:
                        break
                if len(posts) >= max_results:
                    break
            self.logger.info(f"Retrieved {len(posts)} Reddit posts (PRAW) for query: {query}")
        except Exception as e:
            self.logger.error(f"Error fetching Reddit posts (PRAW) for {query}: {e}")
        return posts

    def get_stock_prices(self, ticker: str, period: str = None) -> pd.DataFrame:
        """
        Fetch historical stock price data using yfinance
        
        Args:
            ticker: Stock ticker symbol (e.g., 'TSLA')
            period: Time period for data (default from config)
            
        Returns:
            pandas DataFrame with stock price data
        """
        if period is None:
            period = config.DEFAULT_PERIOD
            
        try:
            stock = yf.Ticker(ticker)
            data = stock.history(period=period)
            
            if data.empty:
                self.logger.error(f"No stock data found for ticker: {ticker}")
                return pd.DataFrame()
                
            self.logger.info(f"Retrieved {len(data)} days of stock data for {ticker}")
            return data
            
        except Exception as e:
            self.logger.error(f"Error fetching stock data for {ticker}: {e}")
            return pd.DataFrame()
    
    def get_tweets(self, query: str, max_results: int = None) -> List[str]:
        """
        Fetch recent StockTwits messages for a ticker.
        
        Args:
            query: Search query (ticker symbol)
            max_results: Maximum number of messages to fetch
            
        Returns:
            List of message texts
        """
        if max_results is None:
            max_results = config.MAX_TWEETS
            
        tweets: List[str] = []
        symbol = (query or "").strip().upper()
        if not symbol:
            return tweets

        self.last_stocktwits_error = None
        try:
            url = f"https://api.stocktwits.com/api/2/streams/symbol/{symbol}.json"
            headers = {
                "Accept": "application/json",
                "User-Agent": "multi-modal-stock-market-analysis/1.0",
            }
            params = {}
            token = getattr(config, "STOCKTWITS_ACCESS_TOKEN", None)
            if token:
                params["access_token"] = token

            response = requests.get(url, headers=headers, params=params, timeout=(5, 15))
            response.raise_for_status()
            payload = response.json() or {}
            messages = payload.get("messages") or []

            for msg in messages[:max_results]:
                body = str((msg or {}).get("body") or "").strip()
                if body:
                    tweets.append(body)

            if tweets:
                self.logger.info(f"Retrieved {len(tweets)} StockTwits messages for symbol: {symbol}")
            else:
                self.logger.info(f"No StockTwits messages found for symbol: {symbol}")
                
        except Exception as e:
            self.last_stocktwits_error = str(e)
            self.logger.error(f"Error fetching StockTwits messages for {symbol}: {e}")
            
        return tweets
    
    def get_reddit_posts(self, query: str, max_results: int = None) -> List[str]:
        """
        Reddit posts: live PRAW search or local CSV (e.g. Kaggle WSB posts dataset).

        See REDDIT_SOURCE, REDDIT_DATASET_PATH in config.
        """
        if max_results is None:
            max_results = config.MAX_REDDIT_POSTS

        self.last_reddit_error = None
        mode = (getattr(config, "REDDIT_SOURCE", "auto") or "auto").lower()
        dataset_path = self._resolve_reddit_dataset_path()

        use_dataset = False
        if mode == "dataset":
            use_dataset = True
        elif mode == "praw":
            use_dataset = False
        else:
            use_dataset = dataset_path is not None

        try:
            if use_dataset:
                if not dataset_path:
                    self.logger.warning(
                        "REDDIT_SOURCE=dataset but REDDIT_DATASET_PATH is missing or not a file"
                    )
                    return []
                posts = self._get_reddit_posts_from_dataset(query, max_results)
                self.logger.info(
                    "Retrieved %d Reddit posts (dataset) for query: %s",
                    len(posts),
                    query,
                )
                return posts

            if not self.reddit_client:
                self.logger.warning("Reddit PRAW not configured — skipping live Reddit posts")
                return []
            return self._get_reddit_posts_praw(query, max_results)
        except Exception as e:
            self.last_reddit_error = str(e)
            self.logger.error(f"Error fetching Reddit posts for {query}: {e}")
            return []
    
    def get_news(self, query: str, max_results: int = None) -> List[Dict[str, str]]:
        """
        Fetch recent news articles mentioning the stock
        
        Args:
            query: Search query (e.g., company name)
            max_results: Maximum number of articles to fetch
            
        Returns:
            List of dictionaries containing article data
        """
        if max_results is None:
            max_results = config.MAX_NEWS_ARTICLES
            
        articles = []
        
        if not self.news_client:
            self.logger.warning("News API client not initialized - skipping news articles")
            return articles
            
        try:
            # Get articles from the past week
            from_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
            
            response = self.news_client.get_everything(
                q=query,
                from_param=from_date,
                sort_by='relevancy',
                language='en',
                page_size=min(max_results, 100)  # API limit
            )
            
            if response['articles']:
                for article in response['articles'][:max_results]:
                    src = article.get('source') or {}
                    articles.append({
                        'title': article.get('title') or '',
                        'description': article.get('description') or '',
                        'content': article.get('content') or '',
                        'url': article.get('url') or '',
                        'published_at': article.get('publishedAt') or '',
                        'source': (src.get('name') or '') if isinstance(src, dict) else '',
                    })
                
                self.logger.info(f"Retrieved {len(articles)} news articles for query: {query}")
            else:
                self.logger.info(f"No news articles found for query: {query}")
                
        except Exception as e:
            self.logger.error(f"Error fetching news articles for {query}: {e}")
            
        return articles
    
    def gather_all_data(self, ticker: str, company_name: str = None) -> Dict:
        """
        Gather all data for a given stock ticker (synchronous, sequential I/O).

        .. note::
            Prefer :meth:`agents.data_agent.DataAgent.gather_all_data` for application code:
            it runs source fetches concurrently, applies TTL caching, and returns a
            :class:`~models.data_models.DataResult`. This method remains for scripts and backward
            compatibility; it does not share the same cache or status metadata.
        """
        self.logger.info(f"Starting data gathering for {ticker}")
        
        # If no company name provided, use ticker for all searches
        if company_name is None:
            company_name = ticker
        
        # Gather all data in parallel-like fashion
        stock_data = self.get_stock_prices(ticker)
        tweets = self.get_tweets(ticker)
        reddit_posts = self.get_reddit_posts(ticker)
        news_articles = self.get_news(company_name)
        
        result = {
            'ticker': ticker,
            'stock_prices': stock_data,
            'tweets': tweets,
            'reddit_posts': reddit_posts,
            'news_articles': news_articles,
            'data_gathered_at': datetime.now()
        }
        
        self.logger.info(f"Data gathering completed for {ticker}")
        return result

