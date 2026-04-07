"""
Application configuration (API keys, TTLs, feature flags).
Loaded from project-root `.env` regardless of current working directory.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env")

# API Configuration
TWITTER_BEARER_TOKEN = os.getenv("TWITTER_BEARER_TOKEN")
STOCKTWITS_ACCESS_TOKEN = os.getenv("STOCKTWITS_ACCESS_TOKEN")
REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID")
REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET")
REDDIT_USER_AGENT = os.getenv("REDDIT_USER_AGENT", "StockAnalysisBot/1.0")
# Reddit: "praw" (live API), "dataset" (local CSV, e.g. Kaggle WSB posts), or "auto" (dataset if file exists, else PRAW)
REDDIT_SOURCE = os.getenv("REDDIT_SOURCE", "auto").strip().lower()
# Path to CSV from https://www.kaggle.com/datasets/gpreda/reddit-wallstreetsbets-posts (e.g. reddit_wsb.csv)
REDDIT_DATASET_PATH = os.getenv("REDDIT_DATASET_PATH", "").strip() or None
# 0 = read entire file (may be large); set e.g. 200000 to cap rows on load
REDDIT_DATASET_MAX_ROWS = int(os.getenv("REDDIT_DATASET_MAX_ROWS", "0"))
NEWS_API_KEY = os.getenv("NEWS_API_KEY")

# Neo4j Configuration
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

# Model Configuration
SENTIMENT_MODEL = "ProsusAI/finbert"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
SPACY_MODEL = "en_core_web_sm"
EMOTION_MODEL = os.getenv("EMOTION_MODEL", "j-hartmann/emotion-english-distilroberta-base")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

# Data Configuration
DEFAULT_PERIOD = "1y"
MAX_TWEETS = 100
MAX_REDDIT_POSTS = 50
MAX_NEWS_ARTICLES = 20

# Analysis Configuration
PREDICTION_DAYS = 5
TOP_ARTICLES = 5
USE_LSTM = os.getenv("USE_LSTM", "false").lower() == "true"

# Streaming backend configuration
LIVE_STREAM_TICKERS = [
    t.strip().upper()
    for t in os.getenv("LIVE_STREAM_TICKERS", "AAPL,MSFT,NVDA,TSLA").split(",")
    if t.strip()
]
LIVE_STREAM_INTERVAL_SECONDS = int(os.getenv("LIVE_STREAM_INTERVAL_SECONDS", "15"))
# Cap concurrent live pipeline runs (shared across tickers) to avoid API/model overload
LIVE_STREAM_MAX_CONCURRENT = max(1, int(os.getenv("LIVE_STREAM_MAX_CONCURRENT", "6")))
# Optional extra tickers joined at runtime via WebSocket subscribe (hard cap)
LIVE_STREAM_MAX_TICKERS = max(1, int(os.getenv("LIVE_STREAM_MAX_TICKERS", "32")))

# WebSocket production tuning
WS_HEARTBEAT_INTERVAL_SECONDS = int(os.getenv("WS_HEARTBEAT_INTERVAL_SECONDS", "25"))
WS_CHAT_RATE_LIMIT_PER_MINUTE = int(os.getenv("WS_CHAT_RATE_LIMIT_PER_MINUTE", "30"))
WS_LIVE_INBOUND_RATE_LIMIT_PER_MINUTE = int(os.getenv("WS_LIVE_INBOUND_RATE_LIMIT_PER_MINUTE", "120"))

# Advisor (Gemini)
ADVISOR_PROVIDER = os.getenv("ADVISOR_PROVIDER", "gemini").strip().lower()
ADVISOR_GEMINI_API_KEY = os.getenv("ADVISOR_GEMINI_API_KEY") or GEMINI_API_KEY or os.getenv("GOOGLE_API_KEY")
ADVISOR_GEMINI_MODEL = os.getenv("ADVISOR_GEMINI_MODEL", GEMINI_MODEL)
ADVISOR_HISTORY_MAX_TURNS = int(os.getenv("ADVISOR_HISTORY_MAX_TURNS", "5"))
ADVISOR_ASSISTANT_MEMORY_MAX_CHARS = int(os.getenv("ADVISOR_ASSISTANT_MEMORY_MAX_CHARS", "4000"))
ADVISOR_MAX_NEWS_HEADLINES = int(os.getenv("ADVISOR_MAX_NEWS_HEADLINES", "8"))

# Caching (optional Redis)
REDIS_URL = os.getenv("REDIS_URL", "").strip() or None
CACHE_TTL_STOCK_SECONDS = int(os.getenv("CACHE_TTL_STOCK_SECONDS", "300"))
CACHE_TTL_TWITTER_SECONDS = int(os.getenv("CACHE_TTL_TWITTER_SECONDS", "120"))
CACHE_TTL_STOCKTWITS_SECONDS = int(os.getenv("CACHE_TTL_STOCKTWITS_SECONDS", str(CACHE_TTL_TWITTER_SECONDS)))
CACHE_TTL_REDDIT_SECONDS = int(os.getenv("CACHE_TTL_REDDIT_SECONDS", "180"))
CACHE_TTL_NEWS_SECONDS = int(os.getenv("CACHE_TTL_NEWS_SECONDS", "600"))

# NLP fallbacks when no text / degraded pipeline
FALLBACK_SENTIMENT_SCORE = float(os.getenv("FALLBACK_SENTIMENT_SCORE", "0.0"))
FALLBACK_EMOTION_LABEL = os.getenv("FALLBACK_EMOTION_LABEL", "neutral")
