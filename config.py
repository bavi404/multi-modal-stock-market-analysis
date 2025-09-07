"""
Configuration file for API keys and settings
"""
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# API Configuration
TWITTER_BEARER_TOKEN = os.getenv('TWITTER_BEARER_TOKEN')
REDDIT_CLIENT_ID = os.getenv('REDDIT_CLIENT_ID')
REDDIT_CLIENT_SECRET = os.getenv('REDDIT_CLIENT_SECRET')
REDDIT_USER_AGENT = os.getenv('REDDIT_USER_AGENT', 'StockAnalysisBot/1.0')
NEWS_API_KEY = os.getenv('NEWS_API_KEY')

# Neo4j Configuration
NEO4J_URI = os.getenv('NEO4J_URI', 'bolt://localhost:7687')
NEO4J_USER = os.getenv('NEO4J_USER', 'neo4j')
NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD')

# Model Configuration
SENTIMENT_MODEL = 'ProsusAI/finbert'
EMBEDDING_MODEL = 'all-MiniLM-L6-v2'
SPACY_MODEL = 'en_core_web_sm'

# Data Configuration
DEFAULT_PERIOD = '1y'  # Default period for stock data
MAX_TWEETS = 100
MAX_REDDIT_POSTS = 50
MAX_NEWS_ARTICLES = 20

# Analysis Configuration
PREDICTION_DAYS = 5  # Number of days to use for price prediction
TOP_ARTICLES = 5     # Number of top articles to recommend

