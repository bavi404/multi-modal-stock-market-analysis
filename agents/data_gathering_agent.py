"""
Data Gathering Agent for collecting stock prices, social media data, and news
"""
import yfinance as yf
import tweepy
import praw
from newsapi import NewsApiClient
import pandas as pd
from typing import List, Dict, Optional
import logging
from datetime import datetime, timedelta
import config


class DataGatheringAgent:
    """Agent responsible for gathering all raw data from various sources"""
    
    def __init__(self):
        """Initialize the data gathering agent with API clients"""
        self.logger = logging.getLogger(__name__)
        
        # Initialize Twitter client
        self.twitter_client = None
        if config.TWITTER_BEARER_TOKEN:
            try:
                self.twitter_client = tweepy.Client(bearer_token=config.TWITTER_BEARER_TOKEN)
            except Exception as e:
                self.logger.warning(f"Failed to initialize Twitter client: {e}")
        
        # Initialize Reddit client
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
        
        # Initialize News API client
        self.news_client = None
        if config.NEWS_API_KEY:
            try:
                self.news_client = NewsApiClient(api_key=config.NEWS_API_KEY)
            except Exception as e:
                self.logger.warning(f"Failed to initialize News API client: {e}")
    
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
        Fetch recent tweets mentioning the stock
        
        Args:
            query: Search query (e.g., ticker symbol)
            max_results: Maximum number of tweets to fetch
            
        Returns:
            List of tweet texts
        """
        if max_results is None:
            max_results = config.MAX_TWEETS
            
        tweets = []
        
        if not self.twitter_client:
            self.logger.warning("Twitter client not initialized - skipping tweets")
            return tweets
            
        try:
            # Search for recent tweets
            search_query = f"{query} -is:retweet lang:en"
            
            response = self.twitter_client.search_recent_tweets(
                query=search_query,
                max_results=min(max_results, 100),  # API limit
                tweet_fields=['created_at', 'public_metrics', 'context_annotations']
            )
            
            if response.data:
                tweets = [tweet.text for tweet in response.data]
                self.logger.info(f"Retrieved {len(tweets)} tweets for query: {query}")
            else:
                self.logger.info(f"No tweets found for query: {query}")
                
        except Exception as e:
            self.logger.error(f"Error fetching tweets for {query}: {e}")
            
        return tweets
    
    def get_reddit_posts(self, query: str, max_results: int = None) -> List[str]:
        """
        Fetch recent Reddit posts mentioning the stock
        
        Args:
            query: Search query (e.g., ticker symbol)  
            max_results: Maximum number of posts to fetch
            
        Returns:
            List of Reddit post texts (title + body)
        """
        if max_results is None:
            max_results = config.MAX_REDDIT_POSTS
            
        posts = []
        
        if not self.reddit_client:
            self.logger.warning("Reddit client not initialized - skipping Reddit posts")
            return posts
            
        try:
            # Search across relevant subreddits
            subreddits = ['stocks', 'investing', 'SecurityAnalysis', 'StockMarket', 'wallstreetbets']
            
            for subreddit_name in subreddits:
                subreddit = self.reddit_client.subreddit(subreddit_name)
                
                # Search for posts containing the query
                for submission in subreddit.search(query, limit=max_results//len(subreddits)):
                    post_text = f"{submission.title}. {submission.selftext}"
                    posts.append(post_text.strip())
                    
                    if len(posts) >= max_results:
                        break
                        
                if len(posts) >= max_results:
                    break
            
            self.logger.info(f"Retrieved {len(posts)} Reddit posts for query: {query}")
            
        except Exception as e:
            self.logger.error(f"Error fetching Reddit posts for {query}: {e}")
            
        return posts
    
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
                    articles.append({
                        'title': article.get('title', ''),
                        'description': article.get('description', ''),
                        'content': article.get('content', ''),
                        'url': article.get('url', ''),
                        'published_at': article.get('publishedAt', ''),
                        'source': article.get('source', {}).get('name', '')
                    })
                
                self.logger.info(f"Retrieved {len(articles)} news articles for query: {query}")
            else:
                self.logger.info(f"No news articles found for query: {query}")
                
        except Exception as e:
            self.logger.error(f"Error fetching news articles for {query}: {e}")
            
        return articles
    
    def gather_all_data(self, ticker: str, company_name: str = None) -> Dict:
        """
        Gather all data for a given stock ticker
        
        Args:
            ticker: Stock ticker symbol
            company_name: Full company name for news search
            
        Returns:
            Dictionary containing all gathered data
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

