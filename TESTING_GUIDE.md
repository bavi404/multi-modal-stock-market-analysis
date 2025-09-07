"""
Component testing script for the Multi-Modal Stock Analysis Framework
"""
import logging
import traceback
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_imports():
    """Test if all required modules can be imported"""
    print("🔍 Testing imports...")
    
    try:
        import yfinance as yf
        print("✅ yfinance imported successfully")
    except ImportError as e:
        print(f"❌ yfinance import failed: {e}")
    
    try:
        from transformers import pipeline
        print("✅ transformers imported successfully")
    except ImportError as e:
        print(f"❌ transformers import failed: {e}")
    
    try:
        import pandas as pd
        print("✅ pandas imported successfully")
    except ImportError as e:
        print(f"❌ pandas import failed: {e}")
    
    try:
        from agents.orchestrator_agent import OrchestratorAgent
        print("✅ Custom agents imported successfully")
    except ImportError as e:
        print(f"❌ Custom agents import failed: {e}")
        print("Make sure you're running from the project root directory")

def test_data_gathering():
    """Test data gathering without API keys"""
    print("\n📊 Testing data gathering...")
    
    try:
        from agents.data_gathering_agent import DataGatheringAgent
        
        agent = DataGatheringAgent()
        
        # Test stock data (should work without API keys)
        print("Testing stock price data...")
        stock_data = agent.get_stock_prices("AAPL", "1mo")
        
        if not stock_data.empty:
            print(f"✅ Stock data retrieved: {len(stock_data)} days")
            print(f"   Latest close: ${stock_data['Close'].iloc[-1]:.2f}")
        else:
            print("❌ No stock data retrieved")
        
        # Test social media (will show warnings without API keys)
        print("Testing social media APIs (may show warnings)...")
        tweets = agent.get_tweets("AAPL", 5)
        reddit_posts = agent.get_reddit_posts("AAPL", 5)
        news = agent.get_news("Apple", 5)
        
        print(f"   Tweets: {len(tweets)}")
        print(f"   Reddit posts: {len(reddit_posts)}")
        print(f"   News articles: {len(news)}")
        
    except Exception as e:
        print(f"❌ Data gathering test failed: {e}")
        traceback.print_exc()

def test_sentiment_analysis():
    """Test sentiment analysis with sample data"""
    print("\n💭 Testing sentiment analysis...")
    
    try:
        from agents.sentiment_agent import SentimentAgent
        
        agent = SentimentAgent()
        
        # Test with sample texts
        sample_texts = [
            "Apple reported strong quarterly earnings beating expectations",
            "Tesla stock is overvalued and due for a correction",
            "The market outlook remains neutral with mixed signals"
        ]
        
        print("Analyzing sample texts...")
        result = agent.analyze(sample_texts)
        
        print(f"✅ Sentiment analysis completed")
        print(f"   Overall sentiment: {result.sentiment_score:.3f}")
        print(f"   Dominant emotion: {result.dominant_emotion}")
        print(f"   Confidence: {result.confidence:.3f}")
        
    except Exception as e:
        print(f"❌ Sentiment analysis test failed: {e}")
        print("This might be due to model download issues or memory constraints")

def test_price_prediction():
    """Test price prediction with sample data"""
    print("\n📈 Testing price prediction...")
    
    try:
        from agents.price_prediction_agent import PricePredictionAgent
        from agents.data_gathering_agent import DataGatheringAgent
        
        # Get some real stock data
        data_agent = DataGatheringAgent()
        stock_data = data_agent.get_stock_prices("AAPL", "3mo")
        
        if stock_data.empty:
            print("❌ No stock data available for prediction test")
            return
        
        pred_agent = PricePredictionAgent()
        
        # Test prediction with neutral sentiment
        result = pred_agent.predict(stock_data, 0.0)
        
        print(f"✅ Price prediction completed")
        print(f"   Predicted price: ${result.predicted_price:.2f}")
        print(f"   Confidence interval: ${result.confidence_interval['lower']:.2f} - ${result.confidence_interval['upper']:.2f}")
        print(f"   Model confidence: {result.model_confidence:.3f}")
        
    except Exception as e:
        print(f"❌ Price prediction test failed: {e}")
        traceback.print_exc()

def test_knowledge_agent():
    """Test knowledge agent with sample articles"""
    print("\n🧠 Testing knowledge agent...")
    
    try:
        from agents.knowledge_agent import KnowledgeAgent
        
        agent = KnowledgeAgent()
        
        # Sample articles
        sample_articles = [
            {
                'title': 'Apple Reports Strong Q4 Earnings',
                'description': 'Apple Inc. exceeded expectations with strong iPhone sales',
                'content': 'Apple CEO Tim Cook announced record revenue',
                'source': 'Tech News'
            },
            {
                'title': 'Tesla Cybertruck Production Begins',
                'description': 'Elon Musk announces start of Cybertruck deliveries',
                'content': 'Tesla begins production of the highly anticipated Cybertruck',
                'source': 'Auto News'
            }
        ]
        
        print("Analyzing sample articles...")
        result = agent.analyze(sample_articles, "AAPL")
        
        print(f"✅ Knowledge analysis completed")
        print(f"   Recommended articles: {len(result.recommended_articles)}")
        print(f"   Entities extracted: {len(result.entities_extracted)}")
        print(f"   Summary: {result.graph_summary}")
        
        # Show some entities
        if result.entities_extracted:
            print("   Sample entities:")
            for entity in result.entities_extracted[:3]:
                print(f"     - {entity.get('text', 'N/A')} ({entity.get('label', 'N/A')})")
        
        agent.close()
        
    except Exception as e:
        print(f"❌ Knowledge agent test failed: {e}")
        print("This might be due to model download issues")

def test_full_orchestrator():
    """Test the complete orchestrator workflow"""
    print("\n🎭 Testing full orchestrator...")
    
    try:
        from agents.orchestrator_agent import OrchestratorAgent
        
        orchestrator = OrchestratorAgent()
        
        # Check status
        status = orchestrator.get_analysis_status()
        print("Agent status:")
        for agent, status_msg in status.items():
            print(f"   {agent}: {status_msg}")
        
        print("\nRunning quick analysis (this may take a few minutes)...")
        
        # Run analysis on a stable stock
        report = orchestrator.run_analysis("MSFT")
        
        print(f"✅ Full analysis completed!")
        print(f"   Ticker: {report.ticker}")
        print(f"   Sentiment score: {report.sentiment_analysis.sentiment_score:.3f}")
        print(f"   Predicted price: ${report.price_prediction.predicted_price:.2f}")
        print(f"   Articles analyzed: {len(report.knowledge_insights.recommended_articles)}")
        
        orchestrator.close()
        
    except Exception as e:
        print(f"❌ Full orchestrator test failed: {e}")
        traceback.print_exc()

def main():
    """Run all component tests"""
    print("🚀 Multi-Modal Stock Analysis Framework - Component Testing")
    print("=" * 70)
    print(f"Test started at: {datetime.now()}")
    
    # Run all tests
    test_imports()
    test_data_gathering()
    test_sentiment_analysis()
    test_price_prediction()
    test_knowledge_agent()
    test_full_orchestrator()
    
    print("\n" + "=" * 70)
    print("🏁 Testing completed!")
    print("\nIf any tests failed:")
    print("1. Check that all dependencies are installed: pip install -r requirements.txt")
    print("2. Install spaCy model: python -m spacy download en_core_web_sm")
    print("3. Add API keys to .env file for full functionality")
    print("4. Ensure you have sufficient memory for ML models")

if __name__ == "__main__":
    main()

