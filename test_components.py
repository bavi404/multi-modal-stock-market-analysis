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

def test_emotion_analysis():
    """Test emotion analysis with sample data"""
    print("\n🎭 Testing emotion analysis...")
    try:
        from agents.emotion_agent import EmotionAgent
        agent = EmotionAgent()
        sample_texts = [
            "Strong iPhone demand boosts Apple's outlook; investors cheer",
            "Regulatory probe raises uncertainty for the sector",
            "Mixed signals keep market participants cautious"
        ]
        print("Analyzing sample texts for emotions...")
        result = agent.analyze(sample_texts)
        print(f"✅ Emotion analysis completed")
        print(f"   Dominant market emotion: {result.dominant_emotion}")
        print(f"   Confidence: {result.confidence:.3f}")
    except Exception as e:
        print(f"❌ Emotion analysis test failed: {e}")

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
        print(f"   Market emotion: {report.emotion_analysis.dominant_emotion}")
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
    test_emotion_analysis()
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

#### Option A: Using pip (Recommended)
```bash
# Install Python dependencies
pip install -r requirements.txt

# Install spaCy English model
python -m spacy download en_core_web_sm

# Run component tests
python test_components.py
```

#### Option B: Using conda
```bash
# Create conda environment
conda create -n stock-analysis python=3.9
conda activate stock-analysis

# Install dependencies
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# Test
python test_components.py
```

### **Step 2: Basic Functionality Test**

#### Test 1: Check System Status
```bash
python main.py --status
```
**Expected Output:**
```
Agent Status Check
• Data Agent: Ready
• Sentiment Agent: Ready (or Model not loaded)
• Prediction Agent: Ready  
• Knowledge Agent: Ready (or Model not loaded)
• Neo4j Connection: Not connected (unless configured)
```

#### Test 2: Quick Analysis (No API Keys Needed)
```bash
python main.py --ticker AAPL
```
**What This Tests:**
- Stock price data retrieval (yfinance)
- Basic sentiment analysis models
- Price prediction algorithms
- System integration

**Expected Behavior:**
- Should retrieve stock data successfully
- May show warnings about missing API keys (normal)
- Should complete analysis with basic functionality

### **Step 3: Component-by-Component Testing**

#### Test Individual Agents
```python
# Run this in Python interpreter or as script
python test_components.py
```

**This will test:**
1. **Import Test** - All required libraries
2. **Data Gathering** - yfinance integration
3. **Sentiment Analysis** - FinBERT model loading
4. **Price Prediction** - ML model functionality  
5. **Knowledge Agent** - NLP and embedding models
6. **Full Orchestrator** - Complete workflow

### **Step 4: API Integration Testing**

#### Set Up API Keys (Optional but Recommended)
1. Copy the environment template:
```bash
cp .env.example .env
```

2. Edit `.env` with your API keys:
```env
# Twitter API (free tier available)
TWITTER_BEARER_TOKEN=your_token_here

# Reddit API (free)
REDDIT_CLIENT_ID=your_id_here
REDDIT_CLIENT_SECRET=your_secret_here

# News API (free tier: 1000 requests/month)
NEWS_API_KEY=your_key_here
```

3. Test with APIs:
```bash
python main.py --ticker TSLA --verbose
```

### **Step 5: Advanced Testing**

#### Test Multiple Stocks
```bash
python example_usage.py
# Select option 2 for batch analysis
```

#### Test Knowledge Graph Features
```bash
python example_usage.py  
# Select option 3 for knowledge graph demo
```

#### Test JSON Export
```bash
python main.py --ticker GOOGL --save-json test_report.json
```

### **Step 6: Troubleshooting Common Issues**

#### Issue 1: "Python was not found"
**Solutions:**
- Install Python from python.org
- Use `py` instead of `python` on Windows
- Use full path: `C:\Python39\python.exe main.py --ticker AAPL`

#### Issue 2: Import Errors
```bash
# Install missing packages
pip install --upgrade pip
pip install -r requirements.txt

# For spaCy model issues
python -m spacy download en_core_web_sm --force
```

#### Issue 3: Memory Issues with ML Models
```bash
# Run with minimal models
python main.py --ticker AAPL 2>&1 | grep -i "memory\|error"
```

#### Issue 4: API Rate Limits
- Use different ticker symbols
- Wait between requests
- Check API key validity

### **Step 7: Performance Testing**

#### Test Analysis Speed
```bash
time python main.py --ticker MSFT
```

#### Test Memory Usage
```bash
# On Windows
python -c "import psutil; import os; print(f'Memory: {psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024:.1f} MB')"

# Run analysis and monitor
python main.py --ticker AAPL --verbose
```

### **Step 8: Integration Testing**

#### Test Web-Ready Output
```python
from agents.orchestrator_agent import OrchestratorAgent

orchestrator = OrchestratorAgent()
report = orchestrator.run_analysis("AAPL")

# Test JSON serialization
import json
json_data = {
    'ticker': report.ticker,
    'sentiment': report.sentiment_analysis.sentiment_score,
    'prediction': report.price_prediction.predicted_price
}
print(json.dumps(json_data, indent=2))
```

## 🎯 Expected Test Results

### **Minimal Configuration (No API Keys)**
- ✅ Stock price data retrieval
- ✅ Basic sentiment analysis  
- ✅ Price prediction
- ⚠️ Limited social media data
- ⚠️ Limited news data
- ✅ Knowledge graph basics

### **Full Configuration (With API Keys)**
- ✅ Complete social media sentiment
- ✅ Real-time news analysis
- ✅ Rich knowledge graphs
- ✅ Comprehensive reports
- ✅ High-quality predictions

### **Performance Benchmarks**
- **Analysis Time**: 2-5 minutes per stock
- **Memory Usage**: 2-4 GB (due to ML models)
- **API Calls**: ~50-100 per analysis
- **Accuracy**: Baseline sentiment and prediction models

## 🚨 Quick Validation Checklist

- [ ] Python 3.8+ installed
- [ ] All dependencies installed (`pip list | grep -E "(yfinance|transformers|spacy)"`)
- [ ] spaCy model downloaded (`python -c "import spacy; spacy.load('en_core_web_sm')"`)
- [ ] Basic analysis works (`python main.py --ticker AAPL`)
- [ ] Status check passes (`python main.py --status`)
- [ ] Component tests pass (`python test_components.py`)

## 🎉 Success Indicators

**You know it's working when you see:**
```
✅ Analysis completed for AAPL!

Key Results:
• Sentiment Score: 0.123 (range: -1.0 to +1.0)
• Dominant Emotion: Optimistic  
• Predicted Price: $185.50
• Articles Analyzed: 15
• Entities Extracted: 42

EXECUTIVE SUMMARY - AAPL Analysis
Current Status:
• Current Price: $183.25
• Market Sentiment: Positive (score: 0.12)
• Predicted Next-Day Price: $185.50 (increase of 1.2%)
```

## 💡 Pro Testing Tips

1. **Start Simple**: Test with `--status` first
2. **Use Verbose Mode**: Add `--verbose` for detailed logs
3. **Test Popular Stocks**: AAPL, MSFT, GOOGL have more data
4. **Check Logs**: Look at `stock_analysis_YYYYMMDD.log`
5. **Monitor Resources**: Watch memory usage during analysis
6. **Test Offline**: Core functionality works without internet

---

**Ready to test your Multi-Modal Stock Analysis Framework!** 🚀

Run `python test_components.py` to start comprehensive testing!
