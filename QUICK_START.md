# 🚀 Quick Start Guide - Fully Functioning Demo

## Step 1: Get API Keys (20 minutes)

Follow the detailed guide in `API_KEYS_SETUP.md` to get:
- ✅ Twitter Bearer Token
- ✅ Reddit Client ID & Secret  
- ✅ News API Key
- ✅ (Optional) Neo4j credentials

**Quick Links:**
- Twitter: https://developer.twitter.com/
- Reddit: https://www.reddit.com/prefs/apps
- News API: https://newsapi.org/
- Neo4j: https://neo4j.com/download/ (or https://neo4j.com/cloud/aura/ for cloud)

## Step 2: Configure .env File

1. Open `.env` file in the project root
2. Replace all `your_xxx_here` placeholders with your actual API keys
3. **Make sure `USE_LSTM=true`** (already set for you!)

Your `.env` should look like:
```env
TWITTER_BEARER_TOKEN=abc123xyz...
REDDIT_CLIENT_ID=your_actual_id
REDDIT_CLIENT_SECRET=your_actual_secret
REDDIT_USER_AGENT=StockAnalysisBot/1.0
NEWS_API_KEY=your_actual_key
USE_LSTM=true
```

## Step 3: Verify Setup

```bash
py main.py --status
```

Expected output:
```
AGENT STATUS CHECK
--------------------------------------------------
  Data Agent: Ready
  Sentiment Agent: Ready
  Prediction Agent: Ready
  Knowledge Agent: Ready
  Emotion Agent: Ready
  Neo4J Connection: Not connected (or Connected if Neo4j is set up)
```

## Step 4: Run the Demo UI 🎨

```bash
streamlit run ui_app.py
```

This will:
- Open your browser automatically
- Show a beautiful UI with ticker input
- Run full multimodal analysis when you click "Run Analysis"
- Display: Sentiment, Emotions, Price Prediction (LSTM), Articles, Knowledge Graph

## Step 4B: Run True Streaming Demo (Gemini + Live Panel)

1. Add Gemini settings to `.env`:
```env
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-2.0-flash
LIVE_STREAM_TICKERS=AAPL,MSFT,NVDA,TSLA
LIVE_STREAM_INTERVAL_SECONDS=15
```

2. Start streaming backend:
```bash
uvicorn backend.server:app --host 0.0.0.0 --port 8000 --reload
```

3. Open:
```text
http://localhost:8000
```

You will get:
- Live updates panel (websocket pushed updates for multiple tickers)
- Gemini token-streaming robo-advisor chat

## Step 5: Test Full Analysis

### Option A: Streamlit UI (Recommended for Demo)
1. Enter a ticker (e.g., `AAPL`, `TSLA`, `MSFT`)
2. Check "Use LSTM predictor" ✅
3. Click "Run Analysis"
4. Wait 2-5 minutes (first run downloads models)
5. See beautiful results!

### Option B: Command Line
```bash
py main.py --ticker AAPL --verbose
```

## What You'll See

### In the UI:
- **Metrics Dashboard**: Sentiment score, dominant emotion, predicted price, article count
- **Sentiment & Emotions**: Text summary + emotion bar chart
- **Price Prediction**: LSTM-based forecast with confidence intervals
- **Top Articles**: Recommended news articles with URLs
- **Executive Summary**: Complete analysis narrative

### In CLI:
- Full formatted report with all analysis components
- Data collection summary (tweets, Reddit posts, news articles)
- Sentiment scores and emotion signals
- Price prediction with LSTM model
- Knowledge graph statistics

## 🎯 Demo Script (5 minutes)

**Perfect for presentations:**

1. **Open Streamlit UI** (`streamlit run ui_app.py`)
2. **Show the interface** - explain multimodal inputs
3. **Enter a popular ticker** (AAPL, TSLA, or MSFT)
4. **Enable LSTM** - check the checkbox
5. **Run Analysis** - show the spinner and explain what's happening
6. **Show Results**:
   - "Here's the sentiment from social media..."
   - "The LSTM model predicts..."
   - "These are the most relevant articles..."
   - "The knowledge graph extracted these entities..."

## 🐛 Troubleshooting

### "No tweets/articles found"
- Check API keys are correct in `.env`
- Verify Twitter/Reddit/News API accounts are active
- Check rate limits haven't been exceeded

### "LSTM not working"
- Verify `USE_LSTM=true` in `.env`
- Restart the application after changing `.env`
- Check logs for LSTM initialization messages

### "Models taking too long"
- First run downloads models (~500MB)
- Subsequent runs are much faster
- Models are cached locally

### "Neo4j connection failed"
- This is optional! System works without it
- Knowledge graph features will be limited
- Check Neo4j Desktop is running (if using local)

## 📊 Expected Performance

- **First Run**: 3-5 minutes (model downloads)
- **Subsequent Runs**: 1-2 minutes
- **Memory Usage**: 2-4 GB (ML models)
- **API Calls**: ~50-100 per analysis

## ✅ Success Checklist

- [ ] All API keys added to `.env`
- [ ] `USE_LSTM=true` in `.env`
- [ ] Status check passes (`py main.py --status`)
- [ ] Streamlit UI opens successfully
- [ ] Analysis completes without errors
- [ ] Results show tweets/articles (not just zeros)
- [ ] LSTM prediction appears in results

## 🎉 You're Ready!

Once all checks pass, you have a **fully functioning multimodal stock analysis system** ready to demonstrate!

**Key Features Demonstrated:**
- ✅ Real-time data from multiple sources
- ✅ Advanced NLP (FinBERT sentiment + emotion detection)
- ✅ LSTM-based price prediction
- ✅ Knowledge graph with entity extraction
- ✅ Beautiful UI for presentations
- ✅ Production-ready architecture

---

**Need help?** Check `API_KEYS_SETUP.md` for detailed API key instructions.
