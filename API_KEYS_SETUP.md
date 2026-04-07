# API Keys Setup Guide

This guide will help you get all the API keys needed for full multimodal stock analysis.

## 🔑 Required API Keys

For a **fully functioning multimodal analysis**, you need:

1. **Twitter API** - For social media sentiment (tweets)
2. **Reddit API** - For community sentiment (Reddit posts)
3. **News API** - For financial news articles
4. **Neo4j** (Optional) - For knowledge graph persistence
5. **Gemini API Key** - For robo-advisor streaming chat synthesis

---

## 1. Twitter API (X API) - Bearer Token

### Steps:

1. **Go to Twitter Developer Portal**
   - Visit: https://developer.twitter.com/
   - Sign in with your Twitter/X account

2. **Create a Developer Account** (if you don't have one)
   - Click "Sign up" or "Apply"
   - Fill out the application form
   - Wait for approval (usually instant for basic access)

3. **Create a Project and App**
   - Go to "Developer Portal" → "Projects & Apps"
   - Click "Create Project" or "Create App"
   - Fill in project details:
     - Project name: "Stock Analysis Bot"
     - Use case: "Making a bot" or "Exploring the API"
   - Create an app within the project

4. **Get Your Bearer Token**
   - In your app settings, go to "Keys and tokens"
   - Under "Bearer Token", click "Generate"
   - **Copy the token immediately** (you won't see it again!)
   - This is your `TWITTER_BEARER_TOKEN`

### Free Tier Limits:
- **Essential**: 10,000 tweets/month (free)
- **Basic**: 10,000 tweets/month ($100/month)
- For demo purposes, Essential tier is sufficient

---

## 2. Reddit API - Client ID & Secret

### Steps:

1. **Go to Reddit App Preferences**
   - Visit: https://www.reddit.com/prefs/apps
   - Sign in to Reddit

2. **Create a New Application**
   - Scroll down and click "create another app..." or "create app"
   - Fill in:
     - **Name**: StockAnalysisBot (or any name)
     - **Type**: Select "script" (for personal use)
     - **Description**: "Stock market analysis bot"
     - **Redirect URI**: `http://localhost:8080` (can be anything for script type)
   - Click "create app"

3. **Get Your Credentials**
   - You'll see a box with your app details
   - **Client ID**: The string under your app name (looks like: `abc123xyz`)
   - **Client Secret**: The "secret" field (looks like: `def456uvw_secret`)
   - **User Agent**: Use format: `StockAnalysisBot/1.0 by YourRedditUsername`

### Free Tier:
- **Unlimited requests** (with rate limiting)
- No cost for personal/script apps

---

## 3. News API - API Key

### Steps:

1. **Go to NewsAPI.org**
   - Visit: https://newsapi.org/
   - Click "Get API Key" or "Sign Up"

2. **Create an Account**
   - Sign up with email
   - Verify your email address

3. **Get Your API Key**
   - After login, go to "API" tab
   - Your API key is displayed on the dashboard
   - **Copy the API key** - this is your `NEWS_API_KEY`

### Free Tier Limits:
- **Developer**: 100 requests/day (free)
- **Business**: 250 requests/day ($449/month)
- For demo purposes, Developer tier is sufficient
- Note: Free tier only works on localhost (localhost:3000, localhost:8080, etc.)

---

## 4. Neo4j (Optional - for Knowledge Graph)

### Option A: Neo4j Desktop (Local)

1. **Download Neo4j Desktop**
   - Visit: https://neo4j.com/download/
   - Download and install Neo4j Desktop (free)

2. **Create a Database**
   - Open Neo4j Desktop
   - Create a new database
   - Set a password (remember this!)
   - Start the database

3. **Get Connection Details**
   - URI: `bolt://localhost:7687` (default)
   - User: `neo4j` (default)
   - Password: The password you set

### Option B: Neo4j Aura (Cloud - Free Tier)

1. **Go to Neo4j Aura**
   - Visit: https://neo4j.com/cloud/aura/
   - Click "Start Free"

2. **Create Free Instance**
   - Sign up with email
   - Create a free instance (Aura Free)
   - Wait for provisioning (~2 minutes)

3. **Get Connection Details**
   - Copy the connection URI (looks like: `neo4j+s://xxxx.databases.neo4j.io`)
   - Username: `neo4j` (default)
   - Password: The password you set during creation

---

## 5. Gemini API Key (for Robo-Advisor Chat)

### Steps:

1. Go to Google AI Studio
   - Visit: https://ai.google.dev/
2. Sign in and create/select a project
3. Generate an API key for Gemini
4. Copy the key into your `.env` as `GEMINI_API_KEY`

Suggested model in `.env`:
```env
GEMINI_MODEL=gemini-2.0-flash
```

---

## 📝 Setting Up Your .env File

1. **Create `.env` file** in the project root directory:
   ```
   C:\Users\sanka\OneDrive\Documents\multi-modal-stock-market-analysis\.env
   ```

2. **Add your API keys** (copy the template below):

```env
# Twitter API (X API)
TWITTER_BEARER_TOKEN=your_twitter_bearer_token_here

# Reddit API
REDDIT_CLIENT_ID=your_reddit_client_id_here
REDDIT_CLIENT_SECRET=your_reddit_client_secret_here
REDDIT_USER_AGENT=StockAnalysisBot/1.0 by YourRedditUsername

# News API
NEWS_API_KEY=your_news_api_key_here

# Neo4j Database (Optional - for knowledge graph)
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_neo4j_password_here

# Enable LSTM for Price Prediction (set to 'true' to use LSTM instead of Linear Regression)
USE_LSTM=true

# Gemini Robo-Advisor
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-2.0-flash

# Live streaming backend
LIVE_STREAM_TICKERS=AAPL,MSFT,NVDA,TSLA
LIVE_STREAM_INTERVAL_SECONDS=15
```

3. **Replace all placeholder values** with your actual keys

4. **Save the file** (make sure it's named exactly `.env` with no extension)

---

## ✅ Verify Your Setup

After adding your API keys, test the configuration:

```bash
py main.py --status
```

You should see:
- ✅ Data Agent: Ready
- ✅ Sentiment Agent: Ready
- ✅ Prediction Agent: Ready
- ✅ Knowledge Agent: Ready
- ✅ Emotion Agent: Ready
- ✅ Neo4j Connection: Connected (if Neo4j is set up)

---

## 🚨 Important Security Notes

1. **Never commit `.env` to Git**
   - The `.env` file should already be in `.gitignore`
   - Never share your API keys publicly

2. **Keep Your Keys Safe**
   - Don't share them in screenshots or documentation
   - Rotate keys if they're accidentally exposed

3. **Rate Limits**
   - Be aware of API rate limits
   - The system handles rate limits gracefully, but avoid excessive testing

---

## 🎯 Quick Reference

| API | What You Get | Free Tier | Setup Time |
|-----|--------------|-----------|------------|
| Twitter | Bearer Token | 10K tweets/month | ~5 minutes |
| Reddit | Client ID + Secret | Unlimited | ~2 minutes |
| News API | API Key | 100 requests/day | ~3 minutes |
| Neo4j | Connection URI + Password | Free local or cloud | ~10 minutes |

**Total setup time: ~20 minutes**

---

## 🆘 Troubleshooting

### Twitter API Issues
- **"Invalid Bearer Token"**: Make sure you copied the full token (it's long!)
- **Rate Limit**: Wait a few minutes between requests

### Reddit API Issues
- **"Invalid credentials"**: Check that Client ID and Secret match your app
- **User Agent**: Must include your Reddit username

### News API Issues
- **"API key missing"**: Make sure the key is in `.env` file
- **"Too many requests"**: Free tier is 100/day - wait or upgrade

### Neo4j Issues
- **"Connection refused"**: Make sure Neo4j Desktop is running
- **"Authentication failed"**: Check username and password

---

## 📚 Additional Resources

- Twitter API Docs: https://developer.twitter.com/en/docs
- Reddit API Docs: https://www.reddit.com/dev/api/
- News API Docs: https://newsapi.org/docs
- Neo4j Docs: https://neo4j.com/docs/

---

**Ready to go!** Once you have all keys in your `.env` file, you can run full multimodal analysis with:

```bash
streamlit run ui_app.py
```

or

```bash
py main.py --ticker AAPL
```
