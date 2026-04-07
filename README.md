# Multi-Modal Stock Market Analysis Framework

An advanced, agentic AI-driven system for comprehensive stock market analysis that combines price data, social media sentiment, and news analysis to provide multi-modal insights and predictions.

## 🚀 Features

- **Agentic Architecture**: Specialized AI agents working together
- **Multi-Modal Analysis**: Price data, social sentiment, and news analysis
- **ML-Powered Predictions**: Price forecasting with sentiment integration
- **Knowledge Graphs**: Neo4j-based entity relationship mapping
- **Real-Time Data**: Live data from multiple sources (Twitter, Reddit, News APIs)
- **NLP Insights**: Advanced sentiment analysis and entity extraction
- **Streaming API**: FastAPI + WebSockets for multi-ticker live updates and advisor chat
- **Evaluation**: Reproducible backtests and pipeline timing (`run_evaluation.py`)

## Architecture overview

The codebase is organized in layers:

| Layer | Role |
|--------|------|
| **CLI / UI** | `main.py` (batch analysis), `ui_app.py` (Streamlit), `run_evaluation.py` (metrics) |
| **Orchestration** | `agents/orchestrator_agent.py` sequences data → NLP → prediction → knowledge graph |
| **Agents** | `data_agent`, `sentiment_agent`, `emotion_agent`, `prediction_agent`, `knowledge_agent`, `advisor_agent` |
| **Domain models** | `models/data_models.py` (Pydantic `AnalysisReport`, signals, live payloads) |
| **HTTP / realtime** | `backend/server.py` (FastAPI), `services/streaming_service.py` (WebSockets, Gemini streaming) |
| **Infrastructure** | `utils/config.py` (env), `utils/response_cache.py` (Redis or memory TTL), `utils/logging.py` |

**Agent responsibilities (summary)**

1. **OrchestratorAgent** — Coordinates the full pipeline and exposes async + sync entry points.
2. **DataAgent** — Pulls prices (yfinance) and optional social/news text when API keys are set.
3. **SentimentAgent** / **EmotionAgent** — Text classification (e.g. FinBERT / emotion transformers).
4. **PredictionAgent** — Price forecast (linear or optional LSTM) using price + sentiment features.
5. **KnowledgeAgent** — Embeddings, article ranking, optional Neo4j graph writes.
6. **AdvisorAgent** — Optional Gemini-backed explanations and chat over the latest signals.

## System design

- **Data flow**: Ticker → gather OHLCV + text → parallel sentiment + emotion → prediction with multimodal features → knowledge enrichment → `AnalysisReport` (or lighter `LiveUpdateResult` for streaming).
- **Async vs sync**: Long-running work uses `asyncio` and `asyncio.to_thread` for blocking ML/I/O so the API server stays responsive. WebSocket live streaming runs **one asyncio task per ticker** with a semaphore (`LIVE_STREAM_MAX_CONCURRENT`) to cap concurrent pipelines.
- **Observability**: Shared log format (`%(asctime)s | LEVEL | logger | message`) via `utils.logging`. CLI writes also to `stock_analysis_YYYYMMDD.log`.
- **Degradation**: Missing API keys or services result in skipped sources and partial reports rather than hard failures where possible.

## How to run

### Prerequisites

- Python 3.8 or higher
- Neo4j database (optional, for knowledge graph features)
- API keys for data sources (see Configuration section)

### Quick start

1. **Clone and install**

```bash
git clone <your-repo-url>
cd multi-modal-stock-market-analysis
pip install -r requirements.txt
```

2. **spaCy model (required for some NLP paths)**

```bash
python -m spacy download en_core_web_sm
```

3. **Environment**

Copy or create a `.env` file in the project root (see `API_KEYS_SETUP.md` for keys). Example:

```bash
# copy template if you maintain one
# cp .env.example .env
```

4. **Run a full analysis (CLI)**

```bash
python main.py --ticker TSLA
python main.py --status
python main.py --ticker AAPL --save-json report.json --verbose
```

5. **Streamlit UI**

```bash
streamlit run ui_app.py
```

6. **Streaming backend (WebSocket live + chat)**

```bash
uvicorn backend.server:app --reload --host 0.0.0.0 --port 8000
```

Open `http://localhost:8000` for the static dashboard; `/health` for JSON health.

7. **Evaluation (metrics + optional plots)**

```bash
python run_evaluation.py --ticker AAPL --plots
```

## 🔧 Installation


### 1. Python Dependencies
```bash
pip install -r requirements.txt
```

### 2. spaCy Language Model
```bash
python -m spacy download en_core_web_sm
```

### 3. Neo4j Setup (Optional)
- Install Neo4j Desktop or use Neo4j Aura
- Create a new database
- Note the connection details for configuration

## ⚙️ Configuration

Create a `.env` file in the project root with your API keys:

```env
# Twitter API (for social sentiment)
TWITTER_BEARER_TOKEN=your_twitter_bearer_token_here

# Reddit API (for community sentiment)
REDDIT_CLIENT_ID=your_reddit_client_id_here
REDDIT_CLIENT_SECRET=your_reddit_client_secret_here
REDDIT_USER_AGENT=StockAnalysisBot/1.0

# News API (for news analysis)
NEWS_API_KEY=your_news_api_key_here

# Neo4j Database (for knowledge graphs)
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_neo4j_password_here
```

### API Key Sources:

- **Twitter API**: [Developer Portal](https://developer.twitter.com/)
- **Reddit API**: [Reddit App Preferences](https://www.reddit.com/prefs/apps)
- **News API**: [NewsAPI.org](https://newsapi.org/)
- **Neo4j**: [Neo4j Aura](https://neo4j.com/cloud/aura/) or local installation

## 🎯 Usage

### Basic Analysis (CLI)
```bash
python main.py --ticker AAPL
```

### Verbose Output
```bash
python main.py --ticker TSLA --verbose
```

### Save Detailed Report
```bash
python main.py --ticker GOOGL --save-json analysis_report.json
```

### Check system status
```bash
python main.py --status
```

### Web UI
```bash
streamlit run ui_app.py
```
Then open the local URL shown by Streamlit.

### Supported Tickers
The system works with any valid stock ticker, with enhanced company name mapping for popular stocks:
- **Tech**: AAPL, GOOGL, MSFT, AMZN, TSLA, META, NVDA, NFLX
- **Finance**: JPM, V, MA, PYPL
- **Consumer**: WMT, HD, DIS, PG
- **Healthcare**: JNJ, UNH
- **Energy**: XOM, CVX
- And many more...

## 📊 Output

The system provides:

1. **Executive Summary**: High-level analysis overview
2. **Sentiment Analysis**: Social media and news sentiment scores
   - Emotion signals: fear/greed/confidence/uncertainty with confidence
3. **Price Prediction**: ML-based next-day price forecast
   - Models: Linear Regression (default) or LSTM (enable via `USE_LSTM=true` in `.env`)
4. **Knowledge Insights**: Key entities and article recommendations
   - Event extraction (earnings, launch, M&A, guidance, regulatory) and Company → IMPACTED_BY → Event
5. **Detailed Metrics**: Confidence intervals, feature importance, etc.

## 🛠️ Development

### Project structure

```
multi-modal-stock-market-analysis/
├── main.py                 # CLI entry point
├── run_evaluation.py       # Metrics + pipeline timing reports
├── config.py               # Legacy re-export of utils.config
├── requirements.txt
├── models/
│   └── data_models.py      # Pydantic domain models
├── agents/
│   ├── orchestrator_agent.py
│   ├── data_agent.py
│   ├── sentiment_agent.py
│   ├── emotion_agent.py
│   ├── prediction_agent.py
│   ├── price_prediction_agent.py
│   ├── knowledge_agent.py
│   └── advisor_agent.py
├── backend/
│   └── server.py           # FastAPI app
├── services/
│   ├── streaming_service.py
│   ├── websocket_manager.py
│   └── ws_messages.py
├── evaluation/             # Backtest + reporting helpers
├── utils/
│   ├── config.py
│   ├── logging.py
│   ├── response_cache.py
│   └── system_health.py
├── frontend/               # Static assets for the streaming UI
└── ui_app.py               # Streamlit UI
```

### Extending the Framework

To add new agents or modify existing ones:

1. **New Agent**: Create in `agents/` directory, inherit from base patterns
2. **New Data Source**: Extend `DataGatheringAgent` with new API integration
3. **New ML Model**: Modify `PricePredictionAgent` (toggle `USE_LSTM`) or `SentimentAgent`
4. **New Features**: Add to `config.py` and update relevant agents

## 🔍 Troubleshooting

### Common Issues

1. **Missing API Keys**: System will skip unavailable data sources gracefully
2. **Model Loading Errors**: Check internet connection and disk space
3. **Neo4j Connection**: Ensure Neo4j is running and credentials are correct
4. **Rate Limits**: Some APIs have rate limits; the system handles these gracefully

### Debug Mode
```bash
python main.py --ticker AAPL --verbose
```

### Log Files
Check `stock_analysis_YYYYMMDD.log` for detailed error information.

## 📈 Performance

- **Analysis Time**: 2-5 minutes per stock (depending on data availability)
- **Memory Usage**: ~2-4GB (due to ML models)
- **API Calls**: Optimized to respect rate limits
- **Caching**: Embeddings and model outputs are cached when possible

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Hugging Face**: For transformer models and tokenizers
- **spaCy**: For NLP and named entity recognition
- **Neo4j**: For graph database capabilities
- **scikit-learn**: For machine learning algorithms
- **yfinance**: For stock price data

## 🔮 Future Enhancements

- [x] Web interface (Streamlit)
- [x] Real-time streaming analysis (FastAPI + WebSockets)
- [ ] Portfolio-level analysis
- [ ] Advanced ML models (LSTM, Transformer-based forecasting)
- [ ] Multi-language support
- [ ] Enhanced visualization
- [ ] Backtesting capabilities
- [ ] Risk assessment metrics

---

**Disclaimer**: This tool is for educational and research purposes only. It should not be used as the sole basis for investment decisions. Always consult with qualified financial advisors and conduct your own research before making investment decisions.

