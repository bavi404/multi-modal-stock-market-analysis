"""
Streamlit dashboard for one-off full analysis runs (synchronous orchestrator).

For real-time multi-ticker streaming, use the FastAPI app in ``backend/server.py``.
"""

from __future__ import annotations

import logging

import streamlit as st

from agents.orchestrator_agent import OrchestratorAgent
from utils.logging import configure_root_logging


def main() -> None:
    st.set_page_config(page_title="Multi-Modal Stock Analysis", layout="wide")
    st.title("Multi-Modal Stock Market Analysis")
    st.caption("Price • Sentiment • Emotions • News • Knowledge Graph")

    ticker = st.text_input("Enter stock ticker", value="AAPL").strip().upper()
    col1, col2, col3 = st.columns(3)
    with col1:
        verbose = st.checkbox("Verbose logs", value=False)
    with col2:
        use_lstm = st.checkbox("Use LSTM predictor (config)", value=False)
    with col3:
        run_btn = st.button("Run Analysis", type="primary")

    if run_btn and ticker:
        configure_root_logging(verbose=verbose, level=logging.DEBUG if verbose else logging.INFO)
        st.info(f"Running analysis for {ticker}. This may take a few minutes on first run (model downloads).")
        orchestrator = OrchestratorAgent()
        # Switch LSTM at runtime if requested
        try:
            if hasattr(orchestrator, 'prediction_agent') and hasattr(orchestrator.prediction_agent, 'use_lstm'):
                orchestrator.prediction_agent.use_lstm = use_lstm
        except Exception:
            pass
        with st.spinner("Analyzing..."):
            report = orchestrator.run_analysis(ticker)
        orchestrator.close()

        # Summary metrics
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("Sentiment Score", f"{report.sentiment_analysis.sentiment_score:.3f}")
        with m2:
            st.metric("Emotion", report.emotion_analysis.dominant_emotion.title())
        with m3:
            st.metric("Predicted Price", f"${report.price_prediction.predicted_price:.2f}")
        with m4:
            st.metric("Articles", len(report.knowledge_insights.recommended_articles))

        # Sentiment & Emotion
        st.subheader("Sentiment & Emotions")
        st.write(report.sentiment_analysis.summary)
        if report.emotion_analysis.emotion_scores:
            st.bar_chart(report.emotion_analysis.emotion_scores)

        # Prediction
        st.subheader("Price Prediction")
        ci = report.price_prediction.confidence_interval
        st.write(f"Predicted: ${report.price_prediction.predicted_price:.2f}  |  CI: ${ci['lower']:.2f} - ${ci['upper']:.2f}")
        if report.price_prediction.features_used:
            st.caption("Features used: " + ", ".join(report.price_prediction.features_used))

        # Articles
        st.subheader("Top Articles")
        for art in report.knowledge_insights.recommended_articles[:5]:
            st.markdown(f"**{art.get('title','No title')}**  ")
            if art.get('source'):
                st.caption(art['source'])
            if art.get('url'):
                st.write(art['url'])
            st.write(art.get('description', ''))
            st.divider()

        # Executive summary
        st.subheader("Executive Summary")
        st.code(report.executive_summary)


if __name__ == "__main__":
    main()


