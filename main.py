"""
Command-line entry point for the multi-modal stock analysis pipeline.

Runs the :class:`~agents.orchestrator_agent.OrchestratorAgent` synchronously and prints
or saves a structured :class:`~models.data_models.AnalysisReport`.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys

from agents.orchestrator_agent import OrchestratorAgent
from models.data_models import AnalysisReport
from utils.logging import setup_logging

logger = logging.getLogger(__name__)


def print_banner() -> None:
    """Print the CLI banner (ASCII fallback on narrow Windows consoles)."""
    try:
        banner = """
╔══════════════════════════════════════════════════════════════════╗
║            Multi-Modal Stock Market Analysis Framework           ║
║                     Agentic AI-Driven Analysis                  ║
╠══════════════════════════════════════════════════════════════════╣
║  Combining: Price Data • Social Sentiment • News Analysis       ║
║  Features: ML Prediction • Knowledge Graphs • NLP Insights      ║
╚══════════════════════════════════════════════════════════════════╝
        """
        print(banner)
    except UnicodeEncodeError:
        # Fallback for Windows terminals that don't support Unicode box drawing
        print("=" * 70)
        print("Multi-Modal Stock Market Analysis Framework")
        print("Agentic AI-Driven Analysis")
        print("=" * 70)
        print("Combining: Price Data | Social Sentiment | News Analysis")
        print("Features: ML Prediction | Knowledge Graphs | NLP Insights")
        print("=" * 70)


def format_analysis_report(report: AnalysisReport) -> str:
    """
    Render an :class:`~models.data_models.AnalysisReport` as a human-readable string.

    Parameters
    ----------
    report
        Completed analysis for one ticker.

    Returns
    -------
    str
        Multi-section text suitable for stdout.
    """
    try:
        # Header
        output = f"\n{'='*80}\n"
        output += f"STOCK ANALYSIS REPORT - {report.ticker}\n"
        output += f"Analysis Date: {report.analysis_date.strftime('%Y-%m-%d %H:%M:%S')}\n"
        output += f"{'='*80}\n\n"
        
        # Executive Summary
        output += "EXECUTIVE SUMMARY\n"
        output += "-" * 50 + "\n"
        output += report.executive_summary + "\n\n"
        
        # Data Collection Summary
        output += "DATA COLLECTION SUMMARY\n"
        output += "-" * 50 + "\n"
        output += f"• Stock Price Data Points: {len(report.stock_data.prices.get('Close', [])) if isinstance(report.stock_data.prices, dict) else 'N/A'}\n"
        output += f"• Tweets Analyzed: {len(report.stock_data.tweets)}\n"
        output += f"• Reddit Posts Analyzed: {len(report.stock_data.reddit_posts)}\n"
        output += f"• News Articles Analyzed: {len(report.stock_data.news_articles)}\n"
        if getattr(report, "data_source_status", None):
            output += "• Per-source status:\n"
            for ds in report.data_source_status:
                msg = f" ({ds.message})" if ds.message else ""
                output += f"  – {ds.source}: {ds.status}{msg}\n"
        output += "\n"
        
        # Sentiment Analysis Details
        output += "SENTIMENT ANALYSIS\n"
        output += "-" * 50 + "\n"
        output += f"• Overall Sentiment Score: {report.sentiment_analysis.sentiment_score:.3f} (range: -1.0 to +1.0)\n"
        output += f"• Dominant Emotion: {report.sentiment_analysis.dominant_emotion.title()}\n"
        output += f"• Analysis Confidence: {report.sentiment_analysis.confidence:.1%}\n"
        output += f"• Summary: {report.sentiment_analysis.summary}\n\n"

        # Emotion Analysis Details
        output += "EMOTION SIGNALS\n"
        output += "-" * 50 + "\n"
        output += f"• Dominant Market Emotion: {report.emotion_analysis.dominant_emotion.title()}\n"
        output += f"• Emotion Confidence: {report.emotion_analysis.confidence:.1%}\n"
        if report.emotion_analysis.emotion_scores:
            # Show top 3 emotions
            top_emotions = sorted(report.emotion_analysis.emotion_scores.items(), key=lambda x: x[1], reverse=True)[:3]
            top_str = ", ".join([f"{k.title()}: {v:.2f}" for k, v in top_emotions])
            output += f"• Top Signals: {top_str}\n"
        output += f"• Summary: {report.emotion_analysis.summary}\n\n"
        
        # Price Prediction Details
        output += "PRICE PREDICTION\n"
        output += "-" * 50 + "\n"
        output += f"• Predicted Price: ${report.price_prediction.predicted_price:.2f}\n"
        output += f"• Confidence Interval: ${report.price_prediction.confidence_interval['lower']:.2f} - ${report.price_prediction.confidence_interval['upper']:.2f}\n"
        output += f"• Prediction Date: {report.price_prediction.prediction_date.strftime('%Y-%m-%d')}\n"
        output += f"• Model Confidence: {report.price_prediction.model_confidence:.1%}\n"
        output += f"• Features Used: {', '.join(report.price_prediction.features_used[:5])}{'...' if len(report.price_prediction.features_used) > 5 else ''}\n"
        exp = report.price_prediction.explainability
        if exp:
            output += f"• Explainability ({exp.attribution_method}):\n"
            for d in exp.drivers[:8]:
                output += f"    – {d.factor} (impact: {d.impact})\n"
            output += f"• Sentiment role: {exp.sentiment_contribution}\n"
            if exp.recent_events:
                output += "• Recent headline context: " + "; ".join(exp.recent_events[:5]) + "\n"
        output += "\n"
        
        # Knowledge Insights
        output += "KNOWLEDGE INSIGHTS\n"
        output += "-" * 50 + "\n"
        output += f"• Recommended Articles: {len(report.knowledge_insights.recommended_articles)}\n"
        output += f"• Entities Extracted: {len(report.knowledge_insights.entities_extracted)}\n"
        output += f"• Graph Relationships: {len(report.knowledge_insights.relationships_created)}\n"
        output += f"• Summary: {report.knowledge_insights.graph_summary}\n\n"
        
        # Top Recommended Articles
        if report.knowledge_insights.recommended_articles:
            output += "TOP RECOMMENDED ARTICLES\n"
            output += "-" * 50 + "\n"
            for i, article in enumerate(report.knowledge_insights.recommended_articles[:3], 1):
                output += f"{i}. {article.get('title', 'No Title')}\n"
                if article.get('source'):
                    output += f"   Source: {article['source']}\n"
                if article.get('url'):
                    output += f"   URL: {article['url']}\n"
                output += "\n"
        
        # Key Entities
        if report.knowledge_insights.entities_extracted:
            output += "KEY ENTITIES IDENTIFIED\n"
            output += "-" * 50 + "\n"
            
            # Group entities by type
            entities_by_type = {}
            for entity in report.knowledge_insights.entities_extracted[:10]:  # Show top 10
                entity_type = entity.get('label', 'Unknown')
                if entity_type not in entities_by_type:
                    entities_by_type[entity_type] = []
                entities_by_type[entity_type].append(entity.get('text', ''))
            
            for entity_type, entities in entities_by_type.items():
                output += f"• {entity_type}: {', '.join(entities[:5])}\n"
            output += "\n"
        
        output += "="*80 + "\n"
        output += "Analysis completed successfully!\n"
        output += "="*80 + "\n"
        
        return output
        
    except Exception as exc:
        return f"Error formatting report: {exc!s}\n\nRaw report data available in logs."


def save_report_json(report: AnalysisReport, filename: str) -> None:
    """
    Serialize key fields of ``report`` to JSON for downstream tooling.

    Parameters
    ----------
    report
        Analysis output to summarize.
    filename
        Destination path (created or overwritten).
    """
    try:
        # Convert report to dictionary for JSON serialization
        report_dict = {
            'ticker': report.ticker,
            'analysis_date': report.analysis_date.isoformat(),
            'executive_summary': report.executive_summary,
            'data_source_status': (
                [ds.model_dump(mode='json') for ds in report.data_source_status]
                if getattr(report, 'data_source_status', None)
                else []
            ),
            'sentiment_analysis': {
                'sentiment_score': report.sentiment_analysis.sentiment_score,
                'dominant_emotion': report.sentiment_analysis.dominant_emotion,
                'confidence': report.sentiment_analysis.confidence,
                'summary': report.sentiment_analysis.summary
            },
            'price_prediction': {
                'predicted_price': report.price_prediction.predicted_price,
                'confidence_interval': report.price_prediction.confidence_interval,
                'prediction_date': report.price_prediction.prediction_date.isoformat(),
                'model_confidence': report.price_prediction.model_confidence,
                'features_used': report.price_prediction.features_used,
                'explainability': (
                    report.price_prediction.explainability.model_dump(mode='json')
                    if report.price_prediction.explainability
                    else None
                ),
            },
            'knowledge_insights': {
                'recommended_articles_count': len(report.knowledge_insights.recommended_articles),
                'entities_count': len(report.knowledge_insights.entities_extracted),
                'relationships_count': len(report.knowledge_insights.relationships_created),
                'graph_summary': report.knowledge_insights.graph_summary,
                'top_articles': report.knowledge_insights.recommended_articles[:5]
            },
            'data_summary': {
                'tweets_count': len(report.stock_data.tweets),
                'reddit_posts_count': len(report.stock_data.reddit_posts),
                'news_articles_count': len(report.stock_data.news_articles)
            }
        }
        
        with open(filename, 'w') as f:
            json.dump(report_dict, f, indent=2, default=str)
        
        print(f"Detailed report saved to: {filename}")
        
    except Exception as exc:
        logger.error("Error saving JSON report: %s", exc)


def main() -> int:
    """Parse CLI arguments, run analysis, and return a process exit code."""
    parser = argparse.ArgumentParser(
        description="Multi-Modal Stock Market Analysis Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --ticker TSLA
  python main.py --ticker AAPL --verbose
  python main.py --ticker GOOGL --save-json analysis_googl.json
  python main.py --ticker MSFT --verbose --save-json msft_analysis.json

Supported tickers include: AAPL, GOOGL, MSFT, AMZN, TSLA, META, NVDA, etc.
        """
    )
    
    parser.add_argument(
        '--ticker', 
        type=str, 
        required=False,
        help='Stock ticker symbol to analyze (e.g., TSLA, AAPL, GOOGL)'
    )
    
    parser.add_argument(
        '--verbose', 
        action='store_true',
        help='Enable verbose logging output'
    )
    
    parser.add_argument(
        '--save-json',
        type=str,
        metavar='FILENAME',
        help='Save detailed analysis report as JSON file'
    )
    
    parser.add_argument(
        '--status',
        action='store_true',
        help='Check the status of all agents and exit'
    )
    
    args = parser.parse_args()
    
    # Set up logging
    setup_logging(args.verbose)
    
    # Print banner
    print_banner()
    
    # Initialize orchestrator
    logger.info("Initializing Multi-Modal Stock Analysis Framework...")
    
    try:
        orchestrator = OrchestratorAgent()
        
        # Check status if requested
        if args.status:
            print("\nAGENT STATUS CHECK")
            print("-" * 50)
            status = orchestrator.get_analysis_status()
            for agent, status_msg in status.items():
                # Use simple bullet for Windows compatibility
                print(f"  {agent.replace('_', ' ').title()}: {status_msg}")
            print("\nNote: Some agents may show 'Model not loaded' until first use.")
            return 0
        
        # Validate ticker is provided when not checking status
        if not args.ticker:
            print("Error: --ticker is required when not using --status")
            parser.print_help()
            return 1
        
        # Validate ticker
        ticker = args.ticker.upper().strip()
        if not ticker or len(ticker) > 10:
            print("Error: Please provide a valid stock ticker symbol (e.g., TSLA)")
            return 1
        
        print(f"\nStarting analysis for {ticker}...")
        print("This may take a few minutes depending on data availability and model loading...")
        
        # Run analysis
        report = orchestrator.run_analysis(ticker)
        
        # Display results
        formatted_report = format_analysis_report(report)
        print(formatted_report)
        
        # Save JSON report if requested
        if args.save_json:
            save_report_json(report, args.save_json)
        
        # Clean up
        orchestrator.close()
        
        print(f"\nAnalysis for {ticker} completed successfully!")
        return 0
        
    except KeyboardInterrupt:
        print("\n\nAnalysis interrupted by user.")
        return 1
        
    except Exception as exc:
        logger.error("Fatal error during analysis: %s", exc)
        print(f"\nError: Analysis failed - {exc!s}")
        print("Check the log file for detailed error information.")
        return 1


if __name__ == "__main__":
    sys.exit(main())

