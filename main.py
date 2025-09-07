"""
Main entry point for the Multi-Modal Stock Market Analysis Framework
"""
import argparse
import logging
import sys
from datetime import datetime
import json

from agents.orchestrator_agent import OrchestratorAgent


def setup_logging(verbose: bool = False):
    """
    Set up logging configuration
    
    Args:
        verbose: Enable verbose logging
    """
    log_level = logging.DEBUG if verbose else logging.INFO
    
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(f'stock_analysis_{datetime.now().strftime("%Y%m%d")}.log')
        ]
    )


def print_banner():
    """Print application banner"""
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


def format_analysis_report(report) -> str:
    """
    Format the analysis report for console display
    
    Args:
        report: AnalysisReport object
        
    Returns:
        Formatted string for display
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
        output += f"• News Articles Analyzed: {len(report.stock_data.news_articles)}\n\n"
        
        # Sentiment Analysis Details
        output += "SENTIMENT ANALYSIS\n"
        output += "-" * 50 + "\n"
        output += f"• Overall Sentiment Score: {report.sentiment_analysis.sentiment_score:.3f} (range: -1.0 to +1.0)\n"
        output += f"• Dominant Emotion: {report.sentiment_analysis.dominant_emotion.title()}\n"
        output += f"• Analysis Confidence: {report.sentiment_analysis.confidence:.1%}\n"
        output += f"• Summary: {report.sentiment_analysis.summary}\n\n"
        
        # Price Prediction Details
        output += "PRICE PREDICTION\n"
        output += "-" * 50 + "\n"
        output += f"• Predicted Price: ${report.price_prediction.predicted_price:.2f}\n"
        output += f"• Confidence Interval: ${report.price_prediction.confidence_interval['lower']:.2f} - ${report.price_prediction.confidence_interval['upper']:.2f}\n"
        output += f"• Prediction Date: {report.price_prediction.prediction_date.strftime('%Y-%m-%d')}\n"
        output += f"• Model Confidence: {report.price_prediction.model_confidence:.1%}\n"
        output += f"• Features Used: {', '.join(report.price_prediction.features_used[:5])}{'...' if len(report.price_prediction.features_used) > 5 else ''}\n\n"
        
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
        
    except Exception as e:
        return f"Error formatting report: {str(e)}\n\nRaw report data available in logs."


def save_report_json(report, filename: str):
    """
    Save the analysis report as JSON
    
    Args:
        report: AnalysisReport object
        filename: Output filename
    """
    try:
        # Convert report to dictionary for JSON serialization
        report_dict = {
            'ticker': report.ticker,
            'analysis_date': report.analysis_date.isoformat(),
            'executive_summary': report.executive_summary,
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
                'features_used': report.price_prediction.features_used
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
        
    except Exception as e:
        logging.error(f"Error saving JSON report: {e}")


def main():
    """Main function"""
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
        required=True,
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
    logger = logging.getLogger(__name__)
    
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
                print(f"• {agent.replace('_', ' ').title()}: {status_msg}")
            print("\nNote: Some agents may show 'Model not loaded' until first use.")
            return 0
        
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
        
    except Exception as e:
        logger.error(f"Fatal error during analysis: {e}")
        print(f"\nError: Analysis failed - {str(e)}")
        print("Check the log file for detailed error information.")
        return 1


if __name__ == "__main__":
    sys.exit(main())

