"""
Example usage of the Multi-Modal Stock Market Analysis Framework
"""
import logging
from agents.orchestrator_agent import OrchestratorAgent


def simple_analysis_example():
    """Simple example of running an analysis"""
    print("🚀 Multi-Modal Stock Analysis Example")
    print("=" * 50)
    
    # Set up basic logging
    logging.basicConfig(level=logging.INFO)
    
    # Initialize the orchestrator
    print("Initializing analysis framework...")
    orchestrator = OrchestratorAgent()
    
    # Check agent status
    print("\nAgent Status:")
    status = orchestrator.get_analysis_status()
    for agent, status_msg in status.items():
        print(f"  • {agent.replace('_', ' ').title()}: {status_msg}")
    
    # Run analysis for a popular stock
    ticker = "AAPL"  # Apple Inc.
    print(f"\nRunning analysis for {ticker}...")
    print("This may take a few minutes...")
    
    try:
        # Perform the analysis
        report = orchestrator.run_analysis(ticker)
        
        # Display key results
        print(f"\n✅ Analysis completed for {ticker}!")
        print("\nKey Results:")
        print(f"  • Sentiment Score: {report.sentiment_analysis.sentiment_score:.3f}")
        print(f"  • Dominant Emotion: {report.sentiment_analysis.dominant_emotion}")
        print(f"  • Market Emotion Signal: {report.emotion_analysis.dominant_emotion} ({report.emotion_analysis.confidence:.2f} conf)")
        print(f"  • Predicted Price: ${report.price_prediction.predicted_price:.2f}")
        print(f"  • Articles Analyzed: {len(report.knowledge_insights.recommended_articles)}")
        print(f"  • Entities Extracted: {len(report.knowledge_insights.entities_extracted)}")
        
        print(f"\nExecutive Summary:")
        print("-" * 30)
        print(report.executive_summary)
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        print("This might be due to:")
        print("  • Missing API keys in .env file")
        print("  • Network connectivity issues")
        print("  • Rate limiting from APIs")
        
    finally:
        # Clean up
        orchestrator.close()


def batch_analysis_example():
    """Example of analyzing multiple stocks"""
    print("🔄 Batch Analysis Example")
    print("=" * 50)
    
    tickers = ["AAPL", "GOOGL", "TSLA"]
    
    # Set up basic logging
    logging.basicConfig(level=logging.WARNING)  # Reduce noise for batch processing
    
    orchestrator = OrchestratorAgent()
    
    results = {}
    
    for ticker in tickers:
        print(f"\nAnalyzing {ticker}...")
        try:
            report = orchestrator.run_analysis(ticker)
            results[ticker] = {
                'sentiment_score': report.sentiment_analysis.sentiment_score,
                'predicted_price': report.price_prediction.predicted_price,
                'articles_count': len(report.knowledge_insights.recommended_articles)
            }
            print(f"✅ {ticker} analysis completed")
            
        except Exception as e:
            print(f"❌ {ticker} analysis failed: {e}")
            results[ticker] = None
    
    # Display batch results
    print("\n📊 Batch Analysis Summary")
    print("-" * 40)
    for ticker, result in results.items():
        if result:
            print(f"{ticker}:")
            print(f"  Sentiment: {result['sentiment_score']:.3f}")
            print(f"  Predicted Price: ${result['predicted_price']:.2f}")
            print(f"  Articles: {result['articles_count']}")
        else:
            print(f"{ticker}: Analysis failed")
    
    orchestrator.close()


def knowledge_graph_example():
    """Example focusing on knowledge graph features"""
    print("🧠 Knowledge Graph Example")
    print("=" * 50)
    
    logging.basicConfig(level=logging.INFO)
    
    # Just initialize the knowledge agent directly
    from agents.knowledge_agent import KnowledgeAgent
    
    knowledge_agent = KnowledgeAgent()
    
    # Example articles (you would normally get these from the data gathering agent)
    example_articles = [
        {
            'title': 'Apple Reports Strong Q4 Earnings',
            'description': 'Apple Inc. reported better than expected earnings with strong iPhone sales.',
            'content': 'Apple CEO Tim Cook announced record revenue driven by iPhone 15 sales and services growth.',
            'source': 'Tech News',
            'url': 'https://example.com/apple-earnings'
        },
        {
            'title': 'Tesla Cybertruck Production Update',
            'description': 'Elon Musk provides update on Cybertruck manufacturing timeline.',
            'content': 'Tesla is ramping up Cybertruck production at its Austin facility with new manufacturing techniques.',
            'source': 'Auto News',
            'url': 'https://example.com/tesla-cybertruck'
        }
    ]
    
    print(f"Analyzing {len(example_articles)} example articles...")
    
    try:
        # Perform knowledge analysis
        result = knowledge_agent.analyze(example_articles, "AAPL")
        
        print(f"\n✅ Knowledge analysis completed!")
        print(f"  • Recommended articles: {len(result.recommended_articles)}")
        print(f"  • Entities extracted: {len(result.entities_extracted)}")
        print(f"  • Relationships created: {len(result.relationships_created)}")
        
        # Show extracted entities
        if result.entities_extracted:
            print(f"\n📋 Extracted Entities:")
            for entity in result.entities_extracted[:5]:  # Show first 5
                print(f"  • {entity['text']} ({entity['label']})")
        
        print(f"\n📝 Summary: {result.graph_summary}")
        
    except Exception as e:
        print(f"❌ Knowledge analysis failed: {e}")
    
    finally:
        knowledge_agent.close()


def main():
    """Main function to run examples"""
    print("🎯 Multi-Modal Stock Analysis Framework Examples")
    print("=" * 60)
    
    examples = {
        '1': ('Simple Analysis Example', simple_analysis_example),
        '2': ('Batch Analysis Example', batch_analysis_example), 
        '3': ('Knowledge Graph Example', knowledge_graph_example)
    }
    
    print("\nAvailable examples:")
    for key, (name, _) in examples.items():
        print(f"  {key}. {name}")
    
    try:
        choice = input("\nSelect an example (1-3) or press Enter for simple analysis: ").strip()
        
        if not choice:
            choice = '1'  # Default to simple analysis
        
        if choice in examples:
            name, func = examples[choice]
            print(f"\nRunning: {name}")
            func()
        else:
            print("Invalid choice. Running simple analysis...")
            simple_analysis_example()
            
    except KeyboardInterrupt:
        print("\n\n👋 Example interrupted by user.")
    except Exception as e:
        print(f"\n❌ Example failed: {e}")


if __name__ == "__main__":
    main()

