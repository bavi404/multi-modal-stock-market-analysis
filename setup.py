"""
Setup script for Multi-Modal Stock Market Analysis Framework
"""
import subprocess
import sys
import os
from pathlib import Path


def run_command(command, description):
    """Run a command and handle errors"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e.stderr}")
        return False


def main():
    """Main setup function"""
    print("🚀 Setting up Multi-Modal Stock Market Analysis Framework")
    print("=" * 60)
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        sys.exit(1)
    
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")
    
    # Install Python dependencies
    if not run_command("pip install -r requirements.txt", "Installing Python dependencies"):
        print("⚠️  Some dependencies may have failed to install. Please check the output above.")
    
    # Install spaCy model
    if not run_command("python -m spacy download en_core_web_sm", "Installing spaCy English model"):
        print("⚠️  spaCy model installation failed. You may need to install it manually.")
    
    # Create .env file if it doesn't exist
    env_file = Path(".env")
    env_example = Path(".env.example")
    
    if not env_file.exists():
        if env_example.exists():
            try:
                env_content = env_example.read_text()
                env_file.write_text(env_content)
                print("✅ Created .env file from template")
                print("📝 Please edit .env file with your API keys")
            except Exception as e:
                print(f"⚠️  Could not create .env file: {e}")
        else:
            # Create basic .env file
            env_content = """# Twitter API
TWITTER_BEARER_TOKEN=your_twitter_bearer_token_here

# Reddit API
REDDIT_CLIENT_ID=your_reddit_client_id_here
REDDIT_CLIENT_SECRET=your_reddit_client_secret_here
REDDIT_USER_AGENT=StockAnalysisBot/1.0

# News API
NEWS_API_KEY=your_news_api_key_here

# Neo4j Database (optional)
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_neo4j_password_here
"""
            try:
                env_file.write_text(env_content)
                print("✅ Created .env file with template")
                print("📝 Please edit .env file with your API keys")
            except Exception as e:
                print(f"⚠️  Could not create .env file: {e}")
    else:
        print("✅ .env file already exists")
    
    # Test basic functionality
    print("\n🧪 Testing basic functionality...")
    
    try:
        # Test imports
        from agents.orchestrator_agent import OrchestratorAgent
        print("✅ Agent imports working")
        
        # Test orchestrator initialization
        orchestrator = OrchestratorAgent()
        status = orchestrator.get_analysis_status()
        print("✅ Orchestrator initialization working")
        
        print("\n📊 Agent Status:")
        for agent, status_msg in status.items():
            icon = "✅" if "Ready" in status_msg or "Connected" in status_msg else "⚠️"
            print(f"  {icon} {agent.replace('_', ' ').title()}: {status_msg}")
        
        orchestrator.close()
        
    except Exception as e:
        print(f"⚠️  Basic functionality test failed: {e}")
        print("   This may be due to missing API keys or dependencies.")
    
    print("\n" + "=" * 60)
    print("🎉 Setup completed!")
    print("\nNext steps:")
    print("1. Edit .env file with your API keys")
    print("2. (Optional) Set up Neo4j database")
    print("3. Run: python main.py --ticker AAPL")
    print("\nFor help: python main.py --help")
    print("Check status: python main.py --status")


if __name__ == "__main__":
    main()

