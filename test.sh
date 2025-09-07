#!/bin/bash

echo "🚀 Multi-Modal Stock Analysis Framework - Quick Test"
echo "==============================================="

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Find Python command
if command_exists python3; then
    PYTHON_CMD="python3"
    echo "✅ Python found: python3 command"
elif command_exists python; then
    PYTHON_CMD="python"
    echo "✅ Python found: python command"
else
    echo "❌ Python not found in PATH"
    echo "Please install Python 3.8+ or add it to PATH"
    exit 1
fi

# Check Python version
PYTHON_VERSION=$($PYTHON_CMD --version 2>&1 | grep -oP '\d+\.\d+')
MAJOR_VERSION=$(echo $PYTHON_VERSION | cut -d. -f1)
MINOR_VERSION=$(echo $PYTHON_VERSION | cut -d. -f2)

if [ "$MAJOR_VERSION" -lt 3 ] || ([ "$MAJOR_VERSION" -eq 3 ] && [ "$MINOR_VERSION" -lt 8 ]); then
    echo "❌ Python 3.8+ required, found $PYTHON_VERSION"
    exit 1
fi

echo "✅ Python version: $PYTHON_VERSION"

echo ""
echo "📦 Installing dependencies..."
$PYTHON_CMD -m pip install -r requirements.txt

echo ""
echo "📚 Installing spaCy model..."
$PYTHON_CMD -m spacy download en_core_web_sm

echo ""
echo "📋 Running component tests..."
$PYTHON_CMD test_components.py

echo ""
echo "📊 Checking system status..."
$PYTHON_CMD main.py --status

echo ""
echo "🧪 Running basic analysis test..."
echo "This may take a few minutes..."
$PYTHON_CMD main.py --ticker AAPL

echo ""
echo "✅ Testing completed!"
echo "Check the output above for any errors."

