#!/bin/bash

# AI Agent Bug Localization - Streamlit Interface
# This script launches the Streamlit web interface for the bug localization system

echo "🔍 AI Agent Bug Localization System"
echo "=================================="
echo

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "⚠️  Virtual environment not found. Creating one..."
    python -m venv venv
    echo "✅ Virtual environment created."
fi

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source venv/bin/activate

# Install/upgrade requirements
echo "📦 Installing requirements..."
pip install -r requirements.txt

# Create necessary directories
echo "📁 Creating necessary directories..."
mkdir -p logs/parallel_logs
mkdir -p AgentProjectData/ConstructedQueries
mkdir -p AgentProjectData/SearchResults
mkdir -p AgentProjectData/EvaluationResults

# Set Python path
export PYTHONPATH="${PYTHONPATH}:./src"

echo
echo "🚀 Starting Streamlit application..."
echo "📍 Open your browser and navigate to: http://localhost:8501"
echo "🛑 Press Ctrl+C to stop the application"
echo

# Launch Streamlit
streamlit run streamlit_app.py --server.port 8501 --server.address localhost 