# 🔍 AI Agent Bug Localization - Streamlit Interface

A modern, interactive web interface for the AI Agent Bug Localization system built with Streamlit. This interface provides an intuitive way to manage projects, run processing pipelines, and analyze bug localization results.

## 🚀 Quick Start

### Prerequisites
- Python 3.7+
- Project source codes and bug reports in `AgentProjectData/` directory

### Launch the Interface

```bash
# Make the script executable (Linux/Mac)
chmod +x run_streamlit.sh

# Launch the Streamlit application
./run_streamlit.sh
```

**Or manually:**

```bash
# Activate virtual environment
source venv/bin/activate

# Install requirements
pip install -r requirements.txt

# Run Streamlit
streamlit run streamlit_app.py
```

The interface will be available at: **http://localhost:8501**

## 📋 Features

### 🏠 Dashboard Tab
- **Project Overview**: Statistics for selected projects
- **Bug Reports Count**: Total number of bug reports per project
- **Source Files Count**: Number of source code files analyzed
- **Processing Status**: Current state of each project
- **Comparative Charts**: Visual comparison between projects

### 🚀 Pipeline Execution Tab
- **Individual Pipeline Runs**:
  - Basic Query Generation (NLP-based)
  - KeyBERT Query Generation
  - Reasoning & Reflection Queries
- **Complete Pipeline**: Run all processing steps
- **Real-time Status**: Monitor pipeline execution
- **Configuration Options**:
  - BM25/FAISS weights adjustment
  - Top N documents selection
  - KeyBERT keywords count

### 📝 Search Queries Tab
- **Six Query Types Display**:
  - Baseline – Basic
  - Extended – Basic
  - Baseline – KeyBERT
  - Extended – KeyBERT
  - Baseline – Reason + Reflect
  - Extended – Reason + Reflect
- **Bug Report Breakdown**: Queries organized by bug ID
- **Query Content Viewer**: Full text of generated search queries

### 🔍 Search Results Tab
- **Localization Results**: Top-ranked source files for each query
- **Ranked Lists**: Files ordered by relevance score
- **Score Information**: BM25+FAISS combined scores
- **Multiple Query Comparison**: Side-by-side result analysis

### 📈 Evaluation Tab
- **Performance Metrics**:
  - Mean Reciprocal Rank (MRR)
  - Mean Average Precision (MAP)
  - Hit@K (K=1,5,10) percentages
- **Improvement Analysis**:
  - Query improvements count
  - Same results count
  - Degraded results count
- **Interactive Charts**: Plotly-based visualizations
- **Cross-Project Comparison**: Performance across multiple projects

## 🛠️ Configuration Options

### Sidebar Controls

**Project Selection**:
- Multi-select dropdown for available projects
- Projects 3, 13, 14, 20, 24 (aspnetboilerplate, Atlas, ARKStatsExtractor, CodenameOne, mobile-wallet)

**Query Type Filtering**:
- Select specific query types to display/analyze
- Filter results by baseline vs extended queries

**Processing Parameters**:
- **BM25 Weight**: Balance between BM25 and FAISS scoring (0.0-1.0)
- **Top N Documents**: Number of top-ranked files to retrieve (10-500)
- **Top N Keywords**: KeyBERT keyword extraction count (5-50)

## 📊 Understanding the Results

### Query Types Explained

1. **Baseline – Basic**: Standard NLP processing of bug report content
2. **Extended – Basic**: NLP processing including image/attachment content
3. **Baseline – KeyBERT**: Keyword extraction using KeyBERT on basic content
4. **Extended – KeyBERT**: KeyBERT on extended content with attachments
5. **Baseline – Reason + Reflect**: AI reasoning on basic bug reports
6. **Extended – Reason + Reflect**: AI reasoning on extended content

### Evaluation Metrics

- **MRR (Mean Reciprocal Rank)**: Average of 1/rank for first relevant result
- **MAP (Mean Average Precision)**: Precision across all relevant documents
- **Hit@K**: Percentage of queries with relevant result in top K
- **Improvement Count**: Queries where extended outperformed baseline

## 🗂️ File Structure

```
AI-agent-BugLocalization/
├── streamlit_app.py              # Main Streamlit application
├── run_streamlit.sh             # Launch script
├── requirements.txt             # Python dependencies
├── STREAMLIT_README.md          # This documentation
├── src/                         # Source code modules
│   ├── NLP.py                  # Basic query processing
│   ├── KEYBERT.py              # KeyBERT query generation
│   ├── reason.py               # Reasoning pipeline
│   ├── evaluate.py             # Evaluation metrics
│   └── agents.py               # AI agent implementations
└── AgentProjectData/           # Data directory
    ├── SourceCodes/            # Project source code
    ├── ProjectBugReports/      # Bug report data
    ├── ConstructedQueries/     # Generated search queries
    ├── SearchResults/          # Localization results
    └── EvaluationResults/      # Performance metrics
```

## 🔧 Troubleshooting

### Common Issues

**"No projects found"**:
- Ensure `AgentProjectData/SourceCodes/` and `AgentProjectData/ProjectBugReports/` exist
- Check that project directories follow naming convention (Project3, Project13, etc.)

**Pipeline fails to run**:
- Verify all dependencies are installed: `pip install -r requirements.txt`
- Check that source code agents are properly configured
- Review log files in `logs/parallel_logs/` for detailed error messages

**Missing evaluation results**:
- Run the complete pipeline first
- Ensure ground truth data exists for evaluation
- Check `AgentProjectData/EvaluationResults/` directory

### Performance Tips

- **Large Projects**: Adjust "Top N Documents" for faster processing
- **Memory Usage**: Process one project at a time for memory-constrained systems
- **Network**: Ensure stable connection for AI agent API calls

## 🎯 Usage Workflow

1. **Setup**: Ensure data is in `AgentProjectData/` directory
2. **Select Projects**: Choose projects from sidebar
3. **Run Pipeline**: Execute desired query generation methods
4. **View Queries**: Examine generated search queries
5. **Check Results**: Review bug localization results
6. **Analyze Performance**: Study evaluation metrics and comparisons

## 🔮 Advanced Features

### Custom Analysis
- Filter by specific bug IDs
- Compare query effectiveness across projects
- Export results for external analysis

### Batch Processing
- Run multiple projects simultaneously
- Schedule pipeline execution
- Monitor progress with real-time logs

### Integration
- REST API endpoints for external tools
- Database connectivity for result storage
- Custom metric implementations

---

**Need Help?** Check the main project README.md or review the source code in the `src/` directory for implementation details. 