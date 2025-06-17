import streamlit as st
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import asyncio
import subprocess
import sys
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Add the src directory to Python path
sys.path.append('./src')

# Page configuration
st.set_page_config(
    page_title="AI Agent Bug Localization",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #2E4057;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #048A81;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .info-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #048A81;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
    .status-running {
        color: #ff6b35;
        font-weight: bold;
    }
    .status-completed {
        color: #4CAF50;
        font-weight: bold;
    }
    .status-error {
        color: #f44336;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Project configuration
PROJECT_MAPPING = {
    "3": "aspnetboilerplate",
    "13": "Atlas", 
    "14": "ARKStatsExtractor",
    "20": "CodenameOne",
    "24": "mobile-wallet"
}

QUERY_TYPES = [
    "Baseline – Basic",
    "Extended – Basic", 
    "Baseline – KeyBERT",
    "Extended – KeyBERT",
    "Baseline – Reason + Reflect",
    "Extended – Reason + Reflect"
]

def main():
    # Main header
    st.markdown('<h1 class="main-header">🔍 AI Agent Bug Localization System</h1>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 🛠️ Configuration")
        
        # Project selection
        available_projects = get_available_projects()
        selected_projects = st.multiselect(
            "Select Project(s)",
            options=list(available_projects.keys()),
            default=list(available_projects.keys())[:2] if available_projects else [],
            format_func=lambda x: f"Project {x}: {available_projects.get(x, 'Unknown')}"
        )
        
        st.markdown("---")
        
        # Query type selection
        selected_query_types = st.multiselect(
            "Select Query Types",
            options=QUERY_TYPES,
            default=QUERY_TYPES[:2]
        )
        
        st.markdown("---")
        
        # Processing options
        st.markdown("### ⚙️ Processing Options")
        
        # BM25 and FAISS weights
        bm25_weight = st.slider("BM25 Weight", 0.0, 1.0, 0.5, 0.1)
        faiss_weight = 1.0 - bm25_weight
        st.write(f"FAISS Weight: {faiss_weight:.1f}")
        
        # Top N documents
        top_n_docs = st.number_input("Top N Documents", min_value=10, max_value=500, value=100, step=10)
        
        # KeyBERT top keywords
        top_n_keywords = st.number_input("Top N Keywords (KeyBERT)", min_value=5, max_value=50, value=10, step=5)
    
    # Main content tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Dashboard", "🚀 Run Pipeline", "📝 Search Queries", "🔍 Search Results", "📈 Evaluation"])
    
    with tab1:
        show_dashboard(selected_projects)
    
    with tab2:
        show_pipeline_runner(selected_projects, selected_query_types, bm25_weight, faiss_weight, top_n_docs, top_n_keywords)
    
    with tab3:
        show_search_queries(selected_projects, selected_query_types)
    
    with tab4:
        show_search_results(selected_projects, selected_query_types)
    
    with tab5:
        show_evaluation_results(selected_projects, selected_query_types)

def get_available_projects():
    """Get available projects from the data directory"""
    projects = {}
    source_codes_path = Path("AgentProjectData/SourceCodes")
    bug_reports_path = Path("AgentProjectData/ProjectBugReports")
    
    if source_codes_path.exists() and bug_reports_path.exists():
        for project_dir in source_codes_path.iterdir():
            if project_dir.is_dir() and project_dir.name.startswith("Project"):
                project_id = project_dir.name.replace("Project", "")
                if (bug_reports_path / project_id).exists():
                    projects[project_id] = PROJECT_MAPPING.get(project_id, f"Project {project_id}")
    
    return projects

def show_dashboard(selected_projects):
    """Show main dashboard with project overview"""
    st.markdown('<h2 class="section-header">📊 Project Dashboard</h2>', unsafe_allow_html=True)
    
    if not selected_projects:
        st.warning("Please select at least one project from the sidebar.")
        return
    
    # Project overview
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>📁 Projects</h3>
            <h2>{}</h2>
            <p>Selected Projects</p>
        </div>
        """.format(len(selected_projects)), unsafe_allow_html=True)
    
    with col2:
        total_bugs = sum([get_bug_count(project_id) for project_id in selected_projects])
        st.markdown("""
        <div class="metric-card">
            <h3>🐛 Bug Reports</h3>
            <h2>{}</h2>
            <p>Total Reports</p>
        </div>
        """.format(total_bugs), unsafe_allow_html=True)
    
    with col3:
        total_files = sum([get_source_file_count(project_id) for project_id in selected_projects])
        st.markdown("""
        <div class="metric-card">
            <h3>📄 Source Files</h3>
            <h2>{}</h2>
            <p>Total Files</p>
        </div>
        """.format(total_files), unsafe_allow_html=True)
    
    with col4:
        query_count = get_generated_query_count(selected_projects)
        st.markdown("""
        <div class="metric-card">
            <h3>🔍 Queries</h3>
            <h2>{}</h2>
            <p>Generated</p>
        </div>
        """.format(query_count), unsafe_allow_html=True)
    
    # Project details table
    st.markdown('<h3 class="section-header">Project Details</h3>', unsafe_allow_html=True)
    
    project_data = []
    for project_id in selected_projects:
        project_data.append({
            "Project ID": project_id,
            "Project Name": PROJECT_MAPPING.get(project_id, f"Project {project_id}"),
            "Bug Reports": get_bug_count(project_id),
            "Source Files": get_source_file_count(project_id),
            "Status": get_project_status(project_id)
        })
    
    df = pd.DataFrame(project_data)
    st.dataframe(df, use_container_width=True)
    
    # Visualization
    if len(selected_projects) > 1:
        st.markdown('<h3 class="section-header">📊 Project Comparison</h3>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Bug reports per project
            fig1 = px.bar(
                df, 
                x="Project Name", 
                y="Bug Reports",
                title="Bug Reports per Project",
                color="Bug Reports",
                color_continuous_scale="viridis"
            )
            st.plotly_chart(fig1, use_container_width=True)
        
        with col2:
            # Source files per project
            fig2 = px.bar(
                df, 
                x="Project Name", 
                y="Source Files",
                title="Source Files per Project",
                color="Source Files",
                color_continuous_scale="plasma"
            )
            st.plotly_chart(fig2, use_container_width=True)

def show_pipeline_runner(selected_projects, selected_query_types, bm25_weight, faiss_weight, top_n_docs, top_n_keywords):
    """Show pipeline execution interface"""
    st.markdown('<h2 class="section-header">🚀 Pipeline Execution</h2>', unsafe_allow_html=True)
    
    if not selected_projects:
        st.warning("Please select at least one project from the sidebar.")
        return
    
    # Pipeline configuration summary
    st.markdown("""
    <div class="info-box">
        <h4>🔧 Current Configuration</h4>
        <ul>
            <li><strong>Projects:</strong> {}</li>
            <li><strong>Query Types:</strong> {}</li>
            <li><strong>BM25 Weight:</strong> {:.1f}</li>
            <li><strong>FAISS Weight:</strong> {:.1f}</li>
            <li><strong>Top Documents:</strong> {}</li>
            <li><strong>Top Keywords:</strong> {}</li>
        </ul>
    </div>
    """.format(
        ", ".join([f"Project {p}" for p in selected_projects]),
        ", ".join(selected_query_types),
        bm25_weight,
        faiss_weight,
        top_n_docs,
        top_n_keywords
    ), unsafe_allow_html=True)
    
    # Pipeline execution buttons
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔄 Run Basic Queries", use_container_width=True):
            run_basic_pipeline(selected_projects)
    
    with col2:
        if st.button("🎯 Run KeyBERT Queries", use_container_width=True):
            run_keybert_pipeline(selected_projects, top_n_keywords)
    
    with col3:
        if st.button("🧠 Run Reasoning Queries", use_container_width=True):
            run_reasoning_pipeline(selected_projects)
    
    # Full pipeline execution
    st.markdown("---")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if st.button("🚀 Run Complete Pipeline", use_container_width=True, type="primary"):
            run_complete_pipeline(selected_projects)
    
    with col2:
        if st.button("🧮 Run Evaluation", use_container_width=True):
            run_evaluation(selected_projects)
    
    # Pipeline status and logs
    st.markdown('<h3 class="section-header">📋 Pipeline Status</h3>', unsafe_allow_html=True)
    
    # Check for running processes
    show_pipeline_status()
    
    # Show recent logs
    show_recent_logs()

def show_search_queries(selected_projects, selected_query_types):
    """Show generated search queries"""
    st.markdown('<h2 class="section-header">📝 Constructed Search Queries</h2>', unsafe_allow_html=True)
    
    if not selected_projects:
        st.warning("Please select at least one project from the sidebar.")
        return
    
    # Query type filter
    query_type_mapping = {
        "Baseline – Basic": "basic",
        "Extended – Basic": "basic",
        "Baseline – KeyBERT": "keyBERT", 
        "Extended – KeyBERT": "keyBERT",
        "Baseline – Reason + Reflect": "reasoning",
        "Extended – Reason + Reflect": "reasoning"
    }
    
    for project_id in selected_projects:
        st.markdown(f'<h3 class="section-header">Project {project_id}: {PROJECT_MAPPING.get(project_id, "Unknown")}</h3>', unsafe_allow_html=True)
        
        queries_data = get_project_queries(project_id, selected_query_types)
        
        if not queries_data:
            st.info(f"No queries found for Project {project_id}. Run the pipeline first.")
            continue
        
        # Display queries by bug report
        for bug_id, queries in queries_data.items():
            with st.expander(f"🐛 Bug Report {bug_id}"):
                for query_type, content in queries.items():
                    if query_type in selected_query_types:
                        st.markdown(f"**{query_type}:**")
                        st.code(content, language="text")
                        st.markdown("---")

def show_search_results(selected_projects, selected_query_types):
    """Show bug localization search results"""
    st.markdown('<h2 class="section-header">🔍 Bug Localization Search Results</h2>', unsafe_allow_html=True)
    
    if not selected_projects:
        st.warning("Please select at least one project from the sidebar.")
        return
    
    for project_id in selected_projects:
        st.markdown(f'<h3 class="section-header">Project {project_id}: {PROJECT_MAPPING.get(project_id, "Unknown")}</h3>', unsafe_allow_html=True)
        
        results_data = get_project_search_results(project_id, selected_query_types)
        
        if not results_data:
            st.info(f"No search results found for Project {project_id}. Run the pipeline first.")
            continue
        
        # Display results by bug report
        for bug_id, results in results_data.items():
            with st.expander(f"🐛 Bug Report {bug_id} - Search Results"):
                for query_type, content in results.items():
                    if query_type in [qt.replace(" – ", "_").replace(" ", "_").lower() for qt in selected_query_types]:
                        st.markdown(f"**{query_type.replace('_', ' ').title()}:**")
                        
                        # Parse search results
                        lines = content.strip().split('\n')
                        if lines:
                            result_data = []
                            for i, line in enumerate(lines[:10], 1):  # Show top 10 results
                                if ',' in line:
                                    parts = line.split(',')
                                    if len(parts) >= 2:
                                        result_data.append({
                                            "Rank": i,
                                            "File": parts[0],
                                            "Score": parts[1] if len(parts) > 1 else "N/A"
                                        })
                            
                            if result_data:
                                df_results = pd.DataFrame(result_data)
                                st.dataframe(df_results, use_container_width=True)
                            else:
                                st.code(content[:500] + "..." if len(content) > 500 else content)
                        
                        st.markdown("---")

def show_evaluation_results(selected_projects, selected_query_types):
    """Show evaluation results and metrics"""
    st.markdown('<h2 class="section-header">📈 Evaluation Results</h2>', unsafe_allow_html=True)
    
    if not selected_projects:
        st.warning("Please select at least one project from the sidebar.")
        return
    
    # Load evaluation results
    evaluation_data = load_evaluation_results(selected_projects)
    
    if not evaluation_data:
        st.info("No evaluation results found. Run the evaluation first.")
        return
    
    # Metrics overview
    st.markdown('<h3 class="section-header">📊 Performance Metrics</h3>', unsafe_allow_html=True)
    
    for project_id in selected_projects:
        project_eval = evaluation_data.get(project_id)
        if not project_eval:
            continue
        
        st.markdown(f"**Project {project_id}: {PROJECT_MAPPING.get(project_id, 'Unknown')}**")
        
        # Metrics in columns
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("MRR Baseline", f"{project_eval.get('mrr_baseline', 0):.3f}")
            st.metric("MAP Baseline", f"{project_eval.get('map_baseline', 0):.1f}%")
        
        with col2:
            st.metric("MRR Extended", f"{project_eval.get('mrr_extended', 0):.3f}")
            st.metric("MAP Extended", f"{project_eval.get('map_extended', 0):.1f}%")
        
        with col3:
            st.metric("Improvements", project_eval.get('improvement_count', 0))
            st.metric("Same Results", project_eval.get('same_count', 0))
        
        with col4:
            st.metric("Worse Results", project_eval.get('worse_count', 0))
            st.metric("Total Queries", project_eval.get('improvement_count', 0) + project_eval.get('same_count', 0) + project_eval.get('worse_count', 0))
        
        # Hit@K metrics
        hit_baseline = project_eval.get('hit_at_k_baseline_percent', {})
        hit_extended = project_eval.get('hit_at_k_extended_percent', {})
        
        if hit_baseline or hit_extended:
            st.markdown("**Hit@K Performance:**")
            hit_data = {
                'K': [1, 5, 10],
                'Baseline': [hit_baseline.get(k, 0) for k in [1, 5, 10]],
                'Extended': [hit_extended.get(k, 0) for k in [1, 5, 10]]
            }
            
            fig = go.Figure()
            fig.add_trace(go.Bar(name='Baseline', x=hit_data['K'], y=hit_data['Baseline']))
            fig.add_trace(go.Bar(name='Extended', x=hit_data['K'], y=hit_data['Extended']))
            fig.update_layout(
                title=f'Hit@K Performance - Project {project_id}',
                xaxis_title='K',
                yaxis_title='Percentage (%)',
                barmode='group'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
    
    # Comparative analysis
    if len(selected_projects) > 1:
        st.markdown('<h3 class="section-header">🔄 Comparative Analysis</h3>', unsafe_allow_html=True)
        
        # Create comparison charts
        comparison_data = []
        for project_id in selected_projects:
            project_eval = evaluation_data.get(project_id, {})
            comparison_data.append({
                'Project': f"Project {project_id}",
                'MRR Baseline': project_eval.get('mrr_baseline', 0),
                'MRR Extended': project_eval.get('mrr_extended', 0),
                'MAP Baseline': project_eval.get('map_baseline', 0),
                'MAP Extended': project_eval.get('map_extended', 0),
                'Improvements': project_eval.get('improvement_count', 0)
            })
        
        df_comparison = pd.DataFrame(comparison_data)
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig1 = px.bar(
                df_comparison,
                x='Project',
                y=['MRR Baseline', 'MRR Extended'],
                title='MRR Comparison Across Projects',
                barmode='group'
            )
            st.plotly_chart(fig1, use_container_width=True)
        
        with col2:
            fig2 = px.bar(
                df_comparison,
                x='Project',
                y=['MAP Baseline', 'MAP Extended'],
                title='MAP Comparison Across Projects',
                barmode='group'
            )
            st.plotly_chart(fig2, use_container_width=True)

# Helper functions
def get_bug_count(project_id):
    """Get number of bug reports for a project"""
    bug_path = Path(f"AgentProjectData/ProjectBugReports/{project_id}")
    if bug_path.exists():
        return len([d for d in bug_path.iterdir() if d.is_dir()])
    return 0

def get_source_file_count(project_id):
    """Get number of source files for a project"""
    source_path = Path(f"AgentProjectData/SourceCodes/Project{project_id}")
    if source_path.exists():
        count = 0
        for root, dirs, files in os.walk(source_path):
            count += len([f for f in files if f.endswith(('.java', '.py', '.cpp', '.c', '.js', '.ts', '.cs'))])
        return count
    return 0

def get_generated_query_count(selected_projects):
    """Get total number of generated queries"""
    count = 0
    for project_id in selected_projects:
        query_path = Path(f"AgentProjectData/ConstructedQueries")
        if query_path.exists():
            for subdir in query_path.iterdir():
                if subdir.is_dir():
                    project_dir = subdir / project_id
                    if project_dir.exists():
                        count += len(list(project_dir.glob("*.txt")))
    return count

def get_project_status(project_id):
    """Get processing status of a project"""
    # Check if results exist
    search_path = Path(f"AgentProjectData/SearchResults/{project_id}")
    if search_path.exists() and any(search_path.iterdir()):
        return "✅ Completed"
    
    # Check if queries exist
    query_path = Path(f"AgentProjectData/ConstructedQueries")
    for subdir in query_path.iterdir():
        if subdir.is_dir():
            project_dir = subdir / project_id
            if project_dir.exists() and any(project_dir.iterdir()):
                return "🔄 Partially Processed"
    
    return "⏳ Pending"

def get_project_queries(project_id, selected_query_types):
    """Get constructed queries for a project"""
    queries_data = {}
    
    # Map query types to directory names
    query_dirs = {
        "Baseline – Basic": "BaselineVsExtended",
        "Extended – Basic": "BaselineVsExtended", 
        "Baseline – KeyBERT": "BaselineVsKeyBERT",
        "Extended – KeyBERT": "BaselineVsKeyBERT",
        "Baseline – Reason + Reflect": "BaselineVsReason",
        "Extended – Reason + Reflect": "BaselineVsReason"
    }
    
    for query_type in selected_query_types:
        if query_type not in query_dirs:
            continue
        
        query_dir = query_dirs[query_type]
        query_path = Path(f"AgentProjectData/ConstructedQueries/{query_dir}/{project_id}")
        
        if not query_path.exists():
            continue
        
        # Find query files
        for query_file in query_path.glob("*.txt"):
            parts = query_file.stem.split('_')
            if len(parts) >= 2:
                bug_id = parts[0]
                
                if bug_id not in queries_data:
                    queries_data[bug_id] = {}
                
                try:
                    with open(query_file, 'r', encoding='utf-8') as f:
                        content = f.read().strip()
                        queries_data[bug_id][query_type] = content
                except Exception as e:
                    queries_data[bug_id][query_type] = f"Error reading file: {str(e)}"
    
    return queries_data

def get_project_search_results(project_id, selected_query_types):
    """Get search results for a project"""
    results_data = {}
    
    results_path = Path(f"AgentProjectData/SearchResults/{project_id}")
    if not results_path.exists():
        return results_data
    
    for bug_dir in results_path.iterdir():
        if not bug_dir.is_dir():
            continue
        
        bug_id = bug_dir.name
        results_data[bug_id] = {}
        
        for result_file in bug_dir.glob("*.txt"):
            try:
                with open(result_file, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    results_data[bug_id][result_file.stem] = content
            except Exception as e:
                results_data[bug_id][result_file.stem] = f"Error reading file: {str(e)}"
    
    return results_data

def load_evaluation_results(selected_projects):
    """Load evaluation results for selected projects"""
    evaluation_data = {}
    
    eval_path = Path("AgentProjectData/EvaluationResults")
    if not eval_path.exists():
        return evaluation_data
    
    for project_id in selected_projects:
        # Look for evaluation files
        for eval_file in eval_path.glob(f"*{project_id}*.json"):
            try:
                with open(eval_file, 'r', encoding='utf-8') as f:
                    evaluation_data[project_id] = json.load(f)
                break
            except Exception as e:
                st.error(f"Error loading evaluation for Project {project_id}: {str(e)}")
    
    return evaluation_data

def run_basic_pipeline(selected_projects):
    """Run basic query pipeline"""
    with st.spinner("Running basic query pipeline..."):
        try:
            result = subprocess.run([sys.executable, "src/NLP.py"], 
                                  capture_output=True, text=True, cwd=".")
            if result.returncode == 0:
                st.success("Basic query pipeline completed successfully!")
            else:
                st.error(f"Pipeline failed: {result.stderr}")
        except Exception as e:
            st.error(f"Error running pipeline: {str(e)}")

def run_keybert_pipeline(selected_projects, top_n_keywords):
    """Run KeyBERT query pipeline"""
    with st.spinner("Running KeyBERT query pipeline..."):
        try:
            result = subprocess.run([sys.executable, "src/KEYBERT.py"], 
                                  capture_output=True, text=True, cwd=".")
            if result.returncode == 0:
                st.success("KeyBERT query pipeline completed successfully!")
            else:
                st.error(f"Pipeline failed: {result.stderr}")
        except Exception as e:
            st.error(f"Error running pipeline: {str(e)}")

def run_reasoning_pipeline(selected_projects):
    """Run reasoning query pipeline"""
    with st.spinner("Running reasoning query pipeline..."):
        try:
            result = subprocess.run([sys.executable, "src/reason.py"], 
                                  capture_output=True, text=True, cwd=".")
            if result.returncode == 0:
                st.success("Reasoning query pipeline completed successfully!")
            else:
                st.error(f"Pipeline failed: {result.stderr}")
        except Exception as e:
            st.error(f"Error running pipeline: {str(e)}")

def run_complete_pipeline(selected_projects):
    """Run complete pipeline"""
    with st.spinner("Running complete pipeline..."):
        try:
            result = subprocess.run(["bash", "run.sh"], 
                                  capture_output=True, text=True, cwd=".")
            if result.returncode == 0:
                st.success("Complete pipeline finished successfully!")
                st.info("Check the logs directory for detailed output.")
            else:
                st.error(f"Pipeline failed: {result.stderr}")
        except Exception as e:
            st.error(f"Error running pipeline: {str(e)}")

def run_evaluation(selected_projects):
    """Run evaluation"""
    with st.spinner("Running evaluation..."):
        try:
            result = subprocess.run([sys.executable, "src/evaluate.py"], 
                                  capture_output=True, text=True, cwd=".")
            if result.returncode == 0:
                st.success("Evaluation completed successfully!")
            else:
                st.error(f"Evaluation failed: {result.stderr}")
        except Exception as e:
            st.error(f"Error running evaluation: {str(e)}")

def show_pipeline_status():
    """Show current pipeline status"""
    # Check for running processes
    log_files = [
        "logs/parallel_logs/NLP_log.txt",
        "logs/parallel_logs/keybert_log.txt", 
        "logs/parallel_logs/reason_log.txt"
    ]
    
    status_found = False
    for log_file in log_files:
        if Path(log_file).exists():
            status_found = True
            mtime = datetime.fromtimestamp(Path(log_file).stat().st_mtime)
            if (datetime.now() - mtime).seconds < 300:  # Updated in last 5 minutes
                st.markdown(f'<span class="status-running">🔄 {log_file.split("/")[-1].replace("_log.txt", "").upper()} pipeline is running</span>', unsafe_allow_html=True)
            else:
                st.markdown(f'<span class="status-completed">✅ {log_file.split("/")[-1].replace("_log.txt", "").upper()} pipeline completed</span>', unsafe_allow_html=True)
    
    if not status_found:
        st.info("No pipeline activity detected.")

def show_recent_logs():
    """Show recent log entries"""
    st.markdown("**📋 Recent Activity:**")
    
    log_files = [
        ("NLP", "logs/parallel_logs/NLP_log.txt"),
        ("KeyBERT", "logs/parallel_logs/keybert_log.txt"),
        ("Reasoning", "logs/parallel_logs/reason_log.txt")
    ]
    
    for name, log_file in log_files:
        if Path(log_file).exists():
            with st.expander(f"{name} Logs"):
                try:
                    with open(log_file, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                        recent_lines = lines[-20:] if len(lines) > 20 else lines
                        st.code(''.join(recent_lines), language="text")
                except Exception as e:
                    st.error(f"Error reading {name} logs: {str(e)}")

if __name__ == "__main__":
    main() 