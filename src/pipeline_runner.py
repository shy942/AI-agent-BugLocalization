# pipeline_runner.py
import os
import asyncio
from reason import main_async  # or your actual file

# Define base paths
BASE_PATH = "./AgentProjectData"
BUG_REPORTS_ROOT = os.path.join(BASE_PATH, "ProjectBugReports")
QUERIES_OUTPUT_ROOT = os.path.join(BASE_PATH, "ConstructedQueries/BaselineVsReason/")
SEARCH_RESULT_PATH = os.path.join(BASE_PATH, "SearchResults")
SOURCE_CODES_ROOT = os.path.join(BASE_PATH, "SourceCodes")
BM25_FAISS_DIR = os.path.join(BASE_PATH, "BM25andFAISS")

# List of projects
#PROJECTS = ["24", "40", "42", "43", "44", "46", "47", "49", "50"]
# def get_available_projects():
#     root = "./AgentProjectData/ProjectBugReports"
#     return sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)))

def get_available_projects():
    root = os.path.abspath("AgentProjectData/ProjectBugReports")
    if not os.path.exists(root):
        return []
    return sorted(
        d for d in os.listdir(root)
        if os.path.isdir(os.path.join(root, d))
    )


async def process_selected_projects(selected_projects):
    os.makedirs("./logs/parallel_logs", exist_ok=True)
    open("./logs/parallel_logs/reason_log.txt", "w").close()

    for project_id in selected_projects:
        print(f"\n=== Processing Project {project_id} ===")
        project_source_dir = os.path.join(SOURCE_CODES_ROOT, f"Project{project_id}")
        
        if os.path.exists(project_source_dir):
            subdirs = [d for d in os.listdir(project_source_dir) 
                       if os.path.isdir(os.path.join(project_source_dir, d)) and d != "Corpus"]
            source_code_dir = os.path.join(project_source_dir, subdirs[0]) if subdirs else project_source_dir
            src_dir = os.path.join(source_code_dir, "src")
            if os.path.exists(src_dir):
                source_code_dir = src_dir
        else:
            print(f"Warning: Missing source code dir for Project {project_id}")
            continue

        await main_async(project_id, BUG_REPORTS_ROOT, QUERIES_OUTPUT_ROOT, 
                         SEARCH_RESULT_PATH, source_code_dir, BM25_FAISS_DIR)
        print(f"=== Done Project {project_id} ===")
