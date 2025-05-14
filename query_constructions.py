import os
from agents import (
    readBugReportContent_agent,
    processBugReportContent_agent,
    processBugRepotQueryKeyBERT_agent
)


def read_file_with_fallback(path):
    """Reads the file with fallback encoding, helper function for the loading of image content"""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except UnicodeDecodeError:
        with open(path, "r", encoding="iso-8859-1") as f:
            return f.read().strip()
        
def load_image_content(bug_dir, bug_id):
    """Loads all image content files for a given bug report, helper function for the extended query"""
    content = ""
    image_files = sorted([
        f for f in os.listdir(bug_dir)
        if f.startswith(bug_id) and f.endswith("ImageContent.txt")
    ])
    for image_file in image_files:
        image_path = os.path.join(bug_dir, image_file)
        content += "\n" + read_file_with_fallback(image_path)
    return content.strip()


def construct_queries_for_project(project_id, bug_reports_root, queries_output_root):
    project_bug_path = os.path.join(bug_reports_root, project_id)
    project_query_path = os.path.join(queries_output_root, project_id)
    os.makedirs(project_query_path, exist_ok=True)

    for bug_id in os.listdir(project_bug_path):
        bug_dir = os.path.join(project_bug_path, bug_id)
        if not os.path.isdir(bug_dir):
            continue

        print(f"\n[INFO] Processing bug {bug_id}...")
        output_dir = os.path.join(project_query_path, bug_id)
        os.makedirs(output_dir, exist_ok=True)

        # === BASELINE QUERY ===
        # Step 1: Read
        baseline_raw = readBugReportContent_agent.run(bug_dir).get("file_content", "")

        # Step 2: Process content
        baseline_processed = processBugReportContent_agent.run(baseline_raw).get("file_content", "")

        # Step 3: generate queries and save
        baseline_keywords = processBugRepotQueryKeyBERT_agent.run(baseline_processed).get("file_content", [])
        if isinstance(baseline_keywords, list):
            #baseline_query = baseline_processed + " " + " ".join(baseline_keywords)
            baseline_query = " ".join(baseline_keywords)
        else:
            baseline_query = str(baseline_keywords)

        with open(os.path.join(output_dir, "baseline_query.txt"), "w", encoding="utf-8") as f:
            f.write(baseline_query)

        # === EXTENDED QUERY ===
        # Step 1: Read baseline content + image content
        extended_raw = baseline_raw + "\n" + load_image_content(bug_dir, bug_id)
        print(f"[DEBUG] Image content for bug {bug_id}:\n{load_image_content(bug_dir, bug_id)}")

        # Step 2: Process content
        extended_processed = processBugReportContent_agent.run(extended_raw).get("file_content", "")
        

        # Step 3: generate queries and save
        extended_keywords = processBugRepotQueryKeyBERT_agent.run(extended_processed).get("file_content", [])
        if isinstance(extended_keywords, list):
            #extended_query = extended_processed + " " + " ".join(extended_keywords)
            extended_query = " ".join(extended_keywords)
        else:
            extended_query = str(extended_keywords)

        with open(os.path.join(output_dir, "extended_keyBERT_query.txt"), "w", encoding="utf-8") as f:
            f.write(extended_query)

        print(f" Saved: {output_dir}")
