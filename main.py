import os
from query_constructions import (
    load_image_content
)
from agents import (
    readBugReportContent_agent,
    processBugReportContent_agent,
    processBugRepotQueryKeyBERT_agent
)

    

def main_manager(project_id, bug_reports_root, queries_output_root):
    project_bug_path = os.path.join(bug_reports_root, project_id)
    project_query_path = os.path.join(queries_output_root, project_id)
    os.makedirs(project_query_path, exist_ok=True)
    top_n = 10
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
        baseline_keywords = processBugRepotQueryKeyBERT_agent.run(baseline_processed, top_n).get("file_content", [])
        if isinstance(baseline_keywords, list):
            #I need to keep the original query as well
            #baseline_query = baseline_processed + " " + " ".join(baseline_keywords)
            #When I only consider the keywords extracted by KeyBERT
            baseline_query = " ".join(baseline_keywords)
        else:
            baseline_query = str(baseline_keywords)

        with open(os.path.join(output_dir, "baseline_keyBERT_query.txt"), "w", encoding="utf-8") as f:
            f.write(baseline_query)

        # === EXTENDED QUERY ===
        # Step 1: Read baseline content + image content
        extended_raw = baseline_raw + "\n" + load_image_content(bug_dir, bug_id)
        print(f"[DEBUG] Image content for bug {bug_id}:\n{load_image_content(bug_dir, bug_id)}")

        # Step 2: Process content
        extended_processed = processBugReportContent_agent.run(extended_raw).get("file_content", "")
        

        # Step 3: generate queries and save
        extended_keywords = processBugRepotQueryKeyBERT_agent.run(extended_processed, top_n).get("file_content", [])
        if isinstance(extended_keywords, list):
            #I need to keep the original query as well
            #extended_query = extended_processed + " " + " ".join(extended_keywords)
            #When I only consider the keywords extracted by KeyBERT
            extended_query = " ".join(extended_keywords)
        else:
            extended_query = str(extended_keywords)

        with open(os.path.join(output_dir, "extended_keyBERT_query.txt"), "w", encoding="utf-8") as f:
            f.write(extended_query)

        print(f" Saved: {output_dir}")


if __name__ == "__main__":
    project_id = "103"
    BugReportPath = os.path.expanduser("./ExampleProjectData/ProjectBugReports/")
    SearchQueryPath = os.path.expanduser("./ExampleProjectData/ConstructedQueries/")

    main_manager(project_id, BugReportPath, SearchQueryPath)
