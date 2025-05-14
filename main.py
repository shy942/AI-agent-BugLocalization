import os
import asyncio
from query_constructions import load_image_content
from agents import (
    readBugReportContent_agent,
    processBugReportContent_agent,
    processBugReportQueryKeyBERT_agent
)
read_queue = asyncio.Queue()
process_queue = asyncio.Queue()
keybert_queue = asyncio.Queue()
async def read_worker():
    '''
    Reads the content of each bug report (title + description)
    Once read, the raw text content is passed to the process queue along with bug ID.
    '''
    while True:
        bug_dir, bug_id = await read_queue.get()
        raw = readBugReportContent_agent.run(bug_dir).get("file_content", "")
        await process_queue.put((bug_dir, bug_id, raw))
        read_queue.task_done()

async def process_worker():
    '''
    Processes the raw bug report content by cleaning and normalizing the text.
    Takes (bug_dir, bug_id, raw_text) from process_queue and puts (bug_dir, bug_id, raw, processed) into keybert_queue.
    '''
    while True:
        bug_dir, bug_id, raw = await process_queue.get()
        processed = processBugReportContent_agent.run(raw).get("file_content", "")
        await keybert_queue.put((bug_dir, bug_id, raw, processed))
        process_queue.task_done()

async def keybert_worker(output_base, top_n):
    '''
    Uses KeyBERT to extract keywords from both baseline and extended bug report content.
    Saves the resulting queries to "baseline_keyBERT_query.txt" and "extended_keyBERT_query.txt" for each bug report.
    '''
    while True:
        bug_dir, bug_id, raw, processed = await keybert_queue.get()

        keywords = processBugReportQueryKeyBERT_agent.run(processed, top_n).get("file_content", [])
        baseline_query = " ".join(keywords) if isinstance(keywords, list) else str(keywords)

        output_dir = os.path.join(output_base, bug_id)
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "baseline_keyBERT_query.txt"), "w", encoding="utf-8") as f:
            f.write(baseline_query)

        extended_raw = raw + "\n" + load_image_content(bug_dir, bug_id)
        extended_processed = processBugReportContent_agent.run(extended_raw).get("file_content", "")
        extended_keywords = processBugReportQueryKeyBERT_agent.run(extended_processed, top_n).get("file_content", [])
        extended_query = " ".join(extended_keywords) if isinstance(extended_keywords, list) else str(extended_keywords)

        with open(os.path.join(output_dir, "extended_keyBERT_query.txt"), "w", encoding="utf-8") as f:
            f.write(extended_query)

        print(f"Saved: {output_dir}")
        keybert_queue.task_done()

async def main_async(project_id, bug_reports_root, queries_output_root):
    '''
    Main async pipeline controller:
    - Prepares the input queue with all bug reports in the project.
    - Starts read, process, and keybert workers in parallel.
    - And it will wait for all queues to be processed and then shuts down workers.
    '''
    top_n = 10
    bug_path = os.path.join(bug_reports_root, project_id)
    output_base = os.path.join(queries_output_root, project_id)
    os.makedirs(output_base, exist_ok=True)

    # Enqueue bug dirs
    for bug_id in os.listdir(bug_path):
        bug_dir = os.path.join(bug_path, bug_id)
        if os.path.isdir(bug_dir):
            await read_queue.put((bug_dir, bug_id))

    # Start!
    workers = [
        asyncio.create_task(read_worker()),
        asyncio.create_task(process_worker()),
        asyncio.create_task(keybert_worker(output_base, top_n))
    ]

    # Wait for all items in the 3 queues to be processed
    await read_queue.join()
    await process_queue.join()
    await keybert_queue.join()

    # Shutdown 
    for w in workers:
        w.cancel()

if __name__ == "__main__":
    project_id = "103"
    BugReportPath = os.path.expanduser("./ExampleProjectData/ProjectBugReports/")
    SearchQueryPath = os.path.expanduser("./ExampleProjectData/ConstructedQueries/")

    asyncio.run(main_async(project_id, BugReportPath, SearchQueryPath))
