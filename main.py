import os
import asyncio
from query_constructions import load_image_content
from agents import (
    readBugReportContent_agent,
    processBugReportContent_agent,
    processBugReportQueryKeyBERT_agent,
    index_source_code_agent,
    load_index_bm25_and_faiss_agent,
    bug_localization_BM25_and_FAISS_agent
)

# Queues between pipeline stages
read_queue = asyncio.Queue()
process_queue = asyncio.Queue()
keybert_queue = asyncio.Queue()
localization_queue = asyncio.Queue()

# Shared index data
bm25_index = None
faiss_index = None

async def read_worker():
    '''Reads title + description from bug report folder'''
    while True:
        bug_dir, bug_id = await read_queue.get()
        raw = readBugReportContent_agent.run(bug_dir).get("file_content", "")
        await process_queue.put((bug_dir, bug_id, raw))
        read_queue.task_done()

async def process_worker():
    '''Processes baseline and extended content'''
    while True:
        bug_dir, bug_id, raw = await process_queue.get()
        processed = processBugReportContent_agent.run(raw).get("file_content", "")
        extended_raw = raw + "\n" + load_image_content(bug_dir, bug_id)
        extended_processed = processBugReportContent_agent.run(extended_raw).get("file_content", "")
        await keybert_queue.put((bug_dir, bug_id, processed, extended_processed, raw))
        process_queue.task_done()

async def keybert_worker(output_base, top_n):
    '''Extracts keywords for both baseline and extended queries and writes them to disk'''
    while True:
        bug_dir, bug_id, baseline_processed, extended_processed, raw = await keybert_queue.get()

        # === Baseline ===
        keywords = processBugReportQueryKeyBERT_agent.run(baseline_processed, top_n).get("file_content", [])
        baseline_query = " ".join(keywords) if isinstance(keywords, list) else str(keywords)

        # === Extended ===
        extended_keywords = processBugReportQueryKeyBERT_agent.run(extended_processed, top_n).get("file_content", [])
        extended_query = " ".join(extended_keywords) if isinstance(extended_keywords, list) else str(extended_keywords)

        output_dir = os.path.join(output_base, bug_id)
        os.makedirs(output_dir, exist_ok=True)

        with open(os.path.join(output_dir, "baseline_keyBERT_query.txt"), "w", encoding="utf-8") as f:
            f.write(baseline_query)
        with open(os.path.join(output_dir, "extended_keyBERT_query.txt"), "w", encoding="utf-8") as f:
            f.write(extended_query)

        print(f" Saved: {output_dir}")
        await localization_queue.put((extended_query, bug_id))
        keybert_queue.task_done()

async def localize_worker(top_n):
    '''Runs BM25+FAISS localization on extended queries'''
    while True:
        extended_query, bug_id = await localization_queue.get()
        bug_localization_BM25_and_FAISS_agent.run(extended_query, top_n, bm25_index, faiss_index)
        print(f" Localized bug {bug_id}")
        localization_queue.task_done()

async def main_async(project_id, bug_reports_root, queries_output_root, index_paths, top_n=10):
    global bm25_index, faiss_index

    # Load precomputed indexes 
    bm25_index, faiss_index = load_index_bm25_and_faiss_agent.run(*index_paths).get("file_content", "")

    bug_path = os.path.join(bug_reports_root, project_id)
    output_base = os.path.join(queries_output_root, project_id)
    os.makedirs(output_base, exist_ok=True)

    # Fill read queue with bug IDs
    for bug_id in os.listdir(bug_path):
        bug_dir = os.path.join(bug_path, bug_id)
        if os.path.isdir(bug_dir):
            await read_queue.put((bug_dir, bug_id))

    # Start
    workers = [
        asyncio.create_task(read_worker()),
        asyncio.create_task(process_worker()),
        asyncio.create_task(keybert_worker(output_base, top_n)),
        asyncio.create_task(localize_worker(top_n))
    ]

    # Wait for all queues to finish
    await read_queue.join()
    await process_queue.join()
    await keybert_queue.join()
    await localization_queue.join()

    # Finish
    for w in workers:
        w.cancel()

if __name__ == "__main__":
    project_id = "103"
    BugReportPath = os.path.expanduser("./ExampleProjectData/ProjectBugReports/")
    SearchQueryPath = os.path.expanduser("./ExampleProjectData/ConstructedQueries/")
    SourceCodeDir = os.path.expanduser("./ExampleProjectData/SourceCodes/Project103/tables/src/")

    # Generate indexes, I think this optional here if it's precomputed
    index_source_code_agent.run(SourceCodeDir).get("file_content", "")
    bm25_path = "./bm25_index.pkl"
    faiss_path = "./faiss_index_dir"

    asyncio.run(main_async(project_id, BugReportPath, SearchQueryPath, (bm25_path, faiss_path)))
