import os
import asyncio, functools, time
from query_constructions import load_image_content
from agents import (
    readBugReportContent_agent,
    processBugReportContent_agent,
    processBugReportQueryKeyBERT_agent,
    index_source_code_agent,
    bug_localization_BM25_and_FAISS_agent
)

# Queues between pipeline stages, initialized in main_async
read_queue = process_queue = keybert_queue = localization_queue = None


# Shared index data
bm25_index = None
faiss_index = None
bm25_weight = 0.3
faiss_weight = 0.7
top_n = 15 # Number of top keywords to retrieve
top_n_documents = 100 # Number of top documents to retrieve, but default is 100



# Helper : run blocking CPU-bound code in a thread
async def run_blocking(fn, *args, **kw):
    loop = asyncio.get_running_loop()
    part = functools.partial(fn, *args, **kw)
    return await loop.run_in_executor(None, part)

#Helper: Log events to the pipeline_log.txt file
log_lock = asyncio.Lock()
async def log_event(tag, bug_id, stage, project_id):
    ts = time.strftime("%H:%M:%S", time.localtime())
    line = f"[{ts}] [{tag}] Project {project_id} Bug {bug_id} at stage: {stage}\n"
    async with log_lock:
        with open("./logs/parallel_logs/keybert_log.txt", "a", encoding="utf-8") as f:
            f.write(line)

async def read_worker(project_id):
    '''Reads title + description from bug report folder'''
    while True:
        bug_dir, bug_id = await read_queue.get()
        await log_event("READ", bug_id, "start", project_id)
        raw = readBugReportContent_agent.run(bug_dir).get("file_content", "")
        extended_raw = raw + "\n" + load_image_content(bug_dir, bug_id)
        await log_event("READ", bug_id, "done", project_id)
        await process_queue.put((bug_dir, bug_id, raw, extended_raw))
        read_queue.task_done()

async def process_worker(project_id):
    '''Processes baseline and extended content'''
    while True:
        bug_dir, bug_id, raw, extended_raw= await process_queue.get()
        await log_event("PROCESS", bug_id, "start", project_id)
        processed = processBugReportContent_agent.run(raw).get("file_content", "")
        extended_processed = processBugReportContent_agent.run(extended_raw).get("file_content", "")
        await log_event("PROCESS", bug_id, "done", project_id)
        await keybert_queue.put((bug_dir, bug_id, processed, extended_processed))
        process_queue.task_done()

async def keybert_worker(output_base, top_n, project_id):
    '''Extracts keywords for both baseline and extended queries and writes them to disk'''
    while True:
        bug_dir, bug_id, baseline_processed, extended_processed= await keybert_queue.get()
        await log_event("KEYBERT", bug_id, "start", project_id)

        # === Baseline ===
        keywords = await run_blocking(processBugReportQueryKeyBERT_agent.run, baseline_processed, top_n)
        baseline_query = " ".join(keywords.get("file_content", [])) 

        # === Extended ===
        extended_keywords = await run_blocking(processBugReportQueryKeyBERT_agent.run, extended_processed, top_n)
        extended_query = " ".join(extended_keywords.get("file_content", [])) 

        output_dir = os.path.join(output_base, bug_id)
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "baseline_keyBERT_query.txt"), "w", encoding="utf-8") as f:
            f.write(baseline_query)
        with open(os.path.join(output_dir, "extended_keyBERT_query.txt"), "w", encoding="utf-8") as f:
            f.write(extended_query)

        await log_event("KEYBERT", bug_id, "done", project_id)
        await localization_queue.put((bug_id, baseline_query, extended_query))
        keybert_queue.task_done()

async def localize_worker(search_base, top_n_documents, processed_documents, project_id):
    '''Runs BM25+FAISS localization on extended queries'''
    while True:
        bug_id, baseline_q, extended_q = await localization_queue.get()

        await log_event("LOCALIZE", bug_id, "baseline start", project_id)
        baseline_search_results = await run_blocking(
            bug_localization_BM25_and_FAISS_agent.run,
            bug_id, baseline_q, top_n_documents,
            bm25_index, faiss_index, processed_documents,
            bm25_weight, faiss_weight
        )

        out_dir = os.path.join(search_base, bug_id)
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, f"{bug_id}_baseline_keyBERT_query_result.txt"),
                  "w", encoding="utf-8") as f:
            f.write(baseline_search_results.get("file_content", ""))
        await log_event("LOCALIZE", bug_id, "baseline done", project_id)

        await log_event("LOCALIZE", bug_id, "extended start", project_id)
        extended_search_results = await run_blocking(
            bug_localization_BM25_and_FAISS_agent.run,
            bug_id, extended_q, top_n_documents,
            bm25_index, faiss_index, processed_documents,
            bm25_weight, faiss_weight
        )
        with open(os.path.join(out_dir, f"{bug_id}_extended_keyBERT_query_result.txt"),
                  "w", encoding="utf-8") as f:
            f.write(extended_search_results.get("file_content", ""))
        await log_event("LOCALIZE", bug_id, "extended done", project_id)
        localization_queue.task_done()

async def main_async(project_id, bug_reports_root, queries_output_root, search_result_path, source_code_dir, bm25_faiss_dir):
    global read_queue, process_queue, keybert_queue, localization_queue
    global bm25_index, faiss_index
    read_queue         = asyncio.Queue()
    process_queue      = asyncio.Queue()
    keybert_queue      = asyncio.Queue()
    localization_queue = asyncio.Queue()
    
    # index source code and load indexes (bm25_index, faiss_index) and processed documents
    bm25_index, faiss_index, processed_documents = index_source_code_agent.run(source_code_dir, f"project{project_id}", bm25_faiss_dir).get("file_content", "")
    top_n_documents = len(processed_documents) # Number of top documents to retrieve, but default is 100

    bug_path = os.path.join(bug_reports_root, project_id)
    output_base = os.path.join(queries_output_root, project_id)
    os.makedirs(output_base, exist_ok=True)
    search_results_base = os.path.join(search_result_path, project_id)
    os.makedirs(search_results_base, exist_ok=True)
    os.makedirs("./logs/parallel_logs", exist_ok=True)

    # Fill read queue with bug IDs
    for bug_id in os.listdir(bug_path):
        bug_dir = os.path.join(bug_path, bug_id)
        if os.path.isdir(bug_dir):
            await read_queue.put((bug_dir, bug_id))

    # Start
    workers = [
        asyncio.create_task(read_worker(project_id)),
        asyncio.create_task(process_worker(project_id)),
        asyncio.create_task(keybert_worker(output_base, top_n, project_id)),
        *[asyncio.create_task(localize_worker(search_results_base, top_n_documents, processed_documents, project_id)) for _ in range(4)],
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
    # Define base paths for AgentProjectData
    base_path = "./AgentProjectData"
    bug_reports_root = os.path.join(base_path, "ProjectBugReports")
    queries_output_root = os.path.join(base_path, "ConstructedQueries")
    search_result_path = os.path.join(base_path, "SearchResults")
    source_codes_root = os.path.join(base_path, "SourceCodes")
    bm25_faiss_dir = os.path.join(base_path, "BM25andFAISS")
    
    # Process all 5 projects
    projects = ["3", "13", "14", "20", "24"]
    
    # Clear the log file at the start
    os.makedirs("./logs/parallel_logs", exist_ok=True)
    open("./logs/parallel_logs/keybert_log.txt", "w").close()
    
    async def process_all_projects():
        for project_id in projects:
            print(f"\n=== Processing Project {project_id} with KeyBERT ===")
            
            # Find the source code directory for this project
            project_source_dir = os.path.join(source_codes_root, f"Project{project_id}")
            
            # Find the actual source directory (exclude Corpus directory)
            if os.path.exists(project_source_dir):
                subdirs = [d for d in os.listdir(project_source_dir) 
                          if os.path.isdir(os.path.join(project_source_dir, d)) and d != "Corpus"]
                if subdirs:
                    # Take the first non-Corpus subdirectory as the source directory
                    source_code_dir = os.path.join(project_source_dir, subdirs[0])
                    # If there's a 'src' directory inside, use that
                    src_dir = os.path.join(source_code_dir, "src")
                    if os.path.exists(src_dir):
                        source_code_dir = src_dir
                else:
                    source_code_dir = project_source_dir
            else:
                print(f"Warning: Source code directory not found for Project {project_id}")
                continue
            
            print(f"Using source code directory: {source_code_dir}")
            
            await main_async(project_id, bug_reports_root, queries_output_root, 
                           search_result_path, source_code_dir, bm25_faiss_dir)
            
            print(f"=== Completed Project {project_id} ===")
    
    asyncio.run(process_all_projects())