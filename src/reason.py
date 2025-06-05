import os
import asyncio, functools, time
from query_constructions import load_image_content
from agents import (
    readBugReportContent_agent,
    processBugReportQueryReasoning_agent,
    processBugReportContentPostReasoning_agent,
    index_source_code_agent,
    bug_localization_BM25_and_FAISS_agent
)

read_queue = reason_queue = process_queue = localization_queue = None
bm25_index = faiss_index = None
bm25_weight = 0.3
faiss_weight = 0.7
top_n_documents = 100

async def run_blocking(fn, *args, **kw):
    loop = asyncio.get_running_loop()
    part = functools.partial(fn, *args, **kw)
    return await loop.run_in_executor(None, part)

log_lock = asyncio.Lock()
async def log_event(tag, bug_id, stage, project_id):
    ts = time.strftime("%H:%M:%S", time.localtime())
    line = f"[{ts}] [{tag}] Project {project_id} Bug {bug_id} at stage: {stage}\n"
    async with log_lock:
        with open("./logs/parallel_logs/reason_log.txt", "a", encoding="utf-8") as f:
            f.write(line)

async def read_worker(project_id):
    while True:
        bug_dir, bug_id = await read_queue.get()
        await log_event("READ", bug_id, "start", project_id)
        raw = readBugReportContent_agent.run(bug_dir).get("file_content", "")
        extended_raw = raw + "\n" + load_image_content(bug_dir, bug_id)
        await reason_queue.put((bug_dir, bug_id, raw, extended_raw))
        await log_event("READ", bug_id, "done", project_id)
        read_queue.task_done()

async def reason_worker(project_id):
    while True:
        bug_dir, bug_id, raw, extended_raw = await reason_queue.get()
        await log_event("REASON", bug_id, "start", project_id)
        reasoned_raw = processBugReportQueryReasoning_agent.run(raw).get("file_content", "")
        reasoned_extended = processBugReportQueryReasoning_agent.run(extended_raw).get("file_content", "")
        await process_queue.put((bug_dir, bug_id, reasoned_raw, reasoned_extended))
        await log_event("REASON", bug_id, "done", project_id)
        reason_queue.task_done()

async def process_worker(output_base, project_id):
    while True:
        bug_dir, bug_id, raw_reasoned, ext_reasoned = await process_queue.get()
        await log_event("PROCESS", bug_id, "start", project_id)
        baseline_query = processBugReportContentPostReasoning_agent.run(raw_reasoned).get("file_content", "")
        extended_query = processBugReportContentPostReasoning_agent.run(ext_reasoned).get("file_content", "")

        out_dir = os.path.join(output_base, bug_id)
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, f"{bug_id}_baseline_reasoning_query.txt"), "w", encoding="utf-8") as f:
            f.write(baseline_query)
        with open(os.path.join(out_dir, f"{bug_id}_extended_reasoning_query.txt"), "w", encoding="utf-8") as f:
            f.write(extended_query)

        await log_event("PROCESS", bug_id, "done", project_id)
        await localization_queue.put((bug_id, baseline_query, extended_query))
        process_queue.task_done()

async def localize_worker(search_base, top_n_documents, processed_documents, project_id):
    while True:
        bug_id, baseline_q, extended_q = await localization_queue.get()

        out_dir = os.path.join(search_base, bug_id)
        os.makedirs(out_dir, exist_ok=True)

        await log_event("LOCALIZE", bug_id, "baseline start", project_id)
        baseline_res = await run_blocking(
            bug_localization_BM25_and_FAISS_agent.run,
            bug_id, baseline_q, top_n_documents,
            bm25_index, faiss_index, processed_documents,
            bm25_weight, faiss_weight
        )
        with open(os.path.join(out_dir, f"{bug_id}_baseline_reasoning_query_result.txt"), "w", encoding="utf-8") as f:
            f.write(baseline_res.get("file_content", ""))

        await log_event("LOCALIZE", bug_id, "extended start", project_id)
        extended_res = await run_blocking(
            bug_localization_BM25_and_FAISS_agent.run,
            bug_id, extended_q, top_n_documents,
            bm25_index, faiss_index, processed_documents,
            bm25_weight, faiss_weight
        )
        with open(os.path.join(out_dir, f"{bug_id}_extended_reasoning_query_result.txt"), "w", encoding="utf-8") as f:
            f.write(extended_res.get("file_content", ""))
        await log_event("LOCALIZE", bug_id, "done", project_id)
        localization_queue.task_done()

async def main_async(project_id, bug_reports_root, queries_output_root, search_result_path, source_code_dir, bm25_faiss_dir):
    global read_queue, reason_queue, process_queue, localization_queue
    global bm25_index, faiss_index

    read_queue         = asyncio.Queue()
    reason_queue       = asyncio.Queue()
    process_queue      = asyncio.Queue()
    localization_queue = asyncio.Queue()

    bm25_index, faiss_index, processed_documents = index_source_code_agent.run(source_code_dir, f"project{project_id}", bm25_faiss_dir).get("file_content", "")
    top_n_documents = len(processed_documents)

    bug_path = os.path.join(bug_reports_root, project_id)
    output_base = os.path.join(queries_output_root, project_id)
    search_base = os.path.join(search_result_path, project_id)
    os.makedirs(output_base, exist_ok=True)
    os.makedirs(search_base, exist_ok=True)
    os.makedirs("./logs/parallel_logs", exist_ok=True)

    for bug_id in os.listdir(bug_path):
        bug_dir = os.path.join(bug_path, bug_id)
        if os.path.isdir(bug_dir):
            await read_queue.put((bug_dir, bug_id))

    workers = [
        asyncio.create_task(read_worker(project_id)),
        asyncio.create_task(reason_worker(project_id)),
        asyncio.create_task(process_worker(output_base, project_id)),
        *[asyncio.create_task(localize_worker(search_base, top_n_documents, processed_documents, project_id)) for _ in range(4)],
    ]

    await read_queue.join()
    await reason_queue.join()
    await process_queue.join()
    await localization_queue.join()

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
    open("./logs/parallel_logs/reason_log.txt", "w").close()
    
    async def process_all_projects():
        for project_id in projects:
            print(f"\n=== Processing Project {project_id} with Reasoning ===")
            
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
