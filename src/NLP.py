import os
import asyncio, functools, time
from query_constructions import load_image_content
from agents import (
    readBugReportContent_agent,
    processBugReportContent_agent,
    index_source_code_agent,
    bug_localization_BM25_and_FAISS_agent,
)

# Queues between pipeline stages
read_queue = process_queue = localization_queue = None

# Shared index data
bm25_index = None
faiss_index = None
bm25_weight = 0.3
faiss_weight = 0.7
top_n_documents = 100

# Helper: run blocking code in thread
async def run_blocking(fn, *args, **kw):
    loop = asyncio.get_running_loop()
    part = functools.partial(fn, *args, **kw)
    return await loop.run_in_executor(None, part)

# Helper: log events to file
log_lock = asyncio.Lock()
async def log_event(tag, bug_id, stage):
    ts = time.strftime("%H:%M:%S", time.localtime())
    line = f"[{ts}] [{tag}] Bug {bug_id} at stage: {stage}\n"
    async with log_lock:
        with open("./logs/parallel_logs/NLP_log.txt", "a", encoding="utf-8") as f:
            f.write(line)

async def read_worker():
    while True:
        bug_dir, bug_id = await read_queue.get()
        await log_event("READ", bug_id, "start")
        raw = readBugReportContent_agent.run(bug_dir).get("file_content", "")
        extended_raw = raw + "\n" + load_image_content(bug_dir, bug_id)
        await log_event("READ", bug_id, "done")
        await process_queue.put((bug_id, raw, extended_raw))
        read_queue.task_done()

async def process_worker(output_base):
    while True:
        bug_id, raw, extended_raw = await process_queue.get()
        await log_event("PROCESS", bug_id, "start")

        baseline = processBugReportContent_agent.run(raw).get("file_content", "")
        extended = processBugReportContent_agent.run(extended_raw).get("file_content", "")

        out_dir = os.path.join(output_base, bug_id)
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, f"{bug_id}_baseline_basic_query.txt"), "w", encoding="utf-8") as f:
            f.write(baseline)
        with open(os.path.join(out_dir, f"{bug_id}_extended_basic_query.txt"), "w", encoding="utf-8") as f:
            f.write(extended)

        await log_event("PROCESS", bug_id, "done")
        await localization_queue.put((bug_id, baseline, extended))
        process_queue.task_done()

async def localize_worker(search_base, top_n_documents, processed_documents):
    while True:
        bug_id, baseline_q, extended_q = await localization_queue.get()
        out_dir = os.path.join(search_base, bug_id)
        os.makedirs(out_dir, exist_ok=True)

        await log_event("LOCALIZE", bug_id, "baseline start")
        baseline_res = await run_blocking(
            bug_localization_BM25_and_FAISS_agent.run,
            bug_id, baseline_q, top_n_documents,
            bm25_index, faiss_index, processed_documents,
            bm25_weight, faiss_weight
        )
        with open(os.path.join(out_dir, f"{bug_id}_baseline_basic_query_result.txt"), "w", encoding="utf-8") as f:
            f.write(baseline_res.get("file_content", ""))
        await log_event("LOCALIZE", bug_id, "baseline done")

        await log_event("LOCALIZE", bug_id, "extended start")
        extended_res = await run_blocking(
            bug_localization_BM25_and_FAISS_agent.run,
            bug_id, extended_q, top_n_documents,
            bm25_index, faiss_index, processed_documents,
            bm25_weight, faiss_weight
        )
        with open(os.path.join(out_dir, f"{bug_id}_extended_basic_query_result.txt"), "w", encoding="utf-8") as f:
            f.write(extended_res.get("file_content", ""))
        await log_event("LOCALIZE", bug_id, "extended done")

        localization_queue.task_done()

async def main_async(project_id, bug_reports_root, constructed_query_root, search_result_path):
    global read_queue, process_queue, localization_queue
    global bm25_index, faiss_index

    read_queue = asyncio.Queue()
    process_queue = asyncio.Queue()
    localization_queue = asyncio.Queue()

    bm25_index, faiss_index, processed_documents = index_source_code_agent.run(SourceCodeDir).get("file_content", "")
    top_n_documents = len(processed_documents)

    bug_path = os.path.join(bug_reports_root, project_id)
    constructed_base = os.path.join(constructed_query_root, project_id)
    search_base = os.path.join(search_result_path, project_id)
    os.makedirs(constructed_base, exist_ok=True)
    os.makedirs(search_base, exist_ok=True)
    os.makedirs("./logs/parallel_logs", exist_ok=True)
    open("./logs/parallel_logs/NLP_log.txt", "w").close()

    for bug_id in os.listdir(bug_path):
        bug_dir = os.path.join(bug_path, bug_id)
        if os.path.isdir(bug_dir):
            await read_queue.put((bug_dir, bug_id))

    workers = [
        asyncio.create_task(read_worker()),
        asyncio.create_task(process_worker(constructed_base)),
        *[asyncio.create_task(localize_worker(search_base, top_n_documents, processed_documents)) for _ in range(4)],
    ]

    await read_queue.join()
    await process_queue.join()
    await localization_queue.join()

    for w in workers:
        w.cancel()

if __name__ == "__main__":
    project_id = "103"
    BugReportPath = os.path.expanduser("./ExampleProjectData/ProjectBugReports/")
    SourceCodeDir = os.path.expanduser("./ExampleProjectData/SourceCodes/Project103/tables/src/")
    ConstructedQueryPath = os.path.expanduser("./ExampleProjectData/ConstructedQueries/")
    SearchResultPath = os.path.expanduser("./ExampleProjectData/SearchResults/")
    
    asyncio.run(main_async(project_id, BugReportPath, ConstructedQueryPath, SearchResultPath))
