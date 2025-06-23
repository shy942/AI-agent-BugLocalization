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
bm25_weight = 0.5
faiss_weight = 0.5
top_n_documents = 100

# Helper: run blocking code in thread
async def run_blocking(fn, *args, **kw):
    loop = asyncio.get_running_loop()
    part = functools.partial(fn, *args, **kw)
    return await loop.run_in_executor(None, part)

# Helper: log events to file
log_lock = asyncio.Lock()
async def log_event(tag, bug_id, stage, project_id):
    ts = time.strftime("%H:%M:%S", time.localtime())
    line = f"[{ts}] [{tag}] Project {project_id} Bug {bug_id} at stage: {stage}\n"
    async with log_lock:
        with open("./logs/parallel_logs/NLP_log.txt", "a", encoding="utf-8") as f:
            f.write(line)

async def read_worker(project_id):
    while True:
        bug_dir, bug_id = await read_queue.get()
        await log_event("READ", bug_id, "start", project_id)
        raw = readBugReportContent_agent.run(bug_dir).get("file_content", "")
        extended_raw = raw + "\n" + load_image_content(bug_dir, bug_id)
        await log_event("READ", bug_id, "done", project_id)
        await process_queue.put((bug_id, raw, extended_raw))
        read_queue.task_done()

async def process_worker(output_base, project_id):
    while True:
        bug_id, raw, extended_raw = await process_queue.get()
        await log_event("PROCESS", bug_id, "start", project_id)

        #baseline = processBugReportContent_agent.run(raw).get("file_content", "")
        extended = processBugReportContent_agent.run(extended_raw).get("file_content", "")

        # out_dir = os.path.join(output_base, project_id)
        # os.makedirs(out_dir, exist_ok=True)
        # # with open(os.path.join(out_dir, f"{bug_id}_baseline_basic_query.txt"), "w", encoding="utf-8") as f:
        #     f.write(baseline)
        with open(os.path.join(output_base, f"{bug_id}_baseline_reasoning_query.txt"), "w", encoding="utf-8") as f:
            f.write(extended)

        await log_event("PROCESS", bug_id, "done", project_id)
        #await localization_queue.put((bug_id, baseline, extended))
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
        with open(os.path.join(out_dir, f"{bug_id}_baseline_basic_query_result.txt"), "w", encoding="utf-8") as f:
            f.write(baseline_res.get("file_content", ""))
        await log_event("LOCALIZE", bug_id, "baseline done", project_id)

        await log_event("LOCALIZE", bug_id, "extended start", project_id)
        extended_res = await run_blocking(
            bug_localization_BM25_and_FAISS_agent.run,
            bug_id, extended_q, top_n_documents,
            bm25_index, faiss_index, processed_documents,
            bm25_weight, faiss_weight
        )
        with open(os.path.join(out_dir, f"{bug_id}_extended_basic_query_result.txt"), "w", encoding="utf-8") as f:
            f.write(extended_res.get("file_content", ""))
        await log_event("LOCALIZE", bug_id, "extended done", project_id)

        localization_queue.task_done()

async def main_async(project_id, bug_reports_root, constructed_query_root, search_result_path, source_code_dir, bm25_faiss_dir):
    global read_queue, process_queue, localization_queue
    global bm25_index, faiss_index

    read_queue = asyncio.Queue()
    process_queue = asyncio.Queue()
    localization_queue = asyncio.Queue()

    # bm25_index, faiss_index, processed_documents = index_source_code_agent.run(source_code_dir, f"project{project_id}", bm25_faiss_dir).get("file_content", "")
    # top_n_documents = len(processed_documents)

    bug_path = os.path.join(bug_reports_root, project_id)
    constructed_base = os.path.join(constructed_query_root, project_id+"_no_stem")
    search_base = os.path.join(search_result_path, project_id)
    os.makedirs(constructed_base, exist_ok=True)
    os.makedirs(search_base, exist_ok=True)
    os.makedirs("./logs/parallel_logs", exist_ok=True)

    for bug_id in os.listdir(bug_path):
        bug_dir = os.path.join(bug_path, bug_id)
        if os.path.isdir(bug_dir):
            await read_queue.put((bug_dir, bug_id))

    workers = [
        asyncio.create_task(read_worker(project_id)),
        asyncio.create_task(process_worker(constructed_base, project_id)),
        #*[asyncio.create_task(localize_worker(search_base, top_n_documents, processed_documents, project_id)) for _ in range(4)],
    ]

    await read_queue.join()
    await process_queue.join()
    await localization_queue.join()

    for w in workers:
        w.cancel()

if __name__ == "__main__":
    # Define base paths for AgentProjectData
    base_path = "./AgentProjectData"
    bug_reports_root = os.path.join(base_path, "ProjectBugReports")
    constructed_query_root = os.path.join(base_path, "ConstructedQueries/BaselineVsReason/")
    search_result_path = os.path.join(base_path, "SearchResults")
    source_codes_root = os.path.join(base_path, "SourceCodes")
    bm25_faiss_dir = os.path.join(base_path, "BM25andFAISS")
    
    # Process all projects
    #projects = ["1","3","5","6","7","8","9","11","12","13","14","18","20","21", "22", "24", "26", "27", "31","33","38","40","42","43", "44","46","47","49","50",
     #           "59","60","62","63", "66","68","69","70","71","76","77","82","77","84","86","87","90","91","97","98","99","101","103","106","107",
     #           "114","116","122","125","128","131","135","138","140","144","146","147","148","149","154","155","156","161","162","165","169","170","176","179",
     #           "181","185","187","188","191","195","197","198","199","201","202","203","207","209","212","223","227","228","232","236","249","259","260","263"]
     #           "265","266","272","279","281","282","288","295","296","299","300","301","306","310","313","324","332","333","335","337","338","342","347","348"]
     #           "351","365","366","371","384","393","401","409","416","422","423","438","442","446","456","463","481","487","492","496","498","508","525","527"]
     #           "528","538","558","582","595","599","614","623","639","643","652","654"]
    projects = ["668","675","681","694","696","699","710","715","736","742","747","760","795"]
    # Clear the log file at the start
    os.makedirs("./logs/parallel_logs", exist_ok=True)
    open("./logs/parallel_logs/NLP_log.txt", "w").close()
    
    async def process_all_projects():
        for project_id in projects:
            print(f"\n=== Processing Project {project_id} with NLP ===")
            
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
            
            await main_async(project_id, bug_reports_root, constructed_query_root, 
                           search_result_path, source_code_dir, bm25_faiss_dir)
            
            print(f"=== Completed Project {project_id} ===")
    
    asyncio.run(process_all_projects())
