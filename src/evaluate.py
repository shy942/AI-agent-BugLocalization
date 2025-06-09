import os
import itertools
from collections import defaultdict

# folder where all projects source codes are contained
source_codes_root = os.path.join(".", "AgentProjectData", "SourceCodes")

# folder where all projects constructed search results are stored
search_results_root = os.path.join(".", "AgentProjectData", "SearchResults")

# folder to store each projects evalution of their respective search results
evaluation_results_root = os.path.join(".", "AgentProjectData", "EvaluationResults")

# Project mapping
project_mapping = {
    "3": "aspnetboilerplate",
    "13": "Atlas", 
    "14": "ARKStatsExtractor",
    "20": "CodenameOne",
    "24": "mobile-wallet"
}

# compute all query evaluators
def compute_evaluation(groundtruth_data, search_data):

    improvement_count = 0
    same_count = 0
    worse_count = 0
    
    bug_reports_affected = []
    bug_reports_missing_groundtruth = []
    
    bug_report_ranks = []
    total_queries = 0
    
    hit_at_k_baseline = {1: 0, 5: 0, 10: 0}
    hit_at_k_extended = {1: 0, 5: 0, 10: 0}
    
    mrr_baseline_sum = 0
    mrr_extended_sum = 0
    
    map_baseline_sum = 0
    map_extended_sum = 0
    
    # iterate over each baseline query, gathering both the baseline and extended queries
    for query, search_results in search_data.items():
    
        query_name, query_type = query
        if query_type != 'baseline':
            continue
        extended_results = search_data[(query_name, 'extended')]
        
        # gather the groundtruth data for comparison against search results
        groundtruth_set, missing_truth_count = groundtruth_data.get(query_name, (set(), 0))
        
        # prevent further calculations if no groundtruth exists
        if not groundtruth_set:
            bug_reports_missing_groundtruth.append(query_name)
            continue
        elif missing_truth_count > 0:
            bug_reports_affected.append(query_name)
        
        # gather the search results for comparison against groundtruth data
        baseline_files = [result.split(',')[0] for result in search_results]
        extended_files = [result.split(',')[0] for result in extended_results]
        
        # compute all baseline and extended ranks
        baseline_ranks = [i + 1 for i, result in enumerate(baseline_files) if result in groundtruth_set]
        extended_ranks = [i + 1 for i, result in enumerate(extended_files) if result in groundtruth_set]

        # Retrieve the first rank if available, otherwise set to None
        baseline_rank = baseline_ranks[0] if baseline_ranks else float('inf')
        extended_rank = extended_ranks[0] if extended_ranks else float('inf')
        
        # store individual ranks (lists of all ranks found)
        bug_report_ranks.append({
            'query_name': query_name,
            'baseline_rank': baseline_ranks if baseline_ranks else None,
            'extended_rank': extended_ranks if extended_ranks else None
        })
        
        # store whether rank improved with the extended query
        if extended_rank < baseline_rank:
            improvement_count += 1
        elif extended_rank == baseline_rank:
            same_count += 1
        else:
            worse_count += 1
        
        # calculate mrr
        if baseline_rank != float('inf'):
            mrr_baseline_sum += 1 / baseline_rank
        if extended_rank != float('inf'):
            mrr_extended_sum += 1 / extended_rank
        
        # Calculate Average Precision (AP) for baseline and extended queries
        def calculate_average_precision(retrieved_files):
            hits = 0
            precision_sum = 0
            for i, file in enumerate(retrieved_files):
                if file in groundtruth_set:
                    hits += 1
                    precision = hits / (i + 1)
                    precision_sum += precision
            return precision_sum / hits if hits > 0 else 0
        
        ap_baseline = calculate_average_precision(baseline_files)
        ap_extended = calculate_average_precision(extended_files)
        
        map_baseline_sum += ap_baseline
        map_extended_sum += ap_extended
        
        # Calculate Hit@K for baseline
        for k in hit_at_k_baseline:
            if any(result in groundtruth_set for result in baseline_files[:k]):
                hit_at_k_baseline[k] += 1
        
        # Calculate Hit@K for extended
        for k in hit_at_k_extended:
            if any(result in groundtruth_set for result in extended_files[:k]):
                hit_at_k_extended[k] += 1
        
        total_queries += 1
    
    # compute k percentages
    hit_at_k_baseline_percent = {
        k: (count / total_queries) * 100 if total_queries != 0 else 0 
        for k, count in hit_at_k_baseline.items()
    }
    hit_at_k_extended_percent = {
        k: (count / total_queries) * 100 if total_queries != 0 else 0 
        for k, count in hit_at_k_extended.items()
    }

    # Calculate final MRR by dividing the sum by the total number of queries
    mrr_baseline = mrr_baseline_sum / total_queries if total_queries > 0 else 0
    mrr_extended = mrr_extended_sum / total_queries if total_queries > 0 else 0
    
    # Calculate final MAP by dividing the sum by the total number of queries
    map_baseline = (map_baseline_sum / total_queries) * 100 if total_queries > 0 else 0
    map_extended = (map_extended_sum / total_queries) * 100 if total_queries > 0 else 0
    
    return {
        'improvement_count': improvement_count,
        'same_count': same_count,
        'worse_count': worse_count,
        'bug_reports_affected': bug_reports_affected,
        'bug_reports_missing_groundtruth': bug_reports_missing_groundtruth,
        'hit_at_k_baseline_percent': hit_at_k_baseline_percent,
        'hit_at_k_extended_percent': hit_at_k_extended_percent,
        'bug_report_ranks': bug_report_ranks,
        'mrr_baseline': mrr_baseline,
        'mrr_extended': mrr_extended,
        'map_baseline': map_baseline,
        'map_extended': map_extended
    }

def generate_possible_paths(dotted_path):
    parts = dotted_path.split('.')
    base_parts, extension = parts[:-1], parts[-1]
    n = len(base_parts)

    all_paths = []
    for combo in itertools.product(['.', os.sep], repeat=n-1):
        path = base_parts[0]
        for sep, part in zip(combo, base_parts[1:]):
            path += sep + part
        path += '.' + extension  # append extension
        all_paths.append(path)
    
    return all_paths

# read and format the groundtruth to a dictionary
def parse_groundtruth(groundtruth_file, groundtruth_found_file, source_code_root, search_data):

    # these are the only bug reports to consider for evaluation
    bug_reports_with_queries = {key[0] for key in search_data.keys()}

    # Read the groundtruthFound file to get valid bug IDs
    valid_bug_ids = set()
    with open(groundtruth_found_file, 'r') as file:
        for line in file:
            bug_id = line.strip()
            if bug_id:
                valid_bug_ids.add(bug_id)

    # datasets to keep track of necessary groundtruth data
    groundtruth_data = {}
    all_groundtruth = set()
    missing_groundtruth = set()
    bugs_all_missing = []
    bugs_some_missing = []
    
    with open(groundtruth_file, 'r') as file:
        while True:
            query_line = file.readline().strip()
            
            # exit if end of file
            if not query_line:
                break
                
            # setup for data retrieval
            query_name, num_lines = query_line.split()
            num_lines = int(num_lines)
            
            # Only process bugs that are in the groundtruthFound file
            if query_name not in valid_bug_ids:
                # Skip this bug and its lines
                for _ in range(num_lines):
                    file.readline()
                continue
                
            groundtruth_entries = set()
            non_existent_count = 0
            
            for _ in range(num_lines):
                line = file.readline().strip()
                
                # Generate all possible paths from the dot notation
                possible_paths = generate_possible_paths(line)
                
                # Check which path actually exists in the source code directory
                found_path = None
                for possible_path in possible_paths:
                    full_path = os.path.join(source_code_root, possible_path)
                    if os.path.exists(full_path):
                        found_path = possible_path
                        break
                
                if found_path:
                    # File exists - normalize the path to match search results format
                    # Convert dots to path separators in path components (except file extension)
                    path_parts = found_path.split('.')
                    if len(path_parts) > 1:
                        # Keep the extension as is, convert dots to path separators in the rest
                        normalized_path = os.sep.join(path_parts[:-1]) + '.' + path_parts[-1]
                    else:
                        normalized_path = found_path
                    
                    groundtruth_entries.add(normalized_path)
                    all_groundtruth.add(os.path.join(source_code_root, found_path))
                else:
                    # No valid path found - mark as missing
                    non_existent_count += 1
                    # Add the original dotted path to missing (for tracking)
                    missing_groundtruth.add(os.path.join(source_code_root, line))
            
            # categorize bugs based on ground truth file existence
            if len(groundtruth_entries) == 0 and num_lines > 0:
                bugs_all_missing.append(query_name)
            elif non_existent_count > 0:
                bugs_some_missing.append(query_name)
            
            # store the formatted data in a dictionary (only for bugs with queries for evaluation)
            if query_name in bug_reports_with_queries:
                groundtruth_data[query_name] = (groundtruth_entries, non_existent_count)
    
    # Total bugs = number of bugs in groundtruthFound file
    total_bugs = len(valid_bug_ids)
    
    # Total considered bugs = total bugs - bugs with all missing groundtruth files
    total_considered_bugs = total_bugs - len(bugs_all_missing)
    
    return groundtruth_data, len(all_groundtruth), len(missing_groundtruth), total_bugs, bugs_all_missing, bugs_some_missing, total_considered_bugs

# read and format the stored query search results to a dictionary
def parse_search_results(search_results_dir, query_type, project_name):
    search_data = {}
    
    # iterate through each bug directory
    for bug_id in os.listdir(search_results_dir):
        bug_dir = os.path.join(search_results_dir, bug_id)
        if not os.path.isdir(bug_dir):
            continue
            
        # load baseline results
        baseline_file = os.path.join(bug_dir, f"{bug_id}_baseline_{query_type}_query_result.txt")
        if os.path.exists(baseline_file):
            baseline_results = []
            with open(baseline_file, 'r') as file:
                for line in file:
                    line = line.strip()
                    if line:
                        parts = line.split(',')
                        if len(parts) >= 2:
                            filename = parts[1].strip()
                            # normalize filename: remove project prefix and convert to path format
                            if filename.startswith(f'{project_name}.'):
                                filename = filename[len(project_name)+1:]
                            # convert remaining dots to path separators except for file extension
                            filename_parts = filename.split('.')
                            if len(filename_parts) > 1:
                                filename = os.path.join(*filename_parts[:-1]) + '.' + filename_parts[-1]
                            baseline_results.append(filename)
            search_data[(bug_id, 'baseline')] = baseline_results
        
        # load extended results  
        extended_file = os.path.join(bug_dir, f"{bug_id}_extended_{query_type}_query_result.txt")
        if os.path.exists(extended_file):
            extended_results = []
            with open(extended_file, 'r') as file:
                for line in file:
                    line = line.strip()
                    if line:
                        parts = line.split(',')
                        if len(parts) >= 2:
                            filename = parts[1].strip()
                            # normalize filename: remove project prefix and convert to path format
                            if filename.startswith(f'{project_name}.'):
                                filename = filename[len(project_name)+1:]
                            # convert remaining dots to path separators except for file extension
                            filename_parts = filename.split('.')
                            if len(filename_parts) > 1:
                                filename = os.path.join(*filename_parts[:-1]) + '.' + filename_parts[-1]
                            extended_results.append(filename)
            search_data[(bug_id, 'extended')] = extended_results
            
    return search_data

def evaluate_project(project_id, project_name):
    """Evaluate a single project and generate evaluation files"""
    
    # query types to evaluate
    query_types = ["basic", "keyBERT", "reasoning"]
    
    source_path = os.path.join(source_codes_root, f"Project{project_id}")
    search_results_dir = os.path.join(search_results_root, project_id)
    
    # find the path to the source code and corpus
    source_corpus = None
    source_code_root = None
    for file in os.listdir(source_path):
        if file.startswith("Corpus"):
            source_corpus = os.path.join(source_path, file)
        elif file == project_name:
            source_code_root = os.path.join(source_path, file)
    
    if not source_corpus or not source_code_root:
        print(f"Error with groundtruth location:{source_corpus} or source code location:{source_code_root}")
        return
    
    # find the path to the groundtruth file
    groundtruth_file = None
    groundtruth_found_file = None
    for file in os.listdir(source_corpus):
        if file.startswith("groundtruth_"):
            groundtruth_file = file
        elif file.startswith("groundtruthFound_"):
            groundtruth_found_file = file
    
    if not groundtruth_file:
        print("Error: no ground truth file found")
        return
    
    if not groundtruth_found_file:
        print("Error: no groundtruthFound file found")
        return
    
    groundtruth_path = os.path.join(source_corpus, groundtruth_file)
    groundtruth_found_path = os.path.join(source_corpus, groundtruth_found_file)
    
    # Create project-specific evaluation directory
    project_evaluation_dir = os.path.join(evaluation_results_root, f"Project{project_id}")
    os.makedirs(project_evaluation_dir, exist_ok=True)
    
    # evaluate each query type
    for query_type in query_types:
        print(f"Starting evaluation for Project {project_id} - {query_type}")
        
        # gather the search results data
        search_data = parse_search_results(search_results_dir, query_type, project_name)
        print(f"Found {len(search_data)} search results")
        
        # gather the groundtruth data
        groundtruth_data, total_groundtruth_count, missing_groundtruth_count, total_bugs, bugs_all_missing, bugs_some_missing, total_considered_bugs = parse_groundtruth(groundtruth_path, groundtruth_found_path, source_code_root, search_data)
        print(f"Found {len(groundtruth_data)} groundtruth entries")
        
        # compute all query evaluators
        data = compute_evaluation(groundtruth_data, search_data)
        
        # save search results
        bug_reports_considered_count = len(data['bug_report_ranks'])
        
        storage_path = os.path.join(project_evaluation_dir, f"evaluation_{query_type.lower()}.txt")
        
        with open(storage_path, 'w') as file:
            file.write(f"Project {project_id} ({project_name}):\n\n")
            file.write(f"Total number of groundtruth files: {total_groundtruth_count}\n")
            file.write(f"Total number of bugs: {total_bugs}\n")
            file.write(f"Total amount of groundtruth files not found in source code: {missing_groundtruth_count}\n")
            
            file.write(f"Total number of Bug reports where all groundtruth files do not exist: {len(bugs_all_missing)}\n")
            file.write(f"Bug reports where all groundtruth files do not exist: {bugs_all_missing}\n")
            
            file.write(f"Total number of bug reports where some groundtruth files were missing: {len(bugs_some_missing)}\n")
            file.write(f"Bug reports where some groundtruth files were missing: {bugs_some_missing}\n")
            
            file.write(f"Total number of considered bugs: {total_considered_bugs}\n")
            
            file.write(f"\nQE Improved Count: {data['improvement_count']}\n")
            file.write(f"QE Identical Count: {data['same_count']}\n")
            file.write(f"QE Worse Count: {data['worse_count']}\n")
    
            file.write(f"\nHit@K for baseline queries:\n")
            for k, percentage in data['hit_at_k_baseline_percent'].items():
                file.write(f"Hit@{k}: {percentage:.2f}%\n")
        
            file.write(f"\nHit@K for extended queries:\n")
            for k, percentage in data['hit_at_k_extended_percent'].items():
                file.write(f"Hit@{k}: {percentage:.2f}%\n")
                
            file.write(f"\nMRR baseline queries: {data['mrr_baseline']}\n")
            file.write(f"MRR extended queries: {data['mrr_extended']}\n")
            
            file.write(f"\nMAP baseline queries: {data['map_baseline']}\n")
            file.write(f"MAP extended queries: {data['map_extended']}\n")
        
            file.write("\nIndividual Results:\n")
            for rank_info in data['bug_report_ranks']:
                file.write(f"{rank_info['query_name']}, 'Baseline', {rank_info['baseline_rank']}\n")
                file.write(f"{rank_info['query_name']}, 'Extended', {rank_info['extended_rank']}\n")
                
        print(f"Stored evaluation for Project {project_id} - {query_type} to {storage_path}")

def main():
    """Main function to evaluate all projects"""
    
    print("Starting evaluation for all projects in AgentProjectData...")
    
    # Process all projects
    for project_id, project_name in project_mapping.items():
        print(f"\n=== Evaluating Project {project_id} ({project_name}) ===")
        evaluate_project(project_id, project_name)
        print(f"=== Completed Project {project_id} ===")
    
    print(f"\nAll evaluations completed! Results stored in {evaluation_results_root}")

if __name__ == "__main__":
    main() 