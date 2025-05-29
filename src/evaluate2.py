import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Set
import argparse


def load_ground_truth(ground_truth_file: str) -> Dict[str, Set[str]]:
    """
    Load ground truth data from file.
    
    Format expected:
    bug_id count
    file1.php
    file2.php
    ...
    
    Returns:
        Dictionary mapping bug_id to set of relevant files
    """
    ground_truth = {}
    
    if not os.path.exists(ground_truth_file):
        print(f"Warning: Ground truth file not found: {ground_truth_file}")
        return ground_truth
    
    with open(ground_truth_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue
            
        # Parse bug_id and count
        parts = line.split()
        if len(parts) != 2:
            i += 1
            continue
            
        bug_id = parts[0]
        try:
            count = int(parts[1])
        except ValueError:
            i += 1
            continue
        
        # Collect the relevant files for this bug
        relevant_files = set()
        for j in range(i + 1, min(i + 1 + count, len(lines))):
            file_path = lines[j].strip()
            if file_path:
                # Normalize the file path format
                normalized_path = normalize_file_path(file_path)
                relevant_files.add(normalized_path)
        
        ground_truth[bug_id] = relevant_files
        i += 1 + count
    
    return ground_truth


def normalize_file_path(file_path: str) -> str:
    """
    Normalize file path to match the format used in search results.
    
    Convert from ground truth format (e.g., "src.app.Services.Data.php") 
    to search result format (e.g., "src/app/Services/Data.php")
    """
    # Remove leading/trailing whitespace
    path = file_path.strip()
    
    # Convert dot notation to path format
    parts = path.split('.')
    if len(parts) > 1:
        # Join all parts except the last one with '/', then add the last part with its extension
        path = os.path.join(*parts[:-1]) + '.' + parts[-1]
    
    return path


def load_search_results(result_file: str) -> List[str]:
    """
    Load search results from file.
    
    Format expected:
    rank,filename,score
    1,tables.src.Services.Data.php,0.856
    2,tables.src.Template.Builder.php,0.743
    ...
    
    Returns:
        List of filenames in ranked order (normalized to match ground truth format)
    """
    results = []
    
    if not os.path.exists(result_file):
        print(f"Warning: Result file not found: {result_file}")
        return results
    
    with open(result_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            parts = line.split(',')
            if len(parts) >= 2:
                filename = parts[1].strip()
                # Normalize filename: remove 'tables.' prefix and convert to path format
                if filename.startswith('tables.'):
                    filename = filename[7:]  # Remove 'tables.' prefix
                # Convert remaining dots to path separators except for file extension
                filename_parts = filename.split('.')
                if len(filename_parts) > 1:
                    # Join all parts except the last one with '/', then add the last part with extension
                    filename = os.path.join(*filename_parts[:-1]) + '.' + filename_parts[-1]
                
                results.append(filename)
    
    return results


def check_source_code_existence(source_codes_root: str, ground_truth: Dict[str, Set[str]]) -> Tuple[Dict[str, Set[str]], Dict[str, Set[str]], List[str], List[str]]:
    """
    Check which ground truth files exist in the source code and categorize bugs.
    
    Returns:
        - existing_files: mapping bug_id to set of existing files
        - missing_files: mapping bug_id to set of missing files  
        - bugs_all_missing: list of bug IDs where all ground truth files are missing
        - bugs_some_missing: list of bug IDs where some ground truth files are missing
    """
    existing_files = {}
    missing_files = {}
    bugs_all_missing = []
    bugs_some_missing = []
    
    # Find the actual source code directory (should be under tables/)
    source_code_dir = None
    tables_path = os.path.join(source_codes_root, "tables")
    if os.path.exists(tables_path):
        source_code_dir = tables_path
    else:
        source_code_dir = source_codes_root
    
    for bug_id, gt_files in ground_truth.items():
        existing = set()
        missing = set()
        
        for gt_file in gt_files:
            full_path = os.path.join(source_code_dir, gt_file)
            if os.path.exists(full_path):
                existing.add(gt_file)
            else:
                missing.add(gt_file)
        
        existing_files[bug_id] = existing
        missing_files[bug_id] = missing
        
        if len(existing) == 0 and len(gt_files) > 0:
            bugs_all_missing.append(bug_id)
        elif len(missing) > 0:
            bugs_some_missing.append(bug_id)
    
    return existing_files, missing_files, bugs_all_missing, bugs_some_missing


def compute_evaluation(ground_truth_data: Dict[str, Set[str]], search_data: Dict[Tuple[str, str], List[str]]) -> Dict:
    """
    Compute evaluation metrics following the reference implementation method.
    """
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
            
        extended_key = (query_name, 'extended')
        if extended_key not in search_data:
            continue
            
        extended_results = search_data[extended_key]
        
        # gather the groundtruth data for comparison against search results
        groundtruth_set = ground_truth_data.get(query_name, set())
        
        # prevent further calculations if no groundtruth exists
        if not groundtruth_set:
            bug_reports_missing_groundtruth.append(query_name)
            continue
        
        # gather the search results for comparison against groundtruth data
        baseline_files = search_results
        extended_files = extended_results
        
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


def evaluate_query_type(search_results_dir: str, ground_truth: Dict[str, Set[str]], 
                       existing_files: Dict[str, Set[str]], query_type: str) -> Dict:
    """
    Evaluate a specific query type for both baseline and extended variants.
    
    Args:
        search_results_dir: Directory containing search results
        ground_truth: Ground truth data
        existing_files: Files that exist in source code
        query_type: Type of query (basic, keyBERT, reasoning)
    
    Returns:
        Dictionary containing evaluation results
    """
    # Build search data dictionary
    search_data = {}
    
    # Only consider bugs where some ground truth files exist
    considered_bugs = [bug_id for bug_id, files in existing_files.items() if len(files) > 0]
    
    for bug_id in considered_bugs:
        # Load baseline results
        baseline_file = os.path.join(search_results_dir, bug_id, f"{bug_id}_baseline_{query_type}_query_result.txt")
        baseline_results = load_search_results(baseline_file)
        
        # Load extended results
        extended_file = os.path.join(search_results_dir, bug_id, f"{bug_id}_extended_{query_type}_query_result.txt")
        extended_results = load_search_results(extended_file)
        
        if baseline_results and extended_results:
            search_data[(bug_id, 'baseline')] = baseline_results
            search_data[(bug_id, 'extended')] = extended_results
    
    # Use existing_files as ground truth data (only files that exist)
    ground_truth_data = existing_files
    
    # Compute evaluation using the reference method
    evaluation_results = compute_evaluation(ground_truth_data, search_data)
    
    # Format results to match expected output structure
    results = {
        'baseline': {
            'hit1': evaluation_results['hit_at_k_baseline_percent'][1],
            'hit5': evaluation_results['hit_at_k_baseline_percent'][5],
            'hit10': evaluation_results['hit_at_k_baseline_percent'][10],
            'mrr': evaluation_results['mrr_baseline'],
            'map': evaluation_results['map_baseline']
        },
        'extended': {
            'hit1': evaluation_results['hit_at_k_extended_percent'][1],
            'hit5': evaluation_results['hit_at_k_extended_percent'][5],
            'hit10': evaluation_results['hit_at_k_extended_percent'][10],
            'mrr': evaluation_results['mrr_extended'],
            'map': evaluation_results['map_extended']
        },
        'qe_stats': {
            'improved': evaluation_results['improvement_count'],
            'identical': evaluation_results['same_count'],
            'worse': evaluation_results['worse_count']
        },
        'considered_bugs': len(evaluation_results['bug_report_ranks']),
        'individual_results': []
    }
    
    # Format individual results
    for rank_info in evaluation_results['bug_report_ranks']:
        results['individual_results'].append((rank_info['query_name'], 'Baseline', rank_info['baseline_rank']))
        results['individual_results'].append((rank_info['query_name'], 'Extended', rank_info['extended_rank']))
    
    return results


def save_evaluation_results(ground_truth: Dict[str, Set[str]], existing_files: Dict[str, Set[str]], 
                           missing_files: Dict[str, Set[str]], bugs_all_missing: List[str], 
                           bugs_some_missing: List[str], results: Dict, query_type: str, 
                           output_file: str):
    """
    Save evaluation results to file in the specified format.
    """
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Calculate statistics
    total_gt_files = sum(len(files) for files in ground_truth.values())
    total_bugs = len(ground_truth)
    total_missing_files = sum(len(files) for files in missing_files.values())
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("Project 103:\n\n")
        
        f.write(f"Total number of groundtruth files: {total_gt_files}\n")
        f.write(f"Total number of bugs: {total_bugs}\n")
        f.write(f"Total amount of groundtruth files not found in source code: {total_missing_files}\n")
        f.write(f"Total number of Bug reports where all groundtruth files do not exist: {len(bugs_all_missing)}\n")
        f.write(f"Bug reports where all groundtruth files do not exist: {bugs_all_missing}\n")
        f.write(f"Total number of bug reports where some groundtruth files were missing: {len(bugs_some_missing)}\n")
        f.write(f"Bug reports where some groundtruth files were missing: {bugs_some_missing}\n")
        f.write(f"Total number of considered bugs: {results['considered_bugs']}\n\n")
        
        f.write(f"QE Improved Count: {results['qe_stats']['improved']}\n")
        f.write(f"QE Identical Count: {results['qe_stats']['identical']}\n")
        f.write(f"QE Worse Count: {results['qe_stats']['worse']}\n\n")
        
        f.write("Hit@K for baseline queries:\n")
        f.write(f"Hit@1: {results['baseline']['hit1']:.2f}%\n")
        f.write(f"Hit@5: {results['baseline']['hit5']:.2f}%\n")
        f.write(f"Hit@10: {results['baseline']['hit10']:.2f}%\n\n")
        
        f.write("Hit@K for extended queries:\n")
        f.write(f"Hit@1: {results['extended']['hit1']:.2f}%\n")
        f.write(f"Hit@5: {results['extended']['hit5']:.2f}%\n")
        f.write(f"Hit@10: {results['extended']['hit10']:.2f}%\n\n")
        
        f.write(f"MRR baseline queries: {results['baseline']['mrr']:.16f}\n")
        f.write(f"MRR extended queries: {results['extended']['mrr']:.16f}\n\n")
        
        f.write(f"MAP baseline queries: {results['baseline']['map']:.14f}\n")
        f.write(f"MAP extended queries: {results['extended']['map']:.14f}\n\n")
        
        f.write("Individual Results:\n")
        for bug_id, variant, positions in results['individual_results']:
            f.write(f"{bug_id}, '{variant}', {positions}\n")


def main():
    """Main evaluation function."""
    
    # Configuration
    project_id = "103"  # Default project ID, can be made configurable
    source_codes_root = "./ExampleProjectData/SourceCodes/Project103"
    search_results_root = f"./ExampleProjectData/SearchResults/{project_id}"
    evaluation_results_root = "./ExampleProjectData/EvaluationResults"
    
    # Ground truth file
    ground_truth_file = os.path.join(source_codes_root, "Corpus", "groundtruth_tables.txt")
    
    # Query types to evaluate
    query_types = ["basic", "keyBERT", "reasoning"]
    
    print("Loading ground truth data...")
    ground_truth = load_ground_truth(ground_truth_file)
    print(f"Loaded ground truth for {len(ground_truth)} bugs")
    
    if not ground_truth:
        print("Error: No ground truth data found. Exiting.")
        sys.exit(1)
    
    print("Checking source code existence...")
    existing_files, missing_files, bugs_all_missing, bugs_some_missing = check_source_code_existence(
        source_codes_root, ground_truth
    )
    
    # Evaluate each query type
    for query_type in query_types:
        print(f"\nEvaluating {query_type} queries...")
        
        # Evaluate this query type
        results = evaluate_query_type(
            search_results_root, ground_truth, existing_files, query_type
        )
        
        # Save results
        output_file = os.path.join(
            evaluation_results_root, 
            f"evaluation_{query_type.lower()}.txt"
        )
        
        save_evaluation_results(
            ground_truth, existing_files, missing_files, 
            bugs_all_missing, bugs_some_missing, results, 
            query_type, output_file
        )
        
        print(f"Results saved to: {output_file}")
    
    print("\nEvaluation completed successfully!")
    print(f"All results saved in: {evaluation_results_root}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate bug localization performance")
    parser.add_argument("--project-id", default="103", help="Project ID to evaluate")
    parser.add_argument("--source-root", default="./ExampleProjectData/SourceCodes/Project103", 
                       help="Source codes root directory")
    parser.add_argument("--results-root", help="Search results root directory")
    parser.add_argument("--output-root", default="./ExampleProjectData/EvaluationResults",
                       help="Evaluation results output directory")
    
    args = parser.parse_args()
    
    # Update configuration if provided
    if args.results_root:
        search_results_root = args.results_root
    else:
        search_results_root = f"./ExampleProjectData/SearchResults/{args.project_id}"
    
    main()
















import os

# folder where all projects source codes are contained
source_codes_root = "./ExampleProjectData/SourceCodes"

# folder where all projects constructed search results are stored
search_results_root = "./ExampleProjectData/SearchResults"

# folder to store each projects evalution of their respective search results
evaluation_results_root = "./ExampleProjectData/EvaluationResults"

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

# read and format the groundtruth to a dictionary
def parse_groundtruth(groundtruth_file, source_code_root, search_data):

    # these are the only bug reports to consider
    bug_reports = {key[0] for key in search_data.keys()}

    # datasets to keep track of necessary groundtruth data
    groundtruth_data = {}
    all_groundtruth = set()
    missing_groundtruth = set()
    
    with open(groundtruth_file, 'r') as file:
        while True:
            query_line = file.readline().strip()
            
            # exit if end of file
            if not query_line:
                break
                
            # setup for data retrieval
            query_name, num_lines = query_line.split()
            num_lines = int(num_lines)
            groundtruth_entries = set()
            non_existent_count = 0
            
            for _ in range(num_lines):
                line = file.readline().strip()
                
                # skip groundtruth data corresponding to non-existant bug reports
                if not query_name in bug_reports:
                    continue
                    
                # format each path for comparison to search results
                parts = line.split('.')
                if len(parts) > 1:
                    line = os.path.join(*parts[:-1]) + '.' + parts[-1]

                # track whether the path actually exists
                full_path = os.path.join(source_code_root, line)
                all_groundtruth.add(full_path)
                if os.path.exists(full_path):
                    groundtruth_entries.add(line)
                else:
                    non_existent_count += 1
                    missing_groundtruth.add(full_path)
            
            # store the formatted data in a dictionary
            if query_name in bug_reports:
                groundtruth_data[query_name] = (groundtruth_entries, non_existent_count)
    
    return groundtruth_data, len(all_groundtruth), len(missing_groundtruth)

# read and format the stored query search results to a dictionary
def parse_search_results(search_results_dir, query_type):
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
                            # normalize filename: remove 'tables.' prefix and convert to path format
                            if filename.startswith('tables.'):
                                filename = filename[7:]
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
                            # normalize filename: remove 'tables.' prefix and convert to path format
                            if filename.startswith('tables.'):
                                filename = filename[7:]
                            # convert remaining dots to path separators except for file extension
                            filename_parts = filename.split('.')
                            if len(filename_parts) > 1:
                                filename = os.path.join(*filename_parts[:-1]) + '.' + filename_parts[-1]
                            extended_results.append(filename)
            search_data[(bug_id, 'extended')] = extended_results
            
    return search_data

def main():
    
    # query types to evaluate
    query_types = ["basic", "keyBERT", "reasoning"]
    project = "103"
    
    source_path = os.path.join(source_codes_root, f"Project{project}")
    search_results_dir = os.path.join(search_results_root, project)
    
    # find the path to the source code
    source_corpus = None
    source_code_root = None
    for file in os.listdir(source_path):
        if file.startswith("Corpus"):
            source_corpus = os.path.join(source_path, file)
        elif file == "tables":
            source_code_root = os.path.join(source_path, file)
    
    if not source_corpus or not source_code_root:
        print(f"Error with groundtruth location:{source_corpus} or source code location:{source_code_root}")
        return
    
    # find the path to the groundtruth file
    groundtruth_file = None
    for file in os.listdir(source_corpus):
        if file.startswith("groundtruth_"):
            groundtruth_file = file
            break
    
    if not groundtruth_file:
        print("Error: no ground truth file found")
        return
    
    groundtruth_path = os.path.join(source_corpus, groundtruth_file)
    
    # evaluate each query type
    for query_type in query_types:
        print(f"starting project {project}_{query_type}")
        
        # gather the search results data
        search_data = parse_search_results(search_results_dir, query_type)
        
        # gather the groundtruth data
        groundtruth_data, total_groundtruth_count, missing_groundtruth_count = parse_groundtruth(groundtruth_path, source_code_root, search_data)
        
        # compute all query evaluators
        data = compute_evaluation(groundtruth_data, search_data)
        
        # save search results
        bug_reports_considered_count = len(data['bug_report_ranks'])
        bug_reports_missing_count = len(data['bug_reports_missing_groundtruth'])
        bug_report_count = bug_reports_considered_count + bug_reports_missing_count
        
        storage_path = os.path.join(evaluation_results_root, f"evaluation_{query_type.lower()}.txt")
        os.makedirs(evaluation_results_root, exist_ok=True)
        
        with open(storage_path, 'w') as file:
            file.write(f"Project {project}:\n\n")
            file.write(f"Total number of groundtruth files: {total_groundtruth_count}\n")
            file.write(f"Total number of bugs: {bug_report_count}\n")
            file.write(f"Total amount of groundtruth files not found in source code: {missing_groundtruth_count}\n")
            
            file.write(f"Total number of Bug reports where all groundtruth files do not exist: {bug_reports_missing_count}\n")
            file.write(f"Bug reports where all groundtruth files do not exist: {data['bug_reports_missing_groundtruth']}\n")
            
            file.write(f"Total number of bug reports where some groundtruth files were missing: {len(data['bug_reports_affected'])}\n")
            file.write(f"Bug reports where some groundtruth files were missing: {data['bug_reports_affected']}\n")
            
            file.write(f"Total number of considered bugs: {bug_reports_considered_count}\n")
            
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
                
        print(f"stored evaluation for project {project}_{query_type} to {storage_path}")

if __name__ == "__main__":
    main() 