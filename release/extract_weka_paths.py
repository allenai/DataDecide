#!/usr/bin/env python3
"""
Extract local paths to model revisions from weka_paths.jsonl for specified repositories.

This script reads the weka_paths.jsonl file and repo_names.txt file, then generates a 
line-separated file containing local paths for model revisions that correspond to the 
specified repository names. Uses the same model name to repo name mapping logic as 
the upload script.

Usage:
    python extract_weka_paths.py [--input INPUT_FILE] [--output OUTPUT_FILE] [--repo_names REPO_NAMES_FILE]

Arguments:
    --input       Path to input weka_paths.jsonl file (default: ../checkpoints/weka_paths.jsonl)
    --output      Path to output file with extracted paths (default: ./filtered_weka_paths.txt)
    --repo_names  Path to repo_names.txt file (default: ./repo_names.txt)
    --help        Show this help message and exit

Example:
    python extract_weka_paths.py
    python extract_weka_paths.py --input /path/to/weka_paths.jsonl --output /path/to/output.txt
"""

import json
import argparse
import os
import sys
from pathlib import Path


# Mapping for model names to repo names (copied from upload_checkpoints.py)
recipe_display_name = {
    "dolma17": "dolma1_7",
    "no_code": "dolma1_7-no-code", 
    "no_math_no_code": "dolma1_7-no-math-code",
    "no_reddit": "dolma1_7-no-reddit",
    "no_flan": "dolma1_7-no-flan",
    "dolma-v1-6-and-sources-baseline": "dolma1_6plus",
    "c4": "c4",
    "prox_fineweb_pro": "fineweb-pro",
    "fineweb_edu_dedup": "fineweb-edu", 
    "falcon": "falcon",
    "falcon_and_cc": "falcon-and-cc",
    "falcon_and_cc_eli5_oh_top10p": "falcon-and-cc-qc-10p",
    "falcon_and_cc_eli5_oh_top20p": "falcon-and-cc-qc-20p",
    "falcon_and_cc_og_eli5_oh_top10p": "falcon-and-cc-qc-orig-10p",
    "falcon_and_cc_tulu_qc_top10": "falcon-and-cc-qc-tulu-10p",
    "DCLM-baseline": "dclm-baseline",
    "dolma17-75p-DCLM-baseline-25p": "dclm-baseline-25p-dolma1.7-75p",
    "dolma17-50p-DCLM-baseline-50p": "dclm-baseline-50p-dolma1.7-50p", 
    "dolma17-25p-DCLM-baseline-75p": "dclm-baseline-75p-dolma1.7-25p",
    "dclm_ft7percentile_fw2": "dclm-baseline-qc-7p-fw2",
    "dclm_ft7percentile_fw3": "dclm-baseline-qc-7p-fw3",
    "dclm_fw_top10": "dclm-baseline-qc-fw-10p",
    "dclm_fw_top3": "dclm-baseline-qc-fw-3p",
    "pos_eli5_oh_neg_dclm_refinedweb_steps_2000_lr3e4_top10p": "dclm-baseline-qc-10p",
    "pos_eli5_oh_neg_dclm_refinedweb_steps_2000_lr3e4_top20p": "dclm-baseline-qc-20p",
}


def model_name_to_repo_name(model_name):
    """
    Convert model name to repo name using the same logic as upload_checkpoints.py
    
    Args:
        model_name (str): The model name from weka_paths.jsonl
        
    Returns:
        str or None: The repo name or None if it should be filtered out
    """
    seed = int(model_name.split('-')[-1])
    if seed not in [6198, 4, 5, 14, 15]:
        return None
    
    size = model_name.split('-')[-2]
    if not (size.endswith('M') or size.endswith('B')):
        return None
    
    recipe = '-'.join(model_name.split('-')[:-2])
    if recipe == 'baseline':
        recipe = 'dolma17'
    
    # Filter out specific recipes that return None in upload script
    if recipe in ['DCLM-baseline-25p', 'DCLM-baseline-50p', 'DCLM-baseline-75p']:
        return None
    
    if recipe not in recipe_display_name:
        return None
    
    recipe = recipe_display_name[recipe]
    return f"DataDecide-{recipe}-{size}"


def load_repo_names(repo_names_file):
    """
    Load repository names from file.
    
    Args:
        repo_names_file (str): Path to repo_names.txt file
        
    Returns:
        set: Set of repository names to filter by
    """
    if not os.path.exists(repo_names_file):
        print(f"Error: Repo names file {repo_names_file} does not exist.", file=sys.stderr)
        sys.exit(1)
    
    repo_names = set()
    try:
        with open(repo_names_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    repo_names.add(line)
        
        print(f"Loaded {len(repo_names)} repository names from {repo_names_file}")
        return repo_names
    
    except Exception as e:
        print(f"Error reading repo names file {repo_names_file}: {e}", file=sys.stderr)
        sys.exit(1)


def extract_paths(input_file, output_file, repo_names_file=None):
    """
    Extract local paths from weka_paths.jsonl, optionally filtering by repository names.
    
    Args:
        input_file (str): Path to input weka_paths.jsonl file
        output_file (str): Path to output file for extracted paths
        repo_names_file (str, optional): Path to repo_names.txt file for filtering
    
    Returns:
        int: Number of paths extracted
    """
    if not os.path.exists(input_file):
        print(f"Error: Input file {input_file} does not exist.", file=sys.stderr)
        sys.exit(1)
    
    # Load repository names for filtering if provided
    target_repo_names = None
    if repo_names_file:
        target_repo_names = load_repo_names(repo_names_file)
    
    paths = []
    skipped_models = []
    filtered_models = []
    processed_models = []
    
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    data = json.loads(line)
                    
                    # Validate required fields
                    if 'checkpoints_location' not in data or 'revisions' not in data or 'model_name' not in data:
                        print(f"Warning: Line {line_num} missing required fields, skipping.", file=sys.stderr)
                        continue
                    
                    model_name = data['model_name']
                    base_path = data['checkpoints_location']
                    revisions = data['revisions']
                    
                    # Convert model name to repo name if filtering is enabled
                    if target_repo_names:
                        repo_name = model_name_to_repo_name(model_name)
                        
                        if repo_name is None:
                            skipped_models.append(model_name)
                            continue
                        
                        if repo_name not in target_repo_names:
                            filtered_models.append((model_name, repo_name))
                            continue
                        
                        processed_models.append((model_name, repo_name))
                    
                    # Generate full paths for each revision
                    for revision in revisions:
                        full_path = f"{base_path}/{revision}"
                        paths.append(full_path)
                
                except json.JSONDecodeError as e:
                    print(f"Warning: Line {line_num} is not valid JSON: {e}, skipping.", file=sys.stderr)
                    continue
    
    except Exception as e:
        print(f"Error reading input file {input_file}: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Write paths to output file
    try:
        # Ensure output directory exists
        output_dir = os.path.dirname(os.path.abspath(output_file))
        os.makedirs(output_dir, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for path in paths:
                f.write(f"{path}\n")
        
        print(f"Successfully extracted {len(paths)} paths to {output_file}")
        if target_repo_names:
            print(f"\nFiltering Summary:")
            print(f"  - Processed models: {len(processed_models)}")
            print(f"  - Skipped models (invalid/filtered recipes): {len(skipped_models)}")
            print(f"  - Filtered out models (not in target repo names): {len(filtered_models)}")
            
            if skipped_models:
                print(f"\nSkipped models details:")
                for model in sorted(skipped_models):
                    print(f"  - {model}")
            
            if filtered_models:
                print(f"\nFiltered out models details (repo not in target list):")
                for model, repo in sorted(filtered_models):
                    print(f"  - {model} -> {repo}")
        
        return len(paths)
    
    except Exception as e:
        print(f"Error writing to output file {output_file}: {e}", file=sys.stderr)
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Extract local paths to model revisions from weka_paths.jsonl, optionally filtering by repo names",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split('Usage:')[1].split('Example:')[0] + "\nExample:\n" + __doc__.split('Example:')[1]
    )
    
    # Default paths relative to script location
    script_dir = Path(__file__).parent
    default_input = script_dir / "../checkpoints/weka_paths.jsonl"
    default_output = script_dir / "filtered_weka_paths.txt"
    default_repo_names = script_dir / "repo_names.txt"
    
    parser.add_argument(
        '--input',
        default=str(default_input),
        help=f"Path to input weka_paths.jsonl file (default: {default_input})"
    )
    
    parser.add_argument(
        '--output',
        default=str(default_output),
        help=f"Path to output file with extracted paths (default: {default_output})"
    )
    
    parser.add_argument(
        '--repo_names',
        default=str(default_repo_names),
        help=f"Path to repo_names.txt file for filtering (default: {default_repo_names}). Use 'none' to disable filtering."
    )
    
    args = parser.parse_args()
    
    print(f"Reading from: {args.input}")
    print(f"Writing to: {args.output}")
    
    # Handle repo names filtering
    repo_names_file = None
    if args.repo_names.lower() != 'none':
        repo_names_file = args.repo_names
        print(f"Filtering by repo names from: {repo_names_file}")
    else:
        print("No filtering by repo names (extracting all paths)")
    
    num_paths = extract_paths(args.input, args.output, repo_names_file)
    
    print(f"Extraction complete! Found {num_paths} total revision paths.")


if __name__ == "__main__":
    main()
