#!/usr/bin/env python3
"""
Script to verify that optimizer states (optim.pt) have been uploaded to Hugging Face repositories.

This script verifies:
1. If each repository in repo_names.txt exists
2. If each expected branch has the optim.pt file at training/optim.pt
3. Samples revisions evenly based on the same logic as upload_optim.py

IMPORTANT: This script ONLY uses metadata APIs and does NOT download any model files or data.
It only checks repository existence, branch existence, and file metadata.

Usage examples:
    # Check all repositories
    python verify_optim.py

    # Check only repositories containing 'falcon'
    python verify_optim.py --repo_filter falcon

    # Check only 60M models
    python verify_optim.py --size_filter 60M

    # Check a specific repository with detailed output
    python verify_optim.py --single_repo DataDecide-falcon-60M --detailed

    # Check with different number of revisions (default is 5)
    python verify_optim.py --num_revisions 3

    # Check with different organization
    python verify_optim.py --org_name myorg

    # Use longer delays to avoid rate limiting (default is 0.2s)
    python verify_optim.py --delay 0.5
"""

import os
import json
import argparse
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import List, Dict, Set, Optional
# METADATA-ONLY imports - these do NOT download model files
from huggingface_hub import HfApi
try:
    from huggingface_hub.errors import RepositoryNotFoundError, RevisionNotFoundError
except ImportError:
    try:
        from huggingface_hub.utils._errors import RepositoryNotFoundError, RevisionNotFoundError
    except ImportError:
        # Fallback for older versions
        RepositoryNotFoundError = Exception
        RevisionNotFoundError = Exception
import time
from tqdm import tqdm
import random

# Constants
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_NAMES_FILE = os.path.join(SCRIPT_DIR, "repo_names.txt")
CHECKPOINTS_FILE = os.path.join(SCRIPT_DIR, "../checkpoints/weka_paths.jsonl")

SEED_MAPPING = {
    6198: "default",
    14: "small-aux-2",
    15: "small-aux-3",
    4: "large-aux-2",
    5: "large-aux-3"
}

FULL_SCHEDULE_LAST_STEP_PER_MODEL = {
    "4M": 5725,  # min step value from {5745, 5725, 5735}
    "6M": 9182,
    "8M": 13039,
    "10M": 15117,
    "14M": 21953,
    "16M": 24432,
    "20M": 14584,  # min step value from {14584, 14594}
    "60M": 29042,  # min step value from {29042, 29052, 29062}
    "90M": 29901,
    "150M": 38157,
    "300M": 45787,
    "530M": 57786,
    "750M": 63589,
    "1B": 69369,
}

# have to round up for the models that ran to long to make sure we get a checkpoint after the LR fully decays
def round_up(value, increment):
    return (value + increment - 1) // increment * increment

for model, step in FULL_SCHEDULE_LAST_STEP_PER_MODEL.items():
    if model == '1B':
        FULL_SCHEDULE_LAST_STEP_PER_MODEL[model] = round_up(step, 2500)
    else:
        FULL_SCHEDULE_LAST_STEP_PER_MODEL[model] = round_up(step, 1250)

# Mapping for model names to repo names (copied from upload_optim.py)
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

@dataclass
class OptimStatus:
    branch_name: str
    branch_exists: bool
    optim_exists: bool
    file_size: Optional[int] = None
    error: Optional[str] = None

@dataclass
class RepoOptimStatus:
    name: str
    repo_exists: bool
    expected_branches: Set[str]
    optim_statuses: List[OptimStatus]
    error: Optional[str] = None

def retry_with_backoff(max_retries=3, base_delay=1.0):
    """Decorator to retry API calls with exponential backoff."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    error_str = str(e).lower()
                    # Check if it's a retryable error (rate limiting, server errors)
                    if any(x in error_str for x in ['503', '429', '502', '504', 'rate limit', 'service temporarily unavailable']):
                        if attempt < max_retries - 1:
                            delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                            print(f"  Retrying in {delay:.1f}s due to: {e}")
                            time.sleep(delay)
                            continue
                    # Re-raise the exception if it's not retryable or we've exhausted retries
                    raise e
            return None
        return wrapper
    return decorator

def model_name_to_repo_name(model_name):
    """Convert model name from weka_paths.jsonl to repo name format."""
    seed = int(model_name.split('-')[-1])
    assert seed in [6198, 4, 5, 14, 15], f"Unknown seed {seed}"
    size = model_name.split('-')[-2]
    assert size.endswith('M') or size.endswith('B'), f"Unknown size {size}"
    recipe = '-'.join(model_name.split('-')[:-2])
    if recipe == 'baseline':
        recipe = 'dolma17'
    if recipe in ['DCLM-baseline-25p', 'DCLM-baseline-50p', 'DCLM-baseline-75p']:
        return None
    assert recipe in recipe_display_name, f"Unknown recipe {recipe}"
    recipe = recipe_display_name[recipe]
    
    return f"DataDecide-{recipe}-{size}"

def extract_step_number(revision):
    """Extract step number from revision string like 'step1250-unsharded-hf'."""
    match = re.search(r'step(\d+)', revision)
    if match:
        return int(match.group(1))
    return 0

def extract_model_size(model_name):
    """Extract model size from model name like 'falcon_and_cc_tulu_qc_top10-60M-15'."""
    parts = model_name.split('-')
    # The size is always the second-to-last part in the format: {recipe}-{size}-{seed}
    if len(parts) >= 2:
        size_candidate = parts[-2]
        if size_candidate.endswith('M') or size_candidate.endswith('B'):
            return size_candidate
    
    # Fallback: search for any part that looks like a size
    for part in parts:
        if part.endswith('M') or part.endswith('B'):
            return part
    return None

def filter_revisions_by_schedule(revisions, model_name):
    """
    Filter revisions to only include steps within the training schedule.
    
    Args:
        revisions: List of revision strings
        model_name: Model name to extract size from
    
    Returns:
        List of filtered revision strings
    """
    model_size = extract_model_size(model_name)
    if model_size not in FULL_SCHEDULE_LAST_STEP_PER_MODEL:
        print(f"Warning: Model size {model_size} not found in schedule, using all revisions")
        return revisions
    
    max_step = FULL_SCHEDULE_LAST_STEP_PER_MODEL[model_size]
    filtered_revisions = []
    
    for revision in revisions:
        step = extract_step_number(revision)
        if step <= max_step:
            filtered_revisions.append(revision)
    
    return filtered_revisions

def sample_revisions_evenly(revisions, num_samples):
    """
    Sort revisions by step number and sample evenly across the sorted list.
    
    Args:
        revisions: List of revision strings
        num_samples: Number of revisions to sample
    
    Returns:
        List of sampled revision strings
    """
    if num_samples >= len(revisions):
        return revisions
    
    # Sort by step number
    sorted_revisions = sorted(revisions, key=extract_step_number)
    
    # Sample evenly across the sorted list
    indices = []
    if num_samples == 1:
        indices = [len(sorted_revisions) - 1]  # Take the last one
    else:
        step = (len(sorted_revisions) - 1) / (num_samples - 1)
        indices = [int(i * step) for i in range(num_samples)]
    
    return [sorted_revisions[i] for i in indices]

def get_checkpoints(repo_name):
    """Find all intermediate checkpoints for the given repo name."""
    checkpoints = []

    with open(CHECKPOINTS_FILE, "r") as file:
        for line in file:
            obj = json.loads(line)
            model_name = obj["model_name"]
            model_name = model_name_to_repo_name(model_name)
            if model_name is None:
                continue

            if model_name.startswith(repo_name):
                checkpoints.append(obj)

    seeds = [int(model['model_name'].split('-')[-1]) for model in checkpoints]
    assert all(seed in SEED_MAPPING for seed in seeds), f"Unknown seed {seeds}"
    if len(checkpoints) == 5:
        assert repo_name.split('-')[-1] == "1B", f"1B model should have 5 checkpoints, got {len(checkpoints)}"
    else:
        assert len(checkpoints) == 3, f"Expected 3 or 5 checkpoints, got {len(checkpoints)}"

    return checkpoints

def get_expected_branches_for_repo(repo_name, num_revisions=5):
    """Get expected branch names for a repository that should have optim.pt files."""
    branches = set()
    
    checkpoints = get_checkpoints(repo_name)
    
    for checkpoint in checkpoints:
        seed = int(checkpoint["model_name"].split("-")[-1])
        seed_name = SEED_MAPPING[seed].replace(" ", "-")
        
        # Filter revisions by training schedule first
        filtered_revisions = filter_revisions_by_schedule(checkpoint["revisions"], checkpoint["model_name"])
        
        # Sample revisions evenly from filtered list
        sampled_revisions = sample_revisions_evenly(filtered_revisions, num_revisions)
        
        for step in sampled_revisions:
            branch_name = f"{step.replace('-unsharded-hf','')}-seed-{seed_name}"
            branches.add(branch_name)
    
    return branches

@retry_with_backoff(max_retries=3, base_delay=1.0)
def check_repo_exists_with_retry(api: HfApi, full_repo_name: str):
    """Check if a repository exists with retry logic."""
    return api.repo_info(repo_id=full_repo_name)

@retry_with_backoff(max_retries=3, base_delay=1.0)
def list_repo_files_with_retry(api: HfApi, full_repo_name: str, revision: str):
    """List repository files with retry logic."""
    return api.list_repo_files(
        repo_id=full_repo_name,
        revision=revision,
        repo_type="model"
    )

@retry_with_backoff(max_retries=3, base_delay=1.0)
def get_file_info_with_retry(api: HfApi, full_repo_name: str, paths: List[str], revision: str):
    """Get file info with retry logic."""
    return api.get_paths_info(
        repo_id=full_repo_name,
        paths=paths,
        revision=revision,
        repo_type="model"
    )

def check_optim_in_branch(api: HfApi, org_name: str, repo_name: str, branch_name: str) -> OptimStatus:
    """Check if optim.pt exists in a specific branch."""
    full_repo_name = f"{org_name}/{repo_name}"
    
    try:
        # First check if branch exists
        try:
            check_repo_exists_with_retry(api, full_repo_name)
        except RevisionNotFoundError:
            return OptimStatus(
                branch_name=branch_name,
                branch_exists=False,
                optim_exists=False,
                error="Branch not found"
            )
        
        # List files in the branch to check for training/optim.pt
        try:
            files = list_repo_files_with_retry(api, full_repo_name, branch_name)
            
            optim_file = "training/optim.pt"
            optim_exists = optim_file in files
            
            file_size = None
            if optim_exists:
                # Get file info to check size (metadata only)
                try:
                    file_info = get_file_info_with_retry(api, full_repo_name, [optim_file], branch_name)
                    if file_info and len(file_info) > 0:
                        file_size = file_info[0].size
                except Exception:
                    # If we can't get file size, that's okay
                    pass
            
            return OptimStatus(
                branch_name=branch_name,
                branch_exists=True,
                optim_exists=optim_exists,
                file_size=file_size,
                error=None if optim_exists else "training/optim.pt not found"
            )
            
        except Exception as e:
            return OptimStatus(
                branch_name=branch_name,
                branch_exists=True,
                optim_exists=False,
                error=f"Failed to list files: {str(e)}"
            )
            
    except Exception as e:
        return OptimStatus(
            branch_name=branch_name,
            branch_exists=False,
            optim_exists=False,
            error=str(e)
        )

def check_repo_optim_status(api: HfApi, org_name: str, repo_name: str, num_revisions: int) -> RepoOptimStatus:
    """Check the optimizer file status for a complete repository."""
    
    # Check if repo exists
    try:
        check_repo_exists_with_retry(api, f"{org_name}/{repo_name}")
    except RepositoryNotFoundError:
        return RepoOptimStatus(
            name=repo_name,
            repo_exists=False,
            expected_branches=set(),
            optim_statuses=[],
            error="Repository not found"
        )
    except Exception as e:
        return RepoOptimStatus(
            name=repo_name,
            repo_exists=False,
            expected_branches=set(),
            optim_statuses=[],
            error=str(e)
        )
    
    # Get expected branches
    try:
        expected_branches = get_expected_branches_for_repo(repo_name, num_revisions)
    except Exception as e:
        return RepoOptimStatus(
            name=repo_name,
            repo_exists=True,
            expected_branches=set(),
            optim_statuses=[],
            error=f"Failed to get expected branches: {str(e)}"
        )
    
    # Check each expected branch
    optim_statuses = []
    for branch_name in expected_branches:
        status = check_optim_in_branch(api, org_name, repo_name, branch_name)
        optim_statuses.append(status)
        # Small delay to avoid rate limiting
        time.sleep(0.1)  # Increased from 0.01 to 0.1
    
    return RepoOptimStatus(
        name=repo_name,
        repo_exists=True,
        expected_branches=expected_branches,
        optim_statuses=optim_statuses
    )

def load_repo_names() -> List[str]:
    """Load repository names from repo_names.txt."""
    with open(REPO_NAMES_FILE, 'r') as f:
        return [line.strip() for line in f if line.strip()]

def print_summary(repo_statuses: List[RepoOptimStatus]):
    """Print a summary of optimizer file statuses."""
    
    total_repos = len(repo_statuses)
    existing_repos = sum(1 for status in repo_statuses if status.repo_exists)
    
    # Count various statistics
    repos_with_errors = sum(1 for status in repo_statuses if status.error)
    total_branches = sum(len(status.expected_branches) for status in repo_statuses if status.repo_exists)
    branches_with_optim = sum(
        sum(1 for optim_status in status.optim_statuses if optim_status.optim_exists)
        for status in repo_statuses if status.repo_exists
    )
    branches_missing_optim = sum(
        sum(1 for optim_status in status.optim_statuses if optim_status.branch_exists and not optim_status.optim_exists)
        for status in repo_statuses if status.repo_exists
    )
    branches_not_found = sum(
        sum(1 for optim_status in status.optim_statuses if not optim_status.branch_exists)
        for status in repo_statuses if status.repo_exists
    )
    
    # Repositories with issues
    repos_with_missing_optim = sum(
        1 for status in repo_statuses 
        if status.repo_exists and any(
            optim_status.branch_exists and not optim_status.optim_exists 
            for optim_status in status.optim_statuses
        )
    )
    
    print(f"\n{'='*80}")
    print(f"OPTIMIZER FILE VERIFICATION SUMMARY")
    print(f"{'='*80}")
    print(f"Total repositories checked: {total_repos}")
    print(f"Existing repositories: {existing_repos}")
    print(f"Missing repositories: {total_repos - existing_repos}")
    print(f"Repositories with API errors: {repos_with_errors}")
    print(f"")
    print(f"Total expected branches: {total_branches}")
    print(f"Branches with optim.pt: {branches_with_optim}")
    print(f"Branches missing optim.pt: {branches_missing_optim}")
    print(f"Branches not found: {branches_not_found}")
    print(f"")
    print(f"Repositories with missing optim files: {repos_with_missing_optim}")
    
    if branches_with_optim > 0:
        success_rate = (branches_with_optim / total_branches) * 100
        print(f"Success rate: {success_rate:.1f}%")
    
    # Show repositories with issues
    problematic_repos = [
        status for status in repo_statuses 
        if (not status.repo_exists or 
            status.error or 
            any(optim_status.branch_exists and not optim_status.optim_exists 
                for optim_status in status.optim_statuses))
    ]
    
    if problematic_repos:
        print(f"\nREPOSITORIES WITH ISSUES ({len(problematic_repos)}):")
        for status in problematic_repos:
            if not status.repo_exists:
                print(f"  - {status.name}: Repository not found")
            elif status.error:
                print(f"  - {status.name}: {status.error}")
            else:
                missing_optim = [
                    optim_status for optim_status in status.optim_statuses 
                    if optim_status.branch_exists and not optim_status.optim_exists
                ]
                if missing_optim:
                    print(f"  - {status.name}: {len(missing_optim)}/{len(status.optim_statuses)} branches missing optim.pt")

def main():
    parser = argparse.ArgumentParser(description="Verify optimizer states in Hugging Face repositories")
    parser.add_argument("--org_name", type=str, default="allenai", 
                       help="Organization name on Hugging Face Hub")
    parser.add_argument("--num_revisions", type=int, default=5,
                       help="Number of revisions per seed that should have optim.pt (matches upload_optim.py default)")
    parser.add_argument("--detailed", action="store_true",
                       help="Show detailed output for each repository")
    parser.add_argument("--repo_filter", type=str,
                       help="Check only repositories containing this string")
    parser.add_argument("--single_repo", type=str,
                       help="Check only this specific repository name")
    parser.add_argument("--size_filter", type=str,
                       help="Check only repositories ending with this size (e.g., '1B', '60M')")
    parser.add_argument("--delay", type=float, default=0.2,
                       help="Delay between API calls in seconds (default: 0.2)")
    args = parser.parse_args()
    
    # Initialize HF API
    api = HfApi()
    
    # Load repository names
    print("Loading repository names...")
    all_repo_names = load_repo_names()
    
    # Apply repository filters
    if args.single_repo:
        repo_names = [args.single_repo] if args.single_repo in all_repo_names else []
        if not repo_names:
            print(f"Repository '{args.single_repo}' not found in repo_names.txt")
            return
    elif args.repo_filter:
        repo_names = [name for name in all_repo_names if args.repo_filter in name]
        if not repo_names:
            print(f"No repositories found matching filter '{args.repo_filter}'")
            return
    elif args.size_filter:
        repo_names = [name for name in all_repo_names if name.endswith(args.size_filter)]
        if not repo_names:
            print(f"No repositories found ending with size '{args.size_filter}'")
            return
    else:
        repo_names = all_repo_names
    
    print(f"Found {len(repo_names)} repositories to check")
    print(f"Looking for {args.num_revisions} revisions per seed with optim.pt files")
    
    # Check each repository
    repo_statuses = []
    
    with tqdm(total=len(repo_names), desc="Verifying optim files") as pbar:
        for repo_name in repo_names:
            status = check_repo_optim_status(api, args.org_name, repo_name, args.num_revisions)
            repo_statuses.append(status)
            
            if args.detailed:
                print(f"\n{repo_name}: {'✓' if status.repo_exists else '✗'}")
                if status.repo_exists and status.optim_statuses:
                    optim_count = sum(1 for s in status.optim_statuses if s.optim_exists)
                    total_count = len(status.optim_statuses)
                    print(f"  Optim files: {optim_count}/{total_count}")
                    
                    if args.detailed and optim_count < total_count:
                        missing = [s for s in status.optim_statuses if s.branch_exists and not s.optim_exists]
                        for missing_status in missing[:3]:  # Show first 3 missing
                            print(f"    Missing: {missing_status.branch_name}")
                        if len(missing) > 3:
                            print(f"    ... and {len(missing) - 3} more")
            
            pbar.update(1)
            # Rate limiting - use configurable delay
            time.sleep(args.delay)
    
    # Print summary
    print_summary(repo_statuses)
    
    # Save detailed results to JSON
    results_file = os.path.join(SCRIPT_DIR, "optim_verification_results.json")
    results = []
    for status in repo_statuses:
        result = {
            "name": status.name,
            "repo_exists": status.repo_exists,
            "error": status.error,
            "expected_branches": list(status.expected_branches),
            "optim_details": [
                {
                    "branch_name": s.branch_name,
                    "branch_exists": s.branch_exists,
                    "optim_exists": s.optim_exists,
                    "file_size": s.file_size,
                    "error": s.error
                } for s in status.optim_statuses
            ],
            "summary": {
                "total_branches": len(status.optim_statuses),
                "branches_with_optim": sum(1 for s in status.optim_statuses if s.optim_exists),
                "branches_missing_optim": sum(1 for s in status.optim_statuses if s.branch_exists and not s.optim_exists),
                "branches_not_found": sum(1 for s in status.optim_statuses if not s.branch_exists)
            }
        }
        results.append(result)
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nDetailed results saved to: {results_file}")

if __name__ == "__main__":
    main()
