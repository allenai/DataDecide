#!/usr/bin/env python3
"""
Script to check the status of Hugging Face repositories.

This script verifies:
1. If each repository in repo_names.txt exists
2. If each repository has the expected branches for all steps and seeds from weka_paths.jsonl
3. If each branch has commits on model.safetensors files after the initial commit

IMPORTANT: This script ONLY uses metadata APIs and does NOT download any model files or data.
It only checks repository existence, branch existence, and commit metadata.
"""

import os
import json
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import List, Dict, Set, Optional
# METADATA-ONLY imports - these do NOT download model files
from huggingface_hub import HfApi, list_repo_commits
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
try:
    import requests
except ImportError:
    requests = None

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

@dataclass
class BranchStatus:
    name: str
    exists: bool
    has_model_commits: bool
    commit_count: int
    error: Optional[str] = None

@dataclass
class RepoStatus:
    name: str
    exists: bool
    expected_branches: Set[str]
    found_branches: Set[str]
    branch_statuses: List[BranchStatus]
    error: Optional[str] = None

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

def load_repo_names() -> List[str]:
    """Load repository names from repo_names.txt."""
    with open(REPO_NAMES_FILE, 'r') as f:
        return [line.strip() for line in f if line.strip()]

def load_checkpoints() -> Dict[str, List[dict]]:
    """Load and group checkpoints by repository name."""
    checkpoints_by_repo = {}
    
    with open(CHECKPOINTS_FILE, 'r') as f:
        for line in f:
            obj = json.loads(line)
            model_name = obj["model_name"]
            repo_name = model_name_to_repo_name(model_name)
            
            if repo_name is None:
                continue
                
            if repo_name not in checkpoints_by_repo:
                checkpoints_by_repo[repo_name] = []
            checkpoints_by_repo[repo_name].append(obj)
    
    return checkpoints_by_repo

def get_expected_branches(checkpoints: List[dict], step_filter: Optional[str] = None) -> Set[str]:
    """Get expected branch names for a repository based on its checkpoints."""
    branches = set()
    
    for checkpoint in checkpoints:
        seed = int(checkpoint["model_name"].split("-")[-1])
        seed_name = SEED_MAPPING[seed].replace(" ", "-")
        
        for step in checkpoint["revisions"]:
            # Apply step filter if provided
            if step_filter and step_filter not in step:
                continue
                
            branch_name = f"{step.replace('-unsharded-hf','')}-seed-{seed_name}"
            branches.add(branch_name)
    
    return branches

def check_repo_exists(api: HfApi, org_name: str, repo_name: str) -> bool:
    """Check if a repository exists."""
    try:
        api.repo_info(repo_id=f"{org_name}/{repo_name}")
        return True
    except RepositoryNotFoundError:
        return False
    except Exception:
        return False

def check_branch_commits(api: HfApi, org_name: str, repo_name: str, branch_name: str) -> BranchStatus:
    """Check if a branch exists and has the expected model commit.
    Note: We cannot reliably check if model.safetensors was modified in the commit
    due to HuggingFace API limitations, so we trust the commit message.
    """
    full_repo_name = f"{org_name}/{repo_name}"
    
    try:
        # First check if branch exists using lightweight repo_info call
        repo_info = api.repo_info(repo_id=full_repo_name, revision=branch_name)
        
        # Get commit history (metadata only, no file downloads)
        commits = list(list_repo_commits(
            repo_id=full_repo_name,
            revision=branch_name,
            repo_type="model"
        ))
        
        if not commits:
            return BranchStatus(
                name=branch_name,
                exists=True,
                has_model_commits=False,
                commit_count=0,
                error="No commits found"
            )
        
        # Get the latest commit
        latest_commit = commits[0]  # list_repo_commits returns newest first
        
        # Check if the latest commit message matches the expected pattern
        expected_message = f"Pushing model to {branch_name} branch"
        commit_message_matches = latest_commit.title == expected_message
        
        if not commit_message_matches:
            return BranchStatus(
                name=branch_name,
                exists=True,
                has_model_commits=False,
                commit_count=len(commits),
                error=f"Latest commit: '{latest_commit.title}' (expected: '{expected_message}')"
            )
        
        # Check if model.safetensors exists in the repo
        try:
            files_at_commit = api.list_repo_files(
                repo_id=full_repo_name,
                revision=latest_commit.commit_id,
                repo_type="model"
            )
            # Check for model file(s) depending on model size
            size = repo_name.split('-')[-1]
            if size == "1B":
                has_model_file = (
                    "model-00001-of-00002.safetensors" in files_at_commit and
                    "model-00002-of-00002.safetensors" in files_at_commit
                )
            else:
                has_model_file = "model.safetensors" in files_at_commit
            
            if not has_model_file:
                return BranchStatus(
                    name=branch_name,
                    exists=True,
                    has_model_commits=False,
                    commit_count=len(commits),
                    error="model.safetensors not found in repository"
                )
            
            # If commit message is correct and model file exists, assume it's valid
            # This is the best we can do without access to commit diff information
            return BranchStatus(
                name=branch_name,
                exists=True,
                has_model_commits=True,
                commit_count=len(commits),
                error=None
            )
                
        except Exception as e:
            return BranchStatus(
                name=branch_name,
                exists=True,
                has_model_commits=False,
                commit_count=len(commits),
                error=f"Failed to check files: {str(e)}"
            )
        
    except RevisionNotFoundError:
        return BranchStatus(
            name=branch_name,
            exists=False,
            has_model_commits=False,
            commit_count=0,
            error="Branch not found"
        )
    except Exception as e:
        return BranchStatus(
            name=branch_name,
            exists=False,
            has_model_commits=False,
            commit_count=0,
            error=str(e)
        )

def check_repository_status(api: HfApi, org_name: str, repo_name: str, 
                          expected_branches: Set[str]) -> RepoStatus:
    """Check the complete status of a repository."""
    
    # Check if repo exists
    if not check_repo_exists(api, org_name, repo_name):
        return RepoStatus(
            name=repo_name,
            exists=False,
            expected_branches=expected_branches,
            found_branches=set(),
            branch_statuses=[],
            error="Repository not found"
        )
    
    # Get actual branches
    try:
        git_refs = api.list_repo_refs(repo_id=f"{org_name}/{repo_name}")
        found_branches = {branch.name for branch in git_refs.branches}
    except Exception as e:
        return RepoStatus(
            name=repo_name,
            exists=True,
            expected_branches=expected_branches,
            found_branches=set(),
            branch_statuses=[],
            error=f"Could not list branches: {str(e)}"
        )
    
    # Check each expected branch
    branch_statuses = []
    for branch_name in expected_branches:
        status = check_branch_commits(api, org_name, repo_name, branch_name)
        branch_statuses.append(status)
        # Small delay to avoid rate limiting
        time.sleep(0.01)
    
    return RepoStatus(
        name=repo_name,
        exists=True,
        expected_branches=expected_branches,
        found_branches=found_branches,
        branch_statuses=branch_statuses
    )

def print_summary(repo_statuses: List[RepoStatus]):
    """Print a summary of repository statuses."""
    
    total_repos = len(repo_statuses)
    existing_repos = sum(1 for status in repo_statuses if status.exists)
    repos_with_errors = sum(1 for status in repo_statuses if status.error)
    
    print(f"\n{'='*80}")
    print(f"REPOSITORY STATUS SUMMARY")
    print(f"{'='*80}")
    print(f"Total repositories checked: {total_repos}")
    print(f"Existing repositories: {existing_repos}")
    print(f"Missing repositories: {total_repos - existing_repos}")
    print(f"Repositories with errors: {repos_with_errors}")
    
    # Missing repositories
    missing_repos = [status for status in repo_statuses if not status.exists]
    if missing_repos:
        print(f"\nMISSING REPOSITORIES ({len(missing_repos)}):")
        for status in missing_repos:
            print(f"  - {status.name}")
    
    # Repositories with branch issues
    print(f"\nBRANCH STATUS:")
    for status in repo_statuses:
        if not status.exists:
            continue
            
        missing_branches = status.expected_branches - status.found_branches
        branches_without_commits = [b for b in status.branch_statuses 
                                  if b.exists and not b.has_model_commits]
        
        if missing_branches or branches_without_commits:
            print(f"\n  {status.name}:")
            if missing_branches:
                print(f"    Missing branches ({len(missing_branches)}): {', '.join(sorted(missing_branches))}")
            if branches_without_commits:
                print(f"    Branches without model commits ({len(branches_without_commits)}): {', '.join([b.name for b in branches_without_commits])}")

def main():
    parser = argparse.ArgumentParser(description="Check Hugging Face repository status")
    parser.add_argument("--org_name", type=str, default="allenai", 
                       help="Organization name on Hugging Face Hub")
    parser.add_argument("--max_workers", type=int, default=5,
                       help="Maximum number of concurrent workers")
    parser.add_argument("--detailed", action="store_true",
                       help="Show detailed output for each repository")
    parser.add_argument("--repo_filter", type=str,
                       help="Check only repositories containing this string (e.g., 'falcon-60M')")
    parser.add_argument("--step_filter", type=str,
                       help="Check only branches with this step (e.g., 'step1250')")
    parser.add_argument("--single_repo", type=str,
                       help="Check only this specific repository name")
    args = parser.parse_args()
    
    # Initialize HF API
    api = HfApi()
    
    # Load data
    print("Loading repository names and checkpoint data...")
    all_repo_names = load_repo_names()
    checkpoints_by_repo = load_checkpoints()
    
    # Apply repository filters
    if args.single_repo:
        # Check only the specific repository
        repo_names = [args.single_repo] if args.single_repo in all_repo_names else []
        if not repo_names:
            print(f"Repository '{args.single_repo}' not found in repo_names.txt")
            return
    elif args.repo_filter:
        # Filter repositories by substring
        repo_names = [name for name in all_repo_names if args.repo_filter in name]
        if not repo_names:
            print(f"No repositories found matching filter '{args.repo_filter}'")
            return
    else:
        repo_names = all_repo_names
    
    print(f"Found {len(repo_names)} repositories to check")
    print(f"Found checkpoint data for {len(checkpoints_by_repo)} repositories")
    
    if args.step_filter:
        print(f"Filtering branches by step: {args.step_filter}")
    
    # Check each repository
    repo_statuses = []
    
    with tqdm(total=len(repo_names), desc="Checking repositories") as pbar:
        for repo_name in repo_names:
            expected_branches = set()
            if repo_name in checkpoints_by_repo:
                expected_branches = get_expected_branches(
                    checkpoints_by_repo[repo_name], 
                    step_filter=args.step_filter
                )
            
            status = check_repository_status(api, args.org_name, repo_name, expected_branches)
            repo_statuses.append(status)
            
            if args.detailed:
                print(f"\n{repo_name}: {'✓' if status.exists else '✗'}")
                if status.exists and expected_branches:
                    missing = expected_branches - status.found_branches
                    if missing:
                        print(f"  Missing branches: {len(missing)}")
                        if args.step_filter:
                            print(f"  (filtered by step: {args.step_filter})")
            
            pbar.update(1)
            # Rate limiting
            time.sleep(0.02)
    
    # Print summary
    print_summary(repo_statuses)
    
    # Save detailed results to JSON
    results_file = os.path.join(SCRIPT_DIR, "repo_status_check.json")
    results = []
    for status in repo_statuses:
        result = {
            "name": status.name,
            "exists": status.exists,
            "error": status.error,
            "expected_branches": list(status.expected_branches),
            "found_branches": list(status.found_branches),
            "missing_branches": list(status.expected_branches - status.found_branches),
            "branch_details": [
                {
                    "name": b.name,
                    "exists": b.exists,
                    "has_model_commits": b.has_model_commits,
                    "commit_count": b.commit_count,
                    "error": b.error
                } for b in status.branch_statuses
            ]
        }
        results.append(result)
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nDetailed results saved to: {results_file}")

if __name__ == "__main__":
    main()
