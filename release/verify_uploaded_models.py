#!/usr/bin/env python3
"""
Script to verify that uploaded model weights on Hugging Face Hub match local weights on Weka.

This script:
1. Randomly samples one branch per repository
2. Downloads the model weights from HF Hub
3. Loads the corresponding local model from Weka
4. Compares the weights tensor by tensor
5. Cleans up downloaded files to avoid storage issues
6. Logs all operations to a configurable log file

Usage:
    python verify_uploaded_models.py --org_name allenai
    python verify_uploaded_models.py --debug  # Test only one small model
    python verify_uploaded_models.py --debug_seed_diff  # Test that different seeds have different weights
    python verify_uploaded_models.py --test_functions  # Test basic functionality without downloads
    python verify_uploaded_models.py --repo_filter falcon-60M  # Test specific repos
    python verify_uploaded_models.py --dry_run  # Show what would be tested
    python verify_uploaded_models.py --log_file /path/to/custom.log  # Custom log file location
"""

import os
import json
import argparse
import tempfile
import shutil
import random
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Set, Optional, Tuple
import torch
from huggingface_hub import HfApi, snapshot_download
from hf_olmo import OLMoForCausalLM
from transformers import AutoTokenizer
import numpy as np
from tqdm import tqdm
import time

# Constants
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_NAMES_FILE = os.path.join(SCRIPT_DIR, "repo_names.txt")
CHECKPOINTS_FILE = os.path.join(SCRIPT_DIR, "../checkpoints/weka_paths.jsonl")
WEKA_PATH = "/data/input/"

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

def get_available_branches(api: HfApi, org_name: str, repo_name: str) -> List[str]:
    """Get all available branches for a repository."""
    try:
        git_refs = api.list_repo_refs(repo_id=f"{org_name}/{repo_name}")
        return [branch.name for branch in git_refs.branches if branch.name != "main"]
    except Exception as e:
        print(f"Error getting branches for {repo_name}: {e}")
        return []

def find_local_path(branch_name: str, checkpoints: List[dict]) -> Optional[str]:
    """Find the local Weka path for a given branch."""
    # Parse branch name: step-seed-<seed_name>
    # Example: step1250-seed-default
    parts = branch_name.split('-seed-')
    if len(parts) != 2:
        return None
    
    step_part = parts[0]
    seed_name = parts[1]
    
    # Find matching checkpoint and step
    for checkpoint in checkpoints:
        seed = int(checkpoint["model_name"].split("-")[-1])
        expected_seed_name = SEED_MAPPING[seed].replace(" ", "-")
        
        if expected_seed_name != seed_name:
            continue
            
        # Look for matching step
        step_with_hf = f"{step_part}-unsharded-hf"
        if step_with_hf in checkpoint["revisions"]:
            location = checkpoint['checkpoints_location'].replace("weka://oe-eval-default/", WEKA_PATH)
            return os.path.join(location, step_with_hf)
    
    return None

def compare_model_weights(hf_model_path: str, local_model_path: str, tolerance: float = 1e-6) -> Tuple[bool, str]:
    """Compare model weights between HF downloaded model and local model."""
    try:
        print(f"Loading HF model from {hf_model_path}")
        hf_model = OLMoForCausalLM.from_pretrained(hf_model_path)
        
        print(f"Loading local model from {local_model_path}")
        local_model = OLMoForCausalLM.from_pretrained(local_model_path)
        
        # Get state dicts
        hf_state_dict = hf_model.state_dict()
        local_state_dict = local_model.state_dict()
        
        # Check if they have the same keys
        if set(hf_state_dict.keys()) != set(local_state_dict.keys()):
            missing_in_hf = set(local_state_dict.keys()) - set(hf_state_dict.keys())
            missing_in_local = set(hf_state_dict.keys()) - set(local_state_dict.keys())
            return False, f"Key mismatch. Missing in HF: {missing_in_hf}, Missing in local: {missing_in_local}"
        
        # Compare each tensor
        mismatched_params = []
        for key in hf_state_dict.keys():
            hf_tensor = hf_state_dict[key]
            local_tensor = local_state_dict[key]
            
            if hf_tensor.shape != local_tensor.shape:
                mismatched_params.append(f"{key}: shape mismatch ({hf_tensor.shape} vs {local_tensor.shape})")
                continue
            
            # Check if tensors are close
            if not torch.allclose(hf_tensor, local_tensor, atol=tolerance, rtol=tolerance):
                max_diff = torch.max(torch.abs(hf_tensor - local_tensor)).item()
                mismatched_params.append(f"{key}: values differ (max diff: {max_diff})")
        
        if mismatched_params:
            return False, f"Weight mismatches: {mismatched_params[:5]}"  # Show first 5 mismatches
        
        return True, "All weights match"
        
    except Exception as e:
        return False, f"Error comparing models: {str(e)}"

def test_basic_functionality():
    """Test basic functionality without downloading models."""
    print("Testing basic functionality...")
    
    # Test model name conversion
    print("\n1. Testing model name conversion...")
    test_cases = [
        ("falcon-4M-6198", "DataDecide-falcon-4M"),
        ("dolma17-60M-14", "DataDecide-dolma1_7-60M"),
        ("baseline-1B-4", "DataDecide-dolma1_7-1B"),
        ("DCLM-baseline-25p-4M-5", None),  # Should return None
    ]
    
    for model_name, expected in test_cases:
        result = model_name_to_repo_name(model_name)
        status = "✓" if result == expected else "✗"
        print(f"  {status} {model_name} -> {result}")
    
    # Test data loading
    print("\n2. Testing data loading...")
    try:
        repo_names = load_repo_names()
        print(f"  ✓ Loaded {len(repo_names)} repository names")
        
        checkpoints = load_checkpoints()
        print(f"  ✓ Loaded checkpoint data for {len(checkpoints)} repositories")
        
        if repo_names and checkpoints:
            sample_repo = list(checkpoints.keys())[0]
            sample_checkpoints = checkpoints[sample_repo]
            print(f"  Sample: {sample_repo} ({len(sample_checkpoints)} checkpoints)")
            
    except Exception as e:
        print(f"  ✗ Error loading data: {e}")
    
    # Test branch parsing
    print("\n3. Testing branch parsing...")
    mock_checkpoint = {
        "model_name": "falcon-4M-6198",
        "checkpoints_location": "weka://oe-eval-default/ai2-llm/checkpoints/test",
        "revisions": ["step1250-unsharded-hf", "step2500-unsharded-hf"]
    }
    
    test_branches = [
        "step1250-seed-default",
        "step2500-seed-default", 
        "step1250-seed-large-aux-2",
        "invalid-branch-name"
    ]
    
    for branch in test_branches:
        path = find_local_path(branch, [mock_checkpoint])
        status = "✓" if path else "✗"
        print(f"  {status} {branch} -> {path or 'No path found'}")
    
    print("\nBasic functionality test completed!")

def debug_seed_differences(api: HfApi, org_name: str, checkpoints_by_repo: Dict[str, List[dict]], 
                          temp_dir: str, tolerance: float = 1e-6) -> None:
    """Debug function to verify that different seeds produce different weights."""
    print("Debug: Testing that different seeds have different weights...")
    
    # Find a repo with multiple seeds
    suitable_repo = None
    for repo_name, checkpoints in checkpoints_by_repo.items():
        if len(checkpoints) >= 2:  # Need at least 2 different seeds
            suitable_repo = repo_name
            break
    
    if not suitable_repo:
        print("No repository found with multiple seeds for testing")
        return
    
    print(f"Testing with repository: {suitable_repo}")
    
    # Get available branches for this repo
    branches = get_available_branches(api, org_name, suitable_repo)
    if len(branches) < 2:
        print(f"Not enough branches found for {suitable_repo}")
        return
    
    # Find branches from the same step but different seeds
    step_to_branches = {}
    for branch in branches:
        if '-seed-' in branch:
            step = branch.split('-seed-')[0]
            if step not in step_to_branches:
                step_to_branches[step] = []
            step_to_branches[step].append(branch)
    
    # Find a step with multiple seeds
    test_step = None
    test_branches = None
    for step, step_branches in step_to_branches.items():
        if len(step_branches) >= 2:
            test_step = step
            test_branches = step_branches[:2]  # Take first 2
            break
    
    if not test_branches:
        print("No step found with multiple seeds")
        return
    
    print(f"Comparing branches: {test_branches[0]} vs {test_branches[1]}")
    
    # Download and compare the two models
    repo_temp_dir1 = os.path.join(temp_dir, f"{suitable_repo}_{test_branches[0]}")
    repo_temp_dir2 = os.path.join(temp_dir, f"{suitable_repo}_{test_branches[1]}")
    
    try:
        os.makedirs(repo_temp_dir1, exist_ok=True)
        os.makedirs(repo_temp_dir2, exist_ok=True)
        
        print(f"Downloading first model: {test_branches[0]}")
        snapshot_download(
            repo_id=f"{org_name}/{suitable_repo}",
            revision=test_branches[0],
            local_dir=repo_temp_dir1,
            allow_patterns=["*.safetensors", "*.json"],
            token=os.getenv("HF_TOKEN")
        )
        
        print(f"Downloading second model: {test_branches[1]}")
        snapshot_download(
            repo_id=f"{org_name}/{suitable_repo}",
            revision=test_branches[1],
            local_dir=repo_temp_dir2,
            allow_patterns=["*.safetensors", "*.json"],
            token=os.getenv("HF_TOKEN")
        )
        
        # Load and compare models
        print("Loading models for comparison...")
        model1 = OLMoForCausalLM.from_pretrained(repo_temp_dir1)
        model2 = OLMoForCausalLM.from_pretrained(repo_temp_dir2)
        
        state_dict1 = model1.state_dict()
        state_dict2 = model2.state_dict()
        
        # Compare weights
        different_params = []
        identical_params = []
        
        for key in state_dict1.keys():
            tensor1 = state_dict1[key]
            tensor2 = state_dict2[key]
            
            if torch.allclose(tensor1, tensor2, atol=tolerance, rtol=tolerance):
                identical_params.append(key)
            else:
                max_diff = torch.max(torch.abs(tensor1 - tensor2)).item()
                different_params.append((key, max_diff))
        
        print(f"\nResults:")
        print(f"  Parameters that are identical: {len(identical_params)}")
        print(f"  Parameters that are different: {len(different_params)}")
        
        if different_params:
            print(f"  ✓ PASS: Different seeds produce different weights (as expected)")
            print(f"  Sample differences:")
            for key, diff in different_params[:3]:  # Show first 3
                print(f"    {key}: max_diff = {diff:.6e}")
        else:
            print(f"  ✗ FAIL: All weights are identical (unexpected!)")
            
        if identical_params:
            print(f"  Note: Some parameters are identical (might be embeddings, etc.)")
    
    except Exception as e:
        print(f"Error during seed difference test: {e}")
    
    finally:
        # Clean up
        for temp_path in [repo_temp_dir1, repo_temp_dir2]:
            if os.path.exists(temp_path):
                shutil.rmtree(temp_path)

def verify_single_repo(api: HfApi, org_name: str, repo_name: str, checkpoints: List[dict], 
                      temp_dir: str, logger: logging.Logger) -> Tuple[bool, str, str]:
    """Verify one randomly sampled branch from a repository."""
    
    logger.info(f"Starting verification for repository: {repo_name}")
    
    # Get available branches
    branches = get_available_branches(api, org_name, repo_name)
    if not branches:
        logger.warning(f"No branches found for {repo_name}")
        return False, "No branches found", ""
    
    logger.info(f"Found {len(branches)} branches for {repo_name}")
    
    # Randomly sample one branch
    branch = random.choice(branches)
    logger.info(f"Randomly selected branch: {branch}")
    
    # Find corresponding local path
    local_path = find_local_path(branch, checkpoints)
    if not local_path:
        logger.error(f"Could not find local path for branch {branch} in repo {repo_name}")
        return False, f"Could not find local path for branch {branch}", branch
    
    logger.info(f"Local path found: {local_path}")
    
    if not os.path.exists(local_path):
        logger.error(f"Local path does not exist: {local_path}")
        return False, f"Local path does not exist: {local_path}", branch
    
    # Create a subdirectory for this download
    repo_temp_dir = os.path.join(temp_dir, f"{repo_name}_{branch}")
    os.makedirs(repo_temp_dir, exist_ok=True)
    logger.info(f"Created temporary directory: {repo_temp_dir}")
    
    try:
        # Download the model from HF Hub
        logger.info(f"Downloading {org_name}/{repo_name} branch {branch}")
        print(f"Downloading {org_name}/{repo_name} branch {branch}")
        
        start_time = time.time()
        hf_model_path = snapshot_download(
            repo_id=f"{org_name}/{repo_name}",
            revision=branch,
            local_dir=repo_temp_dir,
            allow_patterns=["*.safetensors", "*.json"],  # Only download necessary files
            token=os.getenv("HF_TOKEN")
        )
        download_time = time.time() - start_time
        logger.info(f"Download completed in {download_time:.2f} seconds")
        
        # Compare weights
        logger.info(f"Starting weight comparison between HF and local models")
        start_time = time.time()
        weights_match, message = compare_model_weights(hf_model_path, local_path)
        comparison_time = time.time() - start_time
        logger.info(f"Weight comparison completed in {comparison_time:.2f} seconds")
        
        if weights_match:
            logger.info(f"✓ VERIFICATION PASSED for {repo_name} (branch: {branch}): {message}")
        else:
            logger.error(f"✗ VERIFICATION FAILED for {repo_name} (branch: {branch}): {message}")
        
        return weights_match, message, branch
        
    except Exception as e:
        error_msg = f"Error during verification: {str(e)}"
        logger.error(f"Exception in {repo_name} (branch: {branch}): {error_msg}")
        return False, error_msg, branch
    
    finally:
        # Clean up downloaded files
        if os.path.exists(repo_temp_dir):
            shutil.rmtree(repo_temp_dir)
            logger.info(f"Cleaned up temporary directory: {repo_temp_dir}")
            print(f"Cleaned up {repo_temp_dir}")

def setup_logging(log_file: str) -> logging.Logger:
    """Set up logging to both file and console."""
    # Create logger
    logger = logging.getLogger('model_verification')
    logger.setLevel(logging.INFO)
    
    # Clear any existing handlers
    logger.handlers = []
    
    # Create formatters
    file_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_formatter = logging.Formatter('%(message)s')
    
    # File handler
    file_handler = logging.FileHandler(log_file, mode='a')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    return logger

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Verify uploaded model weights match local weights")
    parser.add_argument("--org_name", type=str, default="allenai", 
                       help="Organization name on Hugging Face Hub")
    parser.add_argument("--debug", action="store_true",
                       help="Debug mode: test only one small model (4M)")
    parser.add_argument("--debug_seed_diff", action="store_true",
                       help="Debug mode: test that different seeds have different weights (sanity check)")
    parser.add_argument("--test_functions", action="store_true",
                       help="Test basic functionality without downloading models")
    parser.add_argument("--repo_filter", type=str,
                       help="Test only repositories containing this string")
    parser.add_argument("--single_repo", type=str,
                       help="Test only this specific repository name")
    parser.add_argument("--tolerance", type=float, default=1e-6,
                       help="Tolerance for weight comparison")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for branch sampling")
    parser.add_argument("--dry_run", action="store_true",
                       help="Dry run: show what would be tested without downloading")
    parser.add_argument("--max_repos", type=int,
                       help="Maximum number of repositories to test (for large-scale testing)")
    parser.add_argument("--log_file", type=str, 
                       default=os.path.join(SCRIPT_DIR, "verification.log"),
                       help="Path to log file (default: verification.log in script directory)")
    args = parser.parse_args()
    
    # Set up logging
    logger = setup_logging(args.log_file)
    logger.info("="*80)
    logger.info(f"Starting model verification at {datetime.now()}")
    logger.info(f"Arguments: {vars(args)}")
    logger.info("="*80)
    
    # Set random seed for reproducible sampling
    random.seed(args.seed)
    logger.info(f"Set random seed to {args.seed}")
    
    # Handle test mode
    if args.test_functions:
        logger.info("Running basic functionality tests")
        test_basic_functionality()
        return
    
    # Check for HF token
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        print("Warning: HF_TOKEN environment variable not set. Public repositories only.")
    
    # Initialize HF API
    api = HfApi()
    
    # Load data
    print("Loading repository names and checkpoint data...")
    logger.info("Loading repository names and checkpoint data...")
    all_repo_names = load_repo_names()
    checkpoints_by_repo = load_checkpoints()
    logger.info(f"Loaded {len(all_repo_names)} repository names and {len(checkpoints_by_repo)} checkpoint groups")
    
    # Handle seed difference debug mode
    if args.debug_seed_diff:
        logger.info("Running seed difference debug mode")
        with tempfile.TemporaryDirectory(prefix="seed_diff_test_") as temp_dir:
            debug_seed_differences(api, args.org_name, checkpoints_by_repo, temp_dir, args.tolerance)
        return
    
    # Apply repository filters
    if args.single_repo:
        # Check only the specific repository
        repo_names = [args.single_repo] if args.single_repo in all_repo_names else []
        if not repo_names:
            logger.error(f"Repository '{args.single_repo}' not found in repo_names.txt")
            print(f"Repository '{args.single_repo}' not found in repo_names.txt")
            return
        logger.info(f"Testing single repository: {args.single_repo}")
    elif args.debug:
        # Find a small model (4M) for quick testing
        repo_names = [name for name in all_repo_names if name.endswith('-4M')]
        if repo_names:
            repo_names = [repo_names[0]]  # Just test one
            logger.info(f"Debug mode: testing single 4M model: {repo_names[0]}")
        else:
            logger.error("No 4M models found for debug mode")
            print("No 4M models found for debug mode")
            return
    elif args.repo_filter:
        repo_names = [name for name in all_repo_names if args.repo_filter in name]
        if not repo_names:
            logger.error(f"No repositories found matching filter '{args.repo_filter}'")
            print(f"No repositories found matching filter '{args.repo_filter}'")
            return
        logger.info(f"Filtered to {len(repo_names)} repositories matching '{args.repo_filter}'")
    else:
        repo_names = all_repo_names
        logger.info("Testing all repositories")
    
    # Apply max_repos limit if specified
    if args.max_repos and len(repo_names) > args.max_repos:
        repo_names = repo_names[:args.max_repos]
        logger.info(f"Limited to first {args.max_repos} repositories")
        print(f"Limited to first {args.max_repos} repositories")
    
    logger.info(f"Final repository list: {len(repo_names)} repositories to test")
    print(f"Testing {len(repo_names)} repositories")
    
    # Dry run mode
    if args.dry_run:
        logger.info("DRY RUN MODE - Showing what would be tested")
        print("\nDRY RUN MODE - Showing what would be tested:")
        for repo_name in repo_names:
            if repo_name in checkpoints_by_repo:
                branches = get_available_branches(api, args.org_name, repo_name)
                if branches:
                    sample_branch = random.choice(branches)
                    local_path = find_local_path(sample_branch, checkpoints_by_repo[repo_name])
                    logger.info(f"Would test {repo_name}: branch '{sample_branch}' -> {local_path}")
                    print(f"  {repo_name}: would test branch '{sample_branch}' -> {local_path}")
                else:
                    logger.warning(f"No branches found for {repo_name}")
                    print(f"  {repo_name}: no branches found")
            else:
                logger.warning(f"No checkpoint data for {repo_name}")
                print(f"  {repo_name}: no checkpoint data")
        return
    
    # Create temporary directory for downloads
    with tempfile.TemporaryDirectory(prefix="model_verification_") as temp_dir:
        logger.info(f"Using temporary directory: {temp_dir}")
        print(f"Using temporary directory: {temp_dir}")
        
        results = []
        successful_verifications = 0
        
        logger.info("Starting verification loop")
        for repo_name in tqdm(repo_names, desc="Verifying repositories"):
            if repo_name not in checkpoints_by_repo:
                logger.warning(f"No checkpoint data found for {repo_name}")
                print(f"No checkpoint data found for {repo_name}")
                results.append({
                    "repo_name": repo_name,
                    "success": False,
                    "message": "No checkpoint data",
                    "branch": ""
                })
                continue
            
            success, message, branch = verify_single_repo(
                api, args.org_name, repo_name, checkpoints_by_repo[repo_name], temp_dir, logger
            )
            
            results.append({
                "repo_name": repo_name,
                "success": success,
                "message": message,
                "branch": branch
            })
            
            if success:
                successful_verifications += 1
                print(f"✓ {repo_name} (branch: {branch})")
            else:
                print(f"✗ {repo_name} (branch: {branch}): {message}")
            
            # Small delay to avoid overwhelming the system
            time.sleep(0.5)
    
    # Print summary
    logger.info("="*80)
    logger.info("VERIFICATION SUMMARY")
    logger.info("="*80)
    logger.info(f"Total repositories tested: {len(results)}")
    logger.info(f"Successful verifications: {successful_verifications}")
    logger.info(f"Failed verifications: {len(results) - successful_verifications}")
    logger.info(f"Success rate: {successful_verifications/len(results)*100:.1f}%")
    
    print(f"\n{'='*80}")
    print(f"VERIFICATION SUMMARY")
    print(f"{'='*80}")
    print(f"Total repositories tested: {len(results)}")
    print(f"Successful verifications: {successful_verifications}")
    print(f"Failed verifications: {len(results) - successful_verifications}")
    print(f"Success rate: {successful_verifications/len(results)*100:.1f}%")
    
    # Show failures
    failures = [r for r in results if not r["success"]]
    if failures:
        logger.info(f"FAILED VERIFICATIONS ({len(failures)}):")
        print(f"\nFAILED VERIFICATIONS ({len(failures)}):")
        for failure in failures:
            failure_msg = f"{failure['repo_name']} (branch: {failure['branch']}): {failure['message']}"
            logger.info(f"  FAILED: {failure_msg}")
            print(f"  {failure_msg}")
    
    # Save detailed results
    results_file = os.path.join(SCRIPT_DIR, "model_verification_results.json")
    with open(results_file, 'w') as f:
        json.dump({
            "summary": {
                "total_tested": len(results),
                "successful": successful_verifications,
                "failed": len(results) - successful_verifications,
                "success_rate": successful_verifications/len(results)*100,
                "tolerance": args.tolerance,
                "random_seed": args.seed,
                "timestamp": datetime.now().isoformat(),
                "log_file": args.log_file
            },
            "results": results
        }, f, indent=2)
    
    logger.info(f"Detailed results saved to: {results_file}")
    logger.info(f"Verification completed at {datetime.now()}")
    logger.info("="*80)
    print(f"\nDetailed results saved to: {results_file}")
    print(f"Log file: {args.log_file}")

if __name__ == "__main__":
    main()
