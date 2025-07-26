import os
import argparse
import json
import re
import time
from huggingface_hub import upload_file, HfApi
from pprint import pprint
import tqdm
try:
    from huggingface_hub.errors import RepositoryNotFoundError, RevisionNotFoundError
except ImportError:
    try:
        from huggingface_hub.utils._errors import RepositoryNotFoundError, RevisionNotFoundError
    except ImportError:
        # Fallback for older versions
        RepositoryNotFoundError = Exception
        RevisionNotFoundError = Exception

# Constants
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINTS_FILE = os.path.join(SCRIPT_DIR, "../checkpoints/weka_paths.jsonl")

SEED_MAPPING = {
        6198: "default",
        14: "small aux 2",
        15: "small aux 3",
        4: "large aux 2",
        5: "large aux 3"
    }

WEKA_PATH = "/data/input/"

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

# Mapping for model names to repo names
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
    
    print(f"Filtered revisions for {model_size} model: {len(filtered_revisions)}/{len(revisions)} (max step: {max_step})")
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

def upload_optimizer_state(repo_id, checkpoint_path, branch_name, hf_token):
    """Upload optimizer state to the Hugging Face Hub."""
    optim_path = os.path.join(checkpoint_path, "optim.pt")
    
    if not os.path.exists(optim_path):
        print(f"Warning: optim.pt not found at {optim_path}, skipping.")
        return False
    
    try:
        upload_file(
            path_or_fileobj=optim_path,
            path_in_repo="training/optim.pt",
            repo_id=repo_id,
            revision=branch_name,
            token=hf_token
        )
        return True
    except Exception as e:
        print(f"Error uploading optimizer state for {branch_name}: {e}")
        return False

def check_optim_exists(api: HfApi, repo_id: str, branch_name: str) -> bool:
    """Check if optim.pt already exists in the specified branch."""
    try:
        # First check if branch exists
        try:
            api.repo_info(repo_id=repo_id, revision=branch_name)
        except RevisionNotFoundError:
            return False
        
        # List files in the branch to check for training/optim.pt
        try:
            files = api.list_repo_files(
                repo_id=repo_id,
                revision=branch_name,
                repo_type="model"
            )
            
            optim_file = "training/optim.pt"
            # Small delay to avoid rate limiting
            time.sleep(0.01)
            return optim_file in files
            
        except Exception:
            return False
            
    except Exception:
        return False

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Upload optimizer states to a Hugging Face repository.")
    parser.add_argument("--repo_name", type=str, required=True, help="The Hugging Face repo name (e.g., DataDecide-falcon-and-cc-qc-tulu-10p-60M)")
    parser.add_argument("--org_name", type=str, default="allenai", help="The organization name on Hugging Face Hub.")
    parser.add_argument("--num_revisions", type=int, default=5, help="Number of revisions to upload optimizer states for (sampled evenly across all revisions)")
    parser.add_argument("--force", action="store_true", help="Force upload even if optim.pt already exists in the branch")
    args = parser.parse_args()

    hf_token = os.getenv("HF_TOKEN")  # Ensure your Hugging Face token is set as an environment variable
    assert hf_token is not None, "Please set the HF_TOKEN environment variable."

    # Get checkpoints
    repo_name = args.repo_name
    checkpoints = get_checkpoints(repo_name)
    full_repo_name = f"{args.org_name}/{repo_name}"

    print(f"Uploading optimizer states for repository: {full_repo_name}")
    print(f"Number of revisions per seed: {args.num_revisions}")
    print(f"Force upload (ignore existing): {'Yes' if args.force else 'No'}")

    # Initialize Hugging Face API
    api = HfApi()

    # Check if the repository exists
    try:
        api.repo_info(repo_id=full_repo_name)
        print(f"Repository '{full_repo_name}' exists.")
    except Exception as e:
        raise ValueError(f"Repository '{full_repo_name}' does not exist or is inaccessible. Please create it first.") from e

    # Get the list of branches from the repository
    git_refs = api.list_repo_refs(repo_id=full_repo_name)
    existing_branches = [branch.name for branch in git_refs.branches]

    # Upload optimizer states for selected revisions
    for checkpoint in checkpoints:
        seed = int(checkpoint["model_name"].split("-")[-1])
        seed_name = SEED_MAPPING[seed].replace(" ", "-")
        location = checkpoint['checkpoints_location'].replace("weka://oe-eval-default/", WEKA_PATH)
        
        # Filter revisions by training schedule first
        filtered_revisions = filter_revisions_by_schedule(checkpoint["revisions"], checkpoint["model_name"])
        
        # Sample revisions evenly from filtered list
        sampled_revisions = sample_revisions_evenly(filtered_revisions, args.num_revisions)
        
        print(f"\nProcessing checkpoint for seed {seed_name}:")
        print(f"Total revisions: {len(checkpoint['revisions'])}")
        print(f"Filtered revisions (within schedule): {len(filtered_revisions)}")
        print(f"Sampled revisions: {len(sampled_revisions)}")
        print(f"Sampled steps: {[extract_step_number(rev) for rev in sampled_revisions]}")
        
        progress_bar = tqdm.tqdm(total=len(sampled_revisions), desc=f"Uploading optim states (seed {seed_name})")
        
        successful_uploads = 0
        skipped_existing = 0
        for step in sampled_revisions:
            branch_name = f"{step.replace('-unsharded-hf','')}-seed-{seed_name}"
            
            # Check if branch exists
            if branch_name not in existing_branches:
                print(f"\nWarning: Branch {branch_name} does not exist in repository, skipping.")
                progress_bar.update(1)
                continue
            
            # Check if optim.pt already exists in the branch (unless forced)
            if not args.force and check_optim_exists(api, full_repo_name, branch_name):
                progress_bar.set_postfix_str(f"Skipping {branch_name} (optim.pt exists)")
                skipped_existing += 1
                progress_bar.update(1)
                continue
                
            progress_bar.set_postfix_str(f"Uploading to {branch_name}")
            
            checkpoint_path = os.path.join(location, step)
            success = upload_optimizer_state(full_repo_name, checkpoint_path, branch_name, hf_token)
            
            if success:
                successful_uploads += 1
            
            progress_bar.update(1)
        
        progress_bar.close()
        print(f"Successfully uploaded optimizer states to {successful_uploads}/{len(sampled_revisions)} branches for seed {seed_name}")
        if skipped_existing > 0:
            print(f"Skipped {skipped_existing} branches that already had optim.pt files")

if __name__ == "__main__":
    main()
