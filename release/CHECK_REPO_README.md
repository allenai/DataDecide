# Repository Status Checker

This script checks the status of Hugging Face repositories listed in `repo_names.txt` and verifies they have the expected branches and commits based on the checkpoint data in `../checkpoints/weka_paths.jsonl`.

**⚠️ IMPORTANT: This script does NOT download any model files or data. It only uses Hugging Face metadata APIs to check repository existence, branch existence, and commit history.**

## What it checks:

1. **Repository existence**: Verifies each repository in `repo_names.txt` exists on Hugging Face
2. **Expected branches**: For each repository, checks if all expected branches exist based on the steps and seeds from `weka_paths.jsonl`
3. **Model commits**: Verifies that each branch has commits beyond the initial commit (indicating model files were uploaded)

## Safety Features:

- **No Downloads**: Uses only metadata APIs (`repo_info`, `list_repo_refs`, `list_repo_commits`)
- **Rate Limited**: Includes delays to avoid hitting API limits
- **Read-Only**: Only reads repository information, never modifies anything

## Installation:

```bash
pip install -r check_repo_requirements.txt
```

## Usage:

```bash
# Basic usage (checks allenai org by default)
python check_repo_status.py

# Check a different organization
python check_repo_status.py --org_name your_org_name

# Show detailed output during checking
python check_repo_status.py --detailed

# === LIMITED TESTING OPTIONS ===

# Test just one specific repository
python check_repo_status.py --single_repo "DataDecide-falcon-60M"

# Test repositories matching a pattern
python check_repo_status.py --repo_filter "falcon-60M"

# Test only branches with a specific step
python check_repo_status.py --step_filter "step1250"

# Combine filters: test one repo and one step
python check_repo_status.py --single_repo "DataDecide-falcon-60M" --step_filter "step1250"

# Test falcon repos with only step1250 branches
python check_repo_status.py --repo_filter "falcon" --step_filter "step1250" --detailed

# Use more concurrent workers (default: 5)
python check_repo_status.py --max_workers 10
```

## Output:

The script provides:
- A progress bar showing checking status
- A summary of missing repositories and branch issues
- A detailed JSON report saved as `repo_status_check.json`

## Expected branch naming:

Branches are named as: `{step}-seed-{seed_name}`
- `step`: from the revisions in weka_paths.jsonl (e.g., "step1250", "step5000")
- `seed_name`: mapped from seed numbers (6198 -> "default", 14 -> "small-aux-2", etc.)

Example branch names:
- `step1250-seed-default`
- `step5000-seed-small-aux-2`

## Rate limiting:

The script includes delays to avoid hitting Hugging Face API rate limits. It processes repositories sequentially with small delays between API calls.
