## Setup
```
sudo apt-get install git-lfs
pip install -r requirements.txt
```


## upload final checkpoints with default seed to main branches
```
python make_final_checkpoint_script.py
parallel --joblog progress.log --resume --progress -j1 < final_checkpoint_upload_script.sh
```

## reupload the model card

```
parallel --joblog progress_model_card.log --resume --progress -j10 python upload_model_card.py --repo_id allenai/{} < repo_names.txt
```

## upload checkpoints
```
parallel --joblog progress_checkpoint.log --resume --progress -j10 python upload_checkpoints.py --repo_name {} < repo_names.txt
```

## verify uploaded models
Verify that uploaded model weights match local weights on Weka:

```bash
# Test basic functionality (no downloads)
python verify_uploaded_models.py --test_functions

# Debug mode: test one small model only
python verify_uploaded_models.py --debug

# Debug mode: test that different seeds have different weights (sanity check)
python verify_uploaded_models.py --debug_seed_diff

# Test all models (randomly samples one branch per repo)
python verify_uploaded_models.py

# Test specific model size
python verify_uploaded_models.py --repo_filter 60M

# Test a specific repository
python verify_uploaded_models.py --single_repo DataDecide-falcon-4M

# Dry run: show what would be tested without downloading
python verify_uploaded_models.py --dry_run
```

The script:
- Randomly samples one branch per repository
- Downloads the model from HF Hub to a temporary directory
- Loads the corresponding local model from Weka
- Compares weights tensor by tensor
- Automatically cleans up downloaded files
- Saves results to `model_verification_results.json`

Debug options:
- `--test_functions`: Test basic functionality without downloading models
- `--debug`: Test only one small (4M) model for quick verification
- `--debug_seed_diff`: Sanity check that different seeds produce different weights

## upload eval metrics results
```
python upload_eval.py --private
```

## change privacy

```
parallel --joblog progress_public.log --resume --progress python make_repo_public.py --repo_id allenai/{} < repo_names.txt
```