
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
parallel --joblog progress_model_card.log --resume --progress -j10 python upload_model_card.py --repo_id allenai/{} :::repo_names.txt
```


## upload eval metrics results
```
python upload_eval.py --private
```

## change privacy eg

```
from huggingface_hub import HfApi

# Define your repository details
repo_name = "<>"  # Repository name
organization = "allenai"  # Organization name if applicable
repo_id = f"{organization}/{repo_name}"  # Full repo ID
visibility = "private"  # Change to 'private' to make it private

# Initialize the API
api = HfApi()

# Update the visibility
api.update_repo_visibility(repo_id=repo_id, private=(visibility == "private"), token=hf_token)

print(f"Repository '{repo_id}' is now {visibility}.")
```