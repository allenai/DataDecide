
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