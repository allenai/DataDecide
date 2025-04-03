
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