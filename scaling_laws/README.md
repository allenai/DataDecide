### Quick Start
```sh
git lfs install # .ipynb files are tracked with git lfs! (brew install git-lfs)
pip install -r requirements.txt

# (Optional) If you want to edit the ladder fitting code, you can install locally!
git clone -b datados https://github.com/allenai/OLMo-ladder OLMo-ladder
cd OLMo-ladder
pip install -e ".[plotting]"
```