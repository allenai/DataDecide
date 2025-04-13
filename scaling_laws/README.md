This contains utilites for running scaling law fits on our results.

### Quick Start

```sh
git lfs install # .ipynb files are tracked with git lfs! (brew install git-lfs)
pip install -r requirements.txt

# (Optional) If you want to edit the ladder fitting code, you can install locally!
git clone -b datados https://github.com/allenai/OLMo-ladder OLMo-ladder
cd OLMo-ladder
pip install -e ".[plotting]"
```

**Folder structure**
- `remote/` - contains utilities for pulling evaluation results into a single prediction file
- `utils/` - contains dataloading, ladder usage and plotting code

### Fit scaling laws

```sh
# Fit scaling laws with "--dry-run" to use a subset of results for tesing
python fit_scaling_laws.py --result-path ladder_predictions_dry_run.csv --dry-run

# Fit on all metrics!
python fit_scaling_laws.py --result-path ladder_predictions.csv

# Fit + push results to HuggingFace
python fit_scaling_laws.py --result-path ladder_predictions.csv --push-to-hf allenai/DataDecide-eval-results

# Render tables in our paper
python render_tables.py --result-path ladder_predictions.csv
```

### Notebooks

Scaling laws often have to be de-bugged by looking at the curve fits themselves. Here are some utilities for doing so:

- `example_results.ipynb` -- Example of rendering training curves
- `scaling_law_figures.ipynb` -- Render paper figures to inspect scaling law fits
- `scaling_law_tables.ipynb` -- Interactive version of table rendering
- `math_and_code.ipynb` -- Tables for results on MBPP, HumanEval, etc.