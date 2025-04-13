from pathlib import Path
import os, sys, warnings
sys.path.append(os.path.dirname(os.getcwd()))
from utils.constants import DATA_DIR

import warnings
import numpy as np
import pandas as pd
from utils.scaling_laws import fit_all_mixes
from utils.stats import compute_decision_accuracy
from remote.hf import push_parquet_to_hf, pull_predictions_from_hf

warnings.filterwarnings("ignore", category=RuntimeWarning) # supress function fitting warnings
pd.set_option('display.max_columns', None) # display all pandas cols

SCALING_LAW_CONFIGS = [
    '3_param',
    '3_param-helper_points',
    '3_param-step2=0.5',
    '3_param-helper_points-step2=0.5',
    '5_param-ai2',
    '5_param-1_step-ai2',
    '3_param-1_step',
    '2_param',
    '3_param-intermediate',
    '3_param-intermediate-helper_points'
]

ALL_METRICS = [
    'primary_metric', 'correct_prob', 'correct_prob_per_token', 'correct_prob_per_char', 'margin', 'margin_per_token', 'margin_per_char', 'total_prob', 'total_prob_per_token', 'total_prob_per_char', 'norm_correct_prob', 'norm_correct_prob_per_token', 'norm_correct_prob_per_char',
    'acc_raw', 'acc_per_token', 'acc_per_char'
]

TARGET_SIZE = '1B'

def generate_setups(setup_types, model_sizes):
    """
    Generate fitting configurations by excluding different combinations of small models.
    >>> [
    >>>     3_param-intermediate
    >>>     3_param-intermediate-no_8M
    >>>     3_param-intermediate-no_8M_no_10M
    >>>     3_param-intermediate-no_8M_no_10M_no_14M
    >>>     3_param-intermediate-no_8M_no_10M_no_14M_no_16M
    >>>     ...
    >>> ]
    """
    def all_combinations(lst):
        result = []
        for end in range(3, len(lst)+1):
            result.append(lst[:end])
        for start in range(1, len(lst) - 2):
            result.append(lst[start:])
        return result

    setups = []
    for setup_type in setup_types:
        for combination in all_combinations(model_sizes):
            compliment = [size for size in model_sizes if size not in combination]
            setup_name = f"{setup_type}" + (f"-no_{'_no_'.join(compliment)}" if compliment else "")
            setups.append(setup_name)
    return setups

def run_ladder_fits(data_path, results_path, dry_run=False):
    local_path = pull_predictions_from_hf(data_path, split_name='benchmarks')
    df = pd.read_parquet(local_path)
    print(f'Loaded {len(df):,} model evaluations')

    # For my results, I'm only using the last seed for analysis
    df = df[df['seed'] == 6198]

    # Incomplete results
    df = df[~df['size'].isin(['4M', '6M'])]

    MIXES = df['group'].unique()
    SIZES = df['size'].unique()
    MULT  = df['chinchilla'].unique()

    MODELS = df['model'].unique()
    TASKS  = df['task'].unique()

    # sort sizes
    SIZES = sorted(SIZES, key=lambda x: float(x.replace('M', '').replace('B', '')) * (1000 if 'B' in x else 1))
    SETUPS = generate_setups(SCALING_LAW_CONFIGS, SIZES[:-1])

    if dry_run:
        # Example table on two setups
        results = fit_all_mixes(
            df,
            all_models=MODELS,
            mixes=MIXES[:2], # only test with 2 data mixes
            tasks=TASKS[:2], # only test with 2 tasks
            setups=SETUPS,
            y_metrics=["primary_metric", "acc_per_char", "correct_logit_per_char"], # only test with 3 metrics
            x_metric="correct_logit_per_byte",
        )
    else:
        results = fit_all_mixes(
            df,
            all_models=MODELS,
            mixes=MIXES,
            tasks=TASKS,
            setups=SETUPS,
            y_metrics=ALL_METRICS,
            x_metric="correct_logit_per_byte",
            quiet=True
        )

    # Remove any results where we did not have data to fit the scaling prediction (some of WinoGrande, BoolQ)
    results = results[np.isfinite(results['step_1_pred'])].reset_index(drop=True)

    results = compute_decision_accuracy(df, results, TARGET_SIZE)

    results.to_csv(results_path, index=False)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', type=str, default='allenai/DataDecide-eval-results-davidh',
                      help='HuggingFace dataset to load results from')
    parser.add_argument('--result-path', type=str, default='ladder_predictions.csv',
                      help='Path to save results CSV (default: ladder_predictions.csv)')
    parser.add_argument('--dry-run', action='store_true',
                      help='Run on subset of data for testing')
    parser.add_argument('--push-to-hf', type=str,
                      help='HuggingFace dataset name to push results to (optional)')
    args = parser.parse_args()
    
    data_path    = args.data_path
    results_path = Path(DATA_DIR) / args.result_path
    
    run_ladder_fits(data_path, results_path, dry_run=args.dry_run)

    if args.push_to_hf:
        push_parquet_to_hf(
            parquet_file_path=results_path,
            hf_dataset_name=args.push_to_hf,
            split_name='scaling_law_fit',
            overwrite=True,
            private=True,
        )