## Setup:
```
conda create -n DataDecide-single-scale python=3.11
conda activate DataDecide-single-scale
pip install -r requirements.txt
```

## Process all results given S3 dir path:
```
python process_s3_to_csv.py \
    --s3 s3://<> \
    --csv_name "results_ladder_5xC.csv"
```