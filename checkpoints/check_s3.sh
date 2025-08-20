#!/bin/bash

# Requires: awscli, jq

JSONL_FILE="weka_paths.jsonl"

while IFS= read -r line; do
    # Extract the checkpoints_location field
    weka_path=$(echo "$line" | jq -r '.checkpoints_location')
    # Replace weka://oe-eval-default with s3://oe-eval-default
    s3_path=$(echo "$weka_path" | sed 's|^weka://|s3://|')
    # Check if the path exists on S3
    if aws s3 ls "$s3_path" > /dev/null 2>&1; then
        echo "Exists: $s3_path"
    else
        echo "Missing: $s3_path"
    fi
done < "$JSONL_FILE"