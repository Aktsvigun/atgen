#!/bin/bash

# Script to run subset selection with multiple query sizes
# Query sizes: 10, 25, 50, 100, 250, 500, 1000

set -e  # Exit on any error

# Array of query sizes to test
query_sizes=(250 500 750 1000 1250 1500 2000 2500)

# Base command components
export VLLM_PORT=8001
export CUDA_VISIBLE_DEVICES=0
export DATA_NAME=gsm8k

echo "Starting subset selection runs with multiple query sizes..."
echo "Query sizes to test: ${query_sizes[@]}"
echo "=============================================="
HYDRA_CONFIG_NAME=base python src/atgen/run_scripts/run_subset_selection.py \
        al=kek6 \
        data=${DATA_NAME} \
        al.query_size=100 \
        al.eval_zero_iteration=false \
        seed=9781312

# Loop through each query size
for query_size in "${query_sizes[@]}"; do
    echo
    echo "Running with al.query_size=${query_size}..."
    echo "Time: $(date)"
    echo "----------------------------------------"
    
    # Run the command
    HYDRA_CONFIG_NAME=base python src/atgen/run_scripts/run_subset_selection.py \
        al=kek6 \
        data=${DATA_NAME} \
        al.query_size=${query_size} \
        al.eval_zero_iteration=false \
        seed=824242

    HYDRA_CONFIG_NAME=base python src/atgen/run_scripts/run_subset_selection.py \
        al=kek6 \
        data=${DATA_NAME} \
        al.query_size=${query_size} \
        al.eval_zero_iteration=false \
        seed=814246
    HYDRA_CONFIG_NAME=base python src/atgen/run_scripts/run_subset_selection.py \
        al=kek6 \
        data=${DATA_NAME} \
        al.query_size=${query_size} \
        al.eval_zero_iteration=false \
        seed=3994241

    echo "Completed al.query_size=${query_size}"
    echo "----------------------------------------"
done