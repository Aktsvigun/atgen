#!/bin/bash

# Script to run subset selection with multiple query sizes
# Query sizes: 10, 25, 50, 100, 250, 500, 1000

set -e  # Exit on any error

# Array of query sizes to test
query_sizes=(250)
# Base command components
export VLLM_PORT=8001
export CUDA_VISIBLE_DEVICES=0
export DATA_NAME=gsm8k

echo "Starting subset selection runs with multiple query sizes..."
echo "Query sizes to test: ${query_sizes[@]}"
echo "=============================================="
HYDRA_CONFIG_NAME=base python src/atgen/run_scripts/run_subset_selection.py \
        al=kek \
        data=${DATA_NAME} \
        al.query_size=100 \
        al.eval_zero_iteration=false \
        seed=981312

HYDRA_CONFIG_NAME=base python src/atgen/run_scripts/run_subset_selection.py \
        al=kek2 \
        data=${DATA_NAME} \
        al.query_size=100 \
        al.eval_zero_iteration=false \
        seed=981312

HYDRA_CONFIG_NAME=base python src/atgen/run_scripts/run_subset_selection.py \
        al=kek \
        data=${DATA_NAME} \
        al.query_size=100 \
        al.eval_zero_iteration=false \
        seed=9123

HYDRA_CONFIG_NAME=base python src/atgen/run_scripts/run_subset_selection.py \
        al=kek2 \
        data=${DATA_NAME} \
        al.query_size=100 \
        al.eval_zero_iteration=false \
        seed=9123

HYDRA_CONFIG_NAME=base python src/atgen/run_scripts/run_subset_selection.py \
        al=kek \
        data=${DATA_NAME} \
        al.query_size=100 \
        al.eval_zero_iteration=false \
        seed=213

HYDRA_CONFIG_NAME=base python src/atgen/run_scripts/run_subset_selection.py \
        al=kek2 \
        data=${DATA_NAME} \
        al.query_size=100 \
        al.eval_zero_iteration=false \
        seed=213

# Loop through each query size
for query_size in "${query_sizes[@]}"; do
    echo
    echo "Running with al.query_size=${query_size}..."
    echo "Time: $(date)"
    echo "----------------------------------------"
    
    # Run the command
    HYDRA_CONFIG_NAME=base python src/atgen/run_scripts/run_subset_selection.py \
        al=kek \
        data=${DATA_NAME} \
        al.query_size=${query_size} \
        al.eval_zero_iteration=false \
        seed=981312

    # Run the command
    HYDRA_CONFIG_NAME=base python src/atgen/run_scripts/run_subset_selection.py \
        al=kek \
        data=${DATA_NAME} \
        al.query_size=${query_size} \
        al.eval_zero_iteration=false \
        seed=9123

    # Run the command
    HYDRA_CONFIG_NAME=base python src/atgen/run_scripts/run_subset_selection.py \
        al=kek \
        data=${DATA_NAME} \
        al.query_size=${query_size} \
        al.eval_zero_iteration=false \
        seed=213
    
    echo "Completed al.query_size=${query_size}"
    echo "----------------------------------------"
done

# Loop through each query size
for query_size in "${query_sizes[@]}"; do
    echo
    echo "Running with al.query_size=${query_size}..."
    echo "Time: $(date)"
    echo "----------------------------------------"
    
    # Run the command
    HYDRA_CONFIG_NAME=base python src/atgen/run_scripts/run_subset_selection.py \
        al=kek2 \
        data=${DATA_NAME} \
        al.query_size=${query_size} \
        al.eval_zero_iteration=false \
        seed=981312

    # Run the command
    HYDRA_CONFIG_NAME=base python src/atgen/run_scripts/run_subset_selection.py \
        al=kek2 \
        data=${DATA_NAME} \
        al.query_size=${query_size} \
        al.eval_zero_iteration=false \
        seed=9123

    # Run the command
    HYDRA_CONFIG_NAME=base python src/atgen/run_scripts/run_subset_selection.py \
        al=kek2 \
        data=${DATA_NAME} \
        al.query_size=${query_size} \
        al.eval_zero_iteration=false \
        seed=213
    
    echo "Completed al.query_size=${query_size}"
    echo "----------------------------------------"
done

echo
echo "All runs completed successfully!"
echo "Final time: $(date)" 