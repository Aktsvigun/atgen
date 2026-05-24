#!/bin/bash

# Script to run subset selection with multiple query sizes
# Query sizes: 10, 25, 50, 100, 250, 500, 1000

set -e  # Exit on any error

# Array of query sizes to test
query_sizes=(50 100 250 500 1000)
seeds=(1234567 123456 12345)

# Base command components
export VLLM_PORT=8010
export CUDA_VISIBLE_DEVICES=0
export DATA_NAME=gsm8k

echo "Starting subset selection runs with multiple query sizes..."
echo "Query sizes to test: ${query_sizes[@]}"
echo "=============================================="

for seed in "${seeds[@]}"; do
        for query_size in "${query_sizes[@]}"; do
                echo
                echo "Running with al.query_size=${query_size} and seed=${seed}..."
                echo "Time: $(date)"
                echo "----------------------------------------"
                
                HYDRA_CONFIG_NAME=base python src/atgen/run_scripts/run_subset_selection.py \
                        al=random \
                        data=${DATA_NAME} \
                        al.query_size=${query_size} \
                        al.eval_zero_iteration=false \
                        seed=${seed}
        done
done

echo "All runs finished."