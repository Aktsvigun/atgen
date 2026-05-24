#!/bin/bash

# Parallel version of run_random.sh: distributes (seed × query_size) jobs
# across all 8 GPUs via a shared work queue. Each GPU keeps pulling jobs
# until the queue is drained, so slower jobs don't leave GPUs idle.

set -u

query_sizes=(20 50 100 250 500 1000 2500 5000)
# seeds=(1234567 123456 12345)
seeds=(42 424242 42424242)

export DATA_NAME=gsm8k
NUM_GPUS=8
BASE_VLLM_PORT=8010

LOG_DIR="run/logs/random_parallel_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"
echo "Logs: $LOG_DIR"

# Build job list
jobs=()
for seed in "${seeds[@]}"; do
        for qs in "${query_sizes[@]}"; do
                jobs+=("$seed $qs")
        done
done
echo "Total jobs: ${#jobs[@]}, GPUs: $NUM_GPUS"
echo "=============================================="

# Shared FIFO queue: workers read one job at a time. Reads from the same
# fd are serialized by the kernel, so each line goes to exactly one worker.
fifo=$(mktemp -u)
mkfifo "$fifo"
exec 3<>"$fifo"
rm "$fifo"

for job in "${jobs[@]}"; do
        echo "$job" >&3
done
# Sentinel per worker so each one exits cleanly
for ((i = 0; i < NUM_GPUS; i++)); do
        echo "__DONE__" >&3
done

run_one() {
        local gpu=$1 seed=$2 qs=$3
        local port=$((BASE_VLLM_PORT + gpu))
        local log="$LOG_DIR/gpu${gpu}_seed${seed}_qs${qs}.log"
        echo "[gpu $gpu] start seed=$seed qs=$qs (port=$port) -> $log"
        CUDA_VISIBLE_DEVICES=$gpu VLLM_PORT=$port \
                HYDRA_CONFIG_NAME=base \
                python src/atgen/run_scripts/run_subset_selection.py \
                al=random \
                data="$DATA_NAME" \
                al.query_size="$qs" \
                al.eval_zero_iteration=false \
                seed="$seed" \
                >"$log" 2>&1
        local rc=$?
        if [[ $rc -eq 0 ]]; then
                echo "[gpu $gpu] done  seed=$seed qs=$qs"
        else
                echo "[gpu $gpu] FAIL  seed=$seed qs=$qs (rc=$rc, see $log)"
        fi
}

for ((gpu = 0; gpu < NUM_GPUS; gpu++)); do
        (
                while IFS= read -u 3 -r job; do
                        [[ "$job" == "__DONE__" ]] && break
                        read -r seed qs <<<"$job"
                        run_one "$gpu" "$seed" "$qs"
                done
        ) &
done

wait
exec 3>&-
echo "All runs finished."
