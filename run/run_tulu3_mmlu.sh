#!/bin/bash
# Random-selection AL sweep on Tulu-3 (train pool) → MMLU (eval).
#
# This is the literature-canonical instruction-tuning AL setup. The train pool
# comes from `allenai/tulu-3-sft-mixture` (flat user+assistant rows after
# preprocess_tulu3.py); evaluation is on `cais/mmlu` (multi-choice) via the
# new eval_dataset overlay in configs/data/tulu3_mmlu.yaml.
#
# Prereqs (one-time):
#   python -m atgen.utils.data.preprocess_tulu3 --out cache/tulu3_preprocessed
#   python -m atgen.utils.data.preprocess_mmlu  --out cache/mmlu_preprocessed
#
# (For a smaller pilot pool, use --max-rows 50000 on the tulu3 preprocessor.)
#
# Sweeps query_size x seed for `al=random` on `data=tulu3_mmlu`, distributing
# jobs across 8 GPUs. Random hex prefix on experiment_name prevents folder
# collisions when 8 parallel jobs share the same ${now:%H-%M-%S}.
#
# Aggregate after:
#   python -m atgen.utils.aggregate_metrics outputs/$(date +%F)/*_tulu3_mmlu_random_[0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f]
#
# Headline metric on MMLU is `exact_match` (single-letter A/B/C/D).

set -uo pipefail

NUM_GPUS=8
DATA=tulu3_mmlu
PY=src/atgen/run_scripts/run_subset_selection.py
EXP_NAME=tulu3_mmlu_random

query_sizes=(100 500 2500 10000)
seeds=(42 1234567 31337)

LOG_DIR=run/_tulu3_mmlu_logs/$(date +%F_%H-%M-%S)
mkdir -p "$LOG_DIR"

# Same Qwen3-Instruct-2507 fixes from the MuSiQue/Qwen3-4B-Instruct rounds:
#   - empty assistant_response_start (no <think></think> stub on non-thinking models)
#   - greedy decoding (deterministic, MCQ-friendly)
COMMON_OVERRIDES="model.assistant_response_start='' inference.temperature=0 inference.top_p=1.0 inference.top_k=-1 inference.presence_penalty=0"

echo "Launching $(( ${#query_sizes[@]} * ${#seeds[@]} )) jobs across ${NUM_GPUS} GPUs"
echo "Logs -> $LOG_DIR"
echo "Outputs -> outputs/$(date +%F)/<HH-MM-SS>_<random>_${EXP_NAME}_<hash>/"
echo

idx=0
for q in "${query_sizes[@]}"; do
    for s in "${seeds[@]}"; do
        gpu=$(( idx % NUM_GPUS ))
        prefix=$(printf '%04x%04x' "$RANDOM" "$RANDOM")
        logf="$LOG_DIR/job_$(printf '%03d' "$idx")_gpu${gpu}_q${q}_s${s}.log"
        cmd="HYDRA_CONFIG_NAME=base python ${PY} \
            al=random data=${DATA} \
            al.query_size=${q} al.eval_zero_iteration=false \
            seed=${s} \
            experiment_name=${prefix}_${EXP_NAME} \
            +debug=false \
            ${COMMON_OVERRIDES}"
        echo "[gpu=$gpu] q=$q seed=$s prefix=$prefix"
        ( CUDA_VISIBLE_DEVICES=$gpu bash -c "$cmd" >"$logf" 2>&1 ) &
        idx=$((idx + 1))
        if (( idx % NUM_GPUS == 0 )); then wait; fi
    done
done
wait

echo
echo "All runs finished."
echo "Aggregate with:"
echo "  python -m atgen.utils.aggregate_metrics outputs/$(date +%F)/*_${EXP_NAME}_[0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f]"
