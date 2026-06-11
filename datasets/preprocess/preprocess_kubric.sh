#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   1. Put the Kubric TFDS data under RAW_DIR, or edit RAW_DIR below.
#   2. Edit PROCESSED_DIR and GPUS for your machine.
#   3. Run: datasets/preprocess/preprocess_kubric.sh
#
# Output:
#   For validation split, gen_kubric_tracking.py writes to ${PROCESSED_DIR}_val.
#   The generated directory is consumed by datasets/preprocess/gen_kubric_video.py.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON:-python3}"

RAW_DIR="data/datasets/kubric_tracking/"
PROCESSED_DIR="data/datasets/kubric"
SPLIT="validation"
IMAGE_SIZE=512
INTERVAL=2
TOTAL_FRAMES=24
GPUS=(0 1 2 3)

run_segment() {
    local gpu_id="$1"
    local start_frame="$2"
    local end_frame="$3"

    echo "GPU ${gpu_id}: frames ${start_frame}-${end_frame}"
    for (( start=start_frame; start<end_frame; start+=INTERVAL )); do
        local end=$((start + INTERVAL))
        if (( end > TOTAL_FRAMES )); then
            end="$TOTAL_FRAMES"
        fi

        echo "GPU ${gpu_id}: query frames ${start}-${end}"
        CUDA_VISIBLE_DEVICES="$gpu_id" "$PYTHON_BIN" "$SCRIPT_DIR/gen_kubric_tracking.py" \
            --raw_dir "$RAW_DIR" \
            --processed_dir "$PROCESSED_DIR" \
            --split "$SPLIT" \
            --image_size "$IMAGE_SIZE" \
            --start_frame "$start" \
            --end_frame "$end"
    done
}

num_gpus="${#GPUS[@]}"
num_jobs=$(((TOTAL_FRAMES + INTERVAL - 1) / INTERVAL))
jobs_per_gpu=$(((num_jobs + num_gpus - 1) / num_gpus))

echo "Raw dir: $RAW_DIR"
echo "Processed dir: $PROCESSED_DIR"
echo "Split: $SPLIT"
echo "GPUs: ${GPUS[*]}"

for gpu_index in "${!GPUS[@]}"; do
    start=$((gpu_index * jobs_per_gpu * INTERVAL))
    end=$((start + jobs_per_gpu * INTERVAL))

    if (( start >= TOTAL_FRAMES )); then
        continue
    fi
    if (( end > TOTAL_FRAMES )); then
        end="$TOTAL_FRAMES"
    fi

    run_segment "${GPUS[$gpu_index]}" "$start" "$end" &
done

wait
echo "Kubric tracking preprocessing finished."
