#!/usr/bin/env bash
set -euo pipefail

BASE_JSONL_ROOT="/home/user1-system11/research_dream/llm-clone/extract_clone/data/java-blocks"
BASE_DIR="/home/user1-system11/research_dream/llm-clone/detect_clones/NiCad"
OUT_ROOT="./output/extractable_nicad_block_clones"

LOG_DIR="./logs"
mkdir -p "$OUT_ROOT" "$LOG_DIR"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOG_DIR/block_clone_extractability_${TIMESTAMP}.log"

echo "============================================" | tee -a "$LOG_FILE"
echo "Block Clone Extractability Batch Run" | tee -a "$LOG_FILE"
echo "Started at: $(date)" | tee -a "$LOG_FILE"
echo "============================================" | tee -a "$LOG_FILE"
echo | tee -a "$LOG_FILE"

for repo_dir in "$BASE_JSONL_ROOT"/*-sim0.7; do
    repo_folder="$(basename "$repo_dir")"
    repo_name="${repo_folder%-sim0.7}"

    jsonl_file="$repo_dir/step4_nicad_${repo_name}_sim0.7_filtered_with_func_id.jsonl"

    if [[ ! -f "$jsonl_file" ]]; then
        echo "[SKIP] Missing file: $jsonl_file" | tee -a "$LOG_FILE"
        continue
    fi

    echo "--------------------------------------------" | tee -a "$LOG_FILE"
    echo "[START] $repo_name  ($(date))" | tee -a "$LOG_FILE"
    echo "Using JSONL: $jsonl_file" | tee -a "$LOG_FILE"

    python main.py \
        --jsonl "$jsonl_file" \
        --base-dir "$BASE_DIR" \
        --output "$OUT_ROOT/${repo_name}_clone_visualization.html" \
        --out-jsonl "$OUT_ROOT/${repo_name}_clone_analysis.jsonl" \
        >> "$LOG_FILE" 2>&1

    echo "[DONE] $repo_name  ($(date))" | tee -a "$LOG_FILE"
    echo | tee -a "$LOG_FILE"
done

echo "============================================" | tee -a "$LOG_FILE"
echo "Finished at: $(date)" | tee -a "$LOG_FILE"
echo "Log saved to: $LOG_FILE"
echo "============================================" | tee -a "$LOG_FILE"