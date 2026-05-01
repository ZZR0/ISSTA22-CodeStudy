#!/bin/bash
set -euo pipefail

# ==============================
# Paths
# ==============================

JSONL_DIR="/home/user1-system11/research_dream/llm-clone/extract_clone/output/extractable_nicad_block_clones"
GT_BASE="/home/user1-system11/research_dream/llm-clone/Task/Clone-Detection-BigCloneBench/dataset/nicad_block_java"
PRED_BASE="/home/user1-system11/research_dream/llm-clone/Task/Clone-Detection-BigCloneBench/codebert/saved_models_bcb/predictions_bcb_java_block_function"
PRED_ADAPT_BASE="/home/user1-system11/research_dream/llm-clone/Task/Clone-Detection-BigCloneBench/codebert/saved_models_combined/predictions_bcb_java_block_function"
OUT_BASE="/home/user1-system11/research_dream/llm-clone/extract_clone/output/clone_refactorability_evaluation"

mkdir -p ${OUT_BASE}

# ==============================
# Single Log File
# ==============================

LOG_FILE="${OUT_BASE}/clone_refactorability_batch_$(date +%Y%m%d_%H%M%S).log"

# Redirect ALL output (stdout + stderr) into one log file
exec > >(tee -a "$LOG_FILE") 2>&1

echo "================================================="
echo "Batch clone refactorability evaluation started"
echo "Start time: $(date)"
echo "Log file: $LOG_FILE"
echo "================================================="
echo ""

# ==============================
# Loop through all projects
# ==============================

for jsonl_file in ${JSONL_DIR}/*_clone_analysis.jsonl
do
    filename=$(basename "$jsonl_file")
    project=${filename%%_clone_analysis.jsonl}

    echo "-------------------------------------------------"
    echo "Processing project: $project"
    echo "-------------------------------------------------"

    GT_FILE="${GT_BASE}/${project}/test.txt"
    PRED_FILE="${PRED_BASE}/predictions_${project}_test.txt"
    PRED_ADAPT_FILE="${PRED_ADAPT_BASE}/predictions_${project}_test.txt"
    OUT_FILE="${OUT_BASE}/${project}_refactor_clone_eval.csv"

    # File checks
    if [[ ! -f "$GT_FILE" ]]; then
        echo "[WARNING] Missing GT file for $project, skipping..."
        continue
    fi

    if [[ ! -f "$PRED_FILE" ]]; then
        echo "[WARNING] Missing base prediction file for $project, skipping..."
        continue
    fi

    if [[ ! -f "$PRED_ADAPT_FILE" ]]; then
        echo "[WARNING] Missing adapted prediction file for $project, skipping..."
        continue
    fi

    echo "Running evaluation..."

    python clone_refactorability_evaluation.py \
        --jsonl "$jsonl_file" \
        --gt "$GT_FILE" \
        --pred "$PRED_FILE" \
        --pred_adapt "$PRED_ADAPT_FILE" \
        --out "$OUT_FILE"

    echo "Finished project: $project"
    echo ""
done

echo "================================================="
echo "Batch completed at: $(date)"
echo "================================================="