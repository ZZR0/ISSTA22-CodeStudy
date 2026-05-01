#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# NiCad PY post-process pipeline (Steps: 1, 2a, 2b, 2d, 3 only)
# ============================================================
# This script runs a subset of the full NiCad post-processing pipeline, specifically:
# - Step 1: Convert NiCad XML to JSONL
# - Step 2a: Filter out test functions
# - Step 2b: Filter out groups based on size
# - Step 2d: Filter out __init__ functions
# - Step 3: Generate initial training samples   
# Use the output which keeps the comments for refactoring (i.e., skip Step 2c which removes comments) since we want to keep comments for the refactoring task.
# --------------------
# Config
# --------------------
SIM_TAG="sim0.7"
MIN_SIZE=1
MAX_SIZE=20

# --------------------
# Paths
# --------------------
INPUT_DIR="/home/user1-system11/research_dream/llm-clone/detect_clones/NiCad-py/post_process/input"
OUTPUT_ROOT="/home/user1-system11/research_dream/llm-clone/detect_clones/NiCad-py/post_process/output/py_funcs_step1_2a_2b_2d_3"

# --------------------
# Init Output & Master Log
# --------------------
[[ -d "$INPUT_DIR" ]] || { echo "ERROR: INPUT_DIR not found: $INPUT_DIR" >&2; exit 1; }
mkdir -p "$OUTPUT_ROOT"

MASTER_LOG="$OUTPUT_ROOT/full_pipeline_steps_1_2a_2b_2d_3_$(date +%Y%m%d_%H%M%S).log"

echo "Pipeline Started: $(date)" | tee "$MASTER_LOG"
echo "Input: $INPUT_DIR" | tee -a "$MASTER_LOG"
echo "Steps: 1, 2a, 2b, 2d, 3" | tee -a "$MASTER_LOG"
echo "------------------------------------------------" | tee -a "$MASTER_LOG"

# --------------------
# Counters
# --------------------
SKIPPED_COUNT=0
SKIPPED_DETAILS=""
TOTAL_COUNT=0

# --------------------
# Check Scripts (only the ones we will run)
# --------------------
for s in \
  1_nicad_xml_to_jsonl.py \
  2a_filter_out_test_fun.py \
  2b_filter_out_group.py \
  2d_filter_out_init_fun.py \
  3_gen_init_train_sample.py
do
  [[ -f "$s" ]] || { echo "ERROR: Missing script: $s" | tee -a "$MASTER_LOG"; exit 1; }
done

# --------------------
# Collect XMLs
# --------------------
xmls=( "$INPUT_DIR"/*withsource.xml )
if (( ${#xmls[@]} == 0 )); then
  echo "ERROR: No *withsource.xml files found in $INPUT_DIR" | tee -a "$MASTER_LOG"
  exit 1
fi

TOTAL_COUNT=${#xmls[@]}
echo "Found $TOTAL_COUNT XML files." | tee -a "$MASTER_LOG"

# ============================================================
# Loop over XML files
# ============================================================
for xml in "${xmls[@]}"; do
  fname="$(basename "$xml")"
  system="${fname%%_functions*}"
  system="${system%%.xml}"

  DATA_DIR="$OUTPUT_ROOT/${system}-${SIM_TAG}"
  mkdir -p "$DATA_DIR"

  TS="$(date +%Y%m%d_%H%M%S)"
  LOG="$DATA_DIR/pipeline_steps_1_2a_2b_2d_3_${system}_${SIM_TAG}_${TS}.log"

  echo "------------------------------------------------" | tee -a "$MASTER_LOG"
  echo "PROCESSING: $system" | tee -a "$MASTER_LOG" "$LOG"

  # Step 1
  echo "[STEP 1] 1_nicad_xml_to_jsonl.py" | tee -a "$LOG" "$MASTER_LOG"
  STEP1="$DATA_DIR/step1_nicad_${system}_${SIM_TAG}.jsonl"
  python 1_nicad_xml_to_jsonl.py --xml "$xml" --out "$STEP1" --mode class 2>&1 | tee -a "$LOG" "$MASTER_LOG"

  # Step 2a
  echo "[STEP 2a] 2a_filter_out_test_fun.py" | tee -a "$LOG" "$MASTER_LOG"
  STEP2A="$DATA_DIR/step2a_nicad_${system}_${SIM_TAG}.jsonl"
  python 2a_filter_out_test_fun.py --input "$STEP1" --output "$STEP2A" 2>&1 | tee -a "$LOG" "$MASTER_LOG"

  # Step 2b
  echo "[STEP 2b] 2b_filter_out_group.py" | tee -a "$LOG" "$MASTER_LOG"
  STEP2B="$DATA_DIR/step2b_nicad_${system}_${SIM_TAG}.jsonl"
  python 2b_filter_out_group.py \
    --input "$STEP2A" \
    --output "$STEP2B" \
    --min-size "$MIN_SIZE" \
    --max-size "$MAX_SIZE" \
    2>&1 | tee -a "$LOG" "$MASTER_LOG"

  # CHECK: Empty file skip (0 groups)
  if [[ ! -s "$STEP2B" ]]; then
    MSG="SKIPPED: $system (Reason: 0 clone groups found after Step 2b)"
    echo "$MSG" | tee -a "$LOG" "$MASTER_LOG"
    SKIPPED_COUNT=$((SKIPPED_COUNT + 1))
    SKIPPED_DETAILS+="$MSG\n"
    continue
  fi

  # Step 2d (note: now uses Step2B directly since Step 2c is skipped)
  echo "[STEP 2d] 2d_filter_out_init_fun.py" | tee -a "$LOG" "$MASTER_LOG"
  STEP2D="$DATA_DIR/step2d_nicad_${system}_${SIM_TAG}.jsonl"
  python 2d_filter_out_init_fun.py --input "$STEP2B" --output "$STEP2D" 2>&1 | tee -a "$LOG" "$MASTER_LOG"

  # Step 3
  echo "[STEP 3] 3_gen_init_train_sample.py" | tee -a "$LOG" "$MASTER_LOG"
  STEP3="$DATA_DIR/step3_nicad_${system}_${SIM_TAG}.jsonl"
  python 3_gen_init_train_sample.py --input "$STEP2D" --output "$STEP3" 2>&1 | tee -a "$LOG" "$MASTER_LOG"

  echo "SUCCESS: $system" | tee -a "$LOG" "$MASTER_LOG"
done

# ============================================================
# Final Summary
# ============================================================
echo "============================================================" | tee -a "$MASTER_LOG"
echo "ALL SYSTEMS FINISHED: $(date)" | tee -a "$MASTER_LOG"
echo "Total processed: $TOTAL_COUNT" | tee -a "$MASTER_LOG"
echo "Total skipped:   $SKIPPED_COUNT" | tee -a "$MASTER_LOG"
if (( SKIPPED_COUNT > 0 )); then
  echo -e "Skipped Systems Detail:\n$SKIPPED_DETAILS" | tee -a "$MASTER_LOG"
fi
echo "============================================================" | tee -a "$MASTER_LOG"
echo "Full pipeline log saved to: $MASTER_LOG"