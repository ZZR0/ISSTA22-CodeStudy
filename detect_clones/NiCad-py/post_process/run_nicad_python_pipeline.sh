#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# NiCad PY post-process pipeline
# ============================================================

# --------------------
# Config
# --------------------
SIM_TAG="sim0.7"
SEED=42
MIN_SIZE=1
MAX_SIZE=20

# --------------------
# Paths
# --------------------
INPUT_DIR="/home/user1-system11/research_dream/llm-clone/detect_clones/NiCad-py/post_process/input"
OUTPUT_ROOT="/home/user1-system11/research_dream/llm-clone/detect_clones/NiCad-py/post_process/data"

# --------------------
# Init Output & Master Log
# --------------------
[[ -d "$INPUT_DIR" ]] || { echo "ERROR: INPUT_DIR not found: $INPUT_DIR" >&2; exit 1; }
mkdir -p "$OUTPUT_ROOT"

# [NEW] This file will now contain EVERYTHING (Summary + Detailed Logs)
MASTER_LOG="$OUTPUT_ROOT/full_pipeline_output_$(date +%Y%m%d_%H%M%S).log"

echo "Pipeline Started: $(date)" | tee "$MASTER_LOG"
echo "Input: $INPUT_DIR" | tee -a "$MASTER_LOG"
echo "------------------------------------------------" | tee -a "$MASTER_LOG"

# --------------------
# Counters
# --------------------
SKIPPED_COUNT=0
SKIPPED_DETAILS=""
TOTAL_COUNT=0

# --------------------
# Check Scripts
# --------------------
for s in \
  1_nicad_xml_to_jsonl.py \
  2a_filter_out_test_fun.py \
  2b_filter_out_group.py \
  2c_remove_comment.py \
  2d_filter_out_init_fun.py \
  3_gen_init_train_sample.py \
  4_gen_neg_clone_sample.py \
  5_gen_pos_clone_sample.py \
  6_split_combine_neg_pos_pairs.py \
  7_gen_clone_bench.py
do
  [[ -f "$s" ]] || { echo "ERROR: Missing script: $s" | tee -a "$MASTER_LOG"; exit 1; }
done

xmls=( "$INPUT_DIR"/*withsource.xml )
if (( ${#xmls[@]} == 0 )); then
  echo "ERROR: No *withsource.xml files found" | tee -a "$MASTER_LOG"
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
  # Local Log (Detailed per repo)
  LOG="$DATA_DIR/pipeline_steps_1_7_${system}_${SIM_TAG}_${TS}.log"

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
  python 2b_filter_out_group.py --input "$STEP2A" --output "$STEP2B" --min-size "$MIN_SIZE" --max-size "$MAX_SIZE" 2>&1 | tee -a "$LOG" "$MASTER_LOG"

  # CHECK: Empty file skip (0 groups)
  if [ ! -s "$STEP2B" ]; then
      MSG="SKIPPED: $system (Reason: 0 clone groups found)"
      echo "$MSG" | tee -a "$LOG" "$MASTER_LOG"
      
      SKIPPED_COUNT=$((SKIPPED_COUNT+1))
      SKIPPED_DETAILS+="$MSG\n"
      continue
  fi

  # Step 2c
  echo "[STEP 2c] 2c_remove_comment.py" | tee -a "$LOG" "$MASTER_LOG"
  STEP2C="$DATA_DIR/step2c_nicad_${system}_${SIM_TAG}.jsonl"
  python 2c_remove_comment.py --input "$STEP2B" --output "$STEP2C" 2>&1 | tee -a "$LOG" "$MASTER_LOG"

  # Step 2d
  echo "[STEP 2d] 2d_filter_out_init_fun.py" | tee -a "$LOG" "$MASTER_LOG"
  STEP2D="$DATA_DIR/step2d_nicad_${system}_${SIM_TAG}.jsonl"
  python 2d_filter_out_init_fun.py --input "$STEP2C" --output "$STEP2D" 2>&1 | tee -a "$LOG" "$MASTER_LOG"

  # Step 3
  echo "[STEP 3] 3_gen_init_train_sample.py" | tee -a "$LOG" "$MASTER_LOG"
  STEP3="$DATA_DIR/step3_nicad_${system}_${SIM_TAG}.jsonl"
  python 3_gen_init_train_sample.py --input "$STEP2D" --output "$STEP3" 2>&1 | tee -a "$LOG" "$MASTER_LOG"

  # Step 4 (NEG) - with failure catch
  echo "[STEP 4] 4_gen_neg_clone_sample.py" | tee -a "$LOG" "$MASTER_LOG"
  STEP4_NEG_TXT="$DATA_DIR/step4_nicad_${system}_${SIM_TAG}_neg_pairs.txt"
  STEP4_NEG_JSONL="$DATA_DIR/step4_nicad_${system}_${SIM_TAG}_neg_pairs.jsonl"
  STEP4_NEG_HTML="$DATA_DIR/step4_nicad_${system}_${SIM_TAG}_neg_pairs.html"
  STEP4_NEG_MD="$DATA_DIR/step4_nicad_${system}_${SIM_TAG}_neg_pairs.md"
  
  set +e
  python 4_gen_neg_clone_sample.py \
    --input "$STEP3" \
    --out_txt "$STEP4_NEG_TXT" \
    --out_jsonl "$STEP4_NEG_JSONL" \
    --out_html "$STEP4_NEG_HTML" \
    --out_md "$STEP4_NEG_MD" \
    --seed "$SEED" \
    --verify \
    --cleanup \
    2>&1 | tee -a "$LOG" "$MASTER_LOG"
  
  exit_code=$?
  set -e

  if [ $exit_code -ne 0 ]; then
      MSG="SKIPPED: $system (Reason: Step 4 failed/insufficient groups)"
      echo "$MSG" | tee -a "$LOG" "$MASTER_LOG"
      
      SKIPPED_COUNT=$((SKIPPED_COUNT+1))
      SKIPPED_DETAILS+="$MSG\n"
      continue
  fi

  # Step 5
  echo "[STEP 5] 5_gen_pos_clone_sample.py" | tee -a "$LOG" "$MASTER_LOG"
  STEP5_POS_TXT="$DATA_DIR/step5_nicad_${system}_${SIM_TAG}_pos_pairs.txt"
  STEP5_POS_JSONL="$DATA_DIR/step5_nicad_${system}_${SIM_TAG}_pos_pairs.jsonl"
  STEP5_POS_HTML="$DATA_DIR/step5_display_nicad_${system}_${SIM_TAG}_pos_pairs.html"
  STEP5_POS_MD="$DATA_DIR/step5_display_nicad_${system}_${SIM_TAG}_pos_pairs.md"
  
  python 5_gen_pos_clone_sample.py \
    --input "$STEP3" \
    --out_txt "$STEP5_POS_TXT" \
    --out_jsonl "$STEP5_POS_JSONL" \
    --out_html "$STEP5_POS_HTML" \
    --out_md "$STEP5_POS_MD" \
    --seed "$SEED" \
    --verify \
    --cleanup \
    2>&1 | tee -a "$LOG" "$MASTER_LOG"

  # Step 6
  echo "[STEP 6] 6_split_combine_neg_pos_pairs.py" | tee -a "$LOG" "$MASTER_LOG"
  STEP6_OUT_DIR="$DATA_DIR/${system}"
  mkdir -p "$STEP6_OUT_DIR"
  
  python 6_split_combine_neg_pos_pairs.py --neg "$STEP4_NEG_TXT" --pos "$STEP5_POS_TXT" --out_dir "$STEP6_OUT_DIR" --seed "$SEED" 2>&1 | tee -a "$LOG" "$MASTER_LOG"

  # Step 7
  echo "[STEP 7] 7_gen_clone_bench.py" | tee -a "$LOG" "$MASTER_LOG"
  STEP7_BENCH="$STEP6_OUT_DIR/data.jsonl"
  
  python 7_gen_clone_bench.py --input "$STEP3" --output "$STEP7_BENCH" --dedup 2>&1 | tee -a "$LOG" "$MASTER_LOG"

  echo "SUCCESS: $system" | tee -a "$LOG" "$MASTER_LOG"
done

# ============================================================
# Final Summary
# ============================================================
echo "============================================================" | tee -a "$MASTER_LOG"
echo "ALL SYSTEMS FINISHED: $(date)" | tee -a "$MASTER_LOG"
echo "Total processed: $TOTAL_COUNT" | tee -a "$MASTER_LOG"
echo "Total skipped:   $SKIPPED_COUNT" | tee -a "$MASTER_LOG"
if [ "$SKIPPED_COUNT" -gt 0 ]; then
    echo -e "Skipped Systems Detail:\n$SKIPPED_DETAILS" | tee -a "$MASTER_LOG"
fi
echo "============================================================" | tee -a "$MASTER_LOG"

echo "Full pipeline log saved to: $MASTER_LOG"