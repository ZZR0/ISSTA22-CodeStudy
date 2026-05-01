#!/usr/bin/env bash
set -euo pipefail

# Define directories
DATA_DIR=../data/java
INPUT_DIR=../input
PROJECTS_ROOT=../..

LOG="$DATA_DIR/pipeline.log"
INPUT_XML="$INPUT_DIR/camel-java_functions-blind-clones-0.30-classes-withsource.xml"
STEP1_OUTPUT="$DATA_DIR/step1_nicad_camel_sim0.7_classes.jsonl"
STEP2_OUTPUT="$DATA_DIR/camel_sim0.7.jsonl"
STEP3_OUTPUT="$DATA_DIR/nicad_camel_clone_func.jsonl"
STEP4_OUTPUT="$DATA_DIR/nicad_camel_clone_data.jsonl"
DATAPROC_OUTPUT="../../../../DataProc/data/nicad_camel_clone_data.jsonl"

mkdir -p "$DATA_DIR"

echo "============================================================" | tee -a "$LOG"
echo "[STEP 1] 1_nicad_xml_to_jsonl.py  $(date)" | tee -a "$LOG"
echo "============================================================" | tee -a "$LOG"
python 1_nicad_xml_to_jsonl.py \
  --xml "$INPUT_XML" \
  --out "$STEP1_OUTPUT" \
  --mode class \
  2>&1 | tee -a "$LOG"

echo "------------------------------------------------------------" | tee -a "$LOG"
echo "[STEP 2] 2_java_qmethod_ts.py  $(date)" | tee -a "$LOG"
echo "------------------------------------------------------------" | tee -a "$LOG"
python 2_java_qmethod_ts.py \
  --in "$STEP1_OUTPUT" \
  --out "$STEP2_OUTPUT" \
  --projects-root "$PROJECTS_ROOT" \
  2>&1 | tee -a "$LOG"

echo "------------------------------------------------------------" | tee -a "$LOG"
echo "[STEP 3] 3_filter_out_data.py  $(date)" | tee -a "$LOG"
echo "------------------------------------------------------------" | tee -a "$LOG"
python 3_filter_out_data.py \
  --input "$STEP2_OUTPUT" \
  --output "$STEP3_OUTPUT" \
  --mode drop_group_if_any_test \
  --max_clones 20 \
  2>&1 | tee -a "$LOG"

echo "------------------------------------------------------------" | tee -a "$LOG"
echo "[STEP 4] 4_gen_init_train_sample.py  $(date)" | tee -a "$LOG"
echo "------------------------------------------------------------" | tee -a "$LOG"
python 4_gen_init_train_sample.py \
  "$STEP3_OUTPUT" \
  "$STEP4_OUTPUT" \
  2>&1 | tee -a "$LOG"

echo "------------------------------------------------------------" | tee -a "$LOG"
echo "[STEP 5] 5_gen_neg_clone_sample.py  $(date)" | tee -a "$LOG"
echo "------------------------------------------------------------" | tee -a "$LOG"
python 5_gen_neg_clone_sample.py \
  "$STEP4_OUTPUT" \
  2>&1 | tee -a "$LOG"

# Step 6 (optional)
# echo "------------------------------------------------------------" | tee -a "$LOG"
# echo "[STEP 6] Comparing outputs  $(date)" | tee -a "$LOG"
# echo "------------------------------------------------------------" | tee -a "$LOG"
# diff "$DATAPROC_OUTPUT" "$STEP4_OUTPUT" 2>&1 | tee -a "$LOG"