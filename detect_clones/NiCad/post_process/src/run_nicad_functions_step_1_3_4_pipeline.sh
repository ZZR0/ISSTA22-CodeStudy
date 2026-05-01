#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# NiCad post-process pipeline (Steps 1,3,4)
# - Runs over ALL *withsource.xml files
# - Correct system name extraction (activemq, camel, hadoop, ...)
# - Single similarity tag: SIM_TAG
# - ONE global log file for all systems
# ============================================================

# --------------------
# Config
# --------------------
SIM_TAG="sim0.7"
MAX_CLONES=20

# --------------------
# Paths
# --------------------
INPUT_DIR="/home/user1-system11/research_dream/llm-clone/detect_clones/NiCad/post_process/input"
OUTPUT_ROOT="/home/user1-system11/research_dream/llm-clone/detect_clones/NiCad/post_process/data/java-functions"

# --------------------
# Safety checks
# --------------------
[[ -d "$INPUT_DIR" ]] || { echo "ERROR: INPUT_DIR not found: $INPUT_DIR" >&2; exit 1; }
mkdir -p "$OUTPUT_ROOT"

for s in 1_nicad_xml_to_jsonl.py 3_filter_out_data.py 4_gen_init_train_sample.py; do
  [[ -f "$s" ]] || { echo "ERROR: Missing script: $s" >&2; exit 1; }
done

xmls=( "$INPUT_DIR"/*withsource.xml )
if (( ${#xmls[@]} == 0 )); then
  echo "ERROR: No *withsource.xml files found in $INPUT_DIR" >&2
  exit 1
fi

# --------------------
# Global log (ONE file)
# --------------------
TS="$(date +%Y%m%d_%H%M%S)"
LOG="$OUTPUT_ROOT/pipeline_steps_1_3_4_ALL_${SIM_TAG}_${TS}.log"

# Redirect ALL output to the log (and also show on screen)
exec > >(tee -a "$LOG") 2>&1

trap 'echo "ERROR: line=$LINENO cmd=$BASH_COMMAND exit_code=$? time=$(date)"' ERR

echo "============================================================"
echo "START PIPELINE  $(date)"
echo "INPUT_DIR:   $INPUT_DIR"
echo "OUTPUT_ROOT: $OUTPUT_ROOT"
echo "SIM_TAG:     $SIM_TAG"
echo "MAX_CLONES:  $MAX_CLONES"
echo "LOG:         $LOG"
echo "XML files:   ${#xmls[@]}"
echo "============================================================"
echo

# ============================================================
# Loop over XML files
# ============================================================
for xml in "${xmls[@]}"; do
  fname="$(basename "$xml")"

  # SYSTEM NAME EXTRACTION
  system="${fname%%_functions*}"
  system="${system%-java}"
  system="${system%%.xml}"

  DATA_DIR="$OUTPUT_ROOT/${system}-${SIM_TAG}"
  mkdir -p "$DATA_DIR"

  # Outputs
  STEP1="$DATA_DIR/step1_nicad_${system}_${SIM_TAG}_raw.jsonl"
  STEP3="$DATA_DIR/step3_nicad_${system}_${SIM_TAG}_filtered.jsonl"
  STEP4="$DATA_DIR/step4_nicad_${system}_${SIM_TAG}_filtered_with_func_id.jsonl"

  echo "------------------------------------------------------------"
  echo "SYSTEM:   $system"
  echo "XML:      $xml"
  echo "DATA_DIR: $DATA_DIR"
  echo "------------------------------------------------------------"

  # Step 1
  echo "[STEP 1] 1_nicad_xml_to_jsonl.py  time=$(date)"
  python 1_nicad_xml_to_jsonl.py \
    --xml "$xml" \
    --out "$STEP1"

  # Step 3 (NOTE: input is STEP1 now)
  echo "[STEP 3] 3_filter_out_data.py  time=$(date)"
  python 3_filter_out_data.py \
    --input "$STEP1" \
    --output "$STEP3" \
    --max_clones "$MAX_CLONES" \
    --keep_comments

  # Step 4
  echo "[STEP 4] 4_gen_init_train_sample.py  time=$(date)"
  python 4_gen_init_train_sample.py \
    --input "$STEP3" \
    --output "$STEP4"

  echo "DONE SYSTEM: $system  time=$(date)"
  echo
done

echo "============================================================"
echo "ALL SYSTEMS FINISHED  $(date)"
echo "LOG SAVED AT: $LOG"
echo "============================================================"