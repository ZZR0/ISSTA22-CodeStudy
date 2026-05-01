#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# NiCad post-process pipeline (Steps 1–8)
# - Runs over ALL *withsource.xml files
# - Correct system name extraction (activemq, camel, hadoop, ...)
# - Single similarity tag: SIM_TAG
# - One log file per system (all steps appended)
# ============================================================

# --------------------
# Config
# --------------------
SIM_TAG="sim0.7"       
SEED=42
MAX_CLONES=20

# --------------------
# Paths
# --------------------
INPUT_DIR="/home/user1-system11/research_dream/llm-clone/detect_clones/NiCad/post_process/input"
OUTPUT_ROOT="/home/user1-system11/research_dream/llm-clone/detect_clones/NiCad/post_process/data/java"
PROJECTS_ROOT="/home/user1-system11/research_dream/clone_research/clone_detectors/NiCad"

# --------------------
# Safety checks
# --------------------
[[ -d "$INPUT_DIR" ]] || { echo "ERROR: INPUT_DIR not found: $INPUT_DIR" >&2; exit 1; }
[[ -d "$OUTPUT_ROOT" ]] || mkdir -p "$OUTPUT_ROOT"
[[ -e "$PROJECTS_ROOT" ]] || { echo "ERROR: PROJECTS_ROOT not found: $PROJECTS_ROOT" >&2; exit 1; }

for s in \
  1_nicad_xml_to_jsonl.py \
  2_java_qmethod_ts.py \
  3_filter_out_data.py \
  4_gen_init_train_sample.py \
  5_gen_neg_clone_sample.py \
  6_gen_pos_clone_sample.py \
  7_gen_func_index_corpus.py \
  8_split_combine_neg_pos_pairs.py
do
  [[ -f "$s" ]] || { echo "ERROR: Missing script: $s" >&2; exit 1; }
done

xmls=( "$INPUT_DIR"/*withsource.xml )
if (( ${#xmls[@]} == 0 )); then
  echo "ERROR: No *withsource.xml files found in $INPUT_DIR" >&2
  exit 1
fi

echo "Found ${#xmls[@]} XML files."

# ============================================================
# Loop over XML files
# ============================================================
for xml in "${xmls[@]}"; do
  fname="$(basename "$xml")"

  # ----------------------------------------------------------
  # SYSTEM NAME EXTRACTION (FIXED)
  # Example:
  # activemq-java_functions-blind-clones-0.30-classes-withsource.xml
  # --> activemq
  # ----------------------------------------------------------
  system="${fname%%_functions*}"
  system="${system%-java}"
  system="${system%%.xml}"

  DATA_DIR="$OUTPUT_ROOT/${system}-${SIM_TAG}"
  CORPUS_DIR="$DATA_DIR/${system}"
  mkdir -p "$DATA_DIR" "$CORPUS_DIR"

  TS="$(date +%Y%m%d_%H%M%S)"
  LOG="$DATA_DIR/pipeline_steps_1_8_${system}_${SIM_TAG}_${TS}.log"

  # --------------------
  # Outputs
  # --------------------
  STEP1="$DATA_DIR/step1_nicad_${system}_${SIM_TAG}_raw.jsonl"
  STEP2="$DATA_DIR/step2_nicad_${system}_${SIM_TAG}_fqn.jsonl"
  STEP3="$DATA_DIR/step3_nicad_${system}_${SIM_TAG}_fqn_filtered.jsonl"
  STEP4="$DATA_DIR/step4_nicad_${system}_${SIM_TAG}_fqn_filtered_with_func_id.jsonl"

  STEP5_NEG_TXT="$DATA_DIR/step5_nicad_${system}_${SIM_TAG}_neg_pairs.txt"
  STEP5_NEG_JSONL="$DATA_DIR/step5_nicad_${system}_${SIM_TAG}_neg_pairs.jsonl"
  STEP5_NEG_HTML="$DATA_DIR/step5_display_nicad_${system}_${SIM_TAG}_neg_pairs.html"
  STEP5_NEG_MD="$DATA_DIR/step5_display_nicad_${system}_${SIM_TAG}_neg_pairs.md"

  STEP6_POS_TXT="$DATA_DIR/step6_nicad_${system}_${SIM_TAG}_pos_pairs.txt"
  STEP6_POS_JSONL="$DATA_DIR/step6_nicad_${system}_${SIM_TAG}_pos_pairs.jsonl"
  STEP6_POS_HTML="$DATA_DIR/step6_display_nicad_${system}_${SIM_TAG}_pos_pairs.html"
  STEP6_POS_MD="$DATA_DIR/step6_display_nicad_${system}_${SIM_TAG}_pos_pairs.md"

  STEP7_CORPUS="$CORPUS_DIR/data.jsonl"

  echo "============================================================" | tee -a "$LOG"
  echo "SYSTEM:   $system" | tee -a "$LOG"
  echo "XML:      $xml" | tee -a "$LOG"
  echo "DATA_DIR:   $DATA_DIR" | tee -a "$LOG"
  echo "SIM_TAG:  $SIM_TAG | SEED: $SEED | MAX_CLONES: $MAX_CLONES" | tee -a "$LOG"
  echo "LOG:      $LOG" | tee -a "$LOG"
  echo "============================================================" | tee -a "$LOG"

  # --------------------
  # Step 1
  # --------------------
  echo "[STEP 1] 1_nicad_xml_to_jsonl.py  $(date)" | tee -a "$LOG"
  python 1_nicad_xml_to_jsonl.py \
    --xml "$xml" \
    --out "$STEP1" \
    --mode class \
    2>&1 | tee -a "$LOG"

  # --------------------
  # Step 2
  # --------------------
  echo "[STEP 2] 2_java_qmethod_ts.py  $(date)" | tee -a "$LOG"
  python 2_java_qmethod_ts.py \
    --in "$STEP1" \
    --out "$STEP2" \
    --projects-root "$PROJECTS_ROOT" \
    2>&1 | tee -a "$LOG"

  # --------------------
  # Step 3
  # --------------------
  echo "[STEP 3] 3_filter_out_data.py  $(date)" | tee -a "$LOG"
  python 3_filter_out_data.py \
    --input "$STEP2" \
    --output "$STEP3" \
    --max_clones "$MAX_CLONES" \
    2>&1 | tee -a "$LOG"

  # --------------------
  # Step 4
  # --------------------
  echo "[STEP 4] 4_gen_init_train_sample.py  $(date)" | tee -a "$LOG"
  python 4_gen_init_train_sample.py \
    --input "$STEP3" \
    --output "$STEP4" \
    2>&1 | tee -a "$LOG"

  # --------------------
  # Step 5 (NEG)
  # --------------------
  echo "[STEP 5] 5_gen_neg_clone_sample.py  $(date)" | tee -a "$LOG"
  python 5_gen_neg_clone_sample.py \
    --input "$STEP4" \
    --out_txt "$STEP5_NEG_TXT" \
    --out_jsonl "$STEP5_NEG_JSONL" \
    --out_html "$STEP5_NEG_HTML" \
    --out_md "$STEP5_NEG_MD" \
    --seed "$SEED" \
    --verify \
    --cleanup \
    2>&1 | tee -a "$LOG"

  # --------------------
  # Step 6 (POS)
  # --------------------
  echo "[STEP 6] 6_gen_pos_clone_sample.py  $(date)" | tee -a "$LOG"
  python 6_gen_pos_clone_sample.py \
    --input "$STEP4" \
    --out_txt "$STEP6_POS_TXT" \
    --out_jsonl "$STEP6_POS_JSONL" \
    --out_html "$STEP6_POS_HTML" \
    --out_md "$STEP6_POS_MD" \
    --seed "$SEED" \
    --verify \
    --cleanup \
    2>&1 | tee -a "$LOG"

  # --------------------
  # Step 7 (Bench)
  # --------------------
  echo "[STEP 7] 7_gen_func_index_corpus.py  $(date)" | tee -a "$LOG"
  python 7_gen_func_index_corpus.py \
    --input "$STEP4" \
    --output "$STEP7_CORPUS" \
    --dedup \
    2>&1 | tee -a "$LOG"

  # --------------------
  # Step 8 (Split / Combine)
  # --------------------
  echo "[STEP 8] 8_split_combine_neg_pos_pairs.py  $(date)" | tee -a "$LOG"
  python 8_split_combine_neg_pos_pairs.py \
    --neg "$STEP5_NEG_TXT" \
    --pos "$STEP6_POS_TXT" \
    --out_dir "$CORPUS_DIR" \
    --seed "$SEED" \
    2>&1 | tee -a "$LOG"

  echo "============================================================" | tee -a "$LOG"
  echo "DONE: $system  $(date)" | tee -a "$LOG"
  echo "============================================================" | tee -a "$LOG"

done

echo "ALL SYSTEMS FINISHED: $(date)"
