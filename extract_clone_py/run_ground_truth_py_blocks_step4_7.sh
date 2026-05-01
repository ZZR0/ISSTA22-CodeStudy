#!/usr/bin/env bash
set -Eeuo pipefail

# ============================================================
# NiCad PY post-process pipeline: run STEP 4-7 only
# Starts from existing STEP 3 JSONL files
# ============================================================

# --------------------
# Config (override with env vars if needed)
# --------------------
SIM_TAG="${SIM_TAG:-sim0.7}"
SEED="${SEED:-42}"

# Directory containing existing step-3 style JSONL files
# Examples:
#   augmented_ansible.jsonl
#   step3_nicad_ansible_sim0.7.jsonl
STEP3_DIR="${STEP3_DIR:-/home/user1-system11/research_dream/llm-clone/extract_clone_py/data/augmented_post_nicad_blocks}"

# Directory containing python scripts 4-7
SCRIPT_DIR="${SCRIPT_DIR:-$(pwd)}"

# Output root for steps 4-7 artifacts/logs
OUTPUT_ROOT="${OUTPUT_ROOT:-/home/user1-system11/research_dream/llm-clone/extract_clone_py/output/ground_truth_block_level}"

# Optional filename glob override
STEP3_GLOB="${STEP3_GLOB:-*.jsonl}"

# --------------------
# Init
# --------------------
mkdir -p "$OUTPUT_ROOT"

MASTER_LOG="$OUTPUT_ROOT/full_pipeline_steps_4_7_$(date +%Y%m%d_%H%M%S).log"

echo "Pipeline Started: $(date)" | tee "$MASTER_LOG"
echo "STEP3_DIR:   $STEP3_DIR" | tee -a "$MASTER_LOG"
echo "SCRIPT_DIR:  $SCRIPT_DIR" | tee -a "$MASTER_LOG"
echo "OUTPUT_ROOT: $OUTPUT_ROOT" | tee -a "$MASTER_LOG"
echo "SIM_TAG:     $SIM_TAG" | tee -a "$MASTER_LOG"
echo "SEED:        $SEED" | tee -a "$MASTER_LOG"
echo "------------------------------------------------" | tee -a "$MASTER_LOG"

[[ -d "$STEP3_DIR" ]] || { echo "ERROR: STEP3_DIR not found: $STEP3_DIR" | tee -a "$MASTER_LOG" >&2; exit 1; }
[[ -d "$SCRIPT_DIR" ]] || { echo "ERROR: SCRIPT_DIR not found: $SCRIPT_DIR" | tee -a "$MASTER_LOG" >&2; exit 1; }

# --------------------
# Check Scripts
# --------------------
for s in \
  4_gen_neg_clone_sample.py \
  5_gen_pos_clone_sample.py \
  6_split_combine_neg_pos_pairs.py \
  7_gen_clone_bench.py
do
  [[ -f "$SCRIPT_DIR/$s" ]] || {
    echo "ERROR: Missing script: $SCRIPT_DIR/$s" | tee -a "$MASTER_LOG" >&2
    exit 1
  }
done

# --------------------
# Discover Step-3 JSONL files
# --------------------
shopt -s nullglob
step3_files=( "$STEP3_DIR"/$STEP3_GLOB )
shopt -u nullglob

if (( ${#step3_files[@]} == 0 )); then
  echo "ERROR: No JSONL files found in $STEP3_DIR matching $STEP3_GLOB" | tee -a "$MASTER_LOG"
  exit 1
fi

TOTAL_COUNT=${#step3_files[@]}
SUCCESS_COUNT=0
SKIPPED_COUNT=0
FAILED_COUNT=0

SUCCESS_DETAILS=""
SKIPPED_DETAILS=""
FAILED_DETAILS=""

echo "Found $TOTAL_COUNT candidate JSONL files." | tee -a "$MASTER_LOG"

# --------------------
# Helper: derive system name from filename
# --------------------
derive_system_name() {
  local f="$1"
  local b
  b="$(basename "$f" .jsonl)"

  # Common patterns:
  # step3_nicad_<system>_<simtag>
  # augmented_<system>
  # <system>
  if [[ "$b" =~ ^step3_nicad_(.+)_${SIM_TAG}$ ]]; then
    echo "${BASH_REMATCH[1]}"
  elif [[ "$b" =~ ^step3_nicad_(.+)$ ]]; then
    echo "${BASH_REMATCH[1]}"
  elif [[ "$b" =~ ^augmented_(.+)$ ]]; then
    echo "${BASH_REMATCH[1]}"
  else
    echo "$b"
  fi
}

# --------------------
# Loop over step-3 files
# --------------------
for STEP3 in "${step3_files[@]}"; do
  [[ -s "$STEP3" ]] || {
    MSG="SKIPPED: $(basename "$STEP3") (Reason: empty input file)"
    echo "$MSG" | tee -a "$MASTER_LOG"
    SKIPPED_COUNT=$((SKIPPED_COUNT + 1))
    SKIPPED_DETAILS+="$MSG"$'\n'
    continue
  }

  system="$(derive_system_name "$STEP3")"
  DATA_DIR="$OUTPUT_ROOT/${system}-${SIM_TAG}"
  mkdir -p "$DATA_DIR"

  TS="$(date +%Y%m%d_%H%M%S)"
  LOG="$DATA_DIR/pipeline_steps_4_7_${system}_${SIM_TAG}_${TS}.log"

  echo "------------------------------------------------" | tee -a "$MASTER_LOG" "$LOG"
  echo "PROCESSING: $system" | tee -a "$MASTER_LOG" "$LOG"
  echo "INPUT STEP3: $STEP3" | tee -a "$MASTER_LOG" "$LOG"

  # --------------------
  # Step 4: NEG pairs
  # --------------------
  echo "[STEP 4] 4_gen_neg_clone_sample.py" | tee -a "$LOG" "$MASTER_LOG"

  STEP4_NEG_TXT="$DATA_DIR/step4_nicad_${system}_${SIM_TAG}_neg_pairs.txt"
  STEP4_NEG_JSONL="$DATA_DIR/step4_nicad_${system}_${SIM_TAG}_neg_pairs.jsonl"
  STEP4_NEG_HTML="$DATA_DIR/step4_nicad_${system}_${SIM_TAG}_neg_pairs.html"
  STEP4_NEG_MD="$DATA_DIR/step4_nicad_${system}_${SIM_TAG}_neg_pairs.md"

  set +e
  python "$SCRIPT_DIR/4_gen_neg_clone_sample.py" \
    --input "$STEP3" \
    --out_txt "$STEP4_NEG_TXT" \
    --out_jsonl "$STEP4_NEG_JSONL" \
    --out_html "$STEP4_NEG_HTML" \
    --out_md "$STEP4_NEG_MD" \
    --seed "$SEED" \
    --verify \
    --cleanup \
    2>&1 | tee -a "$LOG" "$MASTER_LOG"
  exit_code=${PIPESTATUS[0]}
  set -e

  if (( exit_code != 0 )); then
    MSG="SKIPPED: $system (Reason: Step 4 failed with exit code $exit_code)"
    echo "$MSG" | tee -a "$LOG" "$MASTER_LOG"
    SKIPPED_COUNT=$((SKIPPED_COUNT + 1))
    SKIPPED_DETAILS+="$MSG"$'\n'
    continue
  fi

  if [[ ! -s "$STEP4_NEG_TXT" || ! -s "$STEP4_NEG_JSONL" ]]; then
    MSG="SKIPPED: $system (Reason: Step 4 produced empty/missing neg outputs)"
    echo "$MSG" | tee -a "$LOG" "$MASTER_LOG"
    SKIPPED_COUNT=$((SKIPPED_COUNT + 1))
    SKIPPED_DETAILS+="$MSG"$'\n'
    continue
  fi

  # --------------------
  # Step 5: POS pairs
  # --------------------
  echo "[STEP 5] 5_gen_pos_clone_sample.py" | tee -a "$LOG" "$MASTER_LOG"

  STEP5_POS_TXT="$DATA_DIR/step5_nicad_${system}_${SIM_TAG}_pos_pairs.txt"
  STEP5_POS_JSONL="$DATA_DIR/step5_nicad_${system}_${SIM_TAG}_pos_pairs.jsonl"
  STEP5_POS_HTML="$DATA_DIR/step5_display_nicad_${system}_${SIM_TAG}_pos_pairs.html"
  STEP5_POS_MD="$DATA_DIR/step5_display_nicad_${system}_${SIM_TAG}_pos_pairs.md"

  set +e
  python "$SCRIPT_DIR/5_gen_pos_clone_sample.py" \
    --input "$STEP3" \
    --out_txt "$STEP5_POS_TXT" \
    --out_jsonl "$STEP5_POS_JSONL" \
    --out_html "$STEP5_POS_HTML" \
    --out_md "$STEP5_POS_MD" \
    --seed "$SEED" \
    --verify \
    --cleanup \
    2>&1 | tee -a "$LOG" "$MASTER_LOG"
  exit_code=${PIPESTATUS[0]}
  set -e

  if (( exit_code != 0 )); then
    MSG="SKIPPED: $system (Reason: Step 5 failed with exit code $exit_code)"
    echo "$MSG" | tee -a "$LOG" "$MASTER_LOG"
    SKIPPED_COUNT=$((SKIPPED_COUNT + 1))
    SKIPPED_DETAILS+="$MSG"$'\n'
    continue
  fi

  if [[ ! -s "$STEP5_POS_TXT" || ! -s "$STEP5_POS_JSONL" ]]; then
    MSG="SKIPPED: $system (Reason: Step 5 produced empty/missing pos outputs)"
    echo "$MSG" | tee -a "$LOG" "$MASTER_LOG"
    SKIPPED_COUNT=$((SKIPPED_COUNT + 1))
    SKIPPED_DETAILS+="$MSG"$'\n'
    continue
  fi

  # --------------------
  # Step 6: split/combine pairs
  # --------------------
  echo "[STEP 6] 6_split_combine_neg_pos_pairs.py" | tee -a "$LOG" "$MASTER_LOG"

  STEP6_OUT_DIR="$DATA_DIR/${system}"
  mkdir -p "$STEP6_OUT_DIR"

  set +e
  python "$SCRIPT_DIR/6_split_combine_neg_pos_pairs.py" \
    --neg "$STEP4_NEG_TXT" \
    --pos "$STEP5_POS_TXT" \
    --out_dir "$STEP6_OUT_DIR" \
    --seed "$SEED" \
    2>&1 | tee -a "$LOG" "$MASTER_LOG"
  exit_code=${PIPESTATUS[0]}
  set -e

  if (( exit_code != 0 )); then
    MSG="FAILED: $system (Reason: Step 6 failed with exit code $exit_code)"
    echo "$MSG" | tee -a "$LOG" "$MASTER_LOG"
    FAILED_COUNT=$((FAILED_COUNT + 1))
    FAILED_DETAILS+="$MSG"$'\n'
    continue
  fi

  # --------------------
  # Step 7: clone bench
  # --------------------
  echo "[STEP 7] 7_gen_clone_bench.py" | tee -a "$LOG" "$MASTER_LOG"

  STEP7_BENCH="$STEP6_OUT_DIR/data.jsonl"

  set +e
  python "$SCRIPT_DIR/7_gen_clone_bench.py" \
    --input "$STEP3" \
    --output "$STEP7_BENCH" \
    --dedup \
    2>&1 | tee -a "$LOG" "$MASTER_LOG"
  exit_code=${PIPESTATUS[0]}
  set -e

  if (( exit_code != 0 )); then
    MSG="FAILED: $system (Reason: Step 7 failed with exit code $exit_code)"
    echo "$MSG" | tee -a "$LOG" "$MASTER_LOG"
    FAILED_COUNT=$((FAILED_COUNT + 1))
    FAILED_DETAILS+="$MSG"$'\n'
    continue
  fi

  if [[ ! -s "$STEP7_BENCH" ]]; then
    MSG="FAILED: $system (Reason: Step 7 output missing/empty: $STEP7_BENCH)"
    echo "$MSG" | tee -a "$LOG" "$MASTER_LOG"
    FAILED_COUNT=$((FAILED_COUNT + 1))
    FAILED_DETAILS+="$MSG"$'\n'
    continue
  fi

  SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
  MSG="SUCCESS: $system"
  echo "$MSG" | tee -a "$LOG" "$MASTER_LOG"
  SUCCESS_DETAILS+="$MSG"$'\n'
done

# --------------------
# Final Summary
# --------------------
echo "============================================================" | tee -a "$MASTER_LOG"
echo "ALL SYSTEMS FINISHED: $(date)" | tee -a "$MASTER_LOG"
echo "Total candidates: $TOTAL_COUNT" | tee -a "$MASTER_LOG"
echo "Total success:    $SUCCESS_COUNT" | tee -a "$MASTER_LOG"
echo "Total skipped:    $SKIPPED_COUNT" | tee -a "$MASTER_LOG"
echo "Total failed:     $FAILED_COUNT" | tee -a "$MASTER_LOG"

if (( SUCCESS_COUNT > 0 )); then
  echo -e "Successful Systems:\n$SUCCESS_DETAILS" | tee -a "$MASTER_LOG"
fi

if (( SKIPPED_COUNT > 0 )); then
  echo -e "Skipped Systems Detail:\n$SKIPPED_DETAILS" | tee -a "$MASTER_LOG"
fi

if (( FAILED_COUNT > 0 )); then
  echo -e "Failed Systems Detail:\n$FAILED_DETAILS" | tee -a "$MASTER_LOG"
fi

echo "============================================================" | tee -a "$MASTER_LOG"
echo "Full pipeline log saved to: $MASTER_LOG" | tee -a "$MASTER_LOG"