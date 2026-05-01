#!/usr/bin/env bash
set -euo pipefail

# =========================
# NiCad post-process pipeline (Steps 1–8)
# Single tag: SIM_TAG=sim0.5
# Logs: everything goes into one file via: 2>&1 | tee -a "$LOG"
# =========================

# ---- Config ----
SIM_TAG="sim0.5"
SEED=42
MAX_CLONES=20

# ---- Paths ----
INPUT_XML="/home/user1-system11/research_dream/llm-clone/detect_clones/NiCad/post_process/input/camel-java_functions-blind-clones-0.50-classes-withsource.xml"
DATA_DIR="/home/user1-system11/research_dream/llm-clone/detect_clones/NiCad/post_process/data/java"
OUT_DIR="$DATA_DIR/camel-sim0.5"
PROJECTS_ROOT="/home/user1-system11/research_dream/clone_research/clone_detectors/NiCad"

# ---- Outputs ----
STEP1="$DATA_DIR/step1_nicad_camel_${SIM_TAG}_raw.jsonl"
STEP2="$DATA_DIR/step2_nicad_camel_${SIM_TAG}_fqn.jsonl"
STEP3="$DATA_DIR/step3_nicad_camel_${SIM_TAG}_fqn_filtered.jsonl"
STEP4="$DATA_DIR/step4_nicad_camel_${SIM_TAG}_fqn_filtered_with_func_id.jsonl"

STEP5_NEG_TXT="$DATA_DIR/step5_nicad_camel_${SIM_TAG}_neg_pairs.txt"
STEP5_NEG_JSONL="$DATA_DIR/step5_nicad_camel_${SIM_TAG}_neg_pairs.jsonl"
STEP5_NEG_HTML="$DATA_DIR/step5_display_nicad_camel_${SIM_TAG}_neg_pairs.html"
STEP5_NEG_MD="$DATA_DIR/step5_display_nicad_camel_${SIM_TAG}_neg_pairs.md"

STEP6_POS_TXT="$DATA_DIR/step6_nicad_camel_${SIM_TAG}_pos_pairs.txt"
STEP6_POS_JSONL="$DATA_DIR/step6_nicad_camel_${SIM_TAG}_pos_pairs.jsonl"
STEP6_POS_HTML="$DATA_DIR/step6_display_nicad_camel_${SIM_TAG}_pos_pairs.html"
STEP6_POS_MD="$DATA_DIR/step6_display_nicad_camel_${SIM_TAG}_pos_pairs.md"

STEP7_BENCH="$OUT_DIR/step7_nicad_camel_${SIM_TAG}_clone_bench.jsonl"

# ---- Log ----
TS="$(date +%Y%m%d_%H%M%S)"
LOG="$DATA_DIR/pipeline_steps_1_8_${SIM_TAG}_${TS}.log"

mkdir -p "$DATA_DIR" "$OUT_DIR"

# ---- Preflight checks ----
if [[ ! -f "$INPUT_XML" ]]; then
  echo "[ERROR] INPUT_XML not found: $INPUT_XML" >&2
  exit 1
fi

for s in \
  1_nicad_xml_to_jsonl.py \
  2_java_qmethod_ts.py \
  3_filter_out_data.py \
  4_gen_init_train_sample.py \
  5_gen_neg_clone_sample.py \
  6_gen_pos_clone_sample.py \
  7_gen_clone_bench.py \
  8_split_comine_neg_pos_pairs.py
do
  [[ -f "$s" ]] || { echo "[ERROR] Missing script: $s" >&2; exit 1; }
done

echo "============================================================" | tee -a "$LOG"
echo "NiCad post_process pipeline started: $(date)" | tee -a "$LOG"
echo "SIM_TAG: $SIM_TAG | SEED: $SEED | MAX_CLONES: $MAX_CLONES" | tee -a "$LOG"
echo "INPUT_XML: $INPUT_XML" | tee -a "$LOG"
echo "LOG: $LOG" | tee -a "$LOG"
echo "============================================================" | tee -a "$LOG"

echo "------------------------------------------------------------" | tee -a "$LOG"
echo "[STEP 1] 1_nicad_xml_to_jsonl.py  $(date)" | tee -a "$LOG"
echo "------------------------------------------------------------" | tee -a "$LOG"
python 1_nicad_xml_to_jsonl.py \
  --xml "$INPUT_XML" \
  --out "$STEP1" \
  --mode class \
  2>&1 | tee -a "$LOG"

echo "------------------------------------------------------------" | tee -a "$LOG"
echo "[STEP 2] 2_java_qmethod_ts.py  $(date)" | tee -a "$LOG"
echo "------------------------------------------------------------" | tee -a "$LOG"
python 2_java_qmethod_ts.py \
  --in "$STEP1" \
  --out "$STEP2" \
  --projects-root "$PROJECTS_ROOT" \
  2>&1 | tee -a "$LOG"

echo "------------------------------------------------------------" | tee -a "$LOG"
echo "[STEP 3] 3_filter_out_data.py  $(date)" | tee -a "$LOG"
echo "------------------------------------------------------------" | tee -a "$LOG"
python 3_filter_out_data.py \
  --input "$STEP2" \
  --output "$STEP3" \
  --max_clones "$MAX_CLONES" \
  2>&1 | tee -a "$LOG"

echo "------------------------------------------------------------" | tee -a "$LOG"
echo "[STEP 4] 4_gen_init_train_sample.py  $(date)" | tee -a "$LOG"
echo "------------------------------------------------------------" | tee -a "$LOG"
python 4_gen_init_train_sample.py \
  --input "$STEP3" \
  --output "$STEP4" \
  2>&1 | tee -a "$LOG"

echo "------------------------------------------------------------" | tee -a "$LOG"
echo "[STEP 5] 5_gen_neg_clone_sample.py  $(date)" | tee -a "$LOG"
echo "------------------------------------------------------------" | tee -a "$LOG"
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

echo "------------------------------------------------------------" | tee -a "$LOG"
echo "[STEP 6] 6_gen_pos_clone_sample.py  $(date)" | tee -a "$LOG"
echo "------------------------------------------------------------" | tee -a "$LOG"
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

echo "------------------------------------------------------------" | tee -a "$LOG"
echo "[STEP 7] 7_gen_clone_bench.py  $(date)" | tee -a "$LOG"
echo "------------------------------------------------------------" | tee -a "$LOG"
python 7_gen_clone_bench.py \
  --input "$STEP4" \
  --output "$STEP7_BENCH" \
  --dedup \
  2>&1 | tee -a "$LOG"

echo "------------------------------------------------------------" | tee -a "$LOG"
echo "[STEP 8] 8_split_comine_neg_pos_pairs.py  $(date)" | tee -a "$LOG"
echo "------------------------------------------------------------" | tee -a "$LOG"
python 8_split_comine_neg_pos_pairs.py \
  --neg "$STEP5_NEG_TXT" \
  --pos "$STEP6_POS_TXT" \
  --out_dir "$OUT_DIR" \
  --seed "$SEED" \
  2>&1 | tee -a "$LOG"

echo "============================================================" | tee -a "$LOG"
echo "Pipeline finished successfully: $(date)" | tee -a "$LOG"
echo "LOG: $LOG" | tee -a "$LOG"
echo "============================================================" | tee -a "$LOG"
