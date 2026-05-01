#!/usr/bin/env bash
set -euo pipefail

# =====================================================
# CONFIG
# =====================================================

PY_SCRIPT="/home/user1-system11/research_dream/llm-clone/AST_Clone_Extractability/join_pred_with_inout_refactorability.py"

# Ground truth predictions (copied *_test.txt files)
PRED_DIR="/home/user1-system11/research_dream/llm-clone/AST_Clone_Extractability/input/ground_truth_test"

# Extractability JSONL directory
JSONL_DIR="/home/user1-system11/research_dream/llm-clone/AST_Clone_Extractability/output/java-functions"

# Output directory (CSV)
OUT_DIR="/home/user1-system11/research_dream/llm-clone/AST_Clone_Extractability/output/ground_truth_test_refactorability"

# Which similarity bucket to use
SIM_TAG="sim0.7"

mkdir -p "$OUT_DIR"

echo "========================================="
echo "Running refactorability evaluation (CSV)"
echo "PRED_DIR : $PRED_DIR"
echo "JSONL_DIR: $JSONL_DIR"
echo "OUT_DIR  : $OUT_DIR"
echo "SIM_TAG  : $SIM_TAG"
echo "========================================="
echo

shopt -s nullglob

for pred in "$PRED_DIR"/*_test.txt; do
  base="$(basename "$pred")"          # camel_test.txt
  system="${base%_test.txt}"          # camel

  echo "Processing system: $system"

  # -------------------------------------------------
  # 1) Prefer exact expected JSONL name
  # -------------------------------------------------
  jsonl="$JSONL_DIR/${system}-${SIM_TAG}_extractability.jsonl"

  # 2) Fallback: try to find a match if naming differs slightly
  if [[ ! -f "$jsonl" ]]; then
    jsonl="$(find "$JSONL_DIR" -maxdepth 1 -type f -name "${system}-${SIM_TAG}_extractability.jsonl" | head -n 1)"
  fi

  if [[ -z "${jsonl:-}" || ! -f "$jsonl" ]]; then
    echo "  ⚠ Missing JSONL for $system: expected ${system}-${SIM_TAG}_extractability.jsonl — skipping"
    echo
    continue
  fi

  out_csv="$OUT_DIR/pred_${system}_with_refactorability.csv"

  echo "  JSONL: $jsonl"
  echo "  PRED : $pred"
  echo "  OUT  : $out_csv"
  echo

  python "$PY_SCRIPT" \
    --jsonl "$jsonl" \
    --pred "$pred" \
    --out "$out_csv"

  echo "  ✓ Done $system"
  echo "-----------------------------------------"
  echo
done

echo "========================================="
echo "All systems processed."
echo "========================================="