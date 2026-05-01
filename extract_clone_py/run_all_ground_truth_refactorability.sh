#!/usr/bin/env bash
set -euo pipefail

# =====================================================
# CONFIG
# =====================================================

PY_SCRIPT="/home/user1-system11/research_dream/llm-clone/extract_clone_py/join_pred_with_inout_refactorability.py"

# Ground truth predictions (copied *_test.txt files)
PRED_DIR="/home/user1-system11/research_dream/llm-clone/extract_clone_py/data/ground_truth_test"

# Augmented JSONL directory (contains augmented_<system>.jsonl)
JSONL_DIR="/home/user1-system11/research_dream/llm-clone/extract_clone_py/data/augmented_post_nicad_func"

# Output directory (CSV)
OUT_DIR="/home/user1-system11/research_dream/llm-clone/extract_clone_py/output/ground_truth_test_refactorability"

mkdir -p "$OUT_DIR"

echo "========================================="
echo "Running refactorability evaluation (CSV)"
echo "PRED_DIR : $PRED_DIR"
echo "JSONL_DIR: $JSONL_DIR"
echo "OUT_DIR  : $OUT_DIR"
echo "========================================="
echo

shopt -s nullglob

find_jsonl_for_system() {
  local system="$1"
  local candidate="$JSONL_DIR/augmented_${system}.jsonl"
  if [[ -f "$candidate" ]]; then
    echo "$candidate"
    return 0
  fi

  # Fallback: case-insensitive match (e.g., Python vs python)
  local found
  found="$(find "$JSONL_DIR" -maxdepth 1 -type f -iname "augmented_${system}.jsonl" | head -n 1 || true)"
  if [[ -n "$found" && -f "$found" ]]; then
    echo "$found"
    return 0
  fi

  echo ""
  return 2
}

for pred in "$PRED_DIR"/*_test.txt; do
  base="$(basename "$pred")"          # ansible_test.txt
  system="${base%_test.txt}"          # ansible

  echo "Processing system: $system"

  jsonl="$(find_jsonl_for_system "$system" || true)"
  if [[ -z "${jsonl:-}" || ! -f "$jsonl" ]]; then
    echo "  ⚠ Missing JSONL for $system: expected augmented_${system}.jsonl — skipping"
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