#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/user1-system11/research_dream/llm-clone/extract_clone_py"
PY_SCRIPT="$ROOT/join_pred_with_inout_refactorability.py"

PRED_ROOT="$ROOT/data/prediction_before_adapt"
JSONL_ROOT="$ROOT/data/augmented_post_nicad_func"
OUT_ROOT="$ROOT/output/pred_before_adapt_refactorability"

MODELS=("codebert" "codegpt" "codet5" "graphcodebert")

RUN_TS="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$OUT_ROOT/logs"
LOG_FILE="$LOG_DIR/batch_join_pred_with_refactorability_${RUN_TS}.log"

mkdir -p "$OUT_ROOT" "$LOG_DIR"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "=============================================================="
echo "Batch Started : $(date)"
echo "ROOT          : $ROOT"
echo "PY_SCRIPT     : $PY_SCRIPT"
echo "PRED_ROOT     : $PRED_ROOT"
echo "JSONL_ROOT    : $JSONL_ROOT"
echo "OUT_ROOT      : $OUT_ROOT"
echo "MODELS        : ${MODELS[*]}"
echo "LOG_FILE      : $LOG_FILE"
echo "=============================================================="
echo

shopt -s nullglob

realpath_f() {
  local p="$1"
  readlink -f "$p" 2>/dev/null || echo "$p"
}

find_preds_for_model() {
  local model="$1"
  local model_root
  model_root="$(realpath_f "$PRED_ROOT/$model")"
  [[ -e "$model_root" ]] || return 2
  find -L "$model_root" -type f -name "predictions_*_test.txt" 2>/dev/null | sort
}

find_jsonl_for_system() {
  local system="$1"
  local exact="$JSONL_ROOT/augmented_${system}.jsonl"
  if [[ -f "$exact" ]]; then
    printf '%s\n' "$exact"
    return 0
  fi

  local found
  found="$(find -L "$JSONL_ROOT" -maxdepth 1 -type f -iname "augmented_${system}.jsonl" | head -n 1 || true)"
  if [[ -n "${found:-}" && -f "$found" ]]; then
    printf '%s\n' "$found"
    return 0
  fi

  return 2
}

total_jobs=0
ok_jobs=0
skip_jobs=0
err_jobs=0

# Sanity checks
[[ -f "$PY_SCRIPT" ]] || { echo "[FATAL] PY_SCRIPT not found: $PY_SCRIPT"; exit 1; }
[[ -d "$JSONL_ROOT" ]] || { echo "[FATAL] JSONL_ROOT not found: $JSONL_ROOT"; exit 1; }
[[ -d "$PRED_ROOT" ]] || { echo "[FATAL] PRED_ROOT not found: $PRED_ROOT"; exit 1; }

for model in "${MODELS[@]}"; do
  OUT_DIR="$OUT_ROOT/$model"
  mkdir -p "$OUT_DIR"

  echo "--------------------------------------------------------------"
  echo "MODEL   : $model"
  echo "MODEL_ROOT (raw): $PRED_ROOT/$model"
  echo "OUT_DIR : $OUT_DIR"
  echo "--------------------------------------------------------------"

  mapfile -t preds < <(find_preds_for_model "$model" || true)

  if [[ "${#preds[@]}" -eq 0 ]]; then
    echo "  [SKIP] No prediction files found for model=$model under $PRED_ROOT/$model (searched recursively)"
    ((skip_jobs++)) || true
    echo
    continue
  fi

  echo "  Found ${#preds[@]} prediction files."

  for pred in "${preds[@]}"; do
    ((total_jobs++)) || true

    base="$(basename "$pred")"
    system="${base#predictions_}"
    system="${system%_test.txt}"

    echo
    echo "[RUN ] model=$model system=$system"
    echo "  pred : $pred"

    # --- JSONL lookup (no candidates file) ---
    set +e
    jsonl="$(find_jsonl_for_system "$system" 2>/dev/null)"
    rc=$?
    set -e

    if [[ "$rc" -ne 0 || -z "${jsonl:-}" ]]; then
      echo "  [ERR ] Missing JSONL for system=$system"
      echo "        Expected: $JSONL_ROOT/augmented_${system}.jsonl"
      ((err_jobs++)) || true
      continue
    fi

    echo "  jsonl: $jsonl"

    out_csv="$OUT_DIR/pred_${system}_with_refactorability.csv"
    echo "  out  : $out_csv"

    set +e
    python3 -u "$PY_SCRIPT" \
      --jsonl "$jsonl" \
      --pred "$pred" \
      --out "$out_csv"
    run_rc=$?
    set -e

    if [[ "$run_rc" -ne 0 ]]; then
      echo "  [ERR ] Join failed (rc=$run_rc) model=$model system=$system"
      ((err_jobs++)) || true
      continue
    fi

    echo "  [OK  ] Done -> $out_csv"
    ((ok_jobs++)) || true
  done

  echo
done

echo "=============================================================="
echo "Batch Finished: $(date)"
echo "Total jobs : $total_jobs"
echo "OK         : $ok_jobs"
echo "SKIP       : $skip_jobs"
echo "ERROR      : $err_jobs"
echo "=============================================================="