#!/usr/bin/env bash
set -euo pipefail

# ------------------------------------------------------------
# Batch: join predictions (4 models) with refactorability CSVs
# Processes:
#   input/prediction_before_adapt/{codebert,codegpt,codet5,graphcodebert}/predictions_prefixed/predictions_*_test.txt
# ------------------------------------------------------------

ROOT="/home/user1-system11/research_dream/llm-clone/AST_Clone_Extractability"
PY_SCRIPT="$ROOT/join_pred_with_inout_refactorability.py"

PRED_ROOT="$ROOT/input/prediction_after_adapt"
JSONL_ROOT="$ROOT/output/java-functions"
OUT_ROOT="$ROOT/output/pred_after_adapt_refactorability"

MODELS=("codebert" "codegpt" "codet5" "graphcodebert")

RUN_TS="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$OUT_ROOT/logs"
LOG_FILE="$LOG_DIR/batch_join_pred_with_refactorability_${RUN_TS}.log"

mkdir -p "$OUT_ROOT" "$LOG_DIR"

# Redirect ALL output (stdout + stderr) to both terminal and log file
exec > >(tee -a "$LOG_FILE") 2>&1

echo "=============================================================="
echo "Batch Started : $(date)"
echo "ROOT          : $ROOT"
echo "PRED_ROOT     : $PRED_ROOT"
echo "JSONL_ROOT    : $JSONL_ROOT"
echo "OUT_ROOT      : $OUT_ROOT"
echo "MODELS        : ${MODELS[*]}"
echo "LOG_FILE      : $LOG_FILE"
echo "=============================================================="
echo

shopt -s nullglob

# Find a unique JSONL for a system; prefer "<system>__extractability.jsonl" if present.
find_jsonl_for_system() {
  local system="$1"

  local preferred="$JSONL_ROOT/${system}_extractability.jsonl"
  if [[ -f "$preferred" ]]; then
    echo "$preferred"
    return 0
  fi

  # Fallback: try to find exactly one matching JSONL
  # This prevents "ant" from matching "ant-ivy".
  local exact
  exact="$(find "$JSONL_ROOT" -maxdepth 1 -type f -name "${system}_extractability.jsonl" | head -n 1 || true)"
  if [[ -n "$exact" && -f "$exact" ]]; then
    echo "$exact"
    return 0
  fi

  # If not found, return "not found"
  echo ""
  return 2
}

total_jobs=0
ok_jobs=0
skip_jobs=0
err_jobs=0

for model in "${MODELS[@]}"; do
  PRED_DIR="$PRED_ROOT/$model/predictions"
  OUT_DIR="$OUT_ROOT/$model"
  mkdir -p "$OUT_DIR"

  echo "--------------------------------------------------------------"
  echo "MODEL: $model"
  echo "PRED_DIR: $PRED_DIR"
  echo "OUT_DIR : $OUT_DIR"
  echo "--------------------------------------------------------------"

  if [[ ! -d "$PRED_DIR" ]]; then
    echo "  [SKIP] Missing predictions directory: $PRED_DIR"
    ((skip_jobs++)) || true
    echo
    continue
  fi

  preds=( "$PRED_DIR"/predictions_*_test.txt )
  if [[ "${#preds[@]}" -eq 0 ]]; then
    echo "  [SKIP] No prediction files found in: $PRED_DIR"
    ((skip_jobs++)) || true
    echo
    continue
  fi

  for pred in "${preds[@]}"; do
    ((total_jobs++)) || true

    base="$(basename "$pred")"
    system="${base#predictions_}"
    system="${system%_test.txt}"

    echo
    echo "[RUN ] model=$model system=$system"
    echo "  pred : $pred"

    jsonl="$(find_jsonl_for_system "$system" 2> /tmp/jsonl_candidates_${RUN_TS}.txt || true)"

    if [[ -z "$jsonl" ]]; then
      rc=$?
      if [[ "$rc" -eq 2 ]]; then
        echo "  [ERR ] No JSONL found for system: $system"
      else
        echo "  [ERR ] Ambiguous JSONL matches for system: $system"
        echo "  Candidates:"
        sed 's/^/    /' /tmp/jsonl_candidates_${RUN_TS}.txt || true
      fi
      ((err_jobs++)) || true
      continue
    fi

    echo "  jsonl: $jsonl"

    out_csv="$OUT_DIR/pred_${system}_with_refactorability.csv"
    echo "  out  : $out_csv"

    python3 -u "$PY_SCRIPT" \
      --jsonl "$jsonl" \
      --pred "$pred" \
      --out "$out_csv"

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