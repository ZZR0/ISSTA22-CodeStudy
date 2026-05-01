#!/usr/bin/env bash
set -u -o pipefail
shopt -s nullglob

DATASET_ROOT="../dataset/nicad_block_java"
OUT_ROOT="./saved_models_combined"
#OUT_ROOT="./saved_models_combined"

EPOCHS="${1:-2}"
SEED="${2:-3}"

mkdir -p "$OUT_ROOT/predictions_bcb_java_block_function" "$OUT_ROOT/logs"

# ---- one single log file for everything ----
RUN_LOG="$OUT_ROOT/logs/test_all_systems_bcb_java_block_function.log"
exec > >(tee -a "$RUN_LOG") 2>&1

# ---------- timing helpers ----------
script_start_ts=$(date +%s)

fmt_duration () {
  local total=$1
  printf "%02dh:%02dm:%02ds" $((total/3600)) $(((total%3600)/60)) $((total%60))
}
# -----------------------------------
# Collect systems (exclude bck/org and require only test.txt and data.jsonl)
systems=()
for d in "$DATASET_ROOT"/*; do
  [[ -d "$d" ]] || continue
  sys="$(basename "$d")"

  case "$sys" in
    bck|org|combined_org10_plus_repos) continue ;;
  esac

  # Only check for test.txt and data.jsonl
  if [[ ! -f "$d/test.txt" || ! -f "$d/data.jsonl" ]]; then
    echo "[SKIP] $sys missing: test.txt or data.jsonl"
    continue
  fi

  systems+=( "$sys" )
done

echo "================================================"
echo "Found ${#systems[@]} systems to test (excluded: bck, org)"
echo "DATASET_ROOT: $DATASET_ROOT"
echo "OUT_ROOT    : $OUT_ROOT"
echo "EPOCHS      : $EPOCHS"
echo "SEED        : $SEED"
echo "Log file    : $RUN_LOG"
echo "Start time  : $(date)"
echo "================================================"
echo

fail=0
tested=0
skipped_runtime=0

for sys in "${systems[@]}"; do
  train="$DATASET_ROOT/$sys/train.txt"
  valid="$DATASET_ROOT/$sys/valid.txt"
  test="$DATASET_ROOT/$sys/test.txt"

  preds_rel="predictions_bcb_java_block_function/predictions_${sys}_test.txt"
  preds_abs="$OUT_ROOT/$preds_rel"

  # optional: don't re-run if predictions already exist
  # comment out this block if you always want to re-test
  if [[ -f "$preds_abs" ]]; then
    echo "[SKIP] predictions already exist for $sys: $preds_abs"
    skipped_runtime=$((skipped_runtime+1))
    continue
  fi

  echo "================= TESTING $sys ================="
  echo "train: $train"
  echo "valid: $valid"
  echo "test : $test"
  echo "pred : $preds_abs"

  sys_start_ts=$(date +%s)

  # NOTE: no per-repo log; everything goes to $RUN_LOG via exec+tee above
  python run.py \
    --output_dir="$OUT_ROOT" \
    --model_type=roberta \
    --config_name=microsoft/codebert-base \
    --model_name_or_path=microsoft/codebert-base \
    --tokenizer_name=roberta-base \
    --do_test \
    --train_data_file="$train" \
    --eval_data_file="$valid" \
    --test_data_file="$test" \
    --epoch "$EPOCHS" \
    --block_size 400 \
    --train_batch_size 16 \
    --eval_batch_size 32 \
    --learning_rate 5e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --predictions_file="$preds_rel" \
    --seed "$SEED"

  rc=$?
  sys_end_ts=$(date +%s)
  sys_dur=$((sys_end_ts - sys_start_ts))

  if [[ $rc -ne 0 ]]; then
    echo "[FAIL] $sys (exit=$rc) duration: $(fmt_duration "$sys_dur")"
    fail=$((fail+1))
  else
    echo "[OK]   $sys duration: $(fmt_duration "$sys_dur")"
    tested=$((tested+1))
  fi
  echo
done

script_end_ts=$(date +%s)
total_dur=$((script_end_ts - script_start_ts))

echo "================================================"
echo "All done."
echo "  tested         : $tested"
echo "  skipped (pred) : $skipped_runtime"
echo "  failed         : $fail"
echo "  total          : $(fmt_duration "$total_dur")"
echo "  finished       : $(date)"
echo "Log saved to     : $RUN_LOG"
echo "================================================"
