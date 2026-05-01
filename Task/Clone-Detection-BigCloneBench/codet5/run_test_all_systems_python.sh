#!/usr/bin/env bash
set -euo pipefail

DATASET_ROOT="../dataset/python"
OUT_ROOT="./saved_models_combined_py/python"
SEED="${SEED:-3}"

# Global checkpoint to reuse for ALL repos:
CKPT_ROOT="${CKPT_ROOT:-/home/user1-system11/research_dream/llm-clone/Task/Clone-Detection-BigCloneBench/codet5/saved_models_combined_py/checkpoint-best-f1}"
CKPT_BIN="$CKPT_ROOT/pytorch_model.bin"

mkdir -p "$OUT_ROOT"
RUN_LOG="$OUT_ROOT/test_all_systems_codet5_bcb_python_combined.log"
exec > >(tee -a "$RUN_LOG") 2>&1

echo "================================================"
echo "DATASET_ROOT : $DATASET_ROOT"
echo "OUT_ROOT     : $OUT_ROOT"
echo "CKPT_BIN     : $CKPT_BIN"
echo "SEED         : $SEED"
echo "Log file     : $RUN_LOG"
echo "Start time   : $(date)"
echo "================================================"
echo

if [[ ! -f "$CKPT_BIN" ]]; then
  echo "[FATAL] Missing checkpoint file: $CKPT_BIN"
  exit 2
fi

# Collect systems (exclude bck/org/combined_org10_plus_repos and require files)
systems=()
for d in "$DATASET_ROOT"/*; do
  [[ -d "$d" ]] || continue
  sys="$(basename "$d")"
  case "$sys" in bck|org|combined_org10_plus_repos) continue ;; esac

  if [[ ! -f "$d/train.txt" || ! -f "$d/valid.txt" || ! -f "$d/test.txt" || ! -f "$d/data.jsonl" ]]; then
    echo "[SKIP] $sys missing one of: train.txt / valid.txt / test.txt / data.jsonl"
    continue
  fi
  systems+=( "$sys" )
done

echo "Found ${#systems[@]} systems to test"
echo

fail=0
tested=0

for sys in "${systems[@]}"; do
  sys_data_dir="$DATASET_ROOT/$sys"
  sys_out_dir="$OUT_ROOT/$sys"

  mkdir -p "$sys_out_dir/cache_data" "$sys_out_dir/predictions" "$sys_out_dir/tensorboard"

  # IMPORTANT: create the checkpoint path that run_clone.py expects
  mkdir -p "$sys_out_dir/checkpoint-best-f1"
  ln -sfn "$CKPT_BIN" "$sys_out_dir/checkpoint-best-f1/pytorch_model.bin"

  preds_rel="predictions/predictions_${sys}_test.txt"

  echo "-------------------- [$sys] $(date) --------------------"
  echo "data_dir     : $sys_data_dir"
  echo "output_dir   : $sys_out_dir"
  echo "checkpoint   : $sys_out_dir/checkpoint-best-f1/pytorch_model.bin -> $CKPT_BIN"
  echo "predictions  : $sys_out_dir/$preds_rel"
  echo

  # TEST ONLY (no training). Keep eval if your script needs it for metrics.
  python run_clone.py \
    --do_eval \
    --do_eval_bleu \
    --do_test \
    --task clone \
    --sub_task none \
    --model_type codet5 \
    --data_num -1 \
    --tokenizer_name Salesforce/codet5-base \
    --tokenizer_path ../../../CodeT5/tokenizer/salesforce \
    --model_name_or_path Salesforce/codet5-base \
    --output_dir "$sys_out_dir" \
    --summary_dir "$sys_out_dir/tensorboard" \
    --data_dir "$sys_data_dir" \
    --cache_path "$sys_out_dir/cache_data" \
    --res_dir predictions \
    --predictions_file "$preds_rel" \
    --res_fn "$sys_out_dir/clone_codet5_base_${sys}.txt" \
    --train_batch_size 4 \
    --eval_batch_size 4 \
    --max_source_length 400 \
    --max_target_length 400 \
    --gradient_accumulation_steps 2 \
    --seed "$SEED"

  rc=$?
  if [[ $rc -ne 0 ]]; then
    echo
    echo "[FAIL] $sys (exit=$rc)"
    fail=$((fail+1))
  else
    echo
    echo "[OK]   $sys"
  fi

  tested=$((tested+1))
  echo
done

echo "================================================"
echo "Finished: tested=$tested fail=$fail"
echo "End time: $(date)"
echo "Log file: $RUN_LOG"
echo "================================================"

exit $fail

