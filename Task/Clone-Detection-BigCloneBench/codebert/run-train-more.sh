#!/bin/bash
set -euo pipefail

# ================================================================
# run-train-more-codebert-single.sh
# Single-process fine-tune + test (original-author style)
# Fine-tune CodeBERT on MIXED train/valid (BCB10% + Other domain)
# and evaluate on OTHER-domain-only test (e.g., Camel).
#
# Inputs:
#   ../dataset/train_mix.txt
#   ../dataset/valid_mix.txt
#   ../dataset/test_camel.txt            (OTHER-domain-only test)
#   ../dataset/mix/data.jsonl            (loaded via --test_type mix)
#
# Notes:
#   - Single process: python run.py (no accelerate, no DDP)
#   - Avoids multi-process race on checkpoint-best-f1/model.bin
# ================================================================

OUTDIR="./saved_models_codebert_mix"
mkdir -p "$OUTDIR"

TRAIN_FILE="../dataset/train_mix.txt"
VALID_FILE="../dataset/valid_mix.txt"
TEST_FILE="../dataset/test_camel.txt"

# Sanity checks
[ -f "$TRAIN_FILE" ] || { echo "ERROR: Missing train file: $TRAIN_FILE"; exit 1; }
[ -f "$VALID_FILE" ] || { echo "ERROR: Missing valid file: $VALID_FILE"; exit 1; }
[ -f "$TEST_FILE" ]  || { echo "ERROR: Missing test file:  $TEST_FILE"; exit 1; }

# Tokenizer setting (paper)
TOKENIZER="roberta-base"

echo ">>> [Single-process] Fine-tune CodeBERT on mixed data + test on other-domain only"
echo "    - Output Dir : $OUTDIR"
echo "    - Train      : $TRAIN_FILE"
echo "    - Valid      : $VALID_FILE"
echo "    - Test (OD)  : $TEST_FILE"
echo "    - Mapping    : ../dataset/mix/data.jsonl (via --test_type mix)"
echo "    - Base Model : microsoft/codebert-base"
echo "    - Tokenizer  : $TOKENIZER"
echo "    - Log File   : $OUTDIR/train_mix.log"

python run.py \
  --output_dir="$OUTDIR/" \
  --model_type=roberta \
  --config_name=microsoft/codebert-base \
  --model_name_or_path=microsoft/codebert-base \
  --tokenizer_name="$TOKENIZER" \
  --test_type mix \
  --do_train \
  --do_test \
  --train_data_file="$TRAIN_FILE" \
  --eval_data_file="$VALID_FILE" \
  --test_data_file="$TEST_FILE" \
  --epoch 2 \
  --block_size 400 \
  --train_batch_size 16 \
  --eval_batch_size 32 \
  --learning_rate 5e-5 \
  --max_grad_norm 1.0 \
  --evaluate_during_training \
  --seed 3 2>&1 | tee "$OUTDIR/train_mix.log"