#!/bin/bash
set -euo pipefail

# ================================================================
# run-test-otherdomain-codebert.sh
# Test-only on OTHER domain (e.g., Camel) for CodeBERT
#
# IMPORTANT (per run.py):
#   - --train_data_file is REQUIRED even for --do_test only
#   - test mapping is loaded from:
#       <dir_of_test_file>/<test_type>/data.jsonl
#     so with TEST_FILE=../dataset/test_camel.txt and --test_type mix:
#       ../dataset/mix/data.jsonl is used.
#
# Notes:
#   - CodeBERT does NOT use --subsample_ratio (TextDataset loads all pairs).
# ================================================================

echo ">>> [Test-only] Cross-domain evaluation on OTHER domain (CodeBERT)"

# -------- Paths --------
OUTDIR="./saved_models"   # <-- change if your checkpoint dir differs
TEST_FILE="../dataset/test_camel.txt"

# Required by run.py argparse (even if we do not train)
# Use any existing train file; we typically point to a mix train file.
DUMMY_TRAIN="../dataset/train_mix.txt"

# Paper tokenizer setting
TOKENIZER="roberta-base"

# -------- Sanity checks --------
[ -d "$OUTDIR" ] || { echo "ERROR: OUTDIR not found: $OUTDIR"; exit 1; }
[ -f "$TEST_FILE" ] || { echo "ERROR: test file not found: $TEST_FILE"; exit 1; }
[ -f "$DUMMY_TRAIN" ] || { echo "ERROR: required train_data_file not found: $DUMMY_TRAIN"; exit 1; }

echo "    - Output Dir     : $OUTDIR"
echo "    - Base Model     : microsoft/codebert-base"
echo "    - Tokenizer      : $TOKENIZER"
echo "    - Test File      : $TEST_FILE"
echo "    - Required dummy : $DUMMY_TRAIN (argparse requires --train_data_file)"
echo "    - Mapping        : ../dataset/mix/data.jsonl (via --test_type mix)"
echo "    - Log File       : $OUTDIR/test_otherdomain.log"

# -------- Run (test-only) --------
accelerate launch run.py \
  --output_dir="$OUTDIR/" \
  --train_data_file="$DUMMY_TRAIN" \
  --model_type=roberta \
  --config_name=microsoft/codebert-base \
  --model_name_or_path=microsoft/codebert-base \
  --tokenizer_name="$TOKENIZER" \
  --test_type mix \
  --do_test \
  --test_data_file="$TEST_FILE" \
  --block_size 400 \
  --eval_batch_size 32 \
  --seed 3 2>&1 | tee "$OUTDIR/test_otherdomain.log"