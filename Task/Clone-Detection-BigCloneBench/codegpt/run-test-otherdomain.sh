#!/bin/bash
set -euo pipefail

# ================================================================
# run-test-otherdomain.sh
# Test-only on OTHER domain (e.g., Camel)
#
# IMPORTANT (per run.py):
#   - --train_data_file is REQUIRED even for --do_test only
#   - test mapping is loaded from:
#       <dir_of_test_file>/<test_type>/data.jsonl
#     so with TEST_FILE=../dataset/test_camel.txt and --test_type mix:
#       ../dataset/mix/data.jsonl is used.
# ================================================================

echo ">>> [Test-only] Cross-domain evaluation on OTHER domain"

OUTDIR="./saved_models_mix"
TEST_FILE="../dataset/test_camel.txt"

# Required by run.py argparse (even if we do not train)
DUMMY_TRAIN="../dataset/train_mix.txt"

[ -d "$OUTDIR" ] || { echo "ERROR: OUTDIR not found: $OUTDIR"; exit 1; }
[ -f "$TEST_FILE" ] || { echo "ERROR: test file not found: $TEST_FILE"; exit 1; }
[ -f "$DUMMY_TRAIN" ] || { echo "ERROR: required train_data_file not found: $DUMMY_TRAIN"; exit 1; }

echo "    - Output Dir     : $OUTDIR"
echo "    - Base Model     : microsoft/CodeGPT-small-java-adaptedGPT2"
echo "    - Test File      : $TEST_FILE"
echo "    - Required dummy : $DUMMY_TRAIN (argparse requires --train_data_file)"
echo "    - Mapping        : ../dataset/mix/data.jsonl (via --test_type mix)"
echo "    - Log File       : $OUTDIR/test_otherdomain.log"

accelerate launch run.py \
  --output_dir="$OUTDIR/" \
  --train_data_file="$DUMMY_TRAIN" \
  --subsample_ratio 1.0 \
  --model_type=gpt2 \
  --config_name=microsoft/CodeGPT-small-java-adaptedGPT2 \
  --model_name_or_path=microsoft/CodeGPT-small-java-adaptedGPT2 \
  --tokenizer_name=microsoft/CodeGPT-small-java-adaptedGPT2 \
  --test_type mix \
  --do_test \
  --test_data_file="$TEST_FILE" \
  --block_size 400 \
  --eval_batch_size 32 \
  --seed 3 2>&1 | tee "$OUTDIR/test_otherdomain.log"
