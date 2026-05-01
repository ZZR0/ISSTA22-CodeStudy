#!/bin/bash

# --- Variables ---
DATA_DIR="$HOME/nicad-clone-azure"
OUTPUT_DIR="./saved_models_bcb"
LOG_FILE="${OUTPUT_DIR}/test-codegpt-bcb-azure.log"

# --- Setup ---
# Create output directory if it doesn't exist (-p prevents error if it exists)
mkdir -p "$OUTPUT_DIR"

# --- Execution ---
python run.py \
    --output_dir="$OUTPUT_DIR" \
    --model_type=gpt2 \
    --config_name=microsoft/CodeGPT-small-java-adaptedGPT2 \
    --model_name_or_path=microsoft/CodeGPT-small-java-adaptedGPT2 \
    --tokenizer_name=microsoft/CodeGPT-small-java-adaptedGPT2 \
    --do_test \
    --train_data_file="${DATA_DIR}/train.txt" \
    --eval_data_file="${DATA_DIR}/valid.txt" \
    --test_data_file="${DATA_DIR}/test.txt" \
    --epoch 2 \
    --block_size 400 \
    --train_batch_size 16 \
    --eval_batch_size 32 \
    --learning_rate 5e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --seed 3 2>&1 | tee "$LOG_FILE"
cp "$LOG_FILE" "${DATA_DIR}/test-codegpt-bcb-azure.log"


: <<'COMMENT'
mkdir ./saved_models/
python run.py \
    --output_dir=./saved_models/ \
    --model_type=gpt2 \
    --config_name=microsoft/CodeGPT-small-java-adaptedGPT2 \
    --model_name_or_path=microsoft/CodeGPT-small-java-adaptedGPT2 \
    --tokenizer_name=microsoft/CodeGPT-small-java-adaptedGPT2 \
    --do_test \
    --train_data_file=../dataset/train.txt \
    --eval_data_file=../dataset/valid.txt \
    --test_data_file=../dataset/test.txt \
    --epoch 2 \
    --block_size 400 \
    --train_batch_size 16 \
    --eval_batch_size 32 \
    --learning_rate 5e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --seed 3 2>&1| tee ./saved_models/test.log
COMMENT
