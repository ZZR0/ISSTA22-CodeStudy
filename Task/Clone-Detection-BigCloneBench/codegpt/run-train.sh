#!/bin/bash

# ==================================================================
# [Final Production] Full Dataset Training Pipeline
# ==================================================================

# ------------------------------------------------------------------
# 1. Safety Checks (Patching run.py)
# ------------------------------------------------------------------
echo ">>> [Pre-Flight Check] Applying safety patches to run.py..."

# Patch A: Disable logic that overrides user's --save_steps
# We comment out the lines where run.py forces save_steps = len(dataloader)
sed -i "s/args.save_steps=len( train_dataloader)/# args.save_steps=len( train_dataloader)/g" run.py
sed -i "s/args.logging_steps=len( train_dataloader)/# args.logging_steps=len( train_dataloader)/g" run.py

# Patch B: Ensure the first checkpoint is saved
# Change best_f1 initialization from 0 to -1 (prevents skipping save if F1 is 0.0)
sed -i "s/best_f1=0/best_f1=-1/g" run.py

echo "    - Patched: run.py now respects --save_steps argument."
echo "    - Patched: Initial best_f1 set to -1."


# ------------------------------------------------------------------
# 2. Hardware Auto-Configuration
# ------------------------------------------------------------------
echo ">>> [System Config] Detecting Hardware..."

# Count available GPUs
NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)

# Hyperparameters
PER_GPU_BATCH=8
TARGET_GLOBAL_BATCH=64

# Calculate Gradient Accumulation Steps
# Formula: Accumulation = 64 / (8 * Num_GPUs)
ACCUM_STEPS=$(( TARGET_GLOBAL_BATCH / (PER_GPU_BATCH * NUM_GPUS) ))

# Safety: Ensure at least 1 accumulation step
[ "$ACCUM_STEPS" -lt 1 ] && ACCUM_STEPS=1

echo "    - GPUs Detected : $NUM_GPUS"
echo "    - Accumulation  : $ACCUM_STEPS"
echo "    - Target Batch  : $TARGET_GLOBAL_BATCH"


# ------------------------------------------------------------------
# 3. Execution (Full Data)
# ------------------------------------------------------------------
LAUNCH_CMD="accelerate launch"
[ "$NUM_GPUS" -gt 1 ] && LAUNCH_CMD="$LAUNCH_CMD --multi_gpu --num_processes $NUM_GPUS"

# Ensure output directory exists
mkdir -p ./saved_models/

echo ">>> [Start] Initiating Full Training..."
echo "    - Output Dir : ./saved_models/"
echo "    - Log File   : ./saved_models/train.log"
echo "    - Save Freq  : Every 500 steps"

$LAUNCH_CMD run.py \
    --output_dir=./saved_models/ \
    --subsample_ratio 1.0 \
    --model_type=gpt2 \
    --config_name=microsoft/CodeGPT-small-java-adaptedGPT2 \
    --model_name_or_path=microsoft/CodeGPT-small-java-adaptedGPT2 \
    --tokenizer_name=microsoft/CodeGPT-small-java-adaptedGPT2 \
    --do_train \
    --do_test \
    --train_data_file=../dataset/train.txt \
    --eval_data_file=../dataset/valid.txt \
    --test_data_file=../dataset/test.txt \
    --block_size 400 \
    --train_batch_size $PER_GPU_BATCH \
    --gradient_accumulation_steps $ACCUM_STEPS \
    --eval_batch_size 32 \
    --epoch 2 \
    --learning_rate 5e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --save_steps 500 \
    --logging_steps 100 \
    --save_total_limit 2 \
    --overwrite_output_dir \
    --seed 3 2>&1 | tee ./saved_models/train.log