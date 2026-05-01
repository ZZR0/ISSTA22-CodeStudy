# GraphCodeBERT Training Optimization & Fix Report

**Date:** January 31, 2026
**Model:** GraphCodeBERT (microsoft/graphcodebert-base)
**Task:** Clone Detection (BigCloneBench)
**Status:** Optimized for Distributed Data Parallel (DDP)

---

## 1. Critical Bug Fixes in `run.py`
Similar to the CodeGPT implementation, the original `run.py` for GraphCodeBERT contained legacy logic that prevented it from running correctly in a modern DDP (`accelerate`) environment.

### Bug A: Missing Local Rank Initialization
The script did not read the `LOCAL_RANK` environment variable provided by `accelerate`. This caused all GPUs to attempt to act as the "master" process, leading to race conditions and Double Loading.

* **The Fix:** Injected code to parse `os.environ['LOCAL_RANK']`.

### Bug B: The "Silent Failure" (Evaluation/Save Skip)
The script contained a hardcoded condition that only allowed evaluation and saving if `local_rank == -1` (Single GPU).
* **Original Code:** `if args.local_rank == -1 and args.evaluate_during_training:`
* **The Issue:** In DDP, the main process is Rank 0. The original code forced Rank 0 to skip evaluation. Consequently, the `results` variable was never created, causing the script to crash with `UnboundLocalError` when attempting to check `if results['eval_f1'] > best_f1`.

* **The Fix:** Updated condition to `if args.local_rank in [-1, 0] ...`.
* **Patch Source:** [Commit 235f6c4](https://github.com/unoselab/ISSTA22-CodeStudy/commit/235f6c41cb0b0fec1a04528e5540bbf12891ecea#diff-f7331c5d651eee74b26539f43a9fd4940522d792881a8aec7663fe015c74dcf9) (Fix DDP save logic)

---

## 2. Optimization Strategy (Auto-Scaling)

We transitioned from `DataParallel` (inefficient single-process) to `DistributedDataParallel` (multi-process).

| Metric | Legacy Strategy | Optimized Strategy (DDP) |
| :--- | :--- | :--- |
| **Process Control** | Single Python Process | **Multi-Process via Accelerate** |
| **GPU Usage** | Unbalanced (GPU 0 bottleneck) | **100% Balanced** |
| **Batch Size Logic** | Fixed Global Batch | **Dynamic Auto-Scaling** |

### Smart Batching Math
GraphCodeBERT is more memory-intensive than CodeGPT due to the Data Flow Graph (DFG). We optimized the batch size to maximize VRAM usage on the RTX 6000 Ada while maintaining a stable Global Batch Size.

* **Target Global Batch:** 64
* **Per-GPU Batch:** 16 (Safe for 48GB VRAM with DFG)
* **Formula:** `Accumulation Steps = 64 / (16 * Num_GPUs)`

| # GPUs | Per GPU Batch | Accumulation Steps | **Total Effective Batch** |
| :--- | :--- | :--- | :--- |
| **1** | 16 | 4 | **64** |
| **2** | 16 | 2 | **64** |
| **3** | 16 | 1 | **48** (approx) |
| **4** | 16 | 1 | **64** |

---

## 3. How to Apply Fixes & Run

### Step 1: Patch `run.py` (One-Time Only)
You must apply these `sed` commands to fix the Python script before running.

```bash
# Fix 1: Initialize Local Rank from Environment
sed -i "/args = parser.parse_args()/a \    if args.local_rank == -1 and 'LOCAL_RANK' in os.environ:\\n        args.local_rank = int(os.environ['LOCAL_RANK'])" run.py

# Fix 2: Allow Rank 0 to Evaluate and Save (Commit 235f6c4)
sed -i "s/if args.local_rank == -1 and args.evaluate_during_training:/if args.local_rank in [-1, 0] and args.evaluate_during_training:/" run.py

```

### Step 2: Create the Auto-Launcher

Save the following as `run_auto.sh`. This script detects your hardware and configures `accelerate` automatically.

```bash
#!/bin/bash
# Auto-detect number of GPUs
NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)

# Smart Batch Calculation for GraphCodeBERT
PER_GPU_BATCH=16
TARGET_GLOBAL_BATCH=64
ACCUM_STEPS=$(( TARGET_GLOBAL_BATCH / (PER_GPU_BATCH * NUM_GPUS) ))
[ "$ACCUM_STEPS" -lt 1 ] && ACCUM_STEPS=1

# Build Command
LAUNCH_CMD="accelerate launch"
[ "$NUM_GPUS" -gt 1 ] && LAUNCH_CMD="$LAUNCH_CMD --multi_gpu --num_processes $NUM_GPUS"

# Execution
mkdir -p ./saved_models/
$LAUNCH_CMD run.py \
    --output_dir=./saved_models/ \
    --model_type=roberta \
    --config_name=microsoft/graphcodebert-base \
    --model_name_or_path=microsoft/graphcodebert-base \
    --tokenizer_name=microsoft/graphcodebert-base \
    --do_train \
    --do_test \
    --train_data_file=../dataset/train.txt \
    --eval_data_file=../dataset/valid.txt \
    --test_data_file=../dataset/test.txt \
    --epoch 2 \
    --code_length 400 \
    --data_flow_length 128 \
    --train_batch_size $PER_GPU_BATCH \
    --eval_batch_size 32 \
    --gradient_accumulation_steps $ACCUM_STEPS \
    --learning_rate 5e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --save_steps 500 \
    --save_total_limit 2 \
    --overwrite_output_dir \
    --seed 3 2>&1 | tee ./saved_models/train_auto_graphcodebert.log

```

### Step 3: Run

```bash
bash run_auto.sh

```
