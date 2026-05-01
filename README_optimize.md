# BigCloneBench Training Optimization Report

**Date:** January 31, 2026  
**Hardware:** 2x NVIDIA RTX 6000 Ada (48GB VRAM each)  
**Model:** CodeGPT-small-java-adaptedGPT2

---

## 1. Performance Comparison

We shifted from legacy `DataParallel` (DP) to `DistributedDataParallel` (DDP) via Hugging Face Accelerate. The difference in hardware utilization and model quality is drastic.

| Metric              | Original Script (Legacy)       | Optimized Script (Final) | Impact                                                   |
| :------------------ | :----------------------------- | :----------------------- | :------------------------------------------------------- |
| **Architecture**    | Single Process (DP)            | **Multi-Process (DDP)**  | Eliminated the bottleneck where GPU 0 manages GPU 1.     |
| **GPU Utilization** | GPU 0: 99%, GPU 1: Idle/Uneven | **100% / 100%**          | Perfect parallel scaling across both cards.              |
| **VRAM Usage**      | ~13 GB per card (73GB wasted)  | **~43 GB per card**      | **3.3x more data** resides in memory per step.           |
| **Context Window**  | 400 Tokens                     | **1024 Tokens**          | Model reads **complete functions** instead of fragments. |
| **Effective Batch** | 16                             | **64**                   | **4x Stability.** See Section 2.C for math.              |

---

## 2. Key Technical Changes & Evidence

### A. The "Double Loading" Bug Fix

The legacy `run.py` script was not compatible with `accelerate`. It attempted to load the model on all GPUs simultaneously without knowing its rank, causing OOM errors.

**The Fix:**
We patched `run.py` to correctly identify the `LOCAL_RANK` from the environment variables, ensuring each process only talks to its assigned GPU.

### B. The "Silent Save Failure" Fix (Critical)

The legacy script only allowed evaluation/saving if `local_rank == -1` (Single GPU). In DDP mode, the main process is `Rank 0`, so the script skipped saving, causing the process to crash at the end or exit without saving.

**The Fix:**
We modified the condition to allow `Rank 0` to perform evaluation and saving.

- **Before:** `if args.local_rank == -1 and ...`
- **After:** `if args.local_rank in [-1, 0] and ...`
- **Patch Source:** [Commit a805deb](https://github.com/unoselab/ISSTA22-CodeStudy/commit/a805debfa4921f18f60cb5ace37fa9a9ad092675) (Fix DDP save logic)

### C. Hyperparameter Tuning (Math Verification)

We replaced the unstable small batch with **Gradient Accumulation** to stabilize training while using the massive 1024-token context.

**Log Evidence:**

- **Original Log:** `n_gpu=2`, `train_batch_size=16` (Split 8 per GPU). Total = **16**.
- **Optimized Log:** `n_gpu=1` (per process), `train_batch_size=8`, `accum=4`.

**Optimized Batch Calculation:**

$$
\text{8 (GPU Batch)} \times \text{2 (Processors)} \times \text{4 (Accum Steps)} = \mathbf{64} \text{ (Global Batch)}
$$

This **4x larger batch size** (64 vs 16) reduces gradient noise, ensuring the model converges smoothly rather than oscillating.

---

## 3. Script Comparison

### Optimized Launcher (`run_optimized.sh`)

_Uses `accelerate` for DDP, 1024 tokens, accumulation, and **safe checkpointing**._

```bash
#!/bin/bash
mkdir -p ./saved_models/

# Launch with 2 processes (one per GPU)
# Added --save_steps and --overwrite_output_dir to ensure checkpoints are saved safely
accelerate launch --multi_gpu --num_processes 2 run.py \
    --output_dir=./saved_models/ \
    --model_type=gpt2 \
    --config_name=microsoft/CodeGPT-small-java-adaptedGPT2 \
    --model_name_or_path=microsoft/CodeGPT-small-java-adaptedGPT2 \
    --tokenizer_name=microsoft/CodeGPT-small-java-adaptedGPT2 \
    --do_train \
    --do_test \
    --train_data_file=../dataset/train.txt \
    --eval_data_file=../dataset/valid.txt \
    --test_data_file=../dataset/test.txt \
    --block_size 1024 \
    --train_batch_size 8 \
    --gradient_accumulation_steps 4 \
    --eval_batch_size 32 \
    --epoch 2 \
    --learning_rate 5e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --save_steps 500 \
    --save_total_limit 2 \
    --overwrite_output_dir \
    --seed 3 2>&1 | tee ./saved_models/train_optimized_v2.log

```

### Original Launcher (`run.sh`)

_Legacy Python execution, 400 tokens, small batch._

```bash
#!/bin/bash
mkdir -p ./saved_models/
python run.py \
    --output_dir=./saved_models/ \
    --model_type=gpt2 \
    --config_name=microsoft/CodeGPT-small-java-adaptedGPT2 \
    --model_name_or_path=microsoft/CodeGPT-small-java-adaptedGPT2 \
    --tokenizer_name=microsoft/CodeGPT-small-java-adaptedGPT2 \
    --do_train \
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
    --seed 3 2>&1| tee ./saved_models/train.log

```

---

## 4. How to Run

### Step 1: Patch the Script (Essential)

You must apply **two patches** to `run.py` to support `accelerate`.

**Patch 1: Initialize Local Rank correctly**

```bash
sed -i "/args = parser.parse_args()/a \    if args.local_rank == -1 and 'LOCAL_RANK' in os.environ:\\n        args.local_rank = int(os.environ['LOCAL_RANK'])" run.py

```

**Patch 2: Enable Saving for Rank 0 ([Commit a805deb](https://github.com/unoselab/ISSTA22-CodeStudy/commit/a805debfa4921f18f60cb5ace37fa9a9ad092675))**

```bash
sed -i "s/if args.local_rank == -1 and args.evaluate_during_training:/if args.local_rank in [-1, 0] and args.evaluate_during_training:/" run.py

```

### Step 2: Execution & Monitoring

Run inside `tmux` to ensure persistence.

```bash
# Start Training
bash run_optimized.sh

# Monitor Logs
tail -f ./saved_models/train_optimized_v2.log

# Monitor GPU Usage (Expect ~43GB/card)
watch -n 1 nvidia-smi
```
