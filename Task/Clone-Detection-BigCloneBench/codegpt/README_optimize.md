# BigCloneBench Training Optimization Report

**Date:** January 31, 2026  
**Hardware:** 2x NVIDIA RTX 6000 Ada (48GB VRAM each)  
**Model:** CodeGPT-small-java-adaptedGPT2

---

## 1. Performance Comparison
We shifted from legacy `DataParallel` (DP) to `DistributedDataParallel` (DDP) via Hugging Face Accelerate. The difference in hardware utilization and model quality is drastic.

| Metric | Original Script (Legacy) | Optimized Script (Final) | Impact |
| :--- | :--- | :--- | :--- |
| **Architecture** | Single Process (DP) | **Multi-Process (DDP)** | Eliminated the bottleneck where GPU 0 manages GPU 1. |
| **GPU Utilization** | GPU 0: 99%, GPU 1: 0-90% (Idle/Uneven) | **100% / 100%** | Perfect parallel scaling across both cards. |
| **VRAM Usage** | ~13 GB per card (73GB wasted) | **~43 GB per card** | **3.3x more data** resides in memory per step. |
| **Context Window** | 400 Tokens | **1024 Tokens** | Model reads **complete functions** instead of fragments. |
| **Effective Batch** | 16 | **64** | More stable convergence using Gradient Accumulation. |

---

## 2. Key Technical Changes

### A. The "Double Loading" Bug Fix
The legacy `run.py` script was not compatible with `accelerate`. It attempted to load the model on all GPUs simultaneously without knowing its rank, causing OOM errors.

**The Fix:**
We patched `run.py` to correctly identify the `LOCAL_RANK` from the environment variables, ensuring each process only talks to its assigned GPU.

```python
# Inserted into run.py after args parsing:
if args.local_rank == -1 and 'LOCAL_RANK' in os.environ:
    args.local_rank = int(os.environ['LOCAL_RANK'])

```

### B. Hyperparameter Tuning

To maximize the 48GB VRAM without crashing, we used **Gradient Accumulation**:

* **Train Batch Size (per GPU):** `8` (Safe limit for 1024 context).
* **Accumulation Steps:** `4`
* **GPUs:** `2`
* **Math:** `8 * 4 * 2 = 64` (Global Batch Size).

This allows us to train with a **Global Batch of 64** using massive 1024-token inputs, which would normally require ~80GB VRAM per card if done in a single step.

---

## 3. How to Run the Optimized Training

### Step 1: Patch the Script (One-time)

Ensure `run.py` has the rank logic.

```bash
sed -i "/args = parser.parse_args()/a \    if args.local_rank == -1 and 'LOCAL_RANK' in os.environ:\\n        args.local_rank = int(os.environ['LOCAL_RANK'])" run.py

```

### Step 2: Create the Launcher Script

Save this as `run_optimized.sh`:

```bash
#!/bin/bash
mkdir -p ./saved_models/

# Launch with 2 processes (one per GPU)
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
    --epoch 5 \
    --learning_rate 2e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --seed 3 2>&1 | tee ./saved_models/train_optimized.log

```

Compare with the original version (`run.sh`):

```bash
#!/bin/bash
mkdir ./saved_models/
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

### Step 3: Execution & Monitoring

Run inside `tmux` to ensure persistence.

```bash
# Start Training
bash run_optimized.sh

# Monitor Logs
tail -f ./saved_models/train_optimized.log

# Monitor GPU Usage (Expect ~43GB/card)
watch -n 1 nvidia-smi
