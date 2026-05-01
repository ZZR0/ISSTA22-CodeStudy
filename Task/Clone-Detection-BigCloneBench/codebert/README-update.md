# README-update.md (CodeBERT Experiment Updates) — 2026-02-05

This document summarizes the updates and experimental setup applied to  
`Clone-Detection-BigCloneBench/codebert/`, focusing on reproducible training,
data handling, and paper-faithful reproduction.

---

## 0) Directory Structure

### codebert/
- `run.py` : Main training / evaluation script
- `run-train.sh` : Training script (paper reproduction setting)
- `sample_data.py` : Script for deterministic 10% dataset generation
- `eval.sh`, `run.sh`, `model.py` : Original files (unchanged)

### dataset/ (relative path: `../dataset/`)

../dataset/
├── data.jsonl
├── train.txt
├── valid.txt
├── test.txt
├── train_10percent.txt
├── valid_10percent.txt
├── bcb/
└── camel/

---

## 1) Background: Hard-coded 10% Sampling in `run.py`

The original `run.py` contained the following logic, which implicitly reduced
train/validation data to 10%:

```python
if 'test' not in postfix:
    data = random.sample(data, int(len(data)*0.1))

This behavior made it difficult to control dataset size externally and caused
confusion when combined with pre-sampled datasets.

Design decision:
All sampling is now handled outside run.py, via explicit dataset files.

⸻

2) Dataset Preparation: Deterministic & Balanced 10% Sampling

2.1 Goals
	•	Deterministic sampling using a fixed random seed
	•	Explicit control of dataset size via files
	•	Optional label balancing (0/1 = 50:50)

2.2 Script

sample_data.py is used to generate 10% subsets of train.txt and valid.txt.

Example execution:

python sample_data.py --seed 3 --mode balanced

2.3 Generated Files
	•	../dataset/train_10percent.txt
	•	../dataset/valid_10percent.txt

Example statistics:
	•	Train: 901,028 → 90,102 (10%)
	•	Valid: 415,416 → 41,540 (10%)

Balanced label distribution:
	•	train_10percent: label0 = 45,051 / label1 = 45,051
	•	valid_10percent: label0 = 20,770 / label1 = 20,770

Note:
	•	--mode stratified preserves the original label ratio
	•	--mode balanced enforces a 50:50 label split

⸻

3) Updates to run.py

3.1 Disable Internal 10% Sampling

The hard-coded sampling logic in TextDataset has been commented out:

# NOTE: 10% sampling is handled externally (train_10percent.txt / valid_10percent.txt)
# if 'test' not in postfix:
#     data = random.sample(data, int(len(data)*0.1))

This ensures that datasets are loaded exactly as provided by the file paths.

⸻

3.2 Remove Unused File Descriptor

An unused open() call was removed to avoid unnecessary file handles:

# f = open(index_filename)  # unused file descriptor


⸻

3.3 Dataset Size Logging

To verify correct dataset loading during execution, the following log was added
before feature construction:

logger.info(f"[Dataset] postfix={postfix} pairs_loaded={len(data)}")

Expected values:
	•	train_10percent → pairs_loaded=90102
	•	valid_10percent → pairs_loaded=41540

⸻

4) Training Script: run-train.sh

4.1 Purpose
	•	Reproduce the original paper setting
	•	Use externally generated 10% datasets
	•	Keep tokenizer consistent with the original implementation

4.2 Script Overview

# TOKENIZER=microsoft/codebert-base   # solution default (future use)
TOKENIZER=roberta-base  # paper setting; original paper reproduction

TRAIN=../dataset/train_10percent.txt
VALID=../dataset/valid_10percent.txt
TEST=../dataset/test.txt

mkdir -p ./saved_models/

python run.py \
  --output_dir=./saved_models/ \
  --model_type=roberta \
  --config_name=microsoft/codebert-base \
  --model_name_or_path=microsoft/codebert-base \
  --tokenizer_name=$TOKENIZER \
  --do_train \
  --do_test \
  --train_data_file=$TRAIN \
  --eval_data_file=$VALID \
  --test_data_file=$TEST \
  --epoch 2 \
  --block_size 400 \
  --train_batch_size 16 \
  --eval_batch_size 32 \
  --learning_rate 5e-5 \
  --max_grad_norm 1.0 \
  --evaluate_during_training \
  --seed 3 2>&1 | tee ./saved_models/train.log

4.3 Execution

bash run-train.sh


⸻

5) Log Verification Checklist

5.1 Dataset Loading

grep -n "$begin:math:display$Dataset$end:math:display$" ./saved_models/train.log | head

Expected:
	•	train: pairs_loaded=90102
	•	valid: pairs_loaded=41540

⸻

5.2 Training Summary

grep -nE "Running training|Num examples|Total optimization steps|Total train batch size|Gradient Accumulation" \
  ./saved_models/train.log | head -n 30


⸻

5.3 Evaluation Results

grep -n "Eval results" ./saved_models/train.log | tail -n 40


⸻

6) Next Step: Solution-Oriented Configuration

After completing paper-faithful reproduction, the next planned step is a
solution-oriented configuration focused on maximizing performance:
	•	Switch tokenizer to:

TOKENIZER=microsoft/codebert-base


	•	Run an ablation experiment with all other settings unchanged
	•	Compare results against the paper setting

This separation ensures:
	•	Fair reproduction of prior work
	•	Clear attribution of performance gains in the proposed solution
