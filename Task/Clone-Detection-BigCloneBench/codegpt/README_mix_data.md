# Mixed-Domain Code Clone Detection Pipeline

This repository hosts a robust pipeline for training and evaluating code clone detection models (specifically **CodeGPT**) on mixed-domain datasets. It is designed to merge the standard **BigCloneBench (BCB)** dataset with custom "Other-Domain" datasets (e.g., Azure, Camel) while ensuring data integrity, reproducibility, and correct cross-domain evaluation.

---

## 📂 Detailed File Descriptions

### 1. Data Processing Scripts

#### `1_sample_data.py` (The Sampler)

A utility to create deterministic, sampled subsets of the BigCloneBench dataset.

- **Key Functionality**: Reads `train.txt` and `valid.txt` and outputs sampled versions (e.g., `train_10percent.txt`).
- **Sampling Modes** (`--mode`):
- `balanced`: Downsamples the majority class to achieve a perfect 50:50 split.

- **Reproducibility**: Uses a fixed `--seed` (default: 3) to ensure the exact same samples are generated every time.

#### `2_mix_data.py` (The Mixer)

The core logic for merging two distinct datasets into a unified "Mixed" dataset.

- **Collision Prevention**: Automatically prefixes IDs from different sources to prevent conflicts (e.g., `bcb_123` vs `azure_123`).
- **JSONL Merging**: Combines the source code mapping files (`data.jsonl`) from both domains into a single `dataset/mix/data.jsonl` file.
- **Test Set Generation**: Optionally processes an "Other-Domain" test file (e.g., `test.txt`) into a labeled test set (e.g., `test_azure.txt`) for isolated cross-domain evaluation.

#### `3_verify_data.py` (The Validator)

A data integrity checker that ensures the generated datasets are valid before training begins.

- **Integrity Check**: Scans `train`, `valid`, and `test` pair files to verify that every ID referenced exists in the master `data.jsonl` mapping.
- **Reporting**: Provides detailed error reports, listing specific line numbers and missing IDs if the merge process failed.
- **Strict Mode**: Can be run with `--strict` to force a non-zero exit code upon finding any missing pairs, stopping the pipeline immediately.

---

### 2. Orchestration Scripts (Shell)

#### `run-mix-data.sh` (Data Pipeline Orchestrator)

The master script that automates the data preparation workflow.

1. **Sampling**: Calls `1_sample_data.py` to create a balanced 10% subset of BCB.
2. **Mixing**: Calls `2_mix_data.py` to merge the BCB subset with the target Azure/Camel dataset.
3. **Verification**: Calls `3_verify_data.py` to confirm the output files are valid.

- _Customization_: Edit the `DATA_DIR` variable to point to your specific target domain data.

#### `run-train-more.sh` (Training Orchestrator)

A specialized script for fine-tuning **CodeGPT** on the mixed dataset.

- **Hardware Auto-Detection**: Automatically counts available GPUs using `nvidia-smi` and calculates the optimal `gradient_accumulation_steps` to maintain a target global batch size of 64.
- **Distributed Training**: Uses `accelerate launch` to handle multi-GPU processes.
- **Model Config**: Fine-tune pre-trained LLMs (e.g., `microsoft/CodeGPT-small-java-adaptedGPT2`) with a block size of 400 and 2 training epochs.

#### `run-test-otherdomain.sh` (Inference Orchestrator)

Designed for running **inference only** on a specific target domain without retraining.

- **Test-Only Mode**: Executes `run.py` with `--do_test` while explicitly disabling training.
- **Constraint Handling**: Automatically supplies a "dummy" training file path to satisfy `run.py`'s argument requirements, allowing the script to proceed purely for evaluation.

---

### 3. Core Training Engine

#### `run.py` (The Trainer)

The main Python driver for training and evaluation based on Hugging Face Transformers.

- **Model Support**: Compatible with GPT-2, BERT, RoBERTa, and DistilBERT architectures.
- **Dynamic Loading**: Uses a `TextDataset` class that reads pairs (`id1`, `id2`) and dynamically retrieves source code from the JSONL mapping during training.
- **Subsampling**: Includes a `--subsample_ratio` argument to train on a random fraction of the data for rapid prototyping.
- **Metrics**: Automatically calculates and reports Precision, Recall, and F1 scores, saving the best checkpoint based on the F1 score.

---

## 🚀 Quick Start

### 1. Prepare Data

```bash
# Generates dataset/mix_azure/ containing merged train/valid/test files
./run-mix-data.sh

```

### 2. Train Model

```bash
# Fine-tunes CodeGPT on the mixed data using all available GPUs
./run-train-more.sh

```

### 3. Evaluate (Cross-Domain)

```bash
# Tests the trained model on the 'test_camel.txt' file
./run-test-otherdomain.sh

```
