# CodeBERT on BigCloneBench (BCB)

This document details the evaluation of the **CodeBERT-BCB** model on the BigCloneBench (BCB) dataset. This specific run focuses on **binary code clone detection** within the provided framework.

## 🏗 Model & Dataset Overview

### Model Specifications

* **Base Model:** `microsoft/codebert-base`
* **Architecture:** RoBERTa-based encoder pre-trained on source code.
* **Framework:** HuggingFace Transformers + PyTorch.

### Dataset Details

* **Benchmark:** BigCloneBench (BCB)
* **Test Source:** `dataset/bcb/test.txt`
* **Pair Count:** 415,416 code pairs.
* **Evaluation Mode:** Within-project (BCB  BCB).

---

## ⚙️ Environment & Configuration

### Runtime Environment

| Component | Specification |
| --- | --- |
| **Python** | 3.11 |
| **PyTorch** | GPU-enabled |
| **Hardware** | 2 × CUDA Devices |
| **Mixed Precision** | Disabled (FP32) |

### Key Parameters

The following configuration was used for the test-only evaluation:

```bash
# Example Execution Command
python run_clone.py \
    --model_type roberta \
    --model_name_or_path microsoft/codebert-base \
    --do_test \
    --train_data_file dataset/bcb/train.txt \
    --test_data_file dataset/bcb/test.txt \
    --block_size 400 \
    --eval_batch_size 32 \
    --threshold 0.5

```

---

## 📊 Execution Summary

### 1. Feature Construction

* **Total Pairs:** 415,416
* **Time Elapsed:** ~4 minutes
* **Notes:** Successfully generated features; observed minor throughput fluctuations at intermediate stages.

### 2. Evaluation Process

* **Steps:** 12,982
* **Duration:** ~35 minutes
* **Stability:** 0 errors. Encountered non-fatal deprecation warnings from HuggingFace/PyTorch APIs.

---

## 📈 Performance Results

Evaluation conducted with a classification threshold of **0.5**.

| Metric | Score |
| --- | --- |
| **Precision** | 0.9574 |
| **Recall** | 0.9692 |
| **F1-Score** | **0.9632** |

> [!TIP]
> **Key Observation:** The high recall () demonstrates CodeBERT's high efficacy in identifying clone pairs when training and testing distributions are aligned.

---

## 📝 Future Work & Notes

* **Baseline:** This run serves as the primary baseline for cross-domain and domain-adaptation research.
* **Planned Comparisons:** Comparative analysis with **GraphCodeBERT-BCB** and **CodeT5-BCB** is ongoing.
* **Expansion:** Future iterations will explore multi-domain training data to mitigate performance degradation in cross-domain scenarios.

---
