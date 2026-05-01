# [Technical Report] CodeBERT Cross-Domain Evaluation

**Project:** Clone Detection Generalization (BCB → Camel)

**Date:** 2026-02-07

**Model:** `microsoft/codebert-base`

**Task:** Binary Clone Detection (BigCloneBench & NiCad-Camel)

---

### 1. Research Objective

This evaluation quantifies the **domain gap** and the impact of **domain-adaptive fine-tuning** on CodeBERT. We compare how a model trained on a large-scale generic dataset (BigCloneBench) generalizes to a specific unseen project (Apache Camel).

| Experiment | Fine-tuning Source (Training) | Evaluation Target (Test) | Status |
| --- | --- | --- | --- |
| **Exp A** (Baseline) | BCB 10% (Balanced) | **Camel-only** | **Complete** ✅ |
| **Exp B** (Adaptive) | BCB 10% + Camel (Mixed) | **Camel-only** | **Pending** ⏳ |

---

### 2. Experimental Setup

#### 2.1 Training & Validation Data (Source Domain)

* **Dataset:** BigCloneBench (BCB) 10% subsample.
* **Files:** `train_10percent.txt`, `valid_10percent.txt`.
* **Characteristics:** Balanced label distribution to ensure stable convergence.

#### 2.2 Evaluation Data (Target Domain)

* **Test Set:** `test_camel.txt` (Project-specific: Apache Camel).
* **Source:** NiCad post-processed clone pairs.
* **Volume:** 2,034 clone pairs.
* **Reference:** Data mapping resolved via `../dataset/mix/data.jsonl`.

---

### 3. Methodology & Evaluation Protocol

The evaluation is conducted in **test-only mode** to ensure the weights remain frozen during cross-domain inference.

* **Inference Mode:** `--do_test` (weights loaded from `checkpoint-best-f1`)
* **Decision Threshold:** 0.5
* **Resolution Type:** `--test_type mix` (Mapping project-specific IDs to source code snippets)
* **Metrics:** F1-Score, Precision, Recall.

---

### 4. Results & Analysis

#### 4.1 Comparative Performance Table

| ID | Training Configuration | Test Set | F1 | Precision | Recall |
| --- | --- | --- | --- | --- | --- |
| **Exp A** | **BCB 10% Only** | Camel | **0.5455** | 0.6813 | 0.5978 |
| **Exp B** | **BCB 10% + Camel** | Camel | *TBD* | *TBD* | *TBD* |

#### 4.2 Interpretation of Baseline (Exp A)

* **Within-Domain (Reference):** Previous benchmarks show CodeBERT achieves **F1 ≈ 0.96** on internal BCB test sets.
* **Domain Shift Impact:** Performance dropped to **F1 = 0.5455** when applied to Camel.
* **Finding:** There is a significant performance degradation (~43% drop), confirming that "out-of-the-box" CodeBERT lacks robustness against project-specific coding styles and NiCad-based clone definitions.

---

### 5. Reproducibility: Execution Script

To replicate the Experiment A baseline, use the following bash configuration:

```bash
#!/bin/bash
# CodeBERT Cross-Domain Test: BCB-10% Model -> Camel Test Set
set -euo pipefail

OUTDIR="./saved_models_codebert_bcb10"
TEST_FILE="../dataset/test_camel.txt"
DUMMY_TRAIN="../dataset/train_mix.txt" # Required by run.py argument parser

accelerate launch run.py \
  --output_dir="$OUTDIR/" \
  --model_type=roberta \
  --config_name=microsoft/codebert-base \
  --model_name_or_path=microsoft/codebert-base \
  --tokenizer_name="roberta-base" \
  --do_test \
  --test_data_file="$TEST_FILE" \
  --train_data_file="$DUMMY_TRAIN" \
  --test_type mix \
  --block_size 400 \
  --eval_batch_size 32 \
  --seed 3

```

---

### 6. Roadmap & Next Steps

1. **Resume Exp B:** Restart the interrupted mixed fine-tuning (BCB + Camel).
2. **Delta Analysis:** Calculate  and Percentage Gain to quantify the benefit of including target-domain data during training.
3. **Error Analysis:** Inspect False Positives/Negatives in Camel to see if the model struggles with specific NiCad clone types (e.g., Type-3 literal changes).
4. **Statistical Significance:** Apply Wilcoxon signed-rank test once multiple runs are complete.

---

**Summary:** Current data confirms a **strong domain gap**. Experiment B is essential to determine if domain-adaptive fine-tuning can recover the lost performance for practical deployment on new software systems.