# Cross-Domain Training and Evaluation of CodeGPT (CodeGPT-BCB)

This document summarizes a three-step experimental pipeline evaluating **CodeGPT-small-java-adaptedGPT2** for code clone detection across domains.

---

## Overall Performance Comparison

| Scenario | Training Data | Test Data | F1 Score |
| --- | --- | --- | --- |
| **Step 1** | BCB (10%) | BCB | 0.9495 |
| **Step 2** | BCB (10%) | Azure | 0.4261 |
| **Step 3** | BCB + Azure | Azure | 0.8618 |

---

## Step 1: Training on BigCloneBench (BCB – 10%)

### Dataset

* **BigCloneBench (BCB)**
* Used 10% of: `train.txt`, `valid.txt`, and `test.txt`.
* Followed the same experimental setup as the original authors.

### Key Modifications

#### 1. Safety Patches to `run.py`

* Disabled forced override of `--save_steps`.
* Changed initialization:
> `best_f1 = 0`  `best_f1 = -1`


* Ensures the first checkpoint is saved even if F1 starts at 0.

#### 2. Hardware-Aware Configuration

* Auto-detect number of GPUs.
* Target global batch size = 64.
* Per-GPU batch size = 8.
* **Gradient accumulation computed dynamically:**



#### 3. Training Configuration

* **Model:** `microsoft/CodeGPT-small-java-adaptedGPT2`
* **Block size:** 400
* **Epochs:** 2
* **Learning rate:** 
* **Save Interval:** Every 500 steps
* **Evaluation:** Enabled during training

### Results (In-Domain: BCB Test Set)

| Metric | Value |
| --- | --- |
| **F1** | 0.9495 |
| **Precision** | 0.9313 |
| **Recall** | 0.9702 |
| **Threshold** | 0.5 |

**Observation:** The model achieves very strong in-domain performance on BCB.

---

## Step 2: Cross-Domain Testing (Azure – Python)

### Setup

* **Model:** CodeGPT-BCB (trained only on BCB).
* **Tested on:** Azure dataset.
* **Language:** Written in Python.
* **Mode:** No additional training (test-only mode).

### Configuration Highlights

* `--do_test` only
* 2 GPUs
* Batch size: 16
* Block size: 400

### Results (Cross-Domain: Azure)

| Metric | Value |
| --- | --- |
| **F1** | 0.4261 |
| **Precision** | 0.6578 |
| **Recall** | 0.5364 |
| **Threshold** | 0.5 |

**Observation:**

* **Massive performance drop:** F1: 
* Indicates poor cross-domain generalization.
* Model trained on Java (BCB) struggles significantly on Python (Azure).
* Domain and language shift significantly affect performance.

---

## Step 3: Fine-Tuning on Mixed Dataset (BCB + Azure)

**Goal:** Improve cross-domain performance by exposing the model to Azure data during training.

### Training Data

* **Mixed training set:** BCB (10%) + Azure domain.
* **Validation:** Mixed.
* **Test:** Azure-only.

### Configuration

* Same base model and learning rate.
* **Epochs:** 2.
* **Global batch size:** 64 (via gradient accumulation).
* **Flags:** `--test_type mix_azure`, `--subsample_ratio 1.0`.

### Results (Mixed Training → Azure Test)

| Metric | Value |
| --- | --- |
| **F1** | 0.8618 |
| **Precision** | 0.8681 |
| **Recall** | 0.8612 |
| **Threshold** | 0.5 |

---

## Overall Performance Comparison

| Scenario | Training Data | Test Data | F1 Score |
| --- | --- | --- | --- |
| **Step 1** | BCB (10%) | BCB | 0.9495 |
| **Step 2** | BCB (10%) | Azure | 0.4261 |
| **Step 3** | BCB + Azure | Azure | 0.8618 |

---

## Key Insights

1. **Strong In-Domain Performance:** CodeGPT performs extremely well when trained and tested within the same domain.
2. **Severe Domain Shift Problem:** Training on Java (BCB) does not generalize well to different projects (Azure) or different languages (Python). Cross-domain F1 drops by more than 50%.
3. **Mixed-Domain Fine-Tuning Works:** Including Azure data during training restores performance significantly, improving F1 from . This demonstrates strong domain adaptation capability.

---

## Conclusion

* CodeGPT is highly effective in-domain but suffers from significant transfer issues.
* Fine-tuning with mixed-domain data dramatically improves cross-domain generalization.
* **Domain-aware training** is critical for robust clone detection across diverse projects and languages.

---

Would you like me to help you draft a specific README for your GitHub repository based on these results?