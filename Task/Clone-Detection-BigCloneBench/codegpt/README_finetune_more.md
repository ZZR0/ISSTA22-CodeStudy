# Fine-tuning CodeGPT with BCB (10%) + Other-Domain Mix

_(Cross-Domain Evaluation on Other Domain Only)_

**Date:** 2026-02-06

**Model:** CodeGPT (Java)

**Goal:**
Improve **cross-project / cross-domain generalization** by:

- **Training & validation** on a mixed dataset:
    - **BCB (BigCloneBench):** 10% subset (label-balanced)
    - **Other Domain (e.g., Camel):** NiCad post-processed clone pairs

- **Testing exclusively on the other domain** (e.g., Camel) to evaluate
  how well a model trained primarily on BCB generalizes to unseen projects.

---

## 1. Experimental Setting (Summary)

| Split | Dataset Composition                 |
| ----- | ----------------------------------- |
| Train | BCB (10%) + Other domain            |
| Valid | BCB (10%) + Other domain            |
| Test  | **Other domain only** (e.g., Camel) |

> **Important:**
> No BCB test pairs are used during evaluation.
> This avoids test leakage and enables strict cross-domain evaluation.

---

## 2. Directory Structure

```text
../dataset/
├── train_10percent.txt           # BCB 10% train
├── valid_10percent.txt           # BCB 10% valid
├── train_mix.txt                 # Mixed train (BCB + other domain)
├── valid_mix.txt                 # Mixed valid (BCB + other domain)
├── test_<otherdomain>.txt        # Other-domain-only test (e.g., test_camel.txt)
├── data.jsonl                    # Original BCB mapping
└── mix/
    ├── data.jsonl                # Combined mapping (BCB + other domain)
    ├── train_bcb.pref.txt
    ├── valid_bcb.pref.txt
    ├── train_other.pref.txt
    └── valid_other.pref.txt
```

---

## 3. Why Prefixing and Mixing Are Required

`run.py` resolves each clone pair ID via **a single mapping file** (`data.jsonl`).

```python
if url1 not in url_to_code or url2 not in url_to_code:
    continue
```

Therefore:

- All IDs appearing in `train / valid / test` **must exist** in the mapping.
- ID collisions across datasets must be avoided.

### Solution

- Prefix IDs by dataset:
    - `bcb_<id>` for BigCloneBench
    - `<otherdomain>_<id>` for the other domain (e.g., `camel_259_340`)

- Build a unified mapping: `../dataset/mix/data.jsonl`

---

## 4. Dataset Preparation Pipeline

### Step 1 — Sample BCB 10% (Balanced)

```bash
python 1_sample_data.py --seed 3 --mode balanced
```

**Result:**

- Train: 90,102 pairs (50/50 labels)
- Valid: 41,540 pairs (50/50 labels)

---

### Step 2 — Mix BCB (10%) + Other Domain

**Script:** `2_mix_data.py`
**Key option:** `--otherdomain_name`

```bash
python 2_mix_data.py \
  --otherdomain_name camel \
  --train_data_file_bcb ../dataset/train_10percent.txt \
  --valid_data_file_bcb ../dataset/valid_10percent.txt \
  --train_data_file_more ../../../detect_clones/NiCad/post_process/data/java/camel/train.txt \
  --valid_data_file_more ../../../detect_clones/NiCad/post_process/data/java/camel/valid.txt \
  --test_data_file_otherdomain ../../../detect_clones/NiCad/post_process/data/java/camel/test.txt \
  --bcb_jsonl ../dataset/data.jsonl \
  --more_jsonl ../../../detect_clones/NiCad/post_process/data/java/camel/data.jsonl \
  --seed 3
```

**Outputs:**

- `train_mix.txt`
- `valid_mix.txt`
- `mix/data.jsonl`
- `test_camel.txt` (**Camel-only test**)

---

## 5. Data Integrity Verification (Critical)

Before training, **all pairs are verified** to ensure their IDs exist in the mapping.

```bash
python 3_verify_data.py --strict --otherdomain_name camel
```

### Verification Results

- `train_mix.txt`: 94,170 pairs → **0 missing**
- `valid_mix.txt`: 43,574 pairs → **0 missing**
- `test_camel.txt`: 2,034 pairs → **0 missing**

```
✅ PASS: All pairs are resolvable in the mapping JSONL.
```

This guarantees that `run.py` will **not silently drop data**.

---

## 6. Fine-tuning CodeGPT

### Key Flags

- `--test_type mix`
  → loads `../dataset/mix/data.jsonl`
- `--test_data_file ../dataset/test_camel.txt`
  → **other-domain-only evaluation**

### Training Command

```bash
accelerate launch run.py \
  --output_dir=./saved_models_mix/ \
  --subsample_ratio 1.0 \
  --model_type=gpt2 \
  --config_name=microsoft/CodeGPT-small-java-adaptedGPT2 \
  --model_name_or_path=microsoft/CodeGPT-small-java-adaptedGPT2 \
  --tokenizer_name=microsoft/CodeGPT-small-java-adaptedGPT2 \
  --test_type mix \
  --do_train \
  --do_test \
  --train_data_file=../dataset/train_mix.txt \
  --eval_data_file=../dataset/valid_mix.txt \
  --test_data_file=../dataset/test_camel.txt \
  --block_size 400 \
  --train_batch_size 8 \
  --gradient_accumulation_steps 4 \
  --epoch 2 \
  --learning_rate 5e-5 \
  --seed 3
```

---

## 7. Interpretation of Results

- Training benefits from **diverse supervision** (BCB + other domain)
- Testing strictly measures **generalization to a new domain**
- The setup avoids:
    - ID mismatch errors
    - Silent test data drops
    - Test leakage from training domain

This configuration is suitable for:

- Cross-project clone detection studies
- Domain adaptation experiments
- Reproducible ML/SE research

---

## 8. Next Step: CodeBERT

The **same dataset artifacts** can be reused for CodeBERT:

- `train_mix.txt`
- `valid_mix.txt`
- `test_<otherdomain>.txt`
- `mix/data.jsonl`

Only the model backend changes.

---

### ✔ Status

- Dataset preparation: **DONE**
- Data verification: **PASS**
- Ready for training & evaluation
