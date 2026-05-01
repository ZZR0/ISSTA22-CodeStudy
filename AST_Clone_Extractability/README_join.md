# Clone Prediction + Refactorability Join Pipeline

This module joins:

* **LLM clone detection predictions**
* **Ground-truth clone pairs**
* **AST-based extractability analysis results**

and produces **pair-level CSV files** containing:

* clone prediction (0/1 or ground truth)
* function-level extractability
* pair-level type compatibility
* pair-level refactorability

The pipeline supports:

1. Ground-truth evaluation
2. Before domain adaptation
3. After domain adaptation

---

# 1. Overview

The join process connects:

```
Predictions / Ground Truth
        +
Extractability JSONL (AST analysis)
        ↓
Pair-level CSV with:
    - In/Out counts
    - Type compatibility
    - Extractability
    - Refactorability
```

---

# 2. Components

## A. `run_join_all.sh`

Batch runner for **model predictions** (before or after adaptation).

Processes all systems for all models and generates per-model CSV outputs.

---

## B. `run_all_ground_truth_refactorability.sh`

Batch runner for **ground-truth clone pairs**.

Processes:

```
input/ground_truth_test/<system>_test.txt
```

and joins them with extractability JSONL to produce:

```
output/ground_truth_test_refactorability/<system>_with_refactorability.csv
```

This allows evaluation of:

* True clone pairs
* Their extractability
* Their refactorability feasibility

---

## C. `join_pred_with_inout_refactorability.py`

Core Python join script that:

* Loads extractability JSONL
* Loads prediction or ground-truth pair file
* Computes pair-level type compatibility
* Computes pair-level refactorability
* Writes CSV
* Prints summary statistics

---

# 3. Directory Structure

```
AST_Clone_Extractability/
│
├── input/
│   │
│   ├── java-functions/
│   │   └── *-sim0.7/
│   │       └── step4_nicad_*_filtered_with_func_id.jsonl
│   │
│   ├── ground_truth_test/
│   │   └── <system>_test.txt
│   │       # Ground-truth clone pairs
│   │
│   ├── prediction_before_adapt/
│   │   ├── codebert/
│   │   │   └── predictions_prefixed -> <symlink>
│   │   ├── codegpt/
│   │   │   └── predictions_prefixed -> <symlink>
│   │   ├── codet5/
│   │   │   └── predictions_prefixed/
│   │   │       └── predictions_<system>_test.txt
│   │   └── graphcodebert/
│   │       └── predictions_prefixed -> <symlink>
│   │
│   └── prediction_after_adapt/
│       ├── codebert/
│       │   └── predictions/
│       ├── codegpt/
│       ├── codet5/
│       └── graphcodebert/
│
├── output/
│   │
│   ├── java-functions/
│   │   └── <system>_extractability.jsonl
│   │       # AST-based extractability analysis
│   │
│   ├── ground_truth_test_refactorability/
│   │   └── <system>_with_refactorability.csv
│   │
│   ├── pred_before_adapt_refactorability/
│   │   ├── codebert/
│   │   ├── codegpt/
│   │   ├── codet5/
│   │   ├── graphcodebert/
│   │   └── logs/
│   │
│   └── pred_after_adapt_refactorability/
│       ├── codebert/
│       ├── codegpt/
│       ├── codet5/
│       ├── graphcodebert/
│       └── logs/
│
├── run_join_all.sh
├── run_all_ground_truth_refactorability.sh
└── join_pred_with_inout_refactorability.py
```

---

# 4. Three Evaluation Pipelines

## 1️⃣ Ground Truth → Refactorability

* Input:

  ```
  input/ground_truth_test/
  ```
* Output:

  ```
  output/ground_truth_test_refactorability/
  ```

Purpose:
Evaluate how many true clone pairs are extractable and refactorable.

---

## 2️⃣ Before Adaptation → Refactorability

* Input:

  ```
  input/prediction_before_adapt/
  ```
* Output:

  ```
  output/pred_before_adapt_refactorability/
  ```

Purpose:
Measure baseline model performance + feasibility.

---

## 3️⃣ After Adaptation → Refactorability

* Input:

  ```
  input/prediction_after_adapt/
  ```
* Output:

  ```
  output/pred_after_adapt_refactorability/
  ```

Purpose:
Measure performance improvement after domain adaptation.

---

# 5. Running the Pipelines

## A. Run All Model Predictions (Batch)

```
bash run_join_all.sh
```

Generates:

```
output/pred_after_adapt_refactorability/<model>/pred_<system>_with_refactorability.csv
```

Logs stored in:

```
output/pred_after_adapt_refactorability/logs/
```

---

## B. Run Ground Truth Join (Batch)

```
bash run_all_ground_truth_refactorability.sh
```

Generates:

```
output/ground_truth_test_refactorability/<system>_with_refactorability.csv
```

---

## C. Single System Run

```
python3 join_pred_with_inout_refactorability.py \
  --jsonl output/java-functions/camel_extractability.jsonl \
  --pred input/prediction_after_adapt/codebert/predictions/predictions_camel_test.txt \
  --out  output/pred_after_adapt_refactorability/codebert/pred_camel_with_refactorability.csv
```

---

# 6. Pair-Level Logic

## InType Compatibility

```
pair_inType_ok ⇔ InType(left) == InType(right)
```

---

## OutType Compatibility

```
pair_outType_ok ⇔
    (Out(left) = ∅ AND Out(right) = ∅)
 OR (Out(left) ≠ ∅ AND Out(right) ≠ ∅ AND OutType(left) = OutType(right))
```

---

## Combined Type Compatibility

```
pair_type_ok ⇔ pair_inType_ok AND pair_outType_ok
```

---

# 7. Refactorability (Final Logic)

Final definition:

```
pair_refactorable =
    left_extractable
    AND right_extractable
    AND pair_type_ok
```

This ensures that a clone pair is considered refactorable only if:

* Both sides are individually extractable
* Input types match
* Output types are compatible

---

# 8. Output CSV Columns

Each CSV contains:

* pair_left_func
* pair_right_func
* clone_predict
* pair_left_in
* pair_right_in
* pair_left_out
* pair_right_out
* pair_inType_ok
* pair_outType_ok
* pair_type_ok
* pair_left_extractable
* pair_right_extractable
* pair_refactorable

---

# 9. Printed Statistics

After execution:

```
===== STATISTICS =====
Total parsed pairs
clone_predict == 1
pair_refactorable == True
pair_type_ok == True
```

Rates are printed automatically.

---