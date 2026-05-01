# Clone Prediction & Refactorability Pipeline

This pipeline integrates LLM clone detection predictions with static analysis data (AST/Dataflow) to evaluate the **refactorability** of identified code clones.

## 1. Core Logic

The pipeline determines if a clone pair can be refactored based on function-level extractability and type compatibility.

### Definitions

* **Type Compatibility (`pair_type_ok`):** * `pair_inType_ok`: Input signatures must match.
* `pair_outType_ok`: Output signatures must match (or both must have no outputs).
* `pair_type_ok = pair_inType_ok ∧ pair_outType_ok`


* **Refactorability (`pair_refactorable`):**
* `pair_refactorable = left_extractable ∧ right_extractable ∧ pair_type_ok`

---

## 2. Pipeline Architecture

extract_clone_py/
├── main_python.py                      # Augments JSONL with AST/IO data
├── join_pred_with_inout_refactorability.py # Core join & analysis engine
├── run_*.sh                            # Batch execution scripts
├── templates/                          # HTML report layouts
├── data/
│   ├── py_funcs_step1_2a_2b_2d_3/      # Raw NiCad inputs
│   ├── augmented_post_nicad_func/      # Generated augmented JSONL + HTML
│   ├── ground_truth_test/              # Test sets for ground truth eval
│   ├── prediction_before_adapt/        # Symlinks to baseline predictions
│   └── prediction_after_adapt/         # Symlinks to adapted model predictions
└── output/
    ├── ground_truth_test_refactorability/
    ├── pred_before_adapt_refactorability/
    └── pred_after_adapt_refactorability/


### Input Data

1. **Clone Predictions/Ground Truth:** Model output files or benchmark sets.
2. **Augmented JSONL:** Contains AST, I/O metadata, types, and `extractable` flags.

### Processing Steps

| Component | Purpose |
| --- | --- |
| `main_python.py` | Generates the augmented JSONL from NiCad input; creates HTML visualizations. |
| `join_pred_with_inout_refactorability.py` | Core engine that joins data, computes compatibility/refactorability, and exports CSVs. |
| **Batch Runners** | Automates execution across `ground_truth`, `before_adapt`, and `after_adapt` datasets. |

---

## 3. Evaluation Pipelines

The system supports three distinct evaluation modes to measure performance improvements:

1. **Ground Truth Analysis:** Measures the theoretical upper bound of refactorable clones in your test set.
2. **Before Adaptation:** Establishes the baseline performance of pre-trained LLMs.
3. **After Adaptation:** Measures performance gains achieved through domain-specific fine-tuning.

---

## 4. Setup & Execution Guide

### Step 1: Data Preparation

Ensure your environment is set up by creating the necessary symlinks to your model predictions.

* **Before/After Adaptation:** Use your existing batch scripts to link `saved_models` directories to the `data/prediction_*/` directories.
* **Ground Truth:** Run `./copy_all_ground_truth.sh` to populate the test directory.

### Step 2: Augmentation

Generate the required static analysis metrics for your systems:

```bash
# Run for all systems
./run_main_python_all.sh

```

### Step 3: Execution

Run the join scripts to generate CSV reports and summary statistics for each evaluation pipeline:

```bash
./run_all_ground_truth_refactorability.sh
./run_join_all_before_adapt.sh
./run_join_all_after_adapt.sh

```

---

## 5. Output Format

### CSV Columns

Each generated CSV contains the following features for analysis:

* **Identifiers:** `pair_left_func`, `pair_right_func`
* **Prediction:** `clone_predict`
* **I/O Metadata:** `pair_left_in/out`, `pair_right_in/out`
* **Compatibility:** `pair_inType_ok`, `pair_outType_ok`, `pair_type_ok`
* **Feasibility:** `pair_left_extractable`, `pair_right_extractable`, `pair_refactorable`

### Statistics

The pipeline automatically outputs a summary after each run:

* Total parsed pairs
* Clone prediction count
* Refactorable pair count
* Type compatibility rate
