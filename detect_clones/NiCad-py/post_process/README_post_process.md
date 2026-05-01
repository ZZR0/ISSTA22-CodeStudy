# NiCad Clone Data Processing Pipeline

This repository provides a comprehensive suite of Python utilities to transform, filter, and normalize **NiCad** clone detection results into high-quality datasets for machine learning. The pipeline is designed for **data traceability**, **code normalization**, and **balanced sampling**.

---

## 🚀 Pipeline Steps

### 1. Ingestion & Conversion

**`1_nicad_xml_to_jsonl.py`**
Converts NiCad's `-classes-withsource.xml` files into **JSON Lines (JSONL)**.

* **Key Feature:** Uses a regex-based parser to handle large XML files efficiently.
* **Cleaning:** Strips illegal ASCII control characters and offers optional HTML unescaping.

```bash
python 1_nicad_xml_to_jsonl.py \
    --xml ./input/azure-sdk-for-python_functions-clones-0.30-classes-withsource.xml \
    --out ./data/step1_nicad_azure_sim0.7.jsonl \
    --mode class \
    > ./data/step1_nicad_azure_sim0.7.log 2>&1

```

### 2. Data Refinement & Filtering

These stages remove noise (tests and boilerplate) and normalize the source code structure.

* **Filter Test Code (`2a_filter_out_test_fun.py`):** Removes unit tests and test-related functions using file-path heuristics and token-level `assert` detection.
```bash
python 2a_filter_out_test_fun.py \
    --input ./data/step1_nicad_azure_sim0.7.jsonl \
    --output ./data/step2a_nicad_azure_sim0.7.jsonl 

```


* **Arity Filtering (`2b_filter_out_group.py`):** Excludes "micro-clones" or overly large clusters by keeping clone groups within a specific size range (e.g., 1 to 20 fragments).
```bash
python 2b_filter_out_group.py \
    --input ./data/step2a_nicad_azure_sim0.7.jsonl \
    --output ./data/step2b_nicad_azure_sim0.7.jsonl \
    --min-size 1 --max-size 20 \
    > ./data/step2b_nicad_azure_sim0.7.log 2>&1

```


* **Remove Comments (`2c_remove_comment.py`):** Strips Python comments and docstrings. It uses a "smart" approach: attempting AST-based parsing first and falling back to a token-based method for partial code snippets.
```bash
python 2c_remove_comment.py \
    --input ./data/step2b_nicad_azure_sim0.7.jsonl \
    --output ./data/step2c_nicad_azure_sim0.7.jsonl \
    > ./data/step2c_nicad_azure_sim0.7.log 2>&1

```


* **Filter Constructors (`2d_filter_out_init_fun.py`):** Targeted removal of Python `__init__` methods to prevent boilerplate assignments from biasing the model.
```bash
python 2d_filter_out_init_fun.py \
    --input ./data/step2c_nicad_azure_sim0.7.jsonl \
    --output ./data/step2d_nicad_azure_sim0.7.jsonl 

```



### 3. Diagnostics & Indexing

Analyze the distribution of the cleaned data and ensure every function has a permanent identifier.

* **Visualize Distribution (`display_clone_group_sizes.py`):**
Generates an ASCII histogram of clone group sizes (Arity) to verify data density.
```bash
python display_clone_group_sizes.py \
    --input ./data/step2d_nicad_azure_sim0.7.jsonl 2>&1 \
    | tee ./data/display_clone_group_sizes1.log

```


* **Generate Stable IDs (`3_gen_init_train_sample.py`):**
Assigns a unique `func_id` (format: `{classid}_{index}`) to every function. This ensures reproducibility even if the data is shuffled later.
```bash
python 3_gen_init_train_sample.py \
    --input ./data/step2d_nicad_azure_sim0.7.jsonl \
    --output ./data/step3_nicad_azure_sim0.7.jsonl 

```



### 4. Training Set Generation

Construct the final pairwise dataset for binary classification.

* **Negative Samples (Label 0) (`4_gen_neg_clone_sample.py`):**
Randomly pairs functions from **different** clone classes.
```bash
python 4_gen_neg_clone_sample.py \
  --input ./data/step3_nicad_azure_sim0.7.jsonl \
  --out_txt ./data/step4_nicad_azure_sim0.7_neg_pairs.txt \
  --out_jsonl ./data/step4_nicad_azure_sim0.7_neg_pairs.jsonl \
  --out_html ./data/step4_nicad_azure_sim0.7_neg_pairs.html \
  --out_md ./data/step4_nicad_azure_sim0.7_neg_pairs.md \
  --seed 42 --verify --cleanup

```


* **Positive Samples (Label 1) (`5_gen_pos_clone_sample.py`):**
Generates combinations of functions within the **same** clone classes.
```bash
python 5_gen_pos_clone_sample.py \
  --input ./data/step3_nicad_azure_sim0.7.jsonl \
  --out_txt ./data/step5_nicad_azure_sim0.7_pos_pairs.txt \
  --out_jsonl ./data/step5_nicad_azure_sim0.7_pos_pairs.jsonl \
  --out_html ./data/step5_display_nicad_azure_sim0.7_pos_pairs.html \
  --out_md ./data/step5_display_nicad_azure_sim0.7_pos_pairs.md \
  --seed 42 --verify --cleanup

```



---

## 🛠️ Requirements & Technical Notes

* **Environment:** Python 3.x
* **Dependencies:** None (Uses the Python Standard Library).
* **Reproducibility:** Use the `--seed` flag in sampling scripts to ensure consistent dataset generation.
* **Verification:** Use the `--verify` flag to cross-check pair files against source IDs.