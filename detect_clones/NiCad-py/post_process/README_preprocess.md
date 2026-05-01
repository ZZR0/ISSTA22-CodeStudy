# Preprocessing Pipeline for Python Clone Data

> **Context:** Post-processing pipeline for NiCad clone detection output.
> **Target Applications:** Clone analysis, dataset construction, and LLM-based clone detection training/evaluation.

## 1. Overview

This directory contains a deterministic, language-aware pipeline designed to clean, filter, and analyze Python clone data produced by NiCad. The pipeline emphasizes semantic preservation while removing noise (comments, docstrings) and irrelevant artifacts (test code) to ensure high-quality input for downstream tasks.

**Key Features:**

- **Semantic Cleaning:** Safely removes Python comments and docstrings using the native `tokenize` module.
- **Test Code Elimination:** Filters out test-related code to prevent data contamination.
- **Statistical Profiling:** Measures code size distributions (LOC, Tokens) to identify outliers.
- **Heuristic-Based:** Fully explainable and reproducible preprocessing steps.

---

## 2. Input Data Specifications

- **Format:** JSON Lines (`.jsonl`)
- **Source:** NiCad clone detection output (e.g., `data/step1_nicad_azure_sim0.7.jsonl`)
- **Structure:** Each line represents a **clone group** or a single code snippet.
- **Fields per Source:**
- `file` / `file_path`: Origin path of the source code.
- `code` / `func`: The raw source code string.
- `start_line` / `end_line`: Metadata from NiCad.

---

## 3. Pipeline Methodology

### Step 1: Python Comment & Docstring Removal

**File:** `remove_python_comments.py`

**Function:** `remove_python_comments(code: str) -> str`

This step reduces code to its semantic core by stripping human-readable annotations.

| **What is Removed** ❌    | **What is Preserved** ✅                        |
| ------------------------- | ----------------------------------------------- |
| Line comments (`# ...`)   | `#` inside string literals                      |
| Inline comments           | URLs with fragments (e.g., `https://x.y/#frag`) |
| Module docstrings         | Triple-quoted strings used as data variables    |
| Class/Function docstrings | f-strings and raw strings (`r"..."`)            |

**Implementation Details:**

- Utilizes Python’s built-in `tokenize` module for accurate parsing (not regex).
- Detects docstrings as `STRING` tokens appearing as the first statement in a block.
- **Post-cleaning:** Removes tokenizer artifacts, trailing whitespace, and empty lines.

### Step 2: Validation & Robustness

**File:** `test_remove_python_comments.py`

Ensures correctness across edge cases before processing the full dataset.

- **Coverage:** Full-line/inline comments, complex URLs, shebangs, encoding declarations, and import-like text within strings.
- **Output:** Prints `BEFORE` / `AFTER` code and `PASS/FAIL` status for every test case.

### Step 3: Filtering Out Test Code

**File:** `2_filter_out_data.py`

Excludes entire clone groups if **any** source within the group is identified as test code. This prevents training data contamination and inflation of clone similarity metrics.

**Detection Heuristics:**

1. **File Path:** Contains `/test/`, `/tests/`, or matches patterns like `test_*.py`, `*_test.py`.
2. **Function Name:** Defined as `def test_*`.
3. **Assertions:** Presence of `assert` statements (token-level detection).
4. **Imports:** Imports of testing frameworks (`pytest`, `unittest`).

_All exclusions are logged to stderr with specific reasons (e.g., `why=FILENAME,ASSERT`)._

### Step 4: Statistical Analysis (LOC & Tokens)

Metrics are computed **after** Step 1 (cleaning) to ensure they reflect semantic code size.

- **Metrics:**
- **LOC:** Non-empty lines of code.
- **Tokens:** Python lexical tokens (via `tokenize`, distinct from LLM subword tokens).

- **Reporting:**
- Distributions for **All** (Input), **Remaining** (Output), and **Excluded** functions.
- Statistics: Count, Min/Max, Mean, Median.
- Bucketed histograms and Top-K exact values for outlier analysis.

---

## 4. Usage

Run the main filtering and analysis script:

```bash
python3 2_filter_out_data.py \
  --input  ./data/step1_nicad_azure_sim0.7.jsonl \
  --output ./data/step2_nicad_azure_sim0.7.jsonl

```

**Optional Arguments:**

- `--quiet-exclusions`: Suppress detailed logs of excluded files.
- `--loc-bucket-size`: Bin size for LOC histogram (default: 5).
- `--tok-bucket-size`: Bin size for Token histogram (default: 50).
- `--loc-topk`: Number of top outliers to display for LOC.

---

## 5. Design Principles

1. **Language-Awareness:** relies on Python `tokenize` rather than fragile regex heuristics.
2. **Conservative Fallback:** If tokenization fails (e.g., syntax errors in snippets), the pipeline retains the original text to avoid data loss.
3. **Explainability:** Every excluded line is logged with a specific heuristic reason ("White-box" filtering).
4. **Reproducibility:** Deterministic processing ensures consistent datasets across runs.

## 6. Intended Use & Extensions

**Primary Use Cases:**

- Data curation for Large Language Model (LLM) training.
- Empirical studies on code clone similarity.
- Analysis of code size and token distribution outliers.

**Potential Extensions:**

- Emitting the cleaned code (`code_clean`) directly into the output JSONL.
- Implementing Max-LOC / Max-Token filters for curriculum learning.
- Visualizing the correlation between Clone Similarity and Code Length.
