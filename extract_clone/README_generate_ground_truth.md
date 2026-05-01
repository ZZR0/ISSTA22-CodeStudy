# Ground Truth Dataset Generation Pipeline

This repository contains the automated bash pipeline for generating a fully structured ground truth dataset. It consumes the mathematically proven, extractable code clones from the previous AST analysis step and orchestrates the generation of positive samples, negative samples, and deduplicated function corpora across multiple repositories.

This pipeline automates **Steps 5 through 8** of the dataset creation process. It processes repositories sequentially, handling failures gracefully to ensure massive batch runs complete without interruption.

## Pipeline Workflow

For every repository `.jsonl` file found in the input directory, the script executes the following sequence:

1. **Step 5: Negative Sample Generation (`5_gen_neg_clone_sample.py`)**
* Intentionally pairs functions from *different* clone classes to create negative examples (Label `0`).
* Outputs textual pairs, JSONL records, and rich HTML/Markdown inspection reports.


2. **Step 6: Positive Sample Generation (`6_gen_pos_clone_sample.py`)**
* Pairs functions from the *same* clone class to create true clone examples (Label `1`).
* Applies memory-cap protections and includes verification checks.


3. **Step 7: Corpus Generation (`7_gen_func_index_corpus.py`)**
* Extracts every unique function involved in the dataset, assigning it a unique ID.
* Outputs a deduplicated `data.jsonl` file containing the raw source code and metadata.


4. **Step 8: Pair Combination (`8_combine_neg_pos_pairs.py`)**
* Merges the positive and negative `.txt` files into a single, balanced `test.txt` file.
* The IDs in this file perfectly map to the functions indexed in Step 7.



## Output Directory Structure

To ensure compatibility with standard ML data loaders (like those used in CodeXGLUE), the script enforces a strict nested folder hierarchy for each repository:

```text
./output/ground_truth/
└── <repo_name>_sim0.7/               # TOP_DIR (Contains reports & raw generation files)
    ├── step5_nicad_<repo>_sim0.7_neg_pairs.txt
    ├── step5_display_nicad_<repo>_sim0.7_neg_pairs.html
    ├── step6_nicad_<repo>_sim0.7_pos_pairs.txt
    ├── step6_display_nicad_<repo>_sim0.7_pos_pairs.html
    └── <repo_name>/                  # OUT_DIR (The clean ML dataset folder)
        ├── data.jsonl                # Deduplicated function corpus
        └── test.txt                  # Combined positive and negative pairs (id1 \t id2 \t label)

```

## Key Features

* **Fault Isolation:** Each repository is processed inside an isolated subshell (`(...)` with `set -e`). If one repository fails (e.g., due to an empty file or memory limit), the script logs the error, safely aborts that specific repo, and immediately continues to the next.
* **Deterministic Generation:** Uses a hardcoded `--seed 42` parameter passed to the Python scripts, ensuring the exact same positive/negative pairs are generated across different runs.
* **Centralized Logging:** Captures all standard output and standard errors into timestamped log files (e.g., `full_pipeline_20260227_165400.log`).
* **Failure Tracking:** Maintains a dedicated text file (`failed_systems_YYYYMMDD_HHMMSS.txt`) explicitly listing any repositories that returned a non-zero exit code for easy debugging.

## Usage

### 1. Configuration

Open the bash script and ensure the absolute paths at the top match your environment:

* `INPUT_DIR`: The directory containing the output `.jsonl` files from your AST Extractability analysis.
* `OUTPUT_ROOT`: The destination folder for your ML datasets,`/home/user1-system11/research_dream/llm-clone/extract_clone/output/extractable_nicad_block_clones`.

### 2. Execution

Run the bash script directly from your terminal. Because the script uses `tee`, you can monitor the progress live while it safely writes to the logs.

```bash
chmod +x run_ground_truth_prepare_step5_8.sh
./run_ground_truth_prepare_step5_8.sh

```
