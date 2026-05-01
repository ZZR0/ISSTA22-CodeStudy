# Clone Refactoring Inspection Pipeline

## Overview

The `integrate_refactoring_pipeline.py` script is an end-to-end evaluation tool designed for large-scale clone detection research. It automates the extraction, filtering, and visualization of code clone pairs specifically optimized for **"Extract Method"** refactoring in IDEs like Eclipse or VS Code.

Instead of running separate evaluation and sampling scripts, this unified pipeline:

1. **Ingests** raw AST metadata (`.jsonl`), ground truth annotations, and CodeBERT predictions (both pre- and post-adaptation) across all repositories.
2. **Evaluates** each clone pair for `Refactorable` structural parity (matching input/output counts and types).
3. **Filters** the pool to strictly include intra-file clones (`same_file == 1`).
4. **Samples** a precise 100-pair subset for manual inspection based on distinct success/failure cases.
5. **Generates** an interactive, syntax-highlighted HTML report mapping the side-by-side clone logic.

## The 100-Sample Inspection Cases

To systematically analyze the impact of the model adaptation, the pipeline enforces a strict 100-sample extraction distributed across three specific scenarios.

**All extracted pairs must reside in the same Java file (`same_file == 1`) to ensure they can be refactored without massive architectural redesign.**

* **Case 1 (40 Samples): Baseline Stability**
* *Condition:* Both baseline and adapted models correctly identified the pair, and the pair is perfectly refactorable.
* *Goal:* Prove that the adaptation did not degrade performance on "easy" or standard clones.


* **Case 2 (30 Samples): False Negative Recovery ("The Success Story")**
* *Condition:* The baseline model missed the clone, but the adapted model successfully found it, AND it is perfectly refactorable.
* *Goal:* Demonstrate that the adaptation uncovers complex, highly-valuable refactoring opportunities that standard models ignore.


* **Case 3 (30 Samples): False Positive Rejection ("Noise Reduction")**
* *Condition:* The baseline model hallucinated a clone, but the adapted model correctly rejected it as a non-clone.
* *Goal:* Prove the adaptation reduces developer fatigue by filtering out non-extractable "garbage" clones.



## Pipeline Outputs

The script generates three distinct artifacts in the designated output folder:

1. **`all_evaluated_pairs.jsonl`**: The master dataset. Contains every processed clone pair across all repositories, augmented with full AST metadata, evaluation flags (`same_file`, `Refactorable`), and transition states.
2. **`sampled_100_pairs.jsonl`**: The strict subset containing exactly 100 pairs adhering to the 40/30/30 distribution rules.
3. **`manual_inspection_report.html`**: A standalone HTML file rendering the 100 sampled pairs side-by-side. It features syntax highlighting for the method signature, context lines, and dynamically highlights in-clone variables (matching `main.py` aesthetics).

## Usage & Execution

Execute the pipeline by providing the paths to your respective metadata directories and defining the desired output paths.

To run the pipeline silently and redirect all terminal output (including the sampling breakdown) to a permanent log file:

```bash
python3 integrate_refactoring_pipeline.py \
  --jsonl_dir "/home/user1-system11/research_dream/llm-clone/extract_clone/output/extractable_nicad_block_clones" \
  --gt_dir "/home/user1-system11/research_dream/llm-clone/Task/Clone-Detection-BigCloneBench/dataset/nicad_block_java" \
  --pred_dir "/home/user1-system11/research_dream/llm-clone/Task/Clone-Detection-BigCloneBench/codebert/saved_models_bcb/predictions_bcb_java_block_function" \
  --pred_adapt_dir "/home/user1-system11/research_dream/llm-clone/Task/Clone-Detection-BigCloneBench/codebert/saved_models_combined/predictions_bcb_java_block_function" \
  --out_all_jsonl "/home/user1-system11/research_dream/llm-clone/extract_clone/output/sampling_clones/all_evaluated_pairs.jsonl" \
  --out_sampled_jsonl "/home/user1-system11/research_dream/llm-clone/extract_clone/output/sampling_clones/sampled_100_pairs.jsonl" \
  --out_html "/home/user1-system11/research_dream/llm-clone/extract_clone/output/sampling_clones/manual_inspection_report.html" \
  > "/home/user1-system11/research_dream/llm-clone/extract_clone/output/sampling_clones/pipeline_run.log" 2>&1

```

Once execution is complete, you can review the sampling breakdown by executing:
`cat /home/user1-system11/research_dream/llm-clone/extract_clone/output/sampling_clones/pipeline_run.log`