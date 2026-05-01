# Python Clone Refactorer: Extract Method

This tool automates the **Extract Method** refactoring process for Python code clones. It identifies duplicated code blocks (clones) across a codebase, synthesizes a common method, and replaces the original clones with calls to that new method.

## ## Overview

The script uses **Tree-sitter** for robust Abstract Syntax Tree (AST) parsing, ensuring that code extraction respects Python's scope and indentation rules. It is specifically designed to work with clone detection outputs (like NiCad) that have been processed into JSON or JSONL formats.

### Key Features

* **AST-Aware Extraction:** Uses `python-tree-sitter` to identify function and class boundaries.
* **Signature Synthesis:** Automatically determines necessary parameters by analyzing free variables within the clone block.
* **Smart Insertion:** Places the newly extracted method at the appropriate class or module level.
* **Indentation Management:** Handles Python's whitespace-sensitive syntax during code injection.

---

## ## Requirements

Ensure you have the following dependencies installed in your environment (e.g., `ast-analysis`):

* **Python 3.11+**
* **tree_sitter**
* **python_treesitter_parser** (local module)
* **util_ast_python** (local module)

---

## ## Usage

Run the script by providing an input file containing clone data and the root directory of the source code.

### Basic Command

```bash
python extract_method_refactor_python.py \
  --input /home/user1-system11/research_dream/llm-clone/automate_extract_method_refactoring_py/data/sampled_pairs_refactored_true_20.jsonl \
  --source-root /home/user1-system11/research_dream/llm-clone/detect_clones/NiCad-py-block \
  --output-dir ./output/refactor_out

```

### Arguments

| Argument | Description |
| --- | --- |
| `--input` | **Required.** Path to the `.jsonl` or `.json` file containing clone class data. |
| `--source-root` | The base directory where the source files are located. |
| `--output-dir` | Directory where the refactored files and reports will be saved. |
| `--full-report` | (Optional) Include detailed metadata in the output JSON. |
| `--merged-report-name` | Custom name for the final aggregated results file. |

---

## ## Output Structure

The tool creates a structured output directory organized by `classid`:

```text
output/refactor_out/
├── summary.json                  # High-level summary of all refactors
├── all_refactor_results.json      # Detailed JSON report of all changes
└── [classid]/                    # Subfolder for each clone class
    └── systems/
        └── [project]/
            └── [module].py       # The refactored .py file

```
