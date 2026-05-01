# AST-Based Python Code Clone Extraction & Analysis

This project provides an automated pipeline for analyzing Python code clones. Using the `tree-sitter` parsing library, the pipeline extracts Abstract Syntax Trees (AST) from target Python source files, performs detailed data-flow analysis across specific code regions (Pre, Within, and Post clone boundaries), detects code extraction hazards, and outputs an augmented dataset alongside an interactive HTML visualization.

## Directory Structure

```text
.
├── data/                               # Generated datasets and target source code
│   ├── augmented_sample_python.jsonl   # Output: Augmented JSONL with data-flow & hazard metadata
│   ├── clone_visualization.html        # Output: HTML visual report of the clones
│   ├── sample_python.jsonl             # Input: Raw clone pair boundaries
│   └── systems/                        # Target Python source code directory
│       └── security_helper.py          # Sample Python file containing clone methods
├── templates/                          # HTML templates for the visualization report
│   ├── class_start.html
│   ├── footer.html
│   ├── header.html
│   ├── instance_error.html
│   └── instance_meta.html
├── main_python.py                      # Main orchestrator script
├── python_treesitter_parser.py         # Lightweight wrapper for the tree-sitter Python grammar
├── run.sh                              # Shell script to execute the pipeline
├── setup_test_data.py                  # Utility to generate mock source files and input JSONL
├── test_ast_python.py                  # Standalone test script for AST traversal functions
└── util_ast_python.py                  # Core AST extraction and Read/Write data-flow logic

```

## Prerequisites

Ensure your environment has the correct version of the Tree-sitter Python bindings. Because the pipeline relies on the `0.25.x` C-ABI, you must use version `0.25.0`.

```bash
pip install tree-sitter tree-sitter-python==0.25.0

```

## Pipeline Execution

### 1. Generate Test Data

If you are running this for the first time or need to reset the sample data, generate the continuous, functional mock code and the corresponding input JSONL:

```bash
python setup_test_data.py

```

This will populate the `data/` folder with `sample_python.jsonl` and `systems/security_helper.py`.

### 2. Run the Analysis Orchestrator

Execute the main script to parse the source code, extract data-flow contexts, filter out hazardous extractions, and generate the output artifacts:

```bash
python main_python.py \
    --jsonl ./data/sample_python.jsonl \
    --base-dir ./data \
    --output ./data/clone_visualization.html \
    --out-jsonl ./data/augmented_sample_python.jsonl

```

*(Note: You can also use `./run.sh` if you have configured it to run these steps sequentially.)*

## Core Architecture

### 1. AST Traversal & Scope Tracking (`util_ast_python.py`)

This module handles all Tree-sitter node queries. It bridges the gap between Python's dynamic typing and rigid block-scoped languages by simulating scope boundaries (classes, functions, lambdas). It maps exactly what variables are defined and mutated using assignments (`=`), augmented assignments (`+=`), and loop bindings (`for x in...`).

### 2. Data-Flow Analysis

The pipeline divides the enclosing function into three distinct execution regions relative to the clone code block:

* **PRE Region**: Code executed before the clone. Tracks initialized variables and parameters.
* **WITHIN Region**: The clone itself. Tracks variables Read (VR) and Written (VW) to determine inputs (`In`) and outputs (`Out`) required to extract the clone into a standalone function.
* **POST Region**: Code executed after the clone. Tracks variables that rely on the clone's output.

### 3. Hazard Detection & Filtering (`main_python.py`)

Clones are evaluated for extraction safety. A clone is dropped from the dataset if:

* **Full-Function Clone**: The clone spans the entire body of the enclosing method.
* **Data-Flow Hazard**: The clone writes to multiple local variables that are subsequently used in the POST region (Python methods standardly return a single execution flow, making multiple state mutations complex to cleanly extract).
* **Control-Flow Hazard**: The clone contains unbalanced jumps (`return`, `break`, `continue`) that disrupt the enclosing method's execution path.
* **Type Mismatches**: Sub-instances within a clone class have diverging inferred signatures (e.g., one requires a `dict` input, while the other requires a `list`).

## Output Artifacts

1. **`augmented_sample_python.jsonl`**: A rich dataset where each clone object includes inferred method signatures, `In(i)` / `Out(i)` sets, type maps, and hazard boolean flags.
2. **`clone_visualization.html`**: A syntax-highlighted HTML report mapping the clones line-by-line, visually marking Read/Write variables, inferred parameters, and regional bounds.

### 4. Data-Flow Extraction Sets (In and Out)

To safely extract a clone block $i$ into a standalone function, the pipeline mathematically calculates the exact input parameters and return variables required. It does this by analyzing variable read ($V_r$) and write ($V_w$) operations across the three execution regions.

**The Input Set:** $In(i)$
The input set represents all variables that must be passed as parameters to the extracted method. A variable is an input if it is defined before the clone and read within the clone.
$$In(i) = Use_{within}(i) \cap Def_{before}(i)$$

Where:
* $Use_{within}(i)$ is the set of all variables read during the clone block execution: $V_r(WITHIN)$.
* $Def_{before}(i)$ is the set of all variables initialized prior to the clone, including function parameters and variables written in the pre-region: $V_w(PRE) \cup Parameters$.

**The Output Set:** $Out(i)$
The output set represents all variables that must be returned from the extracted method back to the enclosing scope. A variable is an output if it is modified within the clone and subsequently read after the clone.
$$Out(i) = Def_{within}(i) \cap Use_{after}(i)$$

Where:
* $Def_{within}(i)$ is the set of all variables written to (mutated or assigned) inside the clone block: $V_w(WITHIN)$.
* $Use_{after}(i)$ is the set of all variables read in the code following the clone block: $V_r(POST)$.

*Note: If $|Out(i)| > 1$, the pipeline flags the clone with a **Data-Flow Hazard** (Multiple Outputs). While Python technically supports returning multiple values via tuples, cleanly extracting code blocks that mutate excessive external state is considered an anti-pattern for standard automated refactoring.*