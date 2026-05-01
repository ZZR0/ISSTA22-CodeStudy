# AST Clone Extractability

AST-based static analysis framework for evaluating **Extract Method refactoring feasibility** of code clones detected by NiCad.

This tool analyzes clone instances using Tree-sitter AST parsing and computes:

* Variable read/write sets (`V_r`, `V_w`)
* Region-based slicing (pre / within / post)
* Data-flow artifacts: `In(i)` and `Out(i)`
* Control-flow hazard detection (`CFHazard`)
* Final extractability decision (`Extractable`)

The analysis follows conservative safety checks to prevent semantic changes during refactoring.

---

## 📦 Project Structure

```
AST_Clone_Extractability/
│
├── main.py                         # Entry point
├── io_nicad.py                     # NiCad JSON/JSONL loader
├── index_methods.py                # Method indexing per file
├── rw_vars.py / rw_vars_v1.py      # Read/Write variable analysis
├── hazards.py                      # Control-flow hazard detection
├── feasibility.py                  # In/Out + Extractability decision
├── java_treesitter_parser.py       # Tree-sitter Java parser wrapper
├── java_class_method_visitor.py    # Method AST visitor
├── util_ast.py                     # AST utilities
│
├── input/                          # NiCad input files
├── output/                         # Generated results
└── systems/                        # Analyzed source code (NiCad systems)
```

---

## 🔬 Analysis Pipeline

For each clone instance:

1. Locate enclosing method via AST index
2. Slice method into:

   * `CloneRegion_pre`
   * `CloneRegion_within`
   * `CloneRegion_post`
3. Extract:

   * `V_r` (read variables)
   * `V_w` (write variables)
4. Compute:

   * `In(i)`  → variables required as parameters
   * `Out(i)` → variables that must be returned
5. Detect:

   * Control-flow hazards (`return`, `break`, `continue`, `throw`, `yield`)
6. Decide:

   * `Extractable = True/False` based on:

     * |In(i)| ≤ P
     * |Out(i)| ≤ R
     * No control-flow hazard

---

## ⚙️ Environment Setup

Create environment:

```bash
conda create -n ast-analysis python=3.11 -y
conda activate ast-analysis
```
## Install packages
```bash
python -m pip install --upgrade pip
pip install "tree-sitter>=0.25,<0.26" tree-sitter-java
```
---

## ▶️ Running the Analysis

```bash
PYTHONPATH=.. python main.py \
  --input input/nicad_ant.jsonl \
  --output output/nicad_ant_feasibility.jsonl \
  --P 7 \
  --R 1 \
  --debug-hazard \
  > output/nicad_ant_feasibility_hazard.log 2>&1
```

This saves all hazard diagnostics into:

```
output/nicad_feasibility_hazard.log
```
### Parameters

| Argument         | Description                        |        |   |
| ---------------- | ---------------------------------- | ------ | - |
| `--input`        | NiCad JSON or JSONL file           |        |   |
| `--output`       | Output file path                   |        |   |
| `--P`            | Maximum allowed parameter count (  | In(i)  | ) |
| `--R`            | Maximum allowed return variables ( | Out(i) | ) |
| `--debug-hazard` | Print control-flow hazard details  |        |   |


---

## 📊 Output Format

For each clone instance, the output includes:

```json
{
  "CloneRegion_pre": {
    "var_read": [],
    "var_write": []
  },
  "CloneRegion_within": {
    "var_read": ["buffer"],
    "var_write": ["i", "ch"]
  },
  "CloneRegion_post": {
    "var_read": [],
    "var_write": []
  },
  "In": ["buffer"],
  "Out": [],
  "CFHazard": true,
  "Extractable": false
}
```

---

## 🛡 Control-Flow Hazard Definition

A clone instance is marked as hazardous if it contains:

* `return_statement` (non-tail)
* `break_statement`
* `continue_statement`
* `throw_statement`
* `yield_statement`

Nested lambda/method bodies are excluded from hazard detection.

The analysis is conservative to ensure refactoring safety.

---
