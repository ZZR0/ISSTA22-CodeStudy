# Extract Method Refactoring for Java Clone Classes

This program automates **Extract Method** refactoring for Java clone classes using structural information from parsed source code and clone metadata.

It reads clone-class JSON input, analyzes the enclosing Java methods, generates a new extracted method, rewrites the original clone locations with method calls, and writes both rewritten Java files and a merged JSON report.

---

## Features

- Supports **Java clone-class based Extract Method refactoring**
- Uses **method-level AST location** via `tree-sitter`-backed indexing
- Handles clone fragments inside:
  - regular statement blocks
  - `if / else` branches
  - `switch case` blocks
- Infers:
  - extracted method visibility
  - `static` modifier
  - parameter list
  - additional required variables from context
- Rewrites original clone sites with extracted method calls
- Writes:
  - rewritten Java source files
  - one merged JSON report for all processed clone classes
  - a summary file

---

## Project Structure

Typical dependencies used by this script:

- `extract_method_refactor.py` — main refactoring program
- `index_methods.py` — method index builder / lookup
- `util_ast.py` — AST utility helpers such as `enclosing_class_node`

Input data is expected in JSON format, either:

- a **single clone-class object**, or
- a **list of clone-class objects**

---

## Input Format

Each clone-class entry should look like this:

```json
{
  "classid": "hive_1516_1388_vs_hive_1516_1389",
  "nclones": 2,
  "similarity": 100.0,
  "project": "hive",
  "same_file": 1,
  "actual_label": 1,
  "clone_predict": 1,
  "clone_predict_after_adapt": 1,
  "Refactorable": 1,
  "inspection_case": "Case 1 (Both Found - Refactorable)",
  "sources": [
    {
      "file": "systems/hive-java/ql/src/java/org/apache/hadoop/hive/ql/exec/vector/VectorizedBatchUtil.java",
      "range": "405-414",
      "nlines": 10,
      "code": "...",
      "func_id": "hive_1516_1388",
      "In": {
        "In(i)": ["batch", "colIndex", "offset", "rowIndex", "writableCol"],
        "InType": {
          "InType": ["VectorizedRowBatch", "int", "int", "int", "Object"],
          "Defbefore(i)": ["batch", "colIndex", "offset", "rowIndex", "writableCol"]
        }
      },
      "Out": {
        "Out(i)": [],
        "OutType": []
      },
      "enclosing_function": {
        "qualified_name": "VectorizedBatchUtil.setVector",
        "fun_range": "295-536",
        "fun_nlines": 242,
        "func_code": "..."
      }
    }
  ]
}
````

### Required fields

At clone-class level:

* `classid`
* `sources`

At source level:

* `file`
* `range`
* `code`
* `func_id`
* `enclosing_function`

Inside `enclosing_function`:

* `qualified_name`
* `fun_range`
* `func_code`

---

## Output

The program produces three kinds of output:

### 1. Rewritten Java files

For each processed clone class, rewritten files are written under:

```text
<output-dir>/<classid>/<original-relative-path>
```

### 2. Merged JSON report

A single merged report is written, for example:

```text
refactor_out/all_refactor_results.json
```

This file contains all refactor results.

### 3. Summary report

A short summary file is written to:

```text
refactor_out/summary.json
```

---

## Command Line Usage

```bash
python extract_method_refactor.py \
  --input ./data/case.json \
  --source-root /path/to/project/root \
  --output-dir ./data/refactor_out
```

### Arguments

* `--input`
  Path to the clone-class JSON input file

* `--source-root`
  Root directory used to resolve relative source file paths from the JSON input

* `--output-dir`
  Directory where rewritten files and reports will be written

* `--full-report`
  Write a verbose merged JSON report instead of a concise one

* `--merged-report-name`
  Output filename for the merged report
  Default: `all_refactor_results.json`

Example:

```bash
python extract_method_refactor.py \
  --input ./data/case.json \
  --source-root /home/user1-system11/research_dream/llm-clone/automate_extract_method_refactoring \
  --output-dir ./data/refactor_out \
  --full-report
```

---

## How It Works

### 1. Load clone metadata

The program reads the clone-class JSON and converts it into structured Python dataclasses.

### 2. Locate clone regions and enclosing methods

Using method indexing and AST helpers, the program finds:

* the enclosing method
* the enclosing class
* the precise byte/line range of each clone

### 3. Normalize the clone body

The selected clone fragment is normalized into a candidate extracted body.

Special handling exists for:

* `case X: { ... }`
* `if (...) { ... }`
* `else { ... }`
* partial `if` selections where the selected clone corresponds to only the `then` branch of a larger `if/else`

### 4. Infer extracted method signature

The tool derives:

* method name
* visibility
* `static` modifier
* common input parameters
* additional context variables referenced by the extracted body

### 5. Render extracted method

The normalized body is wrapped into a Java method definition.

### 6. Rewrite clone call sites

Each clone instance is replaced with a call to the extracted method.

### 7. Insert extracted method into class

The new method is inserted into the enclosing class before the closing class brace.

---

## Example Output Shape

Concise merged report entry:

```json
{
  "classid": "hive_1516_1388_vs_hive_1516_1389",
  "extracted_method": {
    "method_name": "extracted",
    "visibility": "private",
    "is_static": true,
    "return_type": "void",
    "parameters": [
      "VectorizedRowBatch batch",
      "int colIndex",
      "int offset",
      "int rowIndex",
      "Object writableCol"
    ],
    "code": "private static void extracted(...) { ... }"
  },
  "sources": [
    {
      "func_id": "hive_1516_1388",
      "file": "...",
      "range": "405-414",
      "replacement_code": "case TIMESTAMP: { extracted(...); }"
    }
  ],
  "updated_files": [
    {
      "file": "...",
      "inserted_extracted_method": true,
      "rewritten_file_path": "..."
    }
  ]
}
```

---

## Current Limitations

This tool is intended for a **first-step Extract Method pass on Type-1 style clones**.

### Important limitation

If clone bodies are not truly behavior-equivalent, plain Extract Method may be unsafe.

Example:

* one clone uses `TimestampColumnVector`
* another uses `LongColumnVector`

These are not the same behavior, even if they look structurally similar.

In such cases, the safer action is to mark the clone class as **not refactorable by plain Extract Method**.

### Recommended policy

Reject clone classes as non-refactorable when:

* normalized clone bodies are not semantically equivalent
* type casts differ in meaningful ways
* method calls differ in behavior
* assignments differ beyond trivial renaming
* the clone would require template extraction or parameterized code generation rather than plain Extract Method

A future improvement is to add a **structural compatibility check** that:

1. normalizes clone bodies
2. compares them structurally
3. measures semantic differences
4. rejects non-Type-1-compatible clones with a reason such as:

```json
{
  "refactorable_by_extract_method": false,
  "reason": "Requires parameterized template extraction"
}
```

---

## Known Safe Use Case

Best suited for clone classes where:

* clone fragments are syntactically equivalent or near-equivalent
* variable differences can be explained by parameter passing
* behavior is preserved by moving the exact common body to a helper method

---

## Troubleshooting

### Error: `TypeError: list indices must be integers or slices, not str`

Cause: the input JSON is a list, but code assumes a single object.
Fix: ensure the loader supports both a single dict and a list of dicts.

### Error: method key not found

Cause: the enclosing method key derived from:

* `qualified_name`
* `fun_range`

does not match the method index.

Fix:

* verify `qualified_name`
* verify `fun_range`
* verify the source file belongs under `--source-root`

### Wrong extracted body for `if / else`

Cause: selected clone range may correspond only to a branch of a larger `if_statement`.

Fix:

* trim whitespace around selected range
* match AST consequence / alternative nodes exactly
* handle partial `if (...) { then }` selections when the actual AST node includes an `else`

### Wrong call arguments in second clone

Cause: parameter mapping may still rely too much on exact names.

Fix:

* derive argument mapping from each clone’s own normalized body
* later improve this with AST/data-flow role matching

---

## Future Improvements

* Structural compatibility check for strict Type-1 Extract Method
* Reject unsafe clone classes automatically
* Better role-based parameter matching from AST/data-flow
* Return value synthesis
* Multi-variable output handling
* Better method naming strategy
* Formatting cleanup for generated Java output
* Optional dry-run mode
* Optional diff output

---

## Requirements

* Python 3.10+
* Use `conda activate ast-analysis` 
* Java parsing/index support from local project modules:

  * `index_methods.py`
  * `util_ast.py`

If your method index depends on `tree-sitter`, ensure the related parser environment is installed and working.

