# NiCad Post-Processing Pipeline (Java / Camel)

This document describes the full post-processing pipeline used to convert NiCad XML clone results into JSONL datasets suitable for training and analysis.

The pipeline consists of **5 steps**, executed via `src/run.sh`.

---

## Directory Structure

Expected layout:

```
NiCad/post_process/
├── input/
│   └── camel-java_functions-blind-clones-0.30-classes-withsource.xml
├── data/
│   └── java/
│       ├── pipeline.log
│       ├── step1_nicad_camel_sim0.7_classes.jsonl
│       ├── camel_sim0.7.jsonl
│       ├── nicad_camel_clone_func.jsonl
│       ├── nicad_camel_clone_data.jsonl
│       ├── nicad_camel_neg_samples.jsonl
│       ├── nicad_camel_neg_samples.txt
│       └── display_neg_sample.md
└── src/
    ├── run.sh
    ├── 1_nicad_xml_to_jsonl.py
    ├── 2_java_qmethod_ts.py
    ├── 3_filter_out_data.py
    ├── 4_gen_init_train_sample.py
    └── 5_gen_neg_clone_sample.py
```

---

## Quick Start (Run All Steps)

From the `src/` directory:

```bash
cd NiCad/post_process/src
chmod +x run.sh
./run.sh
```

All console output is also written to: `../data/java/pipeline.log`

---

## Pipeline Steps

### STEP 1 — Convert NiCad XML → JSONL clone groups

**Script:** `src/1_nicad_xml_to_jsonl.py`

**Input:**

- `input/camel-java_functions-blind-clones-0.30-classes-withsource.xml`

**Output:**

- `data/java/step1_nicad_camel_sim0.7_classes.jsonl`

**What it does:**

- Parses NiCad XML `<class>` blocks
- Produces one JSON object per clone group
- Stores clone sources in `sources[]` with file, range, code, etc.

---

### STEP 2 — Add Fully Qualified Names (FQN) to each function

**Script:** `src/2_java_qmethod_ts.py`

**Input:**

- `data/java/step1_nicad_camel_sim0.7_classes.jsonl`

**Output:**

- `data/java/camel_sim0.7.jsonl`

**What it does:**

- For each source function, adds: `sources[].qualified_name`
- The qualified name includes: package name + class name + method signature

**Example:**

```json
"qualified_name": "org.apache.camel.util.AnnotationHelper.getAnnotationValue(Method method, String fqnAnnotationName, String key)"
```

---

### STEP 3 — Filter clone groups + remove comments

**Script:** `src/3_filter_out_data.py`

**Input:**

- `data/java/camel_sim0.7.jsonl`

**Output:**

- `data/java/nicad_camel_clone_func.jsonl`

**What it does:**

1. **Filters test-related code**
    - Default mode: drop the entire group if any source is from test paths (`src/test`, `tests`, `integration-test`, etc.)

2. **Limits clone group size**
    - Drops groups where `nclones >= 20`

3. **Removes Java comments** (default)
    - Strips `//...` and `/*...*/` while preserving string literals
    - Helps reduce token usage for downstream processing

---

### STEP 4 — Assign unique func_id for each function

**Script:** `src/4_gen_init_train_sample.py`

**Input:**

- `data/java/nicad_camel_clone_func.jsonl`

**Output:**

- `data/java/nicad_camel_clone_data.jsonl`

**What it does:**

- Adds a unique ID to every function inside a clone group: `sources[].func_id`

**Format:** `{classid}_{global_counter}`

**Examples:**

- `1423_0`
- `1423_1`
- `99_2`

This ID is required for generating positive/negative training pairs.

---

### STEP 5 — Generate negative (non-clone) pairs

**Script:** `src/5_gen_neg_clone_sample.py`

**Input:**

- `data/java/nicad_camel_clone_data.jsonl`

**Outputs:**

- **JSONL pairs:** `data/java/nicad_camel_neg_samples.jsonl`
- **TXT pairs (for training):** `data/java/nicad_camel_neg_samples.txt`
- **Markdown inspection report:** `data/java/display_neg_sample.md`
- **HTML inspection report:** `data/java/display_neg_sample.html` (recommended)

**What it does:**

- Loads all functions with `func_id`
- Computes dataset capacity:
    - Total functions
    - Possible pairs
    - Positive pairs (within same clone group)
    - Max negative pairs (cross-group)
- Randomly generates balanced negative pairs:
    - `label 0`
    - Functions from different clone groups
- Writes outputs + verifies correctness

**TXT format:**

```
func_id_1<TAB>func_id_2<TAB>0
```

---

## Output Summary

| File                                     | Description                                                      |
| ---------------------------------------- | ---------------------------------------------------------------- |
| `step1_nicad_camel_sim0.7_classes.jsonl` | NiCad XML parsed into clone groups                               |
| `camel_sim0.7.jsonl`                     | Clone groups with `qualified_name` added                         |
| `nicad_camel_clone_func.jsonl`           | Filtered clone groups (no tests, nclones < 20, comments removed) |
| `nicad_camel_clone_data.jsonl`           | Clone groups with unique `func_id` per function                  |
| `nicad_camel_neg_samples.jsonl`          | Negative sample pairs (label 0)                                  |
| `nicad_camel_neg_samples.txt`            | Negative sample pairs (training format)                          |
| `display_neg_sample.md/html`             | Human-readable inspection reports                                |

---

## Notes / Tips

### Git Issues

If you see "push rejected (fetch first)", run:

```bash
git pull --rebase origin main
```

### Large File Handling

If large XML files cannot be opened in VS Code, use terminal tools:

```bash
ls -lh <file>
head -n 50 <file>
```

---
