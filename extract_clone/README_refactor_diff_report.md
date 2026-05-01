# Refactoring Diff Report Generator

This tool analyzes machine-learning-based code clone extractions and generates a professional, standalone HTML inspection report. It enables researchers to compare "Original Clone" snippets against "Manual Ground Truth" refactorings, complete with automated quality metrics.

## Features

* **Side-by-Side Diff View**: Directly compare original clones with refactored ground truth side-by-side.
* **Automated Quality Metrics**: Automatically calculates **LOC Reduction** and **Interface Quality** (parameter coupling analysis) for every refactored pair.
* **Metadata Integration**: Displays file paths, line ranges, and parameter/return derivation logic (In/Out sets) for deep analysis.
* **Statistical Overview**: Provides a high-level dashboard summarizing success/failure rates of the refactoring process.

## Prerequisites

* Python 3.x
* `pandas` library (for data processing):
```bash
pip install pandas

```



## Usage

1. **Prepare your data**: Ensure your `sampled_70pairs_after_ref.json` is formatted as an array of objects, each containing a `classid` and a list of `sources` (functions).
2. **Execute the script**:
```bash
python generate_diff_report.py

```


3. **View the output**: The script will generate a standalone HTML file at `./output/sampling_clones/refactor_diff_report.html`.

## Understanding the Report

* **Statistical Header**: Displays the `Total Pairs`, `Success` (refactored), and `Failed` counts.
* **Clone Class ID**: The unique identifier for the clone pair being analyzed.
* **Derivation Box**: Highlights the `Extracted Signature`, `In(i)` parameters, and `Out(i)` return variables derived from your analysis pipeline.
* **Metric Card**:
* **LOC Reduction**: Measures the reduction in lines of code achieved by the refactoring.
* **Interface Quality**: Flags "Balanced" (signature params match input logic) or "Mismatched" (potential coupling issues).

