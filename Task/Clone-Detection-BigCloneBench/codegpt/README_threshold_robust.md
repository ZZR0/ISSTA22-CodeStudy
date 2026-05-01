# Plan: Threshold Robustness Analysis for Cross-Domain Clone Detection

**Date:** 2026-02-06  
**Model:** CodeGPT-small-java  
**Task:** Cross-domain code clone detection  
**Source Domain:** BigCloneBench (BCB)  
**Target Domain:** Camel (NiCad post-processed)

---

## 1. Motivation

Cross-domain clone detection performance can be sensitive to the choice of
similarity threshold. To ensure that the observed performance improvement of
the domain-adapted model is **not an artifact of a specific threshold choice**,
we conduct a **threshold robustness analysis**.

This analysis supports formal statistical significance
tests performed at the per-instance level.

---

## 2. What This Analysis Is (and Is Not)

### This analysis IS:
- A **robustness / sensitivity analysis**
- A way to visualize how performance varies as the NiCad similarity threshold changes
- A distributional comparison between:
  - **Baseline model** (BCB10 → Camel)
  - **Domain-adapted model** (BCB10 + Camel → Camel)

- An analysis for statistical significance testing
- A basis for claiming a larger sample size

---

## 3. Threshold Range Design

Instead of evaluating only a few isolated thresholds, we group thresholds into
**ranges** to mitigate arbitrariness and better capture robustness.

Planned NiCad similarity threshold ranges include:

* 01: **100 – 95**
* 02: **95 – 90**
* 03: **90 – 85**
* 04: **85 – 80**
* 05: **80 – 75**
* 06: **75 – 70**
* 07: **70 – 65**
* 08: **65 – 60**
* 09: **60 – 55**
* 10: **55 – 50**

Within each range, multiple thresholds are swept and evaluated.

---

## 4. Metrics and Visualization

For each threshold range and each model (Exp A / Exp B):

- Precision, Recall, and F1 are computed at each threshold
- Results are summarized using **box plots**, capturing:
  - Median performance
  - Interquartile range (IQR)
  - Variability induced by threshold choice

These plots visualize **distributional shifts** rather than single-point
performance differences.

---

## 5. Expected Outcome

We expect to observe that:

- Across all threshold ranges,
  **the domain-adapted model consistently outperforms the baseline**
- The performance gap is stable and does not collapse at specific thresholds
- Improvements are **robust to threshold selection**

This supports the claim that the observed gains are **structural**, not
threshold-specific.

---

## 6. Relation to Statistical Significance Testing

This threshold robustness analysis is **complementary** to statistical tests.

- **Primary statistical evidence**:
  - Per-instance paired tests (e.g., Wilcoxon signed-rank, McNemar)
  - Sample size = number of test instances (e.g., 2034 Camel pairs)

- **Secondary robustness evidence**:
  - Threshold range sweeps and box plots (this plan)

Both are necessary for a complete and defensible evaluation.

---

## 7. Reporting Language (Guideline)

When reporting these results:

- Statistical significance based on threshold sweeps alone
- Use language such as:
  - “robust across thresholds”
  - “consistent improvement”
  - “not sensitive to threshold selection”

### Example Sentence

> *We further evaluated robustness across multiple ranges of NiCad similarity
> thresholds. Across all threshold ranges, the domain-adapted model consistently
> outperformed the baseline, indicating that the observed improvement is not
> sensitive to threshold selection.*

---

## 8. Deliverables

- Threshold sweep evaluation scripts
- Box plot figures for each threshold range
- One robustness figure section (main paper or appendix)
- Clear separation between robustness analysis and statistical significance tests

---
