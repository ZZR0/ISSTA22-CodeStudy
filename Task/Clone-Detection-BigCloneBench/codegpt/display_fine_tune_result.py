#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# -----------------------------
# Results (confirmed from logs)
# -----------------------------
# In-domain (BCB -> BCB test)
BCB_IN_P = 0.9314
BCB_IN_R = 0.9699
BCB_IN_F1 = 0.9494

# Cross-domain baseline (BCB10 -> Camel test)  [Experiment A]
A_P = 0.6769
A_R = 0.5324
A_F1 = 0.4125

# Cross-domain adapted (BCB10 + Camel exposure -> Camel test) [Experiment B]
B_P = 0.7781
B_R = 0.7591
B_F1 = 0.7584

CAMEL_TEST_N = 2034
THRESHOLD = 0.5

OUTDIR = Path("graph")
OUTDIR.mkdir(parents=True, exist_ok=True)
OUTDIR.mkdir(parents=True, exist_ok=True)

def savefig(name: str):
    path = OUTDIR / name
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()
    print(f"[OK] saved: {path.resolve()}")

def add_value_labels(bars):
    for b in bars:
        h = b.get_height()
        plt.text(b.get_x() + b.get_width()/2, h + 0.02, f"{h:.3f}",
                 ha="center", va="bottom", fontsize=9)

def graph_ab_metrics_two_exp():
    # Exp A vs Exp B on Camel test
    metrics = ["Precision", "Recall", "F1"]
    exp_a = [A_P, A_R, A_F1]
    exp_b = [B_P, B_R, B_F1]

    x = np.arange(len(metrics))
    width = 0.35

    plt.figure(figsize=(7.2, 4.2))
    b1 = plt.bar(x - width/2, exp_a, width, label="Exp A: BCB10 → Camel")
    b2 = plt.bar(x + width/2, exp_b, width, label="Exp B: BCB10 + Camel → Camel")
    plt.xticks(x, metrics)
    plt.ylim(0, 1.0)
    plt.ylabel("Score")
    plt.title(f"Camel-only Test (N={CAMEL_TEST_N}, threshold={THRESHOLD})")
    plt.legend()
    add_value_labels(b1)
    add_value_labels(b2)

    savefig("fine-tune-camel.png")  # (기존 README에서 이미 사용 중)

def graph_three_way_f1():
    labels = ["BCB → BCB (In-domain)", "BCB10 → Camel (Exp A)", "BCB10+Camel → Camel (Exp B)"]
    vals = [BCB_IN_F1, A_F1, B_F1]

    x = np.arange(len(labels))
    plt.figure(figsize=(8.8, 4.2))
    bars = plt.bar(x, vals)
    plt.xticks(x, labels, rotation=15, ha="right")
    plt.ylim(0, 1.0)
    plt.ylabel("F1")
    plt.title("F1 Across Evaluation Scenarios")
    add_value_labels(bars)

    savefig("f1_three_way.png")

def graph_three_way_prf():
    labels = ["BCB → BCB", "Exp A (BCB10 → Camel)", "Exp B (BCB10+Camel → Camel)"]
    P = [BCB_IN_P, A_P, B_P]
    R = [BCB_IN_R, A_R, B_R]
    F = [BCB_IN_F1, A_F1, B_F1]

    x = np.arange(len(labels))
    width = 0.25

    plt.figure(figsize=(9.2, 4.2))
    b1 = plt.bar(x - width, P, width, label="Precision")
    b2 = plt.bar(x,         R, width, label="Recall")
    b3 = plt.bar(x + width, F, width, label="F1")

    plt.xticks(x, labels, rotation=10, ha="right")
    plt.ylim(0, 1.0)
    plt.ylabel("Score")
    plt.title("Precision / Recall / F1 Across Scenarios")
    plt.legend()
    add_value_labels(b1); add_value_labels(b2); add_value_labels(b3)

    savefig("prf_three_way.png")

def graph_domain_gap_recovery():
    # Gap = In-domain F1 - baseline cross-domain F1
    gap = BCB_IN_F1 - A_F1
    recovered = B_F1 - A_F1
    remaining = BCB_IN_F1 - B_F1

    labels = ["Recovered", "Remaining Gap"]
    vals = [recovered, remaining]

    plt.figure(figsize=(7.2, 4.2))
    bars = plt.bar(labels, vals)
    plt.ylim(0, max(0.6, gap + 0.05))
    plt.ylabel("F1 Points")
    plt.title("Domain Gap Recovery (F1)")

    for b in bars:
        h = b.get_height()
        plt.text(b.get_x() + b.get_width()/2, h + 0.01, f"{h:.3f}",
                 ha="center", va="bottom", fontsize=10)

    # annotate totals
    plt.text(-0.4, gap + 0.02, f"Total gap: {gap:.3f} (0.949 − 0.413)", fontsize=10)

    savefig("domain_gap_recovery.png")

def graph_relative_improvement():
    # relative (%) improvements from Exp A -> Exp B
    def rel_gain(a, b):
        return (b - a) / a * 100.0 if a != 0 else float("inf")

    labels = ["Precision", "Recall", "F1"]
    gains = [rel_gain(A_P, B_P), rel_gain(A_R, B_R), rel_gain(A_F1, B_F1)]

    x = np.arange(len(labels))
    plt.figure(figsize=(7.2, 4.2))
    bars = plt.bar(x, gains)
    plt.xticks(x, labels)
    plt.ylabel("Relative Gain (%)")
    plt.title("Relative Improvement (Exp B vs Exp A) on Camel Test")

    for b in bars:
        h = b.get_height()
        plt.text(b.get_x() + b.get_width()/2, h + 1.0, f"{h:.1f}%",
                 ha="center", va="bottom", fontsize=10)

    savefig("relative_gain_percent.png")

def main():
    graph_ab_metrics_two_exp()
    graph_three_way_f1()
    graph_three_way_prf()
    graph_domain_gap_recovery()
    graph_relative_improvement()

    print("\n[Done] Generated all graphs:")
    print(" - fine-tune-camel.png")
    print(" - f1_three_way.png")
    print(" - prf_three_way.png")
    print(" - domain_gap_recovery.png")
    print(" - relative_gain_percent.png")

if __name__ == "__main__":
    main()
