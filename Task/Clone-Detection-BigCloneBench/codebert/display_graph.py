#!/usr/bin/env python3
"""
display_graph.py

Generate publication-ready PNG graphs (saved under ./graph/) for embedding in README.

Default data corresponds to the 3 CodeBERT results you shared:
  Exp-1: BCB -> BCB (within-project)
  Exp-2: BCB -> Camel (cross-domain, no adaptation)
  Exp-3: (BCB+Camel) -> Camel (mixed fine-tune)

Usage:
  python display_graph.py
  python display_graph.py --out_dir graph
  python display_graph.py --json results_codebert.json

Optional JSON format:
{
  "experiments": [
    {"id":"Exp-1", "label":"BCB→BCB", "f1":0.9632, "precision":0.9574, "recall":0.9692},
    {"id":"Exp-2", "label":"BCB→Camel", "f1":0.5455, "precision":0.6813, "recall":0.5978},
    {"id":"Exp-3", "label":"(BCB+Camel)→Camel", "f1":0.8773, "precision":0.8815, "recall":0.8776}
  ]
}

Outputs (PNG):
  - graph/f1_three_way.png
  - graph/prf_three_way.png
  - graph/domain_gap_recovery.png
  - graph/relative_gain_percent.png
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import List, Optional

import matplotlib.pyplot as plt


@dataclass
class Exp:
    id: str
    label: str
    f1: float
    precision: float
    recall: float


DEFAULT_EXPS: List[Exp] = [
    Exp("Exp-1", "BCB→BCB", 0.9632, 0.9574, 0.9692),
    Exp("Exp-2", "BCB→Camel", 0.5455, 0.6813, 0.5978),
    Exp("Exp-3", "(BCB+Camel)→Camel", 0.8773, 0.8815, 0.8776),
]


def _ensure_out_dir(out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def _load_from_json(path: str) -> List[Exp]:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    exps = []
    for e in obj["experiments"]:
        exps.append(
            Exp(
                id=str(e.get("id", "")),
                label=str(e["label"]),
                f1=float(e["f1"]),
                precision=float(e["precision"]),
                recall=float(e["recall"]),
            )
        )
    if len(exps) < 3:
        raise ValueError("Need at least 3 experiments for the default comparison plots.")
    return exps


def _annotate_bars(ax, bars, fmt="{:.3f}"):
    for b in bars:
        h = b.get_height()
        ax.annotate(
            fmt.format(h),
            xy=(b.get_x() + b.get_width() / 2, h),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )


def plot_f1_three_way(exps: List[Exp], out_path: str):
    labels = [e.label for e in exps]
    vals = [e.f1 for e in exps]

    plt.figure(figsize=(8.2, 4.6))
    ax = plt.gca()
    bars = ax.bar(labels, vals)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("F1")
    ax.set_title("CodeBERT: F1 Comparison (Three Experiments)")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    plt.xticks(rotation=15, ha="right")
    _annotate_bars(ax, bars, fmt="{:.4f}")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_prf_three_way(exps: List[Exp], out_path: str):
    labels = [e.label for e in exps]
    p = [e.precision for e in exps]
    r = [e.recall for e in exps]
    f = [e.f1 for e in exps]

    x = list(range(len(labels)))
    width = 0.25

    plt.figure(figsize=(9.2, 4.8))
    ax = plt.gca()

    b1 = ax.bar([i - width for i in x], p, width=width, label="Precision")
    b2 = ax.bar(x, r, width=width, label="Recall")
    b3 = ax.bar([i + width for i in x], f, width=width, label="F1")

    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title("CodeBERT: Precision / Recall / F1 (Three Experiments)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend()

    _annotate_bars(ax, b1, fmt="{:.3f}")
    _annotate_bars(ax, b2, fmt="{:.3f}")
    _annotate_bars(ax, b3, fmt="{:.3f}")

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_domain_gap_recovery(exps: List[Exp], out_path: str):
    """
    Visualize:
      - within-project ceiling (Exp-1 F1)
      - cross-domain baseline (Exp-2 F1)
      - adapted (Exp-3 F1)
    plus:
      - absolute drop (gap): Exp-1 - Exp-2
      - recovered: Exp-3 - Exp-2
      - residual gap: Exp-1 - Exp-3
    """
    # Assume ordering: 0=within, 1=cross baseline, 2=adapted
    within = exps[0].f1
    cross = exps[1].f1
    adapted = exps[2].f1

    gap = within - cross
    recovered = adapted - cross
    residual = within - adapted

    plt.figure(figsize=(9.2, 4.8))
    ax = plt.gca()

    # Bars show the three F1 points
    labels = [exps[0].label, exps[1].label, exps[2].label]
    vals = [within, cross, adapted]
    bars = ax.bar(labels, vals)

    ax.set_ylim(0, 1.0)
    ax.set_ylabel("F1")
    ax.set_title("Domain Gap & Recovery (CodeBERT, F1)")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    plt.xticks(rotation=15, ha="right")
    _annotate_bars(ax, bars, fmt="{:.4f}")

    # Text box summary (no color assumptions)
    txt = (
        f"Gap (Within−Cross): {gap:.4f}\n"
        f"Recovered (Adapted−Cross): {recovered:.4f}\n"
        f"Residual (Within−Adapted): {residual:.4f}"
    )
    ax.text(
        0.98,
        0.06,
        txt,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.9),
    )

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_relative_gain_percent(exps: List[Exp], out_path: str):
    """
    Relative improvement of adapted over cross baseline:
      (Exp-3 - Exp-2) / Exp-2 * 100
    Also show cross-domain drop relative to within:
      (Exp-2 - Exp-1) / Exp-1 * 100
    and residual gap relative to within:
      (Exp-3 - Exp-1) / Exp-1 * 100
    """
    within = exps[0].f1
    cross = exps[1].f1
    adapted = exps[2].f1

    rel_recovery = (adapted - cross) / max(cross, 1e-12) * 100.0
    rel_drop = (cross - within) / max(within, 1e-12) * 100.0
    rel_residual = (adapted - within) / max(within, 1e-12) * 100.0

    labels = [
        "Cross-domain drop\n(BCB→Camel vs BCB→BCB)",
        "Recovery\n((BCB+Camel)→Camel vs BCB→Camel)",
        "Residual gap\n((BCB+Camel)→Camel vs BCB→BCB)",
    ]
    vals = [rel_drop, rel_recovery, rel_residual]

    plt.figure(figsize=(10.2, 5.2))
    ax = plt.gca()
    bars = ax.bar(labels, vals)
    ax.set_ylabel("Percent (%)")
    ax.set_title("Relative Changes (CodeBERT, F1)")
    ax.axhline(0, linestyle="--", linewidth=1)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    plt.xticks(rotation=10, ha="right")

    # annotate with +/- formatting
    for b in bars:
        h = b.get_height()
        ax.annotate(
            f"{h:+.1f}%",
            xy=(b.get_x() + b.get_width() / 2, h),
            xytext=(0, 3 if h >= 0 else -14),
            textcoords="offset points",
            ha="center",
            va="bottom" if h >= 0 else "top",
            fontsize=10,
        )

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="graph", help="Output directory for PNG files (default: graph)")
    ap.add_argument("--json", default=None, help="Optional JSON file containing experiment metrics")
    args = ap.parse_args()

    out_dir = _ensure_out_dir(args.out_dir)
    exps = _load_from_json(args.json) if args.json else DEFAULT_EXPS

    # Core outputs (matching typical README embeds)
    plot_f1_three_way(exps, os.path.join(out_dir, "f1_three_way.png"))
    plot_prf_three_way(exps, os.path.join(out_dir, "prf_three_way.png"))
    plot_domain_gap_recovery(exps, os.path.join(out_dir, "domain_gap_recovery.png"))
    plot_relative_gain_percent(exps, os.path.join(out_dir, "relative_gain_percent.png"))

    print("[OK] Wrote graphs to:", os.path.abspath(out_dir))
    for fn in [
        "f1_three_way.png",
        "prf_three_way.png",
        "domain_gap_recovery.png",
        "relative_gain_percent.png",
    ]:
        print(" -", os.path.join(out_dir, fn))


if __name__ == "__main__":
    main()