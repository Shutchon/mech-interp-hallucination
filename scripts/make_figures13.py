#!/usr/bin/env python
"""Fig 1 (pipeline diagram) + Fig 3 (attention heatmaps) — Mac, no GPU.

Usage: .venv-mac/bin/python scripts/make_figures13.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
F = ROOT / "figures"
F.mkdir(exist_ok=True)


def fig1_pipeline():
    """Vertical layout sized for a single column (~3.4in wide)."""
    fig, ax = plt.subplots(figsize=(3.5, 5.6))
    ax.axis("off")
    W = 30.0

    def box(y, h, text, fc="#eaf2fb", ec="#1f77b4", fs=7.6):
        ax.add_patch(FancyBboxPatch((1, y), W, h, boxstyle="round,pad=0.02",
                                    fc=fc, ec=ec, lw=1.1))
        ax.text(1 + W / 2, y + h / 2, text, ha="center", va="center", fontsize=fs)

    def arrow(y1, y2):
        ax.add_patch(FancyArrowPatch((1 + W / 2, y1), (1 + W / 2, y2), arrowstyle="-|>",
                                     mutation_scale=10, color="#555555", lw=1.1))

    box(72, 13, "CounterFact (21,919 triples)", fc="#fdf3e7", ec="#d62728")
    arrow(72, 66.5)
    box(53, 13, "known-fact filter\n(argmax + gap ≥ 0.5) → 2,000", fc="#fdf3e7", ec="#d62728")
    arrow(53, 47.5)
    box(34, 13, "CM-conflict suite\n3 templates + neutral (8,000)", fc="#fdf3e7", ec="#d62728")
    arrow(34, 28.5)
    box(15, 13, "dev / held-out split\nBY RELATION (anti-circularity)", fc="#fde9ef", ec="#c44e52")
    arrow(15, 9.5)
    box(-4, 13, "activation patching (dev only)\ncoarse → mid (attn vs MLP)\n→ fine (per-head z)", fc="#eaf2fb", ec="#1f77b4")
    arrow(-4, -14.5)
    box(-28, 13, "bidirectional + neutral patching\n(superposition; specialists)", fc="#eaf2fb", ec="#1f77b4")
    arrow(-28, -32.5)
    box(-46, 13, "interventions (held-out only)\nknockout / mean-ablation\n+ random & norm baselines", fc="#eef7ee", ec="#2ca02c")
    arrow(-46, -50.5)
    box(-64, 13, "evaluation\nfollow / truth / neutral · MMLU · PPL\nMcNemar + bootstrap CIs", fc="#eef7ee", ec="#2ca02c")
    ax.text(1 + W / 2, -71.5, "selection: dev relations only\nresults: held-out relations only",
            ha="center", va="top", fontsize=7.4, style="italic", color="#c44e52")
    ax.set_xlim(0, 1 + W + 1); ax.set_ylim(-83, 86)
    fig.tight_layout()
    fig.savefig(F / "fig1_pipeline.png", bbox_inches="tight", dpi=200)
    fig.savefig(F / "fig1_pipeline.pdf", bbox_inches="tight")
    plt.close(fig)


def fig3_attention():
    """3 representative heads, stacked vertically, full token set — column-sized."""
    d = json.load(open(ROOT / "results/analysis/fig3_attention.json"))
    tokens = d["tokens"]
    show = ["L22_H4", "L22_H0", "L25_H4"]  # specialist, specialist, dual-role
    labels = [("⟨bos⟩" if t == "<bos>" else t.replace("▁", " ")) for t in tokens]

    fig, axes = plt.subplots(len(show), 1, figsize=(3.6, 3.05 * len(show)))
    import matplotlib.colors as mcolors
    cmap = mcolors.LinearSegmentedColormap.from_list("m", ["#ffffff", "#1f77b4"])
    cmap.set_bad("#f0f0f0")
    for ax, h in zip(np.atleast_1d(axes), show):
        pat = np.array(d["patterns"][h])
        pat = np.where(np.tril(np.ones_like(pat), k=-1) > 0, np.nan, pat)
        ax.imshow(pat.T, cmap=cmap, vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(len(labels)), labels, rotation=90, fontsize=6.5)
        ax.set_yticks(range(len(labels)), labels, fontsize=6.5)
        kind = "specialist" if h in ("L22_H4", "L22_H0", "L20_H3") else "dual-role"
        ax.set_title(f"{h}  ({kind})", fontsize=9)
        ax.grid(False)
    fig.suptitle("Attention of top heads on a misleading prompt\n"
                 "(true: Israel, distractor: Portsmouth)", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(F / "fig3_attention.png", bbox_inches="tight")
    fig.savefig(F / "fig3_attention.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    fig1_pipeline()
    fig3_attention()
    for f in sorted(F.glob("fig*.png")):
        print("created:", f.name)
