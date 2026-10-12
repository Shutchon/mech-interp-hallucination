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
    """Compact vertical layout for IEICE column (~3.4in wide, ≤4.5in tall)."""
    fig, ax = plt.subplots(figsize=(3.3, 4.0))
    ax.axis("off")
    W = 30.0

    def box(y, h, text, fc="#eaf2fb", ec="#1f77b4", fs=6.8):
        ax.add_patch(FancyBboxPatch((1, y), W, h, boxstyle="round,pad=0.015",
                                    fc=fc, ec=ec, lw=1.0))
        ax.text(1 + W / 2, y + h / 2, text, ha="center", va="center", fontsize=fs)

    def arrow(y1, y2):
        ax.add_patch(FancyArrowPatch((1 + W / 2, y1), (1 + W / 2, y2), arrowstyle="-|>",
                                     mutation_scale=8, color="#555555", lw=1.0))

    box(67, 11, "CounterFact (21,919)", fc="#fdf3e7", ec="#d62728")
    arrow(67, 63)
    box(52, 11, "known-fact filter → 2,000", fc="#fdf3e7", ec="#d62728")
    arrow(52, 48)
    box(37, 11, "CM-conflict suite (3T + neutral)", fc="#fdf3e7", ec="#d62728")
    arrow(37, 33)
    box(18, 15, "dev / held-out split\nBY RELATION (anti-circularity)", fc="#fde9ef", ec="#c44e52")
    arrow(18, 14)
    box(3, 11, "patching (dev): coarse → mid → fine", fc="#eaf2fb", ec="#1f77b4")
    arrow(3, -1)
    box(-14, 13, "bidirectional + neutral patching\n(superposition; specialists)", fc="#eaf2fb", ec="#1f77b4")
    arrow(-14, -18)
    box(-33, 15, "interventions (held-out):\nknockout / mean-ablation + baselines", fc="#eef7ee", ec="#2ca02c")
    arrow(-33, -37)
    box(-52, 15, "eval: follow/truth/neutral\nMMLU · PPL · McNemar + CIs", fc="#eef7ee", ec="#2ca02c")
    ax.set_xlim(0, 1 + W + 1); ax.set_ylim(-54, 80)
    fig.tight_layout()
    fig.savefig(F / "fig1_pipeline.png", bbox_inches="tight", dpi=200)
    fig.savefig(F / "fig1_pipeline.pdf", bbox_inches="tight")
    plt.close(fig)


def fig3_attention():
    """Two contrasting heads side-by-side, column-sized (~3.4 × 3.2 in)."""
    d = json.load(open(ROOT / "results/analysis/fig3_attention.json"))
    tokens = d["tokens"]
    show = [("L22_H4", "specialist"), ("L25_H4", "dual-role")]

    fig, axes = plt.subplots(1, 2, figsize=(3.4, 3.0))
    import matplotlib.colors as mcolors
    cmap = mcolors.LinearSegmentedColormap.from_list("m", ["#ffffff", "#1f77b4"])
    cmap.set_bad("#f0f0f0")
    labels = [("⟨bos⟩" if t == "<bos>" else t.replace("▁", " ")) for t in tokens]
    for i, (ax, (h, kind)) in enumerate(zip(np.atleast_1d(axes), show)):
        pat = np.array(d["patterns"][h])
        pat = np.where(np.tril(np.ones_like(pat), k=-1) > 0, np.nan, pat)
        ax.imshow(pat.T, cmap=cmap, vmin=0, vmax=1, aspect="auto")
        if i == 0:
            ax.set_yticks(range(len(labels)), labels, fontsize=4.5)
        else:
            ax.set_yticks([])
        ax.set_xticks(range(len(labels)), labels, rotation=90, fontsize=4.5)
        ax.set_title(f"{h}\n({kind})", fontsize=7)
        ax.grid(False)
    fig.suptitle("Attention on misleading prompt\n(true: Israel / distractor: Portsmouth)",
                 fontsize=7.5)
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    fig.savefig(F / "fig3_attention.png", bbox_inches="tight")
    fig.savefig(F / "fig3_attention.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    fig1_pipeline()
    fig3_attention()
    for f in sorted(F.glob("fig*.png")):
        print("created:", f.name)
