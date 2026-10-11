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
    fig, ax = plt.subplots(figsize=(10.5, 3.4))
    ax.axis("off")

    def box(x, y, w, h, text, fc="#eaf2fb", ec="#1f77b4", fs=8.5):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02",
                                    fc=fc, ec=ec, lw=1.2))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs)

    def arrow(x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                     mutation_scale=12, color="#555555", lw=1.2))

    # data track
    box(0.5, 62, 13, 22, "CounterFact\n(21,919 triples)", fc="#fdf3e7", ec="#d62728")
    box(16, 62, 13, 22, "known-fact filter\n(argmax + gap ≥ 0.5)\n→ 2,000 facts", fc="#fdf3e7", ec="#d62728")
    box(31.5, 62, 13, 22, "CM-conflict suite\n3 templates + neutral\n(8,000 prompts)", fc="#fdf3e7", ec="#d62728")
    box(47, 62, 15, 22, "dev / held-out split\nBY RELATION\n(anti-circularity)", fc="#fde9ef", ec="#c44e52")

    # localization track
    box(16, 22, 28, 24, "activation patching (dev only)\ncoarse: block × layer × position\nmid: attn vs MLP · fine: per-head z\n2 corruption regimes", fc="#eaf2fb", ec="#1f77b4")
    box(47, 22, 15, 24, "bidirectional +\nneutral patching\n(superposition,\nspecialists)", fc="#eaf2fb", ec="#1f77b4")

    # intervention track
    box(16, -14, 28, 24, "interventions (held-out only)\nknockout / mean-ablation top-k\nspecialist configs · random/norm\nbaselines", fc="#eef7ee", ec="#2ca02c")
    box(47, -14, 15, 24, "evaluation\nfollow/truth/neutral\nMMLU · PPL\nMcNemar + CIs", fc="#eef7ee", ec="#2ca02c")

    arrow(13.5, 73, 16, 73); arrow(29, 73, 31.5, 73); arrow(44.5, 73, 47, 73)
    arrow(38, 73, 38, 84); arrow(38, 84, 30, 84)  # filter feeds down (visual shortcut)
    arrow(30, 62, 30, 46)
    arrow(44, 34, 47, 34)
    arrow(54.5, 22, 54.5, 10)
    arrow(47, 73, 54.5, 73); arrow(54.5, 73, 54.5, 62)
    arrow(30, 22, 30, 10)
    ax.text(38, -20, "selection uses dev relations only — all reported results on held-out relations",
            ha="center", fontsize=8.5, style="italic", color="#c44e52")
    ax.set_xlim(-1, 65); ax.set_ylim(-22, 88)
    fig.tight_layout()
    fig.savefig(F / "fig1_pipeline.png", bbox_inches="tight", dpi=200)
    fig.savefig(F / "fig1_pipeline.pdf", bbox_inches="tight")
    plt.close(fig)


def fig3_attention():
    d = json.load(open(ROOT / "results/analysis/fig3_attention.json"))
    tokens = d["tokens"]
    heads = list(d["patterns"])
    n = len(heads)
    fig, axes = plt.subplots(1, n, figsize=(2.5 * n, 6.2))
    labels = [t.replace("▁", " ") if t not in ("<bos>",) else "⟨bos⟩" for t in tokens]

    import matplotlib.colors as mcolors

    for ax, h in zip(np.atleast_1d(axes), heads):
        pat = np.array(d["patterns"][h])
        pat = np.where(np.tril(np.ones_like(pat), k=-1) > 0, np.nan, pat)  # mask causal zeros
        cmap = mcolors.LinearSegmentedColormap.from_list("m", ["#ffffff", "#1f77b4"])
        cmap.set_bad("#f0f0f0")
        ax.imshow(pat.T, cmap=cmap, vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(len(labels)), labels, rotation=90, fontsize=6)
        ax.set_yticks(range(len(labels)), labels, fontsize=6)
        ax.set_title(h, fontsize=10)
        ax.grid(False)
    fig.suptitle("Attention patterns of top heads — misleading prompt "
                 f"(true: {d['fact']['true']}, distractor: {d['fact']['false']})", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(F / "fig3_attention.png", bbox_inches="tight")
    fig.savefig(F / "fig3_attention.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    fig1_pipeline()
    fig3_attention()
    for f in sorted(F.glob("fig*.png")):
        print("created:", f.name)
