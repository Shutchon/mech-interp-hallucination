#!/usr/bin/env python
"""Render Fig. 3 v2 from fig3_attention_v2.json (review R6+R15).

Layout (full text width, 7.16in):
  (a) 5 attention-pattern matrices (one per head, one example prompt),
      shared colorbar, query/key annotations, BOS and distractor marked.
  (b) Attention FROM the answer position TO BOS / distractor / other,
      mean ± SD over N misleading prompts — the quantitative check of
      the specialist BOS-concentration claim.

Usage: python3 scripts/make_fig3_v2.py   (requires results/analysis/fig3_attention_v2.json)
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
D = json.load(open(ROOT / "results/analysis/fig3_attention_v2.json"))
F = ROOT / "paper" / "figs"
HEADS = list(D["summary"].keys())
KIND = {h: ("specialist" if "spec" else "dual") for h in HEADS}
# summary keys don't carry kind; hardcode per extraction order
KIND = {h: k for h, k in zip(HEADS, ["spec", "spec", "spec", "dual", "dual"])}
N = D["n_prompts"]

plt.rcParams.update({"figure.dpi": 200, "font.size": 8, "axes.labelsize": 8,
                     "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
                     "axes.titlesize": 8})

fig = plt.figure(figsize=(7.16, 2.7))
gs = fig.add_gridspec(1, 8, width_ratios=[1, 1, 1, 1, 1, 0.15, 2.1, 0.6],
                      wspace=0.55)

# ---- (a) matrices
ex = D["example"]
toks = ex["tokens"]
d_idx = ex["distractor_idx"]
vmax = 0.35
axes = []
for i, h in enumerate(HEADS):
    ax = fig.add_subplot(gs[0, i])
    pat = np.array(ex["patterns"][h])
    im = ax.imshow(pat, cmap="viridis", vmin=0, vmax=vmax, aspect="auto")
    ax.set_title(f"{h}\n({KIND[h]})", fontsize=7.5,
                 color=("#d62728" if KIND[h] == "spec" else "#1f77b4"))
    ax.set_xticks([0] + d_idx)
    ax.set_xticklabels(["BOS"] + ["D"] * len(d_idx), fontsize=7)
    ax.set_yticks([0, len(toks) - 1])
    ax.set_yticklabels(["BOS", "ans"], fontsize=7)
    if i == 0:
        ax.set_ylabel("query pos", fontsize=7.5)
    axes.append(ax)
cax = fig.add_subplot(gs[0, 5])
cb = fig.colorbar(im, cax=cax)
cb.set_label("attention weight", fontsize=7)
cb.ax.tick_params(labelsize=7)

# ---- (b) answer-position breakdown, mean±SD over prompts
axb = fig.add_subplot(gs[0, 6])
dests = ["bos", "distractor", "other"]
colors = ["#d62728", "#1f77b4", "#bbbbbb"]
width = 0.25
x = np.arange(len(HEADS))
for j, (dest, c) in enumerate(zip(dests, colors)):
    means = [D["summary"][h][dest]["mean"] for h in HEADS]
    sds = [D["summary"][h][dest]["sd"] for h in HEADS]
    axb.bar(x + (j - 1) * width, means, width, yerr=sds, color=c,
            label={"bos": "→BOS", "distractor": "→distractor",
                   "other": "→other"}[dest],
            error_kw=dict(lw=0.8, capsize=2))
axb.set_xticks(x)
axb.set_xticklabels([h.replace(".", "\n") for h in HEADS], fontsize=7)
axb.set_ylabel("attention from answer position")
axb.set_title(f"(b) answer-position attention\n(mean±SD, {N} prompts)", fontsize=7.5)
axb.legend(fontsize=7, frameon=False)
axb.set_ylim(0, 1.05)

fig.text(0.005, 0.97, "(a) attention patterns, one misleading prompt "
          f"(example: {ex['true']} vs. distractor {ex['distractor']}; D marks distractor columns)",
          fontsize=7.5, va="top")
fig.savefig(F / "fig3_attention.pdf", bbox_inches="tight")
fig.savefig(F / "fig3_attention.png", bbox_inches="tight")
print("written fig3_attention.pdf/png")
