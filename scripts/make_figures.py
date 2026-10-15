#!/usr/bin/env python
"""Generate paper figures from results JSONs (no GPU needed).

Round-2 revision: every figure is created at the exact print size
(single column = 3.39in) so that fonts stay >=7pt in print (IEICE rule).

Fig 2: (a) gemma layer profiles, both corruption regimes; (b) cross-model
       layer profiles at normalised depth.   [stacked, single column]
Fig 4: intervention trade-offs — (a) suite space (follow vs truth);
       (b) capability space (follow vs MMLU retention, PPL annotated).
       [stacked, single column; includes spec1 capability]
Fig 5: superposition scatter — context-push (B) vs restore (A) per head,
       coloured by neutral ratio, spec1 highlighted.

Usage: python3 scripts/make_figures.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"
F = ROOT / "paper" / "figs"
F.mkdir(parents=True, exist_ok=True)
COL = 3.39  # IEICE column width in inches
plt.rcParams.update({"figure.dpi": 200, "font.size": 8, "axes.grid": True,
                     "grid.alpha": 0.3, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.labelsize": 8, "axes.titlesize": 8.5, "xtick.labelsize": 7.5,
                     "ytick.labelsize": 7.5, "legend.fontsize": 7})


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def layer_profile(coarse_path: Path, n_layers: int) -> np.ndarray:
    agg = {}
    for r in load_jsonl(coarse_path):
        agg.setdefault(r["layer"], []).append(r["effect"])
    return np.array([np.mean(agg.get(l, [0.0])) for l in range(n_layers)])


def fig2():
    gemma_mis = layer_profile(R / "e3_sweep/gemma-2-2b/misleading_coarse.jsonl", 26)
    gemma_noise = layer_profile(R / "e3_sweep/gemma-2-2b/noise_coarse.jsonl", 26)
    llama = layer_profile(R / "e6_cross/meta-llama/Llama-3.2-1B/coarse.jsonl", 16)
    qwen = layer_profile(R / "e6_cross/Qwen/Qwen2.5-1.5B/coarse.jsonl", 28)

    fig, axes = plt.subplots(2, 1, figsize=(COL, 3.25))
    ax = axes[0]
    ax.plot(gemma_noise, "-o", ms=2.5, lw=1.2, label="noise (ROME-style)", color="#1f77b4")
    ax.plot(gemma_mis, "-s", ms=2.5, lw=1.2, label="misleading context", color="#d62728")
    ax.set(xlabel="layer", ylabel="restoration\neffect", title="(a) Gemma-2-2B, coarse sweep")
    ax.legend(frameon=False)
    ax = axes[1]
    for prof, name, c in ((gemma_mis, "Gemma-2-2B (26L)", "#d62728"),
                          (llama, "Llama-3.2-1B (16L)", "#1f77b4"),
                          (qwen, "Qwen2.5-1.5B (28L)", "#2ca02c")):
        x = np.linspace(0, 1, len(prof))
        ax.plot(x, prof, "-o", ms=2.5, lw=1.2, label=name, color=c)
    ax.set(xlabel="normalised depth", ylabel="restoration\neffect",
           title="(b) Cross-model layer profiles (misleading regime)")
    ax.legend(frameon=False)
    fig.tight_layout(h_pad=1.0)
    fig.savefig(F / "fig2_layer_profiles.pdf", bbox_inches="tight")
    fig.savefig(F / "fig2_layer_profiles.png", bbox_inches="tight")
    plt.close(fig)


def fig4():
    randr = json.load(open(R / "e5_interventions/gemma-2-2b/random1.json"))["results"]
    randcap = json.load(open(R / "e5_interventions/gemma-2-2b/capability_unified.json"))["configs"]
    r1 = {r["config"]: r for r in json.load(open(R / "e5_interventions/gemma-2-2b/dose_response.json"))["results"]}
    r2 = {r["config"]: r for r in json.load(open(R / "e5_interventions/gemma-2-2b/round2.json"))["results"]}
    cap = dict(json.load(open(R / "e5_interventions/gemma-2-2b/capability.json"))["results"])
    cap.update(json.load(open(R / "e5_interventions/gemma-2-2b/capability_spec1.json"))["results"])

    fig, axes = plt.subplots(2, 1, figsize=(COL, 4.25))
    ax = axes[0]
    base = r1["none"]
    ax.scatter(base["follow_rate"], base["truth_rate"], marker="*", s=140, c="k", label="none", zorder=5)
    # numbered markers + two-column key (matches panel b style)
    pts = [("none", r1["none"])]
    pts += [(f"top-{k}", r1[f"top-{k}"]) for k in ("1", "2", "5", "10", "20")]
    pts += [(name, r2[name]) for name in ("spec1", "spec3", "mean2", "mean5", "meanspec3")]
    pts += [("topnorm-5", r1["topnorm-5"])]
    pair_off = {"top-2": (-2, 7), "mean2": (2, -13),
                "top-5": (-2, 7), "mean5": (2, -13),
                "spec3": (-2, 7), "meanspec3": (2, -13)}
    for i, (name, r) in enumerate(pts, start=1):
        col = "#d62728" if name == "spec1" else ("#7f7f7f" if name == "topnorm-5" else "#1f77b4")
        ax.scatter(r["follow_rate"], r["truth_rate"], c=col, s=36,
                   zorder=6 if name == "spec1" else 3)
        off = pair_off.get(name, (-11, -8) if name == "topnorm-5" else (4, -8))
        ax.annotate(str(i), (r["follow_rate"], r["truth_rate"]), fontsize=7,
                    fontweight="bold", xytext=off, textcoords="offset points")
    # random-1 baselines: uniform (circle) and late-layer (triangle)
    ax.scatter([x["follow_rate"] for x in randr[:5]],
               [x["truth_rate"] for x in randr[:5]], c="#2ca02c", s=26,
               marker="o", alpha=0.9, zorder=2, label="random-1 (uniform)")
    ax.scatter([x["follow_rate"] for x in randr[5:]],
               [x["truth_rate"] for x in randr[5:]], c="#2ca02c", s=26,
               marker="^", alpha=0.9, zorder=2, label="random-1 (late)")
    ax.legend(frameon=False, fontsize=7, loc="lower left")
    key_l = "1  none\n2  top-1\n3  top-2\n4  top-5\n5  top-10\n6  top-20"
    key_r = "7  spec1\n8  spec3\n9  mean2\n10 mean5\n11 meanspec3\n12 topnorm-5"
    ax.text(0.60, 0.03, key_l, transform=ax.transAxes, fontsize=7.5,
            va="bottom", ha="left", linespacing=1.5)
    ax.text(0.83, 0.03, key_r, transform=ax.transAxes, fontsize=7.5,
            va="bottom", ha="left", linespacing=1.5)
    ax.set(xlabel="context-following rate (hallucination)", ylabel="truth rate",
           title="(a) Suite trade-off (held-out)", ylim=(0.12, 0.95))

    ax = axes[1]
    xs, ys, labels = [], [], []
    for name in ("none", "spec1", "top-2", "top-5", "spec3", "mean5"):
        rate = r1.get(name) or r2.get(name)
        xs.append(rate["follow_rate"])
        ys.append(randcap[name]["mmlu_acc"] / randcap["none"]["mmlu_acc"])
        labels.append(name)
    ax.scatter(xs, ys, s=36, zorder=4)
    # numbered markers + side key (labels collide when points cluster)
    names = ("none", "spec1", "top-2", "top-5", "spec3", "mean5", "rand")
    b_dig = {1: (4, 4), 2: (-2, -12), 3: (6, 3), 4: (-4, 5), 5: (5, -11), 6: (5, 4)}
    for i, (x, y) in enumerate(zip(xs, ys), start=1):
        ax.annotate(str(i), (x, y), fontsize=7, fontweight="bold",
                    xytext=b_dig[i], textcoords="offset points")
    ppl = {n: randcap[n]["wiki_ppl"] / randcap["none"]["wiki_ppl"] for n in names[:6]}
    key = "\n".join([f"{i}: {n} (PPL x{ppl[n]:.2f})"
                     for i, n in enumerate(names[:6], start=1)])
    ax.text(0.97, 0.05, key, transform=ax.transAxes, fontsize=7.5,
            ha="right", va="bottom", linespacing=1.5)
    for i, x in enumerate(randr):
        k = ("rand-" if i < 5 else "laterand-") + x["head"].replace(".", "")
        ax.scatter(x["follow_rate"], randcap[k]["mmlu_acc"] / randcap["none"]["mmlu_acc"],
                   c="#2ca02c", s=22, alpha=0.85)
    ax.axhline(1.0, ls="--", c="gray", lw=0.8)
    ax.set(xlabel="context-following rate (hallucination)", ylabel="MMLU retention",
           title="(b) Capability cost", xlim=(0, 0.36), ylim=(0.64, 1.03))
    fig.tight_layout(h_pad=1.0)
    fig.savefig(F / "fig4_tradeoff.pdf", bbox_inches="tight")
    fig.savefig(F / "fig4_tradeoff.png", bbox_inches="tight")
    plt.close(fig)


def fig5():
    table = json.load(open(R / "e4_superposition/gemma-2-2b/superposition_summary.json"))["table"]
    A = np.array([r["effect_A_restore"] for r in table])
    B = np.array([r["effect_B_context_push"] for r in table])
    na = np.array([r["neutral_ratio"] if r["neutral_ratio"] is not None else 0 for r in table])

    fig, ax = plt.subplots(figsize=(COL, 2.4))
    sc = ax.scatter(B, A, c=na, cmap="coolwarm", vmin=-1.5, vmax=1.5, s=42,
                    edgecolors="k", lw=0.4)
    lim = max(A.max(), B.max()) * 1.22
    ax.plot([0, lim], [0, lim], "--", c="gray", lw=0.8, label="A = B (strongly superposed)")
    spec1 = next(r for r in table if r["head"] == "L22_H4")
    ax.scatter([spec1["effect_B_context_push"]], [spec1["effect_A_restore"]],
               marker="*", s=200, facecolors="none", edgecolors="#d62728", lw=1.4,
               label="spec1 (L22.H4)", zorder=6)
    for r in table[:3]:
        ax.annotate(r["head"], (r["effect_B_context_push"], r["effect_A_restore"]),
                    fontsize=7, xytext=(6, -10), textcoords="offset points")
    cb = fig.colorbar(sc, ax=ax, pad=0.02)
    cb.set_label("neutral ratio N/A", fontsize=7.5)
    cb.ax.tick_params(labelsize=7)
    ax.set(xlabel="context-push effect B", ylabel="restore effect A",
           title="Dual roles of influential heads (Gemma-2-2B)")
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(F / "fig5_superposition.pdf", bbox_inches="tight")
    fig.savefig(F / "fig5_superposition.png", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    fig2()
    fig4()
    fig5()
    for f in sorted(F.glob("*.png")):
        print("created:", f.name)
