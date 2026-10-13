#!/usr/bin/env python3
"""Round-2 revision analyses (review R4, R8, R12).

R4:  raw delta-LD reporting, denominator distributions, sensitivity of N
     to the near-zero denominator threshold, and a directly interpretable
     raw-effect ratio (neutral-run vs misleading-run delta-LD, same units).
R8:  fact-clustered bootstrap CIs, accuracy-side exact McNemar for spec1,
     random-baseline seed spread, spec1 neutral-accuracy CI.
R12: table rows for top-10 / mean2 / meanspec3 with CIs.

Inputs (local results/): e4 records + e3 misleading_fine + e5 per-item.
Output: results/analysis/round2_stats.json
"""
import json
import math
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import binom, spearmanr

ROOT = Path(__file__).resolve().parents[1] / "results"
OUT = ROOT / "analysis" / "round2_stats.json"
rng = random.Random(20261013)

# ---------------------------------------------------------------- load data
e4 = [json.loads(l) for l in open(ROOT / "e4_superposition/gemma-2-2b/records.jsonl")]
e3f = [json.loads(l) for l in open(ROOT / "e3_sweep/gemma-2-2b/misleading_fine.jsonl")]
per_item = [json.loads(l) for l in open(ROOT / "e5_interventions/gemma-2-2b/per_item.jsonl")]
round2 = [json.loads(l) for l in open(ROOT / "e5_interventions/gemma-2-2b/round2_per_item.jsonl")]
all_items = per_item + round2

# ------------------------------------------------- R4: normalization audit
KEY_HEADS = [(22, 4), (25, 4), (22, 0), (20, 3), (25, 6)]

# aggregate E4 records per head (records already per case*head; one pos)
per_head = defaultdict(list)
for r in e4:
    per_head[(r["layer"], r["head"])].append(r)

# A-side raw delta-LD from E3 fine (clean value patched into misleading run)
per_head_A = defaultdict(list)
for r in e3f:
    per_head_A[(r["layer"], r["head"])].append(r)

audit = {}
for (layer, head) in KEY_HEADS:
    recs = per_head[(layer, head)]
    fids = per_head_A[(layer, head)]
    if not recs:
        continue
    clean = np.array([r["clean_ld"] for r in recs], dtype=np.float64)
    mis = np.array([r["mis_ld"] for r in recs], dtype=np.float64)
    neu = np.array([r["neu_ld"] for r in recs], dtype=np.float64)
    b_ld = np.array([r["b_ld"] for r in recs], dtype=np.float64)
    n_ld = np.array([r["n_ld"] for r in recs], dtype=np.float64)
    d_mis = clean - mis                # denominator of A and B
    d_neu = clean - neu                # denominator of N
    raw_b = clean - b_ld               # context-push, raw logits
    raw_n = n_ld - neu                 # neutral-run effect, raw logits
    raw_a = None
    if fids:
        raw_a = np.array([f["patched_ld"] - f["corrupted_ld"] for f in fids])
    name = f"L{layer}.H{head}"
    entry = {
        "n_cases_B": len(recs),
        "n_cases_A": len(fids) if fids else 0,
        "raw_dLD_B_mean": float(raw_b.mean()),
        "raw_dLD_N_mean": float(raw_n.mean()),
        "raw_ratio_N_over_B": float(raw_n.mean() / raw_b.mean()) if raw_b.mean() != 0 else None,
        "raw_dLD_A_mean": float(raw_a.mean()) if raw_a is not None else None,
        "D_mis_q": [float(q) for q in np.percentile(d_mis, [5, 25, 50, 75, 95])],
        "D_neu_q": [float(q) for q in np.percentile(d_neu, [5, 25, 50, 75, 95])],
        "D_mis_lt_1e-6": int((np.abs(d_mis) < 1e-6).sum()),
        "D_neu_lt_1e-6": int((np.abs(d_neu) < 1e-6).sum()),
        "D_neu_lt_0p5": int((np.abs(d_neu) < 0.5).sum()),
        "D_neu_lt_1": int((np.abs(d_neu) < 1.0).sum()),
    }
    # N under different near-zero denominator thresholds
    sens = {}
    for eps in [1e-6, 0.5, 1.0, 2.0]:
        eff = np.where(np.abs(d_neu) < eps, 0.0, raw_n / np.where(np.abs(d_neu) < eps, 1.0, d_neu))
        sens[f"eps={eps}"] = float(eff.mean())
    entry["N_sensitivity"] = sens
    audit[name] = entry

# global denominator health over all E4 cases (not just key heads)
all_clean, all_mis, all_neu = [], [], []
for r in e4:
    all_clean.append(r["clean_ld"]); all_mis.append(r["mis_ld"]); all_neu.append(r["neu_ld"])
D_mis = np.array(all_clean) - np.array(all_mis)
D_neu = np.array(all_clean) - np.array(all_neu)
denom_health = {
    "D_mis": {"median": float(np.median(D_mis)), "p5": float(np.percentile(D_mis, 5)),
              "frac_lt_0p5": float((np.abs(D_mis) < 0.5).mean()), "frac_lt_1e-6": float((np.abs(D_mis) < 1e-6).mean())},
    "D_neu": {"median": float(np.median(D_neu)), "p5": float(np.percentile(D_neu, 5)),
              "frac_lt_0p5": float((np.abs(D_neu) < 0.5).mean()), "frac_lt_1e-6": float((np.abs(D_neu) < 1e-6).mean())},
}

# ------------------------------------------- R8: fact-clustered bootstrap
items_by_config = defaultdict(list)
for r in all_items:
    items_by_config[r["config"]].append(r)

def rates(rows):
    mis = [r for r in rows if r["condition"] == "misleading"]
    neu = [r for r in rows if r["condition"] == "neutral"]
    return {
        "follow": np.mean([r["is_distractor"] for r in mis]),
        "truth": np.mean([r["is_true"] for r in mis]),
        "neutral_truth": np.mean([r["is_true"] for r in neu]),
    }

def clustered_bootstrap(rows, n_boot=10000):
    by_fact = defaultdict(list)
    for r in rows:
        by_fact[r["case_id"]].append(r)
    facts = sorted(by_fact)
    out = {k: [] for k in ("follow", "truth", "neutral_truth")}
    for _ in range(n_boot):
        sample = []
        for f in rng.choices(facts, k=len(facts)):
            sample.extend(by_fact[f])
        rr = rates(sample)
        for k in out:
            out[k].append(rr[k])
    return {k: [float(np.percentile(v, [2.5, 97.5])[i]) for i in (0, 1)] for k, v in out.items()}

configs = ["none", "spec1", "spec3", "top-1", "top-2", "top-5", "top-10",
           "top-20", "topnorm-5", "mean2", "mean5", "meanspec3"]
boot = {}
for cfg in configs:
    rows = items_by_config[cfg]
    boot[cfg] = {"point": rates(rows), "ci95_clustered": clustered_bootstrap(rows)}

# accuracy-side exact McNemar for spec1 vs none (misleading condition truth flips)
base = {r["case_id"]: r for r in items_by_config["none"] if r["condition"] == "misleading"}
spec = {r["case_id"]: r for r in items_by_config["spec1"] if r["condition"] == "misleading"}
common = sorted(set(base) & set(spec))
flips = [(base[c]["is_true"], spec[c]["is_true"]) for c in common]
b_acc = sum(1 for x, y in flips if not x and y)   # none wrong -> spec1 right
c_acc = sum(1 for x, y in flips if x and not y)   # none right -> spec1 wrong
p_acc = float(binom.cdf(min(b_acc, c_acc), b_acc + c_acc, 0.5) * 2) if b_acc + c_acc else 1.0
acc_test = {"b_gained": b_acc, "c_lost": c_acc, "exact_p": p_acc}

# neutral-accuracy side test for spec1 (collateral check)
base_n = {r["case_id"]: r for r in items_by_config["none"] if r["condition"] == "neutral"}
spec_n = {r["case_id"]: r for r in items_by_config["spec1"] if r["condition"] == "neutral"}
common_n = sorted(set(base_n) & set(spec_n))
flips_n = [(base_n[c]["is_true"], spec_n[c]["is_true"]) for c in common_n]
b_n = sum(1 for x, y in flips_n if not x and y)
c_n = sum(1 for x, y in flips_n if x and not y)
p_n = float(binom.cdf(min(b_n, c_n), b_n + c_n, 0.5) * 2) if b_n + c_n else 1.0
neu_test = {"b_gained": b_n, "c_lost": c_n, "exact_p": p_n}

# random-baseline seed spread
rand = {f"random5-s{i}": rates(items_by_config[f"random5-s{i}"]) for i in range(5)}
random_spread = {
    k: {"min": float(min(v[k2] for v in rand.values())),
        "max": float(max(v[k2] for v in rand.values())),
        "mean": float(np.mean([v[k2] for v in rand.values()]))}
    for k in ("follow", "truth") for k2 in [k]
}
random_spread = {k: random_spread[k] for k in ("follow", "truth")}

out = {
    "experiment": "analysis_round2",
    "R4_normalization": {"key_heads": audit, "denominator_health": denom_health},
    "R8_uncertainty": {"clustered_bootstrap": boot, "spec1_accuracy_mcnemar": acc_test,
                       "spec1_neutral_mcnemar": neu_test, "random_spread": random_spread,
                       "random_seeds": {k: {m: float(v[m]) for m in v} for k, v in rand.items()}},
}
OUT.write_text(json.dumps(out, indent=1))
print("written", OUT)
print("\n--- spec1 accuracy McNemar:", acc_test)
print("--- spec1 neutral McNemar:", neu_test)
print("\n--- key heads raw ratio (N/B):")
for n, e in audit.items():
    print(f"{n}: rawB={e['raw_dLD_B_mean']:.3f} rawN={e['raw_dLD_N_mean']:.3f} "
          f"ratio={e['raw_ratio_N_over_B'] if e['raw_ratio_N_over_B'] is None else round(e['raw_ratio_N_over_B'],3)} "
          f"N_sens={ {k: round(v,3) for k,v in e['N_sensitivity'].items()} }")
print("\n--- denominator health:", json.dumps(denom_health, indent=1))
print("\n--- clustered CI spec1:", boot["spec1"])
print("\n--- random spread:", random_spread)
