#!/usr/bin/env python
"""Post-VM analyses (review R8, R11).

Reads the outputs of the round-2 VM jobs and computes:
  1. MMLU paired exact McNemar (none vs spec1) from per-question records
  2. Fact-clustered bootstrap CIs for Llama/Qwen held-out interventions
  3. Correct-context control summary (+ McNemar vs none)

Usage: python3 scripts/analysis_round2b.py   (after VM jobs finish)
Output: results/analysis/round2b_stats.json
"""
from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import binom

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"
rng = random.Random(20261013)


def exact_mcnemar(b, c):
    n = b + c
    return float(binom.cdf(min(b, c), n, 0.5) * 2) if n else 1.0


def load_jsonl(p):
    return [json.loads(l) for l in open(p) if l.strip()]


out = {"experiment": "analysis_round2b"}

# 1. MMLU paired test
p = R / "e5_interventions/gemma-2-2b/mmlu_perq.jsonl"
if p.exists():
    rows = load_jsonl(p)
    LET = "ABCD"
    by = defaultdict(dict)
    for r in rows:
        # recompute: pred letter vs integer answer (robust to e5b's letter/int slip)
        ok = LET.index(r["pred"]) == int(r["answer"])
        by[r["config"]][r["q_idx"]] = ok
    acc = {cfg: float(np.mean(list(v.values()))) for cfg, v in by.items()}
    common = sorted(set(by["none"]) & set(by["spec1"]))
    b = sum(1 for q in common if not by["none"][q] and by["spec1"][q])
    c = sum(1 for q in common if by["none"][q] and not by["spec1"][q])
    out["mmlu_paired"] = {"acc": acc, "n": len(common),
                          "b_spec1_right": b, "c_spec1_wrong": c,
                          "exact_p": exact_mcnemar(b, c)}

# 2. Llama/Qwen clustered bootstrap
cross = {}
for m in ("meta-llama/Llama-3.2-1B", "Qwen/Qwen2.5-1.5B"):
    p = R / "e6_cross" / m / "e5_per_item.jsonl"
    if not p.exists():
        continue
    rows = load_jsonl(p)
    by_cfg = defaultdict(list)
    for r in rows:
        by_cfg[r["config"]].append(r)
    entry = {}
    for cfg, recs in by_cfg.items():
        mis = [r for r in recs if r["condition"] == "misleading"]
        follow = np.mean([r["is_distractor"] for r in mis])
        by_fact = defaultdict(list)
        for r in mis:
            by_fact[r["case_id"]].append(r["is_distractor"])
        facts = sorted(by_fact)
        boots = []
        for _ in range(10000):
            s = [rng.choice(facts) for _ in facts]
            boots.append(np.mean([x for f in s for x in by_fact[f]]))
        lo, hi = np.percentile(boots, [2.5, 97.5])
        # McNemar vs none
        nones = {r["case_id"]: r["is_distractor"] for r in by_cfg.get("none", []) if r["condition"] == "misleading"}
        b = sum(1 for r in mis if r["case_id"] in nones and not nones[r["case_id"]] and r["is_distractor"])
        c = sum(1 for r in mis if r["case_id"] in nones and nones[r["case_id"]] and not r["is_distractor"])
        entry[cfg] = {"follow": float(follow),
                      "ci95": [float(lo), float(hi)],
                      "mcnemar_b": b, "mcnemar_c": c,
                      "exact_p": exact_mcnemar(b, c)}
    cross[m] = entry
out["cross_per_item"] = cross

# 3. correct-context control
p = R / "e5_interventions/gemma-2-2b/correct_context.json"
pp = R / "e5_interventions/gemma-2-2b/correct_context_per_item.jsonl"
if p.exists() and pp.exists():
    cc = {"results": json.load(open(p))["results"]}
    rows = load_jsonl(pp)
    by = defaultdict(dict)
    for r in rows:
        by[r["config"]][r["case_id"]] = r["is_true"]
    common = sorted(set(by["none"]) & set(by["spec1"]))
    b = sum(1 for q in common if not by["none"][q] and by["spec1"][q])
    c = sum(1 for q in common if by["none"][q] and not by["spec1"][q])
    cc["spec1_vs_none_mcnemar"] = {"b": b, "c": c, "exact_p": exact_mcnemar(b, c)}
    out["correct_context"] = cc

(R / "analysis/round2b_stats.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1)[:2500])
