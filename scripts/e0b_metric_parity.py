#!/usr/bin/env python
"""E0b — Metric-level precision parity (review R7).

Compares bf16 (experiment precision) against fp32 on the SAME sample at
the level of the metrics we actually report:
  - logit differences LD(clean/misleading/neutral)
  - top-20 head restore effects A (patching, e4-style)
  - Spearman rank correlation of head effects between precisions

Usage: python scripts/e0b_metric_parity.py [--n-facts 300]
Output: results/e0_parity/metric_parity.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"
OUT = R / "e0_parity" / "metric_parity.json"

TL_FLAGS = dict(center_unembed=False, fold_ln=False, center_writing_weights=False)


def load_model(dtype):
    from transformer_lens import HookedTransformer
    m = HookedTransformer.from_pretrained("gemma-2-2b", dtype=dtype, **TL_FLAGS)
    return m.to("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def run_sample(model, facts, suite, top_heads, n):
    """Returns dict: per-condition LD arrays + per-head A effects."""
    ids_by_case = {s["case_id"]: s for s in suite if s["template_idx"] == 0 and s["split"] == "dev" and s["condition"] == "misleading"}
    dev_facts = [f for f in facts if f["split"] == "dev"][:n]
    lds = {"clean": [], "mis": [], "neu": []}
    head_eff = {h: [] for h in top_heads}
    for f in dev_facts:
        cid = f["case_id"]
        s = ids_by_case.get(cid)
        if s is None:
            continue
        tid = int(model.to_tokens(" " + f["target_true"])[0, -1])
        did = int(model.to_tokens(" " + f["target_false"])[0, -1])
        clean_p = f["prompt"]
        mis_p = s["prompt_shown"]
        neu_p = mis_p.replace(f["target_false"], "unknown")
        logits_c = model(clean_p)[0, -1].float()
        logits_m = model(mis_p)[0, -1].float()
        logits_n = model(neu_p)[0, -1].float()
        ld_c = (logits_c[tid] - logits_c[did]).item()
        ld_m = (logits_m[tid] - logits_m[did]).item()
        ld_n = (logits_n[tid] - logits_n[did]).item()
        lds["clean"].append(ld_c); lds["mis"].append(ld_m); lds["neu"].append(ld_n)
        D = ld_c - ld_m
        if abs(D) < 1e-6:
            for h in top_heads:
                head_eff[h].append(0.0)
            continue
        cln = len(model.to_tokens(clean_p)[0]); mln = len(model.to_tokens(mis_p)[0])
        _, cache_c = model.run_with_cache(clean_p, names_filter=lambda n: n.endswith("attn.hook_z"))
        for (l, hd) in top_heads:
            hook = f"blocks.{l}.attn.hook_z"
            acc = []
            for k in range(3):
                cpos, mpos = cln - 3 + k, mln - 3 + k
                val = cache_c[hook][0, cpos, hd, :]
                def hk(act, hook, v=val, p=mpos, h_=hd):
                    act[0, p, h_, :] = v
                    return act
                lp = model.run_with_hooks(mis_p, fwd_hooks=[(hook, hk)])[0, -1].float()
                acc.append(((lp[tid] - lp[did]).item() - ld_m) / D)
            head_eff[(l, hd)].append(float(np.mean(acc)))
        del cache_c
    return {k: np.array(v) for k, v in lds.items()}, {h: np.array(v) for h, v in head_eff.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-facts", type=int, default=300)
    args = ap.parse_args()

    facts = [json.loads(l) for l in open(R / "e1_dataset/facts.jsonl")]
    suite = [json.loads(l) for l in open(R / "e1_dataset/misleading_suite.jsonl")]
    dose = json.load(open(R / "e5_interventions/gemma-2-2b/dose_response.json"))
    rank = dose["dev_head_ranking"][:20]
    top_heads = [tuple(map(int, h_.replace("L", "").split("_H"))) for h_ in rank]

    dev_facts = [f for f in facts if f["split"] == "dev"][: args.n_facts]

    out = {"experiment": "E0b_metric_parity", "n_facts": len(dev_facts), "top_heads": rank}

    # fp32 first
    m32 = load_model(torch.float32)
    lds32, eff32 = run_sample(m32, facts, suite, top_heads, args.n_facts)
    del m32; torch.cuda.empty_cache()
    m16 = load_model(torch.bfloat16)
    lds16, eff16 = run_sample(m16, facts, suite, top_heads, args.n_facts)
    del m16; torch.cuda.empty_cache()

    n = min(len(lds32["clean"]), len(lds16["clean"]))
    out["LD"] = {}
    for cond in ("clean", "mis", "neu"):
        a, b = lds32[cond][:n], lds16[cond][:n]
        out["LD"][cond] = {"mean_abs_diff": float(np.abs(a - b).mean()),
                           "max_abs_diff": float(np.abs(a - b).max()),
                           "mean_abs_ld": float(np.abs(a).mean())}
    common = [h for h in top_heads if len(eff32[h]) and len(eff16[h])]
    m_eff32 = np.array([eff32[h].mean() for h in common])
    m_eff16 = np.array([eff16[h].mean() for h in common])
    out["head_effects"] = {
        "spearman_fp32_vs_bf16": float(spearmanr(m_eff32, m_eff16).statistic),
        "mean_abs_effect_diff": float(np.abs(m_eff32 - m_eff16).mean()),
        "mean_abs_effect": float(np.abs(m_eff32).mean()),
        "max_abs_effect_diff": float(np.abs(m_eff32 - m_eff16).max()),
        "rank_of_L22H4_fp32": int(np.argsort(-m_eff32).tolist().index(
            next(i for i, h in enumerate(common) if h == (22, 4)))),
        "rank_of_L22H4_bf16": int(np.argsort(-m_eff16).tolist().index(
            next(i for i, h in enumerate(common) if h == (22, 4)))),
    }
    OUT.write_text(json.dumps(out, indent=1))
    print(json.dumps(out["LD"], indent=1))
    print(json.dumps(out["head_effects"], indent=1))


if __name__ == "__main__":
    main()
