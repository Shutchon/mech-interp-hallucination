#!/usr/bin/env python
"""E4b — neutral-control effects in float32 (review 9, item 4).

Recomputes, for the key heads, in float32:
  - signed mean N (with the |gap| < 1e-6 -> 0 convention)
  - mean |raw dLD_N| and median |raw dLD_N|
  - whether L22.H4 still passes |N| < 0.01

Usage: python scripts/e4b_neutral_fp32.py
Output: results/e4_superposition/gemma-2-2b/neutral_fp32.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"
OUT = R / "e4_superposition/gemma-2-2b/neutral_fp32.json"
HEADS = [(22, 4), (25, 4), (22, 0), (20, 3)]
N_FACTS = 150


@torch.no_grad()
def main():
    from transformer_lens import HookedTransformer
    model = HookedTransformer.from_pretrained(
        "gemma-2-2b", dtype=torch.float32,
        center_unembed=False, fold_ln=False, center_writing_weights=False
    ).to("cuda" if torch.cuda.is_available() else "cpu")

    facts = [json.loads(l) for l in open(R / "e1_dataset/facts.jsonl")
             if json.loads(l)["split"] == "dev"][:N_FACTS]
    suite = {s["case_id"]: s for s in
             (json.loads(l) for l in open(R / "e1_dataset/misleading_suite.jsonl"))
             if s["template_idx"] == 0 and s["split"] == "dev"
             and s["condition"] == "misleading"}

    out = {"experiment": "E4b_neutral_fp32", "dtype": "float32",
           "n_facts": len(facts), "heads": {}}
    for (l, h) in HEADS:
        hook = f"blocks.{l}.attn.hook_z"
        signed, raws = [], []
        for f in facts:
            s = suite[f["case_id"]]
            neu_p = s["prompt_shown"].replace(f["target_false"], "unknown")
            tid = int(model.to_tokens(" " + f["target_true"])[0, -1])
            did = int(model.to_tokens(" " + f["target_false"])[0, -1])
            cln = len(model.to_tokens(s["prompt_shown"])[0])
            nln = len(model.to_tokens(neu_p)[0])
            _, cache = model.run_with_cache(
                s["prompt_shown"], names_filter=lambda n: n == hook)
            acc = []
            for k in range(3):
                cpos, npos = cln - 3 + k, nln - 3 + k
                val = cache[hook][0, cpos, h, :]
                def hk(act, hook, v=val, p=npos, h_=h):
                    act[0, p, h_, :] = v
                    return act
                lp = model.run_with_hooks(neu_p, fwd_hooks=[(hook, hk)])[0, -1].float()
                lb = model(neu_p)[0, -1].float()
                acc.append(((lp[tid] - lp[did]) - (lb[tid] - lb[did])).item())
            del cache
            raws.extend(acc)
            clean_ld = model(s["prompt_shown"])[0, -1].float()
            neu_ld = model(neu_p)[0, -1].float()
            D_n = (clean_ld[tid] - clean_ld[did] - (neu_ld[tid] - neu_ld[did])).item()
            if abs(D_n) < 1e-6:
                signed.append(0.0)
            else:
                signed.append(float(np.mean(acc)) / D_n)
        raws = np.array(raws)
        out["heads"][f"L{l}.H{h}"] = {
            "N_signed_mean": float(np.mean(signed)),
            "mean_abs_raw_dLD_N": float(np.abs(raws).mean()),
            "median_abs_raw_dLD_N": float(np.median(np.abs(raws))),
            "passes_|N|<0.01": bool(abs(np.mean(signed)) < 0.01),
        }
        print(f"L{l}.H{h}:", out["heads"][f"L{l}.H{h}"])
    OUT.write_text(json.dumps(out, indent=1))
    print("written", OUT)


if __name__ == "__main__":
    main()
