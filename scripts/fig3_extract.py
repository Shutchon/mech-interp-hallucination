#!/usr/bin/env python
"""Fig3 data extraction v2 (review R6).

For 5 heads (3 specialists, 2 dual-role) on N misleading prompts:
  - mean attention FROM the answer position TO: BOS / distractor tokens /
    other (with spread over prompts)
  - mean full pattern over prompts for one qualitative example block
  - token strings for axis labels

Usage: python scripts/fig3_extract.py [--n-prompts 32]
Output: results/analysis/fig3_attention_v2.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"
OUT = R / "analysis" / "fig3_attention_v2.json"
HEADS = [(22, 4, "spec"), (22, 0, "spec"), (20, 3, "spec"),
         (25, 4, "dual"), (25, 6, "dual")]


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-prompts", type=int, default=32)
    args = ap.parse_args()
    from transformer_lens import HookedTransformer
    model = HookedTransformer.from_pretrained(
        "gemma-2-2b", dtype=torch.bfloat16,
        center_unembed=False, fold_ln=False, center_writing_weights=False
    ).to("cuda" if torch.cuda.is_available() else "cpu")

    facts = [json.loads(l) for l in open(R / "e1_dataset/facts.jsonl")
             if json.loads(l)["split"] == "heldout"][: args.n_prompts]
    suite = {s["case_id"]: s for s in
             (json.loads(l) for l in open(R / "e1_dataset/misleading_suite.jsonl"))
             if s["template_idx"] == 0 and s["split"] == "heldout" and s["condition"] == "misleading"}

    stats = {f"L{l}.H{h}": {"bos": [], "distractor": [], "other": [], "distr_read": []}
             for l, h, _ in HEADS}
    example = None
    for f in facts:
        s = suite[f["case_id"]]
        d = f["target_false"]
        toks = model.to_tokens(s["prompt_shown"])[0]
        tok_strs = [model.to_string(int(t)) for t in toks]
        _, cache = model.run_with_cache(
            s["prompt_shown"],
            names_filter=lambda n: n.endswith("attn.hook_pattern"))
        # locate distractor tokens (first occurrence of the distractor string)
        d_low = d.lower()
        d_tok_idx = [i for i, ts in enumerate(tok_strs) if ts.strip().lower() == d_low]
        ans = len(toks) - 1
        for l, h, _ in HEADS:
            pat = cache[f"blocks.{l}.attn.hook_pattern"][0, h].float()  # [q,k]
            row = pat[ans].cpu().numpy()
            bos_w = float(row[0])
            d_w = float(row[d_tok_idx].sum()) if d_tok_idx else 0.0
            other = float(row.sum() - bos_w - d_w)
            stats[f"L{l}.H{h}"]["bos"].append(bos_w)
            stats[f"L{l}.H{h}"]["distractor"].append(d_w)
            stats[f"L{l}.H{h}"]["other"].append(other)
            if d_tok_idx:
                first_d = min(d_tok_idx)
                rows = pat[first_d + 1:, :].cpu().numpy()
                stats[f"L{l}.H{h}"]["distr_read"].append(float(rows[:, d_tok_idx].sum(axis=1).mean()))
        if example is None and not d_tok_idx:
            d_tok_idx = [i for i, ts in enumerate(tok_strs)
                         if d_low in ts.strip().lower() and i > 0]
        if example is None and d_tok_idx:
            example = {
                "prompt": s["prompt_shown"], "tokens": tok_strs,
                "distractor_idx": d_tok_idx, "true": f["target_true"], "distractor": d,
                "patterns": {f"L{l}.H{h}": cache[f"blocks.{l}.attn.hook_pattern"][0, h].float().cpu().numpy().tolist()
                             for l, h, _ in HEADS},
            }
        del cache
    summary = {}
    for k, v in stats.items():
        summary[k] = {m: ({"mean": float(np.mean(x)), "sd": float(np.std(x)),
                           "min": float(np.min(x)), "max": float(np.max(x))}
                          if len(x) else {"mean": None, "sd": None,
                                          "min": None, "max": None})
                      for m, x in v.items()}
    out = {"experiment": "fig3_v2", "n_prompts": len(facts), "answer_position": "last",
           "summary": summary, "example": example}
    OUT.write_text(json.dumps(out))
    print("written", OUT, "prompts:", len(facts))
    for k, v in summary.items():
        print(f"{k}: BOS={v['bos']['mean']:.2f}±{v['bos']['sd']:.2f} "
              f"distr={v['distractor']['mean']:.2f} other={v['other']['mean']:.2f}")


if __name__ == "__main__":
    main()
