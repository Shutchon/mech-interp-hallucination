#!/usr/bin/env python
"""E5d — single-head baselines + full-answer check (review 3, item 7).

1. random-1    : one uniformly random head, 5 seeds
2. late-rand-1 : one random head from late layers 20-25 (layer-matched
                 to where specialists live), 5 seeds
3. full-answer : greedy-generate up to 8 tokens on a held-out sample
                 under none/spec1 and compare the full string to the
                 target (first-token vs full-answer agreement).

Usage: python scripts/e5d_extra_baselines.py
Output: results/e5_interventions/gemma-2-2b/random1.json (+per_item),
        results/e5_interventions/gemma-2-2b/full_answer.json
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"
OUTD = R / "e5_interventions/gemma-2-2b"
ALL_HEADS = [(l, h) for l in range(26) for h in range(8)]
LATE_HEADS = [(l, h) for l in range(20, 26) for h in range(8)]


def zero_hooks(heads):
    fwd = []
    for l, h in heads:
        def mk(l=l, h=h):
            def hk(act, hook):
                act[:, :, h, :] = 0.0
                return act
            return hk
        fwd.append((f"blocks.{l}.attn.hook_z", mk()))
    return fwd


@torch.no_grad()
def main():
    from transformer_lens import HookedTransformer
    model = HookedTransformer.from_pretrained(
        "gemma-2-2b", dtype=torch.bfloat16,
        center_unembed=False, fold_ln=False, center_writing_weights=False
    ).to("cuda" if torch.cuda.is_available() else "cpu")

    suite = [json.loads(l) for l in open(R / "e1_dataset/misleading_suite.jsonl")]
    ho = [s for s in suite if s["split"] == "heldout" and s["template_idx"] == 0]
    facts = {f["case_id"]: f for f in
             (json.loads(l) for l in open(R / "e1_dataset/facts.jsonl"))}
    rng = random.Random(20261014)

    results, per = [], []
    for name, pool in (("random-1", ALL_HEADS), ("late-rand-1", LATE_HEADS)):
        for seed in range(5):
            head = rng.choice(pool)
            hooks = zero_hooks([head])
            n = nf = nt = 0
            for s in ho:
                f = facts[s["case_id"]]
                tid = int(model.to_tokens(" " + f["target_true"])[0, -1])
                did = int(model.to_tokens(" " + f["target_false"])[0, -1])
                logits = model.run_with_hooks(s["prompt_shown"], fwd_hooks=hooks)[0, -1].float()
                pred = int(logits.argmax())
                n += 1
                nf += pred == did
                nt += pred == tid
                per.append({"config": f"{name}-s{seed}", "case_id": s["case_id"],
                            "condition": "misleading",
                            "is_true": bool(pred == tid), "is_distractor": bool(pred == did)})
            results.append({"config": f"{name}-s{seed}", "head": f"L{head[0]}.H{head[1]}",
                            "follow_rate": nf / n, "truth_rate": nt / n})
            print(f"{name}-s{seed} L{head[0]}.H{head[1]}: follow {nf/n:.3f} truth {nt/n:.3f}")
    (OUTD / "random1.json").write_text(json.dumps(
        {"experiment": "E5d_random1", "results": results}, indent=1))
    with open(OUTD / "random1_per_item.jsonl", "w") as fh:
        for r in per:
            fh.write(json.dumps(r) + "\n")

    # full-answer check on 300 held-out facts under none/spec1
    fa = {}
    sample = ho[:300]
    for cfg, heads in (("none", []), ("spec1", [(22, 4)])):
        hooks = zero_hooks(heads)
        agree = first_ok = full_ok = 0
        for s in sample:
            f = facts[s["case_id"]]
            tid = int(model.to_tokens(" " + f["target_true"])[0, -1])
            did = int(model.to_tokens(" " + f["target_false"])[0, -1])
            logits = model.run_with_hooks(s["prompt_shown"], fwd_hooks=hooks)[0, -1].float()
            pred = int(logits.argmax())
            first_ok += pred == tid
            gen = model.generate(s["prompt_shown"], max_new_tokens=8,
                                 do_sample=False, verbose=False)
            txt = gen[0][len(s["prompt_shown"]):].strip().strip(".,")
            full_ok += txt.lower() == f["target_true"].lower()
            agree += (pred == tid) == (txt.lower() == f["target_true"].lower())
        fa[cfg] = {"n": len(sample), "first_token_acc": first_ok / len(sample),
                   "full_answer_acc": full_ok / len(sample),
                   "agreement": agree / len(sample)}
        print(cfg, fa[cfg])
    (OUTD / "full_answer.json").write_text(json.dumps(
        {"experiment": "E5d_full_answer", "results": fa}, indent=1))


if __name__ == "__main__":
    main()
