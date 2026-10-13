#!/usr/bin/env python
"""E5c — Correct-context control (review R11).

Asks whether the specialist knockout breaks the model's ability to USE
context that supports the true answer. Suite: T0 template asserting the
TRUE target ("Earlier you learned that the correct answer is {true}. {prompt}")
on held-out facts, evaluated under none / spec1 / spec3 / top-5 / mean5.

Usage: python scripts/e5c_correct_context.py
Output: results/e5_interventions/gemma-2-2b/correct_context.json + per_item.jsonl
"""
from __future__ import annotations

import json
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"
OUTD = R / "e5_interventions/gemma-2-2b"

T0 = "Earlier you learned that the correct answer is {a}. {p}"

CONFIG_HEADS = {
    "none": [],
    "spec1": [(22, 4)],
    "spec3": [(22, 4), (22, 0), (20, 3)],
    "top-5": [(25, 4), (22, 4), (22, 0), (20, 3), (23, 7)],
}


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

    facts = [json.loads(l) for l in open(R / "e1_dataset/facts.jsonl")
             if json.loads(l)["split"] == "heldout"]
    per_path = OUTD / "correct_context_per_item.jsonl"
    results = []
    with open(per_path, "w") as fh:
        for cfg in ("none", "spec1", "spec3", "top-5"):
            hooks = zero_hooks(CONFIG_HEADS[cfg])
            n = n_true = n_dist = 0
            for f in facts:
                prompt = T0.format(a=f["target_true"], p=f["prompt"])
                tid = int(model.to_tokens(" " + f["target_true"])[0, -1])
                did = int(model.to_tokens(" " + f["target_false"])[0, -1])
                logits = model.run_with_hooks(prompt, fwd_hooks=hooks)[0, -1].float()
                pred = int(logits.argmax())
                n += 1
                n_true += pred == tid
                n_dist += pred == did
                fh.write(json.dumps({"config": cfg, "case_id": f["case_id"],
                                     "is_true": bool(pred == tid),
                                     "is_distractor": bool(pred == did)}) + "\n")
            results.append({"config": cfg, "n": n,
                            "true_rate": n_true / n, "distractor_rate": n_dist / n})
            print(f"{cfg}: truth {n_true/n:.3f}")
    (OUTD / "correct_context.json").write_text(json.dumps(
        {"experiment": "E5c_correct_context", "results": results}, indent=1))
    print("written", OUTD / "correct_context.json")


if __name__ == "__main__":
    main()
