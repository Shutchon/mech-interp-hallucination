#!/usr/bin/env python
"""E5b — MMLU per-question records for paired tests (review R8).

Runs the same fixed 2,000-question MMLU subset as e5_mmlu_ppl.py for the
requested configs and dumps per-question correctness so McNemar can be
computed on paired questions.

Usage: python scripts/e5b_mmlu_perq.py [--configs none,spec1]
Output: results/e5_interventions/gemma-2-2b/mmlu_perq.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"
OUT = R / "e5_interventions/gemma-2-2b/mmlu_perq.jsonl"


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


CONFIG_HEADS = {
    "none": [],
    "spec1": [(22, 4)],
    "spec3": [(22, 4), (22, 0), (20, 3)],
    "top-2": [(25, 4), (22, 4)],
    "top-5": [(25, 4), (22, 4), (22, 0), (20, 3), (23, 7)],
}


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", default="none,spec1")
    ap.add_argument("--n-mmlu", type=int, default=2000)
    args = ap.parse_args()
    from transformer_lens import HookedTransformer
    model = HookedTransformer.from_pretrained(
        "gemma-2-2b", dtype=torch.bfloat16,
        center_unembed=False, fold_ln=False, center_writing_weights=False
    ).to("cuda" if torch.cuda.is_available() else "cpu")

    from datasets import load_dataset
    mmlu = load_dataset("cais/mmlu", "all", split="test")
    rng = random.Random(0)
    idx = rng.sample(range(len(mmlu)), min(args.n_mmlu, len(mmlu)))
    letters = ["A", "B", "C", "D"]
    tok_ids = {L: model.to_single_token(L) for L in letters}

    with open(OUT, "w") as fh:
        for cfg in args.configs.split(","):
            hooks = zero_hooks(CONFIG_HEADS[cfg])
            n_ok = 0
            for qi, i in enumerate(idx):
                row = mmlu[i]
                prompt = (f"The following are multiple choice questions "
                          f"(with answers) about {row['subject']}.\n\n"
                          f"{row['question']}\n"
                          + "\n".join(f"{L}. {c}" for L, c in zip(letters, row["choices"]))
                          + "\nAnswer:")
                logits = model.run_with_hooks(prompt, fwd_hooks=hooks)[0, -1].float()
                pred = letters[int(torch.tensor([logits[tok_ids[L]] for L in letters]).argmax())]
                ok = pred == row["answer"]
                n_ok += ok
                fh.write(json.dumps({"config": cfg, "q_idx": i,
                                     "pred": pred, "answer": row["answer"],
                                     "is_correct": bool(ok)}) + "\n")
            print(f"{cfg}: {n_ok}/{len(idx)} = {n_ok/len(idx):.4f}")
    print("written", OUT)


if __name__ == "__main__":
    main()
