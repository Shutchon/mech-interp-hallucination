#!/usr/bin/env python
"""E5g — unified capability run (review 5, items 1-2).

ONE run, ONE fixed subset, per-question records for everything:
  - Table-5 configs: none, top-2, top-5, spec1, spec3, mean5
  - all ten random-1 / late-rand-1 heads
Each config: MMLU (2,000-question subset, per-question) + WikiText-2 PPL
(100 windows x 1024 tokens, non-overlapping). Single-source numbers for
accuracy, retention, CI, and McNemar.

Usage: python scripts/e5g_capability_unified.py
Output: results/e5_interventions/gemma-2-2b/capability_unified.json
"""
from __future__ import annotations

import glob
import json
import random
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"
OUT = R / "e5_interventions/gemma-2-2b/capability_unified.json"

CONFIGS = [
    ("none", []),
    ("top-2", [(25, 4), (22, 4)]),
    ("top-5", [(25, 4), (22, 4), (22, 0), (20, 3), (23, 7)]),
    ("spec1", [(22, 4)]),
    ("spec3", [(22, 4), (22, 0), (20, 3)]),
    ("mean5", [(25, 4), (22, 4), (22, 0), (20, 3), (23, 7)]),
    ("rand-L23H2", [(23, 2)]), ("rand-L17H3", [(17, 3)]),
    ("rand-L5H4", [(5, 4)]), ("rand-L21H5", [(21, 5)]),
    ("rand-L24H4", [(24, 4)]),
    ("laterand-L21H1", [(21, 1)]), ("laterand-L23H6", [(23, 6)]),
    ("laterand-L24H5", [(24, 5)]), ("laterand-L24H0", [(24, 0)]),
    ("laterand-L22H5", [(22, 5)]),
]


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

    hub = Path.home() / "hf_cache/hub"
    mmlu_pq = glob.glob(str(hub / "datasets--cais--mmlu/snapshots/*/all/test-*.parquet"))[0]
    mmlu = pd.read_parquet(mmlu_pq).to_dict("records")
    rng = random.Random(0)
    idx = rng.sample(range(len(mmlu)), 2000)
    letters = ["A", "B", "C", "D"]
    letter_ids = [int(model.to_tokens(" " + L)[0, -1]) for L in letters]
    prompts = []
    for i in idx:
        row = mmlu[i]
        prompts.append((f"The following are multiple choice questions "
                        f"(with answers) about {row['subject'].replace('_', ' ')}.\n\n"
                        f"{row['question']}\n"
                        + "\n".join(f"{L}. {c}" for L, c in zip(letters, row["choices"]))
                        + "\nAnswer:", row["answer"]))

    wiki_pq = glob.glob(str(hub / "datasets--wikitext/snapshots/*/wikitext-2-raw-v1/test-*.parquet"))
    text = "\n".join(t for t in pd.read_parquet(wiki_pq[0])["text"] if t.strip())
    all_ids = model.to_tokens(text)[0]

    out = {"experiment": "E5g_capability_unified", "n_mmlu": 2000,
           "n_windows": 100, "configs": {}}
    for name, heads in CONFIGS:
        hooks = zero_hooks(heads)
        per_q, correct = [], 0
        for p, ans in prompts:
            logits = model.run_with_hooks(p, fwd_hooks=hooks)[0, -1].float()
            pred = int(torch.tensor([logits[j] for j in letter_ids]).argmax())
            ok = pred == ans
            correct += ok
            per_q.append(bool(ok))
        nll, n_tok, n_win = 0.0, 0, 0
        for w0 in range(0, min(len(all_ids) - 1025, 100 * 1024), 1024):
            window = all_ids[w0:w0 + 1024]
            logits = model.run_with_hooks(window.unsqueeze(0), fwd_hooks=hooks)[0].float()
            lp = torch.log_softmax(logits[:-1], -1)
            nll += -lp[torch.arange(len(window) - 1), window[1:]].sum().item()
            n_tok += len(window) - 1
            n_win += 1
        ppl = torch.exp(torch.tensor(nll / n_tok)).item()
        out["configs"][name] = {"mmlu_acc": correct / len(prompts),
                                "per_question": per_q, "wiki_ppl": ppl}
        print(f"{name:16} MMLU {correct/len(prompts):.4f}  PPL {ppl:.2f}")
    OUT.write_text(json.dumps(out))
    print("written", OUT)


if __name__ == "__main__":
    main()
