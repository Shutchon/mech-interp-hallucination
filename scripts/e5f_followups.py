#!/usr/bin/env python
"""E5f — review-3 follow-ups, take 3 (no TL2.generate, no datasets lib).

1. Full-answer check: manual 8-step greedy generation under hooks; the
   completion must start with the target word.
2. Capability of the best random-1 head (L17.H3): MMLU subset +
   WikiText PPL, reading both datasets as parquet from the HF cache.

Usage: python scripts/e5f_followups.py
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
OUTD = R / "e5_interventions/gemma-2-2b"


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
def greedy_completion(model, prompt, hooks, n_new=8):
    toks = model.to_tokens(prompt)
    orig = toks.shape[1]
    for _ in range(n_new):
        logits = model.run_with_hooks(toks, fwd_hooks=hooks)[0, -1].float()
        toks = torch.cat([toks, torch.tensor([[int(logits.argmax())]],
                                             device=toks.device)], dim=1)
    return model.to_string(toks[0, orig:])


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

    # ---- 1. full-answer
    fa = {}
    for cfg, heads in (("none", []), ("spec1", [(22, 4)])):
        hooks = zero_hooks(heads)
        first_ok = full_ok = 0
        for s in ho[:300]:
            f = facts[s["case_id"]]
            tid = int(model.to_tokens(" " + f["target_true"])[0, -1])
            did = int(model.to_tokens(" " + f["target_false"])[0, -1])
            logits = model.run_with_hooks(s["prompt_shown"], fwd_hooks=hooks)[0, -1].float()
            first_ok += int(logits.argmax()) == tid
            comp = greedy_completion(model, s["prompt_shown"], hooks)
            full_ok += comp.strip().strip(".,").lower().startswith(f["target_true"].lower())
        fa[cfg] = {"n": 300, "first_token_acc": first_ok / 300,
                   "full_answer_acc": full_ok / 300}
        print(cfg, fa[cfg])
    (OUTD / "full_answer.json").write_text(json.dumps(
        {"experiment": "E5f_full_answer", "results": fa}, indent=1))

    # ---- 2. capability of L17.H3
    hub = Path.home() / "hf_cache/hub"
    mmlu_pq = glob.glob(str(hub / "datasets--cais--mmlu/snapshots/*/all/test-*.parquet"))[0]
    mmlu = pd.read_parquet(mmlu_pq).to_dict("records")
    rng = random.Random(0)
    idx = rng.sample(range(len(mmlu)), 2000)
    letters = ["A", "B", "C", "D"]
    letter_ids = [int(model.to_tokens(" " + L)[0, -1]) for L in letters]
    hooks = zero_hooks([(17, 3)])
    correct = 0
    for i in idx:
        row = mmlu[i]
        prompt = (f"The following are multiple choice questions "
                  f"(with answers) about {row['subject'].replace('_', ' ')}.\n\n"
                  f"{row['question']}\n"
                  + "\n".join(f"{L}. {c}" for L, c in zip(letters, row["choices"]))
                  + "\nAnswer:")
        logits = model.run_with_hooks(prompt, fwd_hooks=hooks)[0, -1].float()
        pred = int(torch.tensor([logits[j] for j in letter_ids]).argmax())
        correct += pred == row["answer"]
    mmlu_acc = correct / len(idx)

    wiki_pq = glob.glob(str(hub / "datasets--wikitext/snapshots/*/wikitext-2-raw-v1/test-*.parquet"))
    text = "\n".join(t for t in pd.read_parquet(wiki_pq[0])["text"] if t.strip())
    ids = model.to_tokens(text)[0]
    nll, n_tok = 0.0, 0
    for w0 in range(0, min(len(ids) - 1025, 100 * 1024), 1024):
        window = ids[w0:w0 + 1024]
        logits = model.run_with_hooks(window.unsqueeze(0), fwd_hooks=hooks)[0].float()
        lp = torch.log_softmax(logits[:-1], -1)
        nll += -lp[torch.arange(len(window) - 1), window[1:]].sum().item()
        n_tok += len(window) - 1
        if (w0 // 1024) >= 19:
            break
    ppl = torch.exp(torch.tensor(nll / n_tok)).item()
    base = json.load(open(OUTD / "capability.json"))["results"]["none"]
    out = {"experiment": "E5f_rand_capability", "rand_L17H3": {
        "mmlu_acc": mmlu_acc, "mmlu_retention": mmlu_acc / base["mmlu_acc"],
        "wiki_ppl": ppl, "ppl_ratio": ppl / base["wiki_ppl"]}}
    (OUTD / "rand_capability.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
