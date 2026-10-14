#!/usr/bin/env python
"""E5e — follow-ups for review 3: robust full-answer check + capability
of the best random-1 head (L17.H3).

1. Full-answer: greedy-generate 8 tokens on 300 held-out misleading
   prompts (none/spec1); an answer counts as correct if the target word
   appears at the START of the generated continuation (robust
   containment matching, no exact string slicing).
2. Capability: MMLU subset + WikiText PPL under L17.H3 knockout, for
   comparison with spec1 (Table 5).

Usage: python scripts/e5e_followups.py
"""
from __future__ import annotations

import json
import random
from pathlib import Path

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

    # ---- 1. robust full-answer check
    fa = {}
    sample = ho[:300]
    for cfg, heads in (("none", []), ("spec1", [(22, 4)])):
        hooks = zero_hooks(heads)
        first_ok = full_ok = 0
        for s in sample:
            f = facts[s["case_id"]]
            tid = int(model.to_tokens(" " + f["target_true"])[0, -1])
            did = int(model.to_tokens(" " + f["target_false"])[0, -1])
            logits = model.run_with_hooks(s["prompt_shown"], fwd_hooks=hooks)[0, -1].float()
            first_ok += int(logits.argmax()) == tid
            gen = model.generate(s["prompt_shown"], max_new_tokens=8,
                                 do_sample=False, verbose=False)
            cont = gen[0]
            if cont.startswith(s["prompt_shown"]):
                cont = cont[len(s["prompt_shown"]):]
            cont = cont.strip().strip(".,").lower()
            full_ok += cont.startswith(f["target_true"].lower())
        fa[cfg] = {"n": len(sample), "first_token_acc": first_ok / len(sample),
                   "full_answer_acc": full_ok / len(sample)}
        print(cfg, fa[cfg])
    (OUTD / "full_answer.json").write_text(json.dumps(
        {"experiment": "E5e_full_answer", "results": fa}, indent=1))

    # ---- 2. capability of best random-1 head (L17.H3)
    from datasets import load_dataset
    mmlu = load_dataset("cais/mmlu", "all", split="test")
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
        pred = int(torch.tensor([logits[i2] for i2 in letter_ids]).argmax())
        correct += pred == row["answer"]
    mmlu_acc = correct / len(idx)

    wiki = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n".join(t for t in wiki["text"] if t.strip())
    ids = model.to_tokens(text)[0]
    nll, n_tok = 0.0, 0
    for w0 in range(0, min(len(ids) - 1025, 100 * 1024), 1024):
        window = ids[w0:w0 + 1024]
        logits = model.run_with_hooks(window.unsqueeze(0), fwd_hooks=hooks)[0].float()
        lp = torch.log_softmax(logits[:-1], -1)
        nll += -lp[torch.arange(len(window) - 1), window[1:]].sum().item()
        n_tok += len(window) - 1
    ppl = torch.exp(torch.tensor(nll / n_tok)).item()
    base = json.load(open(OUTD / "capability.json"))["results"]["none"]
    out = {"experiment": "E5e_rand_capability", "rand_L17H3": {
        "mmlu_acc": mmlu_acc, "mmlu_retention": mmlu_acc / base["mmlu_acc"],
        "wiki_ppl": ppl, "ppl_ratio": ppl / base["wiki_ppl"]}}
    (OUTD / "rand_capability.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
