#!/usr/bin/env python
"""E5-extra — closes two reviewer-obvious gaps before submission.

  1. Template robustness: evaluate none / spec1 / top-5 on HELD-OUT
     misleading templates T1 and T2 (round 1-2 only used T0).
  2. Capability of the headline intervention: MMLU subset + WikiText PPL
     under spec1 (capability.json only had spec3).

Usage: python scripts/e5_extra.py
"""
from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

SPECIALIST = (22, 4)
LETTERS = ["A", "B", "C", "D"]


def git_rev() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def first_token_id(model, target: str) -> int:
    return int(model.to_tokens(f" {target}")[0, -1])


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def zero_hooks(heads):
    fwd = []
    for l, h in heads:
        def mk(l=l, h=h):
            def hook(act, hook):
                act[:, :, h, :] = 0
                return act
            return hook
        fwd.append((f"blocks.{l}.attn.hook_z", mk()))
    return fwd


@torch.no_grad()
def eval_suite(model, items, ids, hooks, name):
    fol = tru = n_m = 0
    for it in items:
        logits = (model.run_with_hooks(it["prompt_shown"], fwd_hooks=hooks)[0, -1].float()
                  if hooks else model(it["prompt_shown"])[0, -1].float())
        pred = int(logits.argmax())
        if it["condition"] == "misleading":
            fol += pred == ids[it["case_id"]]["distractor_id"]
            tru += pred == ids[it["case_id"]]["true_id"]
            n_m += 1
    r = {"config": name, "n": n_m, "follow_rate": fol / max(n_m, 1),
         "truth_rate": tru / max(n_m, 1)}
    print(f"[extra] {name}: follow {r['follow_rate']:.1%} truth {r['truth_rate']:.1%}")
    return r


@torch.no_grad()
def eval_capability(model, questions, windows, hooks, name):
    letter_ids = [int(model.to_tokens(" " + L)[0, -1]) for L in LETTERS]
    correct = 0
    for q in questions:
        logits = (model.run_with_hooks(q, fwd_hooks=hooks)[0, -1].float()
                  if hooks else model(q)[0, -1].float())
        correct += int(torch.tensor([logits[i] for i in letter_ids]).argmax()) == q[1]
    losses, ntok = 0.0, 0
    for w in windows:
        logits = (model.run_with_hooks(w, fwd_hooks=hooks)[0].float()
                  if hooks else model(w)[0].float())
        lp = torch.log_softmax(logits[:-1], dim=-1)
        tgt = w[0, 1:]
        losses += -lp.gather(1, tgt.unsqueeze(1)).sum().item()
        ntok += tgt.numel()
    ppl = float(torch.exp(torch.tensor(losses / ntok)))
    print(f"[extra] {name}: MMLU {correct/len(questions):.1%} | PPL {ppl:.2f}")
    return {"config": name, "mmlu_acc": correct / len(questions), "wiki_ppl": ppl}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, default=ROOT / "results" / "e1_dataset")
    ap.add_argument("--e3-dir", type=Path, default=ROOT / "results" / "e3_sweep" / "gemma-2-2b")
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "e5_interventions")
    args = ap.parse_args()

    facts = load_jsonl(args.dataset / "facts.jsonl")
    suite = load_jsonl(args.dataset / "misleading_suite.jsonl")
    ids = {f["case_id"]: {"true_id": None, "distractor_id": None} for f in facts}

    from transformer_lens import HookedTransformer

    model = HookedTransformer.from_pretrained(
        "gemma-2-2b", dtype=torch.bfloat16,
        device="cuda" if torch.cuda.is_available() else "cpu",
        center_unembed=False, center_writing_weights=False, fold_ln=False,
    )
    ids = {f["case_id"]: {"true_id": first_token_id(model, f["target_true"]),
                          "distractor_id": first_token_id(model, f["target_false"])}
           for f in facts}
    top20 = json.load(open(args.e3_dir / "midfine_summary.json"))["regimes"]["misleading"]["fine"]["top20"]
    top5 = [(int(k[1:k.index("_H")]), int(k[k.index("_H") + 2:])) for k in top20[:5]]

    results = {"experiment": "E5_extra", "timestamp": datetime.now(timezone.utc).isoformat(),
               "git_rev": git_rev(), "template_robustness": {}, "capability_spec1": {}}

    # 1) template robustness on held-out, templates 1 and 2
    for t in (1, 2):
        items = [s for s in suite if s["split"] == "heldout" and s["template_idx"] == t
                 and s["condition"] == "misleading"]
        results["template_robustness"][f"T{t}"] = [
            eval_suite(model, items, ids, [], "none"),
            eval_suite(model, items, ids, zero_hooks([SPECIALIST]), "spec1"),
            eval_suite(model, items, ids, zero_hooks(top5), "top-5"),
        ]

    # 2) capability under spec1 (same fixed MMLU subset + windows protocol)
    from datasets import load_dataset

    mmlu = load_dataset("cais/mmlu", "all", split="test")
    rng = random.Random(0)
    questions = []
    for i in rng.sample(range(len(mmlu)), 2000):
        r = mmlu[i]
        questions.append((f"The following are multiple choice questions (with answers) about "
                          f"{r['subject'].replace('_', ' ')}.\n\n{r['question']}\n"
                          + "\n".join(f"{L}. {c}" for L, c in zip(LETTERS, r["choices"]))
                          + "\nAnswer:", int(r["answer"])))
    wiki = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    toks = model.to_tokens("\n".join(t for t in wiki["text"] if t.strip()))[0]
    windows = [toks[i * 1024:(i + 1) * 1024 + 1].unsqueeze(0) for i in range(100)]

    results["capability_spec1"] = {
        "none": eval_capability(model, questions, windows, [], "none"),
        "spec1": eval_capability(model, questions, windows, zero_hooks([SPECIALIST]), "spec1"),
    }

    del model
    torch.cuda.empty_cache()
    out_dir = args.out / "gemma-2-2b"
    (out_dir / "extra.json").write_text(json.dumps(results, indent=2))
    print(f"saved -> {out_dir / 'extra.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
