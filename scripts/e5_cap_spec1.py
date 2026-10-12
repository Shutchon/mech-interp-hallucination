#!/usr/bin/env python
"""E5 — Capability retention under intervention: MMLU subset + WikiText PPL.

lm-eval cannot apply TransformerLens hooks, so both the no-intervention
baseline and the intervention configs are scored HERE with identical code —
the retention ratios are internally consistent (this is the number the paper
reports; absolute MMLU will be near, not identical to, the lm-eval 49.4%).

MMLU : 2000-question fixed subset (seed 0), lm-eval 0-shot format, scored by
       comparing logits of ' A'/' B'/' C'/' D' at the answer position.
PPL  : WikiText-2-raw test, 100 fixed windows of 1024 tokens, mean NLL -> PPL.

Configs: none, top-2, top-5, spec3 (specialist knockout), mean5 (mean-ablation).
Usage: python scripts/e5_mmlu_ppl.py
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

SPECIALISTS = [(22, 4), (22, 0), (20, 3)]
LETTERS = ["A", "B", "C", "D"]


def git_rev() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


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


def mean_hooks(heads, mean_z):
    fwd = []
    for l, h in heads:
        def mk(l=l, h=h, v=mean_z[(l, h)]):
            def hook(act, hook):
                act[:, :, h, :] = v.to(device=act.device, dtype=act.dtype)
                return act
            return hook
        fwd.append((f"blocks.{l}.attn.hook_z", mk()))
    return fwd


@torch.no_grad()
def compute_mean_z(model, prompts, heads):
    acc, n = {(l, h): None for l, h in heads}, 0
    for p in prompts:
        _, cache = model.run_with_cache(p)
        for (l, h) in heads:
            v = cache[f"blocks.{l}.attn.hook_z"][0].float().mean(dim=0)[h].cpu()
            acc[(l, h)] = v if acc[(l, h)] is None else acc[(l, h)] + v
        n += 1
    return {k: v / n for k, v in acc.items()}


@torch.no_grad()
def eval_mmlu(model, questions, hooks):
    letter_ids = [int(model.to_tokens(" " + L)[0, -1]) for L in LETTERS]
    correct = 0
    for q in questions:
        if hooks:
            logits = model.run_with_hooks(q["prompt"], fwd_hooks=hooks)[0, -1].float()
        else:
            logits = model(q["prompt"])[0, -1].float()
        pred = int(torch.tensor([logits[i] for i in letter_ids]).argmax())
        correct += pred == q["answer"]
    return correct / len(questions)


@torch.no_grad()
def eval_ppl(model, windows, hooks):
    losses, ntok = 0.0, 0
    for w in windows:
        if hooks:
            logits = model.run_with_hooks(w, fwd_hooks=hooks)[0].float()
        else:
            logits = model(w)[0].float()
        lp = torch.log_softmax(logits[:-1], dim=-1)
        tgt = w[0, 1:]
        losses += -lp.gather(1, tgt.unsqueeze(1)).sum().item()
        ntok += tgt.numel()
    return float(torch.exp(torch.tensor(losses / ntok)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gemma-2-2b")
    ap.add_argument("--n-mmlu", type=int, default=2000)
    ap.add_argument("--n-windows", type=int, default=100)
    ap.add_argument("--e3-dir", type=Path, default=ROOT / "results" / "e3_sweep" / "gemma-2-2b")
    ap.add_argument("--dataset", type=Path, default=ROOT / "results" / "e1_dataset")
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "e5_interventions")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    from datasets import load_dataset

    mmlu = load_dataset("cais/mmlu", "all", split="test")
    rng = random.Random(0)
    idx = rng.sample(range(len(mmlu)), min(args.n_mmlu, len(mmlu)))
    questions = []
    for i in idx:
        r = mmlu[i]
        prompt = (f"The following are multiple choice questions (with answers) about "
                  f"{r['subject'].replace('_', ' ')}.\n\n{r['question']}\n"
                  + "\n".join(f"{L}. {c}" for L, c in zip(LETTERS, r["choices"]))
                  + "\nAnswer:")
        questions.append({"prompt": prompt, "answer": int(r["answer"])})
    print(f"[CAP] MMLU subset: {len(questions)} questions")

    wiki = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n".join(t for t in wiki["text"] if t.strip())

    from transformer_lens import HookedTransformer

    model = HookedTransformer.from_pretrained(
        args.model, dtype=torch.bfloat16, device=args.device,
        center_unembed=False, center_writing_weights=False, fold_ln=False,
    )
    toks = model.to_tokens(text)[0]
    n_win = min(args.n_windows, (toks.shape[0] - 1) // 1024)
    windows = [toks[i * 1024: (i + 1) * 1024 + 1].unsqueeze(0) for i in range(n_win)]
    print(f"[CAP] PPL windows: {len(windows)} x 1024 tokens")

    top20 = json.load(open(args.e3_dir / "midfine_summary.json"))["regimes"]["misleading"]["fine"]["top20"]
    dev_heads = [(int(k[1:k.index("_H")]), int(k[k.index("_H") + 2:])) for k in top20]
    facts = [json.loads(l) for l in (args.dataset / "facts.jsonl").read_text().splitlines() if l.strip()]
    dev_prompts = [f["prompt"] for f in facts if f["split"] == "dev"][:64]
    need_means = sorted(set(dev_heads[:5]))
    print("[CAP] computing mean-z...")
    mean_z = compute_mean_z(model, dev_prompts, need_means)

    configs = {
        "none": [],
        "spec1": zero_hooks([SPECIALISTS[0]]),
    }
    results = {}
    for name, hooks in configs.items():
        acc = eval_mmlu(model, questions, hooks)
        ppl = eval_ppl(model, windows, hooks)
        results[name] = {"mmlu_acc": acc, "wiki_ppl": ppl}
        print(f"[CAP] {name:>7}: MMLU {acc:.1%} | PPL {ppl:.2f}")

    del model
    torch.cuda.empty_cache()
    base = results["none"]
    for name, r in results.items():
        r["mmlu_retention"] = r["mmlu_acc"] / base["mmlu_acc"]
        r["ppl_ratio"] = r["wiki_ppl"] / base["wiki_ppl"]

    summary = {"experiment": "E5_capability", "timestamp": datetime.now(timezone.utc).isoformat(),
               "git_rev": git_rev(), "model": args.model,
               "n_mmlu": len(questions), "n_windows": len(windows), "results": results}
    out_dir = args.out / args.model
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "capability_spec1.json").write_text(json.dumps(summary, indent=2))
    print(f"\nsummary -> {out_dir / 'capability_spec1.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
