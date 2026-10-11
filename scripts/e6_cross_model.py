#!/usr/bin/env python
"""E6 — Cross-model replication (plan §3, E6): the whole mini-pipeline per model.

Stages (all resume-cached):
  facts   : score CounterFact with THIS model, filter known facts (argmax+gap),
            sample n=600, dev/heldout split by relation, build T0 suite + neutral
  coarse  : misleading-regime coarse sweep (all layers x last-3 positions, dev)
  fine    : per-head sweep on the top-5 layers from coarse (dev)
  e5      : knockout top-1/top-5 on held-out suite (T0) + MMLU subset retention

Usage: python scripts/e6_cross_model.py --hf-llm llama-3.2-1b [--stages facts,coarse,fine,e5]
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

TEMPLATES = {
    "misleading": "Earlier you learned that the correct answer is {distractor}. {prompt}",
    "neutral": "Earlier you learned that the correct answer is unknown. {prompt}",
}
LETTERS = ["A", "B", "C", "D"]


def git_rev() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def first_token_id(model, target: str) -> int:
    return int(model.to_tokens(f" {target}")[0, -1])


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def append_cache(path: Path) -> tuple[dict, object]:
    done = {}
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip():
                rec = json.loads(line)
                done[rec["key"]] = rec
    path.parent.mkdir(parents=True, exist_ok=True)
    return done, path.open("a")


def get_model(name: str, device: str):
    from transformer_lens import HookedTransformer

    return HookedTransformer.from_pretrained(
        name, dtype=torch.bfloat16, device=device,
        center_unembed=False, center_writing_weights=False, fold_ln=False,
    )


@torch.no_grad()
def stage_facts(model, out_dir: Path, cf_path: Path, n_target=600, gap=0.5):
    sys.path.insert(0, str(ROOT / "src"))
    from data.counterfact import load_counterfact, split_by_relation

    all_facts = load_counterfact(cf_path)
    scores_done, s_fh = append_cache(out_dir / "scores.jsonl")
    todo = [f for f in all_facts if f.case_id not in scores_done]  # full pass, resumed
    for i, f in enumerate(todo, 1):
        logits = model(f.prompt)[0, -1].float()
        t_id, d_id = first_token_id(model, f.target_true), first_token_id(model, f.target_false)
        rec = {"key": f.case_id, "case_id": f.case_id,
               "gap": float(logits[t_id] - logits[d_id]),
               "argmax_is_true": bool(logits.argmax() == t_id)}
        scores_done[f.case_id] = rec
        s_fh.write(json.dumps(rec) + "\n")
        if i % 500 == 0:
            s_fh.flush()
            print(f"\r[E6:facts] {i}/{len(todo)}", end="")
    s_fh.close()
    print()
    survivors = [f for f in all_facts
                 if scores_done.get(f.case_id, {}).get("argmax_is_true")
                 and scores_done.get(f.case_id, {}).get("gap", 0) >= gap]
    rng = random.Random(0)
    if len(survivors) > n_target:
        survivors = rng.sample(survivors, n_target)
    survivors = split_by_relation(sorted(survivors, key=lambda f: f.case_id), seed=0)
    suite = []
    for f in survivors:
        for cond, tmpl in TEMPLATES.items():
            prompt = tmpl.format(distractor=f.target_false, prompt=f.prompt)
            suite.append({"case_id": f.case_id, "condition": cond, "prompt_shown": prompt,
                          "split": f.split, "true": f.target_true, "false": f.target_false})
    from dataclasses import asdict
    (out_dir / "facts.jsonl").write_text(
        "".join(json.dumps(asdict(f), ensure_ascii=False) + "\n" for f in survivors))
    (out_dir / "suite.jsonl").write_text(
        "".join(json.dumps(s, ensure_ascii=False) + "\n" for s in suite))
    return {"n_scored": len(scores_done), "n_survivors": len(survivors),
            "split": {"dev": sum(f.split == "dev" for f in survivors),
                      "heldout": sum(f.split == "heldout" for f in survivors)}}


@torch.no_grad()
def sweep(model, facts, suite_by_case, stage: str, layers, heads, out_path: Path):
    """Misleading-regime patch sweep: clean value -> misleading run, last-3 pos."""
    done, fh = append_cache(out_path)
    for fi, f in enumerate(facts, 1):
        mis_item = suite_by_case[(f["case_id"], "misleading")]
        true_id = first_token_id(model, f["target_true"])
        d_id = first_token_id(model, f["target_false"])
        clean_toks = model.to_tokens(f["prompt"])
        mis_toks = model.to_tokens(mis_item["prompt_shown"])

        def ld(logits):
            return float(logits[0, -1, true_id] - logits[0, -1, d_id])

        _, cache = model.run_with_cache(clean_toks)
        clean_ld, mis_ld = ld(model(clean_toks)), ld(model(mis_toks))
        cln, mln = clean_toks.shape[1], mis_toks.shape[1]
        targets = ([(l, "resid") for l in layers] if stage == "coarse"
                  else [(l, h) for l in layers for h in heads])
        for (l, spec) in targets:
            hook_name = (f"blocks.{l}.hook_resid_post" if spec == "resid"
                         else f"blocks.{l}.attn.hook_z")
            for k in range(3):
                cpos, mpos = cln - 3 + k, mln - 3 + k
                key = f"{f['case_id']}:{stage}:{l}:{spec}:{k}"
                if key in done:
                    continue
                cv = (cache[hook_name][0, cpos] if spec == "resid"
                      else cache[hook_name][0, cpos, spec])

                def patch(act, hook, v=cv, p=mpos, sp=spec):
                    if sp == "resid":
                        act[:, p, :] = v.to(act.dtype)
                    else:
                        act[:, p, sp, :] = v.to(act.dtype)
                    return act

                pl = ld(model.run_with_hooks(mis_toks, fwd_hooks=[(hook_name, patch)]))
                denom = clean_ld - mis_ld
                rec = {"key": key, "case_id": f["case_id"], "stage": stage, "layer": l,
                       "spec": spec if spec == "resid" else "head",
                       "head": None if spec == "resid" else spec,
                       "effect": (pl - mis_ld) / denom if abs(denom) > 1e-6 else 0.0}
                done[key] = rec
                fh.write(json.dumps(rec) + "\n")
        if fi % 20 == 20:
            fh.flush()
            print(f"\r[E6:{stage}] {fi}/{len(facts)}", end="")
    fh.close()
    print()
    return done


def top_layers(recs: dict, n=5):
    agg = {}
    for r in recs.values():
        if r["stage"] == "coarse":
            agg.setdefault(r["layer"], []).append(r["effect"])
    return [l for l, v in sorted(((l, sum(v) / len(v)) for l, v in agg.items()),
                                 key=lambda x: x[1], reverse=True)[:n]]


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
def stage_e5(model, out_dir: Path, facts, suite, fine_recs, device):
    dev_facts = [f for f in facts if f["split"] == "dev"]
    heldout = [s for s in suite if s["split"] == "heldout"]
    agg = {}
    for r in fine_recs.values():
        if r["stage"] == "fine":
            agg.setdefault((r["layer"], r["head"]), []).append(r["effect"])
    ranked = sorted(((sum(v) / len(v), l, h) for (l, h), v in agg.items()), reverse=True)
    top = [(l, h) for _, l, h in ranked]
    ids = {f["case_id"]: {"true_id": first_token_id(model, f["target_true"]),
                          "distractor_id": first_token_id(model, f["target_false"])} for f in facts}

    results = []
    for name, heads in [("none", []), ("top-1", top[:1]), ("top-5", top[:5])]:
        hooks = zero_hooks(heads)
        fol = tru = n_m = 0
        ntr = n_n = 0
        for it in heldout:
            ids2 = ids[it["case_id"]]
            logits = (model.run_with_hooks(it["prompt_shown"], fwd_hooks=hooks)[0, -1].float()
                      if hooks else model(it["prompt_shown"])[0, -1].float())
            pred = int(logits.argmax())
            if it["condition"] == "misleading":
                fol += pred == ids2["distractor_id"]
                tru += pred == ids2["true_id"]
                n_m += 1
            else:
                ntr += pred == ids2["true_id"]
                n_n += 1
        results.append({"config": name, "follow_rate": fol / max(n_m, 1),
                        "truth_rate": tru / max(n_m, 1), "neutral_truth_rate": ntr / max(n_n, 1)})
        print(f"[E6:e5] {name}: follow {fol/max(n_m,1):.1%} truth {tru/max(n_m,1):.1%} "
              f"neutral {ntr/max(n_n,1):.1%}")

    # MMLU subset retention for none / top-5
    from datasets import load_dataset

    mmlu = load_dataset("cais/mmlu", "all", split="test")
    rng = random.Random(0)
    qs = []
    for i in rng.sample(range(len(mmlu)), 2000):
        r = mmlu[i]
        qs.append({"prompt": (f"The following are multiple choice questions (with answers) about "
                              f"{r['subject'].replace('_', ' ')}.\n\n{r['question']}\n"
                              + "\n".join(f"{L}. {c}" for L, c in zip(LETTERS, r["choices"]))
                              + "\nAnswer:"), "answer": int(r["answer"])})
    letter_ids = [int(model.to_tokens(" " + L)[0, -1]) for L in LETTERS]
    mmlu_res = {}
    for name in ("none", "top-1", "top-5"):
        hooks = zero_hooks(top[:1] if name == "top-1" else top[:5]) if name != "none" else []
        correct = 0
        for q in qs:
            logits = (model.run_with_hooks(q["prompt"], fwd_hooks=hooks)[0, -1].float()
                      if hooks else model(q["prompt"])[0, -1].float())
            correct += int(torch.tensor([logits[i] for i in letter_ids]).argmax()) == q["answer"]
        mmlu_res[name] = correct / len(qs)
        print(f"[E6:e5] MMLU {name}: {correct/len(qs):.1%}")
    return {"suite": results, "mmlu": mmlu_res, "top_heads": [f"L{l}_H{h}" for l, h in top[:20]]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tl-name", required=True, help="TransformerLens model name")
    ap.add_argument("--stages", default="facts,coarse,fine,e5")
    ap.add_argument("--n-facts", type=int, default=600)
    ap.add_argument("--n-coarse", type=int, default=300)
    ap.add_argument("--n-fine", type=int, default=150)
    ap.add_argument("--cf", type=Path, default=Path.home() / "hf_cache/hub")
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "e6_cross")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    out_dir = args.out / args.tl_name
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {"experiment": "E6_cross", "model": args.tl_name,
               "timestamp": datetime.now(timezone.utc).isoformat(), "git_rev": git_rev()}

    model = get_model(args.tl_name, args.device)
    stages = args.stages.split(",")

    if "facts" in stages:
        cf_path = next(Path(args.cf).glob("**/counterfact.json"), None)
        assert cf_path, "counterfact.json not found in HF cache"
        summary["facts"] = stage_facts(model, out_dir, cf_path, n_target=args.n_facts)
        print(f"[E6:facts] {summary['facts']}")
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    facts = load_jsonl(out_dir / "facts.jsonl")
    suite = load_jsonl(out_dir / "suite.jsonl")
    suite_by_case = {(s["case_id"], s["condition"]): s for s in suite}
    dev = [f for f in facts if f["split"] == "dev"][: args.n_coarse]

    if "coarse" in stages:
        recs = sweep(model, dev, suite_by_case, "coarse",
                     layers=list(range(model.cfg.n_layers)), heads=None,
                     out_path=out_dir / "coarse.jsonl")
        summary["coarse_top_layers"] = top_layers(recs)
        print(f"[E6:coarse] top layers: {summary['coarse_top_layers']}")
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    if "fine" in stages:
        layers = (summary.get("coarse_top_layers")
                  or top_layers(load_jsonl_cache(out_dir / "coarse.jsonl")))[:3]
        dev_f = [f for f in facts if f["split"] == "dev"][: args.n_fine]
        recs = sweep(model, dev_f, suite_by_case, "fine", layers=layers,
                     heads=list(range(model.cfg.n_heads)), out_path=out_dir / "fine.jsonl")
        agg = {}
        for r in recs.values():
            if r["stage"] == "fine":
                agg.setdefault(f"L{r['layer']}_H{r['head']}", []).append(r["effect"])
        ranked = sorted(((sum(v) / len(v), k) for k, v in agg.items()), reverse=True)[:10]
        summary["fine_top10"] = [(k, round(e, 3)) for e, k in ranked]
        print(f"[E6:fine] top10: {summary['fine_top10']}")
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    if "e5" in stages:
        fine_recs = {}
        for r in load_jsonl(out_dir / "fine.jsonl"):
            fine_recs[r["key"]] = r
        summary["e5"] = stage_e5(model, out_dir, facts, suite, fine_recs, args.device)
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    del model
    torch.cuda.empty_cache()
    print(f"\nsummary -> {out_dir / 'summary.json'}")
    return 0


def load_jsonl_cache(path: Path) -> dict:
    return {r["key"]: r for r in load_jsonl(path)}


if __name__ == "__main__":
    raise SystemExit(main())
