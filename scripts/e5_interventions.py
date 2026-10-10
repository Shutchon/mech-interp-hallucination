#!/usr/bin/env python
"""E5 — Intervention dose-response (plan §3, E5) — ROUND 1: knockout on held-out.

Configs (all evaluated on the HELD-OUT split, template 0: misleading + neutral):
  none      : no intervention (sanity re-check of E2 numbers on heldout)
  top-k     : knockout (zero z at all positions) of the top-k heads from the
              DEV misleading ranking, k in {1, 2, 5, 10, 20}
  random-k  : 5 seeds, k=5, heads sampled from ALL 208 heads (fairness baseline)
  topnorm-k : top-5 heads by mean |z| over 64 dev prompts (fairness baseline)

Metrics per config: misleading follow/truth rates + neutral truth rate.
Per-item records saved for paired stats (McNemar vs none) later.
MMLU/PPL and full-template sweeps run at the best k in the next round.

Usage: python scripts/e5_interventions.py
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


def aggregate(recs: list[dict], config_name: str) -> dict:
    """Recompute a config summary from per-item records (resume path)."""
    m = [r for r in recs if r["condition"] == "misleading"]
    n = [r for r in recs if r["condition"] == "neutral"]
    return {"config": config_name, "n_misleading": len(m), "n_neutral": len(n),
            "follow_rate": sum(r["is_distractor"] for r in m) / max(len(m), 1),
            "truth_rate": sum(r["is_true"] for r in m) / max(len(m), 1),
            "neutral_truth_rate": sum(r["is_true"] for r in n) / max(len(n), 1)}


def zero_hooks(heads: list[tuple[int, int]]) -> list:
    fwd = []
    for l, h in heads:
        def mk(l=l, h=h):
            def hook(act, hook):  # act [batch, pos, n_heads, d_head]
                act[:, :, h, :] = 0
                return act
            return hook
        fwd.append((f"blocks.{l}.attn.hook_z", mk()))
    return fwd


@torch.no_grad()
def eval_config(model, items, ids_by_case, hooks, config_name, out_fh, log_every=500):
    follow = truth = n_m = 0
    neu_truth = n_n = 0
    per_item = []
    total = len(items)
    for i, it in enumerate(items, 1):
        ids = ids_by_case[it["case_id"]]
        if hooks:
            logits = model.run_with_hooks(it["prompt_shown"], fwd_hooks=hooks)[0, -1].float()
        else:
            logits = model(it["prompt_shown"])[0, -1].float()
        pred = int(logits.argmax())
        is_true = pred == ids["true_id"]
        rec = {"config": config_name, "case_id": it["case_id"],
               "condition": it["condition"], "is_true": bool(is_true)}
        if it["condition"] == "misleading":
            is_d = pred == ids["distractor_id"]
            rec["is_distractor"] = bool(is_d)
            follow += is_d
            truth += is_true
            n_m += 1
        else:
            neu_truth += is_true
            n_n += 1
        per_item.append(rec)
        if i % log_every == 0:
            print(f"\r[E5:{config_name}] {i}/{total}", end="")
    print()
    for r in per_item:
        out_fh.write(json.dumps(r) + "\n")
    out_fh.flush()
    return {"config": config_name, "n_misleading": n_m, "n_neutral": n_n,
            "follow_rate": follow / max(n_m, 1), "truth_rate": truth / max(n_m, 1),
            "neutral_truth_rate": neu_truth / max(n_n, 1)}


@torch.no_grad()
def head_norms(model, prompts, n_layers, n_heads):
    """Mean |z| per head across prompts and positions."""
    acc = torch.zeros(n_layers, n_heads, dtype=torch.float64)
    for p in prompts:
        _, cache = model.run_with_cache(p)
        for l in range(n_layers):
            z = cache[f"blocks.{l}.attn.hook_z"][0]  # [pos, n_heads, d_head]
            acc[l] += z.float().abs().mean(dim=(0, 2)).double().cpu()
    return (acc / len(prompts)).numpy()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gemma-2-2b")
    ap.add_argument("--template", type=int, default=0)
    ap.add_argument("--ks", default="1,2,5,10,20")
    ap.add_argument("--n-random-seeds", type=int, default=5)
    ap.add_argument("--n-norm-prompts", type=int, default=64)
    ap.add_argument("--dataset", type=Path, default=ROOT / "results" / "e1_dataset")
    ap.add_argument("--e3-dir", type=Path, default=ROOT / "results" / "e3_sweep" / "gemma-2-2b")
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "e5_interventions")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    summary_e3 = json.load(open(args.e3_dir / "midfine_summary.json"))
    top20 = summary_e3["regimes"]["misleading"]["fine"]["top20"]
    dev_heads = [(int(k[1:k.index("_H")]), int(k[k.index("_H") + 2:])) for k in top20]

    facts = load_jsonl(args.dataset / "facts.jsonl")
    suite = load_jsonl(args.dataset / "misleading_suite.jsonl")
    heldout_ids = {f["case_id"] for f in facts if f["split"] == "heldout"}
    items = [s for s in suite if s["split"] == "heldout" and s["template_idx"] == args.template
             and s["condition"] in ("misleading", "neutral")]
    dev_facts = [f for f in facts if f["split"] == "dev"]
    print(f"[E5] {len(items)} held-out items (T{args.template}), {len(dev_heads)} ranked dev heads")

    from transformer_lens import HookedTransformer

    model = HookedTransformer.from_pretrained(
        args.model, dtype=torch.bfloat16, device=args.device,
        center_unembed=False, center_writing_weights=False, fold_ln=False,
    )
    ids_by_case = {}
    for f in facts:
        ids_by_case[f["case_id"]] = {
            "true_id": first_token_id(model, f["target_true"]),
            "distractor_id": first_token_id(model, f["target_false"]),
        }

    out_dir = args.out / args.model
    out_dir.mkdir(parents=True, exist_ok=True)
    per_item_path = out_dir / "per_item.jsonl"
    # Resume support: a config is complete when it has one record per item.
    existing = load_jsonl(per_item_path) if per_item_path.exists() else []
    counts: dict[str, int] = {}
    for r in existing:
        counts[r["config"]] = counts.get(r["config"], 0) + 1
    n_expected = len(items)
    done_cfgs = {c for c, n in counts.items() if n >= n_expected}
    if counts:
        keep = [r for r in existing if counts[r["config"]] >= n_expected]
        if len(keep) != len(existing):  # drop partial configs (e.g. preemption)
            per_item_path.write_text("".join(json.dumps(r) + "\n" for r in keep))
        print(f"[E5] resume: skipping completed configs: {sorted(done_cfgs)}")
    existing_by_cfg: dict[str, list[dict]] = {}
    for r in existing:
        if r["config"] in done_cfgs:
            existing_by_cfg.setdefault(r["config"], []).append(r)

    results = []
    with per_item_path.open("a") as fh:

        def run_or_load(name, hooks):
            if name in done_cfgs:
                return aggregate(existing_by_cfg[name], name)
            return eval_config(model, items, ids_by_case, hooks, name, fh)

        results.append(run_or_load("none", []))

        for k in [int(x) for x in args.ks.split(",")]:
            results.append(run_or_load(f"top-{k}", zero_hooks(dev_heads[:k])))

        rng = random.Random(0)
        all_heads = [(l, h) for l in range(model.cfg.n_layers) for h in range(model.cfg.n_heads)]
        rand_rates = []
        for s in range(args.n_random_seeds):
            heads = rng.sample(all_heads, 5)
            r = run_or_load(f"random5-s{s}", zero_hooks(heads))
            rand_rates.append(r["follow_rate"])
        results.append({"config": "random5-mean", "follow_rate": sum(rand_rates) / len(rand_rates),
                        "follow_rate_sd": (sum((x - sum(rand_rates) / len(rand_rates)) ** 2
                                               for x in rand_rates) / len(rand_rates)) ** 0.5})

        norms = head_norms(model, [f["prompt"] for f in dev_facts[: args.n_norm_prompts]],
                           model.cfg.n_layers, model.cfg.n_heads)
        top_norm_heads = sorted(
            [(l, h) for l in range(model.cfg.n_layers) for h in range(model.cfg.n_heads)],
            key=lambda lh: norms[lh[0], lh[1]], reverse=True)[:5]
        results.append(run_or_load("topnorm-5", zero_hooks(top_norm_heads)))

    del model
    torch.cuda.empty_cache()

    summary = {"experiment": "E5_dose_response_round1", "timestamp": datetime.now(timezone.utc).isoformat(),
               "git_rev": git_rev(), "model": args.model, "template": args.template,
               "dev_head_ranking": top20, "topnorm_heads": [f"L{l}_H{h}" for l, h in top_norm_heads],
               "results": results}
    (out_dir / "dose_response.json").write_text(json.dumps(summary, indent=2))

    print("\n" + "=" * 74)
    print(f"{'config':>14} {'follow_rate':>12} {'truth_rate':>11} {'neutral_truth':>14}")
    for r in results:
        print(f"{r['config']:>14} {r['follow_rate']:>12.1%} "
              f"{r.get('truth_rate', float('nan')):>11.1%} "
              f"{r.get('neutral_truth_rate', float('nan')):>14.1%}")
    print("=" * 74)
    print(f"summary -> {out_dir / 'dose_response.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
