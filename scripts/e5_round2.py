#!/usr/bin/env python
"""E5 round 2 — targeted interventions on HELD-OUT (plan §3, E5 + E4 predictions).

Configs (suite eval, template 0, misleading + neutral, held-out split):
  spec1        : knockout L22_H4 only (top conflict-specialist, E4: N≈0)
  spec3        : knockout {L22_H4, L22_H0, L20_H3} (all N≈0 specialists)
  mean2/mean5  : MEAN-ABLATION of top-2/top-5 dev heads (vs zero-knockout)
  meanspec3    : mean-ablation of the 3 specialists
Baseline "none" reused from round 1 (dose_response.json).

Mean-ablation vector: mean head z over 64 dev reference prompts, positions
pooled. Prediction from E4: spec configs cut follow rate with little neutral
damage (specialists do nothing in normal operation).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

SPECIALISTS = [(22, 4), (22, 0), (20, 3)]  # E4: N/A ≈ 0 (conflict-specialists)


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
    """mean_z: dict (layer,head) -> [d_head] tensor."""
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
    acc = {(l, h): None for l, h in heads}
    n = 0
    for p in prompts:
        _, cache = model.run_with_cache(p)
        for (l, h) in heads:
            z = cache[f"blocks.{l}.attn.hook_z"][0].float().mean(dim=0)  # [n_heads, d]
            v = z[h].cpu()
            acc[(l, h)] = v if acc[(l, h)] is None else acc[(l, h)] + v
        n += 1
    return {k: v / n for k, v in acc.items()}


@torch.no_grad()
def eval_config(model, items, ids_by_case, hooks, config_name, out_fh):
    follow = truth = n_m = 0
    neu_truth = n_n = 0
    per_item = []
    for it in items:
        ids = ids_by_case[it["case_id"]]
        if hooks:
            logits = model.run_with_hooks(it["prompt_shown"], fwd_hooks=hooks)[0, -1].float()
        else:
            logits = model(it["prompt_shown"])[0, -1].float()
        pred = int(logits.argmax())
        rec = {"config": config_name, "case_id": it["case_id"],
               "condition": it["condition"], "is_true": pred == ids["true_id"]}
        if it["condition"] == "misleading":
            rec["is_distractor"] = pred == ids["distractor_id"]
            follow += rec["is_distractor"]
            truth += rec["is_true"]
            n_m += 1
        else:
            neu_truth += rec["is_true"]
            n_n += 1
        per_item.append(rec)
    for r in per_item:
        out_fh.write(json.dumps(r) + "\n")
    out_fh.flush()
    print(f"[E5r2:{config_name}] follow {follow/max(n_m,1):.1%} "
          f"truth {truth/max(n_m,1):.1%} neutral {neu_truth/max(n_n,1):.1%}")
    return {"config": config_name, "follow_rate": follow / max(n_m, 1),
            "truth_rate": truth / max(n_m, 1), "neutral_truth_rate": neu_truth / max(n_n, 1)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gemma-2-2b")
    ap.add_argument("--template", type=int, default=0)
    ap.add_argument("--n-mean-prompts", type=int, default=64)
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
    items = [s for s in suite if s["split"] == "heldout" and s["template_idx"] == args.template
             and s["condition"] in ("misleading", "neutral")]
    dev_facts = [f for f in facts if f["split"] == "dev"]
    print(f"[E5r2] {len(items)} held-out items; specialists={SPECIALISTS}")

    from transformer_lens import HookedTransformer

    model = HookedTransformer.from_pretrained(
        args.model, dtype=torch.bfloat16, device=args.device,
        center_unembed=False, center_writing_weights=False, fold_ln=False,
    )
    ids_by_case = {f["case_id"]: {"true_id": first_token_id(model, f["target_true"]),
                                  "distractor_id": first_token_id(model, f["target_false"])}
                   for f in facts}

    print("[E5r2] computing mean-z vectors over dev reference prompts...")
    mean_z = compute_mean_z(model, [f["prompt"] for f in dev_facts[: args.n_mean_prompts]],
                            dev_heads[:10] + SPECIALISTS)

    out_dir = args.out / args.model
    out_dir.mkdir(parents=True, exist_ok=True)
    results = []
    with (out_dir / "round2_per_item.jsonl").open("a") as fh:
        results.append(eval_config(model, items, ids_by_case, zero_hooks(SPECIALISTS[:1]), "spec1", fh))
        results.append(eval_config(model, items, ids_by_case, zero_hooks(SPECIALISTS), "spec3", fh))
        results.append(eval_config(model, items, ids_by_case, mean_hooks(dev_heads[:2], mean_z), "mean2", fh))
        results.append(eval_config(model, items, ids_by_case, mean_hooks(dev_heads[:5], mean_z), "mean5", fh))
        results.append(eval_config(model, items, ids_by_case, mean_hooks(SPECIALISTS, mean_z), "meanspec3", fh))

    del model
    torch.cuda.empty_cache()

    summary = {"experiment": "E5_round2", "timestamp": datetime.now(timezone.utc).isoformat(),
               "git_rev": git_rev(), "model": args.model,
               "specialists": [f"L{l}_H{h}" for l, h in SPECIALISTS], "results": results}
    (out_dir / "round2.json").write_text(json.dumps(summary, indent=2))
    print(f"\nsummary -> {out_dir / 'round2.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
