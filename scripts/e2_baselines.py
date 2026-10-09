#!/usr/bin/env python
"""E2 — Behavioural baselines on the experiment dataset (plan §3, E2).

For the model under study (default gemma-2-2b):
  1. clean accuracy on facts.jsonl          — memory strength
  2. misleading suite, per template         — RQ2 preliminary: how often does
     the misleading context beat memory (context-following rate)?
  3. neutral control                        — same template, "unknown" in the
     context slot; truth rate should stay ~clean, unknown-following ~0

Standard benchmarks (MMLU / TruthfulQA / WikiText PPL) run separately via
scripts/run_lmeval.sh.

Resumable: per-item predictions append to items.jsonl; rerun skips done keys.
Usage: python scripts/e2_baselines.py [--model gemma-2-2b]
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

from eval.metrics import bootstrap_ci  # noqa: E402


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


@torch.no_grad()
def run_items(model, jobs: list[dict], cache_path: Path) -> dict:
    """jobs: dicts with key/prompt_shown/true_id/distractor_id. Returns map key->record."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    done: dict[str, dict] = {}
    if cache_path.exists():
        for line in cache_path.read_text().splitlines():
            if line.strip():
                rec = json.loads(line)
                done[rec["key"]] = rec

    todo = [j for j in jobs if j["key"] not in done]
    if not todo:
        print(f"[E2] all {len(jobs)} items already evaluated (cache hit)")
        return done

    with cache_path.open("a") as fh:
        for i, j in enumerate(todo, 1):
            logits = model(j["prompt"])[0, -1].float()
            pred = int(logits.argmax())
            rec = {
                "key": j["key"],
                "case_id": j["case_id"],
                "condition": j["condition"],
                "template_idx": j["template_idx"],
                "split": j["split"],
                "pred_id": pred,
                "true_id": j["true_id"],
                "distractor_id": j["distractor_id"],
                "is_true": pred == j["true_id"],
                "is_distractor": pred == j["distractor_id"],
                "logit_gap": float(logits[j["true_id"]] - logits[j["distractor_id"]]),
            }
            done[rec["key"]] = rec
            fh.write(json.dumps(rec) + "\n")
            if i % 250 == 0:
                fh.flush()
                print(f"\r[E2] {i}/{len(todo)}", end="")
    print()
    return done


def rate_block(records: list[dict]) -> dict:
    n = len(records)
    if n == 0:
        return {"n": 0}
    follow = [r["is_distractor"] for r in records]
    truth = [r["is_true"] for r in records]
    return {
        "n": n,
        "truth_rate": sum(truth) / n,
        "follow_rate": sum(follow) / n,
        "follow_ci95": bootstrap_ci(follow),
        "neither_rate": 1 - sum(truth) / n - sum(follow) / n,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gemma-2-2b")
    ap.add_argument("--dataset", type=Path, default=ROOT / "results" / "e1_dataset")
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "e2_baselines")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    out_dir = args.out / args.model
    out_dir.mkdir(parents=True, exist_ok=True)

    facts = load_jsonl(args.dataset / "facts.jsonl")
    suite = load_jsonl(args.dataset / "misleading_suite.jsonl")
    print(f"[E2] {len(facts)} facts, {len(suite)} suite items")

    from transformer_lens import HookedTransformer

    model = HookedTransformer.from_pretrained(
        args.model, dtype=torch.bfloat16, device=args.device,
        center_unembed=False, center_writing_weights=False, fold_ln=False,
    )
    # token ids per fact (true / distractor / neutral 'unknown')
    ids: dict[int, dict] = {}
    for f in facts:
        ids[f["case_id"]] = {
            "true_id": first_token_id(model, f["target_true"]),
            "distractor_id": first_token_id(model, f["target_false"]),
        }
    neutral_id = first_token_id(model, "unknown")

    jobs = []
    for f in facts:  # clean
        jobs.append({"key": f"{f['case_id']}:clean:-1", "prompt": f["prompt"],
                     "case_id": f["case_id"], "condition": "clean", "template_idx": -1,
                     "split": f["split"], **ids[f["case_id"]]})
    for s in suite:  # misleading + neutral
        t_id, d_id = ids[s["case_id"]]["true_id"], ids[s["case_id"]]["distractor_id"]
        jobs.append({"key": f"{s['case_id']}:{s['condition']}:{s['template_idx']}",
                     "prompt": s["prompt_shown"], "case_id": s["case_id"],
                     "condition": s["condition"], "template_idx": s["template_idx"],
                     "split": s["split"], "true_id": t_id,
                     "distractor_id": d_id if s["condition"] == "misleading" else neutral_id})

    recs = run_items(model, jobs, out_dir / "items.jsonl")
    del model
    torch.cuda.empty_cache()

    def sel(cond=None, t_idx=None, split=None):
        return [r for r in recs.values()
                if (cond is None or r["condition"] == cond)
                and (t_idx is None or r["template_idx"] == t_idx)
                and (split is None or r["split"] == split)]

    summary = {
        "experiment": "E2_baselines",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_rev": git_rev(),
        "model": args.model,
        "clean": rate_block(sel("clean")),
        "misleading": {
            "overall": rate_block(sel("misleading")),
            **{f"template_{t}": rate_block(sel("misleading", t)) for t in range(3)},
            "by_split": {sp: rate_block(sel("misleading", split=sp)) for sp in ("dev", "heldout")},
        },
        "neutral": rate_block(sel("neutral")),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    print("\n" + "=" * 62)
    print(f"clean accuracy                : {summary['clean']['truth_rate']:.1%}  (n={summary['clean']['n']})")
    m = summary["misleading"]
    print(f"MISLEADING context-following  : {m['overall']['follow_rate']:.1%}  "
          f"(truth {m['overall']['truth_rate']:.1%}, 95% CI follow "
          f"{m['overall']['follow_ci95'][1]:.1%}-{m['overall']['follow_ci95'][2]:.1%})")
    for t in range(3):
        b = m[f"template_{t}"]
        print(f"  template {t}                  : follow {b['follow_rate']:.1%} / truth {b['truth_rate']:.1%}")
    n = summary["neutral"]
    print(f"NEUTRAL control               : truth {n['truth_rate']:.1%}, unknown-follow {n['follow_rate']:.1%}")
    print(f"outputs                       : {out_dir}/")
    print("=" * 62)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
