#!/usr/bin/env python
"""E1 — Build the experiment dataset (plan §3, E1).

Pipeline: download CounterFact -> score "known facts" with the parity-passed
model (TL2 + bf16) -> filter (argmax == true target AND first-token logit gap
>= 0.5) -> sample N=2000 -> dev/heldout split at the RELATION level -> build
the misleading suite (3 templates) + neutral control.

Usage:
    pilot (recommended first, ~1-2 min):
        python scripts/e1_build_dataset.py --limit 200
    full run (~10-15 min on L4, resumable across spot preemptions):
        python scripts/e1_build_dataset.py
Outputs -> results/e1_dataset/{scores,facts,misleading_suite}.jsonl + summary.json
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

from data.counterfact import load_counterfact, save_facts, split_by_relation  # noqa: E402
from data.misleading import build_misleading_suite  # noqa: E402

DEFAULT_SOURCE = "hf:wangzn2001/counteract:counterfact.json"


def git_rev() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def get_counterfact_path(source: str) -> Path:
    if source.startswith("hf:"):
        _, repo, fname = source.split(":", 2)
        fname = fname or "counterfact.json"
        from huggingface_hub import hf_hub_download

        return Path(hf_hub_download(repo_id=repo, filename=fname, repo_type="dataset"))
    return Path(source)


def first_token_id(model, target: str) -> int:
    """First token of ' {target}' (leading space = natural next-token form)."""
    return int(model.to_tokens(f" {target}")[0, -1])


@torch.no_grad()
def score_facts(model, facts, cache_path: Path) -> dict:
    """Known-fact scoring, resumable: appends one JSONL line per fact so a spot
    preemption never loses progress (plan §5 idempotency rule)."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    done: dict[int, dict] = {}
    if cache_path.exists():
        for line in cache_path.read_text().splitlines():
            if line.strip():
                rec = json.loads(line)
                done[rec["case_id"]] = rec

    todo = [f for f in facts if f.case_id not in done]
    if not todo:
        print(f"[E1] all {len(facts)} facts already scored (cache hit)")
        return done

    with cache_path.open("a") as fh:
        for i, f in enumerate(todo, 1):
            logits = model(f.prompt)[0, -1].float()
            t_id, d_id = first_token_id(model, f.target_true), first_token_id(model, f.target_false)
            rec = {
                "case_id": f.case_id,
                "logit_true": float(logits[t_id]),
                "logit_false": float(logits[d_id]),
                "gap": float(logits[t_id] - logits[d_id]),
                "argmax_is_true": bool(logits.argmax() == t_id),
            }
            done[f.case_id] = rec
            fh.write(json.dumps(rec) + "\n")
            if i % 250 == 0:
                fh.flush()
                print(f"\r[E1] scored {i}/{len(todo)}", end="")
    print()
    return done


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default=DEFAULT_SOURCE,
                    help="hf:<repo>:<file> or local path to counterfact.json")
    ap.add_argument("--model", default="gemma-2-2b")
    ap.add_argument("--n-target", type=int, default=2000)
    ap.add_argument("--gap", type=float, default=0.5,
                    help="min first-token logit gap (true - distractor)")
    ap.add_argument("--limit", type=int, default=None,
                    help="pilot mode: only first K facts")
    ap.add_argument("--relax-argmax", action="store_true",
                    help="drop the argmax==true requirement if survival is too low")
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "e1_dataset")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    cf_path = get_counterfact_path(args.source)
    facts = load_counterfact(cf_path)
    print(f"[E1] loaded {len(facts)} facts from {cf_path}")
    if args.limit:
        facts = facts[: args.limit]
        print(f"[E1] PILOT MODE: limited to first {len(facts)}")

    from transformer_lens import HookedTransformer

    model = HookedTransformer.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        device=args.device,
        center_unembed=False,
        center_writing_weights=False,
        fold_ln=False,
    )
    scores = score_facts(model, facts, args.out / "scores.jsonl")
    del model
    torch.cuda.empty_cache()

    n_argmax = sum(scores[f.case_id]["argmax_is_true"] for f in facts)
    n_gap = sum(scores[f.case_id]["gap"] >= args.gap for f in facts)
    survivors = [
        f for f in facts
        if scores[f.case_id]["gap"] >= args.gap
        and (args.relax_argmax or scores[f.case_id]["argmax_is_true"])
    ]

    rng = random.Random(0)
    if len(survivors) > args.n_target:
        survivors = rng.sample(survivors, args.n_target)
    survivors = split_by_relation(sorted(survivors, key=lambda f: f.case_id), seed=0)

    save_facts(survivors, args.out / "facts.jsonl")
    items = build_misleading_suite(survivors, args.out / "misleading_suite.jsonl")

    summary = {
        "experiment": "E1_dataset",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_rev": git_rev(),
        "source": args.source,
        "model": args.model,
        "params": {"gap": args.gap, "n_target": args.n_target,
                   "relax_argmax": args.relax_argmax, "limit": args.limit},
        "n_loaded": len(facts),
        "n_pass_argmax": int(n_argmax),
        "n_pass_gap": int(n_gap),
        "n_survivors": len(survivors),
        "n_selected": len(survivors),
        "split_counts": {"dev": sum(f.split == "dev" for f in survivors),
                         "heldout": sum(f.split == "heldout" for f in survivors)},
        "suite_items": {"misleading": sum(it.condition == "misleading" for it in items),
                        "neutral": sum(it.condition == "neutral" for it in items)},
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))

    print("\n" + "=" * 60)
    print(f"loaded / argmax==true / gap>={args.gap} / selected : "
          f"{len(facts)} / {n_argmax} / {n_gap} / {len(survivors)}")
    print(f"survival rate (full filter)                        : {len(survivors)}/{len(facts)}"
          f" = {len(survivors)/max(len(facts),1):.1%}")
    print(f"split dev/heldout (by relation)                    : "
          f"{summary['split_counts']['dev']}/{summary['split_counts']['heldout']}")
    print(f"suite items (misleading + neutral)                 : "
          f"{summary['suite_items']['misleading']}+{summary['suite_items']['neutral']}")
    print(f"outputs                                            : {args.out}/")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
