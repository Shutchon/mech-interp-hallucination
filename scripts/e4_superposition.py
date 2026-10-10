#!/usr/bin/env python
"""E4-lite — Superposition & neutral-control analysis on the top heads (plan §3, E4).

For the top-K heads (default 20) of the misleading fine sweep (dev split):

  effect_A (restore)     : from E3 misleading_fine records — how strongly does
                           patching the CLEAN value into the misleading run
                           restore the true answer?
  effect_B (context-push): NEW — patch the MISLEADING value into the CLEAN run.
                           Positive = the head also CONVEYS the misleading
                           context toward the distractor (dual role).
  effect_N (neutral)     : NEW — patch the CLEAN value into the NEUTRAL prompt
                           run ("unknown" in the context slot). Generic memory
                           injection vs context-dependent restoration.

  superposition_idx = |A - B| / (|A| + |B|)  (JuICE, ICML 2025):
      ~1  specialised (pushes one side only)   ~0 superposed (both sides)

Usage: python scripts/e4_superposition.py [--top-k 20] [--n-facts 150]
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
def run(model, facts, suite_by_case, heads: list[tuple[int, int]], template_idx: int,
        out_path: Path, log_every: int = 10) -> list[dict]:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done: set[str] = set()
    if out_path.exists():
        for line in out_path.read_text().splitlines():
            if line.strip():
                done.add(json.loads(line)["key"])
    recs: list[dict] = []
    with out_path.open("a") as fh:
        for fi, f in enumerate(facts, 1):
            true_id = first_token_id(model, f["target_true"])
            d_id = first_token_id(model, f["target_false"])
            clean_toks = model.to_tokens(f["prompt"])
            mis_item = suite_by_case[(f["case_id"], "misleading", template_idx)]
            neu_item = suite_by_case[(f["case_id"], "neutral", template_idx)]
            mis_toks = model.to_tokens(mis_item["prompt_shown"])
            neu_toks = model.to_tokens(neu_item["prompt_shown"])

            def ld(logits):
                return float(logits[0, -1, true_id] - logits[0, -1, d_id])

            _, cache_clean = model.run_with_cache(clean_toks)
            _, cache_mis = model.run_with_cache(mis_toks)
            clean_ld, mis_ld, neu_ld = ld(model(clean_toks)), ld(model(mis_toks)), ld(model(neu_toks))
            cln, mln, nln = clean_toks.shape[1], mis_toks.shape[1], neu_toks.shape[1]

            for (layer, head) in heads:
                for k in range(3):
                    cpos, mpos, npos = cln - 3 + k, mln - 3 + k, nln - 3 + k
                    hook_name = f"blocks.{layer}.attn.hook_z"

                    key = f"{f['case_id']}:{layer}:{head}:{k}"
                    if key in done:
                        continue

                    # B: misleading value -> clean run
                    mis_val = cache_mis[hook_name][0, mpos, head, :]

                    def hook_b(act, hook, v=mis_val, p=cpos, hd=head):
                        act[:, p, hd, :] = v.to(act.dtype)
                        return act

                    b_ld = ld(model.run_with_hooks(clean_toks, fwd_hooks=[(hook_name, hook_b)]))

                    # N: clean value -> neutral run
                    cl_val = cache_clean[hook_name][0, cpos, head, :]

                    def hook_n(act, hook, v=cl_val, p=npos, hd=head):
                        act[:, p, hd, :] = v.to(act.dtype)
                        return act

                    n_ld = ld(model.run_with_hooks(neu_toks, fwd_hooks=[(hook_name, hook_n)]))

                    eB = (clean_ld - b_ld) / (clean_ld - mis_ld) if abs(clean_ld - mis_ld) > 1e-6 else 0.0
                    eN = (n_ld - neu_ld) / (clean_ld - neu_ld) if abs(clean_ld - neu_ld) > 1e-6 else 0.0
                    rec = {"key": key, "case_id": f["case_id"], "layer": layer,
                           "head": head, "pos_k": k,
                           "clean_ld": clean_ld, "mis_ld": mis_ld, "neu_ld": neu_ld,
                           "b_ld": b_ld, "n_ld": n_ld, "effect_B": eB, "effect_N": eN}
                    recs.append(rec)
                    fh.write(json.dumps(rec) + "\n")
            if fi % log_every == 0:
                fh.flush()
                print(f"\r[E4] {fi}/{len(facts)} facts", end="")
    print()
    return recs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gemma-2-2b")
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--n-facts", type=int, default=150)
    ap.add_argument("--template", type=int, default=0)
    ap.add_argument("--dataset", type=Path, default=ROOT / "results" / "e1_dataset")
    ap.add_argument("--e3-dir", type=Path, default=ROOT / "results" / "e3_sweep" / "gemma-2-2b")
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "e4_superposition")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    summary_e3 = json.load(open(args.e3_dir / "midfine_summary.json"))
    top20 = summary_e3["regimes"]["misleading"]["fine"]["top20"][: args.top_k]
    heads = [(int(k[1:k.index("_H")]), int(k[k.index("_H") + 2:])) for k in top20]
    print(f"[E4] top-{len(heads)} heads: {top20}")

    # effect_A per head from the E3 misleading_fine records
    eff_a: dict[tuple[int, int], list] = {}
    for r in load_jsonl(args.e3_dir / "misleading_fine.jsonl"):
        if r["comp"] == "head":
            eff_a.setdefault((r["layer"], r["head"]), []).append(r["effect"])

    facts = [f for f in load_jsonl(args.dataset / "facts.jsonl") if f["split"] == "dev"][: args.n_facts]
    suite_by_case = {(s["case_id"], s["condition"], s["template_idx"]): s
                     for s in load_jsonl(args.dataset / "misleading_suite.jsonl")}

    from transformer_lens import HookedTransformer

    model = HookedTransformer.from_pretrained(
        args.model, dtype=torch.bfloat16, device=args.device,
        center_unembed=False, center_writing_weights=False, fold_ln=False,
    )
    out_dir = args.out / args.model
    recs = run(model, facts, suite_by_case, heads, args.template, out_dir / "records.jsonl")
    del model
    torch.cuda.empty_cache()

    per_head = {}
    for r in recs:
        per_head.setdefault((r["layer"], r["head"]), {"B": [], "N": []})
        per_head[(r["layer"], r["head"])]["B"].append(r["effect_B"])
        per_head[(r["layer"], r["head"])]["N"].append(r["effect_N"])

    table = []
    for (layer, head) in heads:
        A = sum(eff_a.get((layer, head), [0.0])) / max(len(eff_a.get((layer, head), [1])), 1)
        d = per_head.get((layer, head), {"B": [0.0], "N": [0.0]})
        B = sum(d["B"]) / len(d["B"])
        N = sum(d["N"]) / len(d["N"])
        sup = abs(A - B) / (abs(A) + abs(B) + 1e-8)
        table.append({"head": f"L{layer}_H{head}", "layer": layer, "head_idx": head,
                      "effect_A_restore": A, "effect_B_context_push": B,
                      "effect_N_neutral": N, "superposition_idx": sup,
                      "neutral_ratio": (N / A) if abs(A) > 1e-6 else None})
    table.sort(key=lambda r: r["effect_A_restore"], reverse=True)
    summary = {"experiment": "E4_superposition", "timestamp": datetime.now(timezone.utc).isoformat(),
               "git_rev": git_rev(), "model": args.model, "n_facts": len(facts),
               "top_heads": top20, "table": table}
    (out_dir / "superposition_summary.json").write_text(json.dumps(summary, indent=2))

    print("\n" + "=" * 78)
    print(f"{'head':>8} {'A restore':>10} {'B push':>9} {'N neutral':>10} {'sup_idx':>8} {'N/A':>6}")
    for r in table[:12]:
        na = f"{r['neutral_ratio']:.2f}" if r["neutral_ratio"] is not None else "-"
        print(f"{r['head']:>8} {r['effect_A_restore']:>10.3f} {r['effect_B_context_push']:>9.3f} "
              f"{r['effect_N_neutral']:>10.3f} {r['superposition_idx']:>8.2f} {na:>6}")
    print("=" * 78)
    print(f"summary -> {out_dir / 'superposition_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
