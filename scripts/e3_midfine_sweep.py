#!/usr/bin/env python
"""E3 — Mid + fine patching sweep (plan §3, E3 stages 2-3).

Zooms into the layers highlighted by the coarse sweep (default 20-25 for
gemma-2-2b, from coarse_summary.json):

  mid : patch attention-block output vs MLP-block output (hook_attn_out /
        hook_mlp_out) — which block type carries the restorable memory signal?
  fine: patch per-head z-vectors (attn.hook_z) at the last-3 (answer-side)
        positions — produces the TOP-HEAD RANKING on the dev split that E5
        interventions consume (anti-circularity: dev only).

Corruption regimes identical to the coarse sweep (noise / misleading T0).
Resumable per key. Usage: python scripts/e3_midfine_sweep.py [--layers 20-25]
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


def parse_layers(spec: str) -> list[int]:
    out = []
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


@torch.no_grad()
def sweep(model, facts, suite_by_case, regime: str, template_idx: int, layers,
          stage: str, out_path: Path, seed: int, log_every: int = 10) -> list[dict]:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done: set[str] = set()
    if out_path.exists():
        for line in out_path.read_text().splitlines():
            if line.strip():
                done.add(json.loads(line)["key"])

    generator = torch.Generator(device="cpu").manual_seed(seed)
    all_recs: list[dict] = []
    with out_path.open("a") as fh:
        for fi, f in enumerate(facts, 1):
            true_id = first_token_id(model, f["target_true"])
            d_id = first_token_id(model, f["target_false"])
            clean_toks = model.to_tokens(f["prompt"])
            _, cache = model.run_with_cache(clean_toks)

            def ld(logits):
                return float(logits[0, -1, true_id] - logits[0, -1, d_id])

            clean_ld = ld(model(clean_toks))

            if regime == "noise":
                embed = cache["hook_embed"]
                noise = (3.0 * float(embed.std())) * torch.randn(
                    embed.shape, generator=generator, dtype=torch.float32
                ).to(device=embed.device, dtype=embed.dtype)
                noise[:, 0, :] = 0

                def noise_hook(act, hook):
                    act[:, :, :] = (act.float() + noise).to(act.dtype)
                    return act

                corr_toks = clean_toks
                corrupted_ld = ld(model.run_with_hooks(
                    corr_toks, fwd_hooks=[("hook_embed", noise_hook)]))
                # mid stage uses all non-BOS positions (like coarse); fine uses last-3
                if stage == "mid":
                    positions = list(range(1, clean_toks.shape[1]))
                else:
                    ln = clean_toks.shape[1]
                    positions = [ln - 3, ln - 2, ln - 1]
                src_positions = positions
            else:
                corr_toks = model.to_tokens(
                    suite_by_case[(f["case_id"], "misleading", template_idx)]["prompt_shown"])
                corrupted_ld = ld(model(corr_toks))
                cl, cln = corr_toks.shape[1], clean_toks.shape[1]
                positions = [cl - 3, cl - 2, cl - 1]
                src_positions = [cln - 3, cln - 2, cln - 1]

            hook_specs = []
            for layer in layers:
                if stage == "mid":
                    hook_specs.append((f"blocks.{layer}.hook_attn_out", layer, "attn", None))
                    hook_specs.append((f"blocks.{layer}.hook_mlp_out", layer, "mlp", None))
                else:
                    n_heads = model.cfg.n_heads
                    for h in range(n_heads):
                        hook_specs.append((f"blocks.{layer}.attn.hook_z", layer, "head", h))

            for hook_name, layer, comp, head in hook_specs:
                for pos, src in zip(positions, src_positions):
                    key = f"{f['case_id']}:{stage}:{comp}{'' if head is None else head}:{layer}:{pos}"
                    if key in done:
                        continue

                    if comp == "head":
                        clean_val = cache[hook_name][0, src, head, :]
                        patch_name = hook_name

                        def patch_hook(act, hook, v=clean_val, p=pos, hd=head):
                            act[:, p, hd, :] = v.to(act.dtype)
                            return act
                    else:
                        clean_val = cache[hook_name][0, src, :]
                        patch_name = hook_name

                        def patch_hook(act, hook, v=clean_val, p=pos):
                            act[:, p, :] = v.to(act.dtype)
                            return act

                    hooks = [(patch_name, patch_hook)]
                    if regime == "noise":
                        hooks.append(("hook_embed", noise_hook))
                    patched_ld = ld(model.run_with_hooks(corr_toks, fwd_hooks=hooks))
                    denom = clean_ld - corrupted_ld
                    effect = (patched_ld - corrupted_ld) / denom if abs(denom) > 1e-6 else 0.0
                    rec = {"key": key, "regime": regime, "stage": stage, "case_id": f["case_id"],
                           "layer": layer, "comp": comp, "head": head, "pos": pos,
                           "clean_ld": clean_ld, "corrupted_ld": corrupted_ld,
                           "patched_ld": patched_ld, "effect": effect}
                    all_recs.append(rec)
                    fh.write(json.dumps(rec) + "\n")
            if fi % log_every == 0:
                fh.flush()
                print(f"\r[E3:{regime}:{stage}] {fi}/{len(facts)} facts", end="")
    print()
    return all_recs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gemma-2-2b")
    ap.add_argument("--layers", default="20-25", help="e.g. 20-25 or 20,22,25")
    ap.add_argument("--stage", choices=["mid", "fine", "both"], default="both")
    ap.add_argument("--n-noise", type=int, default=100)
    ap.add_argument("--n-misleading", type=int, default=300)
    ap.add_argument("--template", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dataset", type=Path, default=ROOT / "results" / "e1_dataset")
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "e3_sweep")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    layers = parse_layers(args.layers)
    facts = [f for f in load_jsonl(args.dataset / "facts.jsonl") if f["split"] == "dev"]
    suite_by_case = {(s["case_id"], s["condition"], s["template_idx"]): s
                     for s in load_jsonl(args.dataset / "misleading_suite.jsonl")}

    from transformer_lens import HookedTransformer

    model = HookedTransformer.from_pretrained(
        args.model, dtype=torch.bfloat16, device=args.device,
        center_unembed=False, center_writing_weights=False, fold_ln=False,
    )
    out_dir = args.out / args.model
    summary = {"experiment": "E3_midfine_sweep",
               "timestamp": datetime.now(timezone.utc).isoformat(),
               "git_rev": git_rev(), "model": args.model, "layers": layers,
               "regimes": {}}

    for regime in ("noise", "misleading"):
        n = args.n_noise if regime == "noise" else args.n_misleading
        sub = facts[:n]
        summary["regimes"][regime] = {}
        for stage in (["mid", "fine"] if args.stage == "both" else [args.stage]):
            print(f"\n[E3] regime={regime} stage={stage} layers={layers} n={len(sub)}")
            recs = sweep(model, sub, suite_by_case, regime, args.template,
                         layers, stage, out_dir / f"{regime}_{stage}.jsonl", args.seed)
            agg: dict = {}
            for r in recs:
                k = f"L{r['layer']}{'_H' + str(r['head']) if r['head'] is not None else '_' + r['comp']}"
                agg.setdefault(k, []).append(r["effect"])
            means = {k: sum(v) / len(v) for k, v in agg.items()}
            summary["regimes"][regime][stage] = {
                "n_records": len(recs),
                "mean_effect": means,
                "top20": sorted(means, key=means.get, reverse=True)[:20],
            }
            top = summary["regimes"][regime][stage]["top20"][:8]
            print(f"  top: {[(k, round(means[k], 3)) for k in top]}")

    (out_dir / "midfine_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nsummary -> {out_dir / 'midfine_summary.json'}")
    del model
    torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
