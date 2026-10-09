#!/usr/bin/env python
"""E3 — Coarse activation-patching sweep (plan §3, E3, stage 1 of 3).

Coarse granularity: patch residual-stream block outputs (hook_resid_post)
layer-by-layer, position-by-position. Two corruption regimes:

  noise      clean = bare fact prompt; corrupted = same tokens with Gaussian
             noise (3x embedding std) added to all non-BOS embeddings
             (ROME-style denoising; identical lengths -> full layer x position
             heatmap)

  misleading clean = bare fact prompt; corrupted = misleading-context prompt
             (template 0). Prompts differ in length, so we patch the aligned
             question SUFFIX (last 3 tokens of both runs) — the answer-position
             representation feeding the unembedding.

Anti-circularity: sweeps use DEV split facts only (plan §8.3).
Metric: logit-diff restoration effect (Zhang & Nanda ICLR 2024 style):
    effect = (patched_ld - corrupted_ld) / (clean_ld - corrupted_ld)

Resumable per (case_id, layer, position) — JSONL append cache.
Usage:
  python scripts/e3_patching_sweep.py                      # both regimes
  python scripts/e3_patching_sweep.py --regime misleading  # one regime
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
def sweep_regime(model, facts, suite_by_case, regime: str, template_idx: int,
                 out_path: Path, seed: int, log_every: int = 10) -> list[dict]:
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
                corr_toks = clean_toks
                embed = cache["hook_embed"]
                noise = (3.0 * float(embed.std())) * torch.randn(
                    embed.shape, generator=generator, dtype=torch.float32
                ).to(device=embed.device, dtype=embed.dtype)
                noise[:, 0, :] = 0  # keep BOS intact

                def noise_hook(act, hook):
                    act[:, :, :] = (act.float() + noise).to(act.dtype)
                    return act

                corrupted_ld = ld(model.run_with_hooks(
                    corr_toks, fwd_hooks=[("hook_embed", noise_hook)]))
                positions = list(range(1, clean_toks.shape[1]))  # all non-BOS
                src_positions = positions
            else:  # misleading
                corr_toks = model.to_tokens(
                    suite_by_case[(f["case_id"], "misleading", template_idx)]["prompt_shown"])
                corrupted_ld = ld(model(corr_toks))
                corr_len, clean_len = corr_toks.shape[1], clean_toks.shape[1]
                positions = [corr_len - 1 - k for k in (2, 1, 0)]  # last 3, ordered
                src_positions = [clean_len - 1 - k for k in (2, 1, 0)]

            n_layers = model.cfg.n_layers
            for layer in range(n_layers):
                clean_resid = cache["blocks." + str(layer) + ".hook_resid_post"]
                for pos, src in zip(positions, src_positions):
                    key = f"{f['case_id']}:{layer}:{pos}"
                    if key in done:
                        continue

                    def patch_hook(act, hook, src_val=clean_resid[0, src], p=pos):
                        act[:, p, :] = src_val.to(act.dtype)
                        return act

                    if regime == "noise":
                        patched_logits = model.run_with_hooks(
                            corr_toks,
                            fwd_hooks=[("hook_embed", noise_hook),
                                       (f"blocks.{layer}.hook_resid_post", patch_hook)])
                    else:
                        patched_logits = model.run_with_hooks(
                            corr_toks,
                            fwd_hooks=[(f"blocks.{layer}.hook_resid_post", patch_hook)])
                    patched_ld = ld(patched_logits)
                    denom = clean_ld - corrupted_ld
                    effect = (patched_ld - corrupted_ld) / denom if abs(denom) > 1e-6 else 0.0
                    rec = {"key": key, "regime": regime, "case_id": f["case_id"],
                           "layer": layer, "pos": pos, "clean_ld": clean_ld,
                           "corrupted_ld": corrupted_ld, "patched_ld": patched_ld,
                           "effect": effect}
                    all_recs.append(rec)
                    fh.write(json.dumps(rec) + "\n")
            if fi % log_every == 0:
                fh.flush()
                print(f"\r[E3:{regime}] {fi}/{len(facts)} facts", end="")
    print()
    return all_recs


def summarize(recs: list[dict], n_layers: int) -> dict:
    by_layer = {}
    for r in recs:
        by_layer.setdefault(r["layer"], []).append(r["effect"])
    layer_mean = {l: sum(v) / len(v) for l, v in sorted(by_layer.items())}
    top5 = sorted(layer_mean, key=layer_mean.get, reverse=True)[:5]
    corrupted_flip = sum(1 for r in recs if r["corrupted_ld"] < r["clean_ld"]) / max(len(recs), 1)
    return {"n_records": len(recs), "layer_mean_effect": layer_mean,
            "top5_layers_by_effect": top5,
            "mean_effect_top5": sum(layer_mean[l] for l in top5) / len(top5),
            "corrupted_flips_ld": corrupted_flip}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gemma-2-2b")
    ap.add_argument("--regime", choices=["noise", "misleading", "both"], default="both")
    ap.add_argument("--n-noise", type=int, default=100)
    ap.add_argument("--n-misleading", type=int, default=300)
    ap.add_argument("--template", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dataset", type=Path, default=ROOT / "results" / "e1_dataset")
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "e3_sweep")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    facts = [f for f in load_jsonl(args.dataset / "facts.jsonl") if f["split"] == "dev"]
    suite_by_case = {(s["case_id"], s["condition"], s["template_idx"]): s
                     for s in load_jsonl(args.dataset / "misleading_suite.jsonl")}
    print(f"[E3] {len(facts)} dev facts available")

    from transformer_lens import HookedTransformer

    model = HookedTransformer.from_pretrained(
        args.model, dtype=torch.bfloat16, device=args.device,
        center_unembed=False, center_writing_weights=False, fold_ln=False,
    )
    out_dir = args.out / args.model
    regimes = ["noise", "misleading"] if args.regime == "both" else [args.regime]
    summary: dict = {"experiment": "E3_coarse_sweep",
                     "timestamp": datetime.now(timezone.utc).isoformat(),
                     "git_rev": git_rev(), "model": args.model,
                     "template": args.template, "seed": args.seed, "regimes": {}}
    for regime in regimes:
        n = args.n_noise if regime == "noise" else args.n_misleading
        sub = facts[:n]
        print(f"\n[E3] regime={regime} on {len(sub)} dev facts")
        recs = sweep_regime(model, sub, suite_by_case, regime, args.template,
                            out_dir / f"{regime}_coarse.jsonl", args.seed)
        s = summarize(recs, model.cfg.n_layers)
        summary["regimes"][regime] = s
        print(f"  top-5 layers: {s['top5_layers_by_effect']} "
              f"(mean effect on those: {s['mean_effect_top5']:.3f})")

    (out_dir / "coarse_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nsummary -> {out_dir / 'coarse_summary.json'}")
    del model
    torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
