#!/usr/bin/env python
"""E6b — per-item records for the replication models (review R8).

Reruns the e6 held-out evaluation (none / top-1 / top-5 knockout from
each model's own dev ranking) with per-item records so bootstrap CIs and
McNemar tests can be computed.

Usage: python scripts/e6b_peritem.py
Output: results/e6_cross/<hf>/<model>/e5_per_item.jsonl
"""
from __future__ import annotations

import json
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"
MODELS = [("meta-llama/Llama-3.2-1B", "llama-3.2-1b", 16, 32),
          ("Qwen/Qwen2.5-1.5B", "qwen2.5-1.5b", 28, 12)]
TL_NAMES = {"llama-3.2-1b": "Llama-3.2-1B", "qwen2.5-1.5b": "Qwen2.5-1.5B"}


def zero_hooks(heads):
    fwd = []
    for l, h in heads:
        def mk(l=l, h=h):
            def hk(act, hook):
                act[:, :, h, :] = 0.0
                return act
            return hk
        fwd.append((f"blocks.{l}.attn.hook_z", mk()))
    return fwd


@torch.no_grad()
def main():
    from transformer_lens import HookedTransformer
    for hf_id, short, n_layers, n_heads in MODELS:
        d = R / "e6_cross" / hf_id.split("/")[0] / hf_id.split("/")[1]
        summary = json.load(open(d / "summary.json"))
        fine_top10 = [tuple(map(int, h.replace("L", "").split("_H"))) for h, _ in summary["fine_top10"]]
        facts = [json.loads(l) for l in open(d / "facts.jsonl")]
        suite = [json.loads(l) for l in open(d / "suite.jsonl")]
        ho_cases = {f["case_id"]: f for f in facts if f["split"] == "heldout"}
        model = HookedTransformer.from_pretrained(
            TL_NAMES[short], dtype=torch.bfloat16,
            center_unembed=False, fold_ln=False, center_writing_weights=False
        ).to("cuda" if torch.cuda.is_available() else "cpu")
        out_path = d / "e5_per_item.jsonl"
        with open(out_path, "w") as fh:
            for cfg, heads in (("none", []), ("top-1", fine_top10[:1]), ("top-5", fine_top10[:5])):
                hooks = zero_hooks(heads)
                for s in suite:
                    f = ho_cases[s["case_id"]]
                    tid = model.to_single_token(" " + f["target_true"])
                    did = model.to_single_token(" " + f["target_false"])
                    logits = model.run_with_hooks(s["prompt_shown"], fwd_hooks=hooks)[0, -1].float()
                    pred = int(logits.argmax())
                    neu_logits = model.run_with_hooks(
                        s["prompt_shown"].replace(f["target_false"], "unknown"),
                        fwd_hooks=hooks)[0, -1].float()
                    neu_pred = int(neu_logits.argmax())
                    fh.write(json.dumps({
                        "config": cfg, "case_id": s["case_id"],
                        "condition": s.get("condition", "misleading"),
                        "is_true": pred == tid, "is_distractor": pred == did,
                        "neu_is_true": neu_pred == tid}) + "\n")
        print("written", out_path)
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
