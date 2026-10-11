#!/usr/bin/env python
"""Fig 3 data extraction — attention patterns of the top heads (needs GPU).

For a fixed representative dev fact, run the misleading prompt (T0) through
Gemma-2-2B and save the full attention pattern [pos, pos] of each top head,
plus the token strings. Output: results/analysis/fig3_attention.json
(plotted on the Mac afterwards — no GPU needed there).

Usage: python scripts/extract_fig3.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

HEADS = [(22, 0), (25, 4), (20, 3), (25, 6), (22, 4)]  # E3/E4 top heads
FACT_INDEX = 0  # first dev fact (fixed for reproducibility)


def main() -> int:
    facts = [json.loads(l) for l in (ROOT / "results/e1_dataset/facts.jsonl").read_text().splitlines() if l.strip()]
    suite = {(s["case_id"], s["condition"], s["template_idx"]): s
             for s in (json.loads(l) for l in (ROOT / "results/e1_dataset/misleading_suite.jsonl").read_text().splitlines() if l.strip())}
    dev = [f for f in facts if f["split"] == "dev"]
    f = dev[FACT_INDEX]
    prompt = suite[(f["case_id"], "misleading", 0)]["prompt_shown"]
    print(f"[fig3] fact: {f['prompt']!r} true={f['target_true']} false={f['target_false']}")

    from transformer_lens import HookedTransformer

    model = HookedTransformer.from_pretrained(
        "gemma-2-2b", dtype=torch.bfloat16,
        device="cuda" if torch.cuda.is_available() else "cpu",
        center_unembed=False, center_writing_weights=False, fold_ln=False,
    )
    tokens = model.to_str_tokens(prompt)
    _, cache = model.run_with_cache(prompt)
    patterns = {}
    for l, h in HEADS:
        pat = cache[f"blocks.{l}.attn.hook_pattern"][0, :, h, :].float().cpu()  # [pos, pos]
        patterns[f"L{l}_H{h}"] = [[round(float(v), 4) for v in row] for row in pat]
        top_att = pat[-1].argmax().item()
        print(f"[fig3] L{l}_H{h}: last token attends most to -> {tokens[top_att]!r} ({pat[-1, top_att]:.2f})")

    out = {"prompt": prompt, "tokens": tokens,
           "fact": {"true": f["target_true"], "false": f["target_false"]},
           "patterns": patterns}
    out_dir = ROOT / "results/analysis"
    out_dir.mkdir(exist_ok=True)
    (out_dir / "fig3_attention.json").write_text(json.dumps(out))
    print(f"[fig3] saved -> {out_dir / 'fig3_attention.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
