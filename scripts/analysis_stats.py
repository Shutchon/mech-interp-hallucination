#!/usr/bin/env python
"""Analysis — statistical tests and paper numbers from the pushed results.

Outputs results/analysis/stats.json + printed summary:
  1. E5 dose-response: bootstrap 95% CIs on follow rates; McNemar tests
     (vs none) for every intervention config (round 1 + round 2).
  2. Cross-model consistency: Spearman correlation of coarse layer profiles
     at matched depth fractions (gemma vs llama vs qwen).
  3. Head-table: top-10 gemma heads with E4 superposition metrics.

Usage: .venv-mac/bin/python scripts/analysis_stats.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from eval.metrics import bootstrap_ci, mcnemar_test  # noqa: E402

R = ROOT / "results"


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def layer_profile(coarse_path: Path, n_layers: int) -> np.ndarray:
    recs = load_jsonl(coarse_path)
    agg = {}
    for r in recs:
        agg.setdefault(r["layer"], []).append(r["effect"])
    return np.array([np.mean(agg.get(l, [0.0])) for l in range(n_layers)])


def at_depth_fraction(profile: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Interpolate a layer profile onto normalised depth [0,1]."""
    x = np.linspace(0.0, 1.0, len(profile))
    return np.interp(grid, x, profile)


def suite_rates(records: list[dict]) -> dict:
    m = [r for r in records if r["condition"] == "misleading"]
    follow = np.array([r["is_distractor"] for r in m])
    return {"n": len(m), "follow": float(follow.mean()),
            "follow_ci95": bootstrap_ci(follow),
            "truth": float(np.mean([r["is_true"] for r in m]))}


def main() -> int:
    out = {"experiment": "analysis", "models": {}}

    # ---------- 1. E5 round 1 + round 2: CIs and McNemar vs none ----------
    per_item = {}
    for fn in ("per_item.jsonl", "round2_per_item.jsonl"):
        p = R / "e5_interventions" / "gemma-2-2b" / fn
        if p.exists():
            for r in load_jsonl(p):
                per_item.setdefault(r["config"], []).append(r)
    by_cfg = {}
    for cfg, recs in per_item.items():
        # one MISLEADING record per case_id (neutral shares case_id — keep only misleading)
        by_cfg[cfg] = {r["case_id"]: r for r in recs if r["condition"] == "misleading"}
    none_m = {r["case_id"]: r for r in per_item.get("none", []) if r["condition"] == "misleading"}

    e5_stats = {}
    for cfg in sorted(by_cfg):
        if cfg == "none" or cfg == "random5-mean":
            continue
        rates = suite_rates(per_item[cfg])
        b = sum(1 for cid, r in by_cfg[cfg].items()
                if none_m[cid]["is_distractor"] and not r["is_distractor"])
        c = sum(1 for cid, r in by_cfg[cfg].items()
                if not none_m[cid]["is_distractor"] and r["is_distractor"])
        e5_stats[cfg] = {"follow": rates["follow"], "follow_ci95": rates["follow_ci95"],
                         "truth": rates["truth"], "mcnemar_b": b, "mcnemar_c": c,
                         "mcnemar": mcnemar_test(b, c)}
    out["e5_gemma"] = e5_stats

    # ---------- 2. Cross-model layer-profile consistency ----------
    profiles = {
        "gemma-2-2b": layer_profile(R / "e3_sweep/gemma-2-2b/misleading_coarse.jsonl", 26),
        "llama-3.2-1b": layer_profile(R / "e6_cross/meta-llama/Llama-3.2-1B/coarse.jsonl", 16),
        "qwen2.5-1.5b": layer_profile(R / "e6_cross/Qwen/Qwen2.5-1.5B/coarse.jsonl", 28),
    }
    from scipy.stats import spearmanr

    grid = np.linspace(0.05, 0.95, 10)
    interp = {k: at_depth_fraction(v, grid) for k, v in profiles.items()}
    pairs = {}
    names = list(interp)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            rho = spearmanr(interp[names[i]], interp[names[j]]).statistic
            pairs[f"{names[i]}~{names[j]}"] = round(float(rho), 3)
    out["cross_model_profile_spearman"] = pairs
    out["layer_profiles"] = {k: [round(float(x), 4) for x in v] for k, v in profiles.items()}

    # ---------- 3. Head table (gemma) ----------
    sup = json.load(open(R / "e4_superposition/gemma-2-2b/superposition_summary.json"))
    out["head_table"] = sup["table"][:10]

    out_dir = R / "analysis"
    out_dir.mkdir(exist_ok=True)
    (out_dir / "stats.json").write_text(json.dumps(out, indent=2))

    print("=" * 70)
    print("E5 (gemma) — follow rate [95% CI] vs none, with McNemar p")
    for cfg, s in sorted(e5_stats.items()):
        ci = s["follow_ci95"]
        print(f"  {cfg:>12}: {s['follow']:.1%} [{ci[1]:.1%},{ci[2]:.1%}]  "
              f"p={s['mcnemar']['pvalue']:.2e}  (b={s['mcnemar_b']}, c={s['mcnemar_c']})")
    print(f"\nCross-model profile Spearman: {pairs}")
    print(f"\n-> {out_dir / 'stats.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
