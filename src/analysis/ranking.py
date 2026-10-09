"""Head ranking and cross-model analysis (plan §3, E4).

E4 checklist from the plan:
- rank heads by restoration effect (dev split only) + bootstrap 95% CI
- cross-model consistency: Spearman rank correlation of layer profiles
- superposition test (JuICE, ICML 2025): do influential heads favour one side
  (memory / context) or both? — patch in BOTH directions (clean→corrupted and
  corrupted→clean) and compare per-head effects
- copy-suppression control: neutral-context condition (see data/misleading.py)
- entity-popularity stratification with PopQA scores (references.md [K5])
- categorise heads with the 4-stage taxonomy of Zheng et al. 2025 (Patterns)
"""
from __future__ import annotations

import numpy as np
from scipy import stats


def rank_heads(sweep_results, bootstrap_n: int = 10_000):
    """Rank heads by mean restoration effect on the DEV split with bootstrap
    CIs. Selection happens exclusively here — held-out items never touch this
    function (anti-circularity, plan §8.3)."""
    raise NotImplementedError("E4")


def layer_profile_correlation(profile_a: dict, profile_b: dict) -> float:
    """Spearman correlation between layer-wise restoration profiles of two
    models (used for the cross-model consistency claim, RQ1)."""
    layers = sorted(set(profile_a) & set(profile_b))
    rho, _ = stats.spearmanr(
        [profile_a[l] for l in layers], [profile_b[l] for l in layers]
    )
    return float(rho)


def superposition_index(effect_memory_dir: float, effect_context_dir: float) -> float:
    """Symmetry of a head's influence between the two conflict directions.

    Values near 1.0: the head pushes one side only (specialised).
    Values near 0.0: balanced influence on both sides (superposed, cf. JuICE).
    index = |m - c| / (|m| + |c| + eps), m = clean→corrupted restoration,
    c = corrupted→clean (context-boosting) effect.
    """
    m, c = abs(effect_memory_dir), abs(effect_context_dir)
    return (abs(m - c)) / (m + c + 1e-8)
