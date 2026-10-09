"""Metrics and statistical tests for E2/E5 evaluation (plan §4).

Conventions:
- Predictions are greedy argmax token ids (decoding is deterministic everywhere).
- "Hallucination" on the misleading suite means the model answers with the
  distractor (context-following) rather than the memorised true target.
- All confidence intervals are bootstrap over items (10k resamples).
- Multiple comparisons across heads use Benjamini–Hochberg FDR.
"""
from __future__ import annotations

import numpy as np
from scipy import stats


def accuracy(preds: np.ndarray, targets: np.ndarray) -> float:
    """Fraction of greedy predictions equal to the true target."""
    preds = np.asarray(preds)
    targets = np.asarray(targets)
    return float((preds == targets).mean())


def hallucination_rate(
    preds: np.ndarray,
    true_targets: np.ndarray,
    distractor_targets: np.ndarray | None = None,
) -> float:
    """Fraction of answers that follow the misleading context.

    If distractor targets are provided, an item counts as hallucinated when the
    prediction equals the distractor (strict context-following). Otherwise any
    answer different from the true target counts (lenient definition).
    """
    preds = np.asarray(preds)
    if distractor_targets is not None:
        return float((preds == np.asarray(distractor_targets)).mean())
    return float((preds != np.asarray(true_targets)).mean())


def bootstrap_ci(
    values: np.ndarray,
    stat=np.mean,
    n_boot: int = 10_000,
    ci: float = 0.95,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Point estimate and percentile CI for `stat` over bootstrap resamples."""
    values = np.asarray(values)
    rng = np.random.default_rng(seed)
    n = len(values)
    samples = values[rng.integers(0, n, size=(n_boot, n))]
    boot = stat(samples, axis=1)
    alpha = (1 - ci) / 2
    point = stat(values)
    return float(point), float(np.quantile(boot, alpha)), float(np.quantile(boot, 1 - alpha))


def mcnemar_test(b: int, c: int) -> dict:
    """McNemar test on paired correctness before/after intervention (plan §4).

    b = #items correct before & wrong after, c = #items wrong before & correct
    after. Uses the exact binomial test when b+c < 25, else chi-square with
    continuity correction.
    """
    n = b + c
    if n == 0:
        return {"statistic": 0.0, "pvalue": 1.0, "method": "undefined (b+c=0)"}
    if n < 25:
        p = stats.binomtest(min(b, c), n, 0.5).pvalue
        return {"statistic": float(min(b, c)), "pvalue": float(p), "method": "exact_binomial"}
    chi2 = (abs(b - c) - 1) ** 2 / n
    p = 1 - stats.chi2.cdf(chi2, df=1)
    return {"statistic": float(chi2), "pvalue": float(p), "method": "chi2_continuity_corrected"}


def benjamini_hochberg(pvals: np.ndarray) -> np.ndarray:
    """BH-FDR adjusted q-values for a family of per-head tests."""
    pvals = np.asarray(pvals, dtype=float)
    n = len(pvals)
    order = np.argsort(pvals)
    ranked = pvals[order] * n / (np.arange(n) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    q = np.empty(n)
    q[order] = np.clip(ranked, 0, 1)
    return q
