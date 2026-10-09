"""Patching effect metrics (plan §3, E3).

Metric: logit difference between the true target and the distractor target on
the answer position. Restoration effect is normalised so that 0 = corrupted
baseline and 1 = full restoration of the clean run.
"""
from __future__ import annotations

import torch


def logit_diff(logits: torch.Tensor, true_id: int, distractor_id: int) -> float:
    """logit(true target) - logit(distractor target) at the answer position.

    logits: [vocab] or [batch, vocab] — answer-position logits, fp32.
    """
    if logits.dim() == 1:
        return float(logits[true_id] - logits[distractor_id])
    return (logits[:, true_id] - logits[:, distractor_id]).tolist()


def restoration_effect(
    clean_ld: float, corrupted_ld: float, patched_ld: float
) -> float:
    """Normalised restoration effect of a patch.

    effect = (patched - corrupted) / (clean - corrupted)

    Returns 0.0 when clean and corrupted are indistinguishable (item should be
    filtered out by the E1 known-fact criterion anyway). Values can fall
    outside [0, 1] (over/under-shoot) — keep them; do NOT clip for analysis,
    clip only when ranking if desired.
    """
    denom = clean_ld - corrupted_ld
    if abs(denom) < 1e-8:
        return 0.0
    return (patched_ld - corrupted_ld) / denom
