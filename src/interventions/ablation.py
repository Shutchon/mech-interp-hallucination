"""Head ablation hooks and dose–response protocol (plan §3, E5).

Interventions (applied to head z-vectors at the same positions used in E3):
- mean ablation: replace with the mean over a reference set of Dev prompts
- zero knockout: set to zero

Fairness baselines (plan E5, must-have):
- random-k heads (10 seeds)
- top-k heads by activation norm
Dose–response: k = 1, 2, 5, 10, 20 heads → hallucination rate (held-out),
MMLU, WikiText-2 PPL.
"""
from __future__ import annotations

import torch


def make_mean_ablation_hook(mean_z: torch.Tensor):
    """Return a hook replacing the head's z with a precomputed mean vector
    (computed over 64–128 random Dev prompts, plan E5)."""

    def hook(z, hook):  # noqa: ARG001 — TransformerLens hook signature
        z[:] = mean_z.to(device=z.device, dtype=z.dtype)
        return z

    return hook


def make_zero_hook():
    def hook(z, hook):  # noqa: ARG001
        z[:] = 0
        return z

    return hook


def compute_mean_z(model, prompts, layer: int, head: int) -> torch.Tensor:
    """Mean z-vector of one head over reference prompts (position selection
    mirrors the E3 patching positions)."""
    raise NotImplementedError("E5")


def run_dose_response(model, head_ranking, eval_suite, ks=(1, 2, 5, 10, 20), out_path=None):
    """For each k: ablate top-k heads, evaluate on HELD-OUT items + MMLU +
    WikiText-2 PPL. Also runs random-k and top-norm-k baselines."""
    raise NotImplementedError("E5")
