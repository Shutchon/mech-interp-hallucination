"""Corruption regimes for activation patching (plan §3, E3).

Two regimes per sweep:
(a) ROME-style Gaussian noise on subject-token embeddings (denoising setup,
    comparable with the literature)
(b) Misleading-context corruption (the CM-conflict regime — answers RQ2)
"""
from __future__ import annotations

import torch


def corrupt_subject_embeddings(
    embeddings: torch.Tensor,
    subject_token_slice: slice,
    noise_scale: float = 3.0,
    seed: int = 0,
) -> torch.Tensor:
    """Return a copy of token embeddings with Gaussian noise added to the
    subject span (ROME uses scale ~3x the embedding std). Used as the
    corrupted run for regime (a)."""
    raise NotImplementedError("E3")


def corrupt_with_context(prompt: str, misleading_prompt: str):
    """For regime (b): the corrupted input is simply the misleading prompt —
    no embedding noise. Keep this wrapper so both regimes share one interface."""
    return misleading_prompt
