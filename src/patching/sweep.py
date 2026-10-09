"""Activation-patching sweeps (plan §3, E3) — coarse-to-fine.

Coarse: block-level (residual stream, every layer × candidate positions)
Mid:    attention block vs MLP block on top-5 layers from coarse
Fine:   per-head z-vectors on layers of interest
Optional screening pass with Attribution Patching (Syed et al. 2024), then
confirmation by real patching on the top candidates.
"""
from __future__ import annotations

from pathlib import Path

POSITIONS = "subject_tokens + final 3 tokens (plan E3)"


def run_block_sweep(
    model,
    facts,
    corruption: str,               # "noise" | "misleading"
    out_path: Path,
    positions=None,
    resume: bool = True,
):
    """Sweep residual-stream patches across layers × positions.

    Results are appended to out_path as JSONL, one line per
    (fact, layer, position) — idempotent: already-present keys are skipped on
    resume, so Spot preemption never loses or duplicates work.
    """
    raise NotImplementedError("E3")


def run_head_sweep(model, facts, layers: list[int], out_path: Path, resume: bool = True):
    """Per-head z-patching on selected layers (fine pass)."""
    raise NotImplementedError("E3")


def run_attribution_screening(model, facts, out_path: Path):
    """Fast gradient-based attribution screening (Syed, Rager & Conmy 2024).
    Screening ONLY — report in the paper that candidates were confirmed with
    real patching (plan §3 E3 note)."""
    raise NotImplementedError("E3")
