"""Misleading-prompt suite construction for RQ2/RQ3 (plan §3, E1).

Context–memory (CM) conflict prompts per the taxonomy of Xu et al. (EMNLP 2024):
the context asserts the counterfactual distractor; the model's parametric
memory holds the true target. Hallucination = the distractor wins.

Template design notes:
- 3 templates, one of which must be reserved for the E6 template-robustness
  check (never used for head selection).
- Control condition for the copy-suppression confound (Campregher 2025,
  references.md [F7]): replace the distractor with a neutral non-answer token
  — see make_neutral_control().
"""
from __future__ import annotations

TEMPLATES = [
    "Earlier you learned that {subject} {relation_phrase} {distractor}. {prompt}",
    "{subject} {relation_phrase} {distractor}, as stated in the document. {prompt}",
    "Note: {subject} {relation_phrase} {distractor}. Now answer: {prompt}",
]

# reserved for E6 robustness only
HELDOUT_TEMPLATE = "Based on what you read, {subject} {relation_phrase} {distractor}. {prompt}"


def build_misleading_suite(facts, template_idx: int = 0):
    """Build (clean, misleading) prompt pairs from Fact records.

    clean       = the bare CounterFact prompt (no context)
    misleading  = template wrapping the counterfactual distractor
    Returns dicts with prompt ids and target ids for scoring.
    """
    raise NotImplementedError("E1")


def make_neutral_control(fact, template_idx: int = 0):
    """Same template but the context slot holds a neutral non-answer token
    (e.g. 'unknown'). Used to separate selective fact recall from generic
    copy suppression in E4 (plan E4, copy-suppression control)."""
    raise NotImplementedError("E1")
