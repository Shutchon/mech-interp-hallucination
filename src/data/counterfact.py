"""CounterFact loading and known-fact filtering (plan §3, E1).

Fields per fact (ROME/CounterFact schema):
- requested_rewrite: {prompt, subject, target_true, target_false, relation_id}
- PopQA popularity scores are joined later for the E4 popularity analysis.

TODO (E1): verify the current CounterFact download URL (ROME repo or HF mirror)
and its license — tracked in references.md §8.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class Fact:
    prompt: str            # CounterFact prompt, e.g. "The Eiffel Tower is located in the city of"
    subject: str
    target_true: str
    target_false: str      # counterfactual distractor
    relation_id: str
    split: str             # "dev" | "heldout"  (50/50 split by relation, plan E1)


def load_counterfact(path: Path) -> list[Fact]:
    """Load CounterFact triples from the ROME-format JSON into Fact records."""
    raise NotImplementedError("E1: implement loader once dataset URL is verified")


def filter_known_facts(
    facts: list[Fact],
    model,  # HookedTransformer or HF model wrapper
    min_logit_gap: float = 0.5,
) -> list[Fact]:
    """Keep facts the model already knows: greedy argmax == target_true AND
    logit gap (true - runner-up-distractor) >= min_logit_gap (plan E1)."""
    raise NotImplementedError("E1")


def split_by_relation(facts: list[Fact], seed: int = 0) -> list[Fact]:
    """Assign dev/heldout at the RELATION level (not item level) so the
    held-out split shares no relation with the split used for head selection
    (anti-circularity protocol, plan §2/§8.3)."""
    raise NotImplementedError("E1")
