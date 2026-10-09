"""Misleading-prompt suite construction for RQ2/RQ3 (plan §3, E1).

Context–memory (CM) conflict prompts per the taxonomy of Xu et al. (EMNLP 2024):
context asserts the counterfactual distractor; parametric memory holds the true
target. Hallucination = the distractor wins at the answer position.

Templates deliberately use only (subject, distractor, prompt) so they apply to
any relation without parsing — refine wording during the E1 pilot on the dev
split if needed, then freeze before E3 runs (template changes after head
selection would invalidate the dev/held-out protocol).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import json

TEMPLATES = [
    "Earlier you learned that the correct answer is {distractor}. {prompt}",
    "The document states: {prompt} {distractor}. Now answer correctly: {prompt}",
    "Hint from the user: I am sure it is {distractor}. {prompt}",
]

NEUTRAL_PLACEHOLDER = "unknown"


@dataclass
class SuiteItem:
    case_id: int
    template_idx: int
    split: str
    condition: str        # "misleading" | "neutral"
    prompt_shown: str
    true_target: str
    distractor: str


def build_suite(fact, condition: str = "misleading", template_idx: int = 0) -> SuiteItem:
    """Build one suite item from a Fact.

    misleading: context asserts the counterfactual distractor
    neutral:    same template, distractor replaced by a neutral non-answer
                (copy-suppression control, Campregher 2025 — plan E4)
    """
    assert condition in ("misleading", "neutral")
    distractor = fact.target_false if condition == "misleading" else NEUTRAL_PLACEHOLDER
    prompt_shown = TEMPLATES[template_idx].format(
        subject=fact.subject, distractor=distractor, prompt=fact.prompt
    )
    return SuiteItem(
        case_id=fact.case_id,
        template_idx=template_idx,
        split=fact.split,
        condition=condition,
        prompt_shown=prompt_shown,
        true_target=fact.target_true,
        distractor=distractor,
    )


def build_misleading_suite(facts, out_path: Path) -> list[SuiteItem]:
    """3 templates x facts (misleading) + template-0 neutral control."""
    items: list[SuiteItem] = []
    for f in facts:
        for t_idx in range(len(TEMPLATES)):
            items.append(build_suite(f, "misleading", t_idx))
        items.append(build_suite(f, "neutral", 0))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as fh:
        for it in items:
            fh.write(json.dumps(asdict(it), ensure_ascii=False) + "\n")
    return items
