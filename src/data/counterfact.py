"""CounterFact loading and splits (plan §3, E1).

Data source (verified Oct 2026): raw ROME counterfact.json mirrored at
huggingface.co/datasets/wangzn2001/counteract (file: counterfact.json).
Alternative: azhx/counterfact; NeelNanda/counterfact-tracing is a pre-adapted
variant for activation tracing — usable if fields match.

ROME record schema used here:
  record["requested_rewrite"] = {
    "prompt":        "The mother tongue of Danielle Darrieux is",   (or with "{}")
    "subject":       "Danielle Darrieux",
    "target_true":   {"str": "French",  "id": ...},
    "target_new":    {"str": "English", "id": ...},   # the counterfactual distractor
    "relation_id":   "P1412"
  }
"""
from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class Fact:
    case_id: int
    prompt: str                 # completed prompt (subject inlined)
    subject: str
    target_true: str
    target_false: str           # counterfactual distractor (ROME's target_new)
    relation_id: str
    split: str = "unassigned"   # "dev" | "heldout"
    popularity: float | None = None  # PopQA join, E4 (optional)


def _complete_prompt(template: str, subject: str) -> str:
    return template.format(subject) if "{}" in template else template


def load_counterfact(path: Path) -> list[Fact]:
    """Load raw counterfact.json into Fact records (no filtering here)."""
    records = json.loads(Path(path).read_text())
    facts = []
    for rec in records:
        rw = rec["requested_rewrite"]
        facts.append(
            Fact(
                case_id=rec["case_id"],
                prompt=_complete_prompt(rw["prompt"], rw["subject"]).strip(),
                subject=rw["subject"],
                target_true=rw["target_true"]["str"].strip(),
                target_false=rw["target_new"]["str"].strip(),
                relation_id=rw["relation_id"],
            )
        )
    return facts


def split_by_relation(facts: list[Fact], seed: int = 0) -> list[Fact]:
    """Assign dev/heldout at the RELATION level (plan E1 anti-circularity):
    the held-out split shares no relation with the split used for head
    selection, so results in E5 generalise beyond relations seen in E3/E4."""
    relations = sorted({f.relation_id for f in facts})
    rng = random.Random(seed)
    rng.shuffle(relations)
    half = (len(relations) + 1) // 2
    dev_relations = set(relations[:half])
    for f in facts:
        f.split = "dev" if f.relation_id in dev_relations else "heldout"
    return facts


def save_facts(facts: list[Fact], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as fh:
        for f in facts:
            fh.write(json.dumps(asdict(f), ensure_ascii=False) + "\n")
