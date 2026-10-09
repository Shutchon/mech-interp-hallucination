# mech-interp-hallucination

**Locating and Mitigating Factual Hallucination Circuits in Small Language Models via Activation Patching**

Research code for a systematic activation-patching study of context–memory (CM) conflict in sub-2B small language models — Gemma-2-2B (primary), Llama-3.2-1B / Qwen2.5-1.5B (cross-architecture), Gemma-2-9B (coarse sweep) — with quantified hallucination↔capability trade-offs (dose–response curves).

> Status: pre-experiments. E0 (environment + parity gate) is the first runnable script.

## Repository layout

```
.
├── configs/                 # model registry, experiment params, prompt lists
├── scripts/                 # runnable entry points (E0 parity check, VM setup)
├── src/
│   ├── data/                # E1: CounterFact loading/filtering, misleading suites
│   ├── patching/            # E3: corruption regimes, sweep loops, patching metrics
│   ├── analysis/            # E4: head ranking, cross-model correlation
│   ├── interventions/       # E5: mean-ablation / knockout hooks, dose–response
│   └── eval/                # metrics + statistics (CI, McNemar, BH-FDR)
├── results/                 # JSONL outputs (append-only, idempotent)
├── figures/                 # generated figures for the paper
├── paper/                   # manuscript sources
├── PLAN_lab_and_paper_SCOPUS_Q3-Q4.md   # master plan (Thai)
├── references.md            # verified literature review (Thai)
└── GCP_SETUP.md             # cloud runbook (Thai)
```

## Quickstart (on the GCP L4 VM — see GCP_SETUP.md)

```bash
git clone git@github.com:<user>/mech-interp-hallucination.git
cd mech-interp-hallucination
bash scripts/vm_setup.sh          # venv + deps + CUDA sanity check
source .venv/bin/activate
export HF_TOKEN=hf_...            # required for gated models (GCP_SETUP.md §9)

# First gate of the whole project:
python scripts/e0_parity_check.py --n 100
```

Gate criteria (plan §3, E0): mean KL(HF ‖ TransformerLens) < 1e-3 and argmax agreement ≥ 99%. On FAIL → fall back to `pyvene` (works directly on HuggingFace weights; no re-implementation).

## Experiment map

| ID | What | Entry point |
|---|---|---|
| E0 | Environment + TL↔HF parity gate | `scripts/e0_parity_check.py` |
| E1 | CounterFact load + known-fact filter + misleading suite | `src/data/` |
| E2 | Baselines (clean/corrupted acc, TruthfulQA, MMLU, PPL) | `src/eval/` + lm-eval |
| E3 | Patching sweeps (block → MLP/attn → head), 2 corruption regimes | `src/patching/sweep.py` |
| E4 | Head ranking, attention patterns, superposition test | `src/analysis/` |
| E5 | Mean-ablation / knockout dose–response + full evals | `src/interventions/` |
| E6 | Cross-model + template robustness | configs-driven |

All experiment outputs go to `results/<exp>/*.jsonl`, written batch-by-batch (idempotent) so Spot VM preemption never loses a full sweep.

## Planning documents (Thai)

- `PLAN_lab_and_paper_SCOPUS_Q3-Q4.md` — full lab + paper plan, budget, risks, timeline
- `references.md` — gap verification + ~50 annotated references
- `GCP_SETUP.md` — GCP runbook: quota, Spot VM, backups, HF licensing

## Notes

- Models ≤ 9B in bfloat16 all fit a single L4 24GB; Gemma-2 uses logit softcapping, hence the E0 gate.
- Decoding is greedy everywhere; randomness only in mean-ablation estimates and random-head baselines (10 seeds).
