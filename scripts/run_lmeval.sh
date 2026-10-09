#!/usr/bin/env bash
# E2 (standard benchmarks) via lm-evaluation-harness — run AFTER e2_baselines.py
# Usage:
#   bash scripts/run_lmeval.sh google/gemma-2-2b fast     # TruthfulQA MC1/MC2 + WikiText PPL (~15 min)
#   bash scripts/run_lmeval.sh google/gemma-2-2b mmlu     # full MMLU (~1-1.5 h on L4)
set -euo pipefail
MODEL=${1:-google/gemma-2-2b}
STAGE=${2:-fast}
OUT=results/e2_baselines/lmeval/$(basename "$MODEL")
mkdir -p "$OUT"

source .venv/bin/activate

if [ "$STAGE" = "fast" ]; then
  lm_eval --model hf \
    --model_args "pretrained=$MODEL,dtype=bfloat16" \
    --tasks truthfulqa_mc1,truthfulqa_mc2,wikitext \
    --batch_size 8 --output_path "$OUT"
elif [ "$STAGE" = "mmlu" ]; then
  lm_eval --model hf \
    --model_args "pretrained=$MODEL,dtype=bfloat16" \
    --tasks mmlu \
    --batch_size 4 --output_path "$OUT"
else
  echo "unknown stage: $STAGE (use fast|mmlu)"; exit 1
fi
