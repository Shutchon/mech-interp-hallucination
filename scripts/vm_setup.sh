#!/usr/bin/env bash
# VM environment setup — run once after cloning (GCP_SETUP.md §4)
set -euo pipefail

echo "== GPU check =="
nvidia-smi || { echo "nvidia-smi failed — check drivers/image"; exit 1; }

echo "== Python venv =="
python3 -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

echo "== Sanity checks =="
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
PY

cat <<'EOF'

Next steps:
  1) export HF_TOKEN=hf_...        # gated models (see GCP_SETUP.md §9)
  2) echo 'export HF_HOME=$HOME/hf_cache' >> ~/.bashrc && source ~/.bashrc
  3) python scripts/e0_parity_check.py --n 100
EOF
