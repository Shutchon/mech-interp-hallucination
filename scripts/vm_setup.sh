#!/usr/bin/env bash
# VM environment setup — run once after cloning (GCP_SETUP.md §4)
set -euo pipefail

echo "== GPU check =="
nvidia-smi || { echo "nvidia-smi failed — check drivers/image"; exit 1; }

echo "== Ensure python venv support =="
# Slim Ubuntu images ship without python3-venv; install it if missing (needs sudo).
if ! python3 -c "import ensurepip" >/dev/null 2>&1; then
    echo "python3-venv missing — installing (sudo required)"
    sudo apt-get update -qq && sudo apt-get install -y python3-venv
fi
# Remove a partially-created venv left behind by a failed previous run
[ -d .venv ] && [ ! -x .venv/bin/python ] && rm -rf .venv

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
  1) source .venv/bin/activate     # activate the venv (this script ran in a subshell!)
  2) export HF_TOKEN=hf_...        # gated models (see GCP_SETUP.md §9)
     echo 'export HF_TOKEN=hf_...' >> ~/.bashrc   # to make it permanent
  3) echo 'export HF_HOME=$HOME/hf_cache' >> ~/.bashrc && source ~/.bashrc
  4) tmux new -s e0                # run experiments inside tmux (SSH-drop safe)
     python scripts/e0_parity_check.py --n 100
EOF
