#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${1:-venus-phase1}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda is required. Choose a cloud image with Miniconda/Anaconda installed." >&2
  exit 1
fi

eval "$(conda shell.bash hook)"
conda create --name "$ENV_NAME" python=3.10.13 --yes
conda activate "$ENV_NAME"

python -m pip install --upgrade pip==23.2.1
python -m pip install \
  torch==2.0.1+cu118 \
  torchvision==0.15.2+cu118 \
  --index-url https://download.pytorch.org/whl/cu118
python -m pip install --requirement "$SCRIPT_DIR/requirements.txt"
python -m pip check

python - <<'PY'
import torch
import transformers

print(f"torch={torch.__version__}")
print(f"transformers={transformers.__version__}")
print(f"cuda_available={torch.cuda.is_available()}")
print(f"bf16_supported={torch.cuda.is_available() and torch.cuda.is_bf16_supported()}")
if torch.cuda.is_available():
    print(f"gpu={torch.cuda.get_device_name(0)}")
PY

echo "Environment '$ENV_NAME' is ready. Activate it with: conda activate $ENV_NAME"
