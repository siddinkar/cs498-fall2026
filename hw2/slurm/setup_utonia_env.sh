#!/bin/bash
# Run once on a login node: builds a separate venv for the Utonia bonus and
# pre-downloads the pinned checkpoint (compute nodes may lack internet access).
set -euo pipefail
ROOT=/scratch/sdinkar2/cs498-fall2026
VENV=$ROOT/.venv-utonia
UTONIA_SRC=$ROOT/Utonia
UTONIA_COMMIT=de1f9deef4ee65e508d74347f8d0eb848d878abb
HF_REVISION=9457f025e36c7001ee962a542f0d27a5b600e1e7

module load python/3.11.11
python -m venv "$VENV"
source "$VENV/bin/activate"
unset PYTHONPATH
python -m pip install --upgrade pip

# Torch 2.5.0 + CUDA 12.4 wheels, matching the official environment.yml.
pip install "numpy<2" torch==2.5.0 torchvision==0.20.0 --index-url https://download.pytorch.org/whl/cu124
pip install spconv-cu124
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.5.0+cu124.html
pip install huggingface_hub timm addict packaging matplotlib einops scipy

# Official source at the pinned commit (FlashAttention is skipped; the script falls back automatically).
if [ ! -d "$UTONIA_SRC" ]; then
  git clone https://github.com/Pointcept/Utonia.git "$UTONIA_SRC"
fi
git -C "$UTONIA_SRC" checkout "$UTONIA_COMMIT"
# setup.py needs pkg_resources, which new setuptools dropped; a .pth file puts the package on the path instead.
echo "$UTONIA_SRC" > "$VENV/lib/python3.11/site-packages/utonia_src.pth"

# Pre-download the pinned checkpoint.
python - <<PY
from huggingface_hub import hf_hub_download
import os
path = hf_hub_download(repo_id="Pointcept/Utonia", filename="utonia.pth", repo_type="model",
                       revision="$HF_REVISION", local_dir=os.path.expanduser("~/.cache/utonia/ckpt"))
print("checkpoint:", path)
PY

python -c "import torch, spconv, torch_scatter, utonia; print('torch', torch.__version__, '| utonia import OK')"
