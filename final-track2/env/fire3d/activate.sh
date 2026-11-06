#!/usr/bin/env bash
# Source this before running/building Fire3D's CUDA 12.8 extensions.
FIRE3D_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export CONDA_PREFIX="$FIRE3D_ROOT/env/fire3d/.conda"
export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CONDA_PREFIX/bin:$PATH"
export CC="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc"
export CXX="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++"
export CPATH="$CONDA_PREFIX/targets/x86_64-linux/include${CPATH:+:$CPATH}"
for FIRE3D_INCLUDE in "$CONDA_PREFIX"/lib/python3.10/site-packages/nvidia/*/include; do
    export CPATH="$FIRE3D_INCLUDE:$CPATH"
done
export LIBRARY_PATH="$CONDA_PREFIX/targets/x86_64-linux/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"
# Leave architecture detection to PyTorch unless the user explicitly sets
# TORCH_CUDA_ARCH_LIST for a shared/remote build machine.
export MAX_JOBS="${MAX_JOBS:-2}"
# Keep Fire3D's module defaults (dense SDPA, sparse FlashAttention, FlexGEMM).
unset ATTN_BACKEND SPARSE_ATTN_BACKEND SPARSE_CONV_BACKEND
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
