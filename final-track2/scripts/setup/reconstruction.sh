#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
task_root="$PWD"
fire_repo="$task_root/third_party/Fire3D"
if [[ ! -d "$fire_repo/.git" ]]; then
  git clone https://github.com/xiahongchi/Fire3D.git "$fire_repo"
fi
git -C "$fire_repo" checkout --detach 2368dd2f3909120cf90bbf8a17807abe9c41e600
if [[ ! -x env/fire3d/.conda/bin/python ]]; then
  conda env create --prefix "$task_root/env/fire3d/.conda" -f env/fire3d/environment.yml
fi
source env/fire3d/activate.sh
uv pip install --python "$CONDA_PREFIX/bin/python" torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
uv pip install --python "$CONDA_PREFIX/bin/python" -e "$fire_repo" setuptools==75.8.2 wheel spconv-cu118==2.3.8 xformers==0.0.31.post1 matplotlib==3.10.9 plotly==7.1.0 fast-simplification==0.2.0
FLASH_ATTENTION_FORCE_BUILD=TRUE NVCC_THREADS=1 uv pip install --python "$CONDA_PREFIX/bin/python" --no-build-isolation --no-deps flash-attn==2.7.3
uv pip install --python "$CONDA_PREFIX/bin/python" --no-deps 'natten==0.21.0+torch270cu128' --find-links https://whl.natten.org
clone_at() {
  local task_url="$1" task_path="$2" task_commit="$3"
  [[ -d "$task_path/.git" ]] || git clone "$task_url" "$task_path"
  git -C "$task_path" fetch origin "$task_commit"
  git -C "$task_path" checkout --detach "$task_commit"
}
clone_at https://github.com/facebookresearch/dinov3.git "$fire_repo/third_party/dinov3" 31703e4cbf1ccb7c4a72daa1350405f86754b6d1
clone_at https://github.com/facebookresearch/pytorch3d.git "$task_root/third_party/native/pytorch3d" 75ebeeaea0908c5527e7b1e305fbc7681382db47
clone_at https://github.com/JeffreyXiang/FlexGEMM.git "$task_root/third_party/native/flex_gemm" 6dd94a859c26ee8246888502eada3dd8ad85532e
clone_at https://github.com/NVlabs/nvdiffrast.git "$task_root/third_party/native/nvdiffrast" 253ac4fcea7de5f396371124af597e6cc957bfae
for eigen_path in "$fire_repo/trellis2_x2/o-voxel/third_party/eigen" "$fire_repo/trellis2_x2/CuMesh/third_party/cubvh/third_party/eigen"; do
  if [[ ! -d "$eigen_path/Eigen" ]]; then
    git clone --depth 1 --branch 3.4.0 https://gitlab.com/libeigen/eigen.git "$eigen_path"
  fi
done
for extension in "$task_root/third_party/native/pytorch3d" "$task_root/third_party/native/flex_gemm" "$task_root/third_party/native/nvdiffrast" "$fire_repo/trellis2_x2/o-voxel" "$fire_repo/trellis2_x2/CuMesh"; do
  uv pip install --python "$CONDA_PREFIX/bin/python" --no-build-isolation --no-deps "$extension"
done
PYTHONPATH="$task_root/src" HF_HUB_DISABLE_XET=1 "$CONDA_PREFIX/bin/python" -m baselines.fire3d_recon.prepare_models
uv pip freeze --python "$CONDA_PREFIX/bin/python" > env/fire3d/requirements.lock.txt
