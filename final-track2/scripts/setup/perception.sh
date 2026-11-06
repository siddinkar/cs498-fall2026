#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
for task_tool in ffmpeg ffprobe; do
  command -v "$task_tool" >/dev/null || { echo "Install ffmpeg (including ffprobe) before baseline setup. Missing: $task_tool" >&2; exit 1; }
done
# Install the Torch 2.5.1 perception runtime and separately pinned model code.
runtime_dir="env/baseline_2d/.venv"
if [[ ! -x "$runtime_dir/bin/python" ]]; then
  uv venv --python 3.11 "$runtime_dir"
fi
uv pip install --python "$runtime_dir/bin/python" 'torch==2.5.1' 'torchvision==0.20.1' --index-url https://download.pytorch.org/whl/cu121
uv pip install --python "$runtime_dir/bin/python" -r env/pi3x_fusion/requirements.txt
PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}" "$runtime_dir/bin/python" -m baselines.pi3x_fusion.prepare_models
uv pip freeze --python "$runtime_dir/bin/python" > env/pi3x_fusion/requirements.lock.txt
