#!/usr/bin/env bash
# Install the isolated public IO/metrics environment; no GPU or model downloads.
set -euo pipefail
cd "$(dirname "$0")/../.."
bootstrap="env/bootstrap/.venv"
tools="env/tools/.venv"
if [[ ! -x "$bootstrap/bin/python" ]]; then
  python3 -m venv "$bootstrap" || {
    echo "Python 3 with venv is required. On Ubuntu install python3-venv, then retry." >&2
    exit 1
  }
fi
"$bootstrap/bin/python" -m pip install 'uv==0.12.5'
if [[ ! -x "$tools/bin/python" ]]; then
  "$bootstrap/bin/uv" venv --python 3.11.16 --seed "$tools"
fi
"$tools/bin/python" -c "import sys; assert sys.version_info[:2] == (3, 11), 'Use a fresh Python 3.11 tools environment'"
"$bootstrap/bin/uv" pip install --python "$tools/bin/python" -r requirements.txt 'uv==0.12.5'
echo "Tools ready. Activate with: source env/tools/.venv/bin/activate"
