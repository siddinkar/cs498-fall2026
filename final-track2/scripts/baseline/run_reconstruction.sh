#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
if [[ -z "${TRACK2_RECONSTRUCTION_PYTHON:-}" ]]; then
  source env/fire3d/activate.sh
fi
exec "${TRACK2_RECONSTRUCTION_PYTHON:-env/fire3d/.conda/bin/python}" scripts/baseline/reconstruct.py "$@"
