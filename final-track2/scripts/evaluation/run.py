#!/usr/bin/env python3
"""Canonical Dev/Val entry point; no physics simulation or Test GT input."""

import argparse
import os
from pathlib import Path
import sys

for key in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ[key] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from evaluation.portable_local import evaluate
from track2_io import ROOT


def main():
    """Use released Dev/Val GT from the installed data root only."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction-manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=ROOT / "data/scenes")
    parser.add_argument("--run-name", default="default")
    parser.add_argument("--device", choices=["cpu", "gpu"], default="cpu")
    args = parser.parse_args()
    result = evaluate(
        args.prediction_manifest, args.data_root, args.run_name, args.device
    )
    print(result["scene_id"], result["final_score"], result["groups"])


if __name__ == "__main__":
    main()
