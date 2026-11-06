"""Evaluate released Dev/Val scenes; Test GT is never distributed."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _bootstrap
from track2_io import ROOT, scene_path, validate_scene


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=ROOT / "data/scenes")
    parser.add_argument("--run-name", default="default")
    parser.add_argument("--device", choices=["cpu", "gpu"], default="cpu")
    args = parser.parse_args()
    manifest, _ = validate_scene(args.manifest, args.data_root)
    if manifest["scene_id"].startswith("test_"):
        raise ValueError(
            "Test GT is private. Export the final GLBs and submit the generated CSV to Kaggle."
        )
    gt = scene_path(args.data_root, manifest["scene_id"]) / "scene.blend"
    if not gt.is_file():
        raise FileNotFoundError(
            "Install the released Dev/Val evaluation data (see data/README.md)"
        )
    from evaluation.portable_local import evaluate

    result = evaluate(args.manifest, args.data_root, args.run_name, args.device)
    print("SCORE", result["final_score"], "GROUPS", result["groups"])


if __name__ == "__main__":
    main()
